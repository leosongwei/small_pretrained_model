from torch import nn
import torch
from torch import Tensor
import dataclasses
import math


@dataclasses.dataclass
class CustomModelConfig():
    vocab_size: int
    padding_token_id: int
    max_position_embeddings: int = 4096
    hidden_size: int = 1024
    num_heads: int = 16
    #num_kv_heads: int = 16
    MLP_intermediate: int = 5000
    num_layers: int = 16
    attention_dropout: float = 0.1 # https://www.quora.com/What-is-a-good-value-for-the-dropout-rate-in-deep-learning-networks
    # hmmm, what's the good value for attention_dropout?
    dtype: torch.dtype = torch.bfloat16
    training: bool = True
    embedding_dim: int = 0
    linear_imp: type = torch.nn.Linear

    def get_total_params(self):
        if self.embedding_dim == 0:
            embedding_size = self.vocab_size * self.hidden_size
            lm_head_size = self.hidden_size * self.vocab_size
        else:
            embedding_size = self.vocab_size * self.embedding_dim + self.embedding_dim * self.hidden_size
            lm_head_size = self.hidden_size * self.embedding_dim + self.embedding_dim * self.vocab_size

        head_dim = self.hidden_size // self.num_heads

        # attention: no bias
        attention_size = (
            self.hidden_size * self.num_heads * head_dim * 3 + # Q, K, V
            self.hidden_size * self.hidden_size
        )
        
        # MLP
        MLP_size = self.hidden_size * self.MLP_intermediate * 3 # Gate, Up, Down

        layers_size = self.num_layers * (
                attention_size +
                MLP_size +
                self.hidden_size * 2 # input_layernorm, post_attention_layernorm
            )
        
        total_params = layers_size \
            + embedding_size + lm_head_size \
            + self.hidden_size # norm
        
        print("embedding ratio:", embedding_size / total_params)
        print("layers ratio:", layers_size / total_params)
        print("  * attn size ratio:", (attention_size*self.num_layers) / total_params)
        print("  * MLP ratio:", (MLP_size * self.num_layers) / total_params)
        print("lm_head ratio:", lm_head_size / total_params)
        print(f"total:{total_params} (~{total_params//(1000*1000)}M)")
        
        return total_params

    def to_json(self):
        return dataclasses.asdict(self)
    
    @classmethod
    def from_json(cls, dict):
        return cls(**dict)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        taken from the transformers/src/transformers/models/llama/modeling_llama.py
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)
    

class MLPLayer(nn.Module):
    def __init__(self, config: CustomModelConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.MLP_intermediate

        self.gate_proj = config.linear_imp(self.hidden_size, self.intermediate_size, bias=False, dtype=config.dtype)
        self.up_proj = config.linear_imp(self.hidden_size, self.intermediate_size, bias=False, dtype=config.dtype)
        self.down_proj = config.linear_imp(self.intermediate_size, self.hidden_size, bias=False, dtype=config.dtype)

        self.act_fn = nn.SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        gated = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        dotted = torch.mul(gated, up)
        down = self.down_proj(dotted)
        return down


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        """taken from the transformers/src/transformers/models/llama/modeling_llama.py"""
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    def __init__(self, config: CustomModelConfig, layer_idx: int, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.dtype
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        # self.num_key_value_heads = config.num_kv_heads
        self.attn_dropout = config.attention_dropout
        self.training = config.training

        self.max_position_embeddings = config.max_position_embeddings

        assert self.num_heads * self.head_dim == self.hidden_size

        self.q_proj = config.linear_imp(self.hidden_size, self.num_heads * self.head_dim, dtype=config.dtype, bias=False)
        self.k_proj = config.linear_imp(self.hidden_size, self.num_heads * self.head_dim, dtype=config.dtype, bias=False)
        self.v_proj = config.linear_imp(self.hidden_size, self.num_heads * self.head_dim, dtype=config.dtype, bias=False)
        self.o_proj = config.linear_imp(self.hidden_size, self.hidden_size, dtype=config.dtype, bias=False)
        self.rope = LlamaRotaryEmbedding(self.head_dim)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.tensor, causal_mask: torch.tensor):
        """
        hidden_states: bsz, q_len, hidden_dim
        position_ids: bsz, q_len
        cache_positions: q_len,
        attention_mask: 1, 1, max_position_embeddings, max_position_embeddings
        """

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # transpose to: bsz, head, q_len, head_dim
        # TODO: verify the Llama GQA (might not work on small param size)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2,3)) / math.sqrt(self.head_dim)

        attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attn_dropout, training=self.config.training)
        attn_output = torch.matmul(attn_weights, value_states)
        assert attn_output.shape == (bsz, self.num_heads, q_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: CustomModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention = LlamaAttention(config, layer_idx)
        self.mlp = MLPLayer(config)
        self.input_layernorm = LlamaRMSNorm(self.hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(self.hidden_size)
        self.dtype = config.dtype
    
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.tensor, causal_mask: torch.tensor):
        residual = hidden_states
        hidden_states = hidden_states.to(dtype=self.dtype)
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(
            hidden_states=hidden_states,
            position_ids=position_ids,
            causal_mask=causal_mask
        )
        residual = hidden_states + residual
        hidden_states = residual.to(self.dtype)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class CustomLLamaModel(nn.Module):

    def __init__(self, config: CustomModelConfig) -> None:
        super().__init__()
        self.config = config
        self.dtype = config.dtype
        self.padding_idx = config.padding_token_id
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=self.dtype)
        # the entries at padding_idx do not contribute to the gradient
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size,
                                         embedding_dim=config.hidden_size,
                                         padding_idx=self.padding_idx,
                                         dtype=config.dtype)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)
        ])

        self.norm = LlamaRMSNorm(config.hidden_size)


    def _create_causal_mask(self, attention_mask: torch.IntTensor):
        batch_size, q_len = attention_mask.shape
        device = attention_mask.device
        dtype = self.dtype
        min_dtype = torch.finfo(self.dtype).min

        causal_mask = torch.full((q_len, q_len), fill_value=1, dtype=torch.int8, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        padding_mask = attention_mask.to(dtype=torch.int8)[:, None, None, :]
        padding_mask = (-attention_mask+1)[:, None, None, :]

        causal_mask = torch.where(causal_mask < padding_mask, padding_mask, causal_mask)
        causal_mask = causal_mask.to(dtype=dtype) * min_dtype
        return causal_mask
    
    def forward(
            self,
            input_ids: torch.IntTensor,
            attention_mask: torch.Tensor
        ):
        # Output:
        # * logits
        # * loss
        inputs_embeddings = self.embed_tokens(input_ids)

        # position_ids:
        # attention_mask: [[0, 0, 1, 1, 1]]
        # cumsum: [[0, 0, 1, 2, 3]] - 1 = [[0, 0, 0, 1, 2]]
        # masked_fill_: [[1, 1, 0, 1, 2]]
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids = position_ids.masked_fill_(attention_mask == 0, 1)

        causal_mask = self._create_causal_mask(attention_mask)

        hidden_states = inputs_embeddings
        for decoder_layer in self.layers:
            decoder_layer: LlamaAttention = decoder_layer
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask
            )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        # logits = torch.nn.functional.softmax(logits, dim=-1)
        return logits
    
    def compute_loss(self, logits: torch.tensor, labels: torch.tensor):
        assert logits.shape[0] == labels.shape[0] # batch size
        assert logits.shape[1] == labels.shape[1] # q_len
        assert logits.shape[2] == self.vocab_size
        loss_fn = torch.nn.CrossEntropyLoss()
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # flatten
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1).to(shift_logits.device)
        loss = loss_fn(shift_logits, shift_labels.to(dtype=torch.long))
        return loss
    
    def aggressive_decode(self, logits: torch.tensor):
        return torch.argmax(logits, dim=-1)


class EncoderDecoderModel(CustomLLamaModel):
    def __init__(self, config: CustomModelConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(num_embeddings=config.vocab_size,
                                         embedding_dim=config.embedding_dim,
                                         padding_idx=self.padding_idx,
                                         dtype=config.dtype)
        self.embedding_decoder = nn.Linear(config.embedding_dim, self.config.hidden_size, bias=False, dtype=self.dtype)
        self.lm_head_decoder = nn.Linear(self.config.hidden_size, config.embedding_dim, bias=False, dtype=self.dtype)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False, dtype=self.dtype)

    def forward(
            self,
            input_ids: torch.IntTensor,
            attention_mask: torch.Tensor
        ):
        inputs_embeddings = self.embed_tokens(input_ids)

        position_ids = attention_mask.cumsum(-1) - 1
        position_ids = position_ids.masked_fill_(attention_mask == 0, 1)
        causal_mask = self._create_causal_mask(attention_mask)

        hidden_states = self.embedding_decoder(inputs_embeddings)

        for decoder_layer in self.layers:
            decoder_layer: LlamaAttention = decoder_layer
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask
            )

        hidden_states = self.norm(hidden_states)
        output_states = self.lm_head_decoder(hidden_states)
        logits = self.lm_head(output_states)
        return logits

if __name__ == "__main__":
    import glm3_tokenizer
    TOKENIZER = glm3_tokenizer.GLM3Tokenizer()
    CONFIG = CustomModelConfig(
            vocab_size=TOKENIZER.vocab_size(),
            padding_token_id=TOKENIZER.token_pad_id,
            max_position_embeddings=4096,
            hidden_size=1280,
            num_heads=16,
            MLP_intermediate=5500,
            num_layers=16,
            attention_dropout=0.5,
            dtype=torch.bfloat16,
            training=True
        )
    CONFIG.get_total_params()

    # MODEL = CustomLLamaModel(CONFIG).to('cuda')

    # # MODEL = EncoderDecoderModel(CONFIG).to('cuda')
    # TEXTS = ["hello world.", "1,2,3,4,5"]
    # INPUTS = TOKENIZER.encode(TEXTS).to('cuda')
    # RESULTS = MODEL(**INPUTS)
    # TOKEN_IDS = MODEL.aggressive_decode(RESULTS)
    # print(TOKEN_IDS)