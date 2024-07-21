import os
import re
from typing import List

import torch
from sentencepiece import SentencePieceProcessor

import batch_encoding


class GLM3Tokenizer:
    def __init__(self) -> None:
        vocab_file = os.path.join(os.path.dirname(__file__), 'glm3_sp_tokenizer.model')
        self.sp = SentencePieceProcessor(vocab_file)

        self.token_pad = '<pad>'
        self.token_pad_id = 0
        self.token_bos = '<bos>'
        self.token_bos_id = 1
        self.token_eos = '<eos>'
        self.token_eos_id = 2
        self.token_unk = '<unk>'
        self.token_unk_id = 3

        self.special_tokens = {
            self.token_pad: self.token_pad_id,
            self.token_bos: self.token_bos_id,
            self.token_eos: self.token_eos_id,
            self.token_unk: self.token_unk_id
        }
        self.special_tokens_decode = {v: k for k, v in self.special_tokens.items()}
        self._special_tokens_count = len(self.special_tokens.keys())
        self._n_words = self._special_tokens_count + self.sp.vocab_size()
        self.dtype = torch.int32
        self._special_token_regex = "|".join(self.special_tokens.keys())

    def _build_tensor(self, seq):
        return torch.tensor(seq, dtype=self.dtype)
    
    def _sp_encode(self, s: str):
        """sp encode, special token offset"""
        return self._build_tensor(self.sp.encode(s)) + self._special_tokens_count
    
    def _encode_with_special_token(self, s: str):
        last_special_token_end_pos = 0
        collection = []
        for match in re.finditer(self._special_token_regex, s):
            segment = s[last_special_token_end_pos:match.start()]
            if segment:
                collection.append(self._sp_encode(segment))
            special_token_id = self.special_tokens[match.group()]
            collection.append(self._build_tensor([special_token_id]))
            last_special_token_end_pos = match.end()

        segment = s[last_special_token_end_pos:]
        collection.append(self._sp_encode(segment))            
        return torch.concat(collection)
    
    def encode(self, s: str | List[str], parse_special_tokens=False):
        """i'm lazy, left padding only"""
        assert isinstance(s, str) or isinstance(s, list)
        encode_fn = self._encode_with_special_token
        if not parse_special_tokens:
            encode_fn = self._sp_encode
        if isinstance(s, str):
            s = [s]
        encoded = [encode_fn(_s) for _s in s]
        max_len = max([len(_e) for _e in encoded])
        result = torch.zeros((len(encoded), max_len), dtype=self.dtype)
        attn_mask = torch.zeros((len(encoded), max_len), dtype=self.dtype)
        for i, item in enumerate(encoded):
            start_pos = max_len - len(item)
            result[i][start_pos:max_len] = item
            attn_mask[i].masked_fill_(result[i] != 0, 1)
        return batch_encoding.BatchEncoding(result, attn_mask)
    
    def encode_one_no_attn_no_special(self, s: str):
        return self._sp_encode(s)
    
    def _decode_one(self, array: torch.tensor, remove_special_tokens):
        assert array.dim() == 1
        token_list = (array - self._special_tokens_count).tolist()
        
        result_list = []
        current_token_collection = []
        for token_id in token_list:
            if token_id < 0:
                result_list.append(self.sp.decode(current_token_collection))
                current_token_collection = []
                if not remove_special_tokens:
                    result_list.append(self.special_tokens_decode[token_id+self._special_tokens_count])
            else:
                current_token_collection.append(token_id)
        result_list.append(self.sp.decode(current_token_collection))
        return "".join(result_list)

    def decode(self, array: torch.tensor, remove_special_tokens=True):
        assert array.dim() <= 2
        if array.dim() == 1:
            array = torch.stack([array])
        results = [self._decode_one(a, remove_special_tokens) for a in torch.unbind(array)]
        return results

    def vocab_size(self) -> int:
        return self._n_words
    
    def tokenize(self, text) -> List[str]:
        return self.sp.encode(text, out_type=str)


if __name__ == "__main__":
    TOKENIZER = GLM3Tokenizer()
    print(TOKENIZER.vocab_size())
    print(TOKENIZER._sp_encode("你好世界"))
    out = TOKENIZER._encode_with_special_token("<pad><bos>你好世界<eos><bos>走向群星<eos><bos>你好世界<eos>")
    print(out)

    out = TOKENIZER.encode(["<bos>你好世界<eos>", "1234"])
    print(out)

    out.to('cuda')
    print(out)

    out = TOKENIZER.encode(["<pad><bos>你好世界<eos>", "<bos><eos>", "<eos>", "12345"], parse_special_tokens=True)
    print(out)

    result = TOKENIZER.decode(out.input_ids)
    print(result)

    result = TOKENIZER.decode(out.input_ids, remove_special_tokens=False)
    print(result)

    VOCABS = [(i, TOKENIZER.sp.id_to_piece(i)) for i in range(TOKENIZER.sp.get_piece_size())]
    with open("vocab.txt", 'w', encoding="utf8") as file:
        for t in VOCABS:
            print(repr(t), file=file)