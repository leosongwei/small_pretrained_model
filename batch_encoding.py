import torch
import collections
import dataclasses

@dataclasses.dataclass
class BatchEncoding(collections.UserDict):
    input_ids: torch.IntTensor
    attention_mask: torch.IntTensor

    def __post_init__(self):
        self.data = {"input_ids": self.input_ids, "attention_mask": self.attention_mask}

    def to(self, device: str | torch.device, dtype=None) -> "BatchEncoding":
        """The semantics is not correct, Pytorch .to is not inplace,
        but let's follow the HuggingFace transformers BatchEncoding,
        until something goes wrong.
        """
        if dtype is None:
            self.input_ids = self.input_ids.to(device=device)
            self.attention_mask = self.attention_mask.to(device=device)
        else:
            self.input_ids = self.input_ids.to(device=device, dtype=dtype)
            self.attention_mask = self.attention_mask.to(device=device, dtype=dtype)
        self.data = {"input_ids": self.input_ids, "attention_mask": self.attention_mask}
        return self