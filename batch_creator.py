from typing import List, Generator

import torch

import thucnnews_dataset
import batch_encoding
import glm3_tokenizer

class BatchCreator:
    def __init__(self, batch_size: int, seq_len: int, tokenizer: glm3_tokenizer.GLM3Tokenizer, wudao_generator: Generator) -> None:
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        self.dataset_generator = wudao_generator

        self.encoded_items: List[List[torch.Tensor]] = []

    @staticmethod
    def _get_data(generator):
        dataset_out = next(generator)
        fileno = dataset_out["fileno"]
        entryno = dataset_out["entryno"]
        content = dataset_out["content"]
        return fileno, entryno, content


    def get_a_batch(self):
        """
        -> (batch, token_count, fileno, entryno)
        """

        fileno = 0
        entryno = 0

        while len(self.encoded_items) < self.batch_size:
            fileno, entryno, content = self._get_data(self.dataset_generator)
            encoded_content = self.tokenizer.encode_one_no_attn_no_special(content)
            encoded_splitted = encoded_content.split(self.seq_len)
            self.encoded_items.append(list(reversed(encoded_splitted)))

        batch_tensors_list = [tensor_list.pop() for tensor_list in self.encoded_items[:self.batch_size]]
        self.encoded_items = list(filter(lambda x: len(x) > 0, self.encoded_items))

        token_count = sum(map(len, batch_tensors_list))
        
        dtype = self.tokenizer.dtype

        input_ids = torch.zeros((self.batch_size, self.seq_len), dtype=dtype)
        attn_mask = torch.zeros((self.batch_size, self.seq_len), dtype=dtype)
        for i, x in enumerate(batch_tensors_list):
            input_ids[i][:len(x)] = x
            attn_mask[i].masked_fill_(input_ids[i] != 0, 1)
        batch = batch_encoding.BatchEncoding(input_ids, attn_mask)

        return batch, token_count, fileno, entryno
    
if __name__ == "__main__":
    def test():
        thucnnews = thucnnews_dataset.ThuCnNewsSDataset()
        dataset_gen = thucnnews.item_generator(0, 0)
        tokenizer = glm3_tokenizer.GLM3Tokenizer()
        batch_creator = BatchCreator(7, 200, tokenizer, dataset_gen)    
        for i in range(20000):
            batch, token_count, fileno, entryno = batch_creator.get_a_batch()
            if i > 20000 - 10:
                print(repr(tokenizer.decode(batch.input_ids)))

        
    test()