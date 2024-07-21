import time
import json
import os
import math
import shutil
from typing import List, Tuple

import torch
import safetensors.torch
import bitsandbytes

import custom_model
import glm3_tokenizer
import thucnnews_dataset
import batch_encoding
import batch_creator
from friendly_num import friendly_num


class Trainer:

    def __init__(self) -> None:
        self.tokenizer = glm3_tokenizer.GLM3Tokenizer()

        # Deeper config
        # -------------
        # embedding ratio: 0.07983494067795566
        # layers ratio: 0.8403288864902961
        #   * attn size ratio: 0.10093803868223616
        #   * MLP ratio: 0.7392922755046594
        # lm_head ratio: 0.07983494067795566
        # total:415532544 (~415M)
        #
        # self.config = custom_model.CustomModelConfig(
        #     vocab_size=self.tokenizer.vocab_size(),
        #     padding_token_id=self.tokenizer.token_pad_id,
        #     max_position_embeddings=4096,
        #     hidden_size=512,
        #     num_heads=16,
        #     MLP_intermediate=5000,
        #     num_layers=40,
        #     attention_dropout=0.1,
        #     dtype=torch.bfloat16,
        #     training=True,
        #     linear_imp = torch.nn.Linear
        # )

        # Wider config
        # ------------
        # embedding ratio: 0.14889637025887328
        # layers ratio: 0.7022049614505176
        #   * attn size ratio: 0.15060380783858626
        #   * MLP ratio: 0.5515276165963852
        # lm_head ratio: 0.14889637025887328
        # total:445598720 (~445M)
        # self.config = custom_model.CustomModelConfig(
        #     vocab_size=self.tokenizer.vocab_size(),
        #     padding_token_id=self.tokenizer.token_pad_id,
        #     max_position_embeddings=4096,
        #     hidden_size=1024,
        #     num_heads=16,
        #     MLP_intermediate=5000,
        #     num_layers=16,
        #     attention_dropout=0.1,
        #     dtype=torch.bfloat16,
        #     training=True,
        #     linear_imp = torch.nn.Linear
        # )

        # Medium
        # embedding ratio: 0.1030929639406133
        # layers ratio: 0.7938124810060924
        #   * attn size ratio: 0.1254560526721942
        #   * MLP ratio: 0.6682673260237617
        # lm_head ratio: 0.1030929639406133
        # total:442457664 (~442M)
        self.config = custom_model.CustomModelConfig(
            vocab_size=self.tokenizer.vocab_size(),
            padding_token_id=self.tokenizer.token_pad_id,
            max_position_embeddings=4096,
            hidden_size=704,
            num_heads=16,
            MLP_intermediate=5000,
            num_layers=28,
            attention_dropout=0.1,
            dtype=torch.bfloat16,
            training=True,
            linear_imp = torch.nn.Linear
        )



        self.model = custom_model.CustomLLamaModel(self.config).to(device='cuda')
        # self.model = custom_model.EncoderDecoderModel(self.config).to(device='cuda')

        self.lr = 1e-4
        self.betas = (0.9, 0.95)
        self.weight_decay = 0.01
        self.batch_size = 16
        self.train_seq_length = 192
        self.grad_accumulation_steps = 20
        self.clip_grad_max_norm = 1.0

        # log
        self.display_log_per_steps = 20
        self.record_log_per_steps = 20
        self.display_input_sample_per_steps = 600
        self.test_run_and_display_per_steps = 1200

        # checkpoint
        self.save_checkpoint_per_token_consumed = 10 * 1e6
        self.save_checkpoint_per_minutes = 30
        self.checkpoint_keep_recent = 100000

        self.saves_dir = os.path.join(os.path.dirname(__file__), "saves")
        self.overwrite_or_crash = "overwrite"

        print(self.config.get_total_params())
        
        # Status
        self.step = 0
        self.dataset_file_index = 0
        self.dataset_entry_index = 0
        self.token_consumed = 0
        self.training_logs = []
        self.last_save_time = time.time()
        
        self._load_newest_checkpoint()

        self.next_checkpoint_when_token_consumed = self.token_consumed + self.save_checkpoint_per_token_consumed

        self.dataset = thucnnews_dataset.ThuCnNewsSDataset()

        self.dataset_entry_generator = self.dataset.item_generator(self.dataset_file_index, self.dataset_entry_index)
        self.batch_creator = batch_creator.BatchCreator(
            self.batch_size,
            self.train_seq_length,
            self.tokenizer,
            self.dataset_entry_generator
        )

    def _load_newest_checkpoint(self):
        if not os.path.exists(self.saves_dir):
            return

        checkpoints = list(map(lambda x: x.name, os.scandir(self.saves_dir)))
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]), reverse=True)
        if checkpoints:
            print("checkpoints found! resume!")
            newest = checkpoints[0]
            with open(os.path.join(self.saves_dir, newest, "status.json")) as file:
                status_json = json.load(file)
            self.dataset_file_index = status_json["dataset_file_index"]
            self.dataset_entry_index = status_json["dataset_entry_index"]
            self.token_consumed = status_json["token_consumed"]

            model_path = os.path.join(self.saves_dir, newest, "model.safetensor")
            safetensors.torch.load_model(self.model, model_path)

    def _prune_checkpoint(self):
        checkpoints = list(map(lambda x: x.name, os.scandir(self.saves_dir)))
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]), reverse=True)
        to_remove = checkpoints[self.checkpoint_keep_recent:]
        for dir in to_remove:
            model_path = os.path.join(self.saves_dir, dir, "model.safetensor")
            if os.path.exists(model_path):
                os.remove(model_path)
            #shutil.rmtree(path)

    def _save_checkpoint(self):
        time_delta = time.time() - self.last_save_time
        self.last_save_time = time.time()
        print(f"saving checkpoint. time interval: {time_delta/60:.2f}min")

        dirname = f"checkpoint-{self.token_consumed}-{friendly_num(self.token_consumed)}"
        new_save_dir_path = os.path.join(self.saves_dir, dirname)
        if os.path.exists(new_save_dir_path) and self.overwrite_or_crash == "overwrite":
            shutil.rmtree(new_save_dir_path)
        elif os.path.exists(new_save_dir_path):
            raise FileExistsError("new_save_dir_path")
        os.makedirs(new_save_dir_path, exist_ok=True)

        # model
        model_path = os.path.join(new_save_dir_path, "model.safetensor")
        safetensors.torch.save_model(self.model, model_path)

        # status
        status_path = os.path.join(new_save_dir_path, "status.json")
        status_json = {
            "dataset_file_index": self.dataset_file_index,
            "dataset_entry_index": self.dataset_entry_index,
            "token_consumed": self.token_consumed
        }
        with open(status_path, 'w', encoding='utf8') as file:
            json.dump(status_json, file, indent=2)

        # log
        log_path = os.path.join(new_save_dir_path, "logs.json")
        with open(log_path, 'w', encoding='utf8') as file:
            json.dump(self.training_logs, file)
        self.training_logs = []

        print("checkpoint save complete.")

        self._prune_checkpoint()
   
    def _run_test(self):
        encoded = self.tokenizer.encode("奥运")
        input_ids = encoded.input_ids
        attn_mask = encoded.attention_mask

        start_seq_length = len(input_ids[0])

        generate_length = 50
        final_length = start_seq_length + generate_length

        input_ids_tensor = torch.zeros((1, final_length), dtype=self.tokenizer.dtype).to('cuda')
        attn_mask_tensor = torch.zeros((1, final_length), dtype=self.tokenizer.dtype).to('cuda')

        input_ids_tensor[0][:start_seq_length] = input_ids[0]
        attn_mask_tensor[0][:start_seq_length] = attn_mask[0]

        index = start_seq_length - 1

        self.model.config.training = False

        with torch.no_grad():
            while index < final_length - 1:
                logits = self.model.forward(input_ids_tensor, attn_mask_tensor)
                maxed = self.model.aggressive_decode(logits)
                input_ids_tensor[0][index + 1] = maxed[0][index]
                attn_mask_tensor[0][index + 1] = 1
                index += 1

        self.model.config.training = True

        print("test run output (greedy gen 50 token):", self.tokenizer.decode(input_ids_tensor))

    def _get_data_and_create_batches(self):
        batch, token_count, fileno, entryno = self.batch_creator.get_a_batch()
        self.dataset_file_index = fileno
        self.dataset_entry_index = entryno
        return [(batch, token_count)]
    
    def train(self):
        #optimizer = torch.optim.AdamW(
        optimizer = bitsandbytes.optim.AdamW8bit(
            params=self.model.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
            )
        
        batches: List[Tuple[batch_encoding.BatchEncoding, int]] = []

        while True:
            if len(batches) == 0:
                batches += self._get_data_and_create_batches()
            
            batch, token_count = batches.pop()
            batch = batch.to('cuda')
            
            self.token_consumed += token_count

            outputs = self.model.forward(**batch)
            loss = self.model.compute_loss(outputs, batch.input_ids)
            loss.backward()

            if self.step % self.grad_accumulation_steps == 0 and self.step != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_max_norm)
                optimizer.step()
                optimizer.zero_grad()

            if self.step % self.display_log_per_steps == 0:
                print(f"step: {self.step}, loss: {loss}, token consumed: {friendly_num(self.token_consumed)}, next checkpoint(token): {friendly_num(self.next_checkpoint_when_token_consumed)}")

            if self.step % self.record_log_per_steps == 0:
                log_item = {"loss": float(loss), "token_consumed": self.token_consumed}
                self.training_logs.append(log_item)

            if self.step % self.display_input_sample_per_steps == 0:
                input_text = self.tokenizer.decode(batch.input_ids[0])
                print("input sample:", input_text[0])

            if self.step % self.test_run_and_display_per_steps == 0:
                self._run_test()

            if time.time() - self.last_save_time > self.save_checkpoint_per_minutes*60:
                self.last_save_time = time.time()
                self._save_checkpoint()

            self.step += 1

    @staticmethod
    def count_param(model: torch.nn.Module):
        import numpy as np
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_params])
        return params

    
if __name__ == "__main__":
    TRAINER = Trainer()
    TRAINER.train()
    del TRAINER
    print('done')