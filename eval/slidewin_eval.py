# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
import random
import torch
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import sys
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
import os
import logging
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from model.model_factory import create_model
from torch.utils.data.distributed import DistributedSampler
from utils.model_loader import get_max_epoch_step, load_checkpoint, load_model
from utils.misc import gpt_token
import pickle
from utils.data_loader_utils import create_slidewin_dataloader,create_passkey_dataloader


def generate_test_input_with_ldk(model, input_ids):
    # input_ids = tokenizer.encode(s, return_tensors="pt").to('cuda')
    mem_freq = model.config.mem_freq
    chunk_id = model.config.chunk_id
    # input_ids = input_ids[0].tolist()
    input_ids = input_ids[0].tolist()
    for i in range(mem_freq, len(input_ids), mem_freq + 1):
        input_ids.insert(i, chunk_id)
    return torch.tensor([input_ids], dtype=torch.long)

class Evaluator(object):
    def __init__(self, 
                 model,
                 tokenizer,
                 device):
        self.model = model
        self.tokenizer = tokenizer

        self.device = device

    def eval(self, 
             data_loader, 
             stride,
             distributed=False,
             disable_retrieval=False,
             max_steps=-1,
             amp_dtype=torch.bfloat16):

        # total_step = sum(map(lambda x: len(x), data_loaders))

        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.eval()

        total_ppl = 0
        steps = 0
        for inputs in epoch_iterator:
            steps += 1

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            pos_np = inputs['position']
            st = pos_np[0, 0]
            if st == 0:
                strd = -1
            else:
                strd = stride
            with torch.cuda.amp.autocast(dtype=amp_dtype), torch.no_grad():
                result = self.model(**inputs, stride=strd, disable_retrieval=disable_retrieval)
            
            if hasattr(result, 'ar_loss'):
                total_ppl += result.ar_loss
            else:
                total_ppl += result.loss

            if steps % 25 == 0 and steps > 0:
                print(total_ppl / steps)

            if max_steps > 0 and steps >= max_steps:
                break

        return total_ppl / steps

    def eval_passkey_retrieval(
            self, 
            data_loader, 
            stride,
            distributed=False,
            disable_retrieval=False,
            amp_dtype=torch.bfloat16
        ):

        # total_step = sum(map(lambda x: len(x), data_loaders))
        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.eval()

        steps = 0
        hit = 0
        total = 0
        for inputs in epoch_iterator:
            steps += 1
            key_pos = inputs['position']
            label = inputs['input_ids'].clone().to(device)
            inputs['input_ids'][:, key_pos] = 0  # remove correct answer in case of leakage
            inputs.pop('labels')
            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            with torch.cuda.amp.autocast(dtype=amp_dtype), torch.no_grad():
                results = model(**inputs, output_logits=True, segment_len=16 * 1024 // 64 * 65, offloading=True)
            
            # input_ids = inputs['input_ids']
            neg_idx = key_pos - label.shape[1]
            assert label[:, key_pos] == label[:, neg_idx]
            hit += (results.logits[:, neg_idx - 1].argmax(dim=-1) == label[:, key_pos]).sum()
            total += label.shape[0]
            del results
            torch.cuda.empty_cache()
            print(hit/total)

        return hit / total

    def eval_lmk_passkey_retrieval(
        self, 
        data_loader,
        amp_dtype=torch.bfloat16
    ):
    # The way calling forward doens't work for the landmark attention.
        # total_step = sum(map(lambda x: len(x), data_loaders))

        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.eval()
        model.language_model.auto_insert_landmarks = False

        steps = 0
        hit = 0
        total = 0
        for inputs in epoch_iterator:
            steps += 1
            key_pos = inputs['position']
            # for k, v in inputs.items():
            #     if v is not None and isinstance(v, torch.Tensor):
            #         inputs[k] = v.to(device)
            input_ids = inputs['input_ids'][:, :key_pos]
            # print(self.tokenizer.decode(input_ids[0]))
            lmk_input_ids = generate_test_input_with_ldk(self.model, input_ids)
            lmk_input_ids = lmk_input_ids.to(device)
            with torch.cuda.amp.autocast(dtype=amp_dtype), torch.no_grad():
                x = self.model.language_model.generate(lmk_input_ids, max_length=lmk_input_ids.shape[1] + 1, chunk_topk=4)
                print(f"{x[0][-1]}, {inputs['input_ids'][0, key_pos]}")
                print(f"{self.tokenizer.decode(x[0][-5:])} vs {self.tokenizer.decode(inputs['input_ids'][0, key_pos])}")
                if x[0][-1] == inputs['input_ids'][0, key_pos]:
                    hit += 1
            total += 1
            print(hit / total)

        return hit / total

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('NCR pretraining setup')
    cmd.add_argument('--config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--vocab_mapping_path', required=False, type=str)
    cmd.add_argument('--model_type', default='gpt')
    cmd.add_argument('--gradient_checkpointing', action="store_true")
    cmd.add_argument('--max_seq_len', default=16384, type=int)
    cmd.add_argument('--stride', default=1024, type=int)
    cmd.add_argument('--disable_retrieval', action='store_true')
    cmd.add_argument('--passkey_retrieval', default=None, choices=[None, 'single', 'multihop', 'multi'])
    cmd.add_argument('--no_chunk_retrieval', action='store_true')
    cmd.add_argument('--checkpoint_path', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--pool_size', type=int, default=4)
    cmd.add_argument('--max_steps', default=-1, type=int)  # for quick test
    cmd.add_argument('--contriever_vocab_path', required=False, type=str, help='contriever tokenizer path')
    cmd.add_argument('--contriever_path', required=False, type=str, help='contriever path')

    args = cmd.parse_args(sys.argv[1:])

    global_rank = -1
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    vocab_mapping = None
    if args.vocab_mapping_path is not None:
        with open(args.vocab_mapping_path, mode='rb') as f_in:
            vocab_mapping = pickle.load(f_in)
    # logger.info(f'initialize model on {global_rank}')

    print(f'{args.model_type}')
    model = create_model(args.model_type, args.config_path, vocab_mapping=vocab_mapping,
                         gradient_checkpointing=args.gradient_checkpointing,
                         vocab_dir=args.vocab_dir,
                         contriever_path=args.contriever_path,
                         contriever_vocab_path=args.contriever_vocab_path)
    print(model)
    load_model(model, args.checkpoint_path, strict=True)
    
    # model.from_pretrain(args.checkpoint_path, strict=True)
    
    # logger.info(f'move model to gpu:{global_rank}')
    model.to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)


    if args.passkey_retrieval is None:
        dataloader = create_slidewin_dataloader(args.corpus_path, 1, epochs=1,
                                                vocab_mapping=vocab_mapping,
                                                max_seq_len=args.max_seq_len,
                                                shuffle=False, distributed=False, stride=args.stride)
    else:
        dataloader = create_passkey_dataloader(
            tokenizer,
            args.corpus_path, 1, epochs=1,
            vocab_mapping=vocab_mapping,
            max_seq_len=args.max_seq_len,
            token_cnt=5,
            chunk_retrieval=not args.no_chunk_retrieval,
            task_type=args.passkey_retrieval,
            shuffle=False, distributed=False, stride=args.stride
        )
    n_gpu = 1

    
    evaluator = Evaluator(model, device=device, tokenizer=tokenizer)

    amp_dtype=torch.float16
    if torch.cuda.is_bf16_supported():
        amp_dtype=torch.bfloat16
    
    if not args.passkey_retrieval:
        ppl = evaluator.eval(dataloader, stride=args.stride, amp_dtype=amp_dtype, disable_retrieval=args.disable_retrieval, max_steps=args.max_steps)
        print(f'perplexity: {ppl}')
    else: # args.model_type != 'llama_with_landmark':
        acc = evaluator.eval_passkey_retrieval(dataloader, stride=args.stride, amp_dtype=amp_dtype, 
                                               disable_retrieval=args.disable_retrieval)
        print(f'passkey retrieval acc: {acc}')