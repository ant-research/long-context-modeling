# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
import random
import torch
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import argparse
import sys
from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from model.model_factory import create_model
from torch.utils.data.distributed import DistributedSampler
from utils.model_loader import get_max_epoch_step, load_checkpoint, load_model
from utils.misc import gpt_token
import pickle
from utils.data_loader_utils import create_slidewin_dataloader


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
            
            mean_loss = result.ar_loss
            total_ppl += mean_loss

            if steps % 100 == 0 and steps > 0:
                print(f'ppl: {total_ppl / steps}')

        return total_ppl / steps

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('NCR pretraining setup')
    cmd.add_argument('--config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--vocab_mapping_path', required=False, type=str)
    cmd.add_argument('--model_type', default='gpt')
    cmd.add_argument('--max_seq_len', default=16384, type=int)
    cmd.add_argument('--stride', default=1024, type=int)
    cmd.add_argument('--disable_retrieval', action='store_true')
    cmd.add_argument('--checkpoint_path', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--pool_size', type=int, default=4)
    cmd.add_argument('--contriever_vocab_path', required=True, type=str, help='contriever tokenizer path')
    cmd.add_argument('--contriever_path', required=True, type=str, help='contriever path')

    args = cmd.parse_args(sys.argv[1:])

    global_rank = -1
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    print(device)
    vocab_mapping = None
    if args.vocab_mapping_path is not None:
        with open(args.vocab_mapping_path, mode='rb') as f_in:
            vocab_mapping = pickle.load(f_in)
    # logger.info(f'initialize model on {global_rank}')

    model = create_model(args.model_type, args.config_path, vocab_mapping=vocab_mapping,
                         vocab_dir=args.vocab_dir,
                         contriever_path=args.contriever_path,
                         contriever_vocab_path=args.contriever_vocab_path)
    load_model(model, args.checkpoint_path, strict=False)
    
    # model.from_pretrain(args.checkpoint_path, strict=True)
    
    # logger.info(f'move model to gpu:{global_rank}')
    model.to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)


    dataloader = create_slidewin_dataloader(args.corpus_path, 1, epochs=1,
                                            vocab_mapping=vocab_mapping,
                                            max_seq_len=args.max_seq_len,
                                            shuffle=False, distributed=False, stride=args.stride)
    n_gpu = 1

    
    evaluator = Evaluator(model, device=device, tokenizer=tokenizer)

    amp_dtype=torch.float16
    if torch.cuda.is_bf16_supported():
        amp_dtype=torch.bfloat16
    
    
    print(f'perplexity: {evaluator.eval(dataloader, stride=args.stride, amp_dtype=amp_dtype, disable_retrieval=args.disable_retrieval)}')