# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
import random
import torch
from torch.optim import Optimizer
from transformers import AdamW, AutoTokenizer, get_cosine_schedule_with_warmup
# from adabelief_pytorch import AdaBelief
import argparse
from functools import partial
import math
import sys
import pickle
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
from utils.model_loader import get_max_epoch_step, load_checkpoint
from torch.optim.lr_scheduler import LambdaLR
from utils.misc import gpt_token
from utils.model_loader import load_model
from utils.data_loader_utils import create_dataloader


def _get_cosine_schedule_with_warmup_lr_lambda_for_mlm(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float, mlm_steps_rate: float,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_training_steps * mlm_steps_rate:
        num_training_steps = num_training_steps * mlm_steps_rate
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    else:
        return 0.0


def get_cosine_schedule_with_warmup_for_mlm(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1, mlm_steps_rate=1.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda_for_mlm,
        mlm_steps_rate=mlm_steps_rate,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def eval_ppl(data_loader, model, device, amp_dtype=torch.float16, max_chunk_win=-1):

    # total_step = sum(map(lambda x: len(x), data_loaders))

    epoch_iterator = tqdm(data_loader, desc="Iteration")
    model.eval()

    total_ppl = 0
    steps = 0
    for inputs in epoch_iterator:
        steps += 1

        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)                
        with torch.cuda.amp.autocast(dtype=amp_dtype), torch.no_grad():
            result = model(**inputs)
        
        # if steps % 10 == 0 and steps > 0:
        #     print(result.causal_matrix)
        if hasattr(result, 'ar_loss'):
            total_ppl += result.ar_loss
        else:
            total_ppl += result.loss

    return total_ppl / steps

def passkey_eval(data_loader, model, device, amp_dtype=torch.float16, max_chunk_win=-1):

    # total_step = sum(map(lambda x: len(x), data_loaders))

    epoch_iterator = tqdm(data_loader, desc="Iteration")
    model.eval()

    steps = 0
    hit = 0
    total = 0
    for inputs in epoch_iterator:
        steps += 1
        key_pos = inputs['position']
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)                
        with torch.cuda.amp.autocast(dtype=amp_dtype), torch.no_grad():
            results = model(**inputs, output_logits=True)
        
        input_ids = inputs['input_ids']
        # if steps % 20 == 0:
        #     correct_predicts = (results.logits.argmax(dim=-1) == input_ids[:, 1:]).sum()
        #     print(f'correct rate: {correct_predicts / (results.logits.shape[0] * results.logits.shape[1])}')

        hit += (results.logits[:, key_pos - 1].argmax(dim=-1) == input_ids[:, key_pos]).sum()
        total += input_ids.shape[0]

    return -hit / total

class Trainer(object):
    def __init__(self, 
                 model,
                 mlm_prob,
                 tokenizer,
                 device,
                 logger,
                 is_master=True,
                 lr=5e-5):
        self.model = model
        self.mlm_prob = mlm_prob
        self.tokenizer = tokenizer
        self.is_master = is_master
        self.logger = logger

        self.device = device
        self.lr = lr

    def train(self, 
              data_loader, 
              mlm_prob,
              optimizer, 
              scheduler=None, 
              scaler=None,
              output_dir=None,
              amp_dtype=torch.float16,
              valid_dataloader=None,
              coeff_scheduler=None,
              temp_scheduler=None,
              mlm_steps_rate=1.0,
              max_chunk_win=-1,
              min_chunk_win=-1,
              log_steps=100, eval_steps=100, 
              save_steps=100, max_norm=1.0, 
              max_recover_step=-1, accumulation_steps=1,
              eval_func=None):

        total_step = len(data_loader)
        mlm_steps = mlm_steps_rate * total_step
        best_ppl = float('inf')
        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.train()
        chunk_win = -1

        for step, inputs in enumerate(epoch_iterator):
            if step > mlm_steps:
                mlm_prob = 0
            if step <= max_recover_step:
                continue
            max_recover_step = -1
            ratio = min(1.0, (step + 1) / total_step)

            # Find the nearest integer exponent to represent the ratio as a power of (1/2)
            n = math.floor(math.log(ratio, 1/2))

            # Calculate the approximated ratio using the found exponent
            # chunk_win = int(max((1/2) ** n * max_chunk_win, min_chunk_win))

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                result = self.model(**inputs, mask_probability=mlm_prob)
                # print((result.logits[:, -2, :].argmax(dim=-1) == inputs['input_ids'][:, -1]).sum())
            
            scaler.scale((result.total_loss) / accumulation_steps).backward()

            try:
                if (step + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    # print(f'is master: {self.is_master}, param0: {next(model.parameters())[:5]}')
            except RuntimeError as e:
                self.logger.error(e)
            finally:
                if (step + 1) % accumulation_steps == 0:
                    optimizer.zero_grad()

            if self.is_master:
                if step % log_steps == 0 and step > 0:
                    self.logger.info(f'progress:{step}/{total_step} loss: {result.total_loss} ar_loss: {result.ar_loss} ' + \
                        f'ae loss: {result.ae_loss}, ncp loss: {result.ncp_loss}, chunk_win: {chunk_win}, mlm_prob: {mlm_prob}')
                
                if step % save_steps == 0 and step > 0:
                    try:
                        torch.save(self.model.state_dict(),
                                os.path.join(output_dir, f"model0_{step}.bin"))
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, f"optimizer0_{step}.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, f"scheduler0_{step}.pt"))
                        
                        if scaler is not None:
                            torch.save(scaler.state_dict(), os.path.join(output_dir, f'scaler0_{step}.pt'))

                    except RuntimeError as e:
                        self.logger.error(f'{e}')

                if step % eval_steps == 0 and step > 0 and valid_dataloader is not None:
                    if eval_func is not None:
                        ppl = eval_func(valid_dataloader, model, self.device, amp_dtype=amp_dtype, max_chunk_win=chunk_win)
                        model.train()
                        if ppl < best_ppl:
                            best_ppl = ppl
                            self.logger.info(f'new best valid: {best_ppl}')
                            try:
                                torch.save(self.model.state_dict(), os.path.join(output_dir, f"model_best.bin"))
                            except:
                                pass
        
        if self.is_master:
            torch.save(self.model.state_dict(), os.path.join(output_dir, f"model.bin"))
            if eval_func is not None:
                ppl = eval_func(valid_dataloader, model, self.device, amp_dtype=amp_dtype, max_chunk_win=chunk_win)
                if ppl < best_ppl:
                    best_ppl = ppl
                    self.logger.info(f'new best valid: {best_ppl}')
                    try:
                        torch.save(self.model.state_dict(), os.path.join(output_dir, f"model_best.bin"))
                    except:
                        pass

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('slide win pretraining setup')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--batch_size', default=8, type=int, help='training batch size')
    cmd.add_argument('--valid_batch_size', default=1, type=int)
    cmd.add_argument('--epochs', default=10, type=int, help='epochs to pre-train')
    cmd.add_argument('--total_steps', default=-1, type=int)
    cmd.add_argument('--mlm_prob', default=0.15, type=float)
    cmd.add_argument('--max_valid_steps', default=-1, type=int)
    # cmd.add_argument('--mlm_epochs', default=8, type=int)
    cmd.add_argument('--lr', default=5e-5, type=float, help='learning rate')
    cmd.add_argument('--min_lr', default=5e-6, type=float, help='learning rate')
    cmd.add_argument('--config_path', required=True, type=str, help='config for r2d2')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='vocab path')
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--vocab_mapping_path', required=False, type=str)
    cmd.add_argument('--valid_corpus_path', required=False, default=None)
    cmd.add_argument('--accumulation_steps', type=int, default=1)
    cmd.add_argument('--model_type', default='gpt')
    cmd.add_argument('--output_dir', required=True, type=str, help='save dir')
    cmd.add_argument('--checkpoint_dir', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--finetune_path', default=None, required=False, type=str, help='model path for finetuning')
    cmd.add_argument('--num_workers', type=int, default=4)
    cmd.add_argument('--max_seq_len', type=int, default=1024)
    cmd.add_argument('--max_norm', type=float, default=1.0)
    cmd.add_argument('--max_chunk_win', type=int, default=-1)
    cmd.add_argument('--min_chunk_win', type=int, default=-1)
    cmd.add_argument('--seed', type=int, default=404)
    cmd.add_argument('--warm_up', type=float, default=0.01)
    cmd.add_argument('--weight_decay', type=float, default=1e-8)
    cmd.add_argument('--contriever_path', default=None, type=str)
    cmd.add_argument('--contriever_vocab_path', default=None, type=str)
    cmd.add_argument('--log_steps', default=100, type=int)
    cmd.add_argument('--fix_embeddings', action='store_true')
    cmd.add_argument('--save_steps', default=500, type=int)
    cmd.add_argument('--eval_steps', default=500, type=int)
    cmd.add_argument('--mlm_steps_rate', default=1.0, type=float)
    # cmd.add_argument('--passkey_retrieval', action='store_true')
    cmd.add_argument('--task_type', default=None, choices=[None, 'passkey_retrieval', 'summarization'])
    cmd.add_argument('--no_chunk_retrieval', action='store_true')
    cmd.add_argument('--gradient_checkpointing', action='store_true')
    cmd.add_argument('--max_sum_len', type=int, default=100)
    cmd.add_argument('--chunk_size', type=int, default=-1)
    cmd.add_argument('--disable_mlm', action='store_true')

    args = cmd.parse_args(sys.argv[1:])
    set_seed(args.seed)
    # torch.set_printoptions(profile='full')

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = -1

    if local_rank >= 0:
        torch.cuda.set_device(local_rank)  # for multi-process in a single machine with multiple GPUs.
        global_rank = local_rank
        while True:
            try:
                logging.info('init process group')
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                if torch.distributed.is_initialized():
                    break
            except ValueError:
                time.sleep(5)
            except:
                logging.error('Exit with unknown error')
                exit(-1)

        device = torch.device('cuda')
    else:
        global_rank = -1
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')

    is_master = local_rank == -1 or global_rank == 0
    if not os.path.exists(args.output_dir) and is_master:
        os.mkdir(args.output_dir)
    if is_master:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(args.output_dir, 'training_log.txt'), mode='a', encoding="utf-8")
        logger.addHandler(fh)
    else:
        logger = logging

    logger.info(f'initialize model on {global_rank}')
    vocab_mapping = None
    if args.vocab_mapping_path is not None:
        with open(args.vocab_mapping_path, mode='rb') as f_in:
            vocab_mapping = pickle.load(f_in)

    model = create_model(args.model_type, args.config_path, vocab_mapping=vocab_mapping, vocab_dir=args.vocab_dir,
                         contriever_vocab_path=args.contriever_vocab_path, contriever_path=args.contriever_path,
                         gradient_checkpointing=args.gradient_checkpointing)

    max_epoch = -1
    max_step = -1
    
    if args.checkpoint_dir is not None:
        max_epoch, max_step = get_max_epoch_step(args.checkpoint_dir, 'model*_*.bin')
        print(f'detect max_epoch: {max_epoch}, max_step:{max_step}')
        if max_epoch >= 0:
            logger.info(f'load from checkpoint, turn: {max_epoch}_{max_step}')
            load_model(model, os.path.join(args.checkpoint_dir, f'model{max_epoch}_{max_step}.bin'), strict=True)

    if args.finetune_path is not None:
        logger.info(f'finetune mode')
        load_model(model, args.finetune_path, strict=True)
        args.mlm_prob = 0.0
    if args.disable_mlm:
        args.mlm_prob = 0.0
    if args.fix_embeddings:
        print('fix embeddings')
        model.fix_encoder_and_embeddings()

    logger.info(f'move model to gpu:{global_rank}')
    model.to(device=device)

    # named_par_list = list(model.named_parameters())
    # unused_parser_indices = "131 132"
    # unused_parser_indices = [int(t) for t in unused_parser_indices.split()]
    # for idx in unused_parser_indices:
    #     print(named_par_list[idx][0])

    logger.info(f'start loading dataset on {global_rank}')
    task_type = args.task_type
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    valid_dataloader = None
    if args.valid_corpus_path is not None:
        valid_dataloader = create_dataloader(
            args.valid_corpus_path, 
            batch_size=args.valid_batch_size, 
            max_seq_len=args.max_seq_len, 
            epochs=1,
            total_steps=args.max_valid_steps,
            num_workers=1,
            task_type=task_type,
            shuffle=False,
            vocab_mapping=vocab_mapping,
            distributed=False,
            token_cnt=1,
            max_sum_len=args.max_sum_len,
            chunk_retrieval=not args.no_chunk_retrieval,
            chunk_size=args.chunk_size,
            tokenizer=tokenizer
        )

    if global_rank == -1:
        dataloader = create_dataloader(
            args.corpus_path, 
            args.batch_size, 
            args.max_seq_len, 
            args.epochs,
            seed=args.seed, 
            total_steps=args.total_steps,
            vocab_mapping=vocab_mapping,
            num_workers=args.num_workers,
            distributed=False,
            max_sum_len=args.max_sum_len,
            chunk_retrieval=not args.no_chunk_retrieval,
            chunk_size=args.chunk_size,
            tokenizer=tokenizer,
            task_type=task_type,
        )

        n_gpu = 1
        t_total = len(dataloader)
        warm_up_steps = args.warm_up * t_total

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9, 0.95))

    elif global_rank >= 0:
        n_gpu = 1
        dataloader = create_dataloader(
            args.corpus_path, 
            args.batch_size, 
            args.max_seq_len, 
            args.epochs,
            seed=args.seed, 
            total_steps=args.total_steps,
            vocab_mapping=vocab_mapping,
            num_workers=args.num_workers,
            distributed=True,
            max_sum_len=args.max_sum_len,
            chunk_retrieval=not args.no_chunk_retrieval,
            chunk_size=args.chunk_size,
            tokenizer=tokenizer,
            task_type=task_type,
        )

        t_total = len(dataloader)
        warm_up_steps = args.warm_up * t_total
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.001, betas=(0.9, 0.95))
       

        model = DDP(model)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warm_up_steps // args.accumulation_steps,
                                                num_training_steps=t_total // args.accumulation_steps)

    scheduler.base_lrs = [args.lr]
    scheduler.min_lrs = [args.min_lr]
    scaler = torch.cuda.amp.GradScaler()
    
    if max_epoch >= 0:
        modules = [optimizer, scheduler, scaler]
        files = [f'optimizer{max_epoch}_{max_step}.pt', f'scheduler{max_epoch}_{max_step}.pt', \
                f'scaler{max_epoch}_{max_step}.pt']
        load_checkpoint(modules, files, args.checkpoint_dir)
    
    trainer = Trainer(model, #enable_mlm_objective=enable_mlm_objective,
                      mlm_prob=args.mlm_prob,
                      device=device, tokenizer=tokenizer, logger=logger,
                      is_master=is_master)

    amp_dtype=torch.float16
    if torch.cuda.is_bf16_supported():
        amp_dtype=torch.bfloat16

    trainer.train(dataloader, args.mlm_prob, optimizer, 
                  scheduler=scheduler, 
                  scaler=scaler,
                  output_dir=args.output_dir,
                  amp_dtype=amp_dtype,
                  log_steps=args.log_steps, eval_steps=args.eval_steps,
                  mlm_steps_rate=args.mlm_steps_rate,
                  save_steps=args.save_steps,
                  accumulation_steps=args.accumulation_steps,
                  min_chunk_win=args.min_chunk_win,
                  max_chunk_win=args.max_chunk_win,
                  valid_dataloader=valid_dataloader,
                  max_recover_step=max_step, max_norm=args.max_norm,
                  eval_func=eval_ppl if args.task_type != 'passkey_retrieval' else passkey_eval)