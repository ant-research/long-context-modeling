from model.model_factory import create_model
from utils.model_loader import load_model
import torch
import random
import argparse
import math
import numpy as np
from tqdm import tqdm
import pickle
from reader.dataset_xsum import SummarizationDataset
import sys
from transformers import AutoTokenizer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('summary gen setup')
    cmd.add_argument('--model_path', type=str, required=True)
    cmd.add_argument('--model_type', type=str, required=True)
    cmd.add_argument('--config_path', type=str, required=True)
    cmd.add_argument('--vocab_dir', type=str, default='config/gpt2-small')
    cmd.add_argument('--corpus_path', type=str, required=True)
    cmd.add_argument('--output_path', type=str, required=True)
    cmd.add_argument('--max_input_len', type=int, default=8192)
    cmd.add_argument('--chunk_size', type=int, default=-1)
    cmd.add_argument('--insert_lmk', action='store_true')
    args = cmd.parse_args(sys.argv[1:])


    set_seed(1)
    model = create_model(args.model_type, args.config_path)
    load_model(model, args.model_path)
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()
    if args.model_type == 'llama_with_landmark':
        model.language_model.auto_insert_landmarks = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    pad_id = tokenizer.eos_token_id + 1
    assert pad_id != 0
    print(f'eos id: {tokenizer.eos_token_id}')

    dataset = SummarizationDataset(
        args.corpus_path, 
        tokenizer=tokenizer,
        eos_id=tokenizer.eos_token_id
    )

    def generate(model, **kwargs):
        if args.model_type == 'slide_window_lm':
            return model.language_model.generate(**kwargs)
        elif args.model_type == 'llama_with_landmark':
            return model.language_model.generate(chunk_topk=4, **kwargs)
        elif args.model_type == 'DRT':
            return model.generate(**kwargs)
    
    epoch_iterator = tqdm(dataset, desc="Iteration")
    output_ids = []
    for step, inputs in enumerate(epoch_iterator):
        input_ids = inputs['text']
        first_token_id = -1
        if args.chunk_size != -1:
            # padding
            
                # input_ids = F.pad(input_ids, (total_len - input_ids.shape[0], self._pad_id))
            if not args.insert_lmk:
                total_len = math.ceil(input_ids.shape[0] / args.chunk_size) * args.chunk_size + 1
                input_ids = np.concatenate(
                    (np.array([pad_id] * (total_len - input_ids.shape[0]), dtype=np.int32), input_ids),
                    dtype=np.int32
                )
                assert len(input_ids) % args.chunk_size == 1
                # print(input_ids)
                first_token_id=input_ids[-1]
                # print(f'first token id: {first_token_id}')
                input_ids = input_ids[:-1]
            else:
                mem_freq = model.config.mem_freq
                chunk_id = model.config.chunk_id
                input_ids = input_ids.tolist()
                for i in range(mem_freq, len(input_ids), mem_freq + 1):
                    input_ids.insert(i, chunk_id)
            inputs['text'] = np.array(input_ids)
        input_ids = torch.tensor(inputs['text'], device=device).unsqueeze(0)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            try:
                out_ids = np.array([0])
                print(first_token_id)
                if first_token_id != -1:
                    outputs = generate(
                        model, 
                        input_ids=input_ids, 
                        max_length=input_ids.shape[1] + 128, 
                        do_sample=True, 
                        top_k=2, 
                        first_token_id=first_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                else:
                    outputs = generate(
                        model, 
                        input_ids=input_ids, 
                        max_length=input_ids.shape[1] + 128, 
                        do_sample=True, 
                        top_k=2, 
                        eos_token_id=tokenizer.eos_token_id
                    )
                # out = model.language_model(outputs)
                out_ids = outputs.cpu().numpy()[0]
                if out_ids[-1] == tokenizer.eos_token_id:
                    out_ids = out_ids[input_ids.shape[1]:-1]
                if first_token_id != -1:
                    assert out_ids[0] == first_token_id
                    out_ids = out_ids[1:]
            except Exception as e:
                pass
            finally:
                output_ids.append(out_ids)

    with open(args.output_path, 'wb') as file_out:
        pickle.dump(output_ids, file_out)