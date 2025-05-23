import os
import numpy as np
import pickle
import codecs
import tarfile
from itertools import accumulate
import re
import argparse
import sys
import pyarrow.parquet as pq
from tqdm import tqdm
import glob
import json
from pathlib import Path


def get_file_name(text_path):
    file_name = os.path.split(text_path)[1]
    file_name = '.'.join(file_name.split('.')[:-1])
    return file_name

def build_dataset_wiki103(text_path, tokenizer, output_dir, buffer_size = 1024, max_len=-1):
    # index_path = os.path.join(output_dir, f'data.len.pkl')
    file_name = get_file_name(text_path)
    print(f'file name = {file_name}')
    content_path = os.path.join(output_dir, f'{file_name}.bin')
    index_path = os.path.join(output_dir, f'{file_name}.len.pkl')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    with open(text_path, mode='r') as f_in:
        for line in tqdm(f_in):
            line = line.strip()
            if len(line) == 0: # document split
                if len(current_buffer) > 0:
                    flush(current_buffer)
                    current_buffer = []
            else:
                ids = tokenizer.encode(line)
                current_buffer.append(ids)
                if max_len > 0 and len(ids) >= max_len:
                    #drop
                    continue
                    
            
        if len(current_buffer) > 0:
            flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)

def build_openwebtext_from_dir(texts_dir, tokenizer, output_dir, buffer_size = 1024, max_len=-1):
    # filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'data.len.pkl')
    content_path = os.path.join(output_dir, f'data')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    current_buffer = []
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = 0
        for ids in current_buffer:
            total_len += len(ids)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        for ids in current_buffer:
            np_memmap[current_offset: current_offset + len(ids)] = np.array(ids, dtype=np.int32, order='C')
            current_offset += len(ids)
            item_lens.append(len(ids))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    # with open(text_path, mode='r') as f_in:
    processed_files = 0
    for root, dirs, files in os.walk(texts_dir):
        for text_path in files:
            if processed_files % 10 == 0:
                print(f'processed: {processed_files} / {len(files)}', flush=True)
            processed_files += 1
            if text_path.endswith('_data'):
                tar = tarfile.open(os.path.join(root, text_path))
                for member in tar.getmembers():
                    file = tar.extractfile(member)
                    lines = file.readlines()
                    for line in lines:
                        line = line.decode().strip()
                        if len(line) == 0: # document split
                            if len(current_buffer) > 0:
                                flush(current_buffer)
                                current_buffer = []
                                # doc_num += 1
                                # if doc_num > 10:
                                #     break
                        else:
                            # tokenize to ids
                            sents = [line]
                            for sent in sents:
                                ids = tokenizer.encode(sent)
                                current_buffer.append(ids)
                        
                    if len(current_buffer) > 0:
                        flush(current_buffer)
    
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)

def build_dataset_pg19(texts_dir, tokenizer, output_dir, buffer_size = 1024, max_len=-1, text_prefix=None):
    if not text_prefix:
        text_prefix = 'data'
    index_path = os.path.join(output_dir, f'{text_prefix}.len.pkl')
    content_path = os.path.join(output_dir, f'{text_prefix}.bin')
    print(f'text_prefix = {text_prefix}')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = len(buffer)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        
        np_memmap[current_offset: current_offset + len(buffer)] = np.array(buffer, dtype=np.int32, order='C')
        current_offset += len(buffer)
        item_lens.append(len(buffer))
        
        return np_memmap, current_offset
    file_list = []
    for root, dirs, files in os.walk(texts_dir):
        # print(root, dirs, files)
        for file in files:
            file_list.append(os.path.join(root, file))
    for file_name in file_list:
        with open(file_name, "r") as f_in:
            current_buffer = []
            # for line in f_in:
            for line in tqdm(f_in):
                line = line.strip()
                current_buffer.extend(tokenizer.encode(line))
            flush(current_buffer)
            print(f'{file_name} processed successfully.')
    np_memmap.flush()

    # print(item_lens)
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)

def build_dataset_proofpile(texts_dir, tokenizer, output_dir, buffer_size = 1024, max_len=-1, text_prefix="train"):

    def un_gz(file_name):
        import gzip
        f_name = file_name.replace(".gz", "")
        if not os.path.exists(f_name):
            gz_file = gzip.GzipFile(file_name)
            with open(f_name, "wb+") as f_in:
                f_in.write(gz_file.read())
            gz_file.close()
        return f_name

    index_path = os.path.join(output_dir, f'{text_prefix}.len.pkl')
    content_path = os.path.join(output_dir, f'{text_prefix}.bin')
    print(f'text_prefix = {text_prefix}')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = len(buffer)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        
        np_memmap[current_offset: current_offset + len(buffer)] = np.array(buffer, dtype=np.int32, order='C')
        current_offset += len(buffer)
        item_lens.append(len(buffer))
        return np_memmap, current_offset
    
    gz_flie_pattern = os.path.join(texts_dir, f'*{text_prefix}*.gz')
    gz_file_list = glob.glob(gz_flie_pattern)
    for gz_file_name in gz_file_list:
        file_name = un_gz(gz_file_name)
        with open(file_name, "r") as f_in:
            print(f'{file_name} begins to be processed.')
            for line in tqdm(f_in):
                line = json.loads(line)
                meta = line['meta']
                if 'config' not in meta.keys():continue
                if meta['config'] != 'arxiv':continue
                token_ids = tokenizer.encode(line['text'])
                flush(token_ids)
            print(f'{file_name} processed successfully.')
    np_memmap.flush()

    # print(item_lens)
    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)


def build_dataset_wikipedia(texts_dir, tokenizer, output_dir, text_prefix='train'):
    file_list = []
    for root, dirs, files in os.walk(texts_dir):
        for file in files:
            if(file[:len(text_prefix)] == text_prefix):
                file_list.append(os.path.join(root, file))
    texts = []
    for file_name in file_list:
        data = pq.read_table(file_name)
        for doc in tqdm(data['text']):
            # print(doc)
            ids = tokenizer.encode(str(doc))
            texts += ids

    np.array(texts, dtype=np.uint16).tofile(os.path.join(output_dir, 'wikipedia.bin'))

def build_dataset_minipile(texts_dir, tokenizer, output_dir, buffer_size = 1024, max_len=-1, text_prefix='train'):
    # filename = os.path.splitext(os.path.basename(text_path))[0]
    index_path = os.path.join(output_dir, f'{text_prefix}.len.pkl')
    content_path = os.path.join(output_dir, f'{text_prefix}.bin')
    item_lens = []
    current_size = buffer_size
    current_offset = 0
    np_memmap = np.memmap(content_path, dtype=np.int32, mode='w+', order='C', shape=(current_size,))
    
    def flush(buffer):
        nonlocal np_memmap
        nonlocal current_size
        nonlocal current_offset
        total_len = len(buffer)
        while current_offset + total_len > current_size:
            # expand
            np_memmap.flush()
            next_size = (total_len // buffer_size + 1) * buffer_size + current_size
            np_memmap = np.memmap(content_path, dtype=np.int32, order='C', mode='r+', 
                                  shape=(next_size))
            current_size = next_size
        
        np_memmap[current_offset: current_offset + len(buffer)] = np.array(buffer, dtype=np.int32, order='C')
        current_offset += len(buffer)
        item_lens.append(len(buffer))
        
        return np_memmap, current_offset
    
    # doc_num = 0
    # with open(text_path, mode='r') as f_in:
    file_list = []
    for root, dirs, files in os.walk(texts_dir):
        for file in files:
            if(file[:len(text_prefix)] == text_prefix):
                file_list.append(os.path.join(root, file))
    for file_name in file_list:
        data = pq.read_table(file_name)
        for doc in tqdm(data['text']):
            # print(doc)
            ids = tokenizer.encode(str(doc))
            # current_buffer.append(ids)
            flush(ids)
        print(f'{file_name} processed successfully.')
    np_memmap.flush()

    with open(index_path, mode='wb') as index_out:
        pickle.dump(item_lens, index_out)


def print_dataset(index_path, data_path, tokenizer):
    np_memmap = np.memmap(data_path, dtype=np.int32, mode='r', order='C')
    with open(index_path, 'rb') as handle:
        item_lens = pickle.load(handle)
    
    ends = list(accumulate(item_lens))
    prev_end = 0
    for end in ends:
        print(tokenizer.convert_ids_to_tokens(np_memmap[prev_end : end]))
        prev_end = end


def create_directory_if_not_exists(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        print(f"Directory '{path}' is created successfully or already exists.")
    except Exception as e:
        print(f"An error occurred while creating the directory '{path}': {e}")

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Preprocess corpus components')
    cmd.add_argument('--mode', required=True, choices=['wikitext103', 'minipile', 'openwebtext', "pg19", 'arxiv-math', 'wikipedia'], default='wikitext103')
    cmd.add_argument('--vocab_dir', required=True, type=str, help='config for tokenizer')
    cmd.add_argument('--raw_corpus_path', required=True, type=str, help='path for raw corpus')
    cmd.add_argument('--output_path', required=True, type=str, help='path for preprocessed corpus')
    cmd.add_argument('--data_type', required=False, type=str, help='minipile needs to clarify the type of data, rather train or test or validation')

    args = cmd.parse_args(sys.argv[1:])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    create_directory_if_not_exists(args.output_path)
    if args.mode == "wikitext103":
        build_dataset_wiki103(args.raw_corpus_path, tokenizer, args.output_path, 
                buffer_size=4096, max_len=-1)
                # buffer_size=16384, max_len=-1, tokenize_sent=False)
    elif args.mode == "minipile":
        build_dataset_minipile(args.raw_corpus_path, tokenizer, args.output_path, 
                 buffer_size=16384, max_len=-1, text_prefix=args.data_type)
    elif args.mode == 'openwebtext':
        build_openwebtext_from_dir(args.raw_corpus_path, tokenizer, args.output_path, buffer_size=16384)
    elif args.mode == "pg19":
        build_dataset_pg19(args.raw_corpus_path, tokenizer, args.output_path, 
                 buffer_size=16384, max_len=-1)
    elif args.mode == "arxiv-math":
        build_dataset_proofpile(args.raw_corpus_path, tokenizer, args.output_path, 
                 buffer_size=16384, max_len=-1, text_prefix=args.data_type)
    elif args.mode == 'wikipedia':
        build_dataset_wikipedia(args.raw_corpus_path, tokenizer, args.output_path, text_prefix=args.data_type)
    else:
        raise Exception('Mode not suppport')