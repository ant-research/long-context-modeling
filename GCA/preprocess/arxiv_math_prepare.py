import tiktoken
import os
import glob
import json
from tqdm import tqdm
import numpy as np


tokenizer = tiktoken.get_encoding("gpt2")
def build_dataset_proofpile(texts_dir, text_prefix="train"):

    def un_gz(file_name):
        import gzip
        f_name = file_name.replace(".gz", "")
        if not os.path.exists(f_name):
            gz_file = gzip.GzipFile(file_name)
            with open(f_name, "wb+") as f_in:
                f_in.write(gz_file.read())
            gz_file.close()
        return f_name

    print(f'text_prefix = {text_prefix}')
    texts = []
    
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
                token_ids = tokenizer.encode_ordinary(line['text'])
                texts += token_ids
                texts.append(tokenizer.eot_token)
    return np.array(texts, dtype=np.uint16)

dir_to_corpus = ''  # fill in the directory path of the corpus
np_arr = build_dataset_proofpile(f"{dir_to_corpus}/arxiv-math", text_prefix='dev')
np_arr.tofile(f'{dir_to_corpus}/arxiv-dev.bin')
np_arr = build_dataset_proofpile(f"{dir_to_corpus}/arxiv-math", text_prefix='train')
np_arr.tofile(f'{dir_to_corpus}/arxiv-train.bin')