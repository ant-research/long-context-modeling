import nltk
import argparse
import sys
import os
import torch
from fairseq.data.indexed_dataset import MMapIndexedDatasetBuilder

def preprocess_wiki103(tokenizer, dir, output_dir):
    # ASSUMPTION: dir/wiki.train.txt
    #             dir/wiki.valid.txt
    #             dir/wiki.test.txt
    files = ['test', 'valid', 'train']
    for corpus_name in files:
        path = os.path.join(dir, f'wiki.{corpus_name}.txt')
        data_file = os.path.join(output_dir, f'wiki.{corpus_name}.bin')
        index_file = os.path.join(output_dir, f'wiki.{corpus_name}.idx')
        mmap_builder = MMapIndexedDatasetBuilder(data_file)
        with open(path, mode='r') as f_in:
            for line in f_in:
                line = line.strip()
                if len(line) > 0: # document split
                    # tokenize to ids
                    sents = nltk.sent_tokenize(line)
                    for sent in sents:
                        ids = tokenizer.encode(sent)
                        ids[-1] = -ids[-1]
                        mmap_builder.add_item(torch.tensor(ids))
        mmap_builder.finalize(index_file)
        print(f'{corpus_name} done!')


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Preproceccing arguments')
    cmd.add_argument('--corpus-type', choices=["wikitext103", "openwebtext"])
    cmd.add_argument('--dir', type=str, required=True)
    cmd.add_argument('--output-dir', type=str, required=True)
    cmd.add_argument('--vocab-dir', type=str, required=True)

    args = cmd.parse_args(sys.argv[1:])

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    if args.corpus_type == 'wikitext103':
        preprocess_wiki103(tokenizer, args.dir, args.output_dir)
    elif args.corpus_type == 'openwebtext':
        preprocess_openwebtext(tokenizer, args.dir, args.output_dir)