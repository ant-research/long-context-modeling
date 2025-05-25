import argparse

import os
import sys
import evaluate
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from reader.dataset_xsum import SummarizationDataset
from rouge import Rouge



if __name__ == '__main__':
    cmd = argparse.ArgumentParser('summary eval setup')
    cmd.add_argument('--gen_path', type=str, required=True)
    cmd.add_argument('--gold_path', type=str, required=True)
    cmd.add_argument('--vocab_dir', type=str, required=True)
    args = cmd.parse_args(sys.argv[1:])
    metric = evaluate.load('rouge')
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    dataset = SummarizationDataset(
        args.gold_path, 
        tokenizer=tokenizer,
        eos_id=tokenizer.eos_token_id
    )

    pred_ids = pickle.load(open(args.gen_path, 'rb'))
    predictions = [' '.join(tokenizer.convert_ids_to_tokens(ids)) for ids in pred_ids]
    epoch_iterator = tqdm(dataset, desc="Iteration")
    print(predictions[0])
    references = []
    for step, inputs in enumerate(epoch_iterator):
        sum_ids = inputs['summary']
        if sum_ids[-1] == tokenizer.eos_token_id:
            sum_ids = sum_ids[:-1]
        references.append(' '.join(tokenizer.convert_ids_to_tokens(sum_ids)))

    if len(predictions) != len(references):
        print('warning: predictions and references have different lengths')
        assert len(predictions) < len(references)
        references = references[:len(predictions)]
    print(references[0])

    rouge = Rouge()
    scores = rouge.get_scores(predictions, references, avg=True)
    print(scores)
    # print(metric.compute(predictions=predictions, references=references))