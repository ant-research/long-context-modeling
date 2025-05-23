import pickle
import os
import tiktoken
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from tqdm import tqdm


def preprocess_xsum(data_dir, tokenizer, output_path, eos_token_id):
    input_items = []
    for dirpath, dirs, files in os.walk(data_dir):
        # self.input_examples = self.get_examples(data_path_or_dir)
        for filename in files:
            path = os.path.join(dirpath, filename)
            table = pq.read_table(path)
            df = table.to_pandas()
            total_rows = len(df)
            
            for index, row in df.iterrows():
            # for input_example in self.input_examples:
                text = []
                summarytext = []

                ids = tokenizer.encode(row["document"])
                text += ids
                text = text + tokenizer.encode("|Summary") + tokenizer.encode(":")

                ids = tokenizer.encode(row["summary"])
                summarytext += ids + [eos_token_id]
                text = np.array(text, dtype=np.int32)
                summarytext = np.array(summarytext, dtype=np.int32)
                current_item = {"text": text, "summary": summarytext}
                input_items.append(current_item)
                if index % 100 == 0:
                    print(f'progress: {index}/{total_rows}')
                # if len(input_items) > 100:
                #     break
    with open(output_path, 'wb') as file:
        pickle.dump(input_items, file)

def preprocess_cnn(data_dir, tokenizer, output_path, eos_token_id):
    input_items = []
    for dirpath, dirs, files in os.walk(data_dir):
        # self.input_examples = self.get_examples(data_path_or_dir)
        for filename in files:
            path = os.path.join(dirpath, filename)
            table = pq.read_table(path)
            df = table.to_pandas()
            total_rows = len(df)
            
            for index, row in df.iterrows():
                text = []
                summarytext = []
                ids = tokenizer.encode(row["article"])
                text += ids
                text = text + tokenizer.encode("|Summary") + tokenizer.encode(":")

                ids = tokenizer.encode(row["highlights"])
                summarytext += ids + [eos_token_id]
                text = np.array(text, dtype=np.int32)
                summarytext = np.array(summarytext, dtype=np.int32)
                current_item = {"text": text, "summary": summarytext}
                input_items.append(current_item)
                if index % 100 == 0:
                    print(f'progress: {index}/{total_rows}')
    with open(output_path, 'wb') as file:
        pickle.dump(input_items, file)


if __name__ == '__main__':
    tokenizer = tiktoken.get_encoding("gpt2")
    gpt_tokenizer = AutoTokenizer.from_pretrained("config/gpt2-small")
    eos_token_id = gpt_tokenizer.eos_token_id
    assert eos_token_id is not None

    path_to_dir = ''  # fill in the path to directory containing the training corpus
    
    preprocess_xsum(
        f'{path_to_dir}/train', 
        tokenizer,
        f'{path_to_dir}/train.pkl',
        eos_token_id
    )
    preprocess_xsum(
        f'{path_to_dir}/dev', 
        tokenizer,
        f'{path_to_dir}/dev.pkl',
        eos_token_id
    )
    preprocess_xsum(
        f'{path_to_dir}/test', 
        tokenizer,
        f'{path_to_dir}/test.pkl', 
        eos_token_id
    )