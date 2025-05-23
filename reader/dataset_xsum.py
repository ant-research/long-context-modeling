from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import pyarrow.parquet as pq
import numpy as np
import math
import random
import json
from utils.misc import align_spans, get_sentence_from_words
from transformers import AutoTokenizer
from typing import Dict, List
import pickle
import torch.nn.functional as F
import torch
import os


def insert_id_every_x_elements(arr, x, id_value):
    num_ids_to_insert = len(arr) // x
    
    new_length = len(arr) + num_ids_to_insert
    new_arr = np.empty(new_length, dtype=arr.dtype)
    
    new_arr[:new_length: x + 1] = id_value
    new_arr[np.arange(new_length) % (x + 1) != 0] = arr
    
    return new_arr

class SummarizationCollator:
    def __init__(self, max_len, max_sum_len, chunk_size=-1, pad_id=0, tokenizer=None):
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._max_sum_len = max_sum_len
        self._pad_id = pad_id
        self._chunk_size = chunk_size

    def fn(self, batch):
        input_ids_list = []
        labels_list = []
        for item in batch:
            input_ids = item['text']
            sum_ids = item['summary']
            if self._chunk_size != -1:
                #pad input_ids to chunk_size
                total_len = math.ceil(input_ids.shape[0] / self._chunk_size) * self._chunk_size + 1
                # input_ids = F.pad(input_ids, (total_len - input_ids.shape[0], self._pad_id))
                assert type(self._pad_id) == int
                input_ids = np.concatenate(
                    (np.array([self._pad_id] * (total_len - input_ids.shape[0]), dtype=np.int32), input_ids),
                    dtype=np.int32
                )
                assert len(input_ids) % self._chunk_size == 1
            if len(input_ids) + len(sum_ids) > self._max_len:
                # keep sum_ids <= max_sum_len
                sum_ids = sum_ids[-self._max_sum_len: ]
                input_ids = input_ids[:(self._max_len - len(sum_ids))]
            assert len(input_ids) + len(sum_ids) <= self._max_len
            labels = np.concatenate((input_ids, sum_ids), dtype=np.int32)
            labels = np.concatenate((labels, np.array((self._max_len - len(labels)) * [-100])))
            # padded_input_ids = np.concatenate(
            #     (input_ids, sum_ids, np.array([0] * (self._max_len - len(input_ids) - len(sum_ids)), dtype=np.int32)),
            #     dtype=np.int32
            # )
            # print(padded_input_ids.shape)
            # print(labels.shape)
            input_ids_list.append(labels)
            labels_list.append(np.where(labels == self._pad_id, -100, labels))
            # mod_array = insert_id_every_x_elements(labels_list[-1], self._chunk_size, 91)
            # mod_array[mod_array < 0] = 0
            # print(f'pad id: {self._pad_id}')
            # print(f'mod_array: {self._tokenizer.decode(mod_array)}')
        # print(padded_input_ids)
        # print(labels)
        return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long), "labels": torch.tensor(labels_list, dtype=torch.long)}


# {"text": np.array, "sentence_splits": list, "summary": np.array(summary will always be treated as one sentence)}
class SummarizationDataset(data.Dataset):
    
    def __init__(self, data_dir, tokenizer, eos_id, **kwargs):
        # data_name = data_dir + '/' + mode + '.json'
        self._tokenizer = tokenizer
        self._eos_id = eos_id
        self._lines = self.load_pickle(data_dir)

    def _to_ids(self, text):
        ids = self._tokenizer.encode(text)
        return ids

    def load_pickle(self, path):
        with open(path, 'rb') as file:
            input_items = pickle.load(file)

        return input_items

    def __getitem__(self, idx):
        return self._lines[idx]

    def __len__(self):
        return len(self._lines)

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("/ossfs/workspace/nas2/jipy/warpper/Generative-R2D2/data/newgpt2")
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    xsumdataset = SummarizationDataset(data_dir="/ossfs/workspace/antnlp/jipengyu/data/raw_gigaword/train.pkl", tokenizer=tokenizer, eos_id=0)
    lendict = {"0:200":0,"200:500":0,"500:1000":0,"1000:2000":0, '2000:3000': 0, '>3000': 0}
    maxl = -1
    totall = 0
    for item in xsumdataset:
        if 0 <= len(item["text"]) < 200:
            lendict["0:200"] += 1
        elif 200 <= len(item["text"]) < 500:
            lendict["200:500"] += 1
        elif 500 <= len(item["text"]) < 1000:
            lendict["500:1000"] += 1
        elif 1000 <= len(item["text"]) < 2000:
            lendict["1000:2000"] += 1
        elif 2000 <= len(item["text"]) < 3000:
            lendict["2000:3000"] += 1
        else:
            lendict[">3000"] += 1
        if len(item["text"]) > maxl:
            maxl = len(item["text"])
        totall += len(item["text"])
    meanl = totall/len(xsumdataset)
    print(maxl, meanl, lendict)
    # print(len(xsumdataset))
    # text1 = tokenizer.convert_ids_to_tokens(xsumdataset[0]["text"])
    # text2 = tokenizer.convert_ids_to_tokens(xsumdataset[1]["text"])
    # print(xsumdataset[0], text1)
    # print("---------------------------------------next--------------------------------------------------")
    # print(xsumdataset[1], text2)
    # for i in range(1001):
    #     if len(tokenizer.encode(xsumdataset[i]["text"])) < 40:
    #         print("find!")
    #         print(len(tokenizer.convert_ids_to_tokens(xsumdataset[i]["text"])))
    #         print(xsumdataset[i], tokenizer.convert_ids_to_tokens(xsumdataset[i]["text"]))
    #         break
    # print("not find!")

# train: 35243 498.46054985613944 {'0:200': 35236, '200:500': 93042, '500:1000': 56554, '>1000': 19185}
# test: 13584 473.9429983234801 {'0:200': 2330, '200:500': 5102, '500:1000': 2939, '>1000': 962}
# valid: 6494 465.09075659927606 {'0:200': 2364, '200:500': 5164, '500:1000': 2852, '>1000': 947}