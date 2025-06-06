# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""utils for loading text from disk"""
import os
import mmap
import pickle as pkl
import time
import numpy as np
from itertools import accumulate

import torch
from torch.multiprocessing import Lock


def get_lazy_path(path):
    """
    Gets directory path where lazy files are stored.
    """
    return os.path.splitext(path)[0] + '.lazy'


def exists_lazy(path, data_type='data'):
    """
    Check if we've already made a lazy version of this file for the `data_type` field.
    """
    if not os.path.exists(get_lazy_path(path)):
        return False
    contents = os.listdir(get_lazy_path(path))
    if data_type not in contents:
        return False
    if data_type + '.len.pkl' not in contents:
        return False
    return True


def get_scatter_path(path, scatter_rank):
    path = os.path.splitext(path)[0] + '.scatter'
    scatter_path = os.path.join(path, str(scatter_rank))
    return scatter_path


def exists_scatter(path, scatter_num=64, data_type='data'):
    for i in range(scatter_num):
        scatter_path = get_scatter_path(path, scatter_rank=i)
        if not exists_lazy(scatter_path, data_type=data_type):
            return False
    return True


class LazyWriter:
    def __init__(self, path, data_type, is_array=False, array_data_type=np.int32):
        lazypath = get_lazy_path(path)
        if not os.path.exists(lazypath):
            os.makedirs(lazypath)
        self.datapath = os.path.join(lazypath, data_type)
        self.lenpath = os.path.join(lazypath, data_type + '.len.pkl')
        self.array_data_type = array_data_type
        self.output = open(self.datapath, 'wb')
        self.lengths = []
        self.is_array = is_array

    @staticmethod
    def get_len_path(path, data_type):
        lazypath = get_lazy_path(path)
        return os.path.join(lazypath, data_type + '.len.pkl')

    def write(self, s):
        if isinstance(s, dict):
            s = s['text']
        if self.is_array:
            encoded = np.array(s, dtype=self.array_data_type).tobytes(order='C')
            self.output.write(encoded)
            self.lengths.append(len(s))
        else:
            encoded = s.encode('utf-8')
            self.output.write(encoded)
            self.lengths.append(len(encoded))

    def close(self):
        self.output.close()
        with open(self.lenpath, 'wb') as f:
            pkl.dump(self.lengths, f)


def split_strings(strings, start, chr_lens):
    """
    Split strings based on string lengths and given start.
    """
    return [strings[i - start:j - start] for i, j in zip([start] + chr_lens[:-1], chr_lens)]


class ProcessorTokenizer:
    """
    callable class that runs a preprocessing, as well as tokenization step,
    on input text.
    """

    def __init__(self, tokenizer, process_fn=None):
        self.tokenizer = tokenizer
        self.process_fn = process_fn

    def __call__(self, string):
        if self.tokenizer is not None:
            string = self.tokenizer(string, process_fn=self.process_fn)
        elif self.process_fn is not None:
            string = self.process_fn(string)
        return string


class LazyLoader(object):
    """
    Arguments:
        path: path to directory where array entries are concatenated into one big string file
            and the .len file are located
        data_type (str): Some datsets have multiple fields that are stored in different paths.
            `data_type` specifies which of these fields to load in this class

    Example of lazy loader directory structure:
    file.json
    file.lazy/
        data_type1
        data_type1.len.pkl
        data_type2
        data_type2.len.pkl
    """

    def __init__(self, path, data_type='data', is_array=False, array_data_type=np.int32):
        datapath = os.path.join(path, data_type)
        # get file where array entries are concatenated into one big string
        self._file = open(datapath, 'rb')
        self.file = self._file
        self.is_array = is_array
        self.array_data_type = array_data_type
        # memory map file if necessary

        lenpath = os.path.join(path, data_type + '.len.pkl')
        if os.path.exists(lenpath):
            self.file = np.memmap(self.file, dtype=array_data_type, mode='r', order='C')
            lens = pkl.load(open(lenpath, 'rb'))
        else:
            self.file = np.memmap(self.file, dtype=np.uint16, mode='r')
            lens = [self.file.shape[0]]

        self.lens = lens
        self.ends = list(accumulate(self.lens))
        
        print(f"total documents: {len(self.lens)}, total tokens: {self.ends[-1]}")
        self.total_tokens = self.ends[-1]
        self.read_lock = Lock()
        self._tokenizer = None
        self.is_lazy = True

    def __getitem__(self, index):
        """
        read file and splice strings based on string ending array `self.ends`
        """
        if not isinstance(index, slice):
            if index == 0:
                start = 0
            else:
                start = self.ends[index - 1]
            end = self.ends[index]
            rtn = self.file_read(start, end)
        else:
            # if slice, fetch strings with 1 diskread and then splice in memory
            chr_lens = self.ends[index]
            if index.start == 0 or index.start is None:
                start = 0
            else:
                start = self.ends[index.start - 1]
            stop = chr_lens[-1]
            strings = self.file_read(start, stop)
            rtn = split_strings(strings, start, chr_lens)
        return rtn

    def __len__(self):
        return len(self.ends)

    def file_read(self, start=0, end=None):
        """read specified portion of file"""
        data_type_size = np.dtype(self.array_data_type).itemsize
        # atomic reads to avoid race conditions with multiprocess dataloader
        self.read_lock.acquire()

        rtn = self.file[start:end]
        if self.is_array:
            rtn = rtn.copy()
        self.read_lock.release()
        return rtn
    
    def get_text_len(self, idx):
        prev_end = self.ends[idx - 1] if idx > 0 else 0
        return self.ends[idx] - prev_end