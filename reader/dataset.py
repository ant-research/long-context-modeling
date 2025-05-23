from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random
import torch


class GPT2Dataset(data.Dataset):
    
    def __init__(self, ds,
                 max_seq_len=1024,
                 epochs=10,
                 weighted=True,
                 stride=-1,
                 num_samples=-1,
                 random_sampling=True,
                 **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.ds_len = len(self.ds)
        self.max_seq_len = max_seq_len
        if num_samples == -1:
            self.num_samples = (self.ds.total_tokens // max_seq_len) * epochs
        else:
            self.num_samples= num_samples

        self.weighted = weighted
        self.random_sampling = random_sampling
        self.weighting, self.total_len = None, None
        self.total_tokens = self.ds.total_tokens
        self.is_lazy = False
        self.stride=stride
        if stride != -1:
            self.num_samples = (self.ds.total_tokens // stride) * epochs
        if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
            self.is_lazy = True
        self.init_weighting()

    def init_weighting(self):
        if self.weighted:
            if self.is_lazy:
                lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
            else:
                lens = np.array([len(d['text']) if isinstance(d, dict)
                                 else len(d) for d in self.ds])
            self.total_len = np.sum(lens)
            print(f"Dataset document count {len(lens)}, token count {self.total_len}")
            self.weighting = list(accumulate(lens))
        else:
            self.weighting = None

    def get_prev_end(self, idx):
        return self.ds.ends[idx - 1] if idx > 0 else 0

    def get_weighted_samples(self, np_rng):
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return bisect_right(self.weighting, idx)
        else:
            return np_rng.randint(self.ds_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        if self.random_sampling:
            rng = random.Random(idx)
            rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
            # data_idx = self.get_weighted_samples(rng)

            # doc_len = self.ds.get_text_len(data_idx)
            # tokens_to_strip = max(0, doc_len - self.max_seq_len)
            # start = self.get_prev_end(data_idx) + rng.randint(tokens_to_strip + 1)
            # start = min(start, self.total_tokens - self.max_seq_len)
            # tokens = self.ds.file_read(start, start + self.max_seq_len)

            # return torch.tensor(tokens), (start, start + self.max_seq_len)

            start = rng.randint(self.total_tokens - self.max_seq_len - 1)
            tokens = self.ds.file_read(start, start + self.max_seq_len)
            return torch.tensor(np.array(tokens, dtype=np.int32)), (start, start + self.max_seq_len)

        else:
            if self.stride == -1:
                offset = (idx % (self.ds.total_tokens // self.max_seq_len) ) * self.max_seq_len
                offset = min(offset, self.total_tokens - self.max_seq_len)
                tokens = self.ds.file_read(offset, offset + self.max_seq_len)
                assert(tokens.shape[0] == self.max_seq_len)
                return torch.tensor(np.array(tokens, dtype=np.int32)), (offset, offset + self.max_seq_len)
            else:
                offset = idx * self.stride
                offset = min(offset, self.total_tokens - self.max_seq_len)
                tokens = self.ds.file_read(offset, offset + self.max_seq_len)
                assert(tokens.shape[0] == self.max_seq_len)
                return torch.tensor(np.array(tokens, dtype=np.int32)), (offset, offset + self.max_seq_len)
        

    def getidx(self, data_idx):
        token_ids = self.ds[data_idx]

        return token_ids