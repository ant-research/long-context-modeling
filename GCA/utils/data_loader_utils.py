import torch.nn as nn
import torch
from torch.utils import data
import torch.nn.functional as F
import json
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler
from reader.dataset import GPT2Dataset
from reader.lazy_loader import LazyLoader
import numpy as np
import random


def collate_fn(sent_tensors):
    '''
        input_list: [{"text": ..., "sentence_splits":...},...]
    '''
    # split sentence ids with negative ids
    chunk_ids_list = []
    # retrieval_attn_masks = np.zeros((N, max_input_len, 2 * N * max_input_len), dtype=bool)  # (N, L, 2NL)
    for group_i, sent_tensor in enumerate(sent_tensors):
        chunk_ids_list.append(sent_tensor.abs())

    # padding
    return {"input_ids": torch.stack(chunk_ids_list)}


class CollatorWithMapping:
    def __init__(self, vocab_mapping):
        self.vocab_mapping = vocab_mapping

    def collate_fn_with_position(self, tensors_and_positions):
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        # split sentence ids with negative ids
        chunk_ids_list = []
        position_list = []
        # retrieval_attn_masks = np.zeros((N, max_input_len, 2 * N * max_input_len), dtype=bool)  # (N, L, 2NL)
        for group_i, tensor_and_position in enumerate(tensors_and_positions):
            token_ids, position = tensor_and_position
            if self.vocab_mapping is not None:
                chunk_ids_list.append(torch.tensor([self.vocab_mapping[token_id.item()] if token_id.item() in self.vocab_mapping else -100 for token_id in token_ids.abs()], dtype=torch.long))
            else:
                chunk_ids_list.append(torch.tensor(token_ids.abs(), dtype=torch.long))
            position_list.append(position)
        # padding
        return {"input_ids": torch.stack(chunk_ids_list), "position": np.array(position_list),
                "labels": torch.stack(chunk_ids_list)}

class MultiPasskeyRetrieval:
    def __init__(self, tokenizer, chunk_size=64, vocab_low=0, vocab_high=50256, hop=1, noise=5, chunk_retrieval=True, token_cnt=5):
        # self._qa = '|The final node from {} is {}'
        self._qa1 = tokenizer.encode("|The value of ")
        self._qa2 = tokenizer.encode("is")
        self._comma = tokenizer.encode(',')
        self._needle1 = tokenizer.encode('|One of the special magic numbers for ')
        self._needle2 = tokenizer.encode(' is ')
        self._needle3 = tokenizer.encode('|')
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_win_size = -1
        self._low = vocab_low
        self._high = vocab_high
        self._hop = hop
        self._noise = noise
        self._token_cnt = 5
        self._total_needles = (hop + 1) * (noise + 1)
        self._chunk_retrieval = chunk_retrieval
        print(f'needle cnt: {self._total_needles}, chunk_retrieval: {chunk_retrieval}, chunk_size: {chunk_size}')

    def update_chunk_size(self, chunk_size):
        self._chunk_win_size = chunk_size

    def fn(self, tensors_and_positions):
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        # split sentence ids with negative ids
        chunk_ids_list = []
        position_list = []
        # retrieval_attn_masks = np.zeros((N, max_input_len, 2 * N * max_input_len), dtype=bool)  # (N, L, 2NL)
        for group_i, tensor_and_position in enumerate(tensors_and_positions):
            token_ids, position = tensor_and_position
            token_ids = token_ids.numpy()
            # randomly insert passkey from 0 to -128
            rng = random.Random(token_ids[0] * token_ids[-1])
            rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
            total_len = len(token_ids)
            assert total_len % self._chunk_size == 0
            # if self._token_cnt > 1:
            #     # training mode
            #     hop = rng.randint(1, self._hop + 1)
            #     noise = rng.randint(0, self._noise + 1)
            # else:
            hop = self._hop
            noise = self._noise
            total_needles = (hop + 1) * (noise + 1)
            # print(f'hop {hop}, noise: {noise}')
            var_ids = []
            for _ in range(total_needles):
                rand_var = np.array(rng.randint(self._low, self._high, size=self._token_cnt))
                var_ids.append(rand_var)
            
            inserted_ids = []
            for group_idx in range(noise + 1):
                for hop_idx in range(hop):
                    # ndl_str = self.needles.format(
                    #     var_strs[group_idx * (self._hop + 1) + hop_idx], 
                    #     var_strs[group_idx * (self._hop + 1) + hop_idx + 1]
                    # )
                    # ndl_str = ' '.join(ndl_str.split())
                    ndl_ids = np.concatenate((
                        self._needle1,
                        var_ids[group_idx * (hop + 1) + hop_idx],
                        self._needle2,
                        var_ids[group_idx * (hop + 1) + hop_idx + 1],
                        self._needle3
                    ))
                    inserted_ids.append(ndl_ids)
                    assert (inserted_ids[-1][-self._token_cnt-1:-1] == var_ids[group_idx * (hop + 1) + hop_idx + 1]).all(), f'{inserted_ids[-1][-self._token_cnt-1:-1]} vs {var_ids[group_idx * (hop + 1) + hop_idx + 1]}'
            
            np.random.shuffle(inserted_ids)
            # qa = self._qa.format(var_strs[0], var_strs[self._hop])
            # qa = ' '.join(qa.split())
            # qa_ids = self._tokenizer.encode(qa)

            cat_ids = [self._qa1, var_ids[0], self._qa2]

            for idx in range(1, self._hop + 1):
                cat_ids.append(var_ids[idx])
                cat_ids.append(self._comma)

            cat_ids.pop(-1)

            qa_ids = np.concatenate((cat_ids))
            answer_len = self._hop * self._token_cnt + self._hop - 1
            if self._chunk_size != -1:
                assert self._chunk_size > len(qa_ids)
                token_ids = insert_ids(token_ids.tolist(), inserted_ids, reserve_length=self._chunk_size - 1 - answer_len + len(qa_ids))
            else:
                token_ids = insert_ids(token_ids.tolist(), inserted_ids)

            new_array = np.array(token_ids)
           
            if self._chunk_retrieval:
                # print(f'qa: {self._tokenizer.decode(qa_ids)}')
                new_array = np.insert(new_array, total_len - self._chunk_size - len(qa_ids) + answer_len + 1, qa_ids)
                new_array = new_array[:total_len]
                new_array[-(self._chunk_size - answer_len - 1):] = -100
            else:
                new_array = np.insert(new_array, total_len - len(qa_ids), qa_ids)
                new_array = new_array[:total_len]
        
            chunk_ids_list.append(torch.tensor(new_array, dtype=torch.long))
            # mod_array = insert_id_every_x_elements(new_array, self._chunk_size, 91)
            # mod_array[mod_array < 0] = 0
            # print(f'mod_array: {self._tokenizer.decode(mod_array)}')
            # print(f'final pos token: {self._tokenizer.decode(new_array[total_len - (self._chunk_size - answer_len): total_len - (self._chunk_size - answer_len) + 1])}')
            # assert qa_ids[-answer_len - 1] == self._qa2[0]

        if self._chunk_retrieval:
            final_pos = total_len - (self._chunk_size - answer_len)
        else:
            final_pos = total_len - 1
        # padding
        return {"input_ids": torch.stack(chunk_ids_list), "position": final_pos, "labels": torch.stack(chunk_ids_list)}

class PasskeyRetrieval:
    def __init__(self, tokenizer, chunk_size=64, vocab_low=0, vocab_high=50256, token_cnt=10, chunk_retrieval=True):
        self._pass_key_token_ids = tokenizer.encode('The passkey is:')
        self._wrapper_token_id = tokenizer.encode('|')
        self._pass_key_prompt_ids = tokenizer.encode('|What is the passkey? The passkey is')
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_win_size = -1
        self._low = vocab_low
        self._high = vocab_high
        self._token_cnt = token_cnt
        self._chunk_retrieval = chunk_retrieval
        print(f'token cnt: {token_cnt}, chunk_retrieval: {chunk_retrieval}, chunk_size: {chunk_size}')

    def update_chunk_size(self, chunk_size):
        self._chunk_win_size = chunk_size

    def fn(self, tensors_and_positions):
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        # split sentence ids with negative ids
        chunk_ids_list = []
        position_list = []

        total_len = len(tensors_and_positions[0][0])
        if self._chunk_retrieval:
            final_pos = total_len - (self._chunk_size - 1)
        else:
            final_pos = total_len - 1
        # retrieval_attn_masks = np.zeros((N, max_input_len, 2 * N * max_input_len), dtype=bool)  # (N, L, 2NL)
        for group_i, tensor_and_position in enumerate(tensors_and_positions):
            token_ids, position = tensor_and_position
            token_ids = token_ids.numpy()
            # randomly insert passkey from 0 to -128
            rng = random.Random(token_ids[0] * token_ids[-1])
            rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
            total_len = len(token_ids)
            assert total_len % self._chunk_size == 0
            # passkey = chr(ord('a') + rng.randint(25))
            # passkey_id = self._tokenizer.encode(passkey)
            passkey_ids = np.array(rng.randint(self._low, self._high, size=self._token_cnt))
            passkey_ids_ = np.concatenate((self._wrapper_token_id, self._pass_key_token_ids, passkey_ids, self._wrapper_token_id))
            # print(passkey_ids_)
            # if self._chunk_win_size == -1:
            start = rng.randint(len(token_ids) - 2 * self._chunk_size - len(passkey_ids_))  # at least one chunk away
            # else:
            # start = rng.randint(len(token_ids) - (self._chunk_win_size + 1) * self._chunk_size, len(token_ids) - 2 * self._chunk_size - len(passkey_ids))
            # print(type(token_ids))
            new_array = np.insert(token_ids, start, passkey_ids_)
            prompt_ids = np.concatenate((self._pass_key_prompt_ids, passkey_ids))
            # print(prompt_ids)
            if self._chunk_retrieval:
                new_array = np.insert(new_array, total_len - self._chunk_size - len(prompt_ids) + 1 + len(passkey_ids), prompt_ids)
                new_array = new_array[:total_len]
                new_array[-(self._chunk_size - 1 - len(passkey_ids)):] = -100
            else:
                new_array = np.insert(new_array, total_len - len(prompt_ids), prompt_ids)
                new_array = new_array[:total_len]
            
            # if not self._chunk_retrieval:
            #     print(new_array)
            #     print(self._tokenizer.decode(new_array))
            chunk_ids_list.append(torch.tensor(new_array, dtype=torch.long))
            position_list.append(position)

            # mod_array = insert_id_every_x_elements(new_array, self._chunk_size, 91)
            # mod_array[mod_array < 0] = 0
            # print(f'mod_array: {self._tokenizer.decode(mod_array)}, final pos token: {self._tokenizer.decode(new_array[final_pos])}')

        # padding
        return {"input_ids": torch.stack(chunk_ids_list), "position": final_pos, "labels": torch.stack(chunk_ids_list)}

def insert_ids(s, insert_ids, reserve_length=0):
    original_length = len(s)
    n = len(insert_ids)
    insert_len_sum = sum(map(lambda x: len(x), insert_ids))
    assert original_length - insert_len_sum - reserve_length >= n
    positions = sorted(random.sample(range(original_length - insert_len_sum - reserve_length), n))

    for pos, insert_str in zip(reversed(positions), reversed(insert_ids)):
        s = s[:pos] + insert_str.tolist() + s[pos:]

    return s

def insert_id_every_x_elements(arr, x, id_value):
    num_ids_to_insert = len(arr) // x
    
    new_length = len(arr) + num_ids_to_insert
    new_arr = np.empty(new_length, dtype=arr.dtype)
    
    new_arr[:new_length: x + 1] = id_value
    new_arr[np.arange(new_length) % (x + 1) != 0] = arr
    
    return new_arr

class MultiHopRetrieval:
    def __init__(self, tokenizer, chunk_size=64, vocab_low=0, vocab_high=50256, hop=2, noise=1, token_cnt=5, chunk_retrieval=True):
        # self._qa = '|The final node from {} is {}'
        self._qa1 = tokenizer.encode("|The path from")
        self._qa2 = tokenizer.encode("is")
        self._comma = tokenizer.encode(',')
        self._needle1 = tokenizer.encode('|DEF')
        self._needle2 = tokenizer.encode('->')
        self._needle3 = tokenizer.encode('|')
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_win_size = -1
        self._low = vocab_low
        self._high = vocab_high
        self._hop = hop
        self._noise = noise
        self._token_cnt = token_cnt
        self._total_needles = (hop + 1) * (noise + 1)
        self._chunk_retrieval = chunk_retrieval
        print(f'needle cnt: {self._total_needles}, chunk_retrieval: {chunk_retrieval}, chunk_size: {chunk_size}')

    def update_chunk_size(self, chunk_size):
        self._chunk_win_size = chunk_size

    def fn(self, tensors_and_positions):
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        # split sentence ids with negative ids
        chunk_ids_list = []
        position_list = []
        # retrieval_attn_masks = np.zeros((N, max_input_len, 2 * N * max_input_len), dtype=bool)  # (N, L, 2NL)
        for group_i, tensor_and_position in enumerate(tensors_and_positions):
            token_ids, position = tensor_and_position
            token_ids = token_ids.numpy()
            # randomly insert passkey from 0 to -128
            rng = random.Random(token_ids[0] * token_ids[-1])
            rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
            total_len = len(token_ids)
            assert total_len % self._chunk_size == 0
            # if self._token_cnt > 1:
            #     # training mode
            #     hop = rng.randint(1, self._hop + 1)
            #     noise = rng.randint(0, self._noise + 1)
            # else:
            hop = self._hop
            noise = self._noise
            total_needles = (hop + 1) * (noise + 1)
            # print(f'hop {hop}, noise: {noise}')
            var_ids = []
            for _ in range(total_needles):
                rand_var = np.array(rng.randint(self._low, self._high, size=self._token_cnt))
                var_ids.append(rand_var)
            
            inserted_ids = []
            for group_idx in range(noise + 1):
                for hop_idx in range(hop):
                    # ndl_str = self.needles.format(
                    #     var_strs[group_idx * (self._hop + 1) + hop_idx], 
                    #     var_strs[group_idx * (self._hop + 1) + hop_idx + 1]
                    # )
                    # ndl_str = ' '.join(ndl_str.split())
                    ndl_ids = np.concatenate((
                        self._needle1,
                        var_ids[group_idx * (hop + 1) + hop_idx],
                        self._needle2,
                        var_ids[group_idx * (hop + 1) + hop_idx + 1],
                        self._needle3
                    ))
                    inserted_ids.append(ndl_ids)
                    assert (inserted_ids[-1][-self._token_cnt-1:-1] == var_ids[group_idx * (hop + 1) + hop_idx + 1]).all(), f'{inserted_ids[-1][-self._token_cnt-1:-1]} vs {var_ids[group_idx * (hop + 1) + hop_idx + 1]}'
            
            np.random.shuffle(inserted_ids)
            # qa = self._qa.format(var_strs[0], var_strs[self._hop])
            # qa = ' '.join(qa.split())
            # qa_ids = self._tokenizer.encode(qa)

            cat_ids = [self._qa1, var_ids[0], self._qa2]

            for idx in range(1, self._hop + 1):
                cat_ids.append(var_ids[idx])
                cat_ids.append(self._comma)

            cat_ids.pop(-1)

            qa_ids = np.concatenate((cat_ids))
            answer_len = self._hop * self._token_cnt + self._hop - 1
            if self._chunk_size != -1:
                assert self._chunk_size > len(qa_ids)
                token_ids = insert_ids(token_ids.tolist(), inserted_ids, reserve_length=self._chunk_size - 1 - answer_len + len(qa_ids))
            else:
                token_ids = insert_ids(token_ids.tolist(), inserted_ids)

            new_array = np.array(token_ids)
           
            if self._chunk_retrieval:
                # print(f'qa: {self._tokenizer.decode(qa_ids)}')
                new_array = np.insert(new_array, total_len - self._chunk_size - len(qa_ids) + answer_len + 1, qa_ids)
                new_array = new_array[:total_len]
                new_array[-(self._chunk_size - answer_len - 1):] = -100
            else:
                new_array = np.insert(new_array, total_len - len(qa_ids), qa_ids)
                new_array = new_array[:total_len]
        
            chunk_ids_list.append(torch.tensor(new_array, dtype=torch.long))
            # mod_array = insert_id_every_x_elements(new_array, self._chunk_size, 91)
            # mod_array[mod_array < 0] = 0
            # print(f'mod_array: {self._tokenizer.decode(mod_array)}')
            # print(f'final pos token: {self._tokenizer.decode(new_array[total_len - (self._chunk_size - answer_len): total_len - (self._chunk_size - answer_len) + 1])}')
            assert qa_ids[-answer_len - 1] == self._qa2[0]
            # exit()

        if self._chunk_retrieval:
            final_pos = total_len - (self._chunk_size - answer_len)
        else:
            final_pos = total_len - 1
        # padding
        return {"input_ids": torch.stack(chunk_ids_list), "position": final_pos, "labels": torch.stack(chunk_ids_list)}

def format_regularization(json_data):
    id_batches = []
    for item in json_data:
        if isinstance(item, int):
            id_batches.append(item)
        elif isinstance(item, dict):
            id_batches.extend(item['doc_seq'])
        else:
            raise Exception('unrecognized format')
    
    return id_batches

def create_slidewin_dataloader(indexed_data_path, batch_size, max_seq_len, epochs, seed=0, total_steps=-1,
                               num_workers=1, shuffle=True, distributed=True, stride=-1, vocab_mapping=None):
    ds = LazyLoader(indexed_data_path)
    world_size = 1 if not distributed else dist.get_world_size()
    dataset = GPT2Dataset(ds, max_seq_len=max_seq_len,
                          epochs=epochs,
                          weighted=True,
                          stride=stride,
                          num_samples=total_steps * batch_size * world_size if total_steps > 0 else -1,
                          random_sampling=shuffle)
    print(len(dataset))
    collator = CollatorWithMapping(vocab_mapping)
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           collate_fn=collator.collate_fn_with_position,
                           sampler=DistributedSampler(dataset, shuffle=shuffle) if distributed else SequentialSampler(dataset),
                           num_workers=num_workers)

def create_passkey_dataloader(
    tokenizer, 
    indexed_data_path, 
    batch_size, 
    max_seq_len, 
    epochs, 
    seed=0, 
    total_steps=-1, 
    chunk_size=64,
    num_workers=1, 
    shuffle=True, 
    distributed=True, 
    stride=-1, 
    vocab_mapping=None,
    vocab_low=0, 
    vocab_hi=50256, 
    token_cnt=-1, 
    chunk_retrieval=True,
    task_type=None
):
    ds = LazyLoader(indexed_data_path)
    world_size = 1 if not distributed else dist.get_world_size()
    dataset = GPT2Dataset(ds, max_seq_len=max_seq_len,
                          epochs=epochs,
                          weighted=True,
                          stride=stride,
                          num_samples=total_steps * batch_size * world_size if total_steps > 0 else -1,
                          random_sampling=shuffle)
    if task_type == "single" or task_type is None:
        token_cnt = token_cnt if token_cnt != -1 else 10
        collator = PasskeyRetrieval(tokenizer, chunk_size=chunk_size, vocab_low=vocab_low, vocab_high=vocab_hi, 
                                    token_cnt=token_cnt, chunk_retrieval=chunk_retrieval)
    elif task_type == "multi":
        token_cnt = token_cnt if token_cnt != -1 else 10
        collator = MultiPasskeyRetrieval(tokenizer, chunk_size=chunk_size, vocab_low=vocab_low, vocab_high=vocab_hi, 
                                    token_cnt=token_cnt, chunk_retrieval=chunk_retrieval)
    elif task_type == "multihop":
        token_cnt = token_cnt if token_cnt != -1 else 5
        collator = MultiHopRetrieval(tokenizer, chunk_size=chunk_size, vocab_low=vocab_low, vocab_high=vocab_hi, 
                                    token_cnt=token_cnt, chunk_retrieval=chunk_retrieval)
    else:
        raise Exception(f'task type not supported: {task_type}')
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           collate_fn=collator.fn,
                           sampler=DistributedSampler(dataset, shuffle=shuffle) if distributed else SequentialSampler(dataset),
                           num_workers=num_workers)

 
def create_dataloader(indexed_data_path, batch_size, epochs, seed=0, num_workers=2, shuffle=True, distributed=True):
    import fairseq.data.data_utils as data_utils
    from fairseq.data.indexed_dataset import MMapIndexedDataset
    dataset = MMapIndexedDataset(indexed_data_path)
    data_loaders = []
    for epoch in range(epochs):
        data_loaders.append(data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            collate_fn=collate_fn,
                                            sampler=DistributedSampler(dataset, shuffle=shuffle) if distributed else SequentialSampler(dataset),
                                            num_workers=num_workers))

    return data_loaders


def create_dataloader_with_predefined_batches(indexed_data_path, predefined_batch_ids_dir, batch_size, epochs, shard_id=0, total_shards=1, seed=0, num_workers=2):
    import fairseq.data.data_utils as data_utils
    from fairseq.data.indexed_dataset import MMapIndexedDataset
    dataset = MMapIndexedDataset(indexed_data_path)
    total_len = len(dataset)
    # with open(predefined_batch_ids_path, 'r') as f_in:
    #     frozen_batches = json.load(f_in)
    predefined_batches = []
    for root, dirs, files in os.walk(predefined_batch_ids_dir):
        for json_file in files:
            json_path = os.path.join(root, json_file)
            with open(json_path, 'r') as f_in:
                predefined_batch = format_regularization(json.load(f_in))
                predefined_batches.append(list(filter(lambda x: x < total_len, predefined_batch)))

    # org_len = len(frozen_batches)
    # frozen_batches = list(filter(lambda x: x < total_len, frozen_batches))
    # print(f'removed cnt {org_len - len(frozen_batches)}')
    # frozen_batches = frozen_batches[:(total_shards * batch_size) * (len(frozen_batches) // (total_shards * batch_size))]

    data_loaders = []
    for epoch in range(epochs):
        frozen_batches = predefined_batches[epoch % len(predefined_batches)]
        frozen_batches = frozen_batches[:(total_shards * batch_size) * (len(frozen_batches) // (total_shards * batch_size))]
        grouped_batches = []
        for i in range(0, len(frozen_batches), batch_size):
            grouped_batches.append(frozen_batches[i:i + batch_size])

        with data_utils.numpy_seed(seed + epoch):
            np.random.shuffle(grouped_batches)

        # shard_batches = [batch[shard_id * batch_size: shard_id * batch_size + batch_size] for batch in grouped_batches]
        shard_batches = grouped_batches[shard_id::total_shards]
        data_loaders.append(data.DataLoader(dataset,
                                            collate_fn=collate_fn,
                                            batch_sampler=shard_batches,
                                            num_workers=num_workers))

    return data_loaders

def create_dataloader_with_position(dataset, batch_size, epochs, seed=0, num_workers=1, shuffle=False, distributed=True):
    data_loaders = []
    collator = CollatorWithMapping(None)
    for epoch in range(epochs):
        data_loaders.append(data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            collate_fn=collator.collate_fn_with_position,
                                            sampler=DistributedSampler(dataset, shuffle=shuffle) if distributed else SequentialSampler(dataset),
                                            num_workers=num_workers))

    return data_loaders

def create_summary_dataloader(
    tokenizer,
    corpus_dir, 
    batch_size, 
    max_seq_len, 
    epochs,
    seed=0, 
    total_steps=-1,
    vocab_mapping=None,
    num_workers=1,
    chunk_size=-1,
    distributed=True,
    shuffle=True,
    max_sum_len=100
):
    from reader.dataset_xsum import SummarizationCollator, SummarizationDataset
    world_size = 1 if not distributed else dist.get_world_size()
    pad_token_id = tokenizer.eos_token_id + 1
    dataset = SummarizationDataset(
        corpus_dir, 
        tokenizer=tokenizer,
        eos_id=tokenizer.eos_token_id,
        pad_id=pad_token_id
    )
    # print(len(dataset))
    # max_sum_len = kwargs.get('max_sum_len', 100)
    # chunk_size = kwargs.get('chunk_size', -1)
    collator = SummarizationCollator(max_seq_len, max_sum_len, chunk_size, pad_id=pad_token_id, tokenizer=tokenizer)
    return data.DataLoader(dataset,
                           batch_size=batch_size,
                           collate_fn=collator.fn,
                           sampler=DistributedSampler(dataset, shuffle=shuffle) if distributed else SequentialSampler(dataset),
                           num_workers=num_workers)


def create_dataloader(
    corpus_path, 
    batch_size, 
    max_seq_len, 
    epochs,
    task_type=None,
    seed=0, 
    total_steps=-1,
    vocab_mapping=None,
    num_workers=1,
    token_cnt=10,
    distributed=True,
    shuffle=True,
    max_sum_len=-1,
    chunk_retrieval=True,
    chunk_size=64,
    tokenizer=None
):
    print(task_type)
    if task_type is None:
        dataloader = create_slidewin_dataloader(corpus_path, 
                                                batch_size, 
                                                max_seq_len, 
                                                epochs,
                                                seed=seed, 
                                                total_steps=total_steps,
                                                vocab_mapping=vocab_mapping,
                                                num_workers=num_workers,
                                                shuffle=shuffle,
                                                distributed=distributed)
    elif task_type == 'passkey_retrieval':
        dataloader = create_passkey_dataloader(tokenizer,
                                               corpus_path, 
                                               batch_size, 
                                               max_seq_len, 
                                               epochs,
                                               seed=seed, 
                                               total_steps=total_steps,
                                               vocab_mapping=vocab_mapping,
                                               num_workers=num_workers,
                                               distributed=distributed,
                                               chunk_size=chunk_size,
                                               token_cnt=token_cnt,
                                               shuffle=shuffle,
                                               chunk_retrieval=chunk_retrieval)
    elif task_type == 'summarization':
        dataloader = create_summary_dataloader(tokenizer,
                                               corpus_path, 
                                               batch_size, 
                                               max_seq_len, 
                                               epochs,
                                               seed=seed, 
                                               total_steps=total_steps,
                                               vocab_mapping=vocab_mapping,
                                               num_workers=num_workers,
                                               distributed=distributed,
                                               shuffle=shuffle,
                                               max_sum_len=max_sum_len,
                                               chunk_size=chunk_size)
    return dataloader