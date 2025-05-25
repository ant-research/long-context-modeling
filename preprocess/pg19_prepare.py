# Copyright 2023 Amirkeivan Mohtashami, Martin Jaggi
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

import os
import tiktoken

import numpy as np



gpt2_tokenizer = tiktoken.get_encoding("gpt2")
def _read_directory(path):
    texts = []
    for filename in os.listdir(path):
        if filename.endswith(".txt") and filename[:-4].isnumeric():
            print(filename)
            with open(os.path.join(path, filename), 'r') as f:
                texts += gpt2_tokenizer.encode_ordinary(f.read())
                texts.append(gpt2_tokenizer.eot_token)
    return np.array(texts, dtype=np.uint16)


PATH_TO_PG19 = ''
PATH_TO_PROCESSED_PG19 = ''
raw_test_data = _read_directory(f"{PATH_TO_PG19}/train")
raw_test_data.tofile(f'{PATH_TO_PROCESSED_PG19}/train/data')