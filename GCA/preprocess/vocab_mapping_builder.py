from utils.data_loader_utils import create_slidewin_dataloader
import pickle
from tqdm import tqdm
import argparse
import sys

RESERV_IDS = 3  # bos, chunk_id, mask_id

if __name__ == '__main__':
    cmd = argparse.ArgumentParser('Preprocess corpus components')
    cmd.add_argument('--train_path', required=True)
    cmd.add_argument('--valid_path', required=True)
    cmd.add_argument('--save_path', required=True)
    args = cmd.parse_args(sys.argv[1:])

    dataloader = create_slidewin_dataloader(args.train_path, 1, 1024, 1, shuffle=False, distributed=False)

    train_set = set()
    for step, item in enumerate(tqdm(dataloader)):
        ids = item['input_ids']
        for token_id in ids.numpy()[0, :]:
            train_set.add(token_id)

        if step % 10000 == 0:
            print(f'current tokens: {len(train_set)}')

    print(f'total tokens: {len(train_set)}')

    dataloader = create_slidewin_dataloader(args.valid_path, 1, 1024, 1, shuffle=False, distributed=False)

    valid_set = set()
    for step, item in enumerate(tqdm(dataloader)):
        ids = item['input_ids']
        for token_id in ids.numpy()[0, :]:
            valid_set.add(token_id)

        if step % 10000 == 0:
            print(f'current tokens: {len(valid_set)}')

    all_tokens = train_set | valid_set
    token_mapping = {}
    for token_id in sorted(list(all_tokens)):
        token_mapping[token_id] = len(token_mapping) + RESERV_IDS

    with open(args.save_path, mode='wb') as f_out:
        pickle.dump(token_mapping, f_out)