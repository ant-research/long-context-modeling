from unittest import TestCase
from model.model_factory import create_model
import torch
import torch.nn.functional as F
import random
import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TestDRTGeneration(TestCase):
    def testUsability(self):
        from transformers import AutoConfig

        torch.set_printoptions(profile="full")
        device = torch.device("cuda:0")

        set_seed(0)

        model = create_model('DRT', 'config/DRT-unittest/config_res.json')
        device = torch.device('cuda:0')
        model.to(device)
        model.eval()
        input_ids = torch.tensor(torch.randint(0, 100, (1, 126)), device=device)
        # print(input_ids.shape)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
          outputs = model.generate(input_ids, max_length=252)

        outputs = outputs.sequences
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
          model_out = model(outputs)

        pred_ids = model_out.logits.argmax(dim=-1)
        assert torch.all(pred_ids[0, 125:-1] == outputs[0, 126:])