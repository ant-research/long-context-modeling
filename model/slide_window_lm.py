import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import torch.distributed as dist
from typing import Optional
from model.model_common import ModelOutput


class SlideWindowLM(nn.Module):
    def __init__(self, config, language_model):
        super().__init__()
        self.bos_id = config.bos_id
        self.language_model = language_model

    @property
    def pos_ids(self):
        if self._pos_ids is None:
            self._pos_ids = torch.arange(2).to(self.device)
        return self._pos_ids

    def remove_mlm_head(self):
        pass

    def fix_encoder_and_embeddings(self):
        self.language_model.embed_tokens.requires_grad=False

    def forward(self, input_ids=None, stride=-1, labels=None, **kwargs):
        L = input_ids.shape[1]

        outputs = self.language_model(input_ids=torch.where(input_ids < 0, 0, input_ids), labels=labels, stride=stride)
        ar_loss = outputs.loss
        
        result = ModelOutput(ar_loss=ar_loss, ae_loss=0, total_loss=ar_loss, logits=outputs.logits)
        return result