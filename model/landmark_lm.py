import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import torch.distributed as dist
from typing import Optional
from model.model_common import ModelOutput
from model.llama_with_landmark import LlamaForCausalLM


class LandmarkLM(nn.Module):
    def __init__(self, config, language_model):
        super().__init__()
        self.config = config
        self.language_model = language_model

    @property
    def pos_ids(self):
        if self._pos_ids is None:
            self._pos_ids = torch.arange(2).to(self.device)
        return self._pos_ids

    def remove_mlm_head(self):
        pass

    def fix_encoder_and_embeddings(self):
        pass

    def forward(self, input_ids=None, position=None, mask_probability=None, **kwargs):
        L = input_ids.shape[1]
        # print(input_ids.shape)
        if self.training:
            outputs = self.language_model(input_ids=input_ids, labels=input_ids, use_flash=self.config.use_flash)
            # outputs = self.language_model(input_ids=input_ids, labels=input_ids, use_flash=self.config.use_flash, **kwargs)
        else:
            # self.language_model.eval()
            # use_flash = self.config.use_flash
            # offload_cache_to_cpu = use_flash
            # if hasattr(self.config, 'offload_cache_to_cpu'):
            #     offload_cache_to_cpu = self.config.offload_cache_to_cpu
            use_flash = False
            offload_cache_to_cpu = False
            # print(f'use_flash={use_flash}')
            outputs = self.language_model(
                input_ids=input_ids,
                labels=input_ids,
                use_flash=use_flash, offload_cache_to_cpu=offload_cache_to_cpu,
                # use_flash=True, offload_cache_to_cpu=False,
                max_chunk_length=self.config.max_chunk_length if self.config.use_cache else None,
                chunk_topk=self.config.chunk_topk if self.config.use_cache else None,
                use_cache=self.config.use_cache,
                output_hidden_states=False, output_attentions=False, return_dict=True
            )
            # print(outputs.loss)
            # self.language_model.train()

        ar_loss = outputs.loss

        result = ModelOutput(ar_loss=ar_loss, ae_loss=0, total_loss=ar_loss, logits=outputs.logits)
        # print(result.total_loss)
        return result