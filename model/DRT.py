# coding=utf-8
# Author: Xiang Hu
# Copyright 2025 Ant Group

""" PyTorch LLaMA model. Based on transformers==4.35.2"""
import math
import warnings
from typing import List, Optional, Tuple, Union
# from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from einops import rearrange, repeat
from transformers.generation import GenerationMixin

from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D, ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers import LlamaConfig
try:
    # from ltriton.gca_softmax1 import attention as gca_attn_softmax1
    # from ltriton.gca_softmax1 import gca_kv_cache as gca_kv_cache_softmax1

    from ltriton.gca import attention as gca_attn
    from ltriton.gca import gca_kv_cache
except:
    gca_attn = None
    gca_kv_cache = None


try:
    from flash_attn import flash_attn_func
except:
    flash_attn_func = None
# from model.rotary import apply_rotary_emb_func # fused_rotary
from model.llama import (
    LlamaDecoderLayer, 
    LlamaModel, 
    LlamaPreTrainedModel, 
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaMLP,
    GPT2MLP,
)
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from flash_attn.ops.rms_norm import RMSNorm # fused_rmsnorm
from flash_attn.losses.cross_entropy import CrossEntropyLoss # fused_crossentropy

from model.model_common import ModelOutput
from xformers.ops.swiglu_op import swiglu # fused_swiglu
#
# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class ChunkKVManager:
    def __init__(self, offloading=False, group_size=1):
        self.chunk_k = None
        self.chunk_v = None
        self.lmk_embs = None
        self.group_size = group_size
        self._current_chunk_k = [None for _ in range(group_size)]
        self._current_chunk_v = [None for _ in range(group_size)]
        self._current_weights = [None for _ in range(group_size)]
        self._current_hidden_states = None
        self._offloading = offloading

    def append(self, chunk_k, chunk_v, lmk_embs):
        # chunk_k: (N, D, S, dim)
        # lmk_embs: (N, D, dim)
        if self._offloading:
            chunk_k = chunk_k.cpu()
            chunk_v = chunk_v.cpu()
            # still keep lmk_embs in GPU
        if self.chunk_k is None:
            if not self._offloading:
                self.chunk_k = chunk_k # (N, D, S, dim)
            else:
                self.chunk_k = []
                for batch_i in range(chunk_k.shape[0]):
                    self.chunk_k.append(
                        [chunk_k[batch_i, chunk_i].to('cpu', non_blocking=True) for chunk_i in range(chunk_k.shape[1])]
                    )
        else:
            if not self._offloading:
                assert chunk_k.shape[-2] == self.chunk_k.shape[-2]
                self.chunk_k = torch.cat([self.chunk_k, chunk_k], dim=1)
            else:
                for batch_i in range(chunk_k.shape[0]):
                    for chunk_i in range(chunk_k.shape[1]):
                        self.chunk_k[batch_i].append(chunk_k[batch_i, chunk_i].to('cpu', non_blocking=True))
        
        if self.chunk_v is None:
            if not self._offloading:
                self.chunk_v = chunk_v
            else:
                self.chunk_v = []
                for batch_i in range(chunk_v.shape[0]):
                    self.chunk_v.append(
                        [chunk_v[batch_i, chunk_i].to('cpu', non_blocking=True) for chunk_i in range(chunk_v.shape[1])]
                    )
        else:
            if not self._offloading:
                assert chunk_v.shape[-2] == self.chunk_v.shape[-2]
                self.chunk_v = torch.cat([self.chunk_v, chunk_v], dim=1)
            else:
                for batch_i in range(chunk_v.shape[0]):
                    for chunk_i in range(chunk_v.shape[1]):
                        self.chunk_v[batch_i].append(chunk_v[batch_i, chunk_i].to('cpu', non_blocking=True))

        if self.lmk_embs is None:
            self.lmk_embs = lmk_embs
        else:
            self.lmk_embs = torch.cat([self.lmk_embs, lmk_embs], dim=1)

    @property
    def past_lmk_embeds(self):
        return self.lmk_embs

    def current_retrieved_chunk(self, group_idx=0):
        return self._current_chunk_k[group_idx], self._current_chunk_v[group_idx], self._current_weights[group_idx]

    @property
    def lower_hidden_states(self):
        return self._current_hidden_states

    def clear_lower_hidden_states(self):
        self._current_hidden_states = None

    def cache_lower_hidden_states(self, hidden_states):
        # hidden_states: (N, L, dim)
        if self._current_hidden_states is None:
            self._current_hidden_states = hidden_states
        else:
            self._current_hidden_states = torch.cat(
                [self._current_hidden_states, hidden_states], dim=1
            )

    def update_current_retrieved_chunk(self, indices, weights, group_idx):
        N = indices.shape[0]
        org_device = indices.device
        if indices.shape[1] > 0:
            if self._offloading:
                indices = indices.cpu()
            batch_indices = torch.arange(N, device=indices.device).unsqueeze(1)
            if len(indices.shape) == 3:
                indices = indices[:, -1, :]
            if not self._offloading:
                self._current_chunk_k[group_idx] = self.chunk_k[batch_indices, indices]  # (N, K, S, dim)
                self._current_chunk_v[group_idx] = self.chunk_v[batch_indices, indices]
            else:
                gather_chunk_ks = []
                gather_chunk_vs = []
                for batch_i in range(indices.shape[0]):
                    current_chunk_ks = []
                    current_chunk_vs = []
                    for chunk_idx in indices[batch_i].numpy():
                        current_chunk_ks.append(self.chunk_k[batch_i][chunk_idx].to(org_device))
                        current_chunk_vs.append(self.chunk_v[batch_i][chunk_idx].to(org_device))
                    current_chunk_ks = torch.stack(current_chunk_ks, dim=0)
                    current_chunk_vs = torch.stack(current_chunk_vs, dim=0)
                    gather_chunk_ks.append(current_chunk_ks)
                    gather_chunk_vs.append(current_chunk_vs)
                self._current_chunk_k[group_idx] = torch.stack(gather_chunk_ks)
                self._current_chunk_v[group_idx] = torch.stack(gather_chunk_vs)
            self._current_weights[group_idx] = weights[:, -1:, :]
        else:
            assert self.chunk_k.shape[1] == 1
            assert self.chunk_v.shape[1] == 1
            self._current_chunk_k[group_idx] = self.chunk_k
            self._current_chunk_v[group_idx] = self.chunk_v
            self._current_weights[group_idx] = weights

    def retrieve_chunks(self, indices):
        # indices: (N, chunk_num)
        N = indices.shape[0]
        if not self._offloading:
            batch_indices = torch.arange(N, device=indices.device).unsqueeze(1)
            k = self.chunk_k[batch_indices, indices]
            v = self.chunk_v[batch_indices, indices]
            return k, v
        else:
            org_device = indices.device
            indices = indices.cpu().numpy()
            gathered_chunk_k = []
            gathered_chunk_v = []
            for batch_i in range(indices.shape[0]):
                chunk_vs = []
                chunk_ks = []
                for chunk_i in indices[batch_i]:
                    chunk_ks.append(self.chunk_k[batch_i][chunk_i].to(org_device))
                    chunk_vs.append(self.chunk_v[batch_i][chunk_i].to(org_device))
                chunk_ks = torch.stack(chunk_ks, dim=0)
                chunk_vs = torch.stack(chunk_vs, dim=0)  # (D, S, dim)
                gathered_chunk_k.append(chunk_ks)
                gathered_chunk_v.append(chunk_vs)
            chunk_k = torch.stack(gathered_chunk_k, dim=0)
            chunk_v = torch.stack(gathered_chunk_v, dim=0)
            return chunk_k, chunk_v

    def reorder(self, beam_ids):
        # warning: not tested!
        self.chunk_k = self.chunk_k.index_select(0, beam_idx.to(self.chunk_k.device))
        self.chunk_v = self.chunk_v.index_select(0, beam_idx.to(self.chunk_v.device))
        self.lmk_embs = self.lmk_embs.index_select(0, beam_idx.to(self.lmk_embs.device))

        for group_i in range(self.group_size):
            self._current_chunk_k[group_i] = self._current_chunk_k[group_i].index_select(
                0, beam_idx.to(self._current_chunk_k.device)
            )
            self._current_chunk_v[group_i] = self._current_chunk_v[group_i].index_select(
                0, beam_idx.to(self._current_chunk_v.device)
            )


class GroupCrossAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.enc_chunk_size = config.enc_chunk_size
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)

    @staticmethod
    def preprocess_chunk_kv_cache(
        chunk_k: torch.Tensor,  # (N D S dim)
        chunk_v: torch.Tensor,
        weights: torch.Tensor,
        num_heads
    ):
        k = rearrange(chunk_k, 'N L S (h d) -> (N L) S h d', h=num_heads)
        v = rearrange(chunk_v, 'N L S (h d) -> (N L) S h d', h=num_heads)
        return k,  v, weights

    def forward(
        self,
        hidden_states: torch.Tensor,  # (N, D, S, dim)  D: chunk_size, S, seq_len, during inference, D is always 1
        chunk_k: torch.Tensor,  # (N D K, S, h, d)
        chunk_v: torch.Tensor,  # (N D K, S, h, d)
        weights: torch.Tensor,  # (N, D, K)
    ):
        # ret_vals = torch.zeros_like(hidden_states)
        hidden_states = self.pre_norm(hidden_states)
        chunk_size = hidden_states.shape[1]
        q = self.q_proj(hidden_states)  # (N, D, S, dim)
        q = rearrange(q, 'N D S (h d) -> N D h S d', h=self.num_heads)

        K = weights.shape[2]

        if weights.shape[1] * weights.shape[2] == 0:
            ret_vals = torch.zeros(
                (hidden_states.shape[0], hidden_states.shape[1] * hidden_states.shape[2], hidden_states.shape[-1]), 
                device=hidden_states.device
            )
            return ret_vals

        # ignore the first chunk
        if chunk_size > 1:
            exp_q = repeat(q[:, 1:, :, :, :], 'N D h S d->N D K h S d', K=K)
        else:
            exp_q = repeat(q, 'N D h S d->N D K h S d', K=K)
        D = exp_q.shape[1]
        flash_q = rearrange(exp_q, 'N D K h S d->(N D K) S h d')
        assert flash_q.shape[0] == chunk_k.shape[0], f'q shape: {flash_q.shape}, k shape: {chunk_k.shape}'

        hist_chunk_vals = flash_attn_func(flash_q, chunk_k, chunk_v, causal=False)
        hist_chunk_vals = rearrange(hist_chunk_vals, '(N D K) S h d->N D K h S d', D=D, K=K)
        weighted_vals = torch.einsum('N D K h S d, N D K->N D h S d', hist_chunk_vals, weights)

        o = self.o_proj(rearrange(weighted_vals, 'N D h S d -> N (D S) (h d)'))
        if chunk_size > 1:
            ret_vals = torch.zeros(
                (hidden_states.shape[0], hidden_states.shape[1] * hidden_states.shape[2], hidden_states.shape[-1]), 
                device=hidden_states.device
            )
            assert ret_vals.shape[1] - o.shape[1] == self.enc_chunk_size + 1
            ret_vals[:, -o.shape[1]:, :] = o
        else:
            ret_vals = o
        return ret_vals

class TritonGroupCrossAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self._fill_ids = {}
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.enc_chunk_size = config.enc_chunk_size
        self.topK = config.chunk_topk
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)

        self.enable_softmax_one = getattr(config, 'enable_softmax_one', False)
        self.sm_n = 1.0 if self.enable_softmax_one else 0.0

    @staticmethod
    def preprocess_chunk_kv_cache(
        chunk_k: torch.Tensor,  # (N D S dim)
        chunk_v: torch.Tensor,
        weights: torch.Tensor,  # (N D K)
        num_heads
    ):
        K = weights.shape[-1]
        k = rearrange(chunk_k, 'N (D K) S (h d) -> (N D) h K S d', K=K, h=num_heads)
        v = rearrange(chunk_v, 'N (D K) S (h d) -> (N D) h K S d', K=K, h=num_heads)
        weights = rearrange(weights, 'N D K->(N D) K')
        # return k[:, :, :, :-1, :].contiguous(), v[:, :, :, :-1, :].contiguous(), weights
        return k.contiguous(), v.contiguous(), weights

    def _get_fill_ids(self, chunk_num, chunk_size, device):
        key = f'{chunk_num}_{chunk_size}_{device}'
        if key not in self._fill_ids:
            total_len = (chunk_num - 1) * (chunk_size + 1)
            fill_ids = torch.arange(total_len, device=device) + chunk_size + 1
            fill_ids = rearrange(fill_ids, '(L S) -> L S', L=chunk_num-1)
            fill_ids = fill_ids[:, :-1]
            self._fill_ids[key] = fill_ids.flatten().contiguous()
        return self._fill_ids[key]


    def forward(
        self,
        hidden_states: torch.Tensor,  # (N, D, S, dim)  D: chunk_size, S, seq_len, during inference, D is always 1
        chunk_k: torch.Tensor,  # (N D, h, K, S, d)
        chunk_v: torch.Tensor,  # (N D, h, K, S, d)
        weights: torch.Tensor,  # (N D, K):
    ):
        hidden_states = self.pre_norm(hidden_states)
        chunk_size = hidden_states.shape[1]
        q = self.q_proj(hidden_states)  # (N, D, S, dim)
        q = rearrange(q, 'N D S (h d) -> N D h S d', h=self.num_heads)
        
        K = weights.shape[-1]

        if weights.shape[0] * weights.shape[1] == 0:
            ret_vals = torch.zeros(
                (hidden_states.shape[0], hidden_states.shape[1] * hidden_states.shape[2], hidden_states.shape[-1]), 
                device=hidden_states.device
            )
            return ret_vals

        if chunk_size > 1:
            # q = q[:, 1:, :, :-1, :]  # ignore the last landmark token
            # assert q.shape[-2] % 16 == 0, f'q shape: {q.shape}'  # adapt to the triton kernel
            q = q[:, 1:, :, :, :]

        N = q.shape[0]
        q = rearrange(q, 'N D h S d->(N D) h S d')

        assert q.shape[0] == weights.shape[0]
        assert q.shape[0] == chunk_k.shape[0]
        assert chunk_k.shape[-3] == weights.shape[-1], f'chunk_k.shape: {chunk_k.shape}, weights shape: {weights.shape}'

        q = q.contiguous()
        if chunk_size > 1 and self.training:
            weighted_vals = gca_attn(q, chunk_k, chunk_v, weights, 1 / math.sqrt(self.head_dim), self.sm_n)
        else:
            weighted_vals = gca_kv_cache(q, chunk_k, chunk_v, weights, 1 / math.sqrt(self.head_dim), self.sm_n)
        # print(f'gca output shape: {weighted_vals.shape}')
        weighted_vals = rearrange(weighted_vals, '(N D) h S d -> N (D S) (h d)', N=N)
        o = self.o_proj(weighted_vals)

        if chunk_size > 1:
            ret_vals = torch.zeros(
                (hidden_states.shape[0], hidden_states.shape[1] * hidden_states.shape[2], hidden_states.shape[-1]), 
                dtype=o.dtype,
                device=hidden_states.device
            )
            assert ret_vals.shape[1] - o.shape[1] == self.enc_chunk_size + 1
            ret_vals[:, -o.shape[1]:, :] = o
        else:
            # for inference
            ret_vals = o
        return ret_vals


class UpperDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.enc_chunk_size = config.enc_chunk_size
        self.self_attn = (
            LlamaAttention(config=config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else LlamaFlashAttention2(config=config)
        )

        self.gca = (
            GroupCrossAttention(config=config)
            if not getattr(config, "_triton_gca", False)
            else TritonGroupCrossAttention(config=config)
        )

        if hasattr(config, 'gpt2_mlp') and config.gpt2_mlp:
            self.mlp = GPT2MLP(config)
        else:
            self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        chunk_k: torch.Tensor = None,
        chunk_v: torch.Tensor = None,
        chunk_weights: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        if hidden_states.shape[1] % (self.enc_chunk_size + 1) == 0:
            # batch mode:
            hidden_states = rearrange(hidden_states, 'N (D S) H->N D S H', S=self.enc_chunk_size + 1)
            hidden_states = self.gca(hidden_states, chunk_k, chunk_v, chunk_weights)
        else:
            hidden_states = hidden_states.unsqueeze(1)  # (N, 1, L, dim)
            hidden_states = self.gca(hidden_states, chunk_k, chunk_v, chunk_weights)

        hidden_states = residual + hidden_states
        # residual = hidden_states  # DO NOT RESET RESIDUAL, ppl will significantly increase.
        hidden_states = self.post_attention_layernorm(hidden_states)


        # print(f'hidden states shape: {hidden_states.shape}')
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class DRT(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_id
        self.vocab_size = config.vocab_size
        self.enc_chunk_size = config.enc_chunk_size
        self.slide_window = config.slide_window
        self.input_dim = config.hidden_size
        self.chunk_topk = config.chunk_topk
        self.num_heads = config.num_attention_heads
        self.causal_masks = {}
        config.is_causal = True

        if config.vocab_size > 0:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.decoder_layers = config.decoder_layers # only support 1-hop retrieval
        # assert len(config.decoder_layers) == 2 
        
        # self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.lower_layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.decoder_layers[0])])

        self.num_groups = len(config.decoder_layers) - 1
        self.upper_layers = nn.ModuleList(
            [nn.ModuleList([UpperDecoderLayer(config) for _ in range(num_layer)]) for num_layer in config.decoder_layers[1:]]
        )
        self.lmk_norms = nn.ModuleList(
            [RMSNorm(config.hidden_size) for _ in range(self.num_groups)]
        )
        assert sum(map(lambda x: len(x), self.upper_layers)) == sum(self.decoder_layers[1:])

        encoder_config = LlamaConfig(
                            vocab_size=-1,
                            hidden_size=config.hidden_size,
                            intermediate_size=config.intermediate_size,
                            num_hidden_layers=config.encoder_layers,
                            max_position_embeddings=config.enc_chunk_size + 1,
                            num_attention_heads=config.num_attention_heads,
                            num_key_value_heads=config.num_attention_heads,
                            enable_alibi=False,
                            is_causal=False,
                            output_hidden_states=True,
                            slide_window=-1,
                            _flash_attn_2_enabled=True,
                            norm_outputs=True
                        )

        self.encoder = LlamaModel(encoder_config)
        # self.position_embeddings = nn.Embedding(self.enc_chunk_size + 1, config.hidden_size)

        self.preprocess_chunk_kv_cache = (
            GroupCrossAttention.preprocess_chunk_kv_cache 
            if not getattr(config, "_triton_gca", False)
            else TritonGroupCrossAttention.preprocess_chunk_kv_cache
        )

        # self.lmk_q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        # self.lmk_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.chunk_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
        self.chunk_v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)

        self.enc_prenorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def post_init(self):
        super().post_init()
        # self.position_embeddings.weight.data.normal_(mean=0, std=0.02)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _enc_dec_score_mask(self, dec_chunk_num, enc_chunk_num, max_chunk_win=-1):
        key = f'{dec_chunk_num}_{enc_chunk_num}_{max_chunk_win}'
        # print(key)
        if key not in self.causal_masks:
            dec_enc_rate = 1
            # slide_win_chunk_num = math.ceil(self.slide_window / self.enc_chunk_size)
            dec_chunk_ids = torch.arange(enc_chunk_num, device=self.device)
            dec_chunk_ids = dec_chunk_ids[-dec_chunk_num:]
            enc_chunk_ids = torch.arange(enc_chunk_num, device=self.device)
            # mask = torch.triu(torch.ones((L, L), device=self.device, dtype=torch.bool), diagonal=1)
            visible = dec_chunk_ids.unsqueeze(1) > enc_chunk_ids.unsqueeze(0) # (dec_chunk_num, enc_chunk_num)
            if max_chunk_win != -1:
                visible2 = dec_chunk_ids.unsqueeze(1) <= enc_chunk_ids.unsqueeze(0) + max_chunk_win
                visible = visible & visible2

            self.causal_masks[key] = ~visible
            # print(~visible)
        return self.causal_masks[key]
    
    def _perform_batch_retrieval(self, lmk_embs, chunk_kvs, group_idx, use_cache=False):
        past_lmk_embs = chunk_kvs.past_lmk_embeds
        
        scores = torch.einsum('N C D, N E D->N C E', lmk_embs, past_lmk_embs) / math.sqrt(self.input_dim) # (N, dec_chunk_num, enc_chunk_num)
        scores = scores.masked_fill_(self._enc_dec_score_mask(scores.shape[1], scores.shape[2]), float('-inf'))
        # print(f'fwd scores after mask: {scores}')

        if self.training:
            noise = -torch.empty_like(
                scores,
                memory_format=torch.legacy_contiguous_format,
                requires_grad=False).exponential_().log()
        else:
            noise = torch.zeros_like(scores)
        chunk_top_k = min(scores.shape[-1], self.chunk_topk)

        _, indices = torch.topk((scores + noise), dim=2, k=chunk_top_k)  # (N  chunk_num, topk)
        chunk_weights = F.softmax(scores.gather(dim=2, index=indices), dim=-1)
        chunk_weights = torch.nan_to_num(chunk_weights, nan=0.0)  # (N, chunk_num, topk)
        # print(topk_weights)
        indices_ = rearrange(indices[:, :-1, :], 'N D K->N (D K)')  # drop the last chunk
        # print(f'batch retrieval weights: {chunk_weights}, indices: {indices}, group_idx: {group_idx}')
        retrieved_chunk_k, retrieved_chunk_v = chunk_kvs.retrieve_chunks(indices_)
        assert retrieved_chunk_k is not None
        # drop the last chunk
        org_weights = chunk_weights
        chunk_k, chunk_v, chunk_weights = self.preprocess_chunk_kv_cache(
            retrieved_chunk_k, retrieved_chunk_v, chunk_weights[:, :-1, :], self.num_heads
        )
        if use_cache and not self.training:
            chunk_kvs.update_current_retrieved_chunk(indices, org_weights, group_idx)
        # print(f'group idx: {group_idx} batch weights: {org_weights}')
        return chunk_k, chunk_v, chunk_weights, indices

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        offloading = kwargs.get('offloading', False)

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        chunk_kvs = None
        if past_key_values is not None:
            past_key_values, chunk_kvs = past_key_values
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(torch.where(input_ids < 0, self.padding_idx, input_ids))
        else:
            raise Exception('Customized input embeds are not supported')

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.lower_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                # layer_outputs = self._gradient_checkpointing_func(
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

              
        N = input_ids.shape[0]
        L = input_ids.shape[1]
        if chunk_kvs is None:
            chunk_kvs = ChunkKVManager(offloading, self.num_groups)
        
        if L % (self.enc_chunk_size + 1) == 0:
            # inference or training stage
            # chunk_pos_ids = torch.arange(self.enc_chunk_size + 1, device=input_ids.device)  # (enc_chunk_sz + 1)
            # pos_embeds = self.position_embeddings(chunk_pos_ids).unsqueeze(0)
            chunk_hidden_states = rearrange(hidden_states, 'N (C S) D->(N C) S D', S=self.enc_chunk_size + 1)
            enc_outputs = self.encoder(inputs_embeds=self.enc_prenorm(chunk_hidden_states)) # + pos_embeds)
            # enc_outputs = self.encoder(inputs_embeds=chunk_hidden_states)
            chunk_hidden_states = rearrange(enc_outputs.last_hidden_state, '(N C) S D->N C S D', N=N)
            lmk_embs = chunk_hidden_states[:, :, -1, :]
            # lmk_embs = self.lmk_ln(chunk_hidden_states.mean(dim=-2))
            # print(f'batch lmk: {lmk_embs[0, :, :5]}')
            # assert chunk_hidden_states.shape[-2] % 64 == 0, f'chunk_hidden_states.shape: {chunk_hidden_states.shape}'
            chunk_k = self.chunk_k_proj(chunk_hidden_states) # (N C S D)
            assert chunk_k.shape[-2] % 64 == 0 or chunk_k.shape[-2] % 64 == 1
            if chunk_k.shape[-2] % 64 == 0:
                chunk_k[:, :, -1, :] = 0  # set the landmark representation to zero, equivalant to softmax + 1
            elif chunk_k.shape[-2] % 64 == 1:
                chunk_k = chunk_k[:, :, :-1, :]  # remove landmark token
            # chunk_k = chunk_k_hidden[:, :-1, :, :]  # remove the last chunk
            chunk_v = self.chunk_v_proj(chunk_hidden_states)
            if chunk_v.shape[-2] % 64 == 0:
                chunk_v[:, :, -1, :] = 0
            elif chunk_v.shape[-2] % 64 == 1:
                chunk_v = chunk_v[:, :, :-1, :]
            # chunk_v = chunk_v_hidden[:, :-1, :, :]
            # print(f'batch chunk kv: {chunk_k[0, :, 0, :5]}')
            chunk_kvs.append(chunk_k, chunk_v, lmk_embs)

            assert L % (self.enc_chunk_size + 1) == 0
        else:
            assert not self.training
            chunk_kvs.cache_lower_hidden_states(hidden_states)
            if input_ids.shape[1] == 2: # (input_id, lmk_id)
                # update new chunk embeddings based on past kv cache
                assert chunk_kvs.lower_hidden_states.shape[1] == self.enc_chunk_size + 1
                # chunk_pos_ids = torch.arange(self.enc_chunk_size + 1, device=input_ids.device)  # (enc_chunk_sz + 1)
                # pos_embeds = self.position_embeddings(chunk_pos_ids).unsqueeze(0)
                chunk_hidden_states = chunk_kvs.lower_hidden_states  # (N, S + 1, dim)
                chunk_kvs.clear_lower_hidden_states()
                enc_outputs = self.encoder(inputs_embeds=self.enc_prenorm(chunk_hidden_states)) # + pos_embeds)
                chunk_hidden_states = enc_outputs.last_hidden_state  #(N, S + 1, D)
                lmk_embs = chunk_hidden_states[:, -1, :].unsqueeze(1)  # (N, 1, dim)
                # lmk_embs = self.lmk_ln(chunk_hidden_states.mean(dim=-2)).unsqueeze(1)
                # print(f'gen lmk: {lmk_embs[0, :, :5]}')
                chunk_k = self.chunk_k_proj(chunk_hidden_states) # (N, S + 1, D)
                chunk_v = self.chunk_v_proj(chunk_hidden_states) # (N, S + 1, D)
                
                if chunk_k.shape[-2] % 64 == 0:
                    chunk_k[:, -1, :] = 0
                elif chunk_k.shape[-2] % 64 == 1:
                    chunk_k = chunk_k[:, :-1, :]
                
                if chunk_v.shape[-2] % 64 == 0:
                    chunk_v[:, -1, :] = 0
                elif chunk_v.shape[-2] % 64 == 1:
                    chunk_v = chunk_v[:, :-1, :]
                chunk_kvs.clear_lower_hidden_states()  # clear cached hidden states
                chunk_kvs.append(chunk_k[:, None, :, :], chunk_v[:, None, :, :], lmk_embs)

        lower_layer_num_offset = len(self.lower_layers)
        # print(chunk_mark_ids)
        for group_idx in range(self.num_groups):  # 0 ~ group_size - 1
            # perform retrieval/ TODO: L + prev_len
            if L % (self.enc_chunk_size + 1) == 0:
                q_embs = rearrange(hidden_states, 'N (C S) d-> N C S d', S=self.enc_chunk_size + 1)[:, :, -1, :]
                q_embs = self.lmk_norms[group_idx](q_embs)
                chunk_k, chunk_v, chunk_weights, indices = self._perform_batch_retrieval(q_embs, chunk_kvs, group_idx, use_cache=use_cache)
            else:
                assert not self.training
                # for inference stage
                # prepare retrieved chunks
                chunk_k, chunk_v, chunk_weights = chunk_kvs.current_retrieved_chunk(group_idx)
                # chunk_k, (N, 1, K, dim), chunk_weights: (N, 1, K)
                chunk_k, chunk_v, chunk_weights = self.preprocess_chunk_kv_cache(
                    chunk_k, chunk_v, chunk_weights, self.num_heads
                )
                # print(f'input ids shape: {input_ids.shape}')
                if input_ids.shape[1] == 2: # (input_id, lmk_id)
                    past_lmk_embs = chunk_kvs.past_lmk_embeds  # (N, ?, dim)
                    lmk_embs = hidden_states[:, -1:, :]
                    scores = torch.einsum('N C D, N E D->N C E', lmk_embs, past_lmk_embs) / math.sqrt(self.input_dim) # (N, dec_chunk_num, enc_chunk_num)
                    # print(f'update scores: {scores}')
                    scores = scores.masked_fill_(self._enc_dec_score_mask(scores.shape[1], scores.shape[2]), float('-inf'))
                    # print(f'update scores after mask: {scores}')
                    chunk_top_k = min(scores.shape[-1], self.chunk_topk)
                    _, indices = torch.topk(scores, dim=2, k=chunk_top_k)  # (N, 1, K)
                    weights = F.softmax(scores.gather(dim=2, index=indices), dim=-1)
                    indices = indices.squeeze(1)
                    # print(f'single step retrieval weights: {weights}, indices: {indices}, group_idx: {group_idx}')
                    assert weights.shape[1] == 1
                    # print(f'update indices: {indices}, {weights}')
                    chunk_kvs.update_current_retrieved_chunk(indices, weights[:, :1, :], group_idx)
                    # tmp_k, _, _ = chunk_kvs.current_retrieved_chunk(group_idx)
            

            for idx, decoder_layer in enumerate(self.upper_layers[group_idx]):
                idx += lower_layer_num_offset
                # print(f'upper layer idx: {idx}')
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                past_key_value = past_key_values[idx] if past_key_values is not None else None

                if self.gradient_checkpointing and self.training:
                    # layer_outputs = self._gradient_checkpointing_func(
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        decoder_layer.__call__,
                        hidden_states,
                        attention_mask,
                        position_ids,
                        past_key_value,
                        output_attentions,
                        use_cache,
                        chunk_k,
                        chunk_v,
                        chunk_weights
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        chunk_k=chunk_k,
                        chunk_v=chunk_v,
                        chunk_weights=chunk_weights
                    )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            lower_layer_num_offset += len(self.upper_layers[group_idx])

        hidden_states = self.norm(hidden_states)
 
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = (next_decoder_cache, chunk_kvs) if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DRTForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = DRT(config)
        self.vocab_size = config.vocab_size
        self.enc_chunk_size = config.enc_chunk_size
        self.chunk_id = config.chunk_id
        self.pad_id = config.pad_id
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def set_gradient_checkpointing(self, val):
        self.model.gradient_checkpointing = val
        self.model.encoder.gradient_checkpointing = val

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _insert_dec_chunk_special_id(self, input_ids, special_id):
        N = input_ids.shape[0]
        input_ids_ = input_ids.view(N, -1, self.enc_chunk_size)  # (N, L / cz, cz)
        chunk_num = input_ids_.shape[1]
        chunk_id_padding = torch.ones(N, chunk_num, 1, device=input_ids.device, dtype=torch.long).fill_(special_id)
        chunked_input_ids = torch.cat([input_ids_, chunk_id_padding], dim=2)  # (N, L / cz, cz+1)
        chunked_input_ids = chunked_input_ids.view(N, -1)  # (N, L // cz * (cz + 1))
        return chunked_input_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        stride=-1,
        first_token_id=-1,
        output_logits: Optional[bool] = False,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids.shape[1] % self.enc_chunk_size == 0:
            dec_input_ids = self._insert_dec_chunk_special_id(input_ids, self.chunk_id)
        else:
            dec_input_ids = input_ids
            assert dec_input_ids.shape[1] == 1 or dec_input_ids.shape[1] == 2, f'enc chunk size: {self.enc_chunk_size}'

        segment_len = kwargs.get('segment_len', dec_input_ids.shape[1])
        segments = 1
        if segment_len != dec_input_ids.shape[1]:
            assert not self.training
            assert segment_len % (self.enc_chunk_size + 1) == 0
            use_cache = True
            assert self.config.slide_window < segment_len
            segments = math.ceil(dec_input_ids.shape[1] / segment_len)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        for segment_i in range(segments):
            seg_ids = dec_input_ids[:, segment_i * segment_len : (segment_i + 1) * segment_len]
            outputs = self.model(
                input_ids=seg_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

            if not self.training:
                past_kv, chunk_kvs = outputs.past_key_values
                trunc_kvs = []
                for layer_kv in past_kv:
                    trunc_key = layer_kv[0][:, :, -segment_len:, :]
                    trunc_value = layer_kv[1][:, :, -segment_len:, :]
                    trunc_kvs.append((trunc_key, trunc_value))
                past_key_values = (trunc_kvs, chunk_kvs)

        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        # logits = logits.float()

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            target_input_ids = torch.ones_like(labels).fill_(-100)
            # shift right
            target_input_ids[:, :-1] = labels[:, 1:]
            # (N, L)
            target_input_ids = self._insert_dec_chunk_special_id(target_input_ids, -100)
            if stride == -1:
                loss = loss_fct(logits.view(-1, logits.shape[-1]), target_input_ids.view(-1))
            elif stride > 0:
                loss = loss_fct(logits[:, -stride:, :].view(-1, logits.shape[-1]), target_input_ids[:, -stride:].view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        if not output_logits and self.training:
            return ModelOutput(
                ar_loss=loss, 
                ae_loss=0.0, 
                total_loss=loss
            )
        else:
            if input_ids.shape[1] % self.enc_chunk_size == 0:
                logits = rearrange(logits, 'N (C S) V->N C S V', S=self.enc_chunk_size + 1)
                logits = logits[:, :, :-1, :]
                logits = rearrange(logits, 'N C S V->N (C S) V')
                if first_token_id != -1:
                    logits[:, -1, :] = float('-inf')
                    logits[:, -1, first_token_id] = 0
            else:
                assert input_ids.shape[1] == 1 or input_ids.shape[1] == 2, f'input_ids.shape: {input_ids.shape}'
                if input_ids.shape[1] == 2:
                    # only return the first logit
                    logits = logits[:, :1, :]
                    # print(f'logits shape: {logits.shape}')
                
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions
            )


    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.Tensor, 
        past_key_values=None,
        offload_to_cpu=False,
        first_token_id=-1,
        **kwargs
    ):
        past_length = 0
        chunk_kvs = None
        if past_key_values is not None:
            past_key_cache, chunk_kvs = past_key_values
            past_length = past_key_cache[0][0].shape[2] - past_key_cache[0][0].shape[2] // (self.enc_chunk_size + 1)
            # print(f'past_key_cache shape: {past_key_cache[0][0].shape} past_length: {past_length} input_ids: {input_ids.shape}')

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                assert input_ids.shape[1] == 1, f'input_ids shape: {input_ids.shape}, only supports chunk_size * n or 1' # only support passing in one new token
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        # one token or multi tokens:
        if input_ids.shape[1] == 1:
            if (past_length + 1) % self.enc_chunk_size == 0:
                # pad a chunk id
                input_ids = F.pad(input_ids, (0, 1), value=self.chunk_id)
        elif input_ids.shape[1] > 1 and past_key_values is None:
            # multi tokens, padding to chunks
            assert past_length == 0 and chunk_kvs is None
            pad_length = math.ceil(input_ids.shape[1] / self.enc_chunk_size) * self.enc_chunk_size
            past_length = pad_length
            input_ids = F.pad(input_ids, (pad_length - input_ids.shape[1], 0), value=self.pad_id)
            assert input_ids.shape[1] % self.enc_chunk_size == 0
            chunk_kvs = ChunkKVManager(offload_to_cpu)
        else:
            # requires padding previous tokens to a complete chunk and save encoded kv cache to chunk_kvs
            raise Exception(f"Not supported for chat yet")

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "chunk_kvs": chunk_kvs,
                "use_cache": kwargs.get("use_cache"),
                "first_token_id": first_token_id,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # raise Exception("Not supported for beam searching yet")
        # # TODO: add index_select for chunk kv manager.

        past_key_values, chunk_kvs = past_key_values
        chunk_kvs.reorder(beam_idx)
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return (reordered_past, reordered_chunk_kvs)
