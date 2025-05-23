from model.llama import LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers import AutoConfig
from copy import deepcopy
import json


class JSONObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            # if isinstance(value, dict):
            #     setattr(self, key, JSONObject(value))
            # else:
            setattr(self, key, value)



def create_model(model_type, config_path, gradient_checkpointing=False, vocab_mapping=None, vocab_dir=None,
                 contriever_vocab_path=None, contriever_path=None):
    if model_type == "slide_window_lm":
        from model.slide_window_lm import SlideWindowLM
        config = AutoConfig.from_pretrained(config_path)
        if vocab_mapping is not None:
            config.vocab_size = max(vocab_mapping.values()) + 1
            print(f'model vocab size: {config.vocab_size}')
        if config.num_hidden_layers == -1:
            config.num_hidden_layers = sum(config.decoder_layers)
        config.is_causal=True
        config.norm_outputs=True
        llama_lm = LlamaForCausalLM(config)
        llama_lm.gradient_checkpointing = gradient_checkpointing
        model = SlideWindowLM(config, llama_lm)
        return model
    elif model_type == 'rpt_contriever':
        from model.chunk_dec_lm_contriever import ChunkMemAugLLM
        # decoder = LlamaForCausalLM(config)
        config = AutoConfig.from_pretrained(config_path)
        encoder_config = LlamaConfig(vocab_size=-1,
                             hidden_size=config.hidden_size,
                             intermediate_size=config.intermediate_size,
                             num_hidden_layers=config.encoder_layers,
                             max_position_embeddings=-1,
                             num_attention_heads=config.num_attention_heads,
                             num_key_value_heads=config.num_attention_heads,
                             enable_alibi=False,
                             is_causal=False,
                             output_hidden_states=True,
                             slide_window=-1,
                             _flash_attn_2_enabled=True,
                             norm_outputs=True
                             )
        encoder = LlamaModel(encoder_config)

        if vocab_mapping is not None:
            config.vocab_size = max(vocab_mapping.values()) + 1
            print(f'model vocab size: {config.vocab_size}')

        decoders = []
        for idx, layer_num in enumerate(config.decoder_layers):
            dec_config = deepcopy(config)
            dec_config.is_causal = True
            dec_config.num_hidden_layers = layer_num
            if idx < len(config.decoder_layers) - 1:
                dec_config.norm_outputs = False
            else:
                dec_config.norm_outputs = True
            if idx != 0:
                dec_config.vocab_size = -1
            decoders.append(LlamaModel(dec_config))
            decoders[-1].gradient_checkpointing = gradient_checkpointing

        print('rpt contriever')
        mem_aug_lm = ChunkMemAugLLM(config, decoders, encoder, vocab_dir, contriever_vocab_path, contriever_path)
        return mem_aug_lm
    elif model_type == 'blk_rec_tfm':
        from model.block_recurrent_transformer import BlockRecurrentTransformer, RecurrentTrainerWrapper
        config = AutoConfig.from_pretrained(config_path)
        model = BlockRecurrentTransformer(
            num_tokens = config.vocab_size,             # vocab size
            dim = config.hidden_size,                      # model dimensions
            depth = sum(config.decoder_layers),                      # depth
            dim_head = config.hidden_size // config.num_attention_heads,                  # attention head dimensions
            heads = config.num_attention_heads,                      # number of attention heads
            max_seq_len = 2 * config.slide_window,             # the total receptive field of the transformer, in the paper this was 2 * block size
            block_width = config.slide_window,              # block size - total receptive field is max_seq_len, 2 * block size in paper. the block furthest forwards becomes the new cached xl memories, which is a block size of 1 (please open an issue if i am wrong)
            num_state_vectors = config.slide_window,        # number of state vectors, i believe this was a single block size in the paper, but can be any amount
            recurrent_layers = (4,),        # where to place the recurrent layer(s) for states with fixed simple gating
            use_compressed_mem = False,     # whether to use compressed memories of a single block width, from https://arxiv.org/abs/1911.05507
            compressed_mem_factor = 4,      # compression factor of compressed memories
            use_flash_attn = True           # use flash attention, if on pytorch 2.0
        )
        trainer_wrapper = RecurrentTrainerWrapper(model, xl_memories_dropout=0.1, state_dropout=0.1)
        return trainer_wrapper
    elif model_type == 'llama_with_landmark':
        from model.llama_with_landmark import LlamaForCausalLM as LlamaLDK #, LlamaLandmarkConfig
        from model.landmark_lm import LandmarkLM
        config = AutoConfig.from_pretrained(config_path)
        if vocab_mapping is not None:
            config.vocab_size = max(vocab_mapping.values()) + 1
            print(f'model vocab size: {config.vocab_size}')
        # model = LlamaLDK(LlamaLandmarkConfig(config, chunk_id=config.chunk_id, mem_freq=config.mem_freq, train_context_length=config.train_context_length))
        ldk_model = LlamaLDK(config)
        ldk_model.gradient_checkpointing = gradient_checkpointing
        # ldk_model.enable_landmark_insertion()
        return LandmarkLM(config, ldk_model)
    elif model_type == 'DRT':
        from model.DRT import DRTForCausalLM
        config = AutoConfig.from_pretrained(config_path)
        model = DRTForCausalLM(config)
        model.set_gradient_checkpointing(gradient_checkpointing)
        return model