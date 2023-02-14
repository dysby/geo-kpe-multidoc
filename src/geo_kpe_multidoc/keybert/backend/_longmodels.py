import logging
import os
import math
import copy
import torch

from typing import Callable
from transformers  import LongformerSelfAttention
from keybert.backend._utils import select_backend

def create_longformer(model_n : str, save_model_to : str, attention_window : int, max_pos : int ) -> Callable :
    callable_model = select_backend(model_n)

    model = callable_model.embedding_model._modules['0']._modules['auto_model']
    tokenizer = callable_model.embedding_model.tokenizer
    config = model.config
    
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    tokenizer._tokenizer.truncation['max_length'] = attention_window

    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos

    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings
    
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:(k + step)] = model.embeddings.position_embeddings.weight[2:]
        k += step
    
    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers

    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn
    
    if not os.path.exists(save_model_to):
        os.makedirs(save_model_to)
    
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return callable_model