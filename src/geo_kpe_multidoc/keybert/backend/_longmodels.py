import logging
import os
import math
import copy
import torch

from typing import Callable
from transformers  import LongformerSelfAttention
from transformers import BigBirdModel, BigBirdConfig
from transformers import LongformerSelfAttention, LongformerConfig, LongformerModel, LongformerTokenizerFast
from transformers import XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig, AutoModel, AutoTokenizer
from transformers import RobertaForMaskedLM, RobertaTokenizerFast
from ._utils import select_backend
from geo_kpe_multidoc import GEO_KPE_MULTIDOC_MODELS_PATH


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

def load_longformer(embedding_model: str='') -> Callable:
    #longformer_path = f"{os.getcwd()}\\keybert\\backend\\long_models\\"
    longformer_path = GEO_KPE_MULTIDOC_MODELS_PATH 
    if not os.path.exists(longformer_path):
        os.makedirs(longformer_path)
    #model_path = f"{longformer_path}{embedding_model}"
    model_path = os.path.join(longformer_path, embedding_model) 
    if os.path.exists(model_path):
        if os.listdir(model_path):
            # logging.set_verbosity_error()
            callable_model = select_backend(embedding_model[11:])
            callable_model.embedding_model._modules['0']._modules['auto_model'] = XLMRobertaModel.from_pretrained(model_path, output_loading_info=False)
            callable_model.embedding_model.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path, output_loading_info=False)
            callable_model.embedding_model._modules['0']._modules['auto_model'].config = XLMRobertaConfig.from_pretrained(model_path, output_loading_info=False)
            return callable_model
    attention_window = 512
    max_pos = 4096
    return create_long_model(embedding_model, model_path, attention_window, max_pos)

def create_long_model(embedding_model, save_model_to, attention_window, max_pos):
    callable_model = select_backend(embedding_model[11:])
    model = callable_model.embedding_model._modules['0']._modules['auto_model']
    tokenizer = callable_model.embedding_model.tokenizer
    config = model.config
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    tokenizer._tokenizer.truncation['max_length'] = attention_window
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    max_pos += 2
    config.max_position_embeddings = max_pos

    assert max_pos > current_max_pos
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k:k + step] = model.embeddings.position_embeddings.weight[2:]
        k += step

    # From uncompile6
    # if not max_pos > current_max_pos:
    #     raise AssertionError
    # else:
    #     new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    #     k = 2
    #     step = current_max_pos - 2
    #     while True:
    #         if k < max_pos - 1:
    #             new_pos_embed[k:k + step] = model.embeddings.position_embeddings.weight[2:]
    #             k += step

    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos)
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

def create_bigbird(model, save_model_to, attention_window, max_pos):
    pass


def load_longmodel(embedding_model: str='') -> Callable:
    supported_models = {'longformer':create_longformer, 'bigbird':create_bigbird}
    # longmodel_path = f"{os.getcwd()}\\keybert\\backend\\long_models\\"
    longmodel_path = GEO_KPE_MULTIDOC_MODELS_PATH 
    # get model variant
    sliced_t = embedding_model[:embedding_model.index('-')]
    if sliced_t not in supported_models:
        raise ValueError(f"Model {sliced_t} is not in supported types")
    if not os.path.exists(longmodel_path):
        os.makedirs(longmodel_path)
    # get model name
    sliced_m = embedding_model[embedding_model.index('-') + 1:]

    # model_path = os.path.join(longmodel_path, embedding_model) 
    model_path = os.path.join(longmodel_path, embedding_model) 
    #f"{longmodel_path}{embedding_model}"
    # logging.set_verbosity_error()
    attention_window = 512
    max_pos = 4096
    if embedding_model == 'longformer-large-4096':
        model = select_backend('paraphrase-multilingual-mpnet-base-v2')
        model.embedding_model._modules['0']._modules['auto_model'] = LongformerModel.from_pretrained('allenai/longformer-large-4096')
        model.embedding_model.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-large-4096')
        model.embedding_model._modules['0']._modules['auto_model'].config = LongformerConfig.from_pretrained('allenai/longformer-large-4096')
        return model
    # if model does not exist, create new.
    if not os.path.exists(model_path):
        supported_models[sliced_t](sliced_m, model_path, attention_window, max_pos)
    
    callable_model = select_backend(sliced_m)
    if sliced_t == 'longformer':
        callable_model.embedding_model._modules['0']._modules['auto_model'] = XLMRobertaModel.from_pretrained(model_path, output_loading_info=False, output_hidden_states=True, output_attentions=True)
        callable_model.embedding_model.tokenizer = XLMRobertaTokenizer.from_pretrained(model_path, output_loading_info=False, output_hidden_states=True, output_attentions=True)
        callable_model.embedding_model.tokenizer.save_pretrained(model_path)
        callable_model.embedding_model._modules['0']._modules['auto_model'].config = XLMRobertaConfig.from_pretrained(model_path, output_loading_info=False, output_hidden_states=True, output_attentions=True)
    return callable_model