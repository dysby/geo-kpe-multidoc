import copy
import logging
import math
import os
from typing import Callable

import torch
from keybert.backend._sentencetransformers import SentenceTransformerBackend
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoTokenizer,
    BigBirdConfig,
    BigBirdModel,
    LongformerConfig,
    LongformerModel,
    LongformerSelfAttention,
    LongformerTokenizerFast,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizer,
    XLMRobertaTokenizerFast,
)

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_MODELS_PATH

from .roberta2longformer.roberta2longformer import convert_roberta_to_longformer

# from ...keybert.backend._utils import select_backend
from .select_backend import select_backend


class XLMRobertaLongSelfAttention(LongformerSelfAttention):
    """from https://github.com/allenai/longformer/issues/215"""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = any(is_index_global_attn.flatten())
        return super().forward(
            hidden_states,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )


class XLMRobertaLongModel(XLMRobertaModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = XLMRobertaLongSelfAttention(config, layer_id=i)


def to_longformer(base_model, max_pos=4096, attention_window=512):
    """Transform a `base_model` (RoBERTa) into a Longformer with sparce attention."""

    logger.info("Transform to longformer")
    model = base_model._modules["0"]._modules["auto_model"]
    config = model.config
    tokenizer = base_model.tokenizer

    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    tokenizer.padding = "max_lenght"
    # tokenizer._tokenizer.truncation["max_length"] = attention_window

    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos

    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    # copy position embeddings over and over to initialize the new position embeddings

    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k : (k + step)] = model.embeddings.position_embeddings.weight[2:]
        k += step

    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids.data = torch.tensor(
        [i for i in range(max_pos)]
    ).reshape(1, max_pos)

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

    base_model.max_seq_length = tokenizer.model_max_length
    logger.info("Longformer created")
    return model, tokenizer


def create_longformer(
    model_n: str, save_model_to: str, attention_window: int, max_pos: int
) -> Callable:
    callable_model = select_backend(model_n)

    model = callable_model.embedding_model._modules["0"]._modules["auto_model"]
    tokenizer = callable_model.embedding_model.tokenizer
    config = model.config

    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    # tokenizer._tokenizer.truncation["max_length"] = attention_window

    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos

    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    # copy position embeddings over and over to initialize the new position embeddings

    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k : (k + step)] = model.embeddings.position_embeddings.weight[2:]
        k += step

    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids.data = torch.tensor(
        [i for i in range(max_pos)]
    ).reshape(1, max_pos)

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

    # HACK: need manual change config.json
    # DONE: changed config.json architecture XLMRobertaModel to LonformerModel
    #       changed model_type xlm-roberta to longformer
    logger.warning(
        "Need manual change config.json: `architecture` XLMRobertaModel to LonformerModel; `model_type` xlm-roberta to longformer"
    )

    # this does not work...
    # callable_model.embedding_model.max_seq_length = tokenizer.model_max_length

    return callable_model


def load_longformer(embedding_model: str = "") -> Callable:
    # longformer_path = f"{os.getcwd()}\\keybert\\backend\\long_models\\"
    longformer_path = GEO_KPE_MULTIDOC_MODELS_PATH
    if not os.path.exists(longformer_path):
        os.makedirs(longformer_path)
    # model_path = f"{longformer_path}{embedding_model}"
    model_path = os.path.join(longformer_path, embedding_model)
    if os.path.exists(model_path):
        if os.listdir(model_path):
            # logging.set_verbosity_error()
            callable_model = select_backend(embedding_model[11:])
            callable_model.embedding_model._modules["0"]._modules[
                "auto_model"
            ] = XLMRobertaModel.from_pretrained(model_path, output_loading_info=False)
            callable_model.embedding_model.tokenizer = (
                XLMRobertaTokenizer.from_pretrained(
                    model_path, output_loading_info=False
                )
            )
            callable_model.embedding_model._modules["0"]._modules[
                "auto_model"
            ].config = XLMRobertaConfig.from_pretrained(
                model_path, output_loading_info=False
            )
            return callable_model
    attention_window = 512
    max_pos = 4096
    return create_long_model(embedding_model, model_path, attention_window, max_pos)


def create_long_model(embedding_model, save_model_to, attention_window, max_pos):
    callable_model = select_backend(embedding_model[11:])
    model = callable_model.embedding_model._modules["0"]._modules["auto_model"]
    tokenizer = callable_model.embedding_model.tokenizer
    config = model.config
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    # tokenizer._tokenizer.truncation["max_length"] = attention_window
    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    max_pos += 2
    config.max_position_embeddings = max_pos

    assert max_pos > current_max_pos
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k : k + step] = model.embeddings.position_embeddings.weight[2:]
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
    model.embeddings.position_ids.data = torch.tensor(
        [i for i in range(max_pos)]
    ).reshape(1, max_pos)
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value = copy.deepcopy(layer.attention.self.value)

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    if not os.path.exists(save_model_to):
        os.makedirs(save_model_to)

    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return callable_model


def create_bigbird(
    model: str, save_model_to: str, attention_window: int, max_pos: int
) -> Callable:
    callable_model = select_backend(model)

    model = callable_model.embedding_model._modules["0"]._modules["auto_model"]
    tokenizer = callable_model.embedding_model.tokenizer
    config = model.config

    # TODO: Check this
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    tokenizer._tokenizer.truncation["max_length"] = attention_window

    current_max_pos, embed_size = model.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos

    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    # copy position embeddings over and over to initialize the new position embeddings

    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[k : (k + step)] = model.embeddings.position_embeddings.weight[2:]
        k += step

    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids.data = torch.tensor(
        [i for i in range(max_pos)]
    ).reshape(1, max_pos)

    config.attention_window = [attention_window] * config.num_hidden_layers
    roberta_layers = [layer for _, layer in enumerate(model.encoder.layer)]

    big_bird_config = BigBirdConfig()
    big_bird_config.update(config.to_dict())
    big_bird_model = BigBirdModel(big_bird_config)
    big_bird_layers = [layer for _, layer in enumerate(big_bird_model.encoder.layer)]

    for layer, big_bird_layer in zip(roberta_layers, big_bird_layers):
        big_bird_layer.attention.self.query = layer.attention.self.query
        big_bird_layer.attention.self.key = layer.attention.self.key
        big_bird_layer.attention.self.value = layer.attention.self.value
        layer.attention.self = big_bird_layer.attention.self

    if not os.path.exists(save_model_to):
        os.makedirs(save_model_to)

    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return callable_model


def load_longmodel(embedding_model: str = "") -> Callable:
    supported_models = {"longformer": create_longformer, "bigbird": create_bigbird}
    # longmodel_path = f"{os.getcwd()}\\keybert\\backend\\long_models\\"
    longmodel_path = GEO_KPE_MULTIDOC_MODELS_PATH
    # get model variant
    sliced_t = embedding_model[: embedding_model.index("-")]
    if sliced_t not in supported_models:
        raise ValueError(f"Model {sliced_t} is not in supported types")
    if not os.path.exists(longmodel_path):
        os.makedirs(longmodel_path)
    # get model name
    sliced_m = embedding_model[embedding_model.index("-") + 1 :]

    # model_path = os.path.join(longmodel_path, embedding_model)
    model_path = os.path.join(longmodel_path, embedding_model)
    # f"{longmodel_path}{embedding_model}"
    # logging.set_verbosity_error()
    attention_window = 512
    max_pos = 4096
    if embedding_model == "longformer-large-4096":
        model = select_backend("paraphrase-multilingual-mpnet-base-v2")
        model.embedding_model._modules["0"]._modules[
            "auto_model"
        ] = LongformerModel.from_pretrained("allenai/longformer-large-4096")
        model.embedding_model.tokenizer = LongformerTokenizerFast.from_pretrained(
            "allenai/longformer-large-4096"
        )
        model.embedding_model._modules["0"]._modules[
            "auto_model"
        ].config = LongformerConfig.from_pretrained("allenai/longformer-large-4096")
        return model

    # if model does not exist, create new.
    # HACK: bypass with generate
    if not os.path.exists(model_path) and sliced_m != "generate":
        return supported_models[sliced_t](
            sliced_m, model_path, attention_window, max_pos
        )

    # TODO: Hack for local Sentence Transformer Longformer
    """
    .. code-block::
        :caption: validated by encoding a document and asserting model outputs are the same

        >>> keybert_outputs = kb_model.embedding_model.encode(mkduc01['d06']['documents']['LA010890-0031'], show_progress_bar=False, output_value = None)
        >>> sentence_outputs = st_model.encode(mkduc01['d06']['documents']['LA010890-0031'], show_progress_bar=False, output_value = None)
        >>> all(itertools.chain(* np.isclose(sentence_outputs['token_embeddings'], keybert_outputs['token_embeddings'])))
        True
    """
    if sliced_t == "longformer":
        if sliced_m == "generate":
            sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

            # convertion
            lmodel, lmodel_tokenizer = convert_roberta_to_longformer(
                roberta_model=sbert._modules["0"]._modules["auto_model"],
                roberta_tokenizer=sbert._modules["0"].tokenizer,
            )
            sbert.max_seq_length = 4096
            sbert._modules["0"]._modules["auto_model"] = lmodel
            sbert._modules["0"].tokenizer = lmodel_tokenizer

            callable_model = SentenceTransformerBackend(sbert)

        elif sliced_m == "generate-transformers-3":
            tmp_path = os.path.join(GEO_KPE_MULTIDOC_MODELS_PATH, embedding_model)
            raise NotImplemented
            # tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            #     "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
            # )
            # model = XLMRobertaLongModel.from_pretrained(
            #     "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
            # )

        else:
            logger.info(
                f"Loading Longformer from Sentence Transformer model {sliced_t}-{sliced_m}."
            )
            # 3rd
            # return supported_models[sliced_t](
            #     sliced_m, model_path, attention_window, max_pos
            # )
            # 2nd
            # callable_model = select_backend(sliced_m)
            # to_longformer(callable_model.embedding_model)

            # logger.debug(callable_model.embedding_model._modules["0"])
            # 1st
            embedding_model = SentenceTransformer(model_path)
            # DONE: changed config.json architecture XLMRobertaModel to LonformerModel
            #       changed model_type xlm-roberta to longformer
            #       now 3 lines below are not needed.
            # embedding_model._modules["0"]._modules[
            #     "auto_model"
            # ] = LongformerModel.from_pretrained(model_path)
            callable_model = SentenceTransformerBackend(embedding_model)
    else:
        logger.info(f"Loading base model {sliced_m}.")
        callable_model = select_backend(sliced_m)

    # callable_model = select_backend(sliced_m)
    # if sliced_t == "longformer":
    #     callable_model.embedding_model._modules["0"]._modules[
    #         "auto_model"
    #     ] = XLMRobertaModel.from_pretrained(
    #         model_path,
    #         output_loading_info=False,
    #         output_hidden_states=True,
    #         output_attentions=True,
    #     )
    #     callable_model.embedding_model.tokenizer = XLMRobertaTokenizer.from_pretrained(
    #         model_path,
    #         output_loading_info=False,
    #         output_hidden_states=True,
    #         output_attentions=True,
    #     )
    #     callable_model.embedding_model.tokenizer.save_pretrained(model_path)
    #     callable_model.embedding_model._modules["0"]._modules[
    #         "auto_model"
    #     ].config = XLMRobertaConfig.from_pretrained(
    #         model_path,
    #         output_loading_info=False,
    #         output_hidden_states=True,
    #         output_attentions=True,
    #     )
    #     # DONE: Does not work because will use too much memory.
    #     callable_model.embedding_model.max_seq_length = 4096
    logger.debug(
        callable_model.embedding_model._modules["0"]
        ._modules["auto_model"]
        .config.model_type
    )
    return callable_model
