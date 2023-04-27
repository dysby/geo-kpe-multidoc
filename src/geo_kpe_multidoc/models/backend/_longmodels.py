import copy
import logging
import math
import os
from typing import Callable

import torch
from keybert.backend._sentencetransformers import SentenceTransformerBackend
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer  # XLMRobertaTokenizerFast,
from transformers import (  # RobertaForMaskedLM,; RobertaTokenizerFast,; AutoModel,
    LongformerConfig,
    LongformerModel,
    LongformerSelfAttention,
    LongformerTokenizerFast,
    XLMRobertaConfig,
    XLMRobertaModel,
    XLMRobertaTokenizer,
)

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_MODELS_PATH
from geo_kpe_multidoc.models.backend.roberta2longformer.roberta2longformer import (
    convert_roberta_to_longformer,
)

# from ...keybert.backend._utils import select_backend
from .select_backend import select_backend

# from transformers.modeling_longformer import LongformerSelfAttention # v3.0.2


class XLMRobertaLongSelfAttention(LongformerSelfAttention):
    """
    from https://github.com/allenai/longformer/issues/215
    For transformers=4.12.5
    For transformers=4.26

    From XLMRobertaSelfAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,

    to

    LongformerSelfAttention

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        [`LongformerSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LongformerModel.forward`] to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LongformerModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
    """

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
        # is_global_attn = any(is_index_global_attn.flatten()) PR #5811
        is_global_attn = is_index_global_attn.flatten().any().item()
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


class RobertaLongSelfAttention(LongformerSelfAttention):
    """For transformers 3.0.2"""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )


class T3_RobertaLongModel(XLMRobertaModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.roberta.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = RobertaLongSelfAttention(config, layer_id=i)


def to_longformer_ashtonomy_grather_3_4(
    save_model_to, model, tokenizer, attention_window, model_max_length
):
    config = model.config
    position_embeddings = model.roberta.embeddings.position_embeddings

    tokenizer.model_max_length = model_max_length
    tokenizer.init_kwargs["model_max_length"] = model_max_length
    current_model_max_length, embed_size = position_embeddings.weight.shape

    # NOTE: RoBERTa has positions 0,1 reserved
    # embedding size is max position + 2
    model_max_length += 2
    config.max_position_embeddings = model_max_length
    assert (
        model_max_length > current_model_max_length
    ), "New model max_length must be longer than current max_length"

    # BUG for XLM: Need to make all zeros sice too large base model
    new_pos_embed = position_embeddings.weight.new_zeros(model_max_length, embed_size)

    k = 2
    step = current_model_max_length - 2
    while k < model_max_length - 1:
        new_pos_embed[k : (k + step)] = position_embeddings.weight[2:]
        k += step

    # HACK for Huggingface transformers >=3.4.0 and < 4.0
    # https://github.com/huggingface/transformers/issues/6465#issuecomment-719042969
    position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_embeddings.num_embeddings = len(
        new_pos_embed.data
    )
    num_model_embeddings = position_embeddings.num_embeddings
    model.roberta.embeddings.position_ids = torch.arange(0, num_model_embeddings)[None]

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = layer.attention.self.query
        longformer_self_attn.key_global = layer.attention.self.key
        longformer_self_attn.value_global = layer.attention.self.value

        layer.attention.self = longformer_self_attn

    logger.info(f"saving model to {save_model_to}")
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)
    return model, tokenizer


def to_longformer_t_v4(
    base_model: SentenceTransformer,
    max_pos: int = 4096,
    attention_window: int = 512,
    copy_from_position: int = None,
):
    logger.info("Transform SentenceTransformer to longformer using Transformers v 4.26")
    model = base_model._modules["0"]._modules["auto_model"]
    config = model.config
    tokenizer = base_model.tokenizer

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    (
        current_max_pos,
        embed_size,
    ) = model.embeddings.position_embeddings.weight.shape

    # add this to keep shape compability with sentence_transformer
    tokenizer.padding = "max_lenght"

    # test copy only a part of the position embeddings
    if copy_from_position:
        current_max_pos = copy_from_position

    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    # TODO: HACK test copy_from_position == 130
    if torch.any(new_pos_embed.isnan()):
        new_pos_embed = new_pos_embed.nan_to_num()
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        # copy only from 2 to 512 or copy only from 2 to 130, current_max_pos can be
        # set from model or from copy_from_position parameter
        new_pos_embed[k : (k + step)] = model.embeddings.position_embeddings.weight[
            2:current_max_pos
        ]
        k += step
    model.embeddings.position_embeddings.weight.data = new_pos_embed
    model.embeddings.position_ids.data = torch.tensor(
        [i for i in range(max_pos)]  # list(range(max_pos))
    ).reshape(1, max_pos)

    # model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos) # v4.2.0
    # model.roberta.embeddings.position_ids = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos) # v3.0.2

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.encoder.layer):
        longformer_self_attn = XLMRobertaLongSelfAttention(config, layer_id=i)
        longformer_self_attn.query = layer.attention.self.query
        longformer_self_attn.key = layer.attention.self.key
        longformer_self_attn.value = layer.attention.self.value

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    # TODO: check base model is good
    base_model.max_seq_length = tokenizer.model_max_length
    return model, tokenizer


def to_longformer_t_v3(
    base_model: SentenceTransformer, max_pos=4096, attention_window=512
):
    """Transform a `base_model` (RoBERTa) into a Longformer with sparce attention.

    For transformers 3.0.2
    """

    logger.info("Transform to longformer")
    model = base_model._modules["0"]._modules["auto_model"]
    config = model.config
    tokenizer = base_model.tokenizer

    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    # TODO: check this padding, not in original
    # tokenizer.padding = "max_lenght"
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

    # model.roberta.embeddings.position_ids.data = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos) # v4.2.0
    # model.roberta.embeddings.position_ids = torch.tensor([i for i in range(max_pos)]).reshape(1, max_pos) # v3.0.2

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers

    for i, layer in enumerate(model.encoder.layer):
        # longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        # copy from
        longformer_self_attn = RobertaLongSelfAttention(config, layer_id=i)
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


def create_long_model(embedding_model: str, save_model_to, attention_window, max_pos):
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


def load_longmodel(embedding_model: str = "") -> Callable:
    supported_models = {"longformer": create_longformer}  # , "bigbird": create_bigbird}
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
    if not os.path.exists(model_path) and "generate" not in sliced_m:
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
        if sliced_m == "generate-roberta2longformer":
            logger.info(
                f"Convert to Longformer by roberta2longformer from Sentence Transformer model"
            )
            sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

            # convertion by roberta2longformer
            lmodel, lmodel_tokenizer = convert_roberta_to_longformer(
                roberta_model=sbert._modules["0"]._modules["auto_model"],
                roberta_tokenizer=sbert._modules["0"].tokenizer,
            )
            sbert.max_seq_length = 4096
            sbert._modules["0"]._modules["auto_model"] = lmodel
            sbert._modules["0"].tokenizer = lmodel_tokenizer

            callable_model = SentenceTransformerBackend(sbert)

        elif sliced_m == "generate-original":
            """To test in RAM transformation transformer v4 (NOT working transformers 3.0.2)"""
            sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
            logger.info(
                f"Convert to Longformer by original script from Sentence Transformer model"
            )
            long_model, tokenizer = to_longformer_t_v4(sbert)

            sbert.max_seq_length = 4096
            sbert._modules["0"].auto_model = long_model

            # token_type_ids need to be removed (when in model in RAM to longformer)
            del sbert._modules["0"].auto_model.embeddings.token_type_ids

            sbert.tokenizer = tokenizer

            callable_model = SentenceTransformerBackend(sbert)
            return callable_model
        else:
            logger.info(
                f"Loading Longformer from Sentence Transformer model {sliced_t}-{sliced_m}."
            )
            # callable_model = select_backend(sliced_m)
            # to_longformer(callable_model.embedding_model)
            embedding_model = SentenceTransformer(model_path)
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
