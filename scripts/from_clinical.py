import copy

import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizerFast
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention


def test(model, tokenizer):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Sentences we want sentence embeddings for
    sentences = [
        "This framework generates embeddings for each input sentence",
        "Sentences are passed as a list of string.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return sentence_embeddings


def main():
    unpretrained_model_path = ""
    GLOBAL_MAX_POS = 4096
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        unpretrained_model_path, model_max_length=GLOBAL_MAX_POS
    )
    model = XLMRobertaLongModel.from_pretrained(unpretrained_model_path)


# class XLMRobertaLongSelfAttention(LongformerSelfAttention):
#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         encoder_hidden_states=None,
#         encoder_attention_mask=None,
#         output_attentions=False,
#     ):
#         return super().forward(
#             hidden_states,
#             attention_mask=attention_mask,
#             output_attentions=output_attentions,
#         )


# class XLMRobertaLongModel(XLMRobertaModel):
#     def __init__(self, config):
#         super().__init__(config)
#         # for i, layer in enumerate(self.roberta.encoder.layer):
#         for i, layer in enumerate(self.encoder.layer):
#             # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
#             layer.attention.self = XLMRobertaLongSelfAttention(config, layer_id=i)


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


def create_long_model(model, tokenizer, attention_window, max_pos, save_model_to):

    """Starting from the `roberta-base` (or similar) checkpoint, the following function converts it into an instance of `RobertaLong`.
    It makes the following changes:
       1)extend the position embeddings from `512` positions to `max_pos`. In Longformer, we set `max_pos=4096`
       2)initialize the additional position embeddings by copying the embeddings of the first `512` positions.
           This initialization is crucial for the model performance (check table 6 in [the paper](https://arxiv.org/pdf/2004.05150.pdf)
           for performance without this initialization)
       3) replaces `modeling_bert.BertSelfAttention` objects with `modeling_longformer.LongformerSelfAttention` with a attention window size `attention_window`

       The output of this function works for long documents even without pretraining.
       Check tables 6 and 11 in [the paper](https://arxiv.org/pdf/2004.05150.pdf) to get a sense of
       the expected performance of this model before pretraining."""

    model = XLMRobertaModel.from_pretrained(
        model_specified
    )  # ,gradient_checkpointing=True)

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        model_specified, model_max_length=max_pos
    )

    config = model.config

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs["model_max_length"] = max_pos
    (
        current_max_pos,
        embed_size,
    ) = model.roberta.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
    config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.roberta.embeddings.position_embeddings.weight.new_empty(
        max_pos, embed_size
    )
    # copy position embeddings over and over to initialize the new position embeddings
    k = 2
    step = current_max_pos - 2
    while k < max_pos - 1:
        new_pos_embed[
            k : (k + step)
        ] = model.roberta.embeddings.position_embeddings.weight[2:]
        k += step

    model.roberta.embeddings.position_embeddings.weight.data = new_pos_embed
    model.roberta.embeddings.position_embeddings.num_embeddings = len(
        new_pos_embed.data
    )

    # # first, check that model.roberta.embeddings.position_embeddings.weight.data.shape is correct — has to be 4096 (default) of your desired length
    # model.roberta.embeddings.position_ids = torch.arange(
    #     0, model.roberta.embeddings.position_embeddings.num_embeddings
    # )[None]

    model.roberta.embeddings.position_ids.data = torch.tensor(
        [i for i in range(max_pos)]
    ).reshape(1, max_pos)

    # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
    config.attention_window = [attention_window] * config.num_hidden_layers
    for i, layer in enumerate(model.roberta.encoder.layer):
        longformer_self_attn = LongformerSelfAttention(config, layer_id=i)
        longformer_self_attn.query = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value = copy.deepcopy(layer.attention.self.value)

        longformer_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        longformer_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        longformer_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = longformer_self_attn

    # logger.info(f"saving model to {save_model_to}")
    model.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)

    return model, tokenizer