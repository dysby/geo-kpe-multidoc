import torch
from transformers import (
    LongformerSelfAttention,
    XLMRobertaModel,
    XLMRobertaTokenizerFast,
)
from transformers.models.longformer.modeling_longformer import LongformerSelfAttention


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


class SentenceEmbedder:
    def __init__(self) -> None:

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(
            "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
        )
        self.model = XLMRobertaLongModel.from_pretrained(
            "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
        )
        self.max_length = 4096
        self.attention_window = 512

    def encode(self, sentence):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentence,
            # padding="max_length",
            padding=True,
            pad_to_multiple_of=self.attention_window,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        return sentence_embeddings.detach().numpy()