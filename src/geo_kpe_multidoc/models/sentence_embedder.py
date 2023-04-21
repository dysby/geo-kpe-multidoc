from collections import OrderedDict

import torch
from transformers import AutoModel, AutoTokenizer


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
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer) -> None:
        self.model = model
        self.model.eval()
        self.tokenizer = tokenizer
        self.max_length = 4096
        self.attention_window = 512

    def encode(self, sentence, global_attention_mask=None):
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

        if global_attention_mask:
            raise NotImplemented

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        output = OrderedDict(
            {
                "token_embeddings": model_output[0],
                "input_ids": encoded_input["input_ids"],
                "attention_mask": encoded_input["attention_mask"],
                "sentence_embedding": sentence_embedding.squeeze(),
            }
        )

        return output
