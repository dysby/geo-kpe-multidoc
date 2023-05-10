from collections import OrderedDict
from typing import Dict, List, Union

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
        self.max_length = tokenizer.model_max_length
        self.attention_window = int(
            model.encoder.layer[0].attention.self.one_sided_attn_window_size * 2
        )

    def tokenize(self, sentence: Union[str, List[str]], **kwargs) -> Dict:
        # for global_attention
        padding = kwargs.pop("padding", False)

        return self.tokenizer(
            sentence,
            padding=padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            **kwargs,
        )

    def encode(self, sentence, global_attention_mask=None, device=None):
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

        local_mask = encoded_input["attention_mask"].detach().clone()

        if global_attention_mask is not None:
            # 0 masked
            # 1 local attention
            # 2 global attention
            encoded_input["attention_mask"] = (
                encoded_input["attention_mask"] + global_attention_mask
            )

            # raise NotImplemented

        if device:
            encoded_input = batch_to_device(encoded_input, device)
            local_mask.to(device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embedding = mean_pooling(model_output, local_mask)

        # TODO: check cpu
        local_mask = local_mask.cpu()
        output = OrderedDict(
            {
                # TODO: remove batch dimension?
                "token_embeddings": model_output[0].squeeze(),
                "input_ids": encoded_input["input_ids"],
                "attention_mask": local_mask,
                "sentence_embedding": sentence_embedding.squeeze(),
            }
        )

        return output


def batch_to_device(batch, target_device: torch.device):
    """
    copy from sentence_transformers
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch
