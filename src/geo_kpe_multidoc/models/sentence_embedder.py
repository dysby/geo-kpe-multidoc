from collections import OrderedDict
from typing import Dict, List, Protocol, Union

import numpy as np
import torch
from loguru import logger
from numpy import ndarray
from tqdm.autonotebook import trange
from transformers import AutoModel, AutoTokenizer, LongformerModel


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


def _text_length(text: Union[List[int], List[List[int]]]):
    """
    Copy from SentenceTransformer
    Help function to get the length for the input text. Text can be either
    a list of ints (which means a single text as input), or a tuple of list of ints
    (representing several text inputs to the model).
    """

    if isinstance(text, dict):  # {key: value} case
        return len(next(iter(text.values())))
    elif not hasattr(text, "__len__"):  # Object has no len() method
        return 1
    elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
        return len(text)
    else:
        return sum([len(t) for t in text])  # Sum of length of individual strings


class SentenceEmbedder(Protocol):
    def tokenize(self, sentence: Union[str, List[str]], **kwargs) -> Dict:
        """Tokenize the sentence"""

    def encode(
        self, sentence, global_attention_mask=None, output_attentions=False, device=None
    ):
        """Encode the sentence"""


class LongformerSentenceEmbedder:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.attention_window = int(
            model.encoder.layer[0].attention.self.one_sided_attn_window_size * 2
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

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

    def encode(
        self,
        sentence,
        global_attention_mask=None,
        output_attentions=False,
        device=None,
    ):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentence,
            # padding="max_length",
            padding=True,
            pad_to_multiple_of=self.attention_window,
            truncation=True,
            # max_length=self.max_length,
            max_length=256,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # limit only to 128 tokens
        encoded_input["attention_mask"][128:] = 0

        if global_attention_mask is not None:
            if global_attention_mask.size(1) != encoded_input["attention_mask"].size(1):
                global_attention_mask = global_attention_mask[
                    :, : encoded_input["attention_mask"].size(1)
                ]
            encoded_input["global_attention_mask"] = global_attention_mask
            # TODO: old longformer model attention handling
            # 0 masked
            # 1 local attention
            # 2 global attention
            # encoded_input["attention_mask"] = (
            #     encoded_input["attention_mask"] + global_attention_mask
            # )
        elif global_attention_mask is None and isinstance(self.model, LongformerModel):
            global_attention_mask = torch.zeros_like(encoded_input["attention_mask"])
            global_attention_mask[:, 0] = 1  # CLS token
            encoded_input["global_attention_mask"] = global_attention_mask

        device = device if device else self.device
        encoded_input = batch_to_device(encoded_input, device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(
                **encoded_input, output_attentions=output_attentions
            )

        # Perform pooling. In this case, mean pooling
        sentence_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        output = OrderedDict(
            {
                # TODO: remove batch dimension?
                "token_embeddings": model_output[0].squeeze(),
                "input_ids": encoded_input["input_ids"],
                "attention_mask": encoded_input["attention_mask"],
                "sentence_embedding": sentence_embedding.squeeze(),
                # Output includes attention weights when output_attentions=True
                # Size(batch_size, num_heads, sequence_length, sequence_length)
                "attentions": model_output[-1] if output_attentions else None,
            }
        )

        return output

    def encode_stsbenchmark(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
        global_attention_mask=None,
    ) -> Union[List[torch.Tensor], ndarray, torch.Tensor]:
        """
        # Test STSbenchmark
        # Copy from SentenceTransformer
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        if show_progress_bar is None:
            show_progress_bar = False

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-_text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]

            # Tokenize sentences
            features = self.tokenizer(
                sentences_batch,
                # padding="max_length",
                padding=True,
                pad_to_multiple_of=self.attention_window,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )

            if global_attention_mask is not None:
                logger.critical("global_attention_mask cannot be used in batch")
                # features["global_attention_mask"] = global_attention_mask
            elif global_attention_mask is None and isinstance(
                self.model, LongformerModel
            ):
                global_attention_mask = torch.zeros_like(features["attention_mask"])
                global_attention_mask[:, 0] = 1  # CLS token
                features["global_attention_mask"] = global_attention_mask

            device = device if device else self.device
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.model(**features)

                # Perform pooling. In this case, mean pooling
                sentence_embeddings = mean_pooling(
                    out_features, features["attention_mask"]
                )

                out_features["sentence_embedding"] = sentence_embeddings

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(
                        out_features[output_value], out_features["attention_mask"]
                    ):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {
                            name: out_features[name][sent_idx] for name in out_features
                        }
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)
                # must reset global attention mask for another batch
                global_attention_mask = None

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


class BigBirdSentenceEmbedder:
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

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

    def encode(
        self, sentence, global_attention_mask=None, output_attentions=False, device=None
    ):
        # Tokenize sentences
        encoded_input = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        device = device if device else self.device
        encoded_input = batch_to_device(encoded_input, device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(
                **encoded_input, output_attentions=output_attentions
            )

        # Perform pooling. In this case, mean pooling
        sentence_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        output = OrderedDict(
            {
                # TODO: remove batch dimension?
                "token_embeddings": model_output[0].squeeze(),
                "input_ids": encoded_input["input_ids"],
                "attention_mask": encoded_input["attention_mask"],
                "sentence_embedding": sentence_embedding.squeeze(),
                # Output includes attention weights when output_attentions=True
                # Size(batch_size, num_heads, sequence_length, sequence_length)
                "attentions": model_output[-1] if output_attentions else None,
            }
        )

        return output

    def encode_stsbenchmark(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[torch.Tensor], ndarray, torch.Tensor]:
        """
        # Test STSbenchmark
        # Copy from SentenceTransformer
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        if show_progress_bar is None:
            show_progress_bar = False

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-_text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]

            # Tokenize sentences
            features = self.tokenizer(
                sentences_batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )

            device = device if device else self.device
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.model(**features)

                # Perform pooling. In this case, mean pooling
                sentence_embeddings = mean_pooling(
                    out_features, features["attention_mask"]
                )

                out_features["sentence_embedding"] = sentence_embeddings

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(
                        out_features[output_value], out_features["attention_mask"]
                    ):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {
                            name: out_features[name][sent_idx] for name in out_features
                        }
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


class NystromformerSentenceEmbedder(BigBirdSentenceEmbedder):
    def __init__(self, model: AutoModel, tokenizer: AutoTokenizer) -> None:
        super().__init__(model, tokenizer)

    def encode(
        self, sentence, global_attention_mask=None, output_attentions=False, device=None
    ):
        # Tokenize sentences
        # Do dot pad (only supports batch size 1)
        encoded_input = self.tokenizer(
            sentence,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        device = device if device else self.device
        encoded_input = batch_to_device(encoded_input, device)

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(
                **encoded_input, output_attentions=output_attentions
            )

        # Perform pooling. In this case, mean pooling
        sentence_embedding = mean_pooling(model_output, encoded_input["attention_mask"])
        output = OrderedDict(
            {
                # TODO: remove batch dimension?
                "token_embeddings": model_output[0].squeeze(),
                "input_ids": encoded_input["input_ids"],
                "attention_mask": encoded_input["attention_mask"],
                "sentence_embedding": sentence_embedding.squeeze(),
                # Output includes attention weights when output_attentions=True
                # Size(batch_size, num_heads, sequence_length, sequence_length)
                "attentions": model_output[-1] if output_attentions else None,
            }
        )

        return output

    def encode_stsbenchmark(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False,
    ) -> Union[List[torch.Tensor], ndarray, torch.Tensor]:
        """
        # Test STSbenchmark
        # Copy from SentenceTransformer
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        if show_progress_bar is None:
            show_progress_bar = False

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-_text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]

            # Tokenize sentences
            features = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
                return_attention_mask=True,
            )

            device = device if device else self.device
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.model(**features)

                # Perform pooling. In this case, mean pooling
                sentence_embeddings = mean_pooling(
                    out_features, features["attention_mask"]
                )

                out_features["sentence_embedding"] = sentence_embeddings

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(
                        out_features[output_value], out_features["attention_mask"]
                    ):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {
                            name: out_features[name][sent_idx] for name in out_features
                        }
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


def batch_to_device(batch, target_device: torch.device):
    """
    copy from sentence_transformers
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch
