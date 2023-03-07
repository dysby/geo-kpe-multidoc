import re
import string
from typing import Callable, List, Tuple

from keybert.backend._base import BaseEmbedder
from nltk.corpus import stopwords

from .stopwords import ENGLISH_STOP_WORDS

SPECIAL_TOKEN_IDS = {0, 1, 2, 3, 250001}


def remove_punctuation(text: str = "") -> str:
    """
    Quick snippet to remove punctuation marks
    """
    # self.punctuation_regex = (
    #         "[!\"#\$%&'\(\)\*\+,\.\/:;<=>\?@\[\]\^_`{\|}~\-\–\—\‘\’\“\”]"
    #     )
    # TODO: why this is different from document pontuation regex?
    return re.sub("[.,;:\"'!?`´()$£€\-^|=/<>]", " ", text)


def remove_whitespaces(text: str = "") -> str:
    """
    Quick snippet to remove whitespaces
    """
    text = re.sub("\n", " ", text)
    return re.sub("\s{2,}", " ", text)


def remove_stopwords(text: str = "") -> str:
    """
    Quick snippet to remove stopwords
    """

    res = ""
    for word in text.split():
        if word not in stopwords.words("English"):
            res += " {}".format(word)
    return res[1:]


def filter_token_ids(input_ids: List[List[int]]) -> List[int]:
    return [i for i in input_ids.squeeze().tolist() if i not in SPECIAL_TOKEN_IDS]


def tokenize(text: str, model: BaseEmbedder) -> Tuple:
    tokenized = model.embedding_model.tokenizer(
        text, return_tensors="pt", return_attention_mask=True
    )
    tokens = [
        i for i in tokenized.input_ids.squeeze().tolist() if i not in SPECIAL_TOKEN_IDS
    ]

    return tokens, [model.embedding_model.tokenizer.decode(t) for t in tokens]


def tokenize_hf(text: str, model: BaseEmbedder) -> List:
    return model.embedding_model.tokenizer(
        text, padding=True, truncation=True, return_tensors="pt"
    )


def tokenize_attention_embed(text: str, model: BaseEmbedder) -> Tuple:
    inputs = model.embedding_model.tokenizer(text, return_tensors="pt", max_length=2048)
    outputs = model.embedding_model._modules["0"]._modules["auto_model"](**inputs)

    tokens = inputs.input_ids.squeeze().tolist()
    last_layer_attention = outputs.attentions[-1][0]
    return (tokens, last_layer_attention)


def sentence_transformer_tokenize(text: str) -> List[int]:
    tokens_filtered = [
        token for token in text.lower().split() if token not in ENGLISH_STOP_WORDS
    ]

    return tokens_filtered
