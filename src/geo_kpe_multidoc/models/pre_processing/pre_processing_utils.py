import re
import string
from typing import Callable, List, Tuple, Union

import torch
from keybert.backend._base import BaseEmbedder
from nltk.corpus import stopwords
from nltk.stem.api import StemmerI
from nltk.stem.snowball import SnowballStemmer
from simplemma import simplemma, text_lemmatizer

from geo_kpe_multidoc.datasets.language import ISO_to_language

from .stopwords import ENGLISH_STOP_WORDS

# TODO: consider special tokens from AutoTokenizer.
# tokenizer.bos_token_id
# tokenizer.eos_token_id
# tokenizer.unk_token_id
# tokenizer.sep_token_id
# tokenizer.pad_token_id
# tokenizer.cls_token_id
# tokenizer.mask_token_id
# tokenizer.additional_special_tokens_ids

SPECIAL_TOKEN_IDS = {0, 1, 2, 3, 250001}

# TODO: clean from https://github.com/adamwawrzynski/vectorized_documents_benchmark/blob/master/utils/preprocess.py
# def clean_string_longformer(
#         string
# ):
#     return preprocess_text_longformer(string)

# def preprocess_text_longformer(
#         raw,
#         remove_stopwords=False,
#         lemmatize=False,
#         name_entity_extraction=False,
#         contraction_expanding=False,
#         regex1="[^a-zA-Z0-9'\.\?\!\:\;\"\-,]",
# ):
#     text_without_email = re.sub(r'[\w\.-]+@[\w\.-]+', ' ', raw)
#     text = re.sub(regex1, " ", text_without_email)
#     text = re.sub(r'\s+', ' ', text)

#     return text


def remove_punctuation(text: str = "") -> str:
    """
    Quick snippet to remove punctuation marks
    """
    # self.punctuation_regex = (
    #         "[!\"#\$%&'\(\)\*\+,\.\/:;<=>\?@\[\]\^_`{\|}~\-\–\—\‘\’\“\”]"
    #     )
    # TODO: why this is different from document pontuation regex?
    return re.sub("[.,;:\"'!?`´()$£€\-^|=/<>]", " ", text)


def remove_hyphens_and_dots(text: str):
    # TODO: other hyphen forms - – — ?
    return text.replace("-", " ").replace(".", "")


def remove_special_chars(text: str) -> str:
    """
    Quick snippet to remove punctuation marks
    """
    # self.punctuation_regex = (
    #         "[!\"#\$%&'\(\)\*\+,\.\/:;<=>\?@\[\]\^_`{\|}~\-\–\—\‘\’\“\”]"
    #     )
    # TODO: why this is different from document pontuation regex?
    return re.sub(r"[\"'`´()$£€\-_^|=/<>~]", " ", text)


def remove_new_lines_and_tabs(text: str) -> str:
    return re.sub(r"[\n\r\t]+", " ", text)


def remove_whitespaces(text: str = "") -> str:
    """
    Quick snippet to remove whitespaces
    """
    # text = re.sub("\n", " ", text)
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


def lemmatize(text: Union[str, List], lang: str) -> Union[str, List]:
    """Lemmatize text but remove isolated `.` from output.
    This is important because we do not want `Mr. Smith` to be lemmatized to `Mr . Smith`.
    In case of decimals it does not take effect eg. `10.5` og `10.5g` are correctly
    lemmatized, keeping the `.` in place.

    Using text_lemmatizer also lemmatize `inter-national community` correctly to `inter-national community`
    """
    # # old simplemma 0.6.0
    # # simplemma.load_data(lemmatizers[dataset])
    # # simplemma.lemmatize(w, lemmer)

    # from simplemma import __version__

    # if __version__ == "0.6.0":
    #     lemmer = simplemma.load_data(lang)
    #     if isinstance(text, List):
    #         return [
    #             " ".join([w for w in text_lemmatizer(line, lemmer) if w != "."]).lower()
    #             for line in text
    #         ]
    #     return " ".join([w for w in text_lemmatizer(text, lemmer) if w != "."]).lower()
    #     # simplemma.lemmatize(w, lemmer)

    # else:
    #     if isinstance(text, List):
    #         return [
    #             " ".join([w for w in text_lemmatizer(line, lang) if w != "."]).lower()
    #             for line in text
    #         ]
    #     return " ".join([w for w in text_lemmatizer(text, lang) if w != "."]).lower()
    # if isinstance(text, List):
    #     return [
    #         " ".join([w for w in text_lemmatizer(line, lang) if w != "."]).lower()
    #         for line in text
    #     ]
    # return " ".join([w for w in text_lemmatizer(text, lang) if w != "."]).lower()
    if isinstance(text, List):
        return [
            " ".join(
                [
                    simplemma.lemmatize(w, lang)
                    for w in remove_hyphens_and_dots(line).split()
                ]
            ).lower()
            for line in text
        ]
    else:
        return " ".join(
            [
                simplemma.lemmatize(w, lang)
                for w in remove_hyphens_and_dots(text).split()
            ]
        ).lower()


def stemming(text, stemmer):
    return " ".join([stemmer.stem(w) for w in text.split()])


def select_stemmer(lang: str = "en") -> StemmerI:
    # TODO: Multi Language SnowballStemmer
    # copy from https://github.com/LIAAD/kep/blob/master/kep/utility.py

    if lang == "en":
        # create a new instance of a porter stemmer
        stemmer = SnowballStemmer("porter")
    else:
        # create a new instance of a porter stemmer
        stemmer = SnowballStemmer(ISO_to_language[lang], ignore_stopwords=True)
    return stemmer


def filter_special_tokens(input_ids: torch.Tensor) -> List[int]:
    return [i for i in input_ids.squeeze().tolist() if i not in SPECIAL_TOKEN_IDS]


def filter_tokenizer_special_tokens(
    input_ids: torch.Tensor, special_tokens: List[int]
) -> List[int]:
    return [i for i in input_ids.squeeze().tolist() if i not in special_tokens]


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
    # inputs = model.embedding_model.tokenizer(text, return_tensors="pt", max_length=2048)
    inputs = model.embedding_model.tokenizer(text, return_tensors="pt", max_length=4096)
    outputs = model.embedding_model._modules["0"]._modules["auto_model"](**inputs)

    tokens = inputs.input_ids.squeeze().tolist()
    last_layer_attention = outputs.attentions[-1][0]
    return (tokens, last_layer_attention)


def sentence_transformer_tokenize(text: str) -> List[int]:
    tokens_filtered = [
        token for token in text.lower().split() if token not in ENGLISH_STOP_WORDS
    ]

    return tokens_filtered
