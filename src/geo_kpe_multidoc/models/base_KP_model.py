import re
from enum import Enum, auto
from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
from keybert.backend._base import BaseEmbedder
from loguru import logger
from nltk.stem.api import StemmerI

from geo_kpe_multidoc.datasets.datasets import KPEDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.backend.select_backend import select_backend
from geo_kpe_multidoc.models.candidate_extract.candidate_extract_model import (
    CandidateExtractionModel,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    filter_tokenizer_special_tokens,
    remove_punctuation,
    remove_whitespaces,
    tokenize_hf,
)

KPEScore = Tuple[str, float]


class DocMode(Enum):
    GLOBAL_ATTENTION = auto()


class CandidateMode(Enum):
    GLOBAL_ATTENTION = auto()


def _search_mentions(model, candidate_mentions, token_ids):
    mentions = []
    # TODO: mention counts for mean_in_n_out_context
    # mentions_counts = []
    # TODO: simplifys tokenization selection due to non-uniform model classes
    if isinstance(model, BaseEmbedder):
        # original tokenization by KeyBert/SentenceTransformer
        tokenize = partial(tokenize_hf, model=model)
        model_special_ids = model.embedding_model.tokenizer.all_special_ids
    else:
        # tokenize via local SentenceEmbedder Class
        tokenize = model.tokenize
        model_special_ids = model.tokenizer.all_special_ids

    for original_mention in candidate_mentions:
        # TODO: account for special tokenization schemes ' word' vs 'word'"
        # for mention in (original_mention, " " + original_mention):
        for mention in (original_mention,):
            tokenized_candidate = tokenize(mention)
            # filt_ids = filter_special_tokens(tokenized_candidate["input_ids"])
            filt_ids = filter_tokenizer_special_tokens(
                tokenized_candidate["input_ids"],
                model_special_ids,
            )

            # Should not be Empty after filter
            if filt_ids:
                mentions += find_occurrences(filt_ids, token_ids)
    return mentions  # , mentions_counts


def find_occurrences(a: List[int], b: List[int]) -> List[List[int]]:
    occurrences = []
    # TODO: escape search in right padding indexes
    for i in range(len(b) - len(a) + 1):
        if b[i : i + len(a)] == a:
            occurrences.append(list(range(i, i + len(a))))
    return occurrences


def candidate_embedding_from_tokens(
    # TODO: simplify candidate embedding computation
    candidate: List[int],
    doc_input_ids: List[int],
    doc_token_embeddings: torch.Tensor,
) -> torch.Tensor:
    n_tokens, embed_dim = doc_token_embeddings.size()

    candidate_occurrences = find_occurrences(candidate, doc_input_ids)
    if len(candidate_occurrences) != 0:
        embds = torch.empty(size=(len(candidate_occurrences), embed_dim))
        for i, occurrence in enumerate(candidate_occurrences):
            embds[i] = torch.mean(doc_token_embeddings[occurrence, :], dim=0)
        return torch.mean(embds, dim=0)
    else:
        logger.warning(f"Did not find candidate occurrences: {candidate}")
        # return torch.mean(doc_token_embeddings[[], :], dim=0)
        raise ValueError(f"Did not find candidate occurrences: {candidate}")


class BaseKPModel:
    """
    Simple abstract class to encapsulate all KP models
    """

    def __init__(
        self, model, candidate_selection_model: CandidateExtractionModel, **kwargs
    ):
        if model != "":
            self.model = select_backend(model)
        self.name = "{}_{}".format(
            str(self.__str__).split()[3], re.sub("-", "_", model)
        )
        self.candidate_selection_model = candidate_selection_model

        self.counter = 1

    def pre_process(self, txt: str = "", **kwargs) -> str:
        """
        Clean text by removing punctuation and whitespaces
        """
        txt = remove_punctuation(txt)
        return remove_whitespaces(txt)[1:]

    def extract_candidates(
        self, doc, kp_min_len, lemmer, **kwargs
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        return self.candidate_selection_model(
            doc=doc, kp_min_len=kp_min_len, lemmer_lang=lemmer, **kwargs
        )

    def top_n_candidates(
        self, doc, candidate_list, positions, top_n, **kwargs
    ) -> List[Tuple]:
        """
        Abstract method to retrieve top_n candidates
        """
        raise NotImplementedError

    def extract_kp_from_doc(
        self,
        doc: Document,
        top_n: int = -1,
        kp_min_len: int = 0,
        # stemmer: Optional[StemmerI] = None,
        lemmer: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional
        arguments relevant to its specific functionality
        """
        self.extract_candidates(doc, kp_min_len, lemmer, **kwargs)

        top_n_candidates_scores, candidate_set = self.top_n_candidates(
            doc,
            candidate_list=doc.candidate_set,
            positions=doc.candidate_positions,
            top_n=top_n,
            **kwargs,
        )

        logger.debug(f"Document #{self.counter} processed")
        self.counter += 1

        return (top_n_candidates_scores, candidate_set)

    def extract_kp_from_corpus(
        self,
        corpus: KPEDataset,
        dataset,
        top_n=5,
        kp_min_len=0,
        lemmatize=False,
        **kwargs,
    ) -> List[List[Tuple]]:
        """
        Extracts key-phrases from a list of given documents, with optional
        arguments relevant to its specific functionality

        Parameters
        ----------
            corpus: Dataset with topic id, list of documents (txt form) for topic,
            and list of keyphrases for topic.

        """
        raise NotImplementedError


class ExtractionEvaluator(BaseKPModel):
    def __init__(self, model, candidate_selection_model, **kwargs):
        self.candidate_selection_model = candidate_selection_model
        self.counter = 1

    def extract_kp_from_doc(
        self,
        doc: Document,
        top_n,
        kp_min_len,
        stemmer: Optional[StemmerI] = None,
        lemmer: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[List[Tuple], List[str]]:
        """
        Shallow model for candidate extraction evalutation
        """
        self.extract_candidates(doc, kp_min_len, lemmer, **kwargs)

        top_n = len(doc.candidate_set) if top_n == -1 else top_n

        # remove duplicates from promptrank keeping relative positions...
        # candidate selection keeps candidates sorted by its position in source document
        top_n_scores = [
            (k, v) for k, v in dict.fromkeys(doc.candidate_set, 0.5).items()
        ]

        logger.debug(f"Document #{self.counter} processed")
        self.counter += 1

        return (top_n_scores, doc.candidate_set)
