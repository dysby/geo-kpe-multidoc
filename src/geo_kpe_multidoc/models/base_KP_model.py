import re
from enum import Enum, auto
from itertools import chain
from os import path
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

import joblib
import torch
from loguru import logger
from nltk.stem.api import StemmerI

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.datasets.datasets import KPEDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.backend.select_backend import select_backend
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_punctuation,
    remove_whitespaces,
)

KPEScore = Tuple[str, float]


class DocMode(Enum):
    GLOBAL_ATTENTION = auto()


class CandidateMode(Enum):
    GLOBAL_ATTENTION = auto()


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
        return torch.mean(doc_token_embeddings[[], :], dim=0)


class BaseKPModel:
    """
    Simple abstract class to encapsulate all KP models
    """

    def __init__(self, model):
        if model != "":
            self.model = select_backend(model)
        self.name = "{}_{}".format(
            str(self.__str__).split()[3], re.sub("-", "_", model)
        )

        self.grammar = ""
        self.counter = 0
        self.tagger: POS_tagger = None

    def pre_process(self, txt: str = "", **kwargs) -> str:
        """
        Method that defines a pre_processing routine, removing punctuation and whitespaces
        """
        txt = remove_punctuation(txt)
        return remove_whitespaces(txt)[1:]

    def _pos_tag_doc(self, doc: Document, stemming, use_cache, **kwargs) -> None:
        (
            doc.tagged_text,
            doc.doc_sentences,
            doc.doc_sentences_words,
        ) = self.tagger.pos_tag_text_sents_words(doc.raw_text, use_cache, doc.id)

        doc.doc_sentences = [
            sent.text for sent in doc.doc_sentences if sent.text.strip()
        ]

    def extract_candidates(self, tagged_doc, grammar, **kwargs) -> List[str]:
        """
        Abract method to extract all candidates
        """
        raise NotImplemented

    def top_n_candidates(
        self, doc, candidate_list, top_n, min_len, **kwargs
    ) -> List[Tuple]:
        """
        Abstract method to retrieve top_n candidates
        """
        raise NotImplemented

    def extract_kp_from_doc(
        self,
        doc: Document,
        top_n,
        min_len,
        stemmer: Optional[StemmerI] = None,
        lemmer: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """
        self.extract_candidates(doc, min_len, self.grammar, lemmer, **kwargs)

        top_n, candidate_set = self.top_n_candidates(
            doc, top_n, min_len, stemmer, **kwargs
        )

        logger.info(f"Document #{self.counter} processed")
        self.counter += 1
        torch.cuda.empty_cache()

        return (top_n, candidate_set)

    def extract_kp_from_corpus(
        self,
        corpus: KPEDataset,
        dataset,
        top_n=5,
        min_len=0,
        stemming=True,
        lemmatize=False,
        **kwargs,
    ) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality

        Parameters
        ----------
            corpus: Dataset with topic id, list of documents (txt form) for topic, and list of keyphrases for topic.

        """
        raise NotImplemented
