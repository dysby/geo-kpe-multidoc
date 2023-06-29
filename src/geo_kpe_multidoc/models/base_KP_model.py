import re
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import torch
from keybert.backend._base import BaseEmbedder
from loguru import logger
from nltk.stem.api import StemmerI

from geo_kpe_multidoc.datasets.datasets import KPEDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.backend.select_backend import select_backend
from geo_kpe_multidoc.models.candidate_extract.candidate_extract_model import (
    KPECandidateExtractionModel,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    filter_special_tokens,
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
    for mention in candidate_mentions:
        if isinstance(model, BaseEmbedder):
            # original tokenization by KeyBert/SentenceTransformer
            tokenized_candidate = tokenize_hf(mention, model)
        else:
            # tokenize via local SentenceEmbedder Class
            tokenized_candidate = model.tokenize(mention)

        filt_ids = filter_special_tokens(tokenized_candidate["input_ids"])

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
        return torch.mean(doc_token_embeddings[[], :], dim=0)


class BaseKPModel:
    """
    Simple abstract class to encapsulate all KP models
    """

    def __init__(self, model, tagger):
        if model != "":
            self.model = select_backend(model)
        self.name = "{}_{}".format(
            str(self.__str__).split()[3], re.sub("-", "_", model)
        )

        self.candidate_selection_model = KPECandidateExtractionModel(tagger=tagger)

        self.counter = 1

    def pre_process(self, txt: str = "", **kwargs) -> str:
        """
        Method that defines a pre_processing routine, removing punctuation and whitespaces
        """
        txt = remove_punctuation(txt)
        return remove_whitespaces(txt)[1:]

    def _pos_tag_doc(self, doc: Document, stemming, use_cache, **kwargs) -> None:
        self.candidate_selection_model._pos_tag_doc(doc, use_cache)
        # doc.doc_sentences = [
        #     sent.text for sent in doc.doc_sentences if sent.text.strip()
        # ]

    def extract_candidates(self, doc, min_len, lemmer, **kwargs) -> List[str]:
        self.candidate_selection_model(
            doc=doc, min_len=min_len, lemmer_lang=lemmer, **kwargs
        )

    def top_n_candidates(
        self, doc, candidate_list, top_n, min_len, **kwargs
    ) -> List[Tuple]:
        """
        Abstract method to retrieve top_n candidates
        """
        raise NotImplementedError

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
        self.extract_candidates(doc, min_len, lemmer, **kwargs)

        top_n, candidate_set = self.top_n_candidates(
            doc, top_n, min_len, stemmer, **kwargs
        )

        logger.debug(f"Document #{self.counter} processed")
        self.counter += 1

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
        raise NotImplementedError


class ExtractionEvaluator(BaseKPModel):
    def __init__(self, model, tagger):
        self.candidate_selection_model = KPECandidateExtractionModel(tagger=tagger)
        self.counter = 1

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
        Shallow Model for Candidate Extraction Evalutation
        """
        self.extract_candidates(doc, min_len, lemmer, **kwargs)

        top_n = len(doc.candidate_set) if top_n == -1 else top_n

        top_n_scores = [(candidate, 0.5) for candidate in doc.candidate_set[:top_n]]

        logger.debug(f"Document #{self.counter} processed")
        self.counter += 1

        return (top_n_scores, doc.candidate_set)
