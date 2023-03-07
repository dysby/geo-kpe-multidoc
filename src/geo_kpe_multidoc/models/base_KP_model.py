import re
from typing import Callable, List, Optional, Set, Tuple

import torch
from loguru import logger
from nltk.stem.api import StemmerI

from geo_kpe_multidoc.datasets.datasets import KPEDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.backend.select_backend import select_backend
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_punctuation,
    remove_whitespaces,
)

KPEScore = Tuple[str, float]


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

    def pre_process(self, txt: str = "", **kwargs) -> str:
        """
        Method that defines a pre_processing routine, removing punctuation and whitespaces
        """
        txt = remove_punctuation(txt)
        return remove_whitespaces(txt)[1:]

    def pos_tag_doc(self, doc: Document, stemming, memory, **kwargs) -> None:
        (
            doc.tagged_text,
            doc.doc_sentences,
            doc.doc_sentences_words,
        ) = self.tagger.pos_tag_text_sents_words(doc.raw_text, memory, doc.id)

        doc.doc_sentences = [
            sent.text for sent in doc.doc_sentences if sent.text.strip()
        ]

    def extract_candidates(self, tagged_doc, grammar, **kwargs) -> List[str]:
        """
        Abract method to extract all candidates
        """
        pass

    def top_n_candidates(
        self, doc, candidate_list, top_n, min_len, **kwargs
    ) -> List[Tuple]:
        """
        Abstract method to retrieve top_n candidates
        """
        pass

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
        use_cache = kwargs.get("pos_tag_memory", False)

        self.pos_tag_doc(
            doc=doc,
            stemming=None,
            memory=use_cache,
        )

        self.extract_candidates(doc, min_len, self.grammar, lemmer)

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
        pass
