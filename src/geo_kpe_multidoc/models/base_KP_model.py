import re

from typing import List, Tuple, Set

from .backend.select_backend import select_backend

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

    def pre_process(self, doc) -> str:
        """
        Abrstract method that defines a pre_processing routine
        """
        pass

    def pos_tag_doc(self, doc, stemming, **kwargs) -> List[List[Tuple]]:
        """
        Abstract method that handles POS_tagging of an entire document
        """
        pass

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
        self, doc, top_n, min_len, stemming, **kwargs
    ) -> Tuple[List[Tuple], List[str]]:
        """
        Abstract method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """
        pass

    def extract_kp_from_corpus(
        self,
        corpus,
        dataset,
        top_n=5,
        min_len=0,
        stemming=True,
        lemmatize=False,
        **kwargs
    ) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        pass
