from typing import List, Set, Tuple

import numpy as np
import tqdm
from nltk import RegexpParser

from geo_kpe_multidoc.models.base_KP_model import BaseKPModel
from geo_kpe_multidoc.models.pre_processing.language_mapping import (
    choose_lemmatizer,
    choose_tagger,
)
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy


class CandidateExtract(BaseKPModel):
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models
    # TODO: backup from original, not working
    """

    def __init__(self, model, tagger):
        super().__init__(model)

        self.tagger = POS_tagger_spacy(tagger)
        self.grammar = """  NP: 
        {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}"""
        self.single_word_grammar = {"PROPN", "NOUN", "ADJ"}

    def update_tagger(self, dataset: str = "") -> None:
        self.tagger = (
            POS_tagger_spacy(choose_tagger(dataset))
            if choose_tagger(dataset) != self.tagger.name
            else self.tagger
        )

    def pos_tag_doc(
        self, doc: str = "", stemming: bool = True, **kwargs
    ) -> List[List[Tuple]]:
        """
        Method that handles POS_tagging of an entire document, pre-processing or stemming it in the process
        """
        tagged_doc = self.tagger.pos_tag_doc(doc)
        for sent in tagged_doc:
            for i in range(1, len(sent) - 1):
                if i + 1 < len(sent):
                    if sent[i][0] == "-":
                        sent[i] = (f"{sent[i-1][0]}-{sent[i+1][0]}", "NOUN")
                        del sent[i + 1]
                        del sent[i - 1]
        return tagged_doc

    def extract_candidates(
        self, tagged_doc: List[List[Tuple]] = [], **kwargs
    ) -> List[str]:
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document
        """

        use_cache = kwargs.get("pos_tag_memory", False)
        self._pos_tag_doc(
            doc=doc,
            stemming=None,
            use_cache=use_cache,
        )

        candidate_set = set()
        parser = RegexpParser(self.grammar)
        np_trees = parser.parse_sents(tagged_doc)

        for tree in np_trees:
            for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
                candidate_set.add(" ".join(word for word, tag in subtree.leaves()))
                for word, tag in subtree.leaves():
                    if tag in self.single_word_grammar:
                        candidate_set.add(word)

        candidate_set = {kp for kp in candidate_set if len(kp.split()) <= 7}
        return list(candidate_set)

    def extract_kp_from_doc(
        self, doc, top_n, min_len, stemming, **kwargs
    ) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """

        tagged_doc = self.pos_tag_doc(doc, **kwargs)
        candidate_list = self.extract_candidates(tagged_doc, **kwargs)
        print("doc finished\n")
        return ([], candidate_list)

    def extract_kp_from_corpus(
        self, corpus, top_n=5, min_len=0, stemming=False, **kwargs
    ) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        return [
            self.extract_kp_from_doc(doc[0], top_n, min_len, stemming, **kwargs)
            for doc in tqdm(corpus)
        ]
