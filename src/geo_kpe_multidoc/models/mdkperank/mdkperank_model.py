from dataclasses import dataclass
from statistics import mean
from typing import List, Set, Tuple

import numpy as np
from nltk.stem import PorterStemmer

from geo_kpe_multidoc.datasets.datasets import KPEDataset
from geo_kpe_multidoc.document import Document

from ..base_KP_model import BaseKPModel, KPEScore
from ..embedrank import EmbedRank
from ..fusion_model import FusionModel
from ..maskrank import MaskRank
from ..pre_processing.language_mapping import choose_lemmatizer, choose_tagger
from ..pre_processing.pos_tagging import POS_tagger_spacy
from ..pre_processing.pre_processing_utils import remove_punctuation, remove_whitespaces

# from datasets.process_datasets import *


@dataclass
class KpeModelScores:
    candidates: List[str]
    scores: List[float]

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, index):
        return self.candidates[index], self.scores[index]


@dataclass
class MDKPERankOutput:
    results: List[KPEScore]


class MDKPERank(BaseKPModel):
    def __init__(self, model, tagger):
        self.base_model_embed = EmbedRank(model, tagger)
        # TODO: what how to join MaskRank
        # self.base_model_mask = MaskRank(model, tagger)
        # super().__init__(model)

    def extract_kp_from_doc(
        self, doc, top_n, min_len, stemmer=None, lemmer=None, **kwargs
    ) -> Tuple[Document, List[Tuple], List[str]]:  # Tuple[List[Tuple], List[str]]:
        """
        Extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """
        return self.base_model_embed.extract_mdkpe_embeds(
            doc, top_n, min_len, stemmer, lemmer, **kwargs
        )

    def extract_kp_from_topic(
        self,
        topic_docs: List[str],
        top_n: int = 15,
        min_len: int = 5,
        stemming: bool = False,
        lemmatize: bool = False,
        **kwargs,
    ) -> List[KPEScore]:
        """
        Extract keyphrases from list of documents

        Parameters
        ----------
            topic: List[str], a List of Documents for the topic.

        Returns
        -------
            List KPE extracted from the agregation of documents set, with their score(/embeding?)
        """
        topic_res = [
            self.extract_kp_from_doc(
                doc=Document(doc, f"topic_doc_{i}"),
                top_n=-1,
                min_len=min_len,
                stemming=stemming,
                lemmatize=lemmatize,
                **kwargs,
            )
            for i, doc in enumerate(topic_docs)
        ]
        cands = {}
        for _doc, cand_embeds, cand_set in topic_res:
            for candidate, embeding in zip(cand_set, cand_embeds):
                cands.setdefault(candidate, []).append(embeding)

        # The candidate embedding is the average of each word embeding
        # of the candidate in the document.
        cand_embeds = [np.mean(embed, axis=0) for _cand, embed in cands.items()]
        cand_set = list(cands.keys())

        res_p_doc = [
            self.base_model_embed.evaluate_n_candidates(
                doc.doc_embed, cand_embeds, cand_set
            )
            for doc, _, _ in topic_res
        ]
        scores_per_candidate = {}

        for doc, _ in res_p_doc:
            for cand_t in doc:
                if cand_t[0] not in scores_per_candidate:
                    scores_per_candidate[cand_t[0]] = []
                scores_per_candidate[cand_t[0]].append(cand_t[1])

        for cand in scores_per_candidate:
            scores_per_candidate[cand] = mean(scores_per_candidate[cand])

        scores: List[KPEScore] = sorted(
            [(cand, scores_per_candidate[cand]) for cand in scores_per_candidate],
            reverse=True,
            key=lambda x: x[1],
        )

        return scores

    def extract_kp_from_corpus(
        self,
        corpus: KPEDataset,
        top_n: int = 15,
        min_len: int = 5,
        stemming: bool = False,
        lemmatize: bool = False,
        **kwargs,
    ) -> List[List[KPEScore]]:
        """
        Extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality

        Parameters:
        ------
            corpus: TextDataset
                Iterable container with tuple (doc/docs, keyphrases} items.

        Returns:
        ------
            list of results with ((candidate, scores), candidate) items
        """
        res = [
            self.extract_kp_from_topic(
                sample, top_n, min_len, stemming, lemmatize, **kwargs
            )
            for _id, sample, _label in corpus
        ]

        return res
