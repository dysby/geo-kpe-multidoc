from collections import namedtuple
from dataclasses import dataclass
from itertools import chain
from statistics import mean
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from geo_kpe_multidoc.datasets.datasets import KPEDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.mdkperank.mdkperank_strategy import STRATEGIES

from ..base_KP_model import BaseKPModel, KPEScore
from ..embedrank import EmbedRank


@dataclass
class KpeModelScores:
    candidates: List[str]
    scores: List[float]

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, index):
        return self.candidates[index], self.scores[index]


MDKPERankOutput = namedtuple(
    "MDKPERankOutput",
    "top_n_scores candidate_document_matrix documents_embeddings candidate_embeddings",
)


class MDKPERank(BaseKPModel):
    def __init__(self, model: EmbedRank, rank_strategy: str = "MEAN"):
        self.base_model_embed: EmbedRank = model
        # TODO: what how to join MaskRank
        # self.base_model_mask = MaskRank(model, tagger)
        self.name = (
            ".".join([self.__class__.__name__, model.name.split("_")[0]])
            + model.name[model.name.index("_") :]
        )

        self.ranking_strategy = STRATEGIES[rank_strategy]()

    def extract_kp_from_doc(
        self, doc, top_n, min_len, stemmer=None, lemmer=None, **kwargs
    ) -> Tuple[Document, List[np.ndarray], List[str]]:  # Tuple[List[Tuple], List[str]]:
        """
        Extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """

        self.base_model_embed.extract_candidates(doc, min_len, lemmer, **kwargs)

        _, cand_embeds, candidate_set = self.base_model_embed.embed_candidates(
            doc,
            stemmer=stemmer,
            **kwargs,
        )

        return (doc, cand_embeds, candidate_set)

    def _get_locations(
        self, docs_geo_coords, docs_of_candidate: List[str]
    ) -> List[Tuple[float, float]]:
        """
        Get all coordenates of documents tha mention the candidate keyphrase
        TODO: what about repeated locations?
        """
        locations = [docs_geo_coords[doc_id] for doc_id in docs_of_candidate]
        return list(chain(*locations))

    def extract_kp_from_topic_geo(
        self,
        topic_docs: List[Document],
        top_n: int = 15,
        min_len: int = 5,
        stemming: bool = False,
        lemmatize: bool = False,
        **kwargs,
    ) -> List[KPEScore]:
        """
        Extract keyphrases from list of documents.

        Parameters
        ----------
        topic : List[Document]
            A list of Documents for the topic.
        top_n :
            `TODO:` NOT USED
        min_len : int = 5
        stemming : bool = False
        lemmatize : bool = False
        **kwargs
            These parameters will be passed to inner functions calls.

        Returns
        -------
        List[KPEScore]
            List of KPE extracted from the agregation of documents set, with their score(/embedding?)
        """
        topic_res = [
            self.extract_kp_from_doc(
                doc,
                top_n=top_n,
                min_len=min_len,
                stemming=stemming,
                lemmatize=lemmatize,
                **kwargs,
            )
            for doc in topic_docs
        ]
        # Topic_res is a list for each document with (doc, candidades_embeddings, candidates)

        candidates = {}  # candidate: [embedding]
        docs_ids = []
        # a dict with the set of docs the candidate is mentioned
        candidate_document_matrix = {}
        for doc, doc_cand_embeds, doc_cand_set in topic_res:
            for candidate, embedding_in_doc in zip(doc_cand_set, doc_cand_embeds):
                candidates.setdefault(candidate, []).append(embedding_in_doc)
                candidate_document_matrix.setdefault(candidate, set()).add(doc.id)
            docs_ids.append(doc.id)

        # The candidate embedding is the average of each embedding
        # of the candidate in the documents.
        # use sorting to assert arrays have the same index
        candidates_embeddings = [
            np.mean(embeddings_in_docs, axis=0)
            for _candidate, embeddings_in_docs in sorted(candidates.items())
        ]
        candidates = sorted(candidates.keys())

        # compute a candidate <- document score matrix
        ranking_p_doc: Dict[Tuple[List[Tuple[str, float]], List[str]]] = {
            doc.id: self.base_model_embed._rank_candidates(
                doc.doc_embed, candidates_embeddings, candidates
            )
            for doc, _, _ in topic_res
        }
        # this ranking per doc have a candidate list, it can be discarded

        keyphrase_semantic_score = {}
        # n candidates by m documents
        # candidate_document_matrix = np.zeros([len(candidates), len(docs_ids)])
        # Score agregation from all documents
        # TODO: If a candidate is only mentioned in one doc, the overall score can be biased?
        for _, (candidate_score_for_doc, _) in ranking_p_doc.items():
            for candidate, sim_score in candidate_score_for_doc:
                keyphrase_semantic_score.setdefault(candidate, []).append(sim_score)
                # count observation of candidate in document, TODO: COMPUTED BEFORE REMOVE
                # candidate_document_matrix[
                #     candidates.index(candidate), docs_ids.index(doc_id)
                # ] += 1

        keyphrase_semantic_score = {
            cand: np.mean(scores) for cand, scores in keyphrase_semantic_score.items()
        }

        top_n_scores = self._sort_candidate_score(keyphrase_semantic_score)

        # TODO: Get Geo Coordinates?
        # docs_geo_coords = load_topic_geo_locations(topic_docs[0].topic)

        # keyphrase_coordinates = {
        #     candidate: self._get_locations(
        #         docs_geo_coords,
        #         candidate_document_matrix[candidate],
        #     )
        #     for candidate in keyphrase_semantic_score.keys()
        # }

        # Semantic score of per document extracted keyphrases for each document
        # d Documents times k_d Keyphrases, each document can have a different set of extracted keyphrases.
        ranking_p_doc: Dict[Tuple[List[Tuple[str, float]], List[str]]] = {
            doc.id: self.base_model_embed.rank_candidates(
                doc.doc_embed, doc_cand_embeds, doc_cand_set
            )
            for doc, doc_cand_embeds, doc_cand_set in topic_res
        }

        return (
            top_n_scores,
            candidates,
            candidate_document_matrix,
            # keyphrase_coordinates,
            ranking_p_doc,
        )

    def extract_from_topic_original(
        self,
        topic,
        dataset: str = "MKDUC01",
        top_n: int = 15,
        min_len: int = 5,
        stemming: bool = False,
        lemmatize: bool = False,
        **kwargs,
    ) -> List[List[Tuple]]:
        """_summary_

        Parameters
        ----------
        topic : _type_
            _description_
        dataset : str, optional
            _description_, by default "MKDUC01"
        top_n : int, optional
            _description_, by default 15
        min_len : int, optional
            _description_, by default 5
        stemming : bool, optional
            _description_, by default False
        lemmatize : bool, optional
            _description_, by default False

        Returns
        -------
        List[List[Tuple]]
            _description_
        """
        topic_res = [
            self.extract_kp_from_doc(doc, -1, min_len, stemming, lemmatize, **kwargs)
            for doc in topic[0]
        ]
        cands = {}
        for doc in topic_res:
            doc_abs = doc[0]
            cand_embeds = doc[1]
            cand_set = doc[2]

            for i in range(len(cand_set)):
                if cand_set[i] not in cands:
                    cands[cand_set[i]] = []
                cands[cand_set[i]].append(cand_embeds[i])

        cand_embeds = []
        cand_set = []
        for cand in cands:
            cand_embeds.append(np.mean(cands[cand], 0))
            cand_set.append(cand)

        res_p_doc = [
            doc[0].evaluate_n_candidates(cand_embeds, cand_set) for doc in topic_res
        ]
        scores_per_candidate = {}

        for doc in res_p_doc:
            for cand_t in doc[0]:
                if cand_t[0] not in scores_per_candidate:
                    scores_per_candidate[cand_t[0]] = []
                scores_per_candidate[cand_t[0]].append(cand_t[1])

        for cand in scores_per_candidate:
            scores_per_candidate[cand] = mean(scores_per_candidate[cand])

        scores = sorted(
            [(cand, scores_per_candidate[cand]) for cand in scores_per_candidate],
            reverse=True,
            key=lambda x: x[1],
        )
        cand_set = [cand[0] for cand in scores]

        return scores, cand_set

    def extract_kp_from_topic(
        self,
        topic_docs: List[Document],
        top_n: int = 15,
        min_len: int = 5,
        stemming: bool = False,
        lemmatize: bool = False,
        **kwargs,
    ) -> MDKPERankOutput:
        """
        Extract keyphrases from list of documents

        Parameters
        ----------
            topic: List[str], a List of Documents for the topic.

        Returns
        -------
            List KPE extracted from the agregation of documents set, with their score(/embedding?)
        """
        topic_res = [
            self.extract_kp_from_doc(
                doc,
                top_n=top_n,
                min_len=min_len,
                stemming=stemming,
                lemmatize=lemmatize,
                **kwargs,
            )
            for doc in topic_docs
        ]

        documents_embeddings = {}
        candidate_embeddings = {}
        for doc, cand_embeds, cand_set in topic_res:
            documents_embeddings[doc.id] = doc.doc_embed  # .reshape(1, -1)
            # Size([1, 768])

            for candidate, embedding in zip(cand_set, cand_embeds):
                candidate_embeddings.setdefault(candidate, []).append(embedding)

        # The candidate embedding is the average of each embedding
        # of the candidate in the document.
        candidate_embeddings = {
            candidate: np.mean(embeddings, axis=0)
            for candidate, embeddings in candidate_embeddings.items()
        }

        documents_embeddings = pd.DataFrame.from_dict(
            documents_embeddings, orient="index"
        )
        candidate_embeddings = pd.DataFrame.from_dict(
            candidate_embeddings, orient="index"
        )

        top_n_scores = self.ranking_strategy(candidate_embeddings, documents_embeddings)

        # new dataframe with 0
        candidate_document_matrix = pd.DataFrame(
            np.zeros((len(candidate_embeddings), len(documents_embeddings)), dtype=int),
            index=candidate_embeddings.index,
            columns=documents_embeddings.index,
        )
        for doc, _, cand_set in topic_res:
            # Each mention is an observation in the document
            candidate_document_matrix.loc[cand_set, doc.id] += 1

        return MDKPERankOutput(
            top_n_scores,
            # score_per_document,
            candidate_document_matrix,
            documents_embeddings,
            candidate_embeddings,
        )

    def _rank_by_geo(
        self,
        scorer: Callable,
        candidate_scores,
        N,
        candidates,
        geo_association_measure,
        geo_association_candidate_index,
    ) -> List[Tuple[str, float]]:
        """
        For each candidate compute a score based on the semantic score,
        the number of documents the candidate apears in and the geo association measure for each candidate.

        Returns
        -------
            A list with candidates and scores sorted by highest score
        """
        return (
            self._sort_candidate_score(
                [
                    (
                        candidate,
                        scorer(
                            S,
                            N[candidates.index(candidate)],
                            geo_association_measure[
                                geo_association_candidate_index.index(candidate)
                            ],
                        ),
                    )
                    for candidate, S in candidate_scores.items()
                ]
            ),
        )

    def _sort_candidate_score(self, candidates_scores: Iterable):
        """
        Sort {candidate: score} dictionary into list[(candidate, score)]
        in descending order of score (greather first)

        Return
        ------
        List of tuples
        """
        if isinstance(candidates_scores, dict):
            items = candidates_scores.items()
        else:
            items = candidates_scores

        return sorted(
            [(cand, score) for cand, score in items],
            reverse=True,
            key=lambda x: x[1],  # sort by score
        )

    def _score_w_geo_association_I(S, N, I, lambda_=0.5, gamma=0.5):
        return S * lambda_ * (N - (N * gamma * I))

    def _score_w_geo_association_C(S, N, C, lambda_=0.5, gamma=0.5):
        return S * lambda_ * N / (gamma * C)

    def _score_w_geo_association_G(S, N, G, lambda_=0.5, gamma=0.5):
        return S * lambda_ * (N * gamma) * G

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
