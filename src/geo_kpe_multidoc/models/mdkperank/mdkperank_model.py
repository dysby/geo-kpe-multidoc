from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from statistics import mean
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from nltk.stem.api import StemmerI

from geo_kpe_multidoc.datasets.datasets import KPEDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.mdkperank.mdkperank_strategy import STRATEGIES

from ..base_KP_model import BaseKPModel, KPEScore
from ..embedrank import EmbedRank


@dataclass
class MdKPEOutput:
    top_n_scores: List = None
    candidates: List[str] = None
    candidate_document_matrix: pd.DataFrame = None
    documents_embeddings: pd.DataFrame = None
    candidate_embeddings: pd.DataFrame = None
    ranking_p_doc: Dict[str, Tuple[List[Tuple[str, float]], List[str]]] = None


class MDKPERank(BaseKPModel):
    def __init__(self, model: EmbedRank, rank_strategy: str = "MEAN", **kwargs):
        self.base_model_embed: EmbedRank = model
        # TODO: what how to join MaskRank
        # self.base_model_mask = MaskRank(model, tagger)
        self.name = (
            ".".join([self.__class__.__name__, model.name.split("_")[0]])
            + model.name[model.name.index("_") :]
        )

        self.ranking_strategy = STRATEGIES[rank_strategy](**kwargs)

    def _extract_doc_candidate_embeddings(
        self,
        doc,
        top_n: int = -1,
        min_len: int = 0,
        lemmer=None,
        **kwargs,
    ):
        self.base_model_embed.extract_candidates(doc, min_len, lemmer, **kwargs)

        _, cand_embeds, candidate_set = self.base_model_embed.embed_candidates(
            doc,
            **kwargs,
        )

        return (doc, cand_embeds, candidate_set)

    def extract_kp_from_doc(
        self, doc, top_n, min_len, lemmer=None, **kwargs
    ) -> Tuple[Document, List[np.ndarray], List[str]]:  # Tuple[List[Tuple], List[str]]:
        """
        Extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """

        self.base_model_embed.extract_candidates(doc, min_len, lemmer, **kwargs)

        _, cand_embeds, candidate_set = self.base_model_embed.embed_candidates(
            doc,
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
        top_n: int = -1,
        min_len: int = 0,
        stemming: bool = False,
        lemmatize: bool = False,
        **kwargs,
    ) -> MdKPEOutput:
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
        ranking_p_doc: Dict[str, Tuple[List[Tuple[str, float]], List[str]]] = {
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
        ranking_p_doc: Dict[str, Tuple[List[Tuple[str, float]], List[str]]] = {
            doc.id: self.base_model_embed.rank_candidates(
                doc.doc_embed, doc_cand_embeds, doc_cand_set
            )
            for doc, doc_cand_embeds, doc_cand_set in topic_res
        }

        return MdKPEOutput(
            top_n_scores=top_n_scores,
            candidates=candidates,
            candidate_document_matrix=candidate_document_matrix,
            ranking_p_doc=ranking_p_doc,
        )

    def extract_kp_from_topic(
        self,
        topic_docs: List[Document],
        top_n: int = -1,
        min_len: int = 0,
        lemmatize: bool = False,
        **kwargs,
    ) -> MdKPEOutput:
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
                lemmatize=lemmatize,
                **kwargs,
            )
            for doc in topic_docs
        ]

        # if self.whitening:
        #     documents_embeddings = {}
        #     candidate_embeddings = {}

        #     for doc, cand_embeds, cand_set in topic_res:

        #         documents_embeddings[doc.id] = doc.doc_embed  # .reshape(1, -1)
        #         # Size([1, 768])
        #         for candidate, embedding in zip(cand_set, cand_embeds):
        #             candidate_embeddings.setdefault(candidate, []).append(embedding)

        #     candidate_embeddings = {
        #         candidate: np.mean(embeddings, axis=0)
        #         for candidate, embeddings in candidate_embeddings.items()
        #     }

        (
            documents_embeddings,
            candidate_embeddings,
            candidate_document_matrix,
            top_n_scores,
        ) = self.ranking_strategy(topic_res)

        return MdKPEOutput(
            top_n_scores=top_n_scores,
            candidate_document_matrix=candidate_document_matrix,
            documents_embeddings=documents_embeddings,
            candidate_embeddings=candidate_embeddings,
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
            key=itemgetter(1),  # sort by score
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
        top_n: int = -1,
        min_len: int = 0,
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
            self.extract_kp_from_topic(sample, top_n, min_len, lemmatize, **kwargs)
            for _id, sample, _label in corpus
        ]

        return res
