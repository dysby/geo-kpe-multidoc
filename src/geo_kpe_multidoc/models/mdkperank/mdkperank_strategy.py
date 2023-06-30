from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# TODO: Condorcet Fuse combines rankings by sorting the documents according to the pairwise relation r(d1) < r(d2),
#       which is determined for each (d1, d2) by majority vote among the input rankings.
# TODO: CombMNZ requires for each r a corresponding scoring function sr : D → R and a cutoff rank c
#       which all contribute to the CombMNZ score: $CMNZscore(d \in D) = |{r \in R|r(d) \leq c}| \sum_{r|r(d) \leq c} S_r (d)$


class Ranker:
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        raise NotImplementedError

    def __call__(
        self, candidates_embeddings: np.ndarray, documents_embeddings: np.ndarray
    ) -> pd.DataFrame:
        return self._rank(candidates_embeddings, documents_embeddings)


class MeanRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        *args,
        **kwargs
    ) -> pd.DataFrame:
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = score_per_document.mean(axis=1).sort_values(ascending=False)
        return top_n_scores


class ITCSRank(Ranker):
    # Modified inverse sentence frequency-cosine similarity is used to give different weightage to different keyphrases
    # based on "Graph-Based Text Summarization Using Modified TextRank"
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_in_document_matrix,
    ) -> pd.DataFrame:
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = score_per_document.mean(axis=1).sort_values(ascending=False)
        return top_n_scores


class MrrRank(Ranker):
    def _rank(
        self, candidates_embeddings: np.ndarray, documents_embeddings: np.ndarray
    ) -> pd.DataFrame:
        # TODO: add k parameter from original paper default k=60
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )

        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = (
            (1 / score_per_document.rank()).mean(axis=1).sort_values(ascending=False)
        )
        return top_n_scores


class MmrRank(Ranker):
    # TODO: Maximal Relevance in Multidoc???
    # Based on KeyBERT implementation
    def _rank(
        self, candidates_embeddings: np.ndarray, documents_embeddings: np.ndarray
    ) -> pd.DataFrame:
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = score_per_document.mean(axis=1).sort_values(ascending=False)
        return top_n_scores


class Top20MaxSum(Ranker):
    """From KeyBERT
    Calculate Max Sum Distance for extraction of keywords

    We take the 2 x top_n most similar words/phrases to the document.
    Then, we take all top_n combinations from the 2 x top_n words and extract
    the combination that are the least similar to each other by cosine similarity.

    This is O(n^2) and therefore not advised if you use a large top_n
    """

    def _rank(
        self, candidates_embeddings: np.ndarray, documents_embeddings: np.ndarray
    ) -> pd.DataFrame:
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = score_per_document.mean(axis=1).sort_values(ascending=False)
        return top_n_scores


class Top20ClusterCentroids(Ranker):
    def _rank(
        self, candidates_embeddings: np.ndarray, documents_embeddings: np.ndarray
    ) -> pd.DataFrame:
        # 1: compute 20 clusters
        # 2: get keyphrase embedding closest to cluster centroid
        top_n_scores = None
        return top_n_scores


class ComunityRank(Ranker):
    def _rank(
        self, candidates_embeddings: np.ndarray, documents_embeddings: np.ndarray
    ) -> pd.DataFrame:
        # 1: Get network graph based?
        # 2: Comunities and assign a keyphrase from largest comunities
        top_n_scores = None
        return top_n_scores


# TODO: Graph (Graph-Based Text Summarization Using Modified TextRank)
#           The graph is made sparse and partitioned into different clusters with the assumption that the
#           sentences within a cluster are similar to each other and sentences of different cluster represent their dissimilarity.


class EigenRank(Ranker):
    def _rank(
        self, candidates_embeddings: np.ndarray, documents_embeddings: np.ndarray
    ) -> pd.DataFrame:
        # 1: Network graph
        # 2: EigenVector Centrality for the network and get
        top_n_scores = None
        return top_n_scores


STRATEGIES = {
    "MEAN": MeanRank,
    "MMR": MmrRank,
    "MRR": MrrRank,
    "ITCS": ITCSRank,
    "MAXSUM": Top20MaxSum,
    "CLUSTERCENTROIDS": Top20ClusterCentroids,
    "COMUNITY": ComunityRank,
    "EIGEN": EigenRank,
}
