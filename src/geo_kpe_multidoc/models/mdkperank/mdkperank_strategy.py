import itertools

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# TODO: Condorcet Fuse combines rankings by sorting the documents according to the pairwise relation r(d1) < r(d2),
#       which is determined for each (d1, d2) by majority vote among the input rankings.
# TODO: CombMNZ requires for each r a corresponding scoring function sr : D → R and a cutoff rank c
#       which all contribute to the CombMNZ score: $CMNZscore(d \in D) = |{r \in R|r(d) \leq c}| \sum_{r|r(d) \leq c} S_r (d)$


class Ranker:
    def _extract_features(self, topic_extraction_features):
        documents_embeddings = {}
        candidate_embeddings = {}
        # candidate_document_embeddings = {}
        for doc, cand_embeds, cand_set in topic_extraction_features:
            documents_embeddings[doc.id] = doc.doc_embed  # .reshape(1, -1)
            # Size([1, 768])

            for candidate, embedding in zip(cand_set, cand_embeds):
                candidate_embeddings.setdefault(candidate, []).append(embedding)

        #    candidate_document_embeddings[doc.id] = candidate_embeddings

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

        # new dataframe with 0
        candidate_document_matrix = pd.DataFrame(
            np.zeros((len(candidate_embeddings), len(documents_embeddings)), dtype=int),
            index=candidate_embeddings.index,
            columns=documents_embeddings.index,
        )
        for doc, _, cand_set in topic_extraction_features:
            # Each mention is an observation in the document
            candidate_document_matrix.loc[cand_set, doc.id] += 1

        return documents_embeddings, candidate_embeddings, candidate_document_matrix

    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def __call__(self, topic_extraction_features) -> pd.DataFrame:
        (
            documents_embeddings,
            candidate_embeddings,
            candidate_document_matrix,
        ) = self._extract_features(topic_extraction_features)
        return self._rank(
            candidate_embeddings, documents_embeddings, candidate_document_matrix
        )


class MeanRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = score_per_document.mean(axis=1).sort_values(ascending=False)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class ITCSRank(Ranker):
    # Modified inverse sentence frequency-cosine similarity is used to give different weigthage to different keyphrases
    # based on "Graph-Based Text Summarization Using Modified TextRank"
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
    ) -> pd.DataFrame:
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = score_per_document.mean(axis=1).sort_values(ascending=False)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class MrrRank(Ranker):
    """Mean Reciprocal Rank"""

    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
    ) -> pd.DataFrame:
        # TODO: add k parameter from original paper default k=60
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )

        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = (
            (1 / score_per_document.rank()).mean(axis=1).sort_values(ascending=True)
        )
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class MmrRank(Ranker):
    # TODO: Maximal Relevance in Multidoc???
    # Based on KeyBERT implementation
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
    ) -> pd.DataFrame:
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = score_per_document.mean(axis=1).sort_values(ascending=False)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class Top20MaxSum(Ranker):
    """From KeyBERT
    Calculate Max Sum Distance for extraction of keywords

    We take the 2 x top_n most similar words/phrases to the document.
    Then, we take all top_n combinations from the 2 x top_n words and extract
    the combination that are the least similar to each other by cosine similarity.

    This is O(n^2) and therefore not advised if you use a large top_n
    """

    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
    ) -> pd.DataFrame:
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index
        top_n_scores = score_per_document.mean(axis=1).sort_values(ascending=False)

        # We take the 2 x top_n most similar words/phrases to the document.

        TOP_N = 100

        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index
        # top_n_scores = score_per_document.mean(axis=1).sort_values(ascending=False)

        # We take the 2 x top_n most similar words/phrases to the document.

        # We take the 2 x top_n most similar words/phrases to the document.

        top_n = 20
        nr_candidates = 40

        # Calculate distances and extract keywords
        distances = score_per_document.mean(axis=1).to_numpy().reshape(1, -1)
        distances_words = cosine_similarity(
            candidates_embeddings, candidates_embeddings
        )

        # Get 2*top_n words as candidates based on cosine similarity
        words_idx = list(distances.argsort()[0][-nr_candidates:])
        # words_vals = [words[index] for index in words_idx]
        words_vals = candidates_embeddings.iloc[words_idx].index
        candidates = distances_words[np.ix_(words_idx, words_idx)]
        # candidates[np.tril_indices_from(candidates)] = 0
        candidates[np.diag_indices_from(candidates)] = 0

        # Calculate the combination of words that are the least similar to each other
        min_sim = 100_000
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), top_n):
            # sim = sum(
            #     [candidates[i][j] for i in combination for j in combination if i != j]
            # )
            # refactor
            # index = np.array(list(zip(*itertools.combinations(combination, 2)))).T
            # index = np.array(list(zip(*itertools.product(combination, combination)))).T
            # numpy >= 1.24
            index = np.fromiter(
                itertools.product(combination, combination), dtype=(int, 2)
            )
            sim = np.sum(candidates[index[:, 0], index[:, 1]])
            if sim < min_sim:
                candidate = combination
                min_sim = sim

        top_n_scores = [
            (words_vals[idx], round(float(distances[0][words_idx[idx]]), 4))
            for idx in candidate
        ]

        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class Top20ClusterCentroids(Ranker):
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
    ) -> pd.DataFrame:
        # 1: compute 20 clusters
        # 2: get keyphrase embedding closest to cluster centroid
        top_n_scores = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class ComunityRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
    ) -> pd.DataFrame:
        # 1: Get network graph based?
        # 2: Comunities and assign a keyphrase from largest comunities
        top_n_scores = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


# TODO: Graph (Graph-Based Text Summarization Using Modified TextRank)
#           The graph is made sparse and partitioned into different clusters with the assumption that the
#           sentences within a cluster are similar to each other and sentences of different cluster represent their dissimilarity.


class EigenRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
    ) -> pd.DataFrame:
        # 1: Network graph
        # 2: EigenVector Centrality for the network and get
        top_n_scores = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class PageRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: np.ndarray,
        documents_embeddings: np.ndarray,
        candidate_document_matrix,
    ) -> pd.DataFrame:
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )

        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        sentence_score_t_1 = np.random.rand(len(corpus))
        # sentence_score_t_1 = np.zeros(np.asarray(corpus).shape, dtype=float, order='C')
        alpha = 0.6
        epsilon = 0.00001
        first_term = np.multiply(alpha, adjacency_matrix)
        B_transpose = np.transpose(matrix_B)
        temp = np.matmul(matrix_B, B_transpose)
        second_term = np.multiply(1 - alpha, temp)
        constant_matrix = np.add(first_term, second_term)
        while True:
            sentence_score_t = constant_matrix.dot(sentence_score_t_1)
            condition = np.linalg.norm(
                np.subtract(sentence_score_t, sentence_score_t_1)
            )
            if condition * condition > epsilon:
                break
            else:
                sentence_score_t_1 = sentence_score_t

        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


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
