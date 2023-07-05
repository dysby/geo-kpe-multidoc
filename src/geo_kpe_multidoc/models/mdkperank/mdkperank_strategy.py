import itertools

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity

# TODO: Condorcet Fuse combines rankings by sorting the documents according to the pairwise relation r(d1) < r(d2),
#       which is determined for each (d1, d2) by majority vote among the input rankings.
# TODO: CombMNZ requires for each r a corresponding scoring function sr : D → R and a cutoff rank c
#       which all contribute to the CombMNZ score: $CMNZscore(d \in D) = |{r \in R|r(d) \leq c}| \sum_{r|r(d) \leq c} S_r (d)$
# TODO: Proposition-Level Clustering for Multi-Document Summarization (https://github.com/oriern/ProCluster)
# TODO: DDP - Improving the Similarity Measure of Determinantal Point Processes for Extractive Multi-Document Summarization (https://github.com/ucfnlp/summarization-dpp-capsnet)
# TODO: Efficiently Summarizing Text and Graph Encodings of Multi-Document Clusters (https://github.com/amazon-research/bartgraphsumm)
#               we do co-reference resolution within each document and extract open information extraction triplets (OIE) at the sentence
#               level from all input documents.2 Each OIE triplet consists of a subject, a predicate, and an object. Once we have all the triplets,
#               in the second step, we build a graph with subjects and objects as nodes and the predicates as the edge relationship between the nodes.
# TODO: Extractive multi-document summarization using multilayer networks
#            modeling a set of documents as a multilayer network. Intra-document and Inter-document similarity
# TODO: Improved Affinity Graph Based Multi-Document Summarization
#            weight Intra-document and Inter-document similarity
# TODO: CoRank - (https://github.com/bshivangi47/Text-Summarization-using-Corank)
# TODO: Extractive multi-document summarization using population-based multicriteria optimization
# TODO: A Preliminary Exploration of Extractive Multi-Document Summarization in Hyperbolic Space
# TODO: Graph (Graph-Based Text Summarization Using Modified TextRank)
#           The graph is made sparse and partitioned into different clusters with the assumption that the
#           sentences within a cluster are similar to each other and sentences of different cluster represent their dissimilarity.


class Ranker:
    def __init__(self, **kwargs) -> None:
        ...

    def _extract_features(self, topic_extraction_features):
        documents_embeddings = {}
        candidate_embeddings = {}

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
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
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
    """Compute the mean of keyphrase to documents similarity"""

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
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
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix,
        *args,
        **kwargs,
    ):
        top_n_scores = None
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
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
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
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
        top_n_scores = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class MaxSumRank(Ranker):
    """From KeyBERT
    Calculate Max Sum Distance for extraction of keywords

    We take the 2 x top_n most similar words/phrases to the document.
    Then, we take all top_n combinations from the 2 x top_n words and extract
    the combination that are the least similar to each other by cosine similarity.

    This is O(n^2) and therefore not advised if you use a large top_n
    """

    def __init__(self, **kwargs) -> None:
        self.top_n = kwargs.get("top_n", 10)
        self.n_candidates = kwargs.get("n_candidates", 30)

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index
        # We take the 2 x top_n most similar words/phrases to the document.

        # Calculate distances and extract keywords
        distances = score_per_document.mean(axis=1).to_numpy().reshape(1, -1)
        distances_words = cosine_similarity(
            candidates_embeddings, candidates_embeddings
        )

        # Get 2*top_n words as candidates based on cosine similarity
        words_idx = list(distances.argsort()[0][-self.n_candidates :])
        # words_vals = [words[index] for index in words_idx]
        words_vals = candidates_embeddings.iloc[words_idx].index
        candidates = distances_words[np.ix_(words_idx, words_idx)]
        # candidates[np.tril_indices_from(candidates)] = 0
        candidates[np.diag_indices_from(candidates)] = 0

        # Calculate the combination of words that are the least similar to each other
        min_sim = 100_000
        candidate = None
        for combination in itertools.combinations(range(len(words_idx)), self.top_n):
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


class DDPRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
        top_n_scores = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class DCSPRank(Ranker):
    """Yunqing Xia, Yi Liu, Yang Zhang, Wenmin Wang "Clustering Sentences with Density Peaks for Multi-document Summarization" (2015)

    adapted from https://github.com/pvgladkov/density-peaks-sentence-clustering

    $$s^{REP}(i) = \frac{1}{K} \sum_{j=1, j \ne i}^K \cal{X}(sim_{ij} - \delta)$$, with

    $ \delta $ - predefined similarity thereshold
    """

    def _represent_score(sim_matrix, delta):
        # sentences_count = len(sim_matrix)
        # result = np.zeros(shape=(sentences_count,))
        # for i in range(sentences_count):
        #     a = (np.array(sim_matrix[i]) - delta) > 0
        #     t = np.ma.array(a * 1.0, mask=False)
        #     t.mask[i] = True
        #     result[i] = np.sum(t) / sentences_count
        #     if not result[i]:
        #         result[i] = 10 ** (-10)
        # return result

        sim_matrix = sim_matrix.copy()
        sim_matrix[np.diag_indices_from(sim_matrix)] = 0

        sentences_count = len(sim_matrix)
        is_represent = (sim_matrix - delta) > 0
        score = np.sum(is_represent, axis=1) / sentences_count
        return score

    def _diversity_score(sim_matrix, s_rep_vector):
        """
        Diversity score of a sentence is measured by computing the minimum distance
        between the sentence $s_i$ and any other sentences with higher density score
        """
        sentences_count = len(sim_matrix)
        result = np.zeros(shape=(sentences_count,))
        for i in range(sentences_count):
            mask = (s_rep_vector > s_rep_vector[i]) * 1
            if np.any(mask):
                result[i] = 1 - np.max(sim_matrix[i] * mask)
            else:
                t = np.ma.array(sim_matrix[i], mask=False)
                t.mask[i] = True
                result[i] = 1 - np.min(t)
            if not result[i]:
                result[i] = 10 ** (-10)
        return result

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
        sim_matrix = cosine_similarity(candidates_embeddings, candidates_embeddings)
        # effective_lens = [effective_length(s) for s in split_into_sentences(text)]
        # real_lens = [real_length(s) for s in split_into_sentences(text)]

        rep_score = np.log(self._represent_score(sim_matrix, 0.22))
        div_score = np.log(self._diversity_score(sim_matrix, rep_score))
        score = rep_score + div_score
        # len_score = np.log(length_score_(effective_lens, real_lens))
        # score = rep_score + div_score + len_score
        top_n_scores = [
            (sent, s) for sent, s in zip(candidate_document_matrix.index, score)
        ]
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class ClusterCentroidsRank(Ranker):
    """Based on:
    https://github.com/caomanhhaipt/Extractive-Multi-document-Summarization/blob/master/methods/main_method/Kmeans_CentroidBase_MMR_SentencePosition.py
    """

    def __init__(self, **kwargs) -> None:
        self.n_clusters = kwargs.get("n_clusters", 20)

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
        # 1: compute 20 clusters
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans = kmeans.fit(candidates_embeddings)

        avg = []
        for j in range(self.n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        # 2: get keyphrase embedding closest to cluster centroid
        closest, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, candidates_embeddings
        )
        ordering = sorted(range(self.n_clusters), key=lambda k: avg[k])

        top_n_scores = [(candidates_embeddings.index[closest[i]], i) for i in ordering]

        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class ComunityRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
        # 1: Get network graph based?
        # 2: Comunities and assign a keyphrase from largest comunities
        top_n_scores = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
        )


class EigenRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
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
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        *args,
        **kwargs,
    ):
        """based on https://github.com/bshivangi47/Text-Summarization-using-Corank/blob/main/for100files.py"""
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )

        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        adjacency_matrix = cosine_similarity(
            candidates_embeddings, candidates_embeddings
        )

        # Matrix_B= row normalisation of bag of words model
        # matrix_B = np.array([[0 for x in range(len(bag_of_words[0]))] for y in range(len(bag_of_words))])
        matrix_B = candidate_document_matrix / np.linalg.norm(
            candidate_document_matrix, axis=1, keepdims=True
        )

        sentence_score_t_1 = np.random.rand(len(score_per_document.index))
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

        top_n_scores = [
            (sent, s)
            for sent, s in zip(candidate_document_matrix.index, sentence_score_t_1)
        ]
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
    "MAXSUM": MaxSumRank,
    "CLUSTERCENTROIDS": ClusterCentroidsRank,
    "COMUNITY": ComunityRank,
    "EIGEN": EigenRank,
    "PAGERANK": PageRank,
    "DCSP": DCSPRank,
}
