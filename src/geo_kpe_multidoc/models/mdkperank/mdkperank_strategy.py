import itertools
from functools import partial
from operator import itemgetter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch.nn.functional as F
import umap
from keybert._mmr import mmr
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize

# TODO: Condorcet Fuse combines rankings by sorting the documents according to the
#       pairwise relation r(d1) < r(d2), which is determined for each (d1, d2)
#       by majority vote among the input rankings.
# TODO: CombMNZ requires for each r a corresponding scoring function sr : D → R and a
#       cutoff rank c which all contribute to the CombMNZ score:
#       $CMNZscore(d \in D) = |{r \in R|r(d) \leq c}| \sum_{r|r(d) \leq c} S_r (d)$
# TODO: Proposition-Level Clustering for Multi-Document Summarization (https://github.com/oriern/ProCluster)
# TODO: DDP - Improving the Similarity Measure of Determinantal Point Processes for
#       Extractive Multi-Document Summarization (https://github.com/ucfnlp/summarization-dpp-capsnet)
# TODO: Efficiently Summarizing Text and Graph Encodings of Multi-Document Clusters
#       (https://github.com/amazon-research/bartgraphsumm) we do co-reference
#       resolution within each document and extract open information extraction
#       triplets (OIE) at the sentence level from all input documents.2 Each OIE
#       triplet consists of a subject, a predicate, and an object. Once we have all the
#       triplets, in the second step, we build a graph with subjects and objects as
#       nodes and the predicates as the edge relationship between the nodes.
# TODO: Extractive multi-document summarization using multilayer networks modeling a
#       set of documents as a multilayer network. Intra-document and Inter-document
#       similarity.
# TODO: Improved Affinity Graph Based Multi-Document Summarization weight
#       Intra-document and Inter-document similarity
# TODO: CoRank - (https://github.com/bshivangi47/Text-Summarization-using-Corank)
# TODO: Extractive multi-document summarization using population-based multicriteria
#       optimization
# TODO: A Preliminary Exploration of Extractive Multi-Document Summarization in
#       Hyperbolic Space
# TODO: Graph (Graph-Based Text Summarization Using Modified TextRank)
#       The graph is made sparse and partitioned into different clusters with
#       the assumption that the sentences within a cluster are similar to each other
#       and sentences of different cluster represent their dissimilarity.


class Ranker:
    def __init__(self, **kwargs) -> None:
        ...

    def _extract_features(
        self, topic_extraction_features
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Params
        ------
        topic_extraction_features: List[(doc, candidate_embedings, candidate_list, ranking_in_doc), ...]
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            documents_embeddings, candidate_embeddings, candidate_document_matrix_df
        """
        documents_embeddings = {}
        candidate_embeddings = {}
        single_mode_ranking_per_doc = {}

        for doc, cand_embeds, cand_set, ranking_in_doc in topic_extraction_features:
            documents_embeddings[doc.id] = doc.doc_embed  # .reshape(1, -1)
            single_mode_ranking_per_doc[doc.id] = ranking_in_doc  # .reshape(1, -1)
            # Size([1, 768])
            # Not all Single Document Methods compute a candidate_embedding (e.g. PromptRank)
            for candidate, embedding in itertools.zip_longest(cand_set, cand_embeds):
                candidate_embeddings.setdefault(candidate, []).append(embedding)

        #    candidate_document_embeddings[doc.id] = candidate_embeddings

        # The candidate embedding is the average of each embedding
        # of the candidate in the document.
        # For PromptRank candidate_embedding is None
        candidate_embeddings = {
            candidate: np.mean(embeddings, axis=0)
            if isinstance(embeddings[0], np.ndarray)
            else None
            for candidate, embeddings in candidate_embeddings.items()
        }
        # For PromptRank document_embedding is also None
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
        for doc, _, cand_set, _ in topic_extraction_features:
            # Each mention is an observation in the document
            candidate_document_matrix.loc[cand_set, doc.id] += 1

        return (
            documents_embeddings,
            candidate_embeddings,
            candidate_document_matrix,
            single_mode_ranking_per_doc,
        )

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def __call__(
        self, topic_extraction_features, candidate_document_matrix=None, **kwargs
    ) -> pd.DataFrame:
        if kwargs.get("extract_features", True):
            (
                documents_embeddings,
                candidate_embeddings,
                calc_candidate_document_matrix,
                single_mode_ranking_per_doc,
            ) = self._extract_features(topic_extraction_features)
        else:
            (
                documents_embeddings,
                candidate_embeddings,
                calc_candidate_document_matrix,
                single_mode_ranking_per_doc,
            ) = topic_extraction_features

        candidate_document_matrix = (
            calc_candidate_document_matrix
            if candidate_document_matrix is None
            else candidate_document_matrix
        )

        return self._rank(
            candidate_embeddings,
            documents_embeddings,
            candidate_document_matrix,
            single_mode_ranking_per_doc,
        )

    def _score_to_ranking_p_doc(
        self, score_per_document: pd.DataFrame
    ) -> Dict[str, Tuple[List[Tuple[str, float]], List[str]]]:
        # ranking_p_doc: Dict[str, Tuple[List[Tuple[str, float]], List[str]]]
        ranking_p_doc = {
            doc_id: [
                [(candidate, score) for candidate, score in candidate_scores.items()],
                list(candidate_scores.keys()),
            ]
            for doc_id, candidate_scores in score_per_document.transpose()
            .to_dict(orient="index")
            .items()
        }
        return ranking_p_doc


class MeanRank(Ranker):
    """Compute the mean of keyphrase to documents similarity"""

    def __init__(self, **kwargs) -> None:
        self.in_single_mode = kwargs.get("in_single_mode")

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        if self.in_single_mode:
            score_per_document = (
                pd.DataFrame.from_records(
                    [
                        (doc, cand, score)
                        for doc, scores in single_mode_ranking_per_doc.items()
                        for cand, score in scores
                    ],
                    columns=["doc", "candidate", "score"],
                )
                .set_index("candidate")
                .pivot(columns="doc", values="score")
            )
        else:
            score_per_document = pd.DataFrame(
                cosine_similarity(candidates_embeddings, documents_embeddings)
            )
            score_per_document.index = candidates_embeddings.index
            score_per_document.columns = documents_embeddings.index

        top_n_scores = list(
            score_per_document.mean(axis=1).sort_values(ascending=False).items()
        )

        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)

        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class MaxRank(Ranker):
    """Each Candidate score is the maximum similarity to a single document in the multidocument set."""

    def __init__(self, **kwargs) -> None:
        self.in_single_mode = kwargs.get("in_single_mode")

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        if self.in_single_mode:
            score_per_document = (
                pd.DataFrame.from_records(
                    [
                        (doc, cand, score)
                        for doc, scores in single_mode_ranking_per_doc.items()
                        for cand, score in scores
                    ],
                    columns=["doc", "candidate", "score"],
                )
                .set_index("candidate")
                .pivot(columns="doc", values="score")
            )
        else:
            score_per_document = pd.DataFrame(
                cosine_similarity(candidates_embeddings, documents_embeddings)
            )
            score_per_document.index = candidates_embeddings.index
            score_per_document.columns = documents_embeddings.index

        top_n_scores = list(
            score_per_document.max(axis=1).sort_values(ascending=False).items()
        )

        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class MeanTopMaxRank(Ranker):
    """Each Candidate score is the mean of top 50% similarity scores in the
    multidocument set. This strategy is a mix of MaxRank and MeanRank.
    """

    def __init__(self, **kwargs) -> None:
        self.in_single_mode = kwargs.get("in_single_mode")

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        # Function to compute the mean of top 50% values for a row
        def _mean_top_50_percent(row, top_50_percent_threshold):
            threshold = top_50_percent_threshold.loc[row.name]
            top_50_percent_values = row[row >= threshold]
            return top_50_percent_values.mean()

        if self.in_single_mode:
            score_per_document = (
                pd.DataFrame.from_records(
                    [
                        (doc, cand, score)
                        for doc, scores in single_mode_ranking_per_doc.items()
                        for cand, score in scores
                    ],
                    columns=["doc", "candidate", "score"],
                )
                .set_index("candidate")
                .pivot(columns="doc", values="score")
            )
        else:
            score_per_document = pd.DataFrame(
                cosine_similarity(candidates_embeddings, documents_embeddings)
            )
            score_per_document.index = candidates_embeddings.index
            score_per_document.columns = documents_embeddings.index

        # Calculate the top 50% threshold value for each row
        top_50_percent_threshold = score_per_document.apply(
            lambda row: row.quantile(0.5), axis=1
        )
        aggregation = partial(
            _mean_top_50_percent, top_50_percent_threshold=top_50_percent_threshold
        )
        # Apply the function to each row
        mean_top_50p = score_per_document.apply(aggregation, axis=1)

        top_n_scores = list(mean_top_50p.sort_values(ascending=False).items())

        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class MaxMaxRank(Ranker):
    """Compute an alternative candidate embedding representation by Max pooling the
    candidate contextualized embedding from each document. Set the candidate socre as
    the maximum similarity to the  mean of keyphrase to documents similarity.
    # TODO: Not compatible with PromptRank, need change extract_features
    """

    def _extract_features(self, topic_extraction_features):
        documents_embeddings = {}
        candidate_embeddings = {}
        single_mode_ranking_per_doc = {}
        for (
            doc,
            cand_embeds,
            cand_set,
            single_mode_ranking_per_doc,
        ) in topic_extraction_features:
            documents_embeddings[doc.id] = doc.doc_embed  # .reshape(1, -1)
            single_mode_ranking_per_doc[doc.id] = single_mode_ranking_per_doc
            # cand_embeds = cand_embeds if cand_embeds else []
            for candidate, embedding in zip(cand_set, cand_embeds):
                candidate_embeddings.setdefault(candidate, []).append(embedding)

        # The candidate embedding is the average of each embedding
        # of the candidate in the document.
        candidate_embeddings = {
            candidate: np.max(embeddings, axis=0)
            for candidate, embeddings in candidate_embeddings.items()
        }

        documents_embeddings = pd.DataFrame.from_dict(
            documents_embeddings, orient="index"
        )
        candidate_embeddings = pd.DataFrame.from_dict(
            candidate_embeddings, orient="index"
        )

        candidate_document_matrix = pd.DataFrame(
            np.zeros((len(candidate_embeddings), len(documents_embeddings)), dtype=int),
            index=candidate_embeddings.index,
            columns=documents_embeddings.index,
        )
        for doc, _, cand_set, _ in topic_extraction_features:
            # Each mention is an observation in the document
            candidate_document_matrix.loc[cand_set, doc.id] += 1

        return (
            documents_embeddings,
            candidate_embeddings,
            candidate_document_matrix,
            single_mode_ranking_per_doc,
        )

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        top_n_scores = list(
            score_per_document.max(axis=1).sort_values(ascending=False).items()
        )

        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class ITCSRank(Ranker):
    # Modified inverse sentence frequency-cosine similarity is used to give different weigthage to different keyphrases
    # based on "Graph-Based Text Summarization Using Modified TextRank"
    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        top_n_scores = None
        ranking_p_doc = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class RRFRank(Ranker):
    """Reciprocal Rank Fusion"""

    def __init__(self, **kwargs) -> None:
        self.in_single_mode = kwargs.get("in_single_mode")
        self.k = 60

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        if self.in_single_mode:
            score_per_document = (
                pd.DataFrame.from_records(
                    [
                        (doc, cand, score)
                        for doc, scores in single_mode_ranking_per_doc.items()
                        for cand, score in scores
                    ],
                    columns=["doc", "candidate", "score"],
                )
                .set_index("candidate")
                .pivot(columns="doc", values="score")
            )
        else:
            score_per_document = pd.DataFrame(
                cosine_similarity(candidates_embeddings, documents_embeddings)
            )
            score_per_document.index = candidates_embeddings.index
            score_per_document.columns = documents_embeddings.index

        # top_n_scores = list(
        #     (1 / score_per_document.rank())
        #     .mean(axis=1)
        #     .sort_values(ascending=True)
        #     .items()
        # )
        # TODO: add k parameter from original paper default k=60
        top_n_scores = list(
            (
                1
                / (
                    self.k
                    + score_per_document.rank(ascending=False, na_option="bottom")
                )
            )
            .sum(axis=1)
            .sort_values(ascending=False)
            .items()
        )
        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class MmrRank(Ranker):
    # TODO: Maximal Marginal Relevance in Multidoc (query is mean of document embeddings)
    # Based on KeyBERT implementation

    def __init__(self, **kwargs) -> None:
        # TODO: Not compatible with 0 diversity
        if kwargs.get("mmr_diversity") == 0:
            self.diversity = 0
        else:
            self.diversity = kwargs.get("mmr_diversity") or 0.5

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        # document centroid
        documents_centroid = documents_embeddings.mean(axis=0).to_numpy().reshape(1, -1)

        top_n_scores = mmr(
            documents_centroid,
            candidates_embeddings,
            candidates_embeddings.index,
            top_n=len(candidates_embeddings),
            diversity=self.diversity,
        )

        # keep score per document for geo association metrics
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index
        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class MMRPos(Ranker):
    # TODO: Semantice Score + candidate diversity, based on "Unsupervised extractive
    #   multi-document summarization method based on transfer learning from BERT
    #   multi-task fine-tuning"

    def __init__(self, **kwargs) -> None:
        if kwargs.get("mmr_diversity") == 0:
            self.diversity = 0
        else:
            self.diversity = kwargs.get("mmr_diversity") or 0.5

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        score_per_document = (
            pd.DataFrame.from_records(
                [
                    (doc, cand, score)
                    for doc, scores in single_mode_ranking_per_doc.items()
                    for cand, score in scores
                ],
                columns=["doc", "candidate", "score"],
            )
            .set_index("candidate")
            .pivot(columns="doc", values="score")
        )

        words = candidates_embeddings.index
        top_n = len(words)
        word_doc_similarity = (
            np.exp(score_per_document)
            .max(axis=1)
            .loc[candidates_embeddings.index]
            .to_numpy()
            .reshape(-1, 1)
        )

        # Extract similarity within words, and between words and the document
        # word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
        word_similarity = cosine_similarity(candidates_embeddings)

        # Initialize candidates and already choose best keyword/keyphras
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        for _ in range(min(top_n - 1, len(words) - 1)):
            # Extract similarities within candidates and
            # between candidates and selected keywords/phrases
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(
                word_similarity[candidates_idx][:, keywords_idx], axis=1
            )

            # Calculate MMR
            mmr_ = (
                1 - self.diversity
            ) * candidate_similarities - self.diversity * target_similarities.reshape(
                -1, 1
            )
            mmr_idx = candidates_idx[np.argmax(mmr_)]

            # Update keywords & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        # Extract and sort keywords in descending similarity
        keywords = [
            (words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4))
            for idx in keywords_idx
        ]
        top_n_scores = sorted(keywords, key=itemgetter(1), reverse=True)

        # keep score per document for geo association metrics

        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
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
        single_mode_ranking_per_doc: dict,
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

        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class DDPRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        top_n_scores = None
        ranking_p_doc = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class DPSCRank(Ranker):
    """Yunqing Xia, Yi Liu, Yang Zhang, Wenmin Wang "Clustering Sentences with Density Peaks for Multi-document Summarization" (2015)

    adapted from https://github.com/pvgladkov/density-peaks-sentence-clustering

    $$s^{REP}(i) = \frac{1}{K} \sum_{j=1, j \ne i}^K \cal{X}(sim_{ij} - \delta)$$, with

    $ \delta $ - predefined similarity thereshold
    """

    def _represent_score(self, sim_matrix, delta):
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
        sentences_count = len(sim_matrix)
        is_represent = (sim_matrix - delta) > 0
        score = np.sum(is_represent, axis=1) / sentences_count
        return score

    def _diversity_score(self, sim_matrix, s_rep_vector):
        """
        Diversity score of a sentence is measured by computing the minimum distance
        between the sentence $s_i$ and any other sentences with higher density score
        """

        # sentences_count = len(sim_matrix)
        # result = np.zeros(shape=(sentences_count,))
        # for i in range(sentences_count):
        #     result[i] = np.min(sim_matrix[i][s_rep_vector > s_rep_vector[i]])

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
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        sim_matrix = cosine_similarity(candidates_embeddings, candidates_embeddings)
        score_per_document = pd.DataFrame(sim_matrix)
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index
        sim_matrix[np.diag_indices_from(sim_matrix)] = 0
        # effective_lens = [effective_length(s) for s in split_into_sentences(text)]
        # real_lens = [real_length(s) for s in split_into_sentences(text)]

        rep_score = np.log(self._represent_score(sim_matrix, 0.22))
        div_score = np.log(self._diversity_score(sim_matrix, rep_score))
        score = rep_score + div_score
        # len_score = np.log(length_score_(effective_lens, real_lens))
        # score = rep_score + div_score + len_score

        # TODO: check why sorting backwards (reverse=False) give better results
        top_n_scores = sorted(
            [(sent, s) for sent, s in zip(candidate_document_matrix.index, score)],
            key=itemgetter(1),
            reverse=True,
        )

        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class UmapClustersRank(Ranker):
    def __init__(self, **kwargs) -> None:
        self.min = 5

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        candidates_embeddings_l2 = F.normalize(candidates_embeddings.to_numpy()).numpy()

        clusterable_embedding = umap.UMAP(
            # n_neighbors=10,
            # min_dist=0.0,
            # n_components=2,
            # random_state=42,
            # metric="cosine",
            n_components=2,
        ).fit_transform(candidates_embeddings_l2)

        fitted = HDBSCAN(
            min_samples=8,
            min_cluster_size=2,
            store_centers="medoid",
        ).fit(clusterable_embedding)

        avg = []

        # n_clusters = len(fitted.labels_)
        n_clusters = len(fitted.medoids_)

        for j in range(n_clusters):
            idx = np.where(fitted.labels_ == j)[0]
            avg.append(np.mean(idx))
        # 2: get keyphrase embedding closest to cluster centroid
        closest, _ = pairwise_distances_argmin_min(
            # fitted.cluster_centers_, candidates_embeddings, metric="cosine"
            fitted.medoids_,
            clusterable_embedding,
        )

        scores = cosine_similarity(
            candidates_embeddings.iloc[closest],
            documents_embeddings.mean().to_numpy().reshape(1, -1),
        )
        top_n_scores = sorted(
            [
                (candidate, score)
                for candidate, score in zip(
                    candidates_embeddings.iloc[closest].index, scores
                )
            ],
            key=itemgetter(1),
            reverse=True,
        )

        # keep score per document for geo association metrics
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index
        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)

        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class ClusterCentroidsRank(Ranker):
    """Based on:
    https://github.com/caomanhhaipt/Extractive-Multi-document-Summarization/blob/master/methods/main_method/Kmeans_CentroidBase_MMR_SentencePosition.py
    """

    def __init__(self, **kwargs) -> None:
        # self.clustering = KMeans(**kwargs)
        # self.clustering = DBSCAN(**kwargs)
        self.clustering = HDBSCAN(store_centers="medoid")

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        # 1: compute 20 clusters
        # sim_matrix = cosine_similarity(candidates_embeddings, candidates_embeddings)
        # sim_matrix[sim_matrix < 0] = 0
        # distance = sim_matrix - 1
        norm_data = normalize(candidates_embeddings, norm="l2")
        fitted = self.clustering.fit(norm_data)

        avg = []

        n_clusters = len(fitted.medoids_)

        for j in range(n_clusters):
            idx = np.where(fitted.labels_ == j)[0]
            avg.append(np.mean(idx))
        # 2: get keyphrase embedding closest to cluster centroid
        closest, _ = pairwise_distances_argmin_min(
            fitted.medoids_,
            norm_data,
            # fitted.cluster_centers_, candidates_embeddings, metric="cosine"
        )
        ordering = sorted(range(n_clusters), key=lambda k: avg[k])

        top_n_scores = [(candidates_embeddings.index[closest[i]], i) for i in ordering]

        # keep score per document for geo association metrics
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )
        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index
        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)

        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class ComunityRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        # 1: Get network graph based?
        # 2: Comunities and assign a keyphrase from largest comunities
        top_n_scores = None
        ranking_p_doc = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class EigenRank(Ranker):
    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        # 1: Network graph
        # 2: EigenVector Centrality for the network and get
        top_n_scores = None
        ranking_p_doc = None
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class AffinityRank(Ranker):
    """Improved Affinity Graph Based Multi-Document Summarization"""

    def __init__(self, **kwargs) -> None:
        self.gamma = 0.9
        self.damping = 0.85
        self.epsilon = 1e-5

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )

        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        M = cosine_similarity(candidates_embeddings, candidates_embeddings)
        M[np.diag_indices_from(M)] = 0

        M_tilda = np.zeros_like(M)
        for t in range(1, 5 + 1):
            M_tilda = M_tilda + self.gamma ** (t - 1) * np.linalg.matrix_power(M, t)
        M_tilda = M_tilda / np.linalg.norm(M_tilda, axis=1, keepdims=True)

        # TODO: check size
        sentence_score_t_1 = np.random.rand(len(score_per_document.columns))

        first_term = np.multiply(self.damping, M_tilda)
        B_transpose = np.transpose(M_tilda)
        temp = np.matmul(M_tilda, B_transpose)
        second_term = np.multiply(1 - self.damping, temp)
        constant_matrix = np.add(first_term, second_term)
        while True:
            sentence_score_t = constant_matrix.dot(sentence_score_t_1)
            condition = np.linalg.norm(
                np.subtract(sentence_score_t, sentence_score_t_1)
            )
            if condition * condition > self.epsilon:
                break
            else:
                sentence_score_t_1 = sentence_score_t

        top_n_scores = sorted(
            [
                (sent, s)
                for sent, s in zip(candidate_document_matrix.index, sentence_score_t_1)
            ],
            key=itemgetter(1),
            reversed=True,
        )
        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


class PageRank(Ranker):
    def __init__(self, **kwargs) -> None:
        self.alpha = 0.6
        self.delta = 0.3
        self.epsilon = 1e-5

    def _rank(
        self,
        candidates_embeddings: pd.DataFrame,
        documents_embeddings: pd.DataFrame,
        candidate_document_matrix: pd.DataFrame,
        single_mode_ranking_per_doc: dict,
        *args,
        **kwargs,
    ):
        # N = M.shape[1]
        # v = np.ones(N) / N
        # M_hat = (d * M + (1 - d) / N)
        # for i in range(num_iterations):
        #     v = M_hat @ v
        # return v

        """based on https://github.com/bshivangi47/Text-Summarization-using-Corank/blob/main/for100files.py"""
        score_per_document = pd.DataFrame(
            cosine_similarity(candidates_embeddings, documents_embeddings)
        )

        score_per_document.index = candidates_embeddings.index
        score_per_document.columns = documents_embeddings.index

        adjacency_matrix = cosine_similarity(
            candidates_embeddings, candidates_embeddings
        )

        adjacency_matrix[np.diag_indices_from(adjacency_matrix)] = 0
        adjacency_matrix[adjacency_matrix < self.delta] = 0

        # Matrix_B= row normalisation of bag of words model
        # matrix_B = np.array([[0 for x in range(len(bag_of_words[0]))] for y in range(len(bag_of_words))])
        matrix_B = candidate_document_matrix / np.linalg.norm(
            candidate_document_matrix, axis=1, keepdims=True
        )

        # TODO: Check size
        sentence_score_t_1 = np.random.rand(len(score_per_document.columns))
        # sentence_score_t_1 = np.zeros(np.asarray(corpus).shape, dtype=float, order='C')
        first_term = np.multiply(self.alpha, adjacency_matrix)
        B_transpose = np.transpose(matrix_B)
        temp = np.matmul(matrix_B, B_transpose)
        second_term = np.multiply(1 - self.alpha, temp)
        constant_matrix = np.add(first_term, second_term)
        while True:
            sentence_score_t = constant_matrix.dot(sentence_score_t_1)
            condition = np.linalg.norm(
                np.subtract(sentence_score_t, sentence_score_t_1)
            )
            if condition * condition > self.epsilon:
                break
            else:
                sentence_score_t_1 = sentence_score_t

        top_n_scores = sorted(
            [
                (sent, s)
                for sent, s in zip(candidate_document_matrix.index, sentence_score_t_1)
            ],
            key=itemgetter(1),
            reverse=True,
        )

        ranking_p_doc = self._score_to_ranking_p_doc(score_per_document)
        return (
            documents_embeddings,
            candidates_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        )


MD_RANK_STRATEGIES = {
    "MEAN": MeanRank,
    "MAX": MaxRank,
    "MEANTOPMAX": MeanTopMaxRank,
    "MAXMAX": MaxMaxRank,
    "MMR": MmrRank,
    # "MMRPos": MMRPos,
    "RRF": RRFRank,
    "ITCS": ITCSRank,
    "MAXSUM": MaxSumRank,
    "CLUSTERCENTROIDS": ClusterCentroidsRank,
    "UMAPCLUSTERS": UmapClustersRank,
    "COMUNITY": ComunityRank,
    "EIGEN": EigenRank,
    "PAGERANK": PageRank,
    "AFFINITY": AffinityRank,
    "DPSC": DPSCRank,
}
