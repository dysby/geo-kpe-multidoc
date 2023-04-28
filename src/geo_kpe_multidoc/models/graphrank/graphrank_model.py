import math
import re
from functools import reduce
from time import time
from typing import Callable, List, Set, Tuple

import numpy as np
from datasets.process_datasets import *
from models.base_KP_model import BaseKPModel
from models.graphrank.graphrank_document_abstraction import Document
from models.pre_processing.language_mapping import choose_lemmatizer, choose_tagger
from models.pre_processing.pos_tagging import POS_tagger_spacy
from models.pre_processing.pre_processing_utils import (
    remove_punctuation,
    remove_whitespaces,
)
from nltk import RegexpParser
from nltk.stem import PorterStemmer
from torch import cosine_similarity


class GraphRank(BaseKPModel):
    """
    Simple class to encapsulate GraphRank functionality. Uses
    the KeyBert backend to retrieve models
    """

    def __init__(self, model, tagger):
        super().__init__(model)
        self.tagger = POS_tagger_spacy(tagger)
        self.grammar = """  NP: 
        {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}"""
        self.counter = 0

    def update_tagger(self, dataset: str = "") -> None:
        self.tagger = (
            POS_tagger_spacy(choose_tagger(dataset))
            if choose_tagger(dataset) != self.tagger.name
            else self.tagger
        )

    def pre_process(self, doc="", **kwargs) -> str:
        """
        Method that defines a pre_processing routine, removing punctuation and whitespaces
        """
        doc = remove_punctuation(doc)
        return remove_whitespaces(doc)[1:]

    def extract_kp_from_doc(
        self, doc, top_n, min_len, stemmer=None, lemmer=None, **kwargs
    ) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """

        doc = Document(doc, self.counter)

        t = time.time()
        doc.pos_tag(
            self.tagger,
            False if "pos_tag_memory" not in kwargs else kwargs["pos_tag_memory"],
            self.counter,
        )
        print(f"Pos_Tag Doc = {time.time() -  t:.2f}")

        t = time.time()
        doc.extract_candidates(min_len, self.grammar, lemmer)
        print(f"Extract Candidates = {time.time() -  t:.2f}")

        top_n, candidate_set = doc.top_n_candidates(
            self.model, top_n, min_len, stemmer, **kwargs
        )

        print(f"document {self.counter} processed\n")
        self.counter += 1

        return (top_n, candidate_set)

    def extract_kp_from_corpus(
        self,
        corpus,
        dataset: str = "DUC",
        top_n: int = 15,
        min_len: int = 5,
        stemming: bool = False,
        lemmatize: bool = False,
        **kwargs,
    ) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        self.counter = 0
        self.update_tagger(dataset)

        stemer = None if not stemming else PorterStemmer()
        lemmer = None if not lemmatize else choose_lemmatizer(dataset)

        return [
            self.extract_kp_from_doc(doc[0], top_n, min_len, stemer, lemmer, **kwargs)
            for doc in corpus
        ]

    def _pos_tag_doc(self, doc: Document, tagger, memory, id):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        (
            doc.tagged_text,
            doc.doc_sents,
            doc.doc_sents_words,
        ) = tagger.pos_tag_text_sents_words(doc.raw_text, memory, id)
        doc.doc_sents = [sent.text for sent in doc.doc_sents if sent.text.strip()]

    def build_doc_graph_edges(self, doc_graph: dict) -> dict:
        unvisited_clusters = list(doc_graph.keys())
        for l_og in doc_graph:
            del unvisited_clusters[0]
            for cand_og in doc_graph[l_og]:
                for l_dest in unvisited_clusters:
                    for cand_dest in doc_graph[l_dest]:
                        distance_l = [
                            1 / (max(1, abs(pi - pj)))
                            for pi in doc_graph[l_og][cand_og]["pos"]
                            for pj in doc_graph[l_dest][cand_dest]["pos"]
                        ]

                        weight = reduce(lambda x, y: x + y, distance_l)
                        doc_graph[l_og][cand_og]["edges"][cand_dest] = weight
                        doc_graph[l_dest][cand_dest]["edges"][cand_og] = weight
        return doc_graph

    def rebalance_graph_edges(self, doc_graph: dict, a: float = 0.5) -> dict:
        for l_og in doc_graph:
            if len(doc_graph[l_og]) != 1:
                f_cand = sorted(
                    doc_graph[l_og], key=lambda x: doc_graph[l_og][x]["pos"][0]
                )[0]
                pi = doc_graph[l_og][f_cand]["pos"][0]
                other_cand_edges = [
                    doc_graph[l_og][cand]["edges"]
                    for cand in doc_graph[l_og]
                    if cand != f_cand
                ]

                for l_dest in doc_graph:
                    if l_dest != l_og:
                        for cand in doc_graph[l_dest]:
                            doc_graph[l_dest][cand]["edges"][f_cand] += (
                                a
                                * (math.e ** (1 / max(1, pi)))
                                * reduce(
                                    lambda x, y: x + y,
                                    [e[cand] for e in other_cand_edges],
                                )
                            )

        return doc_graph

    def build_doc_graph(
        self,
        doc,
        stemmer: Callable = None,
        clustering_method: str = "OPTICS",
        alpha: float = 0.5,
    ) -> dict:
        """
        Method that builds a graph representation of the document at hand.
        """

        doc.doc_embed = (
            self.model.embed(stemmer.stem(doc.raw_text))
            if stemmer
            else self.model.embed(doc.raw_text)
        )
        candidates = [candidate for candidate in doc.candidate_dic.keys()]
        candidate_embed_list = [
            self.model.embed(stemmer.stem(candidate))
            if stemmer
            else self.model.embed(candidate)
            for candidate in doc.candidate_dic
        ]

        clustering = doc.clustering_methods[clustering_method](
            min_samples=2, metric="cosine"
        ).fit(candidate_embed_list)
        max_label = np.max(clustering.labels_)

        doc_graph = {}
        for i in range(len(clustering.labels_)):
            label = clustering.labels_[i]
            if label == -1:
                max_label += 1
                label = max_label

            candidate = candidates[i]
            candidate_graph = {
                "pos": doc.candidate_dic[candidate],
                "embed": candidate_embed_list[i],
                "doc_sim": float(
                    np.absolute(
                        cosine_similarity(
                            [candidate_embed_list[i]], doc.doc_embed.reshape(1, -1)
                        )
                    )[0][0]
                ),
                "edges": {},
            }

            if label not in doc_graph:
                doc_graph[label] = {}
            doc_graph[label][candidate] = candidate_graph

        doc_graph = self.build_doc_graph_edges(doc_graph)

        return self.rebalance_graph_edges(doc_graph, alpha)

    def extract_candidates(
        self, doc, min_len: int = 5, grammar: str = "", lemmer: Callable = None
    ):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document and
        stores the sentences each candidate occurs in
        """
        candidate_set = set()

        parser = RegexpParser(grammar)
        np_trees = list(parser.parse_sents(doc.tagged_text))

        for i in range(len(np_trees)):
            temp_cand_set = []
            for subtree in np_trees[i].subtrees(filter=lambda t: t.label() == "NP"):
                temp_cand_set.append(" ".join(word for word, tag in subtree.leaves()))

            for candidate in temp_cand_set:
                if len(candidate) > min_len:
                    candidate_set.add(candidate)

        doc.candidate_dic = {candidate: [] for candidate in candidate_set}
        n_words = len(doc.raw_text.split(" "))
        avg_len_word = len(doc.raw_text) / n_words
        for candidate in candidate_set:
            detected = False
            for match in re.finditer(re.escape(candidate), doc.raw_text):
                doc.candidate_dic[candidate].append(
                    math.floor(
                        (match.span()[0] + (match.span()[1] - match.span()[0]) / 2)
                        / avg_len_word
                    )
                )
                detected = True

            # TODO: Removed this valve later
            if not detected:
                doc.candidate_dic[candidate].append(n_words - 1)

    def rank_candidates(self, doc_graph: dict, l_v: float = 0.2) -> List[Tuple]:
        cos_sim_total = reduce(
            lambda x, y: x + y,
            [doc_graph[l][c]["doc_sim"] for l in doc_graph for c in doc_graph[l]],
        )
        prior_s = {
            c: (1 - l_v) * (doc_graph[l][c]["doc_sim"] / cos_sim_total)
            for l in doc_graph
            for c in doc_graph[l]
        }

        # TODO: Check this
        succ_s = {}
        for l in doc_graph:
            for c in doc_graph[l]:
                edges = doc_graph[l][c]["edges"]
                succ_s[c] = (
                    reduce(lambda x, y: x + y, [edges[e] for e in edges])
                    if edges != {}
                    else 1
                )

        res_scores = {}
        for l in doc_graph:
            for c in doc_graph[l]:
                edges = doc_graph[l][c]["edges"]
                post_score = (
                    reduce(
                        lambda x, y: x + y,
                        [(edges[e] * prior_s[c] / succ_s[c]) for e in edges],
                    )
                    if edges != {}
                    else 0
                )

                res_scores[c] = prior_s[c] + l_v * post_score

        return res_scores

    def top_n_candidates(
        self,
        doc,
        top_n: int = 5,
        min_len: int = 5,
        stemmer: Callable = None,
        **kwargs,
    ) -> List[Tuple]:
        t = time.time()
        clustering_alg = (
            "OPTICS"
            if ("clustering" not in kwargs or kwargs["clustering"] == "")
            else kwargs["clustering"]
        )
        alpha_v = 0.5 if "alpha" not in kwargs else kwargs["alpha"]
        doc_graph = self.build_doc_graph(doc, stemmer, clustering_alg, alpha_v)
        print(f"Build Doc Multipartite Graph = {time.time() -  t:.2f}")

        t = time.time()
        lambda_v = 0.2 if "lambda" not in kwargs else kwargs["lambda"]
        cand_scores = sorted(
            self.rank_candidates(doc_graph, lambda_v), reverse=True, key=lambda x: x[1]
        )
        print(f"Ranking Candidates = {time.time() -  t:.2f}")

        if top_n == -1:
            return cand_scores, [candidate[0] for candidate in cand_scores]

        return cand_scores[:top_n], [candidate[0] for candidate in cand_scores]
