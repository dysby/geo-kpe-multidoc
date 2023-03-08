import time
import re
import math
import numpy as np
import simplemma

from functools import reduce
from nltk import RegexpParser

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import OPTICS, DBSCAN

from typing import Dict, List, Tuple, Set, Callable

from keybert.mmr import mmr
from utils.IO import read_from_file

class Document:
    """
    Class to encapsulate document representation and functionality
    """

    def __init__(self, raw_text, id):
        """
        Stores the raw text representation of the doc, a pos_tagger and the grammar to
        extract candidates with.
        
        Attributes:
            self.raw_text -> Raw text representation of the document
            self.id -> Document id in corpus

            self.clustering_methods -> Dictionary with various clustering methods
        """

        self.raw_text = raw_text
        self.id = id
        self.clustering_methods = {"OPTICS" : OPTICS, "DBSCAN" : DBSCAN}

    def pos_tag(self, tagger, memory, id):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        self.tagged_text, self.doc_sents, self.doc_sents_words = tagger.pos_tag_text_sents_words(self.raw_text, memory, id)
        self.doc_sents = [sent.text for sent in self.doc_sents if sent.text.strip()]

    def build_doc_graph_edges(self, doc_graph : Dict) -> Dict:
        unvisited_clusters = list(doc_graph.keys())
        for l_og in doc_graph:
            del unvisited_clusters[0]
            for cand_og in doc_graph[l_og]:

                for l_dest in unvisited_clusters:
                    for cand_dest in doc_graph[l_dest]:
                        distance_l = [ 1 /(max(1, abs(pi - pj))) for pi in doc_graph[l_og][cand_og]["pos"] \
                        for pj in doc_graph[l_dest][cand_dest]["pos"]]

                        weight = reduce(lambda x,y: x+y, distance_l)
                        doc_graph[l_og][cand_og]["edges"][cand_dest] = weight
                        doc_graph[l_dest][cand_dest]["edges"][cand_og] = weight 
        return doc_graph

    def rebalance_graph_edges(self, doc_graph : Dict, a : float = 0.5) -> Dict:
        for l_og in doc_graph:
            if len(doc_graph[l_og]) != 1:
                f_cand = sorted(doc_graph[l_og], key= lambda x: doc_graph[l_og][x]["pos"][0])[0]
                pi = doc_graph[l_og][f_cand]["pos"][0]
                other_cand_edges = [doc_graph[l_og][cand]["edges"] for cand in doc_graph[l_og] if cand != f_cand]

                for l_dest in doc_graph:
                    if l_dest != l_og:
                        for cand in doc_graph[l_dest]:
                            doc_graph[l_dest][cand]["edges"][f_cand] += a * (math.e ** (1 / max(1,pi))) \
                            * reduce(lambda x,y: x + y, [e[cand] for e in other_cand_edges])

        return doc_graph

    def build_doc_graph(self, model, stemmer : Callable = None, clustering_method : str = "OPTICS", alpha : float = 0.5) -> Dict:
        """
        Method that builds a graph representation of the document at hand.
        """

        self.doc_embed = model.embed(stemmer.stem(self.raw_text)) if stemmer else model.embed(self.raw_text)
        candidates = [candidate for candidate in self.candidate_dic.keys()]
        candidate_embed_list = [model.embed(stemmer.stem(candidate)) if stemmer else model.embed(candidate) for candidate in self.candidate_dic]

        clustering = self.clustering_methods[clustering_method](min_samples=2, metric= 'cosine').fit(candidate_embed_list)
        max_label = np.max(clustering.labels_)

        doc_graph = {}
        for i in range(len(clustering.labels_)):
            label = clustering.labels_[i] 
            if label == -1:
                max_label += 1
                label = max_label

            candidate = candidates[i]
            candidate_graph = {"pos" : self.candidate_dic[candidate], "embed" : candidate_embed_list[i], 
                               "doc_sim" : float(np.absolute(cosine_similarity([candidate_embed_list[i]], self.doc_embed.reshape(1, -1)))[0][0]), "edges" : {}}
            
            if label not in doc_graph:
                doc_graph[label] = {}
            doc_graph[label][candidate] = candidate_graph                                             

        doc_graph = self.build_doc_graph_edges(doc_graph)

        return self.rebalance_graph_edges(doc_graph, alpha) 
                
    def extract_candidates(self, min_len : int = 5, grammar : str = "", lemmer : Callable = None):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document and 
        stores the sentences each candidate occurs in
        """
        candidate_set = set()

        parser = RegexpParser(grammar)
        np_trees = list(parser.parse_sents(self.tagged_text))

        for i in range(len(np_trees)):
            temp_cand_set = []
            for subtree in np_trees[i].subtrees(filter = lambda t : t.label() == 'NP'):
                temp_cand_set.append(' '.join(word for word, tag in subtree.leaves()))

            for candidate in temp_cand_set:
                if len(candidate) > min_len:
                    candidate_set.add(candidate)

        self.candidate_dic = {candidate : [] for candidate in candidate_set}
        n_words = len(self.raw_text.split(" "))
        avg_len_word = len(self.raw_text) / n_words
        for candidate in candidate_set:
            detected = False
            for match in re.finditer(re.escape(candidate), self.raw_text):
                self.candidate_dic[candidate].append(math.floor((match.span()[0] + (match.span()[1]-match.span()[0])/2) / avg_len_word))
                detected = True

            #TODO: Removed this valve later
            if not detected:
                self.candidate_dic[candidate].append(n_words-1)
        
    def rank_candidates(self, doc_graph : Dict, l_v : float = 0.2) -> List[Tuple]:
        cos_sim_total = reduce(lambda x, y: x + y, [doc_graph[l][c]["doc_sim"] for l in doc_graph for c in doc_graph[l]])
        prior_s = { c : (1-l_v)*(doc_graph[l][c]["doc_sim"] / cos_sim_total) for l in doc_graph for c in doc_graph[l]}

        #TODO: Check this
        succ_s = {}
        for l in doc_graph:
            for c in doc_graph[l]:
                edges = doc_graph[l][c]["edges"]
                succ_s[c] = reduce(lambda x,y: x + y, [edges[e] for e in edges]) if edges != {} else 1
        
        res_scores = {}
        for l in doc_graph:
            for c in doc_graph[l]:
                edges = doc_graph[l][c]["edges"]
                post_score = reduce(lambda x,y: x + y, [ (edges[e]*prior_s[c] / succ_s[c]) for e in edges]) \
                if edges != {} else 0
                
                res_scores[c] = prior_s[c] + l_v * post_score

        return res_scores

    def top_n_candidates(self, model, top_n: int = 5, min_len : int = 5, stemmer : Callable = None, **kwargs) -> List[Tuple]:

        t = time.time()
        clustering_alg = "OPTICS" if ("clustering" not in kwargs or kwargs["clustering"] == "") else kwargs["clustering"]
        alpha_v = 0.5 if "alpha" not in kwargs else kwargs["alpha"]
        doc_graph = self.build_doc_graph(model, stemmer, clustering_alg, alpha_v)
        print(f'Build Doc Multipartite Graph = {time.time() -  t:.2f}')

        t = time.time()
        lambda_v = 0.2 if "lambda" not in kwargs else kwargs["lambda"]
        cand_scores = sorted(self.rank_candidates(doc_graph, lambda_v), reverse= True, key= lambda x: x[1])
        print(f'Ranking Candidates = {time.time() -  t:.2f}')

        if top_n == -1:
            return cand_scores, [candidate[0] for candidate in cand_scores]

        return cand_scores[:top_n], [candidate[0] for candidate in cand_scores]