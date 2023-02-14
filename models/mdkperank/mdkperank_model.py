import numpy as np
from typing import List, Tuple, Set
from models.pre_processing import post_processing_utils
from nltk.stem import PorterStemmer
from statistics import mean


from datasets.process_datasets import *

from models.base_KP_model import BaseKPModel
from models.embedrank.embedrank_model import EmbedRank
from models.maskrank.maskrank_model import MaskRank
from models.fusion_model import FusionModel

from models.pre_processing.language_mapping import choose_tagger, choose_lemmatizer
from models.pre_processing.pos_tagging import POS_tagger_spacy
from models.pre_processing.pre_processing_utils import remove_punctuation, remove_whitespaces

class MDKPERank(BaseKPModel):
    def __init__(self, model, tagger):
        self.base_model_embed = EmbedRank(model, tagger)
        self.base_model_mask = MaskRank(model, tagger)
        super().__init__(model)

    def extract_kp_from_doc(self, doc, top_n, min_len, stemmer = None, lemmer = None, **kwargs) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """
        return self.base_model_embed.extract_mdkpe_embeds(doc, top_n, min_len, stemmer, lemmer, **kwargs)

    def extract_kp_from_topic(self, topic, dataset: str = "MKDUC01", 
    top_n: int = 15, min_len: int = 5, stemming: bool = False, lemmatize: bool = False, **kwargs) -> List[List[Tuple]]:

        topic_res = [self.extract_kp_from_doc(doc, -1, min_len, stemming, lemmatize, **kwargs) for doc in topic[0]]
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

        res_p_doc = [doc[0].evaluate_n_candidates(cand_embeds, cand_set) for doc in topic_res]
        scores_per_candidate = {}

        for doc in res_p_doc:
            for cand_t in doc[0]:
                if cand_t[0] not in scores_per_candidate:
                    scores_per_candidate[cand_t[0]] = []
                scores_per_candidate[cand_t[0]].append(cand_t[1])

        for cand in scores_per_candidate:
            scores_per_candidate[cand] = mean(scores_per_candidate[cand])

        scores = sorted([(cand, scores_per_candidate[cand]) for cand in scores_per_candidate], reverse= True, key= lambda x: x[1])
        cand_set = [cand[0] for cand in scores]

        return scores, cand_set

    def extract_kp_from_corpus(self, corpus, dataset: str = "MKDUC01", 
    top_n: int = 15, min_len: int = 5, stemming: bool = False, lemmatize: bool = False, **kwargs) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        res = [self.extract_kp_from_topic(topic, dataset, top_n, min_len, stemming, lemmatize, **kwargs) for topic in corpus]

        return res