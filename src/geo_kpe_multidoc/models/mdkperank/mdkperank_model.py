import os
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.datasets.datasets import KPEDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.mdkperank.mdkperank_strategy import MD_RANK_STRATEGIES

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

        self.ranking_strategy = MD_RANK_STRATEGIES[rank_strategy](**kwargs)

    def _extract_doc_candidate_embeddings(
        self,
        doc,
        top_n: int = -1,
        kp_min_len: int = 0,
        lemmer=None,
        **kwargs,
    ):
        self.base_model_embed.extract_candidates(doc, kp_min_len, lemmer, **kwargs)

        _, cand_embeds, candidate_set = self.base_model_embed.embed_candidates(
            doc,
            **kwargs,
        )

        return (doc, cand_embeds, candidate_set)

    def extract_kp_from_doc(
        self, doc, top_n, kp_min_len, lemmer=None, **kwargs
    ) -> Tuple[Document, List[np.ndarray], List[str]]:  # Tuple[List[Tuple], List[str]]:
        """
        Extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """
        ranking_in_doc, _doc_candidates = self.base_model_embed.extract_kp_from_doc(
            doc, kp_min_len, lemmer, **kwargs
        )
        cand_embeds, candidate_set = doc.candidate_set_embed, doc.candidate_set
        # self.base_model_embed.extract_candidates(doc, kp_min_len, lemmer, **kwargs)
        # _, cand_embeds, candidate_set = self.base_model_embed.embed_candidates(
        #     doc,
        #     **kwargs,
        # )

        return (doc, cand_embeds, candidate_set, ranking_in_doc)

    # def _get_locations(
    #     self, docs_geo_coords, docs_of_candidate: List[str]
    # ) -> List[Tuple[float, float]]:
    #     """
    #     Get all coordenates of documents tha mention the candidate keyphrase
    #     TODO: what about repeated locations?
    #     """
    #     locations = [docs_geo_coords[doc_id] for doc_id in docs_of_candidate]
    #     return list(chain(*locations))

    def extract_kp_from_topic(
        self,
        topic_docs: List[Document],
        top_n: int = -1,
        kp_min_len: int = 0,
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

        use_cache = kwargs.get("cache_md_embeddings", False)

        topic_res = None
        if use_cache:
            topic_res = self._read_md_embeddings_from_cache(topic_docs[0].topic)

        if not topic_res:
            topic_res = [
                self.extract_kp_from_doc(
                    doc,
                    top_n=top_n,
                    kp_min_len=kp_min_len,
                    lemmatize=lemmatize,
                    **kwargs,
                )
                for doc in topic_docs
            ]
            if use_cache:
                self._save_md_embeddings_in_cache(topic_res, topic_docs[0].topic)

        (
            documents_embeddings,
            candidate_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        ) = self.ranking_strategy(topic_res)

        return MdKPEOutput(
            top_n_scores=top_n_scores,
            candidate_document_matrix=candidate_document_matrix,
            documents_embeddings=documents_embeddings,
            candidate_embeddings=candidate_embeddings,
            ranking_p_doc=ranking_p_doc,
        )

    # def _score_w_geo_association_I(S, N, I, lambda_=0.5, gamma=0.5):
    #     return S * lambda_ * (N - (N * gamma * I))

    # def _score_w_geo_association_C(S, N, C, lambda_=0.5, gamma=0.5):
    #     return S * lambda_ * N / (gamma * C)

    # def _score_w_geo_association_G(S, N, G, lambda_=0.5, gamma=0.5):
    #     return S * lambda_ * (N * gamma) * G

    def _save_md_embeddings_in_cache(self, topic_res: List, topic_id: str):
        topic_res = []  # # List[(doc, cand_embeds, candidate_set, ranking_in_doc), ...]
        logger.info(f"Saving {topic_id} embeddings in cache dir.")

        cache_file_path = os.path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            self.name[self.name.index("_") + 1 :],
            f"{topic_id}-md-embeddings.gz",
        )

        Path(cache_file_path).parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(
            topic_res,
            cache_file_path,
        )

    def _read_md_embeddings_from_cache(self, topic_id):
        # TODO: implement caching? is usefull only in future analysis
        cache_file_path = os.path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            self.name[self.name.index("_") + 1 :],
            f"{topic_id}-md-embeddings.gz",
        )

        if os.path.exists(cache_file_path):
            topic_res = joblib.load(cache_file_path)
            logger.debug(f"Load embeddings from cache {cache_file_path}")
            return topic_res
        return None
