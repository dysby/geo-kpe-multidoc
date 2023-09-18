import os
from operator import itemgetter
from pathlib import Path
from typing import Callable, List, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models import BaseKPModel
from geo_kpe_multidoc.models.mdkperank.mdkperank_model import MdKPEOutput
from geo_kpe_multidoc.models.mdkperank.mdkperank_strategy import MD_RANK_STRATEGIES
from geo_kpe_multidoc.models.promptrank.promptrank import PromptRank


class MdPromptRank(BaseKPModel):
    def __init__(self, base_model: PromptRank, rank_strategy: str = "MEAN", **kwargs):
        self.base_model = base_model
        # TODO: what how to join MaskRank
        # self.base_model_mask = MaskRank(model, tagger)
        self.name = (
            ".".join([self.__class__.__name__, base_model.name.split("_")[0]])
            + base_model.name[base_model.name.index("_") :]
        )

        self.ranking_strategy = MD_RANK_STRATEGIES[rank_strategy](
            in_single_mode=True, **kwargs
        )

    def extract_kp_from_topic(
        self,
        topic_docs: List[Document],
        top_n: int = -1,
        kp_min_len: int = 5,
        lemmer: Optional[Callable] = None,
        **kwargs,
    ) -> MdKPEOutput:
        """
        Extract keyphrases from list of documents

        Parameters
        ----------
            topic: List[str], a List of Documents for the topic.

        Returns
        -------
            List KPE extracted from the agregation of documents set, with their
            score(/embedding?)
        """

        # 1 - extract candidates for each document to build a topic candidate extraction.
        # 2 - for each document

        use_cache = kwargs.get("cache_md_embeddings", False)

        topic_res = None
        if use_cache:
            topic_res = self._read_md_embeddings_from_cache(topic_docs[0].topic)

        if not topic_res:
            topic_candidates = set()
            candidate_document_matrix = {}
            doc_ids = set()
            for doc in topic_docs:
                doc_ids.add(doc.id)
                doc_candidates, _ = self.base_model.extract_candidates(
                    doc, kp_min_len, lemmer=lemmer, **kwargs
                )
                topic_candidates.update(doc_candidates)
                for candidate in doc_candidates:
                    candidate_document_matrix.setdefault(candidate, []).append(doc.id)

            # TODO: refactor out candidate_document_matrix to pd.DataFrame
            df = (
                pd.DataFrame()
                .reindex(index=topic_candidates, columns=doc_ids)
                .fillna(0)
                .astype(int)
            )
            for cand, docs in candidate_document_matrix.items():
                # TODO: use index slice assignment without looping on docs.
                for doc in docs:
                    df.loc[cand, doc] += 1

            candidate_document_matrix = df

            single_mode_ranking_per_doc = {}
            scores_for_candidate = {}
            topic_res = (
                []
            )  # # List[(doc, cand_embeds, candidate_set, ranking_in_doc), ...]
            for doc in topic_docs:
                # append  missing candidates to check with doc
                missing = list(topic_candidates - set(doc.candidate_set))
                missing_position = [
                    (self.base_model.max_len, self.base_model.max_len)
                    for _ in range(len(missing))
                ]

                doc.candidate_set.extend(missing)
                doc.candidate_positions.extend(missing_position)

                ranking_in_doc, _candidades = self.base_model.top_n_candidates(
                    doc,
                    doc.candidate_set,
                    doc.candidate_positions,
                    top_n=top_n,
                    **kwargs,
                )
                # doc, cand_embeddings (empty list), candidates (same for all doc in topic), ranking
                topic_res.append((doc, [], doc.candidate_set, ranking_in_doc))

            if use_cache:
                self._save_md_embeddings_in_cache(topic_res, topic_docs[0].topic)
        # Rank Candidates
        (
            documents_embeddings,
            candidate_embeddings,
            _wrong_candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        ) = self.ranking_strategy(topic_res)

        candidates = candidate_document_matrix.index.to_list()

        return MdKPEOutput(
            top_n_scores=top_n_scores,
            candidates=candidates,
            candidate_document_matrix=candidate_document_matrix,
            # keyphrase_coordinates,
            ranking_p_doc=ranking_p_doc,
        )

    def _save_md_embeddings_in_cache(self, topic_res: List, topic_id: str):
        # topic_res: List[(doc, cand_embeds, candidate_set, ranking_in_doc), ...]
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
