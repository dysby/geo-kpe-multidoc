from operator import itemgetter
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models import BaseKPModel
from geo_kpe_multidoc.models.mdkperank.mdkperank_model import MdKPEOutput
from geo_kpe_multidoc.models.promptrank.promptrank import PromptRank


class MdPromptRank(BaseKPModel):
    def __init__(self, base_model: PromptRank, rank_strategy: str = "MEAN", **kwargs):
        self.base_model: PromptRank = base_model
        # TODO: what how to join MaskRank
        # self.base_model_mask = MaskRank(model, tagger)
        self.name = (
            ".".join([self.__class__.__name__, base_model.name.split("_")[0]])
            + base_model.name[base_model.name.index("_") :]
        )

    def extract_kp_from_topic(
        self,
        topic_docs: List[Document],
        top_n: int = 15,
        min_len: int = 5,
        stemming: bool = False,
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

        topic_candidates = set()
        candidate_document_matrix = {}
        doc_ids = set()
        for doc in topic_docs:
            doc_ids.add(doc.id)
            doc_candidates, _ = self.base_model.extract_candidates(
                doc, min_len, lemmer=lemmer, **kwargs
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
            for doc in docs:
                df.loc[cand, doc] += 1

        candidate_document_matrix = df

        ranking_p_doc = {}
        scores_for_candidate = {}
        for doc in topic_docs:
            # append  missing candidates to check with doc
            missing = list(topic_candidates - set(doc.candidate_set))
            missing_position = [
                (self.base_model.max_len, self.base_model.max_len)
                for _ in range(len(missing))
            ]

            doc.candidate_set.extend(missing)
            doc.candidate_positions.extend(missing_position)

            ranking_p_doc[doc.id] = self.base_model.top_n_candidates(
                doc, doc.candidate_set, doc.candidate_positions, **kwargs
            )

            for candidate, score in ranking_p_doc[doc.id][0]:
                scores_for_candidate.setdefault(candidate, []).append(score)

        top_n_scores = sorted(
            [
                (candidate, np.mean(scores))
                for candidate, scores in scores_for_candidate.items()
            ],
            key=itemgetter(1),
            reverse=True,
        )
        candidates, _ = list(zip(*top_n_scores))

        return MdKPEOutput(
            top_n_scores=top_n_scores,
            candidates=candidates,
            candidate_document_matrix=candidate_document_matrix,
            # keyphrase_coordinates,
            ranking_p_doc=ranking_p_doc,
        )
