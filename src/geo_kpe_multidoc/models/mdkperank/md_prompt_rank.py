from operator import itemgetter
from typing import Callable, List, Optional, Tuple

from numpy import mean

from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models import BaseKPModel
from geo_kpe_multidoc.models.mdkperank.mdkperank_model import MDKPERankOutput
from geo_kpe_multidoc.models.mdkperank.mdkperank_strategy import STRATEGIES
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
    ) -> MDKPERankOutput:
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
        for doc in topic_docs:
            doc_candidates, _ = self.base_model.extract_candidates(
                doc, min_len, lemmer=None, **kwargs
            )
            topic_candidates.update(doc_candidates)
            for candidate in doc_candidates:
                candidate_document_matrix.setdefault(candidate, set()).add(doc.id)

        for doc in topic_docs:
            for out_candidate in topic_candidates.difference(set(doc.candidate_set)):
                doc.candidate_set.append(out_candidate)
                # TODO: Multi-document PromptRank, how to set candidate position?
                # set position to last token of the document
                doc.candidate_positions.append(
                    (self.base_model.max_len, self.base_model.max_len)
                )

        ranking_p_doc = {}
        scores_for_candidate = {}
        for doc in topic_docs:
            # top_candidates_n_scores, candidates
            ranking_p_doc[doc.id] = self.base_model.extract_kp_from_doc(
                doc,
                top_n=top_n,
                min_len=min_len,
                stemming=stemming,
                lemmer=lemmer,
                **kwargs,
            )

            for candidate, score in ranking_p_doc[doc.id][0]:
                scores_for_candidate.setdefault(candidate, []).append(score)

        top_n_scores = sorted(
            [
                (candidate, mean(scores))
                for candidate, scores in scores_for_candidate.items()
            ],
            key=itemgetter(1),
            reverse=True,
        )
        candidates, _ = list(zip(*top_n_scores))

        return (
            top_n_scores,
            candidates,
            candidate_document_matrix,
            # keyphrase_coordinates,
            ranking_p_doc,
        )
