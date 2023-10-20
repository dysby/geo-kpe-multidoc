import itertools
import math
import os
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models import EmbedRank
from geo_kpe_multidoc.models.mdkperank import MDKPERank
from geo_kpe_multidoc.models.mdkperank.mdkperank_strategy import MD_RANK_STRATEGIES


@dataclass
class MdKPEOutput:
    top_n_scores: List = None
    candidates: List[str] = None
    candidate_document_matrix: pd.DataFrame = None
    documents_embeddings: pd.DataFrame = None
    candidate_embeddings: pd.DataFrame = None
    ranking_p_doc: Dict[str, Tuple[List[Tuple[str, float]], List[str]]] = None


class MDKPERankPosCross(MDKPERank):
    def __init__(self, model: EmbedRank, rank_strategy: str = "MEAN", **kwargs):
        self.base_model_embed: EmbedRank = model
        # TODO: what how to join MaskRank
        # self.base_model_mask = MaskRank(model, tagger)
        self.name = (
            ".".join([self.__class__.__name__, model.name.split("_")[0]])
            + model.name[model.name.index("_") :]
        )

        if kwargs.get("no_position_feature", False):
            raise ValueError(
                "no_position_feature option is not compatible with MDKPERankPosCross"
            )
        if not kwargs.get("md_cross_doc", False):
            raise ValueError(
                "MDKPERankPosCross must run with md_cross_doc option enabled"
            )

        # TODO: Because MDKPERankPosCross computes cross candidate doc ranking
        # the ranking strategy does not need to know it is in cross or single
        # mode, just use the computed rank.
        in_single_mode = True

        self.ranking_strategy = MD_RANK_STRATEGIES[rank_strategy](
            in_single_mode=in_single_mode, **kwargs
        )

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
        self,
        doc: Document,
        top_n=-1,
        kp_min_len=0,
        lemmer: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[
        Document, List[np.ndarray], List[str], List
    ]:  # Tuple[List[Tuple], List[str]]:
        """
        Extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """
        ranking_in_doc, _doc_candidates = self.base_model_embed.extract_kp_from_doc(
            doc, top_n, kp_min_len, lemmer, **kwargs
        )
        cand_embeds, candidate_set = doc.candidate_set_embed, doc.candidate_set
        # self.base_model_embed.extract_candidates(doc, kp_min_len, lemmer, **kwargs)
        # _, cand_embeds, candidate_set = self.base_model_embed.embed_candidates(
        #     doc,
        #     **kwargs,
        # )

        return (doc, cand_embeds, candidate_set, ranking_in_doc)

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

        if topic_res is None:
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

            doc_ids = set()
            topic_candidates = set()
            candidate_document_matrix = {}
            documents_embeddings = {}
            candidate_embeddings = {}
            single_mode_ranking_per_doc = {}

            for doc, cand_embeds, doc_candidates, ranking_in_doc in topic_res:
                doc_ids.add(doc.id)
                topic_candidates.update(doc_candidates)
                for candidate in doc_candidates:
                    candidate_document_matrix.setdefault(candidate, []).append(doc.id)

                documents_embeddings[doc.id] = doc.doc_embed  # .reshape(1, -1)
                single_mode_ranking_per_doc[doc.id] = ranking_in_doc  # .reshape(1, -1)

                # Size([1, 768])
                # Not all Single Document Methods compute a candidate_embedding (e.g. PromptRank)
                for candidate, embedding in itertools.zip_longest(
                    doc_candidates, cand_embeds
                ):
                    candidate_embeddings.setdefault(candidate, []).append(embedding)

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

            # The candidate embedding is the average of each embedding
            # of the candidate in the document.
            candidate_embeddings = {
                candidate: np.mean(embeddings, axis=0)
                # if isinstance(embeddings[0], np.ndarray)
                # else None
                for candidate, embeddings in candidate_embeddings.items()
            }

            scores_per_doc = []
            for doc_id, ranking_in_doc in single_mode_ranking_per_doc.items():
                in_candidates, _ = list(zip(*ranking_in_doc))
                # candidate was not found inside the document
                # ADD candidate as if at the end of the document
                # TODO: candidate was not found inside the document, only for sentence-t5-base
                missing_candidates = list(topic_candidates - set(in_candidates))
                missing_candidates_scores = [
                    (1 + 1.2e8 / (256**3))
                    * math.log(
                        cosine_similarity(
                            documents_embeddings[doc_id],
                            candidate_embeddings[candidate],
                        )
                    )
                    for missing_candidate in missing_candidates
                ]

                for candidate, score in itertools.chain(
                    ranking_in_doc, zip(missing_candidates, missing_candidates_scores)
                ):
                    scores_per_doc.append(
                        {"doc": doc_id, "candidate": candidate, "score": score}
                    )

            candidate_scores_per_doc = pd.DataFrame(scores_per_doc)
            candidate_scores_per_doc = candidate_scores_per_doc.set_index("doc")
            # update single_mode_ranking_per_doc with new cross candidate ranking per doc
            for doc_id in doc_ids:
                single_mode_ranking_per_doc[doc_id] = list(
                    candidate_scores_per_doc.loc[doc_id].itertuples(
                        index=False, name=None
                    )
                )

            documents_embeddings = pd.DataFrame.from_dict(
                documents_embeddings, orient="index"
            )
            candidate_embeddings = pd.DataFrame.from_dict(
                candidate_embeddings, orient="index"
            )

            topic_res = (
                documents_embeddings,
                candidate_embeddings,
                candidate_document_matrix,
                single_mode_ranking_per_doc,
            )

            if use_cache:
                self._save_md_embeddings_in_cache(topic_res, topic_docs[0].topic)

        # topic_res: Tuple( documents_embeddings, candidate_embeddings, candidate_document_matrix, single_mode_ranking_per_doc)
        # single_mode_ranking_per_doc: dict[doc_id, list(tuple(candidate, score))]

        (
            documents_embeddings,
            candidate_embeddings,
            candidate_document_matrix,
            top_n_scores,
            ranking_p_doc,
        ) = self.ranking_strategy(topic_res, extract_features=False)

        return MdKPEOutput(
            top_n_scores=top_n_scores,
            candidate_document_matrix=candidate_document_matrix,
            documents_embeddings=documents_embeddings,
            candidate_embeddings=candidate_embeddings,
            ranking_p_doc=ranking_p_doc,
        )

    def _save_md_embeddings_in_cache(self, topic_res: List, topic_id: str):
        # topic_res: List[(doc, cand_embeds, candidate_set, ranking_in_doc), ...]
        logger.debug(f"Saving {topic_id} embeddings in cache dir.")

        cache_file_path = os.path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            self.name,
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
            self.name,
            f"{topic_id}-md-embeddings.gz",
        )

        if os.path.exists(cache_file_path):
            topic_res = joblib.load(cache_file_path)
            logger.debug(f"Load embeddings from cache {cache_file_path}")
            return topic_res
        return None

    # def _score_w_geo_association_I(S, N, I, lambda_=0.5, gamma=0.5):
    #     return S * lambda_ * (N - (N * gamma * I))

    # def _score_w_geo_association_C(S, N, C, lambda_=0.5, gamma=0.5):
    #     return S * lambda_ * N / (gamma * C)

    # def _score_w_geo_association_G(S, N, G, lambda_=0.5, gamma=0.5):
    #     return S * lambda_ * (N * gamma) * G
