import os
from operator import itemgetter
from pathlib import Path
from time import time
from typing import List, Optional, Tuple

import joblib
import numpy as np
import torch
from keybert._mmr import mmr
from keybert.backend._base import BaseEmbedder
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.base_KP_model import BaseKPModel
from geo_kpe_multidoc.models.embedrank.embedding_strategy import (
    STRATEGIES,
    CandidateEmbeddingStrategy,
    GlobalAttentionCandidateStrategy,
)
from geo_kpe_multidoc.models.pre_processing.post_processing_utils import (
    whitening_np,
    z_score_normalization,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import tokenize_hf
from geo_kpe_multidoc.models.sentence_embedder import LongformerSentenceEmbedder


class EmbedRank(BaseKPModel):
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models

    cand_mode:
        {mentions}?{no_context}+            - candidate embeddings from non
                            contextualized form
        {global_attention}+{dilated_(n)}?   - get embeddings using longformer global
                            attention in all doc tokens where a candidate is present.
                            If dilated mode is selected a sparse pattern is used,
                            considering global attention on every n position.
        in_n_out_context                    - add non contextualized form embedding to
                            mentions embeddings and do the average as candidade
                            embedding.
    """

    def __init__(
        self,
        model,
        candidate_selection_model,
        pooling_strategy: str = "mean",
        candidate_embedding_strategy: str = "mentions_no_context",
        max_seq_len: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model, candidate_selection_model, **kwargs)
        self.counter = 0
        # manualy set SentenceTransformer max seq lenght1
        # self.model.embedding_model.max_seq_length = 384

        # TODO: deal with global_attention and global_attention_dilated
        dilation = 128
        if "dilated" in candidate_embedding_strategy:
            dilation = int("".join(filter(str.isdigit, candidate_embedding_strategy)))
            candidate_embedding_strategy = candidate_embedding_strategy[
                : candidate_embedding_strategy.index("dilated") + len("dilated")
            ]
        strategy = STRATEGIES.get(candidate_embedding_strategy)
        self.pooling_strategy = pooling_strategy

        # TODO: Add support for e5 type models that require "query: " prompted text.
        self.add_query_prefix = (
            "query: " if kwargs.get("add_query_prefix", False) else ""
        )  # for intfloat/multilingual-e5-* models

        if max_seq_len:
            self.model.model_max_lenght = max_seq_len
        # force max sequence length, for sentence-t5* models

        self.candidate_embedding_strategy: CandidateEmbeddingStrategy = strategy(
            add_query_prefix=self.add_query_prefix, dilation=dilation
        )
        logger.info(
            f"Initialize EmbedRank w/ {self.candidate_embedding_strategy.__class__.__name__}"
        )

        self.whitening = kwargs.get("whitening", False)

    def _embed_doc(
        self,
        doc: Document,
        doc_mode: str = "",
        post_processing: List[str] = None,
        output_attentions=None,
    ) -> np.ndarray:
        """
        Method that embeds the document, having several modes according to usage.
        The default value just embeds the document normally.
        """

        # Check why document text is mutated to lower:
        # Used after POS_TAGGING,
        # at 1st stage document POS Tagging uses normal text including capital letters,
        # but later document handling will use only lowered text, embeddings, and such.
        # doc.raw_text = doc.raw_text.lower()
        if doc_mode == "global_attention":
            logger.warning(
                "Global Attention not used in SentenceTransformer API, use LongEmbedRank"
            )

        doc_embeddings = self.model.embedding_model.encode(
            self.add_query_prefix + doc.raw_text,
            show_progress_bar=False,
            output_value=None,
        )

        doc.token_ids = doc_embeddings["input_ids"].squeeze().tolist()
        doc.token_embeddings = doc_embeddings["token_embeddings"].detach().cpu()
        doc.attention_mask = doc_embeddings["attention_mask"].detach().cpu()

        # TODO: longformer attention window 256 with inputs only 128
        # limit_128 = True
        # if limit_128:
        #     doc.token_ids = doc.token_ids[:128]
        return doc_embeddings["sentence_embedding"].detach().cpu().numpy()

    def _aggregate_candidate_mention_embeddings(
        self,
        doc: Document,
        cand_mode: str = "",
        post_processing: List[str] = None,
    ):
        """
        Method that embeds the current candidate set, having several modes according to usage.
        The default value just embeds candidates directly.
        """
        # TODO: keep this init?
        doc.candidate_set_embed = []

        # TODO: deal with global_attention and global_attention_dilated
        self.candidate_embedding_strategy.candidate_embeddings(self.model, doc)

        if "z_score" in post_processing:
            # TODO: Why z_score_normalization by space split?
            doc.candidate_set_embed = z_score_normalization(
                doc.candidate_set_embed, doc.raw_text, self.model
            )

    # def _set_global_attention_on_candidates(self, doc: Document):
    #     mentions = []
    #     for candidate in doc.candidate_set:
    #         # mentions_positions, _ = self._search_mentions(doc, candidate)
    #         mentions_positions = _search_mentions(
    #             self.model, doc.candidate_mentions[candidate], doc.token_ids
    #         )
    #         if len(mentions_positions) > 0:
    #             # candidate mentions where found in document token_ids
    #             mentions.extend(mentions_positions)
    #     mentions = tuple(set(chain(*mentions)))
    #     logger.debug(f"Global Attention in {len(mentions)} tokens")
    #     doc.global_attention_mask[:, mentions] = 1

    def embed_candidates(
        self, doc: Document, **kwargs
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        TODO: Why embed_(n)_candidates?

        Return
        ------
            candidate_set_embed:    np.ndarray of the embeddings for each candidate.
            candicate_set:          List of candidates.
        """
        # doc_mode = kwargs.get("doc_mode", "")
        doc_mode = kwargs.get("cand_mode", "")
        cand_mode = kwargs.get("cand_mode", "")
        post_processing = kwargs.get("post_processing", [""])
        use_cache = kwargs.get("cache_embeddings", False)

        if use_cache:
            # this mutates doc
            cached = self._read_embeddings_from_cache(doc)
            if cached:
                return cached

        # TODO: simplify subclassing tokenization
        # Set Global Attention CLS token for all modes
        if isinstance(self.model, BaseEmbedder):
            # original tokenization by KeyBert/SentenceTransformer
            tokenized_doc = tokenize_hf(
                self.add_query_prefix + doc.raw_text, self.model
            )
        elif isinstance(self.model, LongformerSentenceEmbedder):
            # tokenize via local SentenceEmbedder Class
            tokenized_doc = self.model.tokenize(
                self.add_query_prefix + doc.raw_text,
                padding=True,
                pad_to_multiple_of=self.model.attention_window,
            )
        else:
            # BIGBIRD
            # tokenize via local SentenceEmbedder Class
            tokenized_doc = self.model.tokenize(self.add_query_prefix + doc.raw_text)

        doc.token_ids = tokenized_doc["input_ids"].squeeze().tolist()
        doc.global_attention_mask = torch.zeros(tokenized_doc["input_ids"].shape)
        doc.global_attention_mask[:, 0] = 1  # CLS token

        # TODO: Global Attention alternatives
        self.candidate_embedding_strategy.set_global_attention(self.model, doc)
        # Global attention on the first 100 tokens
        # global_attention_mask[:, :100] = 1
        # # Global attention on periods
        # global_attention_mask[(input_ids == self.tokenizer.convert_tokens_to_ids('.'))] = 1

        output_attentions = "attention_rank" in cand_mode

        start = time()
        doc.doc_embed = self._embed_doc(
            doc, doc_mode, post_processing, output_attentions
        )
        logger.debug(f"Embed Doc in {time() -  start:.2f}s")

        start = time()
        self._aggregate_candidate_mention_embeddings(doc, cand_mode, post_processing)
        logger.debug(f"Embed Candidates in {time() -  start:.2f}s")

        if use_cache:
            self._save_embeddings_in_cache(doc)

        return doc.doc_embed, doc.candidate_set_embed, doc.candidate_set

    def _save_embeddings_in_cache(self, doc: Document):
        logger.info(f"Saving {doc.id} embeddings in cache dir.")

        cache_file_path = os.path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            self.name[self.name.index("_") + 1 :],
            f"{doc.id}-embeddings.pkl",
        )

        Path(cache_file_path).parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(
            {
                "dataset": doc.dataset,
                "topic": doc.topic,
                "doc": doc.id,
                "doc_embedding": doc.doc_embed,
                "token_embeddings": doc.token_embeddings,
                "candidate_embeddings": doc.candidate_set_embed,
                "candidates": doc.candidate_set,
            },
            cache_file_path,
        )

    def _read_embeddings_from_cache(self, doc: Document):
        # TODO: implement caching? is usefull only in future analysis
        cache_file_path = os.path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            self.name[self.name.index("_") + 1 :],
            f"{doc.id}-embeddings.pkl",
        )

        if os.path.exists(cache_file_path):
            cache = joblib.load(cache_file_path)
            doc.doc_embed = cache["doc_embedding"]
            doc.candidate_set_embed = cache["candidate_embeddings"]
            doc.candidate_set = cache["candidates"]
            # TODO: load token embeddings
            logger.debug(f"Load embeddings from cache {cache_file_path}")
            return doc.doc_embed, doc.candidate_set_embed, doc.candidate_set

    def _rank_candidates(
        self,
        doc_embed: np.ndarray,
        candidate_set_embed: List[np.ndarray],
        candidate_set: List[str],
        top_n: int = -1,
        **kwargs,
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        """
        This method is key for each ranking model.
        Here the ranking heuritic is applied according to model definition.

        EmbedRank selects the candidates that have more similarity to the document.
        """
        mmr_mode = kwargs.get("mmr", False)
        mmr_diversity = (
            kwargs.get("mmr_diversity")
            if kwargs.get("mmr_diversity") is not None
            else 0.8
        )
        top_n = len(candidate_set) if top_n == -1 else top_n

        if self.whitening:
            new_embedding = np.concatenate(
                [candidate_set_embed, doc_embed.reshape(1, -1)]
            )
            whiten = whitening_np(new_embedding)

            doc_embed = whiten[-1, :]
            candidate_set_embed = whiten[:-1, :]

        doc_sim = []
        if mmr_mode:
            assert mmr_diversity >= 0
            assert mmr_diversity <= 1

            valid_top_n = len(candidate_set)
            if top_n > 0:
                valid_top_n = min(valid_top_n, top_n)

            doc_sim = mmr(
                doc_embed.reshape(1, -1),
                candidate_set_embed,
                candidate_set,
                top_n=valid_top_n,
                diversity=mmr_diversity,
            )
            candidate_score = doc_sim
        else:
            # TODO: convert ndarray to list. To be uniform with MDKPERank.
            doc_sim = cosine_similarity(candidate_set_embed, doc_embed.reshape(1, -1))

            candidate_score = sorted(
                [
                    (candidate, candidate_doc_sim.item())
                    for (candidate, candidate_doc_sim) in zip(candidate_set, doc_sim)
                ],
                # [(candidate_set[i], doc_sim[i][0]) for i in range(len(doc_sim))],
                reverse=True,
                key=itemgetter(1),
            )

        return candidate_score[:top_n], candidate_set

    def top_n_candidates(
        self, doc, candidate_list, positions, top_n, **kwargs
    ) -> List[Tuple]:
        doc_mode = kwargs.get("doc_mode", "")
        cand_mode = kwargs.get("cand_mode", "")
        post_processing = kwargs.get("post_processing", [""])
        # TODO: remove? use_cache = kwargs.get("embed_memory", False)
        # if cand_mode != "" and cand_mode != "AvgContext":
        #     logger.error(f"Getting Embeddings for word sentence (not used?)")
        #     self.embed_sents_words(doc, stemmer, use_cache)

        self.embed_candidates(doc, **kwargs)

        # candidate_score[:top_n], candidate_set
        ranking = self._rank_candidates(
            doc.doc_embed, doc.candidate_set_embed, doc.candidate_set, top_n, **kwargs
        )

        # DEBUG: check score distribution of found vs not found mentions
        # candidates_out_score = []
        # candidates_in_score = []
        # for candidate, score in ranking[0]:
        #     if candidate in doc.candidate_mentions_not_found:
        #         candidates_out_score.append(score)
        #     else:
        #         candidates_in_score.append(score)

        # import matplotlib.pyplot as plt
        # import pandas as pd

        # _, ax = plt.subplots()
        # density = False
        # pd.Series(candidates_out_score, name="out_score").hist(
        #     ax=ax, alpha=0.5, bins=20, density=density, color="orange"
        # )
        # pd.Series(candidates_in_score, name="in_score").hist(
        #     ax=ax, alpha=0.5, bins=20, density=density, color="blue"
        # )
        # plt.show()

        return ranking
