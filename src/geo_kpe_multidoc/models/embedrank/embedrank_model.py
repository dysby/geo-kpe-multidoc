import os
from itertools import chain, pairwise
from operator import itemgetter
from pathlib import Path
from time import time
from typing import Callable, List, Optional, Set, Tuple, Union

import joblib
import numpy as np
import simplemma
import torch
from keybert._mmr import mmr
from keybert.backend._base import BaseEmbedder
from loguru import logger
from nltk.stem.api import StemmerI
from sklearn.metrics.pairwise import cosine_similarity

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.base_KP_model import BaseKPModel, find_occurrences
from geo_kpe_multidoc.models.pre_processing.post_processing_utils import (
    z_score_normalization,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    filter_special_tokens,
    tokenize_hf,
)
from geo_kpe_multidoc.models.sentence_embedder import LongformerSentenceEmbedder


class EmbedRank(BaseKPModel):
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models

    cand_mode:
        {mentions}?{no_context}+            - candidate embeddings from non contextualized form
        {global_attention}+{dilated_(n)}?   - get embeddings using longformer global attention in all doc tokens where a candidate is present.
                                            If dilated mode is selected a sparse pattern is used, considering global attention on every n position.
        mean_in_n_out_context               - add non contextualized form embedding to mentions embeddings and do the average as candidade embedding.
    """

    def __init__(self, model, tagger):
        super().__init__(model, tagger)
        self.counter = 0

    def _embed_doc(
        self,
        doc: Document,
        stemmer: Callable = None,
        doc_mode: str = "",
        post_processing: List[str] = [],
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
                "Global Attention not used in SentenceTransformer API, use EmbedRankManual"
            )

        doc_embeddings = self.model.embedding_model.encode(
            doc.raw_text, show_progress_bar=False, output_value=None
        )

        doc.token_ids = doc_embeddings["input_ids"].squeeze().tolist()
        doc.token_embeddings = doc_embeddings["token_embeddings"].detach().cpu()
        doc.attention_mask = doc_embeddings["attention_mask"].detach().cpu()

        # TODO: longformer attention window 256 with inputs only 128
        # limit_128 = True
        # if limit_128:
        #     doc.token_ids = doc.token_ids[:128]

        return doc_embeddings["sentence_embedding"].detach().cpu().numpy()

    def _search_mentions(self, candidate_mentions, token_ids):
        mentions = []
        # TODO: mention counts for mean_in_n_out_context
        # mentions_counts = []
        for mention in candidate_mentions:
            if isinstance(self.model, BaseEmbedder):
                # original tokenization by KeyBert/SentenceTransformer
                tokenized_candidate = tokenize_hf(mention, self.model)
            else:
                # tokenize via local SentenceEmbedder Class
                tokenized_candidate = self.model.tokenize(mention)

            filt_ids = filter_special_tokens(tokenized_candidate["input_ids"])

            # Should not be Empty after filter
            if filt_ids:
                mentions += find_occurrences(filt_ids, token_ids)
            # mentions_counts.append(len(mentions))

        # mentions_counts = mentions_counts[:1] + [
        #     y - x for x, y in pairwise(mentions_counts)
        # ]
        return mentions  # , mentions_counts

    def _embedding_in_n_out_context(self, doc: Document):
        # TODO: temp to comparison of out context embeddings vs in context embeddings
        candidate_embeddings = dict()

        for candidate in doc.candidate_set:
            candidate_mentions_embeddings = []
            for mention in doc.candidate_mentions[candidate]:
                if isinstance(self.model, BaseEmbedder):
                    mention_out_of_context_embedding = self.model.embed(mention)
                else:
                    mention_out_of_context_embedding = (
                        self.model.encode(mention, device=self.device)[
                            "sentence_embedding"
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                mentions = self._search_mentions([mention], doc.token_ids)
                if len(mentions) == 0:
                    # backoff procedure, if mention is not found.
                    candidate_mentions_embeddings.append(
                        mention_out_of_context_embedding
                    )

                    # TODO: temp to comparison of out context embeddings vs in context embeddings
                    candidate_embeddings[mention] = {
                        "out_context": mention_out_of_context_embedding,
                        "in_context": [],
                    }

                else:
                    _, embed_dim = doc.token_embeddings.size()
                    embds = torch.empty(size=(len(mentions), embed_dim))
                    for i, occurrence in enumerate(mentions):
                        embds[i] = torch.mean(
                            doc.token_embeddings[occurrence, :], dim=0
                        )
                    embds = embds.numpy()
                    mention_in_context_embedding = np.mean(embds, 0)

                    # TODO: temp to comparison of out context embeddings vs in context embeddings
                    candidate_embeddings[mention] = {
                        "out_context": mention_out_of_context_embedding,
                        "in_context": [embds[i] for i in range(embds.shape[0])],
                    }

                    candidate_mentions_embeddings.append(
                        np.mean(
                            [
                                mention_out_of_context_embedding,
                                mention_in_context_embedding,
                            ],
                            0,
                        )
                    )

            doc.candidate_set_embed.append(np.mean(candidate_mentions_embeddings, 0))

        # TODO: temp to comparison of out context embeddings vs in context embeddings
        filename = os.path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            "temp_embeddings",
            f"{doc.dataset}-{doc.id}.pkl",
        )
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(candidate_embeddings, filename)

    def _embedding_in_context(self, doc: Document, cand_mode: str = ""):
        if "mean_in_n_out_context" in cand_mode:
            self._embedding_in_n_out_context(doc)
            return

        # TODO: temp to comparison of out context embeddings vs in context embeddings
        candidate_embeddings = dict()

        for candidate in doc.candidate_set:
            mentions = self._search_mentions(
                doc.candidate_mentions[candidate], doc.token_ids
            )

            # backoff procedure, if mentions not found.
            if len(mentions) == 0:
                # If this form is not present in token ids (remember max 4096), fallback to embedding without context.
                # Can happen that tokenization gives different input_ids and the candidate form is not found in document
                # input_ids.
                # candidate is beyond max position for emdedding
                # return a non-contextualized embedding.
                # TODO: candidate -> mentions
                for mention in doc.candidate_mentions[candidate]:
                    embds = []
                    if isinstance(self.model, BaseEmbedder):
                        embd = self.model.embed(mention)
                    else:
                        embd = (
                            self.model.encode(mention, device=self.device)[
                                "sentence_embedding"
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    embds.append(embd)

                # TODO: temp to comparison of out context embeddings vs in context embeddings
                candidate_embeddings[candidate] = {"out_context": embds}

                doc.candidate_set_embed.append(np.mean(embds, 0))
                # TODO: problem with original 'andrew - would' vs PoS extracted 'andrew-would'
                logger.debug(
                    f"Candidate {candidate} - mentions not found: {doc.candidate_mentions[candidate]}"
                )
            else:
                _, embed_dim = doc.token_embeddings.size()
                embds = torch.empty(size=(len(mentions), embed_dim))
                for i, occurrence in enumerate(mentions):
                    embds[i] = torch.mean(doc.token_embeddings[occurrence, :], dim=0)

                embds = embds.numpy()
                # TODO: temp to comparison of out context embeddings vs in context embeddings
                candidate_embeddings[candidate] = {
                    "in_context": [embds[i] for i in range(embds.shape[0])]
                }

                if "mean_in_n_out_context" in cand_mode:
                    for mention in doc.candidate_mentions[candidate]:
                        if isinstance(self.model, BaseEmbedder):
                            mention_out_of_context_embedding = self.model.embed(mention)
                        else:
                            mention_out_of_context_embedding = (
                                self.model.encode(mention, device=self.device)[
                                    "sentence_embedding"
                                ]
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        embds = np.concatenate(
                            [embds, [mention_out_of_context_embedding]]
                        )

                        # TODO: temp to comparison of out context embeddings vs in context embeddings
                        candidate_embeddings[candidate].setdefault(
                            "out_context", []
                        ).append(mention_out_of_context_embedding)

                doc.candidate_set_embed.append(np.mean(embds, 0))

        # TODO: temp to comparison of out context embeddings vs in context embeddings
        filename = os.path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            "temp_embeddings",
            f"{doc.dataset}-{doc.id}.pkl",
        )
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(candidate_embeddings, filename)

    def _embedding_out_context(self, doc: Document, cand_mode: str = ""):
        logger.debug(f"Getting candidate embeddings without context.")

        if "mentions" in cand_mode:
            for candidate in doc.candidate_set:
                for mention in doc.candidate_mentions[candidate]:
                    embds = []
                    if isinstance(self.model, BaseEmbedder):
                        embd = self.model.embed(mention)
                    else:
                        embd = (
                            self.model.encode(mention, device=self.device)[
                                "sentence_embedding"
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    embds.append(embd)

                doc.candidate_set_embed.append(np.mean(embds, 0))
        else:
            for candidate in doc.candidate_set:
                if isinstance(self.model, BaseEmbedder):
                    embd = self.model.embed(candidate)
                else:
                    embd = (
                        self.model.encode(candidate, device=self.device)[
                            "sentence_embedding"
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                doc.candidate_set_embed.append(embd)

    def _aggregate_candidate_mention_embeddings(
        self,
        doc: Document,
        stemmer: Optional[StemmerI] = None,
        cand_mode: str = "",
        post_processing: List[str] = [],
    ):
        """
        Method that embeds the current candidate set, having several modes according to usage.
        The default value just embeds candidates directly.

        TODO: deal with subclassing EmbedRankManual
        """
        # TODO: keep this init?
        doc.candidate_set_embed = []

        # special simple case
        if "no_context" in cand_mode:
            self._embedding_out_context(doc, cand_mode)
        else:
            self._embedding_in_context(doc, cand_mode)

        if "z_score" in post_processing:
            # TODO: Why z_score_normalization by space split?
            doc.candidate_set_embed = z_score_normalization(
                doc.candidate_set_embed, doc.raw_text, self.model
            )

    def _set_global_attention_on_candidates(self, doc: Document):
        mentions = []
        for candidate in doc.candidate_set:
            # mentions_positions, _ = self._search_mentions(doc, candidate)
            mentions_positions = self._search_mentions(
                doc.candidate_mentions[candidate], doc.token_ids
            )
            if len(mentions_positions) > 0:
                # candidate mentions where found in document token_ids
                mentions.extend(mentions_positions)
        mentions = tuple(set(chain(*mentions)))
        logger.debug(f"Global Attention in {len(mentions)} tokens")
        doc.global_attention_mask[:, mentions] = 1

    def embed_candidates(
        self, doc: Document, stemmer, **kwargs
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
            tokenized_doc = tokenize_hf(doc.raw_text, self.model)
        elif isinstance(self.model, LongformerSentenceEmbedder):
            # tokenize via local SentenceEmbedder Class
            tokenized_doc = self.model.tokenize(
                doc.raw_text,
                padding=True,
                pad_to_multiple_of=self.model.attention_window,
            )
        else:
            # BIGBIRD
            # tokenize via local SentenceEmbedder Class
            tokenized_doc = self.model.tokenize(doc.raw_text)

        doc.token_ids = tokenized_doc["input_ids"].squeeze().tolist()
        doc.global_attention_mask = torch.zeros(tokenized_doc["input_ids"].shape)
        doc.global_attention_mask[:, 0] = 1  # CLS token

        if "global_attention" in cand_mode:
            if "dilated" in cand_mode:
                dilation = int("".join(filter(str.isdigit, cand_mode)))
                input_size = doc.global_attention_mask.size(1)
                indices = torch.arange(0, input_size, dilation)
                doc.global_attention_mask.index_fill_(1, indices, 1)
            else:
                self._set_global_attention_on_candidates(doc)

        output_attentions = "attention_rank" in cand_mode

        t = time()
        doc.doc_embed = self._embed_doc(
            doc, stemmer, doc_mode, post_processing, output_attentions=output_attentions
        )
        logger.debug(f"Embed Doc in {time() -  t:.2f}s")

        t = time()
        self._aggregate_candidate_mention_embeddings(
            doc, stemmer, cand_mode, post_processing
        )
        logger.debug(f"Embed Candidates in {time() -  t:.2f}s")

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
        TODO: why does not have MMR? - copied mmr from top_n_candidates
        """
        mmr_mode = kwargs.get("mmr", False)
        mmr_diversity = kwargs.get("mmr_diversity", 0.8)
        top_n = len(candidate_set) if top_n == -1 else top_n

        doc_sim = []
        if mmr_mode:
            assert mmr_diversity > 0
            assert mmr_diversity < 1

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
                    (candidate, candidate_doc_sim)
                    for (candidate, candidate_doc_sim) in zip(candidate_set, doc_sim)
                ],
                # [(candidate_set[i], doc_sim[i][0]) for i in range(len(doc_sim))],
                reverse=True,
                key=itemgetter(1),
            )

        return candidate_score[:top_n], candidate_set

    def top_n_candidates(
        self,
        doc: Document,
        top_n: int = 5,
        min_len: int = 5,
        stemmer: Callable = None,
        **kwargs,
    ) -> List[Tuple]:
        doc_mode = kwargs.get("doc_mode", "")
        cand_mode = kwargs.get("cand_mode", "")
        post_processing = kwargs.get("post_processing", [""])
        # TODO: remove? use_cache = kwargs.get("embed_memory", False)
        # if cand_mode != "" and cand_mode != "AvgContext":
        #     logger.error(f"Getting Embeddings for word sentence (not used?)")
        #     self.embed_sents_words(doc, stemmer, use_cache)

        self.embed_candidates(doc, stemmer, **kwargs)

        ranking = self._rank_candidates(
            doc.doc_embed, doc.candidate_set_embed, doc.candidate_set, top_n, **kwargs
        )

        return ranking
        # doc_sim = []
        # if mmr_mode:
        #     assert mmr_diversity > 0
        #     assert mmr_diversity < 1
        #     valid_top_n = len(doc.candidate_set)
        #     if top_n > 0:
        #         valid_top_n = (
        #             len(doc.candidate_set) if len(doc.candidate_set) < top_n else top_n
        #         )
        #     doc_sim = mmr(
        #         doc.doc_embed.reshape(1, -1),
        #         doc.candidate_set_embed,
        #         doc.candidate_set,
        #         valid_top_n,
        #         diversity=mmr_diversity,
        #     )
        # else:
        #     doc_sim = np.absolute(
        #         cosine_similarity(doc.candidate_set_embed, doc.doc_embed.reshape(1, -1))
        #     )

        # candidate_score = sorted(
        #     [(doc.candidate_set[i], doc_sim[i][0]) for i in range(len(doc_sim))],
        #     reverse=True,
        #     key=lambda x: x[1],
        # )

        # if top_n == -1:
        #     return candidate_score, [candidate[0] for candidate in candidate_score]

        # return candidate_score[:top_n], [candidate[0] for candidate in candidate_score]
