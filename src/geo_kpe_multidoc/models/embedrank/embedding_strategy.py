import os
from itertools import chain
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import torch
from keybert.backend._base import BaseEmbedder
from loguru import logger

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.base_KP_model import _search_mentions


class CandidateEmbeddingStrategy(Protocol):
    def candidate_embeddings(self, model, doc: Document):
        ...

    def set_global_attention(self, model, doc: Document):
        ...


class OutContextMentionsEmbedding:
    def __init__(self, add_query_prefix=False, **kwargs) -> None:
        self.add_query_prefix = "query: " if add_query_prefix else ""

    def candidate_embeddings(self, model, doc: Document):
        for candidate in doc.candidate_set:
            # TODO: refactor to batch encode
            embds = []
            for mention in doc.candidate_mentions[candidate]:
                mention = self.add_query_prefix + mention
                # TODO: deal with subclassing LongEmbedRank
                if isinstance(model, BaseEmbedder):
                    embd = model.embed(mention)
                else:
                    embd = (
                        model.encode(mention, device=model.device)["sentence_embedding"]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                embds.append(embd)

            doc.candidate_set_embed.append(np.mean(embds, 0))

    def set_global_attention(self, model, doc: Document):
        pass


class OutContextEmbedding:
    def __init__(self, add_query_prefix=False, **kwargs) -> None:
        self.add_query_prefix = "query: " if add_query_prefix else ""

    def candidate_embeddings(self, model, doc: Document):
        # TODO: refactor to batch encode
        for candidate in doc.candidate_set:
            candidate = self.add_query_prefix + candidate
            if isinstance(model, BaseEmbedder):
                embd = model.embed(candidate)
            else:
                embd = (
                    self.model.encode(candidate, device=model.device)[
                        "sentence_embedding"
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

            doc.candidate_set_embed.append(embd)


class InContextEmbeddings:
    def __init__(self, add_query_prefix=False, **kwargs) -> None:
        self.add_query_prefix = "query: " if add_query_prefix else ""

    def candidate_embeddings(self, model, doc: Document):
        # DEBUG:
        # doc.candidate_mentions_not_found = set()

        for candidate in doc.candidate_set:
            mentions_positions = _search_mentions(
                model, doc.candidate_mentions[candidate], doc.token_ids
            )

            # backoff procedure, if mentions not found.
            # If this form is not present in token ids (remember max 4096), fallback to
            # embedding without context. Can happen that tokenization gives different
            # input_ids and the candidate form is not found in document input_ids.
            # candidate is beyond max position for emdedding
            # return a non-contextualized embedding.
            if len(mentions_positions) == 0:
                embds = []
                for mention in doc.candidate_mentions[candidate]:
                    q_mention = self.add_query_prefix + mention

                    if isinstance(model, BaseEmbedder):
                        embd = model.embed(q_mention)
                    else:
                        embd = (
                            model.encode(q_mention, device=model.device)[
                                "sentence_embedding"
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    embds.append(embd)

                # doc.candidate_mentions_not_found.add(candidate)
                logger.debug(
                    f"Candidate {candidate} - mentions not found: {doc.candidate_mentions[candidate]}"
                )
            else:
                _, embed_dim = doc.token_embeddings.size()
                embds = torch.empty(size=(len(mentions_positions), embed_dim))
                for i, occurrence in enumerate(mentions_positions):
                    embds[i] = torch.mean(doc.token_embeddings[occurrence, :], dim=0)

                embds = embds.numpy()

            doc.candidate_set_embed.append(np.mean(embds, 0))

    def set_global_attention(self, model, doc: Document):
        pass


class InContextPlusClsEmbeddings:
    """Compute each candidate mention embedding as the mean of each token in the mention,
    and also the cls token and the last token of the document embedding
    """

    def __init__(self, add_query_prefix=False, **kwargs) -> None:
        self.add_query_prefix = "query: " if add_query_prefix else ""

    def set_global_attention(self, model, doc: Document):
        pass

    def candidate_embeddings(self, model, doc: Document):
        for candidate in doc.candidate_set:
            mentions_positions = _search_mentions(
                model, doc.candidate_mentions[candidate], doc.token_ids
            )

            # backoff procedure, if mentions not found.
            if len(mentions_positions) == 0:
                embds = []
                for mention in doc.candidate_mentions[candidate]:
                    q_mention = self.add_query_prefix + mention

                    if isinstance(model, BaseEmbedder):
                        embd = model.embed(q_mention)
                    else:
                        embd = (
                            model.encode(q_mention, device=model.device)[
                                "sentence_embedding"
                            ]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    embds.append(embd)

                logger.debug(
                    f"Candidate {candidate} - mentions not found: {doc.candidate_mentions[candidate]}"
                )
            else:
                _, embed_dim = doc.token_embeddings.size()
                embds = torch.empty(size=(len(mentions_positions), embed_dim))
                for i, occurrence in enumerate(mentions_positions):
                    embds[i] = torch.mean(
                        doc.token_embeddings[(0, *occurrence, -1), :], dim=0
                    )

                embds = embds.numpy()

            doc.candidate_set_embed.append(np.mean(embds, 0))


class InAndOutContextEmbeddings:
    def __init__(self, add_query_prefix=False, **kwargs) -> None:
        self.add_query_prefix = "query: " if add_query_prefix else ""

    def set_global_attention(self, model, doc: Document):
        pass

    def candidate_embeddings(self, model, doc: Document):
        # TODO: temp to comparison of out context embeddings vs in context embeddings
        save_embeddings = False
        if save_embeddings:
            candidate_embeddings = {}

        for candidate in doc.candidate_set:
            candidate_mentions_embeddings = []
            for mention in doc.candidate_mentions[candidate]:
                q_mention = self.add_query_prefix + mention
                if isinstance(model, BaseEmbedder):
                    mention_out_of_context_embedding = model.embed(q_mention)
                else:
                    mention_out_of_context_embedding = (
                        model.encode(q_mention, device=model.device)[
                            "sentence_embedding"
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                mention_positions = _search_mentions(model, [mention], doc.token_ids)
                if len(mention_positions) == 0:
                    # backoff procedure, if mention is not found.
                    candidate_mentions_embeddings.append(
                        mention_out_of_context_embedding
                    )

                    # TODO: temp to comparison of out context embeddings vs in context embeddings
                    if save_embeddings:
                        candidate_embeddings[mention] = {
                            "out_context": mention_out_of_context_embedding,
                            "in_context": [],
                        }

                else:
                    _, embed_dim = doc.token_embeddings.size()
                    embds = torch.empty(size=(len(mention_positions), embed_dim))
                    for i, occurrence in enumerate(mention_positions):
                        embds[i] = torch.mean(
                            doc.token_embeddings[occurrence, :], dim=0
                        )
                    embds = embds.numpy()
                    mention_in_context_embedding = np.mean(embds, 0)

                    # TODO: temp to comparison of out context embeddings vs in context embeddings
                    if save_embeddings:
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
        if save_embeddings:
            filename = os.path.join(
                GEO_KPE_MULTIDOC_CACHE_PATH,
                "temp_embeddings",
                f"{doc.dataset}-{doc.id}.pkl",
            )
            Path(filename).parent.mkdir(exist_ok=True, parents=True)
            joblib.dump(candidate_embeddings, filename)


class GlobalAttentionCandidateStrategy(InContextEmbeddings):
    def __init__(self, add_query_prefix=False, **kwargs) -> None:
        super().__init__(add_query_prefix, **kwargs)

    def set_global_attention(self, model, doc: Document):
        mentions = []
        for candidate in doc.candidate_set:
            # mentions_positions, _ = self._search_mentions(doc, candidate)
            mentions_positions = _search_mentions(
                model, doc.candidate_mentions[candidate], doc.token_ids
            )
            if len(mentions_positions) > 0:
                # candidate mentions where found in document token_ids
                mentions.extend(mentions_positions)
        mentions = tuple(set(chain(*mentions)))
        logger.debug(f"Global Attention in {len(mentions)} tokens")
        doc.global_attention_mask[:, mentions] = 1


class GlobalAttentionDilatedStrategy(InContextEmbeddings):
    def __init__(self, dilation: int = 128, **kwargs) -> None:
        self.dilation = dilation
        super().__init__(**kwargs)

    def set_global_attention(self, model, doc: Document):
        input_size = doc.global_attention_mask.size(1)
        indices = torch.arange(0, input_size, self.dilation)
        doc.global_attention_mask.index_fill_(1, indices, 1)


STRATEGIES = {
    "no_context": OutContextEmbedding,
    "mentions_no_context": OutContextMentionsEmbedding,
    "in_context": InContextEmbeddings,
    "in_context_plus_cls": InContextPlusClsEmbeddings,
    "in_n_out_context": InAndOutContextEmbeddings,
    "global_attention": GlobalAttentionCandidateStrategy,
    "global_attention_dilated": GlobalAttentionDilatedStrategy,
}
