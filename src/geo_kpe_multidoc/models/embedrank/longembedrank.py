from typing import List

import numpy as np
import torch
from loguru import logger
from transformers import (
    BigBirdModel,
    LongformerModel,
    NystromformerModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.embedrank.embedding_strategy import (
    STRATEGIES,
    CandidateEmbeddingStrategy,
    InContextEmbeddings,
)
from geo_kpe_multidoc.models.embedrank.embedrank_model import EmbedRank
from geo_kpe_multidoc.models.sentence_embedder import (
    BigBirdSentenceEmbedder,
    LongformerSentenceEmbedder,
    NystromformerSentenceEmbedder,
    SentenceEmbedder,
)


class LongEmbedRank(EmbedRank):
    """
    EmbedRank variant for Longformer model
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        candidate_selection_model,
        device=None,
        name="",
        candidate_embedding_strategy: str = "mentions_no_context",
        **kwargs,
    ):
        # TODO: init super class
        self.candidate_selection_model = candidate_selection_model
        self.counter = 1

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"LongEmbedRank use pytorch device: {device}")
            # if torch.cuda.device_count() > 1:
            #     device = "cuda" if torch.cuda.is_available() else "cpu"
            #     logger.info("LongEmbedRank use pytorch device: {}".format(device))
            #     model = nn.DataParallel(model)

        model.to(device)
        self.device = device

        if isinstance(model, LongformerModel):
            embedder_cls = LongformerSentenceEmbedder
        elif isinstance(model, BigBirdModel):
            embedder_cls = BigBirdSentenceEmbedder
        elif isinstance(model, NystromformerModel):
            embedder_cls = NystromformerSentenceEmbedder
        else:
            raise ValueError(
                f"Model of type '{model.__class__.__name__}' not supported for SentenceEmbedder."
            )

        self.model: SentenceEmbedder = embedder_cls(model, tokenizer)
        self.name = f"LongEmbedRank_{name}"

        # TODO: Add support for e5 type models that require "query: " prefixed text.
        self.add_query_prefix = (
            "query: " if kwargs.get("add_query_prefix") else ""
        )  # for intfloat/multilingual-e5-* models

        dilation = 128
        if "dilated" in candidate_embedding_strategy:
            dilation = int("".join(filter(str.isdigit, candidate_embedding_strategy)))
            candidate_embedding_strategy = candidate_embedding_strategy[
                : candidate_embedding_strategy.index("dilated") + len("dilated")
            ]
        strategy = STRATEGIES.get(candidate_embedding_strategy)

        self.candidate_embedding_strategy: CandidateEmbeddingStrategy = strategy(
            add_query_prefix=self.add_query_prefix, dilation=dilation
        )
        logger.info(
            f"Initialize EmbedRank w/ {self.candidate_embedding_strategy.__class__.__name__}"
        )

        # TODO: add score position factor bias like in PromptRank
        self.enable_pos = not kwargs.get("no_position_feature", False)
        self.position_factor = kwargs.get("position_factor", 1.2e8)

        self.whitening = kwargs.get("whitening", False)

    def _embed_doc(
        self,
        doc: Document,
        doc_mode: str = "",
        post_processing: List[str] = None,
        output_attentions=False,
    ) -> np.ndarray:
        """
        Method that embeds the document, having several modes according to usage.
        The default value just embeds the document normally.
        """
        doc_embeddings = self.model.encode(
            self.add_query_prefix + doc.raw_text,
            global_attention_mask=doc.global_attention_mask,
            device=self.device,
            output_attentions=output_attentions,
        )

        doc.token_ids = doc_embeddings["input_ids"].squeeze().tolist()
        doc.token_embeddings = doc_embeddings["token_embeddings"].detach().cpu()
        doc.attention_mask = doc_embeddings["attention_mask"].detach().cpu()
        doc.attentions = (
            doc_embeddings["attentions"].detach().cpu() if output_attentions else None
        )

        return doc_embeddings["sentence_embedding"].detach().cpu().numpy()
