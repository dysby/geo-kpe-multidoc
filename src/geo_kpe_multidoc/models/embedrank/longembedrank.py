from typing import Callable, List

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
from geo_kpe_multidoc.models.candidate_extract.candidate_extract_model import (
    KPECandidateExtractionModel,
)
from geo_kpe_multidoc.models.embedrank.embedding_strategy import (
    STRATEGIES,
    CandidateEmbeddingStrategy,
    InContextEmbeddings,
)
from geo_kpe_multidoc.models.embedrank.embedrank_model import EmbedRank
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy
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
        tagger: POS_tagger_spacy,
        device=None,
        name="",
        candidate_embedding_strategy: str = "",
        **kwargs,
    ):
        # TODO: init super class
        self.candidate_selection_model = KPECandidateExtractionModel(tagger=tagger)
        self.counter = 1

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("LongEmbedRank use pytorch device: {}".format(device))
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

        # strategies = {
        #     "no_context": OutContextEmbedding,
        #     "mentions_no_context": OutContextMentionsEmbedding,
        #     "in_context": InContextEmbeddings,
        #     "in_n_out_context": InAndOutContextEmbeddings,
        # }
        # TODO: deal with global_attention and global_attention_dilated
        strategy = STRATEGIES.get(candidate_embedding_strategy, InContextEmbeddings)
        self.candidate_embedding_strategy: CandidateEmbeddingStrategy = strategy()
        logger.info(
            f"Initialize EmbedRank w/ {self.candidate_embedding_strategy.__class__.__name__}"
        )
        # TODO: Add support for e5 type models that require "query: " prefixed text.
        self.add_query_prefix = (
            "query: " if kwargs.get("add_query_prefix", False) else ""
        )  # for intfloat/multilingual-e5-* models
        self.whitening = kwargs.get("whitening", False)

    def _embed_doc(
        self,
        doc: Document,
        stemmer: Callable = None,
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
