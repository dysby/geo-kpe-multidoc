from operator import itemgetter
from time import time
from typing import List, Tuple

import numpy as np
import torch
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    BigBirdModel,
    LongformerModel,
    NystromformerModel,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.base_KP_model import BaseKPModel
from geo_kpe_multidoc.models.maskrank.mask_strategy import (
    CandidateMaskEmbeddingStrategy,
    MaskAll,
    MaskFirst,
    MaskHighest,
    MaskSubset,
)
from geo_kpe_multidoc.models.sentence_embedder import (
    BigBirdSentenceEmbedder,
    LongformerSentenceEmbedder,
    NystromformerSentenceEmbedder,
    SentenceEmbedder,
)


class LongformerMaskRank(BaseKPModel):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        candidate_selection_model,
        device=None,
        name="",
        candidate_embedding_strategy: str = "MaskAll",
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
                f'Model of type "{model.__class__.__name__}" not supported for SentenceEmbedder'
            )

        self.model: SentenceEmbedder = embedder_cls(model, tokenizer)

        strategies: dict[str, CandidateMaskEmbeddingStrategy] = {
            "MaskFirst": MaskFirst,
            "MaskAll": MaskAll,
            "MaskHighest": MaskHighest,
            "MaskSubset": MaskSubset,
        }

        # TODO: Add support for e5 type models that require "query: " prefixed text.
        self.add_query_prefix = (
            "query: " if kwargs.get("add_query_prefix") else ""
        )  # for intfloat/multilingual-e5-* models

        self.strategy = strategies[candidate_embedding_strategy]()

        self.name = f"MaskRank-{self.strategy.__class__.__name__}-{name}"

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
            doc_embeddings["attentions"].detach.cpu() if output_attentions else None
        )

        return doc_embeddings["sentence_embedding"].detach().cpu().numpy()

    def _rank_candidates(
        self,
        doc: Document,
        doc_embed,
        candidate_set_embed,
        candidate_set,
        top_n: int = -1,
        **kwargs,
    ):
        """
        This method is key for each ranking model.
        Here the ranking heuritic is applied according to model definition.

        MaskRank selects the candidates that have embeddings of masked document form far
        from emedings of the source document.
        Looking for 1 - similarity.
        """
        cand_mode = kwargs.get("cand_mode", "MaskAll")
        top_n = len(candidate_set) if top_n == -1 else top_n

        doc_sim = []

        doc_embed = doc_embed.reshape(1, -1)
        if cand_mode != "MaskHighest":
            doc_sim = cosine_similarity(candidate_set_embed, doc_embed)
        else:
            # cand_mode == "MaskHighest":
            for mask_cand_occur in candidate_set_embed:
                if mask_cand_occur != []:
                    doc_sim.append(
                        [cosine_similarity(mask_cand_occur, doc_embed).min()]
                    )
                else:
                    doc_sim.append(np.array([1.0]))

        # TODO: refactor candidate scores sorting
        candidate_score = sorted(
            [(candidate_set[i], 1.0 - doc_sim[i][0]) for i in range(len(doc_sim))],
            reverse=True,
            key=itemgetter(1),
        )

        return candidate_score[:top_n], [candidate[0] for candidate in candidate_score]

    def top_n_candidates(
        self,
        doc: Document,
        candidate_list,
        positions=None,
        top_n: int = 5,
        **kwargs,
    ) -> List[Tuple]:
        cand_mode = kwargs.get("cand_mode", "MaskAll")
        attention = kwargs.get("global_attention", "global_attention")

        # t = time()
        # doc.doc_embed = self.embed_doc(doc, stemmer)
        # logger.info(f"Embed Doc in {time() -  t:.2f}s")

        # t = time()
        self.embed_candidates(doc, cand_mode, attention)
        # logger.info(f"Embed Candidates in {time() -  t:.2f}s")

        return self._rank_candidates(
            doc,
            doc.doc_embed,
            doc.candidate_set_embed,
            doc.candidate_set,
            top_n,
            **kwargs,
        )

    def embed_candidates(
        self,
        doc: Document,
        cand_mode: str = "MaskAll",
        attention: str = "",
    ):
        """
        Method that embeds the current candidate set, having several modes according to usage.
            cand_mode
            | MaskFirst only masks the first occurence of a candidate;
            | MaskAll masks all occurences of said candidate

            The default value is MaskAll.
        """
        start = time()
        doc.doc_embed = self._embed_doc(doc)
        logger.debug(f"Embed Doc in {time() -  start:.2f}s")

        doc.candidate_set_embed = []

        start = time()
        self.strategy.candidate_embeddings(self.model, doc)
        logger.debug(f"Embed Candidates in {time() -  start:.2f}s")
