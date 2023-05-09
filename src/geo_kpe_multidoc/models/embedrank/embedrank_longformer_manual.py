from typing import Callable, List

import numpy as np
import torch
from loguru import logger

from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.embedrank.embedrank_model import EmbedRank
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy
from geo_kpe_multidoc.models.sentence_embedder import SentenceEmbedder


class EmbedRankManual(EmbedRank):
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models
    """

    def __init__(self, model, tokenizer, tagger, device=None, name=""):
        self.tagger = POS_tagger_spacy(tagger)
        self.grammar = """  NP: 
        {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}"""
        self.counter = 0

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("EmbedRankManual use pytorch device: {}".format(device))

        model.to(device)
        self.device = device

        self.model = SentenceEmbedder(model, tokenizer)
        self.name = f"EmbedRankManual_{name}"

    def _embed_doc(
        self,
        doc: Document,
        stemmer: Callable = None,
        doc_mode: str = "",
        post_processing: List[str] = [],
    ) -> np.ndarray:
        """
        Method that embeds the document, having several modes according to usage.
        The default value just embeds the document normally.
        """
        doc_embeddings = self.model.encode(
            doc.raw_text,
            global_attention_mask=doc.global_attention_mask,
            device=self.device,
        )

        doc.token_ids = doc_embeddings["input_ids"].squeeze().tolist()
        doc.token_embeddings = doc_embeddings["token_embeddings"].detach().cpu()
        doc.attention_mask = doc_embeddings["attention_mask"].detach().cpu()

        return doc_embeddings["sentence_embedding"].detach().cpu().numpy()
