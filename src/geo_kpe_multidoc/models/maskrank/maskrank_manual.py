import re
from operator import itemgetter
from time import time
from typing import Callable, Dict, List, Optional, Protocol, Set, Tuple

import numpy as np
import torch
from loguru import logger
from nltk.stem import PorterStemmer
from nltk.stem.api import StemmerI
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
from geo_kpe_multidoc.models.candidate_extract.candidate_extract_model import (
    KPECandidateExtractionModel,
)
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy
from geo_kpe_multidoc.models.sentence_embedder import (
    BigBirdSentenceEmbedder,
    LongformerSentenceEmbedder,
    NystromformerSentenceEmbedder,
    SentenceEmbedder,
)


class CandidateMaskEmbeddingStrategy(Protocol):
    def candidate_embeddings(self, model, doc: Document):
        ...


class MaskFirst:
    occurrences = 1

    def candidate_embeddings(self, model, doc: Document):
        # if cand_mode == "MaskFirst" or cand_mode == "MaskAll":
        #     occurences = 1 if cand_mode == "MaskFirst" else 0
        # TODO: does not work (candidate is not found)

        escaped_docs = []

        for _, mentions in doc.candidate_mentions.items():
            text = doc.raw_text
            for mention in mentions:
                if mention not in text:
                    logger.debug(f"Mention '{mention}' - not in text")
                text = text.replace(mention, "<mask>", self.occurrences)
                if self.occurrences == 1:
                    break

            escaped_docs.append(text)

        embd = model.encode_batch(escaped_docs)
        doc.candidate_set_embed = [embd[i] for i in range(len(embd))]


class MaskAll(MaskFirst):
    occurrences = -1

    # escaped_docs = [
    #     re.sub(re.escape(candidate), "<mask>", doc.raw_text, occurences)
    #     for candidate in doc.candidate_set
    # ]
    # doc.candidate_set_embed = self.model.embed(escaped_docs)


class MaskHighest:
    def candidate_embeddings(self, model, doc: Document):
        for candidate in doc.candidate_set:
            candidate = re.escape(candidate)
            candidate_embeds = []

            for match in re.finditer(candidate, doc.raw_text):
                masked_text = f"{doc.raw_text[:match.span()[0]]}<mask>{doc.raw_text[match.span()[1]:]}"
                candidate_embeds.append(model.encode(masked_text))
                # if attention == "global_attention":
                #     candidate_embeds.append(self._embed_global(masked_text))
                # else:
                #     candidate_embeds.append(self.model.embed(masked_text))

        doc.candidate_set_embed.append(candidate_embeds)


class MaskSubset(CandidateMaskEmbeddingStrategy):
    def candidate_embeddings(self, model, doc: Document):
        doc.candidate_set = sorted(
            doc.candidate_set, reverse=True, key=lambda x: len(x)
        )
        seen_candidates = {}

        for candidate in doc.candidate_set:
            prohibited_pos = []
            len_candidate = len(candidate)
            for prev_candidate in seen_candidates:
                if len_candidate == len(prev_candidate):
                    break

                elif candidate in prev_candidate:
                    prohibited_pos.extend(seen_candidates[prev_candidate])

            pos = [
                (match.span()[0], match.span()[1])
                for match in re.finditer(re.escape(candidate), doc.raw_text)
            ]

            seen_candidates[candidate] = pos
            subset_pos = []
            for p in pos:
                subset_flag = True
                for prob in prohibited_pos:
                    if p[0] >= prob[0] and p[1] <= prob[1]:
                        subset_flag = False
                        break
                if subset_flag:
                    subset_pos.append(p)

            masked_doc = doc.raw_text
            for i in range(len(subset_pos)):
                masked_doc = f"{masked_doc[:(subset_pos[i][0] + i*(len_candidate - 5))]}<mask>{masked_doc[subset_pos[i][1] + i*(len_candidate - 5):]}"

            doc.candidate_set_embed.append(model.embed(masked_doc))


class LongformerMaskRank(BaseKPModel):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        tagger: POS_tagger_spacy,
        device=None,
        name="",
    ):
        # TODO: init super class
        self.candidate_selection_model = KPECandidateExtractionModel(tagger=tagger)
        self.counter = 1

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("EmbedRankManual use pytorch device: {}".format(device))
            # if torch.cuda.device_count() > 1:
            #     device = "cuda" if torch.cuda.is_available() else "cpu"
            #     logger.info("EmbedRankManual use pytorch device: {}".format(device))
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
                f'Model of type "{model.__class__.__name__}" not supported for SentenceEmbedder.'
            )

        self.model: SentenceEmbedder = embedder_cls(model, tokenizer)
        self.name = f"MaskRank_{name}"

    def _embed_doc(
        self,
        doc: Document,
        stemmer: Callable = None,
        doc_mode: str = "",
        post_processing: List[str] = [],
        output_attentions=False,
    ) -> np.ndarray:
        """
        Method that embeds the document, having several modes according to usage.
        The default value just embeds the document normally.
        """
        doc_embeddings = self.model.encode(
            doc.raw_text,
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
        self, doc_embed, candidate_set_embed, candidate_set, top_n: int = -1, **kwargs
    ):
        """
        This method is key for each ranking model.
        Here the ranking heuritic is applied according to model definition.

        MaskRank selects the candidates that have embeddings of masked document form far from emedings of the original document.
        Looking for 1 - similarity.
        """
        cand_mode = kwargs.get("cand_mode", "MaskAll")
        top_n = len(candidate_set) if top_n == -1 else top_n

        doc_sim = []

        doc_embed = doc_embed.reshape(1, -1)
        # TODO: simplify cand_mode if to test only MaskHighest
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
        top_n: int = 5,
        min_len: int = 5,
        stemmer: Callable = None,
        **kwargs,
    ) -> List[Tuple]:
        cand_mode = kwargs.get("cand_mode", "MaskAll")
        attention = kwargs.get("global_attention", "global_attention")

        # t = time()
        # doc.doc_embed = self.embed_doc(doc, stemmer)
        # logger.info(f"Embed Doc in {time() -  t:.2f}s")

        # t = time()
        self.embed_candidates(doc, stemmer, cand_mode, attention)
        # logger.info(f"Embed Candidates in {time() -  t:.2f}s")

        return self._rank_candidates(
            doc.doc_embed, doc.candidate_set_embed, doc.candidate_set, top_n, **kwargs
        )

    def embed_candidates(
        self,
        doc: Document,
        stemmer: Callable = None,
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
        t = time()
        doc.doc_embed = self._embed_doc(doc)
        logger.debug(f"Embed Doc in {time() -  t:.2f}s")

        doc.candidate_set_embed = []

        strategies: dict[str, CandidateMaskEmbeddingStrategy] = {
            "MaskFirst": MaskFirst,
            "MaskAll": MaskAll,
            "MaskHighest": MaskHighest,
            "MaskSubset": MaskSubset,
        }

        strategy = strategies[cand_mode]

        t = time()
        strategy().candidate_embeddings(self.model, doc)
        logger.debug(f"Embed Candidates in {time() -  t:.2f}s")
