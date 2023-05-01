import re
from operator import itemgetter
from time import time
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
import torch
from loguru import logger
from nltk import RegexpParser
from nltk.stem import PorterStemmer
from nltk.stem.api import StemmerI
from sklearn.metrics.pairwise import cosine_similarity

from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.base_KP_model import BaseKPModel
from geo_kpe_multidoc.models.pre_processing.language_mapping import (
    choose_lemmatizer,
    choose_tagger,
)
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_punctuation,
    remove_whitespaces,
)


class MaskRank(BaseKPModel):
    """
    Simple class to encapsulate MaskRank functionality.

    Extract candidates and rank by cosine similarity of document embedding with masked document embedding.

    Uses KeyBert backend to retrieve models.
    """

    def __init__(self, model, tagger):
        super().__init__(model)
        self.tagger = POS_tagger_spacy(tagger)
        self.grammar = """  NP: 
        {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}"""
        self.counter = 0

    def update_tagger(self, dataset: str = "") -> None:
        self.tagger = (
            POS_tagger_spacy(choose_tagger(dataset))
            if choose_tagger(dataset) != self.tagger.name
            else self.tagger
        )

    def extract_kp_from_corpus(
        self,
        corpus,
        dataset: str = "DUC",
        top_n: int = 15,
        min_len: int = 5,
        stemming: bool = False,
        lemmatize: bool = False,
        **kwargs,
    ) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality
        """
        self.counter = 0
        self.update_tagger(dataset)

        stemmer = PorterStemmer() if stemming else None
        lemmer = choose_lemmatizer(dataset) if lemmatize else None

        return [
            self.extract_kp_from_doc(doc, top_n, min_len, stemmer, lemmer, **kwargs)
            for doc, _ in corpus
        ]

    def embed_doc(self, doc: Document, stemmer: Callable = None) -> np.ndarray:
        """
        Method that embeds the document.
        """

        # doc_info = model.embed_full(self.raw_text) # encode(documents, show_progress_bar=False, output_value = None)
        doc_embeddings = self.model.embedding_model.encode(
            doc.raw_text, show_progress_bar=False, output_value=None
        )

        doc.token_ids = doc_embeddings["input_ids"].detach().cpu().squeeze().tolist()
        doc.token_embeddings = doc_embeddings["token_embeddings"].detach().cpu()
        doc.attention_mask = doc_embeddings["attention_mask"].detach().cpu()

        return doc_embeddings["sentence_embedding"].detach().cpu().numpy()

    def _embed_global(self, model):
        raise NotImplemented

    def _global_embed_doc(self, model):
        raise NotImplemented

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
        doc.candidate_set_embed = []

        if cand_mode == "MaskFirst" or cand_mode == "MaskAll":
            occurences = 1 if cand_mode == "MaskFirst" else 0

            escaped_docs = [
                re.sub(re.escape(candidate), "<mask>", doc.raw_text, occurences)
                for candidate in doc.candidate_set
            ]
            doc.candidate_set_embed = self.model.embed(escaped_docs)

        elif cand_mode == "MaskHighest":
            for candidate in doc.candidate_set:
                candidate = re.escape(candidate)
                candidate_embeds = []

                for match in re.finditer(candidate, doc.raw_text):
                    masked_text = f"{doc.raw_text[:match.span()[0]]}<mask>{doc.raw_text[match.span()[1]:]}"
                    if attention == "global_attention":
                        candidate_embeds.append(self._embed_global(masked_text))
                    else:
                        candidate_embeds.append(self.model.embed(masked_text))
                doc.candidate_set_embed.append(candidate_embeds)

        elif cand_mode == "MaskSubset":
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
                doc.candidate_set_embed.append(self.model.embed(masked_doc))
        else:
            RuntimeError("cand_mode not set!")

    def extract_candidates(
        self,
        doc: Document,
        min_len: int = 5,
        grammar: str = "",
        lemmer: Callable = None,
        **kwargs,
    ):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document and
        stores the sentences each candidate occurs in
        """

        use_cache = kwargs.get("pos_tag_cache", False)
        candidate_set = set()

        parser = RegexpParser(grammar)
        np_trees = list(parser.parse_sents(doc.tagged_text))

        for i in range(len(np_trees)):
            temp_cand_set = []
            for subtree in np_trees[i].subtrees(filter=lambda t: t.label() == "NP"):
                temp_cand_set.append(" ".join(word for word, tag in subtree.leaves()))

            for candidate in temp_cand_set:
                if len(candidate) > min_len:
                    candidate_set.add(candidate)

        doc.candidate_set = list(candidate_set)

    def embed_n_candidates(
        self, doc: Document, min_len: int, stemmer: Optional[StemmerI] = None, **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Return
        ------
            candidate_set_embed:    np.ndarray of the embeddings for each candidate.
            candicate_set:          List of candidates.
        """
        # TODO: why embed_doc in MaskRank is different, no post_processing or doc_mode
        t = time()
        doc.doc_embed = self.embed_doc(doc, stemmer)
        logger.info(f"Embed Doc in {time() -  t:.2f}s")

        t = time()
        self.embed_candidates(doc, stemmer, cand_mode="MaskAll")
        logger.info(f"Embed Candidates in {time() -  t:.2f}s")

        return doc.candidate_set_embed, doc.candidate_set

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
        if "cand_mode" not in kwargs or kwargs["cand_mode"] != "MaskHighest":
            doc_sim = cosine_similarity(candidate_set_embed, doc_embed)

        elif kwargs["cand_mode"] == "MaskHighest":
            for mask_cand_occur in candidate_set_embed:
                if mask_cand_occur != []:
                    doc_sim.append(
                        [
                            # np.ndarray.min(
                            #     np.absolute(
                            #         cosine_similarity(mask_cand_occur, doc_embed)
                            #     )
                            # )
                            cosine_similarity(mask_cand_occur, doc_embed).min()
                        ]
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
        doc,
        top_n: int = 5,
        min_len: int = 5,
        stemmer: Callable = None,
        **kwargs,
    ) -> List[Tuple]:
        cand_mode = kwargs.get("cand_mode", "MaskAll")
        attention = kwargs.get("global_attention", "global_attention")

        t = time()
        doc.doc_embed = self.embed_doc(doc, stemmer)
        logger.info(f"Embed Doc in {time() -  t:.2f}s")

        t = time()
        self.embed_candidates(doc, stemmer, cand_mode, attention)
        logger.info(f"Embed Candidates in {time() -  t:.2f}s")

        return self._rank_candidates(
            doc.doc_embed, doc.candidate_set_embed, doc.candidate_set, top_n, **kwargs
        )

        # doc_sim = []
        # # TODO: simplify cand_mode if to test only MaskHighest
        # if "cand_mode" not in kwargs or kwargs["cand_mode"] != "MaskHighest":
        #     doc_sim = np.absolute(
        #         cosine_similarity(doc.candidate_set_embed, doc.doc_embed.reshape(1, -1))
        #     )

        # elif kwargs["cand_mode"] == "MaskHighest":
        #     doc_embed = doc.doc_embed.reshape(1, -1)
        #     for mask_cand_occur in doc.candidate_set_embed:
        #         if mask_cand_occur != []:
        #             doc_sim.append(
        #                 [
        #                     np.ndarray.min(
        #                         np.absolute(
        #                             cosine_similarity(mask_cand_occur, doc_embed)
        #                         )
        #                     )
        #                 ]
        #             )
        #         else:
        #             doc_sim.append([1.0])

        # # TODO: refactor candidate scores sorting
        # candidate_score = sorted(
        #     [(doc.candidate_set[i], 1.0 - doc_sim[i][0]) for i in range(len(doc_sim))],
        #     reverse=True,
        #     key=itemgetter(1),
        # )

        # if top_n == -1:
        #     return candidate_score, [candidate[0] for candidate in candidate_score]

        # return candidate_score[:top_n], [candidate[0] for candidate in candidate_score]
