from time import time
from typing import Callable, List, Optional, Set, Tuple, Union

import numpy as np
import simplemma
import torch
from keybert._mmr import mmr
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
from geo_kpe_multidoc.models.pre_processing.post_processing_utils import (
    z_score_normalization,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    filter_token_ids,
    lemmatize,
    tokenize_hf,
)
from geo_kpe_multidoc.utils.IO import read_from_file


class EmbedRank(BaseKPModel):
    """
    Simple class to encapsulate EmbedRank functionality. Uses
    the KeyBert backend to retrieve models
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

    def extract_mdkpe_embeds(
        self, doc: Document, top_n, min_len, stemmer=None, lemmer=None, **kwargs
    ) -> Tuple[Document, List[Tuple], List[str]]:
        use_cache = kwargs.get("pos_tag_memory", False)

        self.pos_tag_doc(
            doc=doc,
            stemming=None,
            memory=use_cache,
        )
        self.extract_candidates(doc, min_len, self.grammar, lemmer)

        cand_embeds, candidate_set = self.embed_n_candidates(
            doc, min_len, stemmer, **kwargs
        )

        logger.info(f"Document #{self.counter} processed")
        self.counter += 1
        torch.cuda.empty_cache()

        return (doc, cand_embeds, candidate_set)

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

    def embed_sents_words(self, doc, stemmer: Optional[StemmerI] = None, memory=False):
        """
        Embed each word in the sentence by it self
        Non-conterxtualized embedding.
        Embed_sent_words (Use only on non-contextualized candicate embeding mode, not used).
        TODO: validate, not used, we always want contexttualized embeddings of the candidades.
        TODO: correct cache/memory usage
        """
        if not memory:
            # self.doc_sents_words_embed = []
            doc.doc_sents_words_embed = [
                self.model.embed(stemmer.stem(word)) for word in doc.doc_sent_words
            ]
            # for i in range(len(self.doc_sents_words)):
            #     self.doc_sents_words_embed.append(
            #         self.embed(stemmer.stem(doc.doc_sents_words[i]))
            #         if stemmer
            #         else model.embed(doc.doc_sents_words[i])
            #     )
        else:
            doc.doc_sents_words_embed = read_from_file(f"{memory}/{doc.id}")

    def embed_doc(
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

        # Check why document text is mutated to lower:
        # Used after POS_TAGGING,
        # at 1st stage document POS Tagging uses normal text including capital letters,
        # but later document handling will use only lowered text, embedings, and such.
        doc.raw_text = doc.raw_text.lower()
        doc_embedings = self.model.embedding_model.encode(
            doc.raw_text, show_progress_bar=False, output_value=None
        )

        doc.token_ids = doc_embedings["input_ids"].squeeze().tolist()
        doc.token_embeddings = doc_embedings["token_embeddings"]
        doc.attention_mask = doc_embedings["attention_mask"]

        return doc_embedings["sentence_embedding"].detach().numpy()

    def global_embed_doc(self, doc):
        raise NotImplemented

    def embed_candidates(
        self,
        doc: Document,
        stemmer: Optional[StemmerI] = None,
        cand_mode: str = "",
        post_processing: List[str] = [],
    ):
        """
        Method that embeds the current candidate set, having several modes according to usage.
            The default value just embeds candidates directly.
        """
        # TODO: keep this init?
        doc.candidate_set_embed = []

        for candidate in doc.candidate_set:
            candidate_embeds = []

            for mention in doc.candidate_mentions[candidate]:
                tokenized_candidate = tokenize_hf(mention, self.model)
                filt_ids = filter_token_ids(tokenized_candidate["input_ids"])

                cand_len = len(filt_ids)

                for i in range(len(doc.token_ids)):
                    if (
                        filt_ids[0] == doc.token_ids[i]
                        and filt_ids == doc.token_ids[i : i + cand_len]
                    ):
                        candidate_embeds.append(
                            np.mean(
                                doc.token_embeddings[i : i + cand_len].detach().numpy(),
                                axis=0,
                            )
                        )
                        # What is global_attention mode?
                        # Used for custom global attention mask of the longformer.
                        # Set attention mask = 1 at all token positions where this candidate is mentioned.
                        # TODO: Use attention_mask for computing a new doc embeding vector?
                        if cand_mode == "global_attention":
                            for j in range(i, i + cand_len):
                                doc.attention_mask[j] = 1

            if candidate_embeds == []:
                # candidate is beyond max position for emdedding
                # return a non-contextualized embedding.
                doc.candidate_set_embed.append(self.model.embed(candidate))

            else:
                doc.candidate_set_embed.append(np.mean(candidate_embeds, axis=0))

        if "z_score" in post_processing:
            # TODO: Why z_score_normalization by space split?
            doc.candidate_set_embed = z_score_normalization(
                doc.candidate_set_embed, doc.raw_text, self.model
            )

        # TODO: If in global attention mode the document embeding should be computed again having the
        # attention mask changed to the candidate positions.
        if cand_mode == "global_attention":
            doc.doc_embed = self.global_embed_doc(doc)

    def extract_candidates(
        self,
        doc: Document,
        min_len: int = 5,
        grammar: str = "",
        lemmer_lang: str = None,
    ):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document and
        stores the sentences each candidate occurs in

        len(candidate.split(" ")) <= 5 avoid too long candidate phrases

        Parameters
        ----------
                min_len: minimum candidate length (chars)
        """
        doc.candidate_set = set()
        doc.candidate_mentions = {}

        parser = RegexpParser(grammar)
        np_trees = list(parser.parse_sents(doc.tagged_text))

        for i in range(len(np_trees)):
            temp_cand_set = []
            for subtree in np_trees[i].subtrees(filter=lambda t: t.label() == "NP"):
                temp_cand_set.append(" ".join(word for word, tag in subtree.leaves()))

            for candidate in temp_cand_set:
                # candidate max number of words is 5 because longer candidates may be overfitting
                if len(candidate) > min_len and len(candidate.split(" ")) <= 5:
                    l_candidate = (
                        lemmatize(candidate, lemmer_lang) if lemmer_lang else candidate
                    )
                    # if l_candidate not in doc.candidate_set:
                    doc.candidate_set.add(l_candidate)

                    if l_candidate not in doc.candidate_mentions:
                        doc.candidate_mentions[l_candidate] = []

                    doc.candidate_mentions[l_candidate].append(candidate)

        doc.candidate_set = sorted(list(doc.candidate_set), key=len, reverse=True)

    def embed_n_candidates(
        self, doc: Document, min_len: int, stemmer, **kwargs
    ) -> Tuple[np.ndarray, List[str]]:
        """
        TODO: Why embed_n_candidates

        Return
        ------
            candidate_set_embed:    np.ndarray of the embedings for each candidate.
            candicate_set:          List of candidates.
        """
        doc_mode = kwargs.get("doc_mode", "")
        cand_mode = kwargs.get("global_attention", "")
        post_processing = kwargs.get("post_processing", [""])

        t = time()
        doc.doc_embed = self.embed_doc(doc, stemmer, doc_mode, post_processing)
        logger.info(f"Embed Doc in {time() -  t:.2f}s")

        t = time()
        self.embed_candidates(doc, stemmer, cand_mode, post_processing)
        print(f"Embed Candidates in {time() -  t:.2f}s")

        if cand_mode == "global_attention":
            doc.doc_embed = self.global_embed_doc(doc)

        return self.candidate_set_embed, self.candidate_set

    def evaluate_n_candidates(
        self, doc_embed: np.ndarray, candidate_set_embed, candidate_set
    ) -> List[Tuple]:
        """
        This method is key for each ranking model.
        Here the ranking heuritic is applied according to model definition.

        EmbedRank selects the candidates that have more similarity to the document.

        TODO: Rename method to rank_candidates()
        """
        # doc_embed = doc.doc_embed.reshape(1, -1)
        doc_sim = np.absolute(
            cosine_similarity(candidate_set_embed, doc_embed.reshape(1, -1))
        )
        candidate_score = sorted(
            [
                (candidate, candidate_doc_sim[0])
                for (candidate, candidate_doc_sim) in zip(candidate_set, doc_sim)
            ],
            # [(candidate_set[i], doc_sim[i][0]) for i in range(len(doc_sim))],
            reverse=True,
            key=lambda x: x[1],
        )

        return candidate_score, [candidate[0] for candidate in candidate_score]

    def top_n_candidates(
        self,
        doc,
        top_n: int = 5,
        min_len: int = 5,
        stemmer: Callable = None,
        **kwargs,
    ) -> List[Tuple]:
        doc_mode = kwargs.get("doc_mode", "")
        cand_mode = kwargs.get("cand_mode", "")
        post_processing = kwargs.get("post_processing", [""])
        use_cache = kwargs.get("embed_memory", False)
        mmr_mode = kwargs.get("mmr", False)
        mmr_diversity = kwargs.get("diversity", 0.8)

        t = time()
        doc.doc_embed = self.embed_doc(doc, stemmer, doc_mode, post_processing)
        logger.info(f"Embed Doc in {time() -  t:.2f}s")

        if cand_mode != "" and cand_mode != "AvgContext":
            self.embed_sents_words(doc, stemmer, use_cache)

        t = time()
        self.embed_candidates(doc, stemmer, cand_mode, post_processing)
        logger.info(f"Embed Candidates in {time() -  t:.2f}s")

        doc_sim = []
        if mmr_mode:
            assert mmr_diversity > 0
            assert mmr_diversity < 1
            valid_top_n = len(doc.candidate_set)
            if top_n > 0:
                valid_top_n = (
                    len(doc.candidate_set) if len(doc.candidate_set) < top_n else top_n
                )
            doc_sim = mmr(
                doc.doc_embed.reshape(1, -1),
                doc.candidate_set_embed,
                doc.candidate_set,
                valid_top_n,
                diversity=mmr_diversity,
            )
        else:
            doc_sim = np.absolute(
                cosine_similarity(doc.candidate_set_embed, doc.doc_embed.reshape(1, -1))
            )

        candidate_score = sorted(
            [(doc.candidate_set[i], doc_sim[i][0]) for i in range(len(doc_sim))],
            reverse=True,
            key=lambda x: x[1],
        )

        if top_n == -1:
            return candidate_score, [candidate[0] for candidate in candidate_score]

        return candidate_score[:top_n], [candidate[0] for candidate in candidate_score]
