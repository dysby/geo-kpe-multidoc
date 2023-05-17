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
from nltk import RegexpParser
from nltk.stem import PorterStemmer
from nltk.stem.api import StemmerI
from sklearn.metrics.pairwise import cosine_similarity

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.base_KP_model import BaseKPModel, find_occurrences
from geo_kpe_multidoc.models.pre_processing.language_mapping import (
    choose_lemmatizer,
    choose_tagger,
)
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy
from geo_kpe_multidoc.models.pre_processing.post_processing_utils import (
    z_score_normalization,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    filter_special_tokens,
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

    # def extract_mdkpe_embeds(
    #     self, doc: Document, top_n, min_len, stemmer=None, lemmer=None, **kwargs
    # ) -> Tuple[Document, List[np.ndarray], List[str]]:
    #     use_cache = kwargs.get("pos_tag_memory", False)

    #     self.pos_tag_doc(
    #         doc=doc,
    #         stemming=None,
    #         use_cache=use_cache,
    #     )
    #     self.extract_candidates(doc, min_len, self.grammar, lemmer)

    #     cand_embeds, candidate_set = self.embed_n_candidates(doc, stemmer, **kwargs)

    #     logger.info(f"Document #{self.counter} processed")
    #     self.counter += 1
    #     torch.cuda.empty_cache()

    #     return (doc, cand_embeds, candidate_set)

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
        Embed_sent_words (Use only on non-contextualized candicate embedding mode, not used).
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

        return doc_embeddings["sentence_embedding"].detach().cpu().numpy()

    def _search_mentions(self, doc, candidate):
        mentions = []
        # TODO: mention counts for mean_in_n_out_context
        # mentions_counts = []
        for mention in doc.candidate_mentions[candidate]:
            if isinstance(self.model, BaseEmbedder):
                # original tokenization by KeyBert/SentenceTransformer
                tokenized_candidate = tokenize_hf(mention, self.model)
            else:
                # tokenize via local SentenceEmbedder Class
                tokenized_candidate = self.model.tokenize(mention)

            filt_ids = filter_special_tokens(tokenized_candidate["input_ids"])

            mentions += find_occurrences(filt_ids, doc.token_ids)
            # mentions_counts.append(len(mentions))

        # mentions_counts = mentions_counts[:1] + [
        #     y - x for x, y in pairwise(mentions_counts)
        # ]
        return mentions  # , mentions_counts

    def _embedding_in_context(self, doc: Document, cand_mode: str = ""):
        for candidate in doc.candidate_set:
            # mentions, mentions_counts = self._search_mentions(doc, candidate)
            mentions = self._search_mentions(doc, candidate)

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

                doc.candidate_set_embed.append(torch.mean(embds, dim=0).numpy())

                #         if "mean_in_context" in cand_mode:
                # # get out of context embedding for the original mention form of his candidate
                # mention_index = [
                #     idx for idx, v in enumerate(mentions_counts) if i <= v
                # ]
                # # We know that we are processing the ith mention form of the candidate
                # mention = doc.candidate_mentions[candidate][mention_index]
                # if isinstance(self.model, BaseEmbedder):
                #     no_context_embedding = self.model.embed(mention)
                # else:
                #     no_context_embedding = (
                #         self.model.encode(mention, device=self.device)[
                #             "sentence_embedding"
                #         ]
                #         .detach()
                #         .cpu()
                #         .numpy()
                #     )
                # embds[i] = torch.mean([embds[i], no_context_embedding])

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
            self._embedding_in_context(doc)

        if "z_score" in post_processing:
            # TODO: Why z_score_normalization by space split?
            doc.candidate_set_embed = z_score_normalization(
                doc.candidate_set_embed, doc.raw_text, self.model
            )

    def _set_global_attention_on_candidates(self, doc: Document):
        mentions = []
        for candidate in doc.candidate_set:
            # mentions_positions, _ = self._search_mentions(doc, candidate)
            mentions_positions = self._search_mentions(doc, candidate)
            if len(mentions_positions) > 0:
                # candidate mentions where found in document token_ids
                mentions.extend(mentions_positions)
        mentions = tuple(set(chain(*mentions)))
        logger.debug(f"Global Attention in {len(mentions)} tokens")
        doc.global_attention_mask[:, mentions] = 1

    def extract_candidates(
        self,
        doc: Document,
        min_len: int = 5,
        grammar: str = None,
        lemmer_lang: str = None,
        **kwargs,
    ):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document
        and stores the sentences each candidate occurs in.

        len(candidate.split(" ")) <= 5 avoid too long candidate phrases

        Parameters
        ----------
                min_len: minimum candidate length (chars)

        TODO: why not use aggregate candidades in stem form?
        DONE: why not lowercase candidates? lemmatize returns lowercase candidates
        DONE: candidate mentions should not be lemmatized, maybe lowercased (document will be embed in lowercase form),
                otherwise mentions are not found in document. Do not change original mention form.
                Model tokenizer is responsible for case handling.

            Baseline NP:
                {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}

        TODO: new grammar (({.*}{HYPH}{.*}){NOUN}*)|(({VBG}|{VBN})?{ADJ}*{NOUN}+) Keyphrase-Vectorizers paper ()
                        r'(({.*}-.*-{.*}){NN}*)|(({VBG}|{VBN})?{JJ}*{NN}+)'
            WORKS! in KeyphraseVectorizer
                        '((<.*>-+<.*>)<NN>*)|((<VBG|VBN>)?<JJ>*<NN>+)'

            SIFRank grammar '<NN.*|JJ>*<NN.*>'  ,  NN = NOUN, JJ = ADJ

            Automatic Extraction of Relevant Keyphrases for the Study of Issue Competition
                (<NOUN>+<ADJ>*<PREP>*)?<NOUN>+<ADJ>*

            UKE-CCRank

                GRAMMAR1 = NP:
                    {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)

                GRAMMAR2 = NP:
                    {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)

                GRAMMAR3 = NP:
                    {<NN.*|JJ|VBG|VBN>*<NN.*>}  # Adjective(s)(optional) + Noun(s)
        """
        cache_pos_tags = kwargs.get("cache_pos_tags", False)
        self._pos_tag_doc(
            doc=doc,
            stemming=None,
            use_cache=cache_pos_tags,
        )

        grammar = self.grammar if not grammar else grammar

        doc.candidate_set = set()
        doc.candidate_mentions = {}

        parser = RegexpParser(grammar)

        # grammar by pos_ or by tag_?
        # here use use pos_, KeyphraseVectorizers use tag_
        # A choice between using a coarse-grained tag set that is consistent across languages (.pos),
        # or a fine-grained tag set (.tag) that is specific to a particular treebank, and hence a particular language.

        np_trees = list(parser.parse_sents(doc.tagged_text))

        for i in range(len(np_trees)):
            # TODO: validate that candidates in correct form, meaning  " ".join is ok?
            temp_cand_set = []
            for subtree in np_trees[i].subtrees(filter=lambda t: t.label() == "NP"):
                temp_cand_set.append(" ".join(word for word, tag in subtree.leaves()))

            # TODO: how to deal with `re-election campain`? join in line above will result in `re - election campain`.
            #       Then the model will nevel find this candidate mentions because the original form is lost.
            #       This is a hack, to handle `-` and `.` in the middle of a candidate.
            #       Check from `pos_tag_text_sents_words` where `-` are joined rto surrounding nouns.

            for candidate in temp_cand_set:
                # candidate max number of words is 5 because longer candidates may be overfitting
                # HACK: DEBUG
                # if candidate in [
                #     "cane crops",
                #     "mile band",
                #     "casualty",
                #     "Pounds",
                #     "Non - Marine Association",
                #     "Texas border",
                #     "Roberts",
                # ]:
                #     pass

                if len(candidate) > min_len and len(candidate.split(" ")) <= 5:
                    # TODO: 'we insurer':{'US INSURERS'} but 'eastern us': {'eastern US'} ...
                    l_candidate = (
                        lemmatize(candidate, lemmer_lang) if lemmer_lang else candidate
                    )
                    doc.candidate_set.add(l_candidate)

                    # Candidate mentions was a list of candidate forms,
                    # it should be a set (no repetitions), when embedding candidate the mentions
                    # will be searched and all occorrences will count.
                    # DONE: keep candidate mentions in lower form. Document is embedded in lower case.
                    # DONE: candidate forms are kept in original form. The tokenizer of the model is
                    # responsible handling text case.
                    doc.candidate_mentions.setdefault(l_candidate, set()).add(
                        candidate  # candidate.lower()
                    )

        doc.candidate_set = sorted(list(doc.candidate_set), key=len, reverse=True)

        return doc.candidate_set, doc.candidate_mentions

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

        # Set Global Attention CLS token for all modes
        if isinstance(self.model, BaseEmbedder):
            # original tokenization by KeyBert/SentenceTransformer
            tokenized_doc = tokenize_hf(doc.raw_text, self.model)
        else:
            # tokenize via local SentenceEmbedder Class
            tokenized_doc = self.model.tokenize(
                doc.raw_text,
                padding=True,
                pad_to_multiple_of=self.model.attention_window,
            )
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
        logger.info(f"Embed Doc in {time() -  t:.2f}s")

        t = time()
        self._aggregate_candidate_mention_embeddings(
            doc, stemmer, cand_mode, post_processing
        )
        logger.info(f"Embed Candidates in {time() -  t:.2f}s")

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
            # TODO: Not same format as cosine_similarity
            logger.error("TODO: Not same format as cosine_similarity")
        else:
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

        return candidate_score[:top_n], [candidate[0] for candidate in candidate_score]

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
        if cand_mode != "" and cand_mode != "AvgContext":
            logger.error(f"Getting Embeddings for word sentence (not used?)")
            # self.embed_sents_words(doc, stemmer, use_cache)

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
