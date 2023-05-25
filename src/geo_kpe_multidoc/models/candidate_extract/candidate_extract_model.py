from os import path
from pathlib import Path
from typing import Callable, List, Set, Tuple

import joblib
import numpy as np
import tqdm
from loguru import logger
from nltk import RegexpParser

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import lemmatize


def extract_kp_from_doc(
    extractor, doc, top_n, min_len, stemming, **kwargs
) -> Tuple[List[Tuple], List[str]]:
    """
    Concrete method that extracts key-phrases from a given document, with optional arguments
    relevant to its specific functionality
    """

    tagged_doc = extractor.pos_tag_doc(doc, **kwargs)
    candidate_list = extractor.extract_candidates(tagged_doc, **kwargs)
    print("doc finished\n")
    return ([], candidate_list)


def extract_kp_from_corpus(
    extractor, corpus, top_n=5, min_len=0, stemming=False, **kwargs
) -> List[List[Tuple]]:
    """
    Concrete method that extracts key-phrases from a list of given documents, with optional arguments
    relevant to its specific functionality
    """
    return [
        extractor.extract_kp_from_doc(doc[0], top_n, min_len, stemming, **kwargs)
        for doc in tqdm(corpus)
    ]


class KPECandidateExtractionModel:
    """
    Keyphrase Candidate identification by grammar pos tag parsing
    """

    def __init__(self, tagger, grammar=None):
        self.tagger = POS_tagger_spacy(tagger)
        self.grammar = (
            grammar if grammar else """NP: {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}"""
        )
        self.parser = RegexpParser(self.grammar)
        self.single_word_grammar = {"PROPN", "NOUN", "ADJ"}

    def __call__(
        self,
        doc: Document,
        min_len: int = 5,
        grammar: str = None,
        lemmer_lang: str = None,
        **kwargs,
    ):
        self._extract_candidates_simple(doc, min_len, grammar, lemmer_lang, **kwargs)

    def _pos_tag_doc(self, doc: Document, stemming, use_cache, **kwargs) -> None:
        (
            doc.tagged_text,
            doc.doc_sentences,
            doc.doc_sentences_words,
        ) = self.tagger.pos_tag_text_sents_words_simple(
            doc.raw_text, use_cache, doc.dataset, doc.id
        )

    def __pos_tag_doc(
        self, doc: str = "", stemming: bool = True, **kwargs
    ) -> List[List[Tuple]]:
        """
        Method that handles POS_tagging of an entire document, pre-processing or stemming it in the process
        """
        tagged_doc = self.tagger.pos_tag_doc(doc)
        for sent in tagged_doc:
            for i in range(1, len(sent) - 1):
                if i + 1 < len(sent):
                    if sent[i][0] == "-":
                        sent[i] = (f"{sent[i-1][0]}-{sent[i+1][0]}", "NOUN")
                        del sent[i + 1]
                        del sent[i - 1]
        return tagged_doc

    def _extract_candidates_simple(
        self,
        doc: Document,
        min_len: int = 4,
        grammar: str = None,
        lemmer_lang: str = None,
        **kwargs,
    ) -> List[str]:
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document
        """

        cache_candidate_selection = kwargs.get("cache_candidate_selection", False)
        if cache_candidate_selection:
            doc.candidate_set, doc.candidate_mentions = self._read_cache(
                doc.dataset, doc.id
            )
            if len(doc.candidate_set) > 0 and (
                len(doc.candidate_set) == len(doc.candidate_mentions)
            ):
                return doc.candidate_set, doc.candidate_mentions

        use_cache = kwargs.get("pos_tag_memory", False)
        self._pos_tag_doc(
            doc=doc,
            stemming=None,
            use_cache=use_cache,
        )

        doc.candidate_set = set()
        doc.candidate_mentions = {}

        np_trees = self.parser.parse_sents(doc.tagged_text)

        for tree in np_trees:
            temp_cand_set = [
                " ".join(word for word, tag in subtree.leaves())
                for subtree in tree.subtrees(filter=lambda t: t.label() == "NP")
            ]
            # for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
            #     candidate_set.add(" ".join(word for word, tag in subtree.leaves()))
            # TODO: add unigram candidates
            # for word, tag in subtree.leaves():
            #     if tag in self.single_word_grammar:
            #         candidate_set.add(word)

            # TODO: Join hyphen nouns
            # TODO: Join " ." nouns
            temp_cand_set = [
                candidate.replace(" - ", "-").replace(" .", ".")
                for candidate in temp_cand_set
            ]

            for candidate in temp_cand_set:
                # TODO: Remove min_len and max words
                if len(candidate) > min_len:  # and len(candidate.split(" ")) <= 5:
                    # TODO: 'we insurer':{'US INSURERS'} but 'eastern us': {'eastern US'} ...
                    l_candidate = (
                        lemmatize(candidate, lemmer_lang) if lemmer_lang else candidate
                    )
                    doc.candidate_set.add(l_candidate)

                    doc.candidate_mentions.setdefault(l_candidate, set()).add(candidate)

        # candidate_set = {kp.lower() for kp in candidate_set if len(kp.split()) <= 7}
        # TODO: limit candidate size
        # TODO: lemmatize and save mentions
        # candidate_set = {kp.lower() for kp in candidate_set}

        doc.candidate_set = sorted(doc.candidate_set, key=len, reverse=True)

        if cache_candidate_selection:
            self._save_cache(
                doc.dataset, doc.doc_id, doc.candidate_set, doc.candidate_mentions
            )

        return doc.candidate_set, doc.candidate_mentions

    def _extract_candidates(
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

        Baseline
            NP:
                {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}

        TODO: new grammar

            (({.*}{HYPH}{.*}){NOUN}*)|(({VBG}|{VBN})?{ADJ}*{NOUN}+) Keyphrase-Vectorizers paper ()
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

        Parameters
        ----------
                min_len: minimum candidate length (chars)
        """
        cache_pos_tags = kwargs.get("cache_pos_tags", False)

        self._pos_tag_doc(
            doc=doc,
            stemming=None,
            use_cache=cache_pos_tags,
        )

        grammar = grammar if grammar else self.grammar

        doc.candidate_set = set()
        doc.candidate_mentions = {}

        # grammar by pos_ or by tag_?
        # here use use pos_, KeyphraseVectorizers use tag_
        # A choice between using a coarse-grained tag set that is consistent across languages (.pos),
        # or a fine-grained tag set (.tag) that is specific to a particular treebank, and hence a particular language.

        np_trees = list(self.parser.parse_sents(doc.tagged_text))

        for tree in np_trees:
            temp_cand_set = [
                " ".join(word for word, tag in subtree.leaves())
                for subtree in tree.subtrees(filter=lambda t: t.label() == "NP")
            ]

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

                # TODO: Remove min_len and max words
                if len(candidate) > min_len and len(candidate.split(" ")) <= 5:
                    # TODO: 'we insurer':{'US INSURERS'} but 'eastern us': {'eastern US'} ...
                    l_candidate = (
                        lemmatize(candidate, lemmer_lang) if lemmer_lang else candidate
                    )
                    doc.candidate_set.add(l_candidate)

                    doc.candidate_mentions.setdefault(l_candidate, set()).add(candidate)

        doc.candidate_set = sorted(doc.candidate_set, key=len, reverse=True)

        return doc.candidate_set, doc.candidate_mentions

    def __extract_candidates(
        self, tagged_doc: List[List[Tuple]] = [], **kwargs
    ) -> List[str]:
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document
        """
        use_cache = kwargs.get("pos_tag_memory", False)
        self._pos_tag_doc(
            doc=doc,
            stemming=None,
            use_cache=use_cache,
        )

        candidate_set = set()
        parser = RegexpParser(self.grammar)
        np_trees = parser.parse_sents(tagged_doc)

        for tree in np_trees:
            for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
                candidate_set.add(" ".join(word for word, tag in subtree.leaves()))
                for word, tag in subtree.leaves():
                    if tag in self.single_word_grammar:
                        candidate_set.add(word)

        candidate_set = {kp for kp in candidate_set if len(kp.split()) <= 7}
        return list(candidate_set)

    def _mask_rank_extract_candidates(
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
        use_cache = kwargs.get("cache_pos_tags", False)
        if use_cache:
            logger.warning("POS Tag Cache in maskrank not implemented")

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

    def _read_cache(self, dataset, doc_id) -> Tuple[set, dict]:
        cache_file_path = path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            "candidates" f"{dataset}-{doc_id}-candidates.cache",
        )
        if path.exists(cache_file_path):
            (candidate_set, candidate_mentions) = joblib.load(cache_file_path)
            logger.debug(f"Load Candidates and Mentions from cache {cache_file_path}")
            return (candidate_set, candidate_mentions)
        else:
            return (set(), {})

    def _save_cache(self, dataset, doc_id, candidate_set, candidate_mentions):
        cache_file_path = path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            "candidates" f"{dataset}-{doc_id}-candidates.cache",
        )
        Path(cache_file_path).parent.mkdir(exist_ok=True, parents=True)
        joblib.dump((candidate_set, candidate_mentions), cache_file_path)
        logger.info(f"Save {doc_id} Keyphrase Candidates in {cache_file_path}")
