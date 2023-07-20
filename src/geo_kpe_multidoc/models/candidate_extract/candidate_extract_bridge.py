import itertools
from collections import Counter
from os import path
from pathlib import Path
from typing import Callable, List, Tuple

import joblib
import nltk
from loguru import logger
from nltk import RegexpParser

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    lemmatize,
    remove_hyphens_and_dots,
    remove_whitespaces,
)


class BridgeKPECandidateExtractionModel:
    """
    Keyphrase Candidate identification by grammar pos tag parsing

        Baseline
            NP:
                {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}

        TODO: new grammar
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

            HAKE: an Unsupervised Approach to Automatic Keyphrase Extraction for Multiple Domains
                    {<NN | NNS | NNP | NNPS | VBN | JJ | JJS | RB>*<NN | NNS | NNP | NNPS | VBG>+}
                    adverbial noun (tag RB) such as “double experience” (RB NN), and a verb in present participle (tag VBG) such as follows:
                    “virtual desktop conferencing” (JJ NN VBG), where the VBG tag can be at the beginning, the middle,
                    or at the end of the noun phrase.

    """

    def __init__(self, tagger, grammar=None):
        self.tagger = POS_tagger_spacy(tagger)
        self.grammar = (
            grammar if grammar else """NP: {<PROPN|NOUN|ADJ>*<PROPN|NOUN>+<ADJ>*}"""
        )
        self.parser = RegexpParser(self.grammar)
        self.single_word_grammar = {"PROPN", "NOUN", "ADJ"}

        self.join_hyphen_in_candidate = False
        # Or use in POS Tagging class (pos_tagging.py)
        self.join_hyphen_pos = False
        self.join_hyphen_pos_valid = False

    def __call__(
        self,
        doc: Document,
        min_len: int = 5,
        grammar: str = None,
        lemmer_lang: str = None,
        **kwargs,
    ):
        return self._extract_candidates_positions(
            doc, min_len, grammar, lemmer_lang, **kwargs
        )

    def _pos_tag_doc(self, doc: Document, stemming, use_cache, **kwargs) -> None:
        (
            doc.tagged_text,
            doc.doc_sentences,
            doc.doc_sentences_words,
        ) = self.tagger.pos_tag_text_sents_words_simple(
            doc.raw_text,
            use_cache,
            doc.dataset,
            doc.id,
            join_hyphen=self.join_hyphen_pos,
            join_hyphen_only_valid_pos=self.join_hyphen_pos_valid,
        )

    def _extract_candidates_positions(
        self,
        doc: Document,
        min_len: int = 0,
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

        use_cache = kwargs.get("cache_pos_tags", False)
        self._pos_tag_doc(
            doc=doc,
            stemming=None,
            use_cache=use_cache,
        )

        doc.candidate_set = set()
        doc.candidate_mentions = dict()
        doc.candidate_positions = dict()

        # np_trees = self.parser.parse_sents(doc.tagged_text)
        tokens_tagged = list(itertools.chain.from_iterable(doc.tagged_text))
        np_pos_tag_tokens = self.parser.parse(tokens_tagged)

        cans_count = Counter()
        keyphrase_candidate = []
        count = 0
        for token in np_pos_tag_tokens:
            if isinstance(token, nltk.tree.Tree) and token._label == "NP":
                np = " ".join(word for word, tag in token.leaves())
                length = len(token.leaves())
                start_end = (count, count + length)
                count += length

                if len(np.split()) == 1:
                    cans_count[np] += 1

                keyphrase_candidate.append((np, start_end))

            else:
                count += 1

        for candidate, start_end in keyphrase_candidate:
            l_candidate = (
                lemmatize(
                    remove_whitespaces(remove_hyphens_and_dots(candidate.lower())),
                    lemmer_lang,
                )
                if lemmer_lang
                else candidate.lower()
            )
            doc.candidate_set.add(l_candidate)
            doc.candidate_mentions.setdefault(l_candidate, set()).add(candidate)
            doc.candidate_positions.setdefault(l_candidate, []).append(start_end)

        doc.candidate_set = sorted(doc.candidate_set, key=len, reverse=True)
        # keep only the first position of the candidate
        doc.candidate_positions = [
            doc.candidate_positions[candidate][0] for candidate in doc.candidate_set
        ]

        if cache_candidate_selection:
            self._save_cache(
                doc.dataset, doc.id, doc.candidate_set, doc.candidate_mentions
            )

        return doc.candidate_set, doc.candidate_positions

    def _read_cache(self, dataset, doc_id) -> Tuple[set, dict]:
        cache_file_path = path.join(
            GEO_KPE_MULTIDOC_CACHE_PATH,
            "candidates",
            dataset,
            f"{doc_id}-candidates.cache",
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
            "candidates",
            dataset,
            f"{doc_id}-candidates.cache",
        )
        Path(cache_file_path).parent.mkdir(exist_ok=True, parents=True)
        joblib.dump((candidate_set, candidate_mentions), cache_file_path)
        logger.info(f"Save {doc_id} Keyphrase Candidates in {cache_file_path}")
