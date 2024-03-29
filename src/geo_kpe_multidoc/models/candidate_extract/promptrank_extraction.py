import re
from typing import List, Tuple

import nltk
from nltk import RegexpParser
from nltk.corpus import stopwords

from geo_kpe_multidoc.datasets.language import ISO_to_language
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    lemmatize,
    remove_hyphens_and_dots,
    remove_whitespaces,
)


def remove_special(text: str):
    text_len = len(text.split())
    remove_chars = "[’!\"#$%&'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+"
    text = re.sub(remove_chars, "", text)
    re_text_len = len(text.split())
    if text_len != re_text_len:
        return True
    else:
        return False


class PromptRankKPECandidateExtractionModel:
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """

    def __init__(self, tagger, language, grammar=None, **kwargs) -> None:
        self.tagger = POS_tagger_spacy(tagger)

        self.grammar = (
            grammar
            if grammar
            else """  NP:
                {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
        )

        self.max_len = kwargs.get("max_seq_len", 512)
        self.kp_min_len = kwargs.get("kp_min_len", 0)
        self.kp_max_words = kwargs.get("kp_max_words", 256)
        self.parser = RegexpParser(self.grammar)
        self.language = ISO_to_language[language]
        # Limit candidate size
        self.stopword_dict = set(stopwords.words(self.language))
        self.enable_filter = kwargs.get("enable_filter", False)

    def __call__(
        self,
        doc: Document,
        kp_min_len: int = 5,
        grammar: str = None,
        lemmer_lang: str = None,
        **kwargs,
    ) -> Tuple[List, List]:
        return self._extract_candidates(doc, kp_min_len, grammar, lemmer_lang, **kwargs)

    def _pos_tag_doc(self, doc: Document, **kwargs) -> None:
        # considered_tags = {"NN", "NNS", "NNP", "NNPS", "JJ"}

        tokens = []

        text = " ".join(doc.raw_text.split()[: self.max_len])
        tagged_doc = self.tagger.pos_tag_str(text)

        tokens_tagged = [
            (token.text, token.tag_)
            for sent in tagged_doc.sents
            if sent.text.strip()
            for token in sent
        ]

        tokens = [token for token, _ in tokens_tagged]

        assert len(tokens) == len(tokens_tagged)
        for i, token in enumerate(tokens):
            if token.lower() in self.stopword_dict:
                tokens_tagged[i] = (token, "IN")
        doc.tagged_text = tokens_tagged

    def _extract_candidates(
        self,
        doc: Document,
        kp_min_len: int = 5,
        grammar: str = None,
        lemmer_lang: str = None,
        **kwargs,
    ) -> Tuple[List, List]:
        """
        Based on part of speech return a list of candidate phrases
        :param text_obj: Input text Representation see @InputTextObj
        :param no_subset: if true won't put a candidate which is the subset of an other candidate
        :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
        """

        self._pos_tag_doc(doc)

        cans_count = dict()

        keyphrase_candidate = []
        np_pos_tag_tokens = self.parser.parse(doc.tagged_text)
        count = 0
        for token in np_pos_tag_tokens:
            if isinstance(token, nltk.tree.Tree) and token._label == "NP":
                np = " ".join(word for word, tag in token.leaves())
                length = len(token.leaves())
                start_end = (count, count + length)
                count += length

                if len(np.split()) == 1:
                    if np not in cans_count.keys():
                        cans_count[np] = 0
                    cans_count[np] += 1

                keyphrase_candidate.append((np, start_end))

            else:
                count += 1
        # What is this filter?
        if self.enable_filter:
            i = 0
            while i < len(keyphrase_candidate):
                can, pos = keyphrase_candidate[i]
                # pos[0] > 50 and
                if can in cans_count.keys() and cans_count[can] == 1:
                    keyphrase_candidate.pop(i)
                    continue
                i += 1

        candidates = []
        for can, pos in keyphrase_candidate:
            if self.enable_filter and (len(can.split()) > 4):
                continue

            # This was moved from generate_doc_pairs to here
            # Why re.sub instead of simple search?
            if remove_special(can):
                #    count += 1
                continue

            if len(can) < self.kp_min_len:
                continue

            # Adictional processing if lemmatization is enabled
            l_candidate = (
                lemmatize(
                    remove_whitespaces(remove_hyphens_and_dots(can.lower())),
                    lemmer_lang,
                )
                if lemmer_lang
                else can.lower()
            )

            if not l_candidate:
                # candidates like '-' are lemmatized to empty string
                continue

            candidates.append([l_candidate, pos])
            # add mentions for Embedrank compability
            doc.candidate_mentions.setdefault(l_candidate, set()).add(can)

        candidates, positions = list(zip(*candidates))

        doc.candidate_set = list(candidates)
        doc.candidate_positions = list(positions)

        return candidates, positions
