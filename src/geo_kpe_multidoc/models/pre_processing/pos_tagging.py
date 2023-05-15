from abc import ABC, abstractmethod
from os import path
from pathlib import Path
from typing import List, Tuple

import joblib
import spacy
import torch
from loguru import logger

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.utils.IO import read_from_file, write_to_file


class POS_tagger(ABC):
    """
    Abstract data class for POS tagging
    """

    @abstractmethod
    def pos_tag_str(self, text: str = "") -> None:
        """
        POS tag a string and return it in model representation form
        """
        ...

    @abstractmethod
    def pos_tag_doc(self, text: str = "") -> List[List[Tuple[str, str]]]:
        """
        POS tag a document and return it's result in form List of sentences with each word as a Tuple (text, token.pos_)
        """
        ...

    @abstractmethod
    def pos_tag_doc_sents(
        self, text: str = ""
    ) -> Tuple[List[List[Tuple[str, str]]], List[str]]:
        """
        POS tag a document and return it's result in Tuple form, with the first element being a List of sentences with each
        word as a Tuple (text, token.pos_), and the second a list of document sentences
        """
        ...

    @abstractmethod
    def pos_tag_text_sents_words(
        self, text: str = "", use_cache: bool = False, id: int = 0
    ) -> Tuple[List[List[Tuple[str, str]]], List[str], List[List[str]]]:
        """
        POS tag a document and return it's result in Tuple form, with the first element being a List of sentences with each
        word as a Tuple (text, token.pos_), the second a list of document sentences and the third a list of words in each sentence.
        """
        ...


class POS_tagger_spacy(POS_tagger):
    """
    Concrete data class for POS tagging using spacy
    """

    def __init__(self, model, exclude=["ner", "lemmatizer"]):
        self.tagger = spacy.load(model, exclude=exclude)
        self.name = model

    def pos_tag_str(self, text: str = "") -> spacy.tokens.doc.Doc:
        return self.tagger(text)

    def pos_tag_doc(self, text: str = "") -> List[List[Tuple]]:
        doc = self.tagger(text)
        return [
            [(token.text, token.pos_) for token in sent]
            for sent in doc.sents
            if sent.text.strip()
        ]

    def pos_tag_doc_sents(self, text: str = "") -> Tuple[List[List[Tuple]], List[str]]:
        doc = self.tagger(text)
        return (
            [
                [(token.text, token.pos_) for token in sent]
                for sent in doc.sents
                if sent.text.strip()
            ],
            list(doc.sents),
        )

    def pos_tag_text_sents_words(
        self, text: str = "", use_cache: bool = False, id: str = ""
    ) -> Tuple[List[List[Tuple[str, str]]], List[str], List[List[str]]]:
        logger.debug(f"Cache:{use_cache} Id:{id}")

        # only bypass spacy PoS tagging, other transformations are not skipped (e.g. joining NOUN HYP NOUN).
        if use_cache:
            cache_file_path = path.join(
                GEO_KPE_MULTIDOC_CACHE_PATH,
                "POS_CACHE",
                f"{self.name}-{id}-PoS.cache",
            )
            if path.exists(cache_file_path):
                doc = joblib.load(cache_file_path)
                logger.debug(f"Load POS tags from cache {cache_file_path}")
            else:
                doc = self.tagger(text)
                Path(cache_file_path).parent.mkdir(exist_ok=True, parents=True)
                joblib.dump(doc, cache_file_path)
                logger.info(f"Save {id} POS tags in {cache_file_path}")
        else:
            doc = self.tagger(text)

        tagged_text = []
        doc_word_sents = []

        for sent in doc.sents:
            if sent.text.strip():
                tagged_text_s = []
                doc_word_sents_s = []

                # HACK: DEBUG {'Non - Marine Association'}
                # if "Marine" in text:
                #     pass
                # if "Mr." in sent.text:
                #     pass

                for token in sent:
                    tagged_text_s.append((token.text, token.pos_))
                    doc_word_sents_s.append(token.text)

                # TODO: join NOUN -(NOUN) NOUN
                #       to deal with `re-election` and `post-tax`.
                # TODO: and `mr. smith`?
                for i in range(1, len(doc_word_sents_s) - 1):
                    if i + 1 < len(doc_word_sents_s):
                        if doc_word_sents_s[i] == "-" and tagged_text_s[i][1] in [
                            "NOUN",
                            "ADJ",
                            "PROPN",
                        ]:
                            # keep original tag given to `-` (NOUN or ADJ)
                            tagged_text_s[i] = (
                                f"{doc_word_sents_s[i-1]}-{doc_word_sents_s[i+1]}",
                                tagged_text_s[i][1],
                            )
                            del tagged_text_s[i + 1]
                            del tagged_text_s[i - 1]

                            doc_word_sents_s[
                                i
                            ] = f"{doc_word_sents_s[i-1]}-{doc_word_sents_s[i+1]}"
                            del doc_word_sents_s[i + 1]
                            del doc_word_sents_s[i - 1]

                        elif doc_word_sents_s[i] == "." and tagged_text_s[i][1] in [
                            "NOUN",
                            "ADJ",
                            "PROPN",
                        ]:
                            # join `.` with last token and keep the same tag.
                            tagged_text_s[i] = (
                                f"{doc_word_sents_s[i-1]}.",
                                tagged_text_s[i][1],
                            )
                            del tagged_text_s[i - 1]

                            doc_word_sents_s[i] = f"{doc_word_sents_s[i-1]}."
                            del doc_word_sents_s[i - 1]

                tagged_text.append(tagged_text_s)
                doc_word_sents.append(doc_word_sents_s)

        return (tagged_text, list(doc.sents), doc_word_sents)

    def _save_on_cache(self, tagged_text, doc_id: str = "") -> None:
        logger.info(f"Caching PoS Tags for Document {doc_id}")
        joblib.dump(
            tagged_text,
            path.join(
                GEO_KPE_MULTIDOC_CACHE_PATH, "POS_CACHE", f"{self.name}-{id}-PoS.cache"
            ),
        )
