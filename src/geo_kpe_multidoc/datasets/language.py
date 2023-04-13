from enum import Enum

"""
From https://github.com/KNOT-FIT-BUT/QBEK
"""


class Language(Enum):
    """
    Representation of a language.
    Also defines set of supported languages.
    """

    CZECH = "cze"
    ENGLISH = "eng"
    GERMAN = "ger"
    FRENCH = "fre"

    @classmethod
    def lang_2_spacy(cls) -> Dict["Language", spacy.language.Language]:
        """
        Mapping from language to its spacy language model.
        """
        return {
            cls.CZECH: Czech,
            cls.ENGLISH: English,
            cls.GERMAN: German,
            cls.FRENCH: French,
        }

    @property
    def code(self) -> str:
        """
        Language code.
        CZECH  -> cze
        """
        return self.value

    @property
    def spacy(self) -> spacy.language.Language:
        """
        Spacy language model class.
        """
        return self.lang_2_spacy()[self]

    @property
    def tokenizer(self) -> Tokenizer:
        """
        Tokenizer for a language.
        """

        return SpaCyTokenizer.init_shared(self.spacy)

    @property
    def lemmatizer(self) -> Lemmatizer:
        """
        Lemmatizer for a language.
        """

        if self == self.CZECH:
            # for czech the morphodite is better
            return MorphoditaLemmatizer.init_shared()
        elif self == self.ENGLISH:
            return PorterStemmer()
        else:
            return SpaCyLemmatizer.init_shared(self.spacy)

    @property
    def lemmatizer_factory(self) -> LemmatizerFactory:
        """
        Lemmatizer factory for a language.
        """

        if self == self.CZECH:
            # for czech the morphodite is better
            return MorphoditaLemmatizerFactory()
        elif self == self.ENGLISH:
            return PorterStemmerFactory()
        else:
            return SpaCyLemmatizerFactory(self.spacy)
