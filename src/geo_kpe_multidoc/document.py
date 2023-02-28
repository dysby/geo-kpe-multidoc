from dataclasses import dataclass, field
from typing import List


@dataclass
class Document:
    """
    Class to encapsulate document representation and functionality
    """

    raw_text: str
    id: str
    punctuation_regex = "[!\"#\$%&'\(\)\*\+,\.\/:;<=>\?@\[\]\^_`{\|}~\-\–\—\‘\’\“\”]"
    single_word_grammar = {"PROPN", "NOUN", "ADJ"}
    doc_sentences: List[str] = field(default_factory=list)
    doc_sentences_words = []
    doc_sents_words_embed = []

    # Tokenized document
    token_ids = []
    token_embeddings = []
    attention_mask = []
    candidate_mentions = {}

    def __init__(self, raw_text, id):
        """
        Stores the raw text representation of the doc, a pos_tagger and the grammar to
        extract candidates with.

        Attributes:
            raw_text -> Raw text representation of the document
            doc_sents -> Document in list form divided by sentences
            punctuation_regex -> regex that covers most punctuation and notation marks

            tagged_text -> The entire document divided by sentences with POS tags in each word
            candidate_set -> Set of candidates in list form, according to the supplied grammar
            candidate_set_sents -> Lists of sentences where candidates occur in the document

            doc_embed -> Document in embedding form
            doc_sents_words_embed -> Document in list form divided by sentences, each sentence in embedding form, word piece by word piece
            candidate_set_embed -> Set of candidates in list form, according to the supplied grammar, in embedding form
        """

        self.raw_text = raw_text
        self.single_word_grammar = {"PROPN", "NOUN", "ADJ"}
        self.doc_sentences = []
        self.doc_sentences_words = []
        self.id = id

        # Tokenized document
        self.token_ids = []
        self.token_embeddings = []
        self.attention_mask = []
        self.candidate_mentions = {}
