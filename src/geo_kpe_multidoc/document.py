from dataclasses import dataclass, field
from typing import Callable, List, Tuple


@dataclass
class Document:
    """
    Class to encapsulate document representation. Stores the raw text representation of the doc,
        a pos_tagger and the grammar to extract candidates with.

    Attributes
    ----------
    raw_text: Raw text representation of the document

    doc_sents: Document in list form divided by sentences

    punctuation_regex: regex that covers most punctuation and notation marks

    tagged_text: The entire document divided by sentences with POS tags in each word

    candidate_set: Set of candidates in list form, according to the supplied grammar
    candidate_set_sents: Lists of sentences where candidates occur in the document

    candidate_mentions: Dictionary with candidate (may be tranformed by lemmer or stemmer) as keys,
            and as values a list of surface forms that can be found in document text.

    doc_embed: Document in embedding form
    candidate_set_embed: Set of candidates in list form, according to the supplied grammar,
            in embedding form
    doc_sents_words_embed: Document in list form divided by sentences, each sentence
            in embedding form, word piece by word piece
    geo_locations: List of (latitude, longitude) locations from geoparsing document text
    """

    raw_text: str
    id: str
    dataset: str = ""
    topic: str = ""
    punctuation_regex = "[!\"#\$%&'\(\)\*\+,\.\/:;<=>\?@\[\]\^_`{\|}~\-\–\—\‘\’\“\”]"
    single_word_grammar = {"PROPN", "NOUN", "ADJ"}
    doc_sentences: List[str] = field(default_factory=list)
    doc_sentences_words = []
    doc_sents_words_embed = []
    tagged_text = None

    # candidates
    candidate_set = []
    candidate_mentions = {}  #
    candidate_scores = []
    candidate_positions = []
    # Tokenized document
    token_ids = []
    token_embeddings = []

    # Embeddings
    doc_embed = None
    attention_mask = []  # Attention mask
    global_attention_mask = None
    candidate_set_embed = []

    # GeoTags
    # geo_locations: List[Tuple[float, float]] = []

    def __init__(
        self,
        raw_text: str,
        id: str,
        dataset: str = "",
        topic: str = "",
        pre_processing_pipeline: List[Callable] = None,
    ):
        pre_processing_pipeline = (
            pre_processing_pipeline if pre_processing_pipeline else []
        )

        for f in pre_processing_pipeline:
            raw_text = f(raw_text)
        self.raw_text = raw_text
        self.id = id
        self.dataset = dataset if dataset else ""
        self.topic = topic if topic else ""
