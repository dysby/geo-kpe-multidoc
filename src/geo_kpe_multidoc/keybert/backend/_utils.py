from ._base import BaseEmbedder


def select_backend(embedding_model) -> BaseEmbedder:
    """ Select an embedding model based on language or a specific sentence transformer models.
    When selecting a language, we choose distilbert-base-nli-stsb-mean-tokens for English and
    xlm-r-bert-base-nli-stsb-mean-tokens for all other languages as it support 100+ languages.
    Returns:
        model: Either a Sentence-Transformer or Flair model
    """
    # keybert language backend
    if isinstance(embedding_model, BaseEmbedder):
        return embedding_model

    # Longmodels model
    elif "longformer" in str(embedding_model) or "bigbird" in str(embedding_model):
        from keybert.backend._longmodels import load_longmodel
        return load_longmodel(embedding_model)

    # Flair word embeddings
    elif "flair" in str(type(embedding_model)):
        from keybert.backend._flair import FlairBackend
        return FlairBackend(embedding_model)

    # Spacy embeddings
    elif "spacy" in str(type(embedding_model)):
        from keybert.backend._spacy import SpacyBackend
        return SpacyBackend(embedding_model)

    # Gensim embeddings
    elif "gensim" in str(type(embedding_model)):
        from keybert.backend._gensim import GensimBackend
        return GensimBackend(embedding_model)

    # USE embeddings
    elif "tensorflow" and "saved_model" in str(type(embedding_model)):
        from keybert.backend._use import USEBackend
        return USEBackend(embedding_model)

    # Sentence Transformer embeddings
    elif "sentence_transformers" in str(type(embedding_model)):
        from ._sentencetransformers import SentenceTransformerBackend
        return SentenceTransformerBackend(embedding_model)

    # Create a Sentence Transformer model based on a string
    elif isinstance(embedding_model, str):
        from ._sentencetransformers import SentenceTransformerBackend
        return SentenceTransformerBackend(embedding_model)
    
    from ._sentencetransformers import SentenceTransformerBackend
    return SentenceTransformerBackend("xlm-r-bert-base-nli-stsb-mean-tokens")