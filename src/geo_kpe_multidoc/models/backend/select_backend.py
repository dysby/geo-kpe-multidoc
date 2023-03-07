from keybert.backend._base import BaseEmbedder
from keybert.backend._utils import select_backend as keybert_select_backend


def select_backend(embedding_model) -> BaseEmbedder:
    # keybert language backend
    if isinstance(embedding_model, BaseEmbedder):
        return embedding_model
    # Longmodels model
    if "longformer" in str(embedding_model) or "bigbird" in str(embedding_model):
        from geo_kpe_multidoc.models.backend._longmodels import load_longmodel

        return load_longmodel(embedding_model)

    return keybert_select_backend(embedding_model)
