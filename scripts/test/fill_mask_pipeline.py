from typing import Dict

import numpy as np
import torch
from transformers.pipelines import FillMaskPipeline
from transformers.pipelines.base import GenericTensor


class LongFillMaskPipeline(FillMaskPipeline):
    """
    Masked language modeling prediction pipeline using any :obj:`ModelWithLMHead`. See the `masked language modeling
    examples <../task_summary.html#masked-language-modeling>`__ for more information.

    This mask filling pipeline can currently be loaded from :func:`~transformers.pipeline` using the following task
    identifier: :obj:`"fill-mask"`.

    The models that this pipeline can use are models that have been trained with a masked language modeling objective,
    which includes the bi-directional models in the library. See the up-to-date list of available models on
    `huggingface.co/models <https://huggingface.co/models?filter=masked-lm>`__.

    .. note::

        This pipeline only works for inputs with exactly one token masked.
    """

    def preprocess(
        self, inputs, return_tensors=None, **preprocess_parameters
    ) -> Dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(
            inputs, padding="max_length", return_tensors=return_tensors
        )
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs


from sentence_transformers import SentenceTransformer

from geo_kpe_multidoc.models.backend._longmodels import to_longformer_t_v4
from geo_kpe_multidoc.models.sentence_embedder import SentenceEmbedder

slong = SentenceEmbedder(
    *to_longformer_t_v4(SentenceTransformer("paraphrase-multilingual-mpnet-base-v2"))
)

del slong.model.embeddings.token_type_ids

fill_mask = LongFillMaskPipeline(model=slong.model, tokenizer=slong.tokenizer)

print(fill_mask("Send these <mask> back!"))
pass
