import torch
from sentence_transformers import SentenceTransformer
from transformers.models.longformer.modeling_longformer import (
    create_position_ids_from_input_ids,
)

olong = SentenceTransformer(
    "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
)
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
inputs = olong.tokenize(["helder"])
olong_inputs_embeds = olong._modules["0"].auto_model.embeddings.word_embeddings(
    inputs["input_ids"]
)
sbert_inputs_embeds = sbert._modules["0"].auto_model.embeddings.word_embeddings(
    inputs["input_ids"]
)
torch.allclose(sbert_inputs_embeds, olong_inputs_embeds)

olong_token_type_ids = torch.zeros(inputs["input_ids"].size(), dtype=torch.long)
sbert_token_type_ids = sbert._modules["0"].auto_model.embeddings.token_type_ids[
    :, : inputs["input_ids"].size()[1]
]

sbert_token_type_embeddings = sbert._modules[
    "0"
].auto_model.embeddings.token_type_embeddings(sbert_token_type_ids)
olong_token_type_embeddings = olong._modules[
    "0"
].auto_model.embeddings.token_type_embeddings(olong_token_type_ids)
torch.allclose(sbert_token_type_embeddings, olong_token_type_embeddings)

olong_position_ids = create_position_ids_from_input_ids(
    inputs["input_ids"], padding_idx=1
)
sbert_position_ids = create_position_ids_from_input_ids(
    inputs["input_ids"], padding_idx=1
)
olong_position_embeddings = olong._modules[
    "0"
].auto_model.embeddings.position_embeddings(olong_position_ids)
sbert_position_embeddings = sbert._modules[
    "0"
].auto_model.embeddings.position_embeddings(sbert_position_ids)
torch.allclose(olong_position_embeddings, sbert_position_embeddings)

olong_embeddings_in = (
    olong_inputs_embeds + olong_token_type_embeddings + olong_position_embeddings
)
sbert_embeddings_in = (
    sbert_inputs_embeds + sbert_token_type_embeddings + sbert_position_embeddings
)
torch.allclose(olong_embeddings_in, sbert_embeddings_in)

olong_embeddings = olong._modules["0"].auto_model.embeddings.LayerNorm(
    olong_embeddings_in
)
sbert_embeddings = sbert._modules["0"].auto_model.embeddings.LayerNorm(
    sbert_embeddings_in
)
torch.allclose(olong_embeddings, sbert_embeddings)

original_sbert_embeddings = sbert._modules["0"].auto_model.embeddings(
    inputs["input_ids"]
)

sbert_embeddings

sbert_embeddings == original_sbert_embeddings

olong_embeddings == original_sbert_embeddings

torch.allclose(olong_embeddings, original_sbert_embeddings)

original_olong_embeddings = olong._modules["0"].auto_model.embeddings(
    inputs["input_ids"]
)

torch.allclose(original_olong_embeddings, original_sbert_embeddings)

olong._modules["0"].auto_model.training

olong_embeddings
original_olong_embeddings

torch.allclose(original_olong_embeddings, olong_embeddings)
