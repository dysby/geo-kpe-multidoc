import torch
from sentence_transformers import SentenceTransformer

from geo_kpe_multidoc.models.backend._longmodels import to_longformer_t_v4

torch_device = "cpu"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
model, tokenizer = to_longformer_t_v4(sbert)

sentence = ["helder"]
attention_window = 512
max_length = 4096

encoded_input = tokenizer(
    sentence,
    # padding="max_length",
    padding=True,
    pad_to_multiple_of=attention_window,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
    return_attention_mask=True,
)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling
sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])


original_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

converted_model = SentenceTransformer(
    "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
)

original_embedding = original_model.encode(
    ["helder"], convert_to_tensor=True, output_value=None
)
converted_embedding = converted_model.encode(
    ["helder"], convert_to_tensor=True, output_value=None
)

torch.allclose(original_embedding, sentence_embeddings.detach().cpu().numpy())
