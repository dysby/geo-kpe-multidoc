import os
from collections import OrderedDict

import click
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_OUTPUT_PATH
from geo_kpe_multidoc.datasets import load_data
from geo_kpe_multidoc.models.backend._longmodels import to_longformer_t_v4
from geo_kpe_multidoc.models.backend.roberta2longformer.roberta2longformer import (
    convert_roberta_to_longformer,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_new_lines_and_tabs,
    remove_whitespaces,
)
from geo_kpe_multidoc.models.sentence_embedder import batch_to_device, mean_pooling


@click.command()
@click.argument("name", default="")
def test(name):
    sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    attention_window = 128
    max_length = 4096
    # longformer, tokenizer = convert_roberta_to_longformer(
    #     sbert._modules["0"].auto_model, sbert.tokenizer, max_length, attention_window
    # )

    longformer, tokenizer = to_longformer_t_v4(
        SentenceTransformer("paraphrase-multilingual-mpnet-base-v2"),
        max_length,
        attention_window,
    )

    # longformer, tokenizer = to_longformer_t_v4(
    #     SentenceTransformer("paraphrase-multilingual-mpnet-base-v2"),
    #     max_length,
    #     attention_window,
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    longformer = longformer.to(device)
    longformer.eval()

    results = pd.DataFrame()

    duc2001 = load_data("DUC2001")
    for doc_id, txt, gold_kp in tqdm(duc2001):
        txt = remove_whitespaces(remove_new_lines_and_tabs(txt))

        encoded_input_longformer = tokenizer(
            txt,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Longformer
        global_attention_mask = torch.zeros_like(
            encoded_input_longformer["attention_mask"]
        )
        global_attention_mask[:, 0] = 1  # CLS token
        encoded_input_longformer["global_attention_mask"] = global_attention_mask
        # encoded_input_longformer["global_attention_mask"] = encoded_input_longformer["attention_mask"].detach().clone() #  All tokens with Global

        # to_longformer_t_v4
        # encoded_input_longformer["attention_mask"] = encoded_input_longformer["attention_mask"] + encoded_input_longformer["attention_mask"]
        # encoded_input_longformer["attention_mask"][:, 0] = 2 # CLS token

        encoded_input_longformer = batch_to_device(encoded_input_longformer, device)

        # Compute token embeddings
        with torch.no_grad():
            longformer_output = longformer(**encoded_input_longformer)

        # Perform pooling. In this case, mean pooling
        sentence_embedding = mean_pooling(
            longformer_output, encoded_input_longformer["attention_mask"]
        )

        longformer_output = OrderedDict(
            {
                # TODO: remove batch dimension?
                "token_embeddings": longformer_output[0].squeeze(),
                "input_ids": encoded_input_longformer["input_ids"].squeeze(),
                "attention_mask": encoded_input_longformer["attention_mask"].squeeze(),
                "sentence_embedding": sentence_embedding.squeeze(),
            }
        )

        sbert_output = sbert.encode(txt, output_value=None, convert_to_tensor=True)

        longformer_output = batch_to_device(longformer_output, torch.device("cpu"))
        sbert_output = batch_to_device(sbert_output, torch.device("cpu"))

        row = {
            "doc": doc_id,
            "doc_similarity": cosine_similarity(
                longformer_output["sentence_embedding"].reshape(1, -1),
                sbert_output["sentence_embedding"].reshape(1, -1),
            ).item(),
            "token_similarity": np.diagonal(
                cosine_similarity(
                    longformer_output["token_embeddings"],
                    sbert_output["token_embeddings"],
                )
            ),
            "long_input_ids_size": longformer_output["input_ids"].size(0),
            "sbert_input_ids_size": sbert_output["input_ids"].size(0),
            "equal_input_ids": torch.allclose(
                longformer_output["input_ids"][: sbert_output["input_ids"].size(0)],
                sbert_output["input_ids"],
                atol=1e-03,
            ),
            "equal_input_ids_size": (
                longformer_output["input_ids"].size(0)
                == sbert_output["input_ids"].size(0)
            ),
            "equal_attention_mask": torch.allclose(
                longformer_output["attention_mask"][
                    : sbert_output["input_ids"].size(0)
                ],
                sbert_output["attention_mask"],
                atol=1e-03,
            ),
        }

        results = pd.concat([results, pd.DataFrame([row])])

    print(results["doc_similarity"].describe().T)

    results.to_csv(
        os.path.join(
            GEO_KPE_MULTIDOC_OUTPUT_PATH, f"sbert_and_longformer128_{name}.csv"
        ),
        index_label="doc_id",
    )


if __name__ == "__main__":
    test()
