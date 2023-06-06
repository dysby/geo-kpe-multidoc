import csv
import gzip
import os
from tempfile import TemporaryDirectory

import click
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.models.Pooling import Pooling
from sentence_transformers.readers import InputExample
from transformers import AutoModel, AutoTokenizer

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH, GEO_KPE_MULTIDOC_OUTPUT_PATH
from geo_kpe_multidoc.models.backend.roberta2longformer.roberta2bigbird import (
    convert_roberta_to_bigbird,
)
from geo_kpe_multidoc.models.backend.roberta2longformer.roberta2longformer import (
    convert_roberta_to_longformer,
)
from geo_kpe_multidoc.models.backend.roberta2longformer.roberta2nyströmformer import (
    convert_roberta_to_nystromformer,
)
from geo_kpe_multidoc.models.sentence_embedder import (
    BigBirdSentenceEmbedder,
    LongformerSentenceEmbedder,
    NystromformerSentenceEmbedder,
)


@click.command()
@click.option(
    "--base_model",
    prompt="Sentence Transformer model name",
    help="Sentence Transformer model name.",
    default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
@click.option("--variant", default="original")
def stsbenchmark(base_model, variant):
    sts_dataset_path = os.path.join(GEO_KPE_MULTIDOC_DATA_PATH, "stsbenchmark.tsv.gz")

    train_samples = []
    dev_samples = []
    test_samples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0  # Normalize score to range 0 ... 1
            inp_example = InputExample(
                texts=[row["sentence1"], row["sentence2"]], label=score
            )

            if row["split"] == "dev":
                dev_samples.append(inp_example)
            elif row["split"] == "test":
                test_samples.append(inp_example)
            else:
                train_samples.append(inp_example)

    base_model = AutoModel.from_pretrained(base_model)
    base_tokenizer = AutoTokenizer.from_pretrained(base_model)

    if variant == "longformer":
        new_model, new_tokenizer = convert_roberta_to_longformer(
            base_model, base_tokenizer, 4096, 128
        )
        model = LongformerSentenceEmbedder(new_model, new_tokenizer)
    elif variant == "bigbird":
        new_model, new_tokenizer = convert_roberta_to_bigbird(
            base_model, base_tokenizer, 4096
        )
        model = BigBirdSentenceEmbedder(new_model, new_tokenizer)
    elif variant == "nystromformer":
        new_model, new_tokenizer = convert_roberta_to_nystromformer(
            base_model, base_tokenizer, 4096
        )
        model = NystromformerSentenceEmbedder(new_model, new_tokenizer)
    else:
        model = base_model

    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples, name=variant, show_progress_bar=True
    )

    test_evaluator(model, output_path=GEO_KPE_MULTIDOC_OUTPUT_PATH)


if __name__ == "__main__":
    stsbenchmark()
