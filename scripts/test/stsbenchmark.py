import csv
import gzip
import os
from tempfile import TemporaryDirectory

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.models.Pooling import Pooling
from sentence_transformers.readers import InputExample

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

# model_save_path = "paraphrase-multilingual-mpnet-base-v2"
# model = SentenceTransformer(model_save_path)

from transformers import AutoModel, AutoTokenizer

# model_path = "/home/helder/doc/mecd/thesis/models/longformer-xlm-robterta"
# # model_path = "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
# model = SentenceTransformer(model_path)
# model._modules["0"]._modules["auto_model"] = LongformerModel.from_pretrained(model_path)
# model._modules["0"].tokenizer = AutoTokenizer.from_pretrained(model_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
#     test_samples, name="sts-test", show_progress_bar=True
# )

# model_path = "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
# model = SentenceTransformer(model_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
#     test_samples, name="sts-longformer", show_progress_bar=True
# )

base_model = AutoModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
base_tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# new_model, new_tokenizer = convert_roberta_to_bigbird(base_model, base_tokenizer, 4096)
new_model, new_tokenizer = convert_roberta_to_longformer(
    base_model, base_tokenizer, 4096, 128
)
# new_model, new_tokenizer = convert_roberta_to_nystromformer(
#     base_model, base_tokenizer, 4096
# )

with TemporaryDirectory() as temp_dir:
    new_model.save_pretrained(temp_dir)
    base_tokenizer.model_max_length = 4096
    base_tokenizer.save_pretrained(temp_dir)
    model = SentenceTransformer(temp_dir)

    model.tokenizer.enable_padding(length=4096, pad_to_multiple_of=128)  # longformer
    # model.tokenizer.enable_padding(length=4096, pad_to_multiple_of=64)  # bigbird
    # model.tokenizer.enable_padding(length=4096, pad_to_multiple_of=64)  # bigbird
    model.tokenizer.enable_truncation(max_length=4096, strategy="longest_first")

    # test_samples, name="sbert", show_progress_bar=True
    # test_samples, name="nystromformer", show_progress_bar=True
    # test_samples, name="", show_progress_bar=True
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    test_samples, name="longformer", show_progress_bar=True
)

test_evaluator(model, output_path=GEO_KPE_MULTIDOC_OUTPUT_PATH)
