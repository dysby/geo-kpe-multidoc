import csv
import gzip

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

sts_dataset_path = "/home/helder/doc/mecd/thesis/data/stsbenchmark.tsv.gz"

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

from transformers import AutoTokenizer, LongformerModel

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

model_path = "allenai/longformer-base-4096"
model = SentenceTransformer(model_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    test_samples, name="sts-allenai-longformer", show_progress_bar=True
)

test_evaluator(model, output_path="/home/helder/doc/mecd/thesis/outputs")