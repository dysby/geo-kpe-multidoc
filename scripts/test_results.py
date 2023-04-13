import joblib
from nltk.stem import PorterStemmer

from geo_kpe_multidoc.evaluation.evaluation_tools import (
    evaluate_kp_extraction,
    postprocess_dataset_labels,
    postprocess_res_labels,
)

model_results_long = joblib.load("model_results_long.pkl")
model_results_sbert = joblib.load("model_results_sbert.pkl")
true_labels = joblib.load("true_labels.pkl")

stemmer = PorterStemmer()
lemmer = "en"

evaluate_kp_extraction(
    postprocess_res_labels(model_results_sbert, stemmer, lemmer),
    postprocess_dataset_labels(true_labels, stemmer, lemmer),
)
