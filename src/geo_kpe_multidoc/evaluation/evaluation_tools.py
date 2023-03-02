import json
from os import write
from pathlib import Path
from time import gmtime, strftime
from typing import Callable, Dict, List, Tuple

import numpy as np
import simplemma
from nltk.stem import PorterStemmer
from nltk.stem.api import StemmerI

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_OUTPUT_PATH
from geo_kpe_multidoc.evaluation.metrics import MAP, f1_score, nDCG, precision, recall
from geo_kpe_multidoc.utils.IO import write_to_file


def extract_res_labels(
    model_results: Dict, stemmer: StemmerI = None, lemmer: Callable = None
):
    """
    Code snippet to correctly model results
    """
    res = {}
    for dataset in model_results:
        res[dataset] = []
        for doc in model_results[dataset]:
            if stemmer:
                res[dataset].append(
                    (
                        [
                            " ".join(
                                [
                                    stemmer.stem(w)
                                    for w in simplemma.simple_tokenizer(kp[0])
                                ]
                            ).lower()
                            for kp in doc[0]
                        ],
                        [
                            " ".join(
                                [
                                    stemmer.stem(w)
                                    for w in simplemma.simple_tokenizer(kp)
                                ]
                            ).lower()
                            for kp in doc[1]
                        ],
                    )
                )
    return res


def extract_res_labels_x(
    model_results, stemmer: StemmerI = None, lemmer: Callable = None
):
    """
    Code snippet to correctly model results
    """
    res = {}
    for dataset in model_results:
        res[dataset] = []
        for doc in model_results[dataset]:
            # TODO: If not stemmer results empty?
            if stemmer:
                res[dataset].append(
                    (
                        [
                            (
                                " ".join(
                                    [
                                        stemmer.stem(w)
                                        for w in simplemma.simple_tokenizer(kp[0])
                                    ]
                                ).lower(),
                                kp[1],
                            )
                            for kp in doc[0]
                        ],
                        [
                            " ".join(
                                [
                                    stemmer.stem(w)
                                    for w in simplemma.simple_tokenizer(kp)
                                ]
                            ).lower()
                            for kp in doc[1]
                        ],
                    )
                )
    return res


def extract_dataset_labels(
    corpus_true_labels, stemmer: StemmerI = None, lemmer: Callable = None
):
    """
    Code snippet to correctly format dataset true labels
    """
    res = {}
    for dataset in corpus_true_labels:
        res[dataset] = []
        for i in range(len(corpus_true_labels[dataset])):
            doc_results = []
            for kp in corpus_true_labels[dataset][i][1]:
                if lemmer:
                    kp = " ".join(
                        [
                            simplemma.lemmatize(w, lemmer)
                            for w in simplemma.simple_tokenizer(kp)
                        ]
                    ).lower()
                if stemmer:
                    kp = " ".join(
                        [stemmer.stem(w) for w in simplemma.simple_tokenizer(kp)]
                    ).lower()
                doc_results.append(kp.lower())
            res[dataset].append(doc_results)
    return res


def evaluate_kp_extraction(
    model_results: Dict[str, List] = {},
    true_labels: Dict[str, Tuple[List]] = {},
    model_name: str = "",
    save: bool = True,
    kp_eval: bool = True,
    **kwargs,
) -> None:
    """
    Evaluate model results for each dataset, considering the true labels.

    Parameters
    ----------
        model_results:  Dict[str, List]
            Dictionary whit dataset Names as keys and Results as values
            ***NOTE*** that each results does not have top n candidate score
            ex: model_results["dataset_name"][(doc1_top_n, doc1_candidates), (doc2...)]
        true_labels: Dict[str, Tuple[List]]
            keys are the dataset names, and values are the list of gold keyphrases for each document
            ex: true_labels["dataset_name"][[doc1_kp1, doc1_kp2], [doc2...]]
    """

    stamp = ""
    if "doc_mode" in kwargs and "cand_mode" in kwargs:
        stamp = f'{strftime("%Y_%m_%d %H_%M", gmtime())} {kwargs["doc_mode"]} {kwargs["cand_mode"]} {model_name}'
    else:
        stamp = f'{strftime("%Y_%m_%d %H_%M", gmtime())} {model_name}'

    res = f"{stamp}\n ------------- \n"
    res_dic = {}

    for dataset in model_results:
        results_c = {"Precision": [], "Recall": [], "F1": []}

        results_kp = {"MAP": [], "nDCG": []}

        k_set = [5, 10, 15] if "k_set" not in kwargs else kwargs["k_set"]

        for k in k_set:
            results_kp[f"Precision_{k}"] = []
            results_kp[f"Recall_{k}"] = []
            results_kp[f"F1_{k}"] = []

        for i in range(len(model_results[dataset])):
            top_kp = [kp for kp, _score in model_results[dataset][i][0]]

            candidates = model_results[dataset][i][1]

            true_label = true_labels[dataset][i]

            # Precision, Recall and F1-Score for candidates
            p = precision(candidates, true_label)
            r = recall(candidates, true_label)
            f1 = f1_score(p, r)

            results_c["Precision"].append(p)
            results_c["Recall"].append(r)
            results_c["F1"].append(f1)

            if kp_eval:
                # Precision_k, Recall_k, F1-Score_k, MAP and nDCG for KP
                for k in k_set:
                    p_k = precision(top_kp[:k], true_label)
                    r_k = recall(top_kp[:k], true_label)
                    f1_k = f1_score(p_k, r_k)

                    results_kp[f"Precision_{k}"].append(p_k)
                    results_kp[f"Recall_{k}"].append(r_k)
                    results_kp[f"F1_{k}"].append(f1_k)

                map = MAP(top_kp, true_label)
                ndcg = nDCG(top_kp, true_label)

                results_kp["MAP"].append(map)
                results_kp["nDCG"].append(ndcg)

        res += f"\nResults for Dataset {dataset}\n --- \n"

        res += "Candidate Extraction Evalution: \n"
        for result in results_c:
            res += f"{result} = {np.mean(results_c[result])*100:.3f}%\n"

        if kp_eval:
            res += "\nKP Ranking Evalution: \n"
            for result in results_kp:
                res += f"{result} = {np.mean(results_kp[result])*100:.3f}%\n"

        if save:
            res_dic[dataset] = {}
            for name, dic in [("candidates", results_c), ("kp", results_kp)]:
                res_dic[name] = {}
                for measure in dic:
                    res_dic[name][measure] = dic[measure]

    if save:
        Path(f"{GEO_KPE_MULTIDOC_OUTPUT_PATH}/raw/{stamp} raw.txt").parent.mkdir(
            exist_ok=True, parents=True
        )
        with open(f"{GEO_KPE_MULTIDOC_OUTPUT_PATH}/raw/{stamp} raw.txt", "a") as f:
            f.write(res.rstrip())

    print(res)


def output_top_cands(
    model_results: Dict[str, List] = {}, true_labels: Dict[str, Tuple[List]] = {}
):
    """
    Print Top N candidates that are in Gold Candidate list

    Parameters:
    -----------
        model_results: values are a list with results for each document [((doc1_top_n_candidates, doc1_top_n_scores], doc1_candidates), ...]
    """
    top_cand_l = []
    for dataset in model_results:
        for i in range(len(model_results[dataset])):
            top_kp_and_score = model_results[dataset][i][0]
            true_label = true_labels[dataset][i]
            top_cand_l += [
                round(float(score), 2)
                for kp, score in top_kp_and_score
                if kp in true_label
            ]
    print(top_cand_l)
