from os import write
from typing import List, Dict, Tuple, Callable
import numpy as np
import simplemma
import json

from nltk.stem import PorterStemmer
from time import gmtime, strftime
from utils.IO import write_to_file

from ...geo_kpe_multidoc import GEO_KPE_MULTIDOC_OUTPUT_PATH


def extract_res_labels(
    model_results, stemmer: Callable = None, lemmer: Callable = None
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
    model_results, stemmer: Callable = None, lemmer: Callable = None
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
    corpus_true_labels, stemmer: Callable = None, lemmer: Callable = None
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
    Function that evaluates the model result in each dataset it ran on, considering the true labels of said dataset.
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
            top_kp = model_results[dataset][i][0]
            len_top_kp = float(len(top_kp))

            candidates = model_results[dataset][i][1]
            len_candidates = float(len(candidates))

            true_label = true_labels[dataset][i]
            len_true_label = float(len(true_label))

            # Precision, Recall and F1-Score for candidates
            p = min(
                1.0, len([kp for kp in candidates if kp in true_label]) / len_candidates
            )
            r = min(
                1.0, len([kp for kp in candidates if kp in true_label]) / len_true_label
            )
            f1 = 0.0

            if p != 0 and r != 0:
                f1 = (2.0 * p * r) / (p + r)

            results_c["Precision"].append(p)
            results_c["Recall"].append(r)
            results_c["F1"].append(f1)

            if kp_eval:
                # Precision_k, Recall_k, F1-Score_k, MAP and nDCG for KP
                for k in k_set:
                    p_k = min(
                        1.0,
                        len([kp for kp in top_kp[:k] if kp in true_label])
                        / float(len(top_kp[:k])),
                    )
                    r_k = min(
                        1.0,
                        (
                            len([kp for kp in top_kp[:k] if kp in true_label])
                            / len_true_label
                        ),
                    )
                    f1_k = 0.0

                    if p_k != 0 and r_k != 0:
                        f1_k = (2.0 * p_k * r_k) / (p_k + r_k)

                    results_kp[f"Precision_{k}"].append(p_k)
                    results_kp[f"Recall_{k}"].append(r_k)
                    results_kp[f"F1_{k}"].append(f1_k)

                ap = [
                    len([k for k in top_kp[:i] if k in true_label]) / float(i)
                    for i in range(1, len(top_kp) + 1)
                    if top_kp[i - 1] in true_label
                ]
                map = np.sum(ap) / float(len(true_label))
                ndcg = np.sum(
                    [
                        1.0 / np.log2(i + 1)
                        for i in range(1, len(top_kp) + 1)
                        if top_kp[i - 1] in true_label
                    ]
                )
                ndcg = ndcg / np.sum(
                    [1.0 / np.log2(i + 1) for i in range(1, len(true_label) + 1)]
                )

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
            for (name, dic) in [("candidates", results_c), ("kp", results_kp)]:

                res_dic[name] = {}
                for measure in dic:
                    res_dic[name][measure] = dic[measure]

    if save:
        with open(f"{GEO_KPE_MULTIDOC_OUTPUT_PATH}/raw/{stamp} raw.txt", "a") as f:
            f.write(res.rstrip())

    print(res)


def output_top_cands(
    model_results: Dict[str, List] = {}, true_labels: Dict[str, Tuple[List]] = {}
):
    top_cand_l = []
    for dataset in model_results:
        for i in range(len(model_results[dataset])):
            top_kp = model_results[dataset][i][0]
            true_label = true_labels[dataset][i]
            top_cand_l += [
                round(float(kp[1]), 2) for kp in top_kp if kp[0] in true_label
            ]
