import os
from itertools import islice
from os import path
from pathlib import Path
from time import gmtime, strftime
from typing import Callable, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from nltk.stem.api import StemmerI
from tqdm import tqdm

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.datasets import KPEDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.evaluation.metrics import MAP, f1_score, nDCG, precision, recall
from geo_kpe_multidoc.models import EmbedRank, MaskRank, MDKPERank
from geo_kpe_multidoc.models.embedrank.longembedrank import LongEmbedRank
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_hyphens_and_dots,
)


def postprocess(kp, preprocessing):
    kp = remove_hyphens_and_dots(kp.lower())
    for transformation in preprocessing:
        kp = transformation(kp)
    return kp


def postprocess_model_outputs(
    model_results: Dict,
    stemmer: StemmerI = None,
    lemmer: Callable = None,
    preprocessing: List[Callable] = None,
):
    """
    Code snippet to correctly model results

    {
        dataset_name: [
                ( top_n_cand_and_scores: List[Tuple[str, float]] , candidates: List[str]),
                ...
                ],
        ...
    }

    Usually candidates are already lemmatized.

    """
    res = {}
    for dataset in model_results:
        res[dataset] = []
        for doc in model_results[dataset]:
            if preprocessing:
                cand_score, cand = doc
                cand_score = [
                    (postprocess(kp, preprocessing), score) for kp, score in cand_score
                ]
                cand = [postprocess(kp, preprocessing) for kp in cand]
                doc = (cand_score, cand)

            if stemmer:
                res[dataset].append(
                    (
                        # each (kp, score)
                        [
                            (
                                " ".join(
                                    [
                                        stemmer.stem(w)
                                        for w in kp.split()
                                        # for w in simplemma.simple_tokenizer(kp)
                                    ]
                                ).lower(),
                                score,
                            )
                            for kp, score in doc[0]
                        ],
                        # each kp candidate
                        [
                            " ".join(
                                [
                                    stemmer.stem(w)
                                    for w in kp.split()
                                    # for w in simplemma.simple_tokenizer(kp)
                                ]
                            ).lower()
                            for kp in doc[1]
                        ],
                    )
                )
    return res


def postprocess_dataset_labels(
    corpus_true_labels,
    stemmer: StemmerI = None,
    lemmer: str = None,
    preprocessing: List[Callable] = None,
):
    """
    Code snippet to correctly format dataset true labels

    {
        dataset_name: [
                doc_labels:List[str],
                ...
                ],
        ...
    }

    """
    res = {}
    for dataset in corpus_true_labels:
        res[dataset] = []
        for i in range(len(corpus_true_labels[dataset])):
            doc_results = []
            for kp in corpus_true_labels[dataset][i]:
                if preprocessing:
                    kp = postprocess(kp, preprocessing)

                # if lemmer:
                #     kp = lemmatize(kp, lemmer)
                if stemmer:
                    kp = " ".join(stemmer.stem(w) for w in kp.lower().split())
                doc_results.append(kp)
            # TODO: remove duplicates and discard empty
            # doc_results = list(set(doc_results).discard(""))
            doc_results = [kp for kp in doc_results if kp]
            res[dataset].append(doc_results)
    return res


def evaluate_kp_extraction_base(
    model_results: Dict[str, List] = {},
    true_labels: Dict[str, Tuple[List]] = {},
    model_name: str = "",
    save: bool = True,
    kp_eval: bool = True,
    stemmer: StemmerI = None,
    **kwargs,
) -> None:
    """
    Function that evaluates the model result in each dataset it ran on, considering the true labels of said dataset.
    """
    results = pd.DataFrame()
    results.index.name = "Dataset"
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
            top_kp = [kp for kp, _ in model_results[dataset][i][0]]
            len_top_kp = float(len(top_kp))

            candidates = model_results[dataset][i][1]
            len_candidates = float(len(candidates))

            true_label = true_labels[dataset][i]
            len_true_label = float(len(true_label))

            # Use stemming for evaluation
            if stemmer:
                candidates = [
                    " ".join([stemmer.stem(w) for w in kp.split()]).lower()
                    for kp in candidates
                ]
                true_label = [
                    " ".join([stemmer.stem(w) for w in kp.split()]).lower()
                    for kp in true_label
                ]

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
            for name, dic in [("candidates", results_c), ("kp", results_kp)]:
                res_dic[name] = {}
                for measure in dic:
                    res_dic[name][measure] = dic[measure]
        row = (
            pd.concat(
                [
                    pd.DataFrame(results_c).mean(axis=0),
                    pd.DataFrame(results_kp).mean(axis=0),
                ]
            )
            .to_frame()
            .T
        )
        row.index = pd.Index([f"{dataset}_base_eval"])
        results = pd.concat([results, row])
    # if save:
    #     with open(f"{RESULT_DIR}/raw/{stamp} raw.txt", "a") as f:
    #         f.write(res.rstrip())

    # print(res)
    return results


def evaluate_kp_extraction(
    model_results: Dict[str, List] = {},
    true_labels: Dict[str, List[List]] = {},
    model_name: str = "",
    kp_eval: bool = True,
    k_set=[5, 10, 15],
    **kwargs,
) -> pd.DataFrame:
    """
    Evaluate model results for each dataset, considering the true labels.

    Parameters
    ----------
        model_results:  Dict[str, List]
            Dictionary with dataset Names as keys and Results as values
            ex: model_results["dataset_name"][(doc1_top_n = (kp1, score_kp1), doc1_candidates), (doc2...)]
        true_labels: Dict[str, List[List]]
            keys are the dataset names, and values are the list of gold keyphrases for each document
            ex: true_labels["dataset_name"][[doc1_kp1, doc1_kp2], [doc2...]]
    """

    results = pd.DataFrame()
    results.index.name = "Dataset"

    for dataset in model_results:
        results_c = {"Precision": [], "Recall": [], "F1": []}

        results_kp = {"MAP": [], "nDCG": []}

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

        row = (
            pd.concat(
                [
                    pd.DataFrame(results_c).mean(axis=0),
                    pd.DataFrame(results_kp).mean(axis=0),
                ]
            )
            .to_frame()
            .T
        )
        row.index = pd.Index([dataset])
        results = pd.concat([results, row])

    return results


def extract_keyphrases_docs(
    dataset: KPEDataset,
    model: Union[EmbedRank, MaskRank, LongEmbedRank],
    top_n=20,
    min_len=0,
    lemmer=None,
    n_docs_limit=-1,
    **kwargs,
):
    """
    Extraction and Evaluation for Single Document mode
    """
    model_results = {dataset.name: []}
    true_labels = {dataset.name: []}
    experiment = kwargs.get("experiment", "debug")
    preprocessing = kwargs.get("preprocessing", [])
    cache_results = kwargs.get("cache_results", False)

    if isinstance(n_docs_limit, str):
        loader = [dataset[dataset.ids.index(n_docs_limit)]]
        n_docs_limit = 1
    elif n_docs_limit == -1:
        loader = dataset
        n_docs_limit = len(dataset)
    else:
        loader = islice(dataset, n_docs_limit)

    for doc_id, txt, gold_kp in tqdm(loader, total=n_docs_limit):
        logger.info(f"KPE for document {doc_id}")
        top_n_and_scores, candidates = model.extract_kp_from_doc(
            Document(
                txt,
                doc_id,
                dataset.name,
                pre_processing_pipeline=preprocessing,
            ),
            top_n=top_n,
            min_len=min_len,
            lemmer=lemmer,
            **kwargs,
        )
        model_results[dataset.name].append(
            (
                top_n_and_scores,
                candidates,
            )
        )

        # TODO: remove preprocessing outside Document class
        # TODO: refactor processing gold to function
        # TODO: preprocessing hyphens and dots
        # if preprocessing:
        #     processed_gold_kp = []
        #     for kp in gold_kp:
        #         kp = remove_hyphens_and_dots(kp.lower())
        #         for transformation in preprocessing:
        #             kp = transformation(kp)
        #         processed_gold_kp.append(kp)
        #     # TODO: remove duplicates and discard empty
        #     processed_gold_kp = set(processed_gold_kp)
        #     processed_gold_kp.discard("")
        #     gold_kp = processed_gold_kp

        # Decision: lemmatize is not applied to gold
        # if lemmer:
        #     gold_kp = lemmatize(gold_kp, lemmer)

        true_labels[dataset.name].append(gold_kp)

        if cache_results:
            os.makedirs(
                path.join(GEO_KPE_MULTIDOC_CACHE_PATH, experiment), exist_ok=True
            )
            joblib.dump(
                {
                    "dataset": dataset.name,
                    "topic": doc_id,
                    "doc": doc_id,
                    "top_n_scores": top_n_and_scores,
                    "gold": gold_kp,
                },
                path.join(
                    GEO_KPE_MULTIDOC_CACHE_PATH,
                    experiment,
                    f"{doc_id}-top_n_scores.pkl",
                ),
            )

    return model_results, true_labels


def extract_keyphrases_topics(
    dataset: KPEDataset,
    model: MDKPERank,
    top_n=-1,
    min_len=0,
    lemmer=None,
    n_docs_limit=-1,
    **kwargs,
):
    """
    Extraction and Evaluation for Multi Document mode
    """
    model_results = {dataset.name: []}
    true_labels = {dataset.name: []}
    cache_results = kwargs.get("cache_results", False)
    experiment = kwargs.get("experiment", "debug")
    preprocessing = kwargs.get("preprocessing", [])

    if n_docs_limit == -1:
        # no limit
        loader = dataset
        n_docs_limit = len(dataset)
    else:
        loader = islice(dataset, n_docs_limit)

    for i, (topic_id, docs, gold_kp) in enumerate(tqdm(loader, total=n_docs_limit)):
        logger.info(f"KPE for topic {topic_id}")
        (
            top_n_scores,
            # score_per_document,
            candidate_document_matrix,
            documents_embeddings,
            candidate_embeddings,
        ) = model.extract_kp_from_topic(
            # top_n_and_scores, candidates = model.extract_kp_from_topic(
            # top_n_and_scores, candidates = model.extract_kp_geo(
            # TODO: ***Warning*** this is only true for MultiDocument Dataset!
            [
                Document(
                    txt,
                    doc_name,
                    dataset.name,
                    topic_id,
                    pre_processing_pipeline=preprocessing,
                )
                for doc_name, txt in docs
            ],  # kpe_model.pre_process(doc),
            top_n=top_n,
            min_len=min_len,
            lemmer=lemmer,
            **kwargs,
        )

        # candidates = candidate_embeddings.index.tolist()
        candidates, _ = list(zip(*top_n_scores))

        model_results[dataset.name].append((top_n_scores, candidates))

        # if len(preprocessing) > 0:
        #     processed_gold_kp = []
        #     for kp in gold_kp:
        #         kp = remove_hyphens_and_dots(kp.lower())
        #         for transformation in preprocessing:
        #             kp = transformation(kp)
        #         processed_gold_kp.append(kp)
        #     # TODO: remove duplicates and discard empty
        #     processed_gold_kp = set(processed_gold_kp)
        #     processed_gold_kp.discard("")
        #     gold_kp = processed_gold_kp

        true_labels[dataset.name].append(gold_kp)

        if cache_results:
            filename = path.join(
                GEO_KPE_MULTIDOC_CACHE_PATH,
                experiment,
                f"{topic_id}-mdkpe-results.pkl",
            )

            Path(filename).parent.mkdir(exist_ok=True, parents=True)

            joblib.dump(
                {
                    "dataset": dataset.name,
                    "topic": topic_id,
                    "top_n_scores": top_n_scores,
                    "candidate_document_matrix": candidate_document_matrix,
                    "gold_kp": gold_kp,
                    "documents_embeddings": documents_embeddings,
                    "candidate_embeddings": candidate_embeddings,
                },
                filename,
            )
            logger.info(f"Saving {topic_id} results in cache dir {filename}")

    return model_results, true_labels


def model_scores_to_dataframe(model_results, true_labels) -> pd.DataFrame:
    df = pd.DataFrame()

    for dataset, doc_candidade_values in model_results.items():
        for i, (candidate_scores, doc_gold_kp) in enumerate(
            zip(doc_candidade_values, true_labels[dataset])
        ):
            candidates_score, _candidates = candidate_scores

            # TODO: Uniform score value class
            rows = [
                {
                    "dataset": dataset,
                    "doc": i,
                    "candidate": candidate,
                    "score": score.item() if isinstance(score, np.ndarray) else score,
                    "in_gold": candidate in doc_gold_kp,
                }
                for candidate, score in candidates_score
            ]
            df = pd.concat([df, pd.DataFrame.from_records(rows)])

    return df
