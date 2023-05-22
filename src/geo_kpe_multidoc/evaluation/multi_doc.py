import os
import re
from functools import partial
from itertools import chain
from typing import Tuple

import joblib
import optuna
import pandas as pd
from nltk.stem.porter import PorterStemmer

import geo_kpe_multidoc.geo.measures
import geo_kpe_multidoc.geo.utils
from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.datasets.process_mordecai import load_topic_geo_locations
from geo_kpe_multidoc.evaluation.evaluation_tools import (
    evaluate_kp_extraction,
    postprocess_dataset_labels,
    postprocess_res_labels,
)
from geo_kpe_multidoc.geo.utils import process_geo_associations_for_topics


def get_cache_files(path: str):
    geo_file_name_pattern = re.compile(r"d\d{2}-mdkpe-geo\.pkl")
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)) and geo_file_name_pattern.match(
            file
        ):
            yield file


def add_gold_label(df, gold):
    """
    Mutate dataframe `df` adding a label column if candidate is in the gold set.
    """
    gold_idx = pd.MultiIndex.from_tuples(
        chain.from_iterable(
            df.index[
                df.index.isin([topic], level=0) & df.index.isin(gold[topic], level=1)
            ]
            for topic in df.index.get_level_values(0).unique()
        ),
        names=["topic", "keyphrases"],
    )

    not_gold_idx = pd.MultiIndex.from_tuples(
        chain.from_iterable(
            df.index[
                df.index.isin([topic], level=0) & ~df.index.isin(gold[topic], level=1)
            ]
            for topic in df.index.get_level_values(0).unique()
        ),
        names=["topic", "keyphrases"],
    )

    df.loc[gold_idx, "gold"] = True
    df.loc[not_gold_idx, "gold"] = False


def multidoc_cache_to_frames(
    experiment: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Read MDKPERank cache files from `experiment` directory and join scores into dataframes.

    Parameters
    ----------
    experiment : str
        Name of experiment cache dir

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, dict, dict]
        Global Topic data, Docs data, Topic Candidate Document Matrix, Gold
    """
    global_topic_data = pd.DataFrame()
    docs_data = pd.DataFrame()
    topic_candidate_document_matrix = dict()
    gold = dict()

    for filename in get_cache_files(
        os.path.join(GEO_KPE_MULTIDOC_CACHE_PATH, experiment)
    ):
        topic_id = filename[:3]
        # {
        #     "dataset": dataset_name,
        #     "topic": topic_id,
        #     "top_n_scores": top_n_scores,
        #     "score_per_document": score_per_document,
        #     "candidate_document_matrix": candidate_document_matrix,
        #     "documents_embeddings": document_embeddings,
        #     "candidates_embeddings": candidates_embeddings,
        #     "gold_kp": gold_kp,
        # }
        topic_results = joblib.load(
            os.path.join(GEO_KPE_MULTIDOC_CACHE_PATH, experiment, filename)
        )

        N = topic_results["candidate_document_matrix"].sum(axis=1)

        topic_docs_data = topic_results["score_per_document"].melt(
            var_name="document", value_name="semantic_score", ignore_index=False
        )
        topic_docs_data["topic"] = topic_id
        topic_docs_data.index.name = "keyphrase"
        topic_docs_data = topic_docs_data.set_index(
            ["topic", "document", topic_docs_data.index]
        )
        topic_docs_data

        topic_data = pd.DataFrame(
            topic_results["top_n_scores"], columns=["semantic_score"]
        )
        topic_data["N"] = N
        topic_data["topic"] = topic_id
        topic_data.index.name = "keyphrase"
        topic_data = topic_data.set_index(["topic", topic_data.index])

        # coordinates[topic_id] = keyphrase_coordinates
        gold[topic_id] = topic_results["gold_kp"]
        topic_candidate_document_matrix[topic_id] = topic_results[
            "candidate_document_matrix"
        ]
        docs_data = pd.concat([docs_data, topic_docs_data])
        global_topic_data = pd.concat([global_topic_data, topic_data])

    add_gold_label(global_topic_data, gold)

    return (global_topic_data, docs_data, topic_candidate_document_matrix, gold)


def gather_geo_locations(experiment: str, use_cache=True) -> pd.DataFrame:
    # gather geo location / coordenates from mordecai parsing
    # need to run only once.

    if use_cache:
        return pd.read_parquet(
            os.path.join(
                GEO_KPE_MULTIDOC_CACHE_PATH,
                "MKDUC01-topic-doc-coordinates-20230504.parquet",
            )
        )

    topic_docs_coordinates = pd.DataFrame()

    for filename in get_cache_files(
        os.path.join(GEO_KPE_MULTIDOC_CACHE_PATH, experiment)
    ):
        topic_id = filename[:3]

        df = (
            pd.DataFrame.from_dict(
                {topic_id: load_topic_geo_locations(topic_id)}, orient="index"
            )
            .stack()
            .explode()
            .to_frame()
        )

        df.columns = ["lat_long"]
        df.index.names = ["topic", "doc"]

        topic_docs_coordinates = pd.concat([topic_docs_coordinates, df])
    return topic_docs_coordinates


def multidoc_df_to_model_results(df: pd.DataFrame, gold: dict, dataset_name: str):
    model_results = {dataset_name: []}
    true_labels = {dataset_name: []}
    for topic in df.index.get_level_values(0).unique():
        model_results[dataset_name].append(
            (
                df.loc[topic, ["semantic_score"]]
                .sort_values("semantic_score", ascending=False)
                .to_records(),
                df.loc[topic].index.to_list(),
            )
        )
        true_labels[dataset_name].append(gold[topic])
    return model_results, true_labels


def objective(
    trial,
    global_topic_data,
    docs_data,
    topic_candidate_document_matrix,
    gold,
    topic_docs_coordinates,
    stemmer,
    lemmer,
):
    params = {
        # "w_function": trial.suggest_categorical("w_function", ("inv_dist", "exp_dist")),
        # "lambda": trial.suggest_int("", 10, 1000),
        # "w_function": trial.suggest_categorical("w_function", ("inv_dist")),
        "w_function": "inv_dist",
        "w_function_param": trial.suggest_float("w_function_param", 1, 30000),
        "geo_ranking": trial.suggest_categorical(
            "geo_ranking",
            (
                "_score_w_geo_association_I",
                "_score_w_geo_association_C",
                "_score_w_geo_association_G",
            ),
        ),
        "lambda": trial.suggest_float("lambda", 0, 1),
        "gamma": trial.suggest_float("gamma", 0, 1),
    }

    copy_df = global_topic_data.copy()

    process_geo_associations_for_topics(
        copy_df,
        docs_data,
        topic_candidate_document_matrix,
        doc_coordinate_data=topic_docs_coordinates,
        w_function=getattr(geo_kpe_multidoc.geo.measures, params["w_function"]),
        w_function_param=params["w_function_param"],
        save_cache=False,
    )

    ranking_function = getattr(geo_kpe_multidoc.geo.utils, params["geo_ranking"])

    copy_df["semantic_score"] = ranking_function(
        copy_df,
        S="semantic_score",
        N="N",
        lambda_=params["lambda"],
        gamma=params["gamma"],
    )

    # F1@k=15 Eval
    model_results, true_labels = multidoc_df_to_model_results(
        copy_df, gold, dataset_name="teste"
    )

    stemmer = stemmer if stemmer else PorterStemmer()
    lemmer = lemmer if lemmer else "en"

    results = evaluate_kp_extraction(
        postprocess_res_labels(
            model_results=model_results, stemmer=stemmer, lemmer=lemmer
        ),
        postprocess_dataset_labels(true_labels, stemmer=stemmer, lemmer=lemmer),
        model_name="teste",
    )

    # print(
    #     results[
    #         ["Precision", "Recall", "F1", "MAP", "nDCG", "F1_5", "F1_10", "F1_15"]
    #     ].style.format("{:,.2%}")
    # )

    return results.loc["teste", "F1_15"]


def optimize(
    experiment: str,
    study_name: str,
    storage: str,
    stemmer=None,
    lemmer=None,
    TRIALS=100,
):
    (
        global_topic_data,
        docs_data,
        topic_candidate_document_matrix,
        gold,
    ) = multidoc_cache_to_frames(experiment)
    topic_docs_coordinates = gather_geo_locations(experiment)

    the_objective = partial(
        objective,
        global_topic_data=global_topic_data,
        docs_data=docs_data,
        topic_candidate_document_matrix=topic_candidate_document_matrix,
        gold=gold,
        topic_docs_coordinates=topic_docs_coordinates,
        stemmer=stemmer,
        lemmer=lemmer,
    )

    if storage:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),  # seed=SEED),
            # pruner=optuna.pruners.MedianPruner(n_warmup_steps=9),
        )
    study.optimize(the_objective, n_trials=TRIALS)  # , timeout=599)

    print("Number of finished trials: ", len(study.trials))
    print(f"Number of trials on the Pareto front: {len(study.best_trials)}")

    print(f"====================================================")
    for t in sorted(study.trials, reverse=True)[:3]:
        print(f"\tnumber: {t.number}")
        print(f"\tparams: {t.params}")
        # print(f"\tuser_attr: {t.user_attrs}")
        print(f"\tvalues: {t.values}")
        print(f"====================================================")
