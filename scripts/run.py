import argparse
import json
import sys
import textwrap
from datetime import datetime
from os import path
from time import time

import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from tabulate import tabulate

import wandb
from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DEBUG, GEO_KPE_MULTIDOC_OUTPUT_PATH
from geo_kpe_multidoc.datasets.datasets import DATASETS, load_dataset
from geo_kpe_multidoc.evaluation.evaluation_tools import (
    evaluate_kp_extraction,
    evaluate_kp_extraction_base,
    extract_keyphrases_docs,
    extract_keyphrases_topics,
    model_scores_to_dataframe,
    postprocess_model_outputs,
)
from geo_kpe_multidoc.evaluation.report import (
    output_one_top_cands,
    plot_score_distribuitions_with_gold,
    table_latex,
)
from geo_kpe_multidoc.models import MDKPERank
from geo_kpe_multidoc.models.factory import kpe_model_factory
from geo_kpe_multidoc.models.maskrank.maskrank_model import MaskRank
from geo_kpe_multidoc.models.mdkperank.mdpromptrank import MdPromptRank
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_new_lines_and_tabs,
    remove_whitespaces,
    select_stemmer,
)
from geo_kpe_multidoc.models.promptrank.promptrank import PromptRank


# fmt: off
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Key-phrase extraction
        --------------------------------
            Multi language
            Multi document
            Geospatial association measures
        """
        ),
    )
    parser.add_argument("--experiment_name", default="run", type=str, help="Name to save experiment results.",)
    parser.add_argument("--dataset_name", type=str, required=True, help="The input dataset name",)
    parser.add_argument("--dataset_source", type=str, default="base", help="Dataset sources [base, preloaded, promptrank]",)
    parser.add_argument("--preprocessing", action="store_true", help="Preprocess text documents by removing pontuation",)
    parser.add_argument("--extraction_variant", default="base", type=str, help="Set Extraction model variant [base, promptrank]",)
    parser.add_argument("--kp_min_len", type=int, default=0, help="Candidate keyphrase minimum length",)
    parser.add_argument("--kp_max_words", type=int, help="Candidate keyphrase maximum words",)
    parser.add_argument("--lemmatization", action="store_true", help="boolean flag to use lemmatization")
    parser.add_argument("--rank_model", default="EmbedRank", type=str, help="The Ranking Model", choices=[ "EmbedRank", "MaskRank", "PromptRank", "FusionRank", "MDKPERank", "MdPromptRank", "ExtractionEvaluator", ],)
    parser.add_argument("--embed_model", type=str, help="Defines the embedding model to use", default="paraphrase-multilingual-mpnet-base-v2",)
    parser.add_argument("--longformer_max_length", type=int, default=4096, help="longformer: max length of the new model",)
    parser.add_argument("--longformer_attention_window", type=int, default=512, help="Longformer: sliding chunk Attention Window size",)
    parser.add_argument("--longformer_only_copy_to_max_position", type=int, help="Longformer: only copy first positions of Pretrained Model position embedding weights",)
    parser.add_argument("--max_seq_len", type=int, help="PromptRank: max input sequence length because each model have a differenter config key especification")
    parser.add_argument("--batch_size", type=int, help="PromptRank: decoding batch size")
    parser.add_argument("--encoder_prompt", type=str, help="PromptRank: encoder prompt, default 'Book: ' ")
    parser.add_argument("--decoder_prompt", type=str, help="PromptRank: decoder prompt, default 'This book mainly talks about ' ")
    parser.add_argument("--no_position_feature", action="store_true", help="PromptRank: use candidate position as aditional feature (default: True)")
    parser.add_argument("--add_query_prefix", action="store_true", help="Add support for e5 type models that require 'query: ' prefixed instruction",)
    parser.add_argument("--candidate_mode", default="mentions_no_context", type=str, help="The method for candidate mode (no_context, mentions_no_context, global_attention, global_attention_dilated_nnn, attention_rank).",)
    parser.add_argument("--md_strategy", default="MEAN", type=str, help="Candidate ranking method for Multi-document keyphrase extraction",)
    parser.add_argument("--md_cross_doc", action="store_true", help="Do a candidate cross doc evaluation even when candidate is not found in doc",)
    parser.add_argument("--mmr", action="store_true", help="boolean flag to use EmbedRank MMR")
    parser.add_argument("--mmr_diversity", type=float, help="EmbedRank MMR diversity parameter value.",)
    parser.add_argument("--whitening", action="store_true", help="Apply whitening to the embeddings")
    parser.add_argument("--tagger_name", type=str, help="Explicit use this Spacy tagger",)
    parser.add_argument("--no_stemming", action="store_true", help="bool flag to use stemming")
    parser.add_argument("--ensemble_mode", type=str, default="weighted", help="Fusion model ensembling mode", choices=["weighted", "harmonic"],)
    parser.add_argument("--weights", nargs="+", help="Weight list for Fusion Rank, in .2f", default="0.50 0.50",)
    parser.add_argument("--top_n", default=-1, type=int, help="Keep only Top N candidates",)
    # parser.add_argument( "--pooling", type=str, default="mean", help="[NOT USED] Embedding Pooling strategy [mean, max]",)
    parser.add_argument("--doc_limit", default=-1, type=int, help="Max number of documents to process from Dataset",)
    parser.add_argument("--doc_name", type=str, help="Doc ID to test from Dataset.",)
    parser.add_argument("--cache_pos_tags", action="store_true", help="Save/Load doc POS Tagging in cache directory.",)
    parser.add_argument("--cache_candidate_selection", action="store_true", help="Save/Load doc Candidat in cache directoryy.",)
    parser.add_argument("--cache_embeddings", action="store_true", help="Save/Load doc and candidates embeddings in cache directory.",)
    parser.add_argument("--cache_md_embeddings", action="store_true", help="Save/Load topic embeddings in cache directory.",)
    parser.add_argument("--cache_results", action="store_true", help="Save KPE Model outputs (top N per doc/topic) to cache directory.",
    )
    return parser.parse_args()
# fmt: on


def save(
    dataset_kpe: pd.DataFrame, performance_metrics: pd.DataFrame, fig: plt.Figure, args
):
    t = datetime.now().strftime(r"%Y%m%d-%H%M")
    filename = "-".join(["kpe", args.experiment_name, t]) + ".csv"
    dataset_kpe.to_csv(path.join(GEO_KPE_MULTIDOC_OUTPUT_PATH, filename))

    filename = "-".join(["results", args.experiment_name, t]) + ".csv"
    performance_metrics.to_csv(path.join(GEO_KPE_MULTIDOC_OUTPUT_PATH, filename))

    filename = filename[:-3] + "txt"
    with open(
        path.join(GEO_KPE_MULTIDOC_OUTPUT_PATH, filename), mode="w", encoding="utf8"
    ) as f:
        # f.write(args.__repr__())
        json.dump(vars(args), f, indent=4)

    filename = filename[:-3] + "pdf"
    fig.savefig(path.join(GEO_KPE_MULTIDOC_OUTPUT_PATH, filename), dpi=100)

    logger.info(f"Results saved in {filename}")


def write_resume_txt(performance_metrics, args):
    with open("runs.resume.txt", "a", encoding="utf-8") as f:
        stamp = datetime.now().strftime(r"%Y%m%d-%H%M")
        print(f"Date: {stamp}", file=f)
        print(f"Args: {args}", file=f)
        print(
            tabulate(
                performance_metrics[
                    [
                        # "Precision",
                        "Recall",
                        # "F1",
                        # "MAP",
                        "nDCG",
                        "F1_5",
                        "F1_10",
                        "F1_15",
                    ]
                ],
                headers="keys",
                floatfmt=".2%",
            ),
            file=f,
        )

    with open("runs.resume.latex.txt", "a", encoding="utf-8") as f:
        stamp = datetime.now().strftime(r"%Y%m%d-%H%M")
        print(f"Date: {stamp}", file=f)
        print(f"Args: {args}", file=f)
        print(
            table_latex(
                performance_metrics[
                    [
                        "Recall",
                        "nDCG",
                        "F1_5",
                        "F1_10",
                        "F1_15",
                    ]
                ],
                caption=str(args),
                label=f"{args.experiment_name}-{stamp}".replace("_", " "),
                percentage=True,
            ),
            file=f,
        )


def _args_to_options(args):
    options = {}

    options["experiment"] = args.experiment_name

    if args.mmr:
        options["mmr"] = True
        logger.warning("MMR is only used with EmbedRank type models.")
        if args.mmr_diversity is not None:
            options["mmr_diversity"] = args.mmr_diversity
        else:
            logger.warning("EmbedRank MMR selected but diversity is default 0.8")
        # if isinstance(kpe_model, MaskRank):
        #     logger.warning("EmbedRank MMR selected but model is not EmbedRank")

    if args.cache_results:
        options["cache_results"] = True
    if args.cache_pos_tags:
        options["cache_pos_tags"] = True
        # options["use_cache"] = True
        # options["pos_tag_cache"] = True

    if args.cache_candidate_selection:
        options["cache_candidate_selection"] = True

    if args.cache_embeddings:
        options["cache_embeddings"] = True
    if args.cache_md_embeddings:
        options["cache_md_embeddings"] = True
    if args.preprocessing:
        # options["preprocess"] = [remove_special_chars, remove_whitespaces]
        options["preprocessing"] = [remove_new_lines_and_tabs, remove_whitespaces]

    if args.candidate_mode:
        options["cand_mode"] = args.candidate_mode

    if args.kp_min_len:
        options["kp_min_len"] = args.kp_min_len

    return options


def main():
    args = parse_args()

    start = time()
    logger.info("Warmup")
    logger.info("Loading models")

    # "longformer-paraphrase-multilingual-mpnet-base-v2"
    BACKEND_MODEL_NAME = args.embed_model

    ds_name = args.dataset_name
    if ds_name not in DATASETS.keys():
        logger.critical(
            f"Dataset {ds_name} is not supported. Select one of {list(DATASETS.keys())}"
        )
        sys.exit(-1)

    TAGGER_NAME = (
        args.tagger_name if args.tagger_name else DATASETS[ds_name].get("tagger")
    )

    LANGUAGE = DATASETS[ds_name].get("language")

    kpe_model = kpe_model_factory(
        BACKEND_MODEL_NAME, TAGGER_NAME, language=LANGUAGE, **vars(args)
    )

    if isinstance(kpe_model, (MDKPERank, MdPromptRank)):
        extract_keyphases_pipeline = extract_keyphrases_topics
        if "MKDUC01" not in ds_name:
            logger.critical("Multi Document Ranking on single document dataset!")
    else:
        extract_keyphases_pipeline = extract_keyphrases_docs

    stemmer = (
        None if args.no_stemming else select_stemmer(DATASETS[ds_name].get("language"))
    )
    assert stemmer is not None
    if not stemmer:
        logger.critical("KPE Evaluation usually need stemmer!")
    lemmer = DATASETS[ds_name].get("language") if args.lemmatization else None

    options = _args_to_options(args)

    dataset = load_dataset(ds_name, datasource=args.dataset_source)

    logger.info(f"Args: {args}")
    logger.info("Start Testing ...")
    logger.info(f"KP extraction for {len(dataset)} examples.")
    logger.info(f"Options: {options}")

    n_docs_limit = args.doc_name or args.doc_limit

    # -------------------------------------------------
    # --------------- Run Experiment ------------------
    # -------------------------------------------------
    with wandb.init(
        project="geo-kpe-multidoc", name=f"{args.experiment_name}", config=vars(args)
    ):
        model_results, true_labels = extract_keyphases_pipeline(
            dataset,
            kpe_model,
            top_n=args.top_n,
            lemmer=lemmer,
            n_docs_limit=n_docs_limit,
            **options,
        )

        # DEBUG PromptRank Original Evaluation
        if isinstance(kpe_model, PromptRank):
            # model_results, true_labels are aligned by key (dataset_name)
            for dataset_model_results, dataset_true_labels in zip(
                model_results.values(), true_labels.values()
            ):
                dataset_top_k = []
                for cand_score, cand in dataset_model_results:
                    top_k, score = list(zip(*cand_score))
                    # append document top_k
                    dataset_top_k.append(top_k)

                dataset_labels = dataset_true_labels
                # if no_stemming option is selected this will break
                kpe_model._evaluate(dataset_top_k, dataset_labels, stemmer)

        if stemmer:
            model_results = postprocess_model_outputs(
                model_results, stemmer, lemmer, options.get("preprocessing", [])
            )
            # True labels are preprocessed while loading Dataset.

        dataset_kpe = model_scores_to_dataframe(model_results, true_labels)

        # xlim = (
        #     (dataset_kpe["score"].min(), dataset_kpe["score"].max())
        #     if isinstance(kpe_model, (PromptRank, MaskRank))
        #     or not args.no_position_feature
        #     else (0, 1)
        # )
        fig = plot_score_distribuitions_with_gold(
            results=dataset_kpe,
            title=args.experiment_name.replace("-", " "),
            # xlim=xlim,
        )

        # mlflow.log_figure(fig, artifact_file="score_distribution.png")
        wandb.log({"score_distribution": wandb.Image(fig)})

        performance_metrics = evaluate_kp_extraction_base(model_results, true_labels)
        performance_metrics = pd.concat(
            [performance_metrics, evaluate_kp_extraction(model_results, true_labels)]
        )

        # mlflow.log_text(kpe_for_doc, artifact_file="first-doc-extraction-sample.txt")
        kpe_for_doc = output_one_top_cands(dataset.ids, model_results, true_labels)
        wandb.log({"first-doc-extraction-sample": kpe_for_doc})

        metric_names = [
            "_base_" + value for value in performance_metrics.iloc[0].index.values
        ]

        metrics = performance_metrics.iloc[0]
        metrics.index = metric_names
        all_metrics = metrics.to_dict()

        # mlflow.log_metrics(metrics.to_dict())
        metrics = performance_metrics.iloc[1]
        # mlflow.log_metrics(metrics.to_dict())

        all_metrics.update(metrics.to_dict())
        wandb.log(all_metrics)

    print(kpe_for_doc)
    logger.info(f"Args: {args}")
    # Print and Save Results
    print(
        tabulate(
            performance_metrics[
                # ["Precision", "Recall", "F1", "MAP", "nDCG", "F1_5", "F1_10", "F1_15"]
                ["Recall", "nDCG", "F1_5", "F1_10", "F1_15"]
            ],
            headers="keys",
            floatfmt=".2%",
        )
    )

    if not GEO_KPE_MULTIDOC_DEBUG:
        write_resume_txt(performance_metrics, args)
        save(dataset_kpe, performance_metrics, fig, args)

    end = time()
    logger.info(f"Processing time: {end - start:.1f}")


if __name__ == "__main__":
    main()
