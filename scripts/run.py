import argparse
import json
import sys
import textwrap
from datetime import datetime
from os import path
from time import time

import torch
from loguru import logger
from nltk.stem import PorterStemmer
from pandas import DataFrame
from sentence_transformers import SentenceTransformer

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH, GEO_KPE_MULTIDOC_DATA_PATH
from geo_kpe_multidoc.datasets.datasets import DATASETS, load_data
from geo_kpe_multidoc.evaluation.evaluation_tools import (
    evaluate_kp_extraction,
    extract_keyphrases_docs,
    extract_keyphrases_topics,
    output_one_top_cands,
    postprocess_dataset_labels,
    postprocess_res_labels,
)
from geo_kpe_multidoc.evaluation.mkduc01_eval import MKDUC01_Eval
from geo_kpe_multidoc.models import EmbedRank, FusionModel, MaskRank, MDKPERank
from geo_kpe_multidoc.models.backend._longmodels import to_longformer_t_v4
from geo_kpe_multidoc.models.embedrank.embedrank_longformer_manual import (
    EmbedRankManual,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_punctuation,
    remove_whitespaces,
)


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

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The input dataset name",
    )
    parser.add_argument(
        "--doc_embed_mode",
        default="mean",
        type=str,
        # required=True,
        help="The method for doc embedding.",
    )
    parser.add_argument(
        "--doc_mode",
        default="",
        type=str,
        # required=True,
        help="The method for doc mode (?).",
    )
    parser.add_argument(
        "--candidate_mode",
        default="",
        type=str,
        # required=True,
        help="The method for candidate mode (?).",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        help="Defines the embedding model to use",
        default="longformer-paraphrase-multilingual-mpnet-base-v2",
    )
    parser.add_argument(
        "--rank_model",
        default="EmbedRank",
        type=str,
        help="The Ranking Model [EmbedRank, EmbedRankManual, MaskRank, and FusionRank], MDKPERank",
        choices=["EmbedRank", "EmbedRankManual", "MaskRank", "FusionRank", "MDKPERank"],
    )
    parser.add_argument(
        "--doc_limit",
        default="-1",
        type=int,
        help="Max number of documents to process from Dataset.",
    )
    parser.add_argument(
        "--top_n",
        default="-1",
        type=int,
        help="Keep only Top N candidates",
    )
    parser.add_argument(
        "--experiment_name",
        default="run",
        type=str,
        help="Name to save experiment results.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        help="Weight list for Fusion Rank, in .2f",
        default="0.50 0.50",
    )
    parser.add_argument(
        "--ensemble_mode",
        type=str,
        default="weighted",
        help="Fusion model ensembling mode",
        # choices=[el.value for el in EnsembleMode],
        choices=["weighted", "harmonic"],
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="bool flag to use pos tags and embeds from cache",
    )
    parser.add_argument(
        "--stemming", action="store_true", help="bool flag to use stemming"
    )
    parser.add_argument(
        "--lemmatization", action="store_true", help="boolean flag to use lemmatization"
    )
    parser.add_argument(
        "--embedrank_mmr", action="store_true", help="boolean flag to use EmbedRank MMR"
    )
    parser.add_argument(
        "--embedrank_diversity",
        type=float,
        help="EmbedRank MMR diversity parameter value.",
    )
    parser.add_argument(
        "--cache_results",
        action="store_true",
        help="Save KPE Model outputs to cache directory.",
    )
    parser.add_argument(
        "--longformer_attention_window",
        type=int,
        default=512,
        help="Longformer: sliding chunk Attention Window size",
    )
    parser.add_argument(
        "--longformer_only_copy_to_max_position",
        type=int,
        help="Longformer: only copy first positions of Pretrained Model position embedding weights",
    )
    parser.add_argument(
        "--longformer_max_length",
        type=int,
        default=4096,
        help="Longformer: max length of the new model",
    )
    parser.add_argument(
        "--preprocessing",
        action="store_true",
        help="Preprocess text documents by removing pontuation",
    )
    return parser.parse_args()


def save(results: DataFrame, args):
    from geo_kpe_multidoc import GEO_KPE_MULTIDOC_OUTPUT_PATH

    t = datetime.now()
    filename = (
        "-".join(["results", args.experiment_name, t.strftime(r"%Y%m%d-%H%M%S")])
        + ".csv"
    )
    results.to_csv(path.join(GEO_KPE_MULTIDOC_OUTPUT_PATH, filename))
    logger.info(f"Results saved in {filename}")

    filename = filename[:-3] + "txt"
    with open(
        path.join(GEO_KPE_MULTIDOC_OUTPUT_PATH, filename), mode="w", encoding="utf8"
    ) as f:
        # f.write(args.__repr__())
        json.dump(args.__dict__, f, indent=4)


def generateLongformerRanker(backend_model_name, tagger_name, args):
    new_max_pos = args.longformer_max_length
    attention_window = args.longformer_attention_window
    copy_from_position = (
        args.longformer_only_copy_to_max_position
        if args.longformer_only_copy_to_max_position
        else None
    )

    model_name = f"longformer_paraphrase_mnet_max{new_max_pos}_attw{attention_window}"
    if copy_from_position:
        model_name += f"_cpmaxpos{copy_from_position}"

    model, tokenizer = to_longformer_t_v4(
        SentenceTransformer(backend_model_name),
        max_pos=new_max_pos,
        attention_window=attention_window,
        copy_from_position=copy_from_position,
    )
    # in RAM convertion to longformer needs this.
    del model.embeddings.token_type_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kpe_model = EmbedRankManual(
        model, tokenizer, tagger_name, device=device, name=model_name
    )
    return kpe_model


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

    TAGGER_NAME = DATASETS[ds_name].get("tagger")

    if args.rank_model == "EmbedRank":
        kpe_model = EmbedRank(BACKEND_MODEL_NAME, TAGGER_NAME)
    elif args.rank_model == "EmbedRankManual":
        kpe_model = generateLongformerRanker(BACKEND_MODEL_NAME, TAGGER_NAME, args)
    elif args.rank_model == "MaskRank":
        kpe_model = MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME)
    elif args.rank_model == "MDKPERank":
        if "longformer" in BACKEND_MODEL_NAME:
            base_name = BACKEND_MODEL_NAME.replace("longformer-", "")
            kpe_model = MDKPERank(
                generateLongformerRanker(base_name, TAGGER_NAME, args)
            )
        else:
            ranker = EmbedRank(BACKEND_MODEL_NAME, TAGGER_NAME)
            kpe_model = MDKPERank(ranker)
    elif args.rank_model == "FusionRank":
        kpe_model = FusionModel(
            [
                EmbedRank(BACKEND_MODEL_NAME, TAGGER_NAME),
                MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME),
            ],
            averaging_strategy=args.ensemble_mode,
            # models_weights=args.weights,
        )
    else:
        # raise ValueError("Model selection must be one of [EmbedRank, MaskRank].")
        logger.critical(
            "Model selection must be one of [EmbedRank, MaskRank, MDKPERank]."
        )
        sys.exit(-1)

    if isinstance(kpe_model, MDKPERank):
        extract_eval = extract_keyphrases_topics
        if ds_name != "MKDUC01":
            logger.critical("Not Multi Document Ranking on single document dataset!")
    else:
        extract_eval = extract_keyphrases_docs

    stemmer = PorterStemmer() if args.stemming else None
    lemmer = DATASETS[ds_name].get("language") if args.lemmatization else None

    if not lemmer:
        logger.warning("Running without lemmatization. Results will be poor.")

    options = dict()

    if args.embedrank_mmr:
        options["mmr"] = True
        if args.embedrank_diversity:
            options["mmr_diversity"] = args.embedrank_diversity
        else:
            logger.warning("EmbedRank MMR selected but diversity is default 0.8")
        if isinstance(kpe_model, MaskRank):
            logger.warning("EmbedRank MMR selected but model is not EmbedRank")

    if args.cache_results:
        options["cache_results"] = True
    if args.use_cache:
        options["use_cache"] = True
        options["pos_tag_cache"] = True

    if args.preprocess:
        options["preprocess"] = [remove_punctuation, remove_whitespaces]

    data = load_data(ds_name, GEO_KPE_MULTIDOC_DATA_PATH)
    logger.info(f"Args: {args}")
    logger.info("Start Testing ...")
    logger.info(f"KP extraction for {len(data)} examples.")
    logger.info(f"Options: {options}")

    options["experiment"] = args.experiment_name
    # -------------------------------------------------
    # --------------- Run Experiment ------------------
    # -------------------------------------------------
    model_results, true_labels = extract_eval(
        data,
        kpe_model,
        top_n=args.top_n,
        min_len=5,
        lemmer=lemmer,
        n_docs_limit=args.doc_limit,
        **options,
    )
    end = time()
    logger.info("Processing time: {}".format(end - start))

    # output_one_top_cands_geo(data.ids, model_results, true_labels)
    output_one_top_cands(data.ids, model_results, true_labels)

    # model_results["dataset_name"][(doc1_top_n, doc1_candidates), (doc2...)]
    assert stemmer != None
    # Transform keyphrases, thru stemming, for KPE evaluation.
    if not stemmer:
        logger.critical("KPE Evaluation usualy need stemmer!")
    else:
        model_results = postprocess_res_labels(model_results, stemmer, lemmer)
        true_labels = postprocess_dataset_labels(true_labels, stemmer, lemmer)
        import joblib

        joblib.dump(
            (model_results, true_labels),
            path.join(GEO_KPE_MULTIDOC_CACHE_PATH, "debug-predict-gold.pkl"),
        )

    results = evaluate_kp_extraction(model_results, true_labels)
    save(results, args)


if __name__ == "__main__":
    main()
