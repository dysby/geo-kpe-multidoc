import argparse
import itertools
import time
from itertools import product
from typing import Callable, List

import numpy as np
from nltk.stem import PorterStemmer
from tqdm import tqdm

from geo_kpe_multidoc.datasets.process_datasets import *
from geo_kpe_multidoc.evaluation.config import EMBEDS_DIR, POS_TAG_DIR
from geo_kpe_multidoc.evaluation.evaluation_tools import (
    evaluate_kp_extraction,
    extract_dataset_labels,
    extract_res_labels,
    extract_res_labels_x,
    output_top_cands,
)
from geo_kpe_multidoc.models.candidate_extract.candidate_extract_model import (
    CandidateExtract,
)
from geo_kpe_multidoc.models.embedrank.embedrank_model import EmbedRank
from geo_kpe_multidoc.models.fusion_model import FusionModel
from geo_kpe_multidoc.models.graphrank.graphrank_model import GraphRank
from geo_kpe_multidoc.models.maskrank.maskrank_model import MaskRank
from geo_kpe_multidoc.models.mdkperank.mdkperank_model import MDKPERank
from geo_kpe_multidoc.models.pre_processing.dataset_embeddings_memory import (
    EmbeddingsMemory,
)
from geo_kpe_multidoc.models.pre_processing.language_mapping import (
    choose_lemmatizer,
    choose_tagger,
)
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy
from geo_kpe_multidoc.models.pre_processing.post_processing_utils import (
    z_score_normalization,
)


def str2bool(input_str: str):
    if str(input_str).lower() in ["yes", "y", "true", "t"]:
        return True
    elif str(input_str).lower() in ["no", "n", "false", "f"]:
        return False
    return None


def save_model_pos_tags(
    dataset_obj: Callable, pos_tagger_model: str, model: Callable
) -> None:
    for dataset in dataset_obj.dataset_content:
        model.update_tagger(dataset)

        if not os.path.isdir(f"{POS_TAG_DIR}{dataset}/"):
            os.mkdir(f"{POS_TAG_DIR}{dataset}/")

        model.tagger.pos_tag_to_file(
            dataset_obj.dataset_content[dataset],
            f"{POS_TAG_DIR}{dataset}/{pos_tagger_model}/",
            0,
        )


def save_model_embeds(
    dataset_obj: Callable, embed_model: str, pos_tagger_model: str, model: Callable
) -> None:
    for dataset in dataset_obj.dataset_content:
        model.update_tagger(dataset)

        if not os.path.isdir(f"{EMBEDS_DIR}{dataset}/"):
            os.mkdir(f"{EMBEDS_DIR}{dataset}/")

        mem = EmbeddingsMemory(dataset_obj)
        mem.save_embeddings(
            dataset_obj,
            model.model,
            f"{embed_model}",
            EMBEDS_DIR,
            POS_tagger_spacy(f"{pos_tagger_model}"),
            False,
            0,
        )


def run_single_model(
    datasets: List[str],
    embed_model: str,
    pos_tagger_model: str,
    model_class: Callable,
    save_pos_tags: bool,
    save_embeds: bool,
    use_memory: bool,
    stemming: bool,
    lemmatize: bool,
    doc_cand_modes: List[List[str]],
    **kwargs,
) -> None:
    dataset_obj = DataSet(datasets)
    model = model_class(f"{embed_model}", f"{pos_tagger_model}")

    if save_pos_tags:
        save_model_pos_tags(dataset_obj, pos_tagger_model, model)

    if save_embeds:
        save_model_embeds(dataset_obj, embed_model, pos_tagger_model, model)

    for d_mode, c_mode in doc_cand_modes:
        res = {}
        for dataset in dataset_obj.dataset_content:
            if use_memory == True:
                pos_tag_memory_dir = f"{POS_TAG_DIR}{dataset}/{pos_tagger_model}/"
                embed_memory_dir = f"{EMBEDS_DIR}{dataset}/{embed_model}/"

                res[dataset] = model.extract_kp_from_corpus(
                    dataset_obj.dataset_content[dataset],
                    dataset,
                    15,
                    5,
                    stemming,
                    lemmatize,
                    doc_mode=d_mode,
                    cand_mode=c_mode,
                    pos_tag_memory=pos_tag_memory_dir,
                    embed_memory=embed_memory_dir,
                    **kwargs,
                )

            else:
                res[dataset] = model.extract_kp_from_corpus(
                    dataset_obj.dataset_content[dataset],
                    dataset,
                    15,
                    5,
                    stemming,
                    lemmatize,
                    doc_mode=d_mode,
                    cand_mode=c_mode,
                    **kwargs,
                )

        stemmer = PorterStemmer()
        lemmer = choose_lemmatizer(dataset) if lemmatize else None
        evaluate_kp_extraction(
            extract_res_labels(res, stemmer, lemmer),
            extract_dataset_labels(dataset_obj.dataset_content, stemmer, lemmer),
            model.name,
            True,
            True,
            k_set=[5, 10, 15],
        )

    return


def run_fusion_model(
    datasets: List[str],
    embed_model: str,
    pos_tagger_model: str,
    models: List[Callable],
    save_pos_tags: bool,
    save_embeds: bool,
    use_memory: bool,
    stemming: bool,
    lemmatize: bool,
    doc_cand_modes: List[List[str]],
    weights: List[float],
    **kwargs,
) -> None:
    dataset_obj = DataSet(datasets)
    model_list = [model(f"{embed_model}", f"{pos_tagger_model}") for model in models]
    kpe_model = FusionModel(model_list, weights)

    if save_pos_tags:
        for model in models:
            save_model_pos_tags(dataset_obj, pos_tagger_model, model)

    if save_embeds:
        for m in models:
            save_model_embeds(dataset_obj, embed_model, pos_tagger_model, model)

    for d_mode, c_mode in doc_cand_modes:
        res = {}
        for dataset in dataset_obj.dataset_content:
            if use_memory == True:
                pos_tag_memory_dir = f"{POS_TAG_DIR}{dataset}/{pos_tagger_model}/"
                embed_memory_dir = f"{EMBEDS_DIR}{dataset}/{embed_model}/"

                print(embed_memory_dir)

                res[dataset] = kpe_model.extract_kp_from_corpus(
                    dataset_obj.dataset_content[dataset],
                    dataset,
                    15,
                    5,
                    stemming,
                    lemmatize,
                    doc_mode=d_mode,
                    cand_mode=c_mode,
                    pos_tag_memory=pos_tag_memory_dir,
                    embed_memory=embed_memory_dir,
                    **kwargs,
                )

            else:
                res[dataset] = kpe_model.extract_kp_from_corpus(
                    dataset_obj.dataset_content[dataset],
                    dataset,
                    15,
                    5,
                    stemming,
                    doc_mode=d_mode,
                    cand_mode=c_mode,
                    **kwargs,
                )

        stemmer = PorterStemmer() if stemming else None
        lemmer = choose_lemmatizer(dataset) if lemmatize else None

        evaluate_kp_extraction(
            extract_res_labels(res, stemmer, lemmer),
            extract_dataset_labels(dataset_obj.dataset_content, stemmer, lemmer),
            model_name=kpe_model.name,
            save=True,
            kp_eval=True,
            k_set=[5, 10, 15],
        )
        output_top_cands(
            extract_res_labels_x(res, stemmer, lemmer),
            extract_dataset_labels(dataset_obj.dataset_content, stemmer, lemmer),
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        "--list",
        nargs="+",
        help="Choose which datasets to run from the list of supported ones",
        default="DUC",
    )
    parser.add_argument(
        "--embed_model",
        type=str,
        help="Defines the embedding model to use",
        default="longformer-paraphrase-multilingual-mpnet-base-v2",
    )
    parser.add_argument(
        "--rank_method",
        "--list",
        nargs="+",
        help="Choose which rank method to use from [EmbedRank, MaskRank and FusionRank]",
        default="EmbedRank",
    )
    parser.add_argument(
        "--weights",
        "--list",
        nargs="+",
        help="Weight list for Fusion Rank, in .2f",
        default="0.50 0.50",
    )
    parser.add_argument(
        "--save_pos_tags", type=str, help="bool flag to save POS tags", default="False"
    )
    parser.add_argument(
        "--save_embeds",
        type=str,
        help="bool flag to save generated embeds",
        default="False",
    )
    parser.add_argument(
        "--use_memory",
        type=str,
        help="bool flag to use pos tags and embeds from memory",
        default="False",
    )
    parser.add_argument(
        "--stemming", type=str, help="bool flag to use stemming", default="False"
    )
    parser.add_argument(
        "--lemmatization",
        type=str,
        help="bool flag to use lemmatization",
        default="False",
    )
    args = parser.parse_args()

    torch.cuda.is_available = lambda: False

    doc_cand_modes = itertools.product([""], [""])

    if len(args.rank_method) == 1:
        run_single_model(
            args.datasets,
            args.embed_model,
            choose_tagger(args.datasets[0]),
            args.rank_method[0],
            doc_cand_modes,
            str2bool(args.save_pos_tags),
            str2bool(args.save_embeds),
            str2bool(args.use_memory),
            str2bool(args.stemming),
            str2bool(args.lemmatization),
        )

    else:
        run_fusion_model(
            args.datasets,
            args.embed_model,
            choose_tagger(args.datasets[0]),
            args.rank_method,
            doc_cand_modes,
            [0.50, 0.50],
            str2bool(args.save_pos_tags),
            str2bool(args.save_embeds),
            str2bool(args.use_memory),
            str2bool(args.stemming),
            str2bool(args.lemmatization),
        )
