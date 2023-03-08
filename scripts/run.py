import argparse
import sys
import textwrap
import time
from datetime import datetime
from itertools import islice
from os import path

import simplemma
from loguru import logger
from nltk.stem import PorterStemmer

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_OUTPUT_PATH
from geo_kpe_multidoc.datasets.datasets import load_data
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.evaluation.evaluation_tools import (
    extract_keyphrases_docs,
    extract_keyphrases_topics,
)
from geo_kpe_multidoc.models.fusion_model import EnsembleMode


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
        Please do not mess up this text!
        --------------------------------
            I have indented it
            exactly the way
            I want it
        """
        ),
    )

    parser.add_argument(
        "--dataset_name",
        default=None,
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
        help="The Ranking Model [EmbedRank, MaskRank, and FusionRank], MDKPERank",
        choices=["EmbedRank", "MaskRank", "FusionRank", "MDKPERank"],
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
        choices=[el.value for el in EnsembleMode],
    )
    parser.add_argument(
        "--save_pos_tags", type=bool, help="bool flag to save POS tags", default=False
    )
    parser.add_argument(
        "--save_embeds",
        type=bool,
        help="bool flag to save generated embeds",
        default=False,
    )
    parser.add_argument(
        "--use_cache",
        type=bool,
        help="bool flag to use pos tags and embeds from cache",
        default=False,
    )
    parser.add_argument(
        "--stemming", type=bool, help="bool flag to use stemming", default=False
    )
    parser.add_argument(
        "--lemmatization",
        type=bool,
        help="bool flag to use lemmatization",
        default=False,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    start = time.time()
    logger.info("Warmup")
    logger.info("Loading models")

    from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH
    from geo_kpe_multidoc.datasets.datasets import DATASETS, KPEDataset
    from geo_kpe_multidoc.evaluation.evaluation_tools import evaluate_kp_extraction
    from geo_kpe_multidoc.evaluation.mkduc01_eval import MKDUC01_Eval
    from geo_kpe_multidoc.models import EmbedRank, FusionModel, MaskRank, MDKPERank
    from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy

    # "longformer-paraphrase-multilingual-mpnet-base-v2"
    BACKEND_MODEL_NAME = args.embed_model

    ds_name = args.dataset_name
    if ds_name not in DATASETS.keys():
        logger.critical(
            f"Dataset {ds_name} is not supported. Select one of {list(DATASETS.keys())}"
        )
        sys.exit(-1)

    TAGGER_NAME = DATASETS[ds_name].get("tagger")

    data = load_data(ds_name, GEO_KPE_MULTIDOC_DATA_PATH)

    match args.rank_model:
        case "EmbedRank":
            kpe_model = EmbedRank(BACKEND_MODEL_NAME, TAGGER_NAME)
        case "MaskRank":
            kpe_model = MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME)
        case "MDKPERank":
            kpe_model = MDKPERank(BACKEND_MODEL_NAME, TAGGER_NAME)
        case "FusionRank":
            kpe_model = FusionModel(
                [
                    EmbedRank(BACKEND_MODEL_NAME, TAGGER_NAME),
                    MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME),
                ],
                averaging_strategy=args.ensemble_mode,
                # models_weights=args.weights,
            )
        case _:
            # raise ValueError("Model selection must be one of [EmbedRank, MaskRank].")
            logger.critical(
                "Model selection must be one of [EmbedRank, MaskRank, MDKPERank]."
            )
            sys.exit(-1)

    # ori_encode_dict = tokenizer.encode_plus(
    #     doc,  # Sentence to encode.
    #     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
    #     max_length=MAX_LEN,  # Pad & truncate all sentences.
    #     padding='max_length',
    #     return_attention_mask=True,  # Construct attn. masks.
    #     return_tensors='pt',  # Return pytorch tensors.
    #     truncation=True
    # )

    # dataloader = DataLoader(dataset, batch_size=args.batch_size)
    logger.info("Start Testing ...")
    logger.info(f"KP extraction for {len(data)} examples.")
    # update SpaCy POS tagging for dataset language
    kpe_model.tagger = POS_tagger_spacy(DATASETS[ds_name]["tagger"])

    if args.doc_limit == -1:
        loader = data
    else:
        loader = islice(data, args.doc_limit)

    if isinstance(kpe_model, MDKPERank):
        extract_eval = extract_keyphrases_topics
    else:
        extract_eval = extract_keyphrases_docs

    stemmer = PorterStemmer() if args.stemming else None
    lemmer = DATASETS[ds_name].get("language") if args.lemmatization else None

    model_results, true_labels = extract_eval(
        loader, kpe_model, top_n=args.top_n, min_len=5, lemmer=lemmer
    )

    # model_results["dataset_name"][(doc1_top_n, doc1_candidates), (doc2...)]
    results = evaluate_kp_extraction(model_results, true_labels)
    # keyphrases_selection(doc_list, labels_stemed, labels, model, dataloader, log)

    end = time.time()
    logger.info("Processing time: {}".format(end - start))

    t = datetime.now()
    filename = (
        "-".join(["results", args.experiment_name, t.strftime("%Y%m%d-%H%M%S")])
        + ".csv"
    )
    results.to_csv(path.join(GEO_KPE_MULTIDOC_OUTPUT_PATH, filename))
    logger.info(f"Results saved in {filename}")


if __name__ == "__main__":
    main()
