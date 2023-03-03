import argparse
import sys
import textwrap
import time
from datetime import datetime
from itertools import islice
from os import path

from loguru import logger

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_OUTPUT_PATH


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
        help="The input dataset name.",
    )
    parser.add_argument(
        "--doc_embed_mode",
        default="mean",
        type=str,
        # required=True,
        help="The method for doc embedding.",
    )
    parser.add_argument(
        "--rank_model",
        default="EmbedRank",
        type=str,
        help="The Ranking Model [EmbedRank, MaskRank]",
    )
    parser.add_argument(
        "--doc_limit",
        default="-1",
        type=int,
        help="Max number of documents to process from Dataset.",
    )

    parser.add_argument(
        "--experiment_name",
        default="run",
        type=str,
        help="Name to save experiment results.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    start = time.time()
    logger.info("Warmup")
    logger.info("Loading models")

    from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH
    from geo_kpe_multidoc.datasets import TextDataset
    from geo_kpe_multidoc.datasets.datasets import DATASETS, KPEDataset
    from geo_kpe_multidoc.evaluation.evaluation_tools import evaluate_kp_extraction
    from geo_kpe_multidoc.evaluation.mkduc01_eval import MKDUC01_Eval
    from geo_kpe_multidoc.models import EmbedRank, MaskRank
    from geo_kpe_multidoc.models.mdkperank.mdkperank_model import MDKPERank
    from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy

    BACKEND_MODEL_NAME = "longformer-paraphrase-multilingual-mpnet-base-v2"
    TAGGER_NAME = "en_core_web_trf"

    match args.rank_model:
        case "EmbedRank":
            kpe_model = EmbedRank(BACKEND_MODEL_NAME, TAGGER_NAME)
        case "MaskRank":
            kpe_model = MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME)
        case _:
            # raise ValueError("Model selection must be one of [EmbedRank, MaskRank].")
            logger.critical("Model selection must be one of [EmbedRank, MaskRank].")
            sys.exit(-1)

    logger.info("Start Testing ...")

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
    model_results = {}
    true_labels = {}
    ds_name = args.dataset_name
    if ds_name not in DATASETS.keys():
        logger.critical(
            f"Dataset {ds_name} is not supported. Select one of {list(DATASETS.keys())}"
        )
        sys.exit(-1)

    data = KPEDataset(
        ds_name, DATASETS[ds_name]["zip_file"], GEO_KPE_MULTIDOC_DATA_PATH
    )

    logger.info(f"KP extraction for {len(data)} examples.")
    # update SpaCy POS tagging for dataset language
    kpe_model.tagger = POS_tagger_spacy(DATASETS[ds_name]["tagger"])

    model_results[ds_name] = []
    true_labels[ds_name] = []

    if args.doc_limit == -1:
        loader = data
    else:
        loader = islice(data, args.doc_limit)

    for _doc_id, doc, gold_kp in loader:
        top_n_and_scores, candicates = kpe_model.extract_kp_from_doc(
            kpe_model.pre_process(doc),
            top_n=20,
            min_len=2,
            lemmer=DATASETS[ds_name]["language"],
        )
        model_results[ds_name].append(
            (
                top_n_and_scores,
                candicates,
            )
        )
        true_labels[ds_name].append(gold_kp)

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
