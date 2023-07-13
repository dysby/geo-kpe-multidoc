import io
import itertools
import json
import pickle
import re
from os import path
from typing import List, Tuple
from zipfile import ZipFile

from loguru import logger

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH
from geo_kpe_multidoc.datasets.kpedataset import KPEDataset
from geo_kpe_multidoc.datasets.promptrank_datasets import load_promptrankdataset

# Datasets from https://github.com/LIAAD/KeywordExtractor-Datasets
DATASETS = {
    # zip_file: str = Path to the zip file with docs and annoations.
    "DUC2001": {
        # source: https://github.com/boudinfl/ake-datasets
        "zip_file": "DUC2001.zip",
        "language": "en",
        "tagger": "en_core_web_trf",
        # "tagger": "en_core_web_lg",
    },
    "MKDUC01": {
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "110-PT-BN-KP": {
        "zip_file": "110-PT-BN-KP.zip",
        "language": "pt",
        "tagger": "pt_core_news_lg",
        # TODO: why pt_core_news_lg not trf?
    },
    "500N-KPCrowd": {
        "zip_file": "500N-KPCrowd-v1.1.zip",
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "cacic": {"zip_file": "cacic.zip", "language": "es", "tagger": "es_dep_news_trf"},
    "citeulike180": {
        "zip_file": "citeulike180.zip",
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "fao30": {"zip_file": "fao30.zip", "language": "en", "tagger": "en_core_web_trf"},
    "fao780": {"zip_file": "fao780.zip", "language": "en", "tagger": "en_core_web_trf"},
    "Inspec": {"zip_file": "Inspec.zip", "language": "en", "tagger": "en_core_web_trf"},
    "kdd": {"zip_file": "kdd.zip", "language": "en", "tagger": "en_core_web_trf"},
    "Krapivin2009": {
        "zip_file": "Krapivin2009.zip",
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "Nguyen2007": {
        "zip_file": "Nguyen2007.zip",
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "pak2018": {
        "zip_file": "pak2018.zip",
        "language": "pl",
        "tagger": "en_core_web_trf",  # TODO: select pl spacy model
    },
    "PubMed": {"zip_file": "PubMed.zip", "language": "en", "tagger": "en_core_web_trf"},
    "Schutz2008": {
        "zip_file": "Schutz2008.zip",
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "SemEval2010": {
        "zip_file": "SemEval2010.zip",
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "SemEval2017": {
        "zip_file": "SemEval2017.zip",
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "theses100": {
        "zip_file": "theses100.zip",
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "wicc": {"zip_file": "wicc.zip", "language": "es", "tagger": "es_dep_news_trf"},
    "wiki20": {"zip_file": "wiki20.zip", "language": "en", "tagger": "en_core_web_trf"},
    "WikiNews": {
        "zip_file": "WikiNews.zip",
        "language": "fr",
        "tagger": "fr_dep_news_trf",
    },
    "www": {"zip_file": "www.zip", "language": "en", "tagger": "en_core_web_trf"},
}


def load_dataset(
    name, datasource="base", root_dir=GEO_KPE_MULTIDOC_DATA_PATH
) -> KPEDataset:
    """
    name: Supported dataset name, must exist in DATASET dict
    root_dir: str = Data path.
    """

    def _read_mdkpe(dataset_dir):
        """Remove topic key and document id and keep only a list of items each corresponding to
        a topic, and each item composed by a list of docs and a list of keyphrases."""
        dataset = {}
        with open(f"{dataset_dir}/MKDUC01/MKDUC01.json", "r") as source_f:
            dataset = json.load(source_f)
        logger.info(f"Load json with {len(dataset)} topics")

        ids = []
        documents = []
        labels = []
        for topic in dataset:
            # TODO: simplify
            docs_content_for_topic = [
                (doc_name, doc_content)
                for doc_name, doc_content in dataset[topic]["documents"].items()
            ]
            kps_for_topic = list(itertools.chain(*dataset[topic]["keyphrases"]))

            ids.append(topic)
            documents.append(docs_content_for_topic)
            labels.append(kps_for_topic)

        if len(ids) == 0:
            logger.warning("Extracted **zero** results")
        return (ids, documents, labels)

    def _read_zip(filename) -> Tuple[List[str], List, List]:
        """
        read documents from docsutf8 dir w/ .txt, and read keys from keys dir w/ .key

        ex:
            "keys/2000_10_11-21_00_00-Jornal2-2-topic-seg.txt-Nr9.key"
        to
            id = "2000_10_11-21_00_00-Jornal2-2-topic-seg.txt-Nr9"
        """
        ids = []
        documents = []
        labels = []

        with ZipFile(filename, mode="r") as zf:
            for file in zf.namelist():
                if file[-4:] == ".key":  # optional filtering by filetype
                    id = file.split("/")[-1][:-4]
                    ids.append(id)
                    with zf.open(file) as f:
                        labels.append(
                            [
                                line.strip()
                                for line in io.TextIOWrapper(f, encoding="utf8")
                            ]
                        )
                    # replace keys dir with doc dir
                    doc_file = re.sub("keys", "docsutf8", file)
                    doc_file = doc_file[:-3] + "txt"
                    with zf.open(doc_file) as f:
                        documents.append(f.read().decode("utf8"))
        return ids, documents, labels

    if datasource == "preloaded":
        return load_preprocessed(name)
    elif datasource == "promptrank":
        return load_promptrankdataset(name)

    zipfile = DATASETS[name].get("zip_file", None)
    if zipfile:
        return KPEDataset(name, *_read_zip(path.join(root_dir, zipfile)))
    else:
        return KPEDataset(name, *_read_mdkpe(root_dir))


def load_preprocessed(name, root_dir=GEO_KPE_MULTIDOC_DATA_PATH) -> KPEDataset:
    subdir = "processed"

    # DE-TeKET_processed.txt  ES-WICC_processed.txt  MKDUC01_processed.txt  PubMed_processed.txt
    # DUC_processed.txt       FR-WIKI_processed.txt  NUS_processed.txt      SemEval_processed.txt
    # ES-CACIC_processed.txt  Inspec_processed.txt   PT-KP_processed.txt

    local_name = {
        "DUC2001": "DUC",
        "MKDUC01": "MKDUC01",
        "110-PT-BN-KP": "PT-KP",
        "cacic": "ES-CACIC",
        "Inspec": "Inspec",
        "Nguyen2007": "NUS",
        "PubMed": "PubMed",
        "SemEval2010": "SemEval",
        "wicc": "ES-WICC",
        "WikiNews": "FR-WIKI",
    }

    with open(
        path.join(root_dir, subdir, f"{local_name[name]}_processed.txt"), mode="rb"
    ) as f:
        docs_and_keys = pickle.load(f)

    ids = list(range(len(docs_and_keys)))
    docs, labels = list(zip(*docs_and_keys))

    return KPEDataset(name, ids, docs, labels)
