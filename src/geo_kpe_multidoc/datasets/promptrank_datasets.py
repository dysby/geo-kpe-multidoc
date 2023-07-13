import codecs
import json
import os
import re

import tqdm
from torch.utils.data import Dataset

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH
from geo_kpe_multidoc.datasets.kpedataset import KPEDataset


class PromptRankDataset(Dataset):
    def __init__(self, docs_pairs):
        self.docs_pairs = docs_pairs
        self.total_examples = len(self.docs_pairs)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        doc_pair = self.docs_pairs[idx]
        en_input_ids = doc_pair[0][0]
        en_input_mask = doc_pair[1][0]
        de_input_ids = doc_pair[2][0]
        dic = doc_pair[3]

        return [en_input_ids, en_input_mask, de_input_ids, dic]


def clean_text(text="", database="inspec"):
    # Specially for Duc2001 Database
    if database == "duc2001" or database == "semeval2017":
        pattern2 = re.compile(r"[\s,]" + "[\n]{1}")
        while True:
            if pattern2.search(text) is not None:
                position = pattern2.search(text)
                start = position.start()
                end = position.end()
                # start = int(position[0])
                text_new = text[:start] + "\n" + text[start + 2 :]
                text = text_new
            else:
                break

    pattern2 = re.compile(r"[a-zA-Z0-9,\s]" + "[\n]{1}")
    while True:
        if pattern2.search(text) is not None:
            position = pattern2.search(text)
            start = position.start()
            end = position.end()
            # start = int(position[0])
            text_new = text[: start + 1] + " " + text[start + 2 :]
            text = text_new
        else:
            break

    pattern3 = re.compile(r"\s{2,}")
    while True:
        if pattern3.search(text) is not None:
            position = pattern3.search(text)
            start = position.start()
            # end = position.end()
            # start = int(position[0])
            text_new = text[: start + 1] + "" + text[start + 2 :]
            text = text_new
        else:
            break

    pattern1 = re.compile(r"[<>[\]{}]")
    text = pattern1.sub(" ", text)
    text = text.replace("\t", " ")
    text = text.replace(" p ", "\n")
    text = text.replace(" /p \n", "\n")
    lines = text.splitlines()
    # delete blank line
    text_new = ""
    for line in lines:
        if line != "\n":
            text_new += line + "\n"

    return text_new


def get_long_data(file_path="promptrank/nus/nus_test.json"):
    """Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(
        os.path.join(GEO_KPE_MULTIDOC_DATA_PATH, file_path), "r", "utf-8"
    ) as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl["keywords"].lower().split(";")
                abstract = jsonl["abstract"]
                fulltxt = jsonl["fulltext"]
                doc = " ".join([abstract, fulltxt])
                doc = re.sub("\. ", " . ", doc)
                doc = re.sub(", ", " , ", doc)

                doc = clean_text(doc, database="nus")
                doc = doc.replace("\n", " ")
                data[jsonl["name"]] = doc
                labels[jsonl["name"]] = keywords
            except Exception:
                raise ValueError
    return data, labels


def get_short_data(file_path="promptrank/kp20k/kp20k_valid2k_test.json"):
    """Load file.jsonl ."""
    data = {}
    labels = {}
    with codecs.open(
        os.path.join(GEO_KPE_MULTIDOC_DATA_PATH, file_path), "r", "utf-8"
    ) as f:
        json_text = f.readlines()
        for i, line in tqdm(enumerate(json_text), desc="Loading Doc ..."):
            try:
                jsonl = json.loads(line)
                keywords = jsonl["keywords"].lower().split(";")
                abstract = jsonl["abstract"]
                doc = abstract
                doc = re.sub("\. ", " . ", doc)
                doc = re.sub(", ", " , ", doc)

                doc = clean_text(doc, database="kp20k")
                doc = doc.replace("\n", " ")
                data[i] = doc
                labels[i] = keywords
            except Exception:
                raise ValueError
    return data, labels


def get_duc2001_data(file_path="promptrank/DUC2001"):
    pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.S)
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(
        os.path.join(GEO_KPE_MULTIDOC_DATA_PATH, file_path)
    ):
        for fname in filenames:
            if fname == "annotations.txt":
                # left, right = fname.split('.')
                infile = os.path.join(dirname, fname)
                f = open(infile, "rb")
                text = f.read().decode("utf8")
                lines = text.splitlines()
                for line in lines:
                    left, right = line.split("@")
                    d = right.split(";")[:-1]
                    doc_id = left
                    labels[doc_id] = d
                f.close()
            else:
                infile = os.path.join(dirname, fname)
                f = open(infile, "rb")
                text = f.read().decode("utf8")
                text = re.findall(pattern, text)[0]

                text = text.lower()
                text = clean_text(text, database="duc2001")
                data[fname] = text.strip("\n")
                # data[fname] = text
    return data, labels


def get_inspec_data(file_path="promptrank/Inspec"):
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(
        os.path.join(GEO_KPE_MULTIDOC_DATA_PATH, file_path)
    ):
        for fname in filenames:
            left, right = fname.split(".")
            if right == "abstr":
                infile = os.path.join(dirname, fname)
                f = open(infile)
                text = f.read()
                text = text.replace("%", "")
                text = clean_text(text)
                data[left] = text
            if right == "uncontr":
                infile = os.path.join(dirname, fname)
                f = open(infile)
                text = f.read()
                text = text.replace("\n", " ")
                text = clean_text(text, database="inspec")
                text = text.lower()
                label = text.split("; ")
                labels[left] = label
    return data, labels


def get_semeval2017_data(
    data_path="promptrank/SemEval2017/docsutf8",
    labels_path="promptrank/SemEval2017/keys",
):
    data = {}
    labels = {}
    for dirname, dirnames, filenames in os.walk(
        os.path.join(GEO_KPE_MULTIDOC_DATA_PATH, data_path)
    ):
        for fname in filenames:
            left, right = fname.split(".")
            infile = os.path.join(dirname, fname)
            # f = open(infile, 'rb')
            # text = f.read().decode('utf8')
            with codecs.open(infile, "r", "utf-8") as fi:
                text = fi.read()
                text = text.replace("%", "")
            text = clean_text(text, database="semeval2017")
            data[left] = text.lower()
            # f.close()
    for dirname, dirnames, filenames in os.walk(
        os.path.join(GEO_KPE_MULTIDOC_DATA_PATH, labels_path)
    ):
        for fname in filenames:
            left, right = fname.split(".")
            infile = os.path.join(dirname, fname)
            f = open(infile, "rb")
            text = f.read().decode("utf8")
            text = text.strip()
            ls = text.splitlines()
            labels[left] = ls
            f.close()
    return data, labels


DATASETS = {
    "DUC2001": {
        "loader": get_duc2001_data,
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "Inspec": {
        "loader": get_inspec_data,
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "Nguyen2007": {
        "loader": get_long_data,
        "language": "en",
        "tagger": "en_core_web_trf",
    },
    "SemEval2017": {
        "loader": get_semeval2017_data,
        "language": "en",
        "tagger": "en_core_web_trf",
    },
}


def load_promptrankdataset(name) -> KPEDataset:
    data, labels = DATASETS[name]["loader"]()

    return KPEDataset(
        name,
        ids=list(data.keys()),
        documents=list(data.values()),
        labels=[labels[key] for key in data],
    )
