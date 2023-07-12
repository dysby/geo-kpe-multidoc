# coding=utf-8
import codecs
import json
import os
import re
from itertools import islice

import nltk
import spacy
from loguru import logger
from nltk.corpus import stopwords
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5TokenizerFast

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH
from geo_kpe_multidoc.datasets.datasets import KPEDataset
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import select_stemmer


class PromptRankExtractor:
    def __init__(self, model_name, **kwargs) -> None:
        self.max_len = kwargs.get("max_len", 512)
        self.temp_en = kwargs.get("temp_en", "Book:")
        self.temp_de = kwargs.get("temp_de", "This book mainly talks about ")
        self.enable_filter = kwargs.get("enable_filter", False)
        self.enable_pos = kwargs.get("enable_pos", True)
        self.position_factor = kwargs.get("position_factor", 1.2e8)
        self.length_factor = kwargs.get("length_factor", 0.6)

        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, model_max_length=self.max_len
        )

        exclude = ["ner", "lemmatizer"]
        tagger = kwargs.get("tagger", "en_core_web_trf")

        self.en_model = spacy.load(tagger, exclude=exclude)
        self.stemmer = select_stemmer(kwargs.get("lang", "en"))

        self.GRAMMAR = """  NP:
                {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

    def extract_candidates(self, tokens_tagged, no_subset=False):
        """
        Based on part of speech return a list of candidate phrases
        :param text_obj: Input text Representation see @InputTextObj
        :param no_subset: if true won't put a candidate which is the subset of an other candidate
        :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
        """

        cans_count = dict()

        np_parser = nltk.RegexpParser(self.GRAMMAR)  # Noun phrase parser
        keyphrase_candidate = []
        np_pos_tag_tokens = np_parser.parse(tokens_tagged)
        count = 0
        for token in np_pos_tag_tokens:
            if isinstance(token, nltk.tree.Tree) and token._label == "NP":
                np = " ".join(word for word, tag in token.leaves())
                length = len(token.leaves())
                start_end = (count, count + length)
                count += length

                if len(np.split()) == 1:
                    if np not in cans_count.keys():
                        cans_count[np] = 0
                    cans_count[np] += 1

                keyphrase_candidate.append((np, start_end))

            else:
                count += 1

        if self.enable_filter:
            i = 0
            while i < len(keyphrase_candidate):
                can, pos = keyphrase_candidate[i]
                # pos[0] > 50 and
                if can in cans_count.keys() and cans_count[can] == 1:
                    keyphrase_candidate.pop(i)
                    continue
                i += 1

        return keyphrase_candidate

    def generate_doc_pairs(self, doc, candidates, idx):
        count = 0
        doc_pairs = []

        en_input = self.tokenizer(
            doc,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        en_input_ids = en_input["input_ids"]
        en_input_mask = en_input["attention_mask"]

        for id, can_and_pos in enumerate(candidates):
            candidate = can_and_pos[0]
            # Remove stopwords in a candidate
            if remove(candidate):
                count += 1
                continue

            de_input = self.temp_de + candidate + " ."
            de_input_ids = self.tokenizer(
                de_input,
                max_length=30,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )["input_ids"]
            de_input_ids[0, 0] = 0
            de_input_len = (de_input_ids[0] == self.tokenizer.eos_token_id).nonzero()[
                0
            ].item() - 2

            #         for i in de_input_ids[0]:
            #             print(tokenizer.decode(i))
            #         print(de_input_len)

            #         x = tokenizer(temp_de, return_tensors="pt")["input_ids"]
            #         for i in x[0]:
            #             print(tokenizer.decode(i))
            #         exit(0)
            dic = {
                "de_input_len": de_input_len,
                "candidate": candidate,
                "idx": idx,
                "pos": can_and_pos[1][0],
            }

            doc_pairs.append([en_input_ids, en_input_mask, de_input_ids, dic])
            # print(tokenizer.decode(en_input_ids[0]))
            # print(tokenizer.decode(de_input_ids[0]))
            # print(candidate)
            # print(de_input_len)
            # print()
            # exit(0)
        return doc_pairs, count

    # def data_process(self, dataset: KPEDataset):
    def data_process(self, dataset_name: str):
        """
        Core API in data.py which returns the dataset
        """

        # init(setting_dict)

        if dataset_name == "SemEval2017":
            data, referneces = get_semeval2017_data()
        elif dataset_name == "DUC2001":
            data, referneces = get_duc2001_data()
        elif dataset_name == "NUS":
            data, referneces = get_long_data()
        elif dataset_name == "Krapivin2009":
            data, referneces = get_long_data()
        elif dataset_name == "kp20k":
            data, referneces = get_short_data()
        elif dataset_name == "SemEval2010":
            data, referneces = get_short_data()
        elif dataset_name == "Inspec":
            data, referneces = get_inspec_data()

        docs_pairs = []
        doc_list = []
        labels = []
        labels_stemed = []
        t_n = 0
        candidate_num = 0

        # Use with geo_kpe_multidoc.datasets.KPEDataset
        # for idx, (doc_id, text, gold_kp) in enumerate(dataset):
        #     text = text.lower()
        #     text = clean_text(text, database="duc2001")
        #     text = text.strip("\n")
        # Use with original data
        for idx, (key, doc) in enumerate(data.items()):
            # Get stemmed labels and document segments
            # labels.append([ref.replace(" \n", "") for ref in gold_kp])
            labels.append([ref.replace(" \n", "") for ref in referneces[key]])
            labels_s = []
            # for l in gold_kp:
            for l in referneces[key]:
                tokens = l.split()
                # if len(tokens) > 0:
                #     labels_s.append(" ".join(self.stemmer.stem(t) for t in tokens))
                labels_s.append(" ".join(self.stemmer.stem(t) for t in tokens))
            # Get stemmed labels and document segments

            doc = " ".join(doc.split()[: self.max_len])
            labels_stemed.append(labels_s)
            doc_list.append(doc)

            # Statistic on empty docs
            empty_doc = 0
            # try:
            #     text_obj = InputTextObj(en_model, doc)
            # except Exception as e:
            #     empty_doc += 1
            #     logger.critical(e)
            #     logger.critical(f"Empty doc: {doc_id}")
            text_obj = InputTextObj(self.en_model, doc)
            # Generate candidates (lower)
            cans = self.extract_candidates(text_obj.tokens_tagged)
            candidates = []
            for can, pos in cans:
                if self.enable_filter and (len(can.split()) > 4):
                    continue
                candidates.append([can.lower(), pos])
            candidate_num += len(candidates)

            # Generate docs_paris for constructing dataset
            # doc = doc.lower()
            doc = self.temp_en + '"' + doc + '"'
            doc_pairs, count = self.generate_doc_pairs(doc, candidates, idx)
            docs_pairs.extend(doc_pairs)
            t_n += count

        logger.debug(f"Extracted candidates: {candidate_num}")
        logger.debug(f"Unmatched: {t_n}")
        dataset = PromptRankDataset(docs_pairs)
        logger.debug(f"Doc-candidate pairs: {dataset.total_examples}")

        return dataset, doc_list, labels, labels_stemed


class InputTextObj:
    """Represent the input text in which we want to extract keyphrases"""

    def __init__(self, en_model, text="", language="english"):
        """
        :param is_sectioned: If we want to section the text.
        :param en_model: the pipeline of tokenization and POS-tagger
        :param considered_tags: The POSs we want to keep
        """
        self.considered_tags = {"NN", "NNS", "NNP", "NNPS", "JJ"}

        self.tokens = []
        self.tokens_tagged = []

        stopword_dict = set(stopwords.words(language))
        doc = en_model(text)

        self.tokens_tagged = [
            (token.text, token.tag_)
            for sent in doc.sents
            if sent.text.strip()
            for token in sent
        ]

        self.tokens = [token for token, _ in self.tokens_tagged]

        assert len(self.tokens) == len(self.tokens_tagged)
        for i, token in enumerate(self.tokens):
            if token.lower() in stopword_dict:
                self.tokens_tagged[i] = (token, "IN")
        # self.keyphrase_candidate = extract_candidates(self.tokens_tagged)


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
            end = position.end()
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


def get_long_data(file_path="nus/nus_test.json"):
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
            except:
                raise ValueError
    return data, labels


def get_short_data(file_path="kp20k/kp20k_valid2k_test.json"):
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
            except:
                raise ValueError
    return data, labels


def get_duc2001_data(file_path="DUC2001"):
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
                    l = left
                    labels[l] = d
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


def get_inspec_data(file_path="Inspec"):
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
    data_path="SemEval2017/docsutf8", labels_path="SemEval2017/keys"
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


def remove(text: str):
    text_len = len(text.split())
    remove_chars = "[’!\"#$%&'()*+,./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+"
    text = re.sub(remove_chars, "", text)
    re_text_len = len(text.split())
    if text_len != re_text_len:
        return True
    else:
        return False
