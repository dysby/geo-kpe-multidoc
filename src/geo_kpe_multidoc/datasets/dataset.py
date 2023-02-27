import itertools
import json
import re
from os import listdir, path
from typing import List, Tuple, Type, TypeVar, Union
from loguru import logger
from bs4 import BeautifulSoup

from torch.utils.data import Dataset

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH
from geo_kpe_multidoc.utils.IO import read_from_file, write_to_file


SUPPORTED_DATASETS = {
    "DUC": "xml",
    "NUS": "xml",
    "Inspec": "xml",
    "SemEval": "txt",
    "PubMed": "xml",
    "ResisBank": "txt",
    "MKDUC01": "mdkpe",
    "PT-KP": "xml",
    "ES-CACIC": "txt",
    "ES-WICC": "txt",
    "FR-WIKI": "txt",
    "DE-TeKET": "txt",
}

_DATA_SUBSET = ["train", "dev", "test"]

T = TypeVar("T", bound="TextDataset")
# typing, MDKPE is for multidocument KPE, and DKPE is for single document KPE
MDKPE_LIST = List[Tuple[List[str], List[str]]]
DKPE_LIST = List[Tuple[str, List[str]]]


class TextDataset(Dataset):
    """
    A data class to store the supported Dataset contents in custom format for processing with the modules.

    content:

    Attributes
    ----------
    content : List[Tuple[List[str], List[str]]]
        a list container with (doc/docs, keyphrases),
        or a list with (topic name, docs, keyphrases)

    Methods
    -------
    build_dataset(names: List[str])
        a factory method.
    """

    def __init__(self, name: str, content: Union[DKPE_LIST, MDKPE_LIST]) -> None:
        self.name = name
        self.content = content

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        """
        Returns
        -----
            (sample, label) or (topic_name, sample, label )

        """
        return self.content[idx]

    @classmethod
    def build_datasets(cls: Type[T], names: List[str] = ["DUC"]) -> List[T]:
        """
        Load datasets from files. Only supported datasets can be loaded.

        Parameters
        ----------
        name : List[str]
            list of names datasets to load.

        Returns
        -------
        TextDataset
        """
        datasets = []
        for name in names:
            if name not in SUPPORTED_DATASETS.keys():
                raise ValueError(
                    f"Requested dataset {name} is not implemented in supported datasets\n {SUPPORTED_DATASETS.keys()}"
                )

            content = extract_from_dataset(name, SUPPORTED_DATASETS[name])
            datasets.append(cls(name, content))
        return datasets


def extract_from_dataset(
    dataset_name: str = "DUC", data_t: str = "xml"
) -> Union[DKPE_LIST, MDKPE_LIST]:
    """Extract from raw datafiles and write pre-processed data to a child directory"""
    dataset_dir = path.join(GEO_KPE_MULTIDOC_DATA_PATH, dataset_name)

    p_data_path = path.join(
        GEO_KPE_MULTIDOC_DATA_PATH,
        "processed_data",
        dataset_name,
        f"{dataset_name}_processed",
    )

    if path.isfile(f"{p_data_path}.txt"):
        logger.info(f"Reading from pickled cache file: {p_data_path}")
        return read_from_file(p_data_path)

    res = []
    if data_t == "xml":
        res = extract_xml(dataset_dir)
    elif data_t == "txt":
        res = extract_txt(dataset_dir)
    elif data_t == "mdkpe":
        res = extract_mdkpe(dataset_dir)

    logger.info(f"Writing in pickle cache file: {p_data_path}")
    write_to_file(p_data_path, res)

    return res


def extract_mdkpe(dataset_dir) -> MDKPE_LIST:
    """Remove topic key and document id and keep only a list of items each corresponding to
    a topic, and each item composed by a list of docs and a list of keyphrases."""
    dataset = {}
    with open(f"{dataset_dir}/MKDUC01.json", "r") as source_f:
        dataset = json.load(source_f)
    logger.info(f"Load json with {len(dataset)} items")

    res = []
    for topic in dataset:
        doc_content_for_topic = [
            doc_content
            for _doc_name, doc_content in dataset[topic]["documents"].items()
        ]
        kps_for_topic = list(itertools.chain(*dataset[topic]["keyphrases"]))
        res.append((topic, doc_content_for_topic, kps_for_topic))

    if len(res) == 0:
        logger.warning(f"Extracted **zero** results")
    return res


def extract_xml(dataset_dir) -> MDKPE_LIST:
    res = []

    dir_cont = listdir(dataset_dir)
    for subset in _DATA_SUBSET:
        if subset in dir_cont:
            subset_dir = f"{dataset_dir}/{subset}"

            with open(f"{dataset_dir}/references/{subset}.json") as ref_file:
                refs = json.load(ref_file)

            # subst_table = { "-LSB-" : "(", "-LRB-" : "(", "-RRB-" : ")", "-RSB-" : ")", "p." : "page", }
            subst_table = {}

            for file in listdir(subset_dir):
                if file[:-4] not in refs:
                    raise RuntimeError(f"Can't find key-phrases for file {file}")

                doc = ""
                soup = BeautifulSoup(open(f"{subset_dir}/{file}").read(), "xml")

                # content = soup.find_all('journal-title')
                # for word in content:
                #    doc += "{}. ".format(re.sub(r'Figs?\.*\s*[0-9]*\.*', r'Fig ', word.get_text()))

                # content = soup.find_all('p')
                # for word in content:
                #    doc += "{} ".format(re.sub(r'Figs?\.*\s*[0-9]*\.*', r'Fig ', word.get_text()))

                # content = soup.find_all(['article-title ', 'title'])
                # for word in content:
                #    doc += "{}. ".format(re.sub(r'Figs?\.*\s*[0-9]*\.*', r'Fig ', word.get_text()))

                content = soup.find_all("word")
                for word in content:
                    text = word.get_text()
                    for key in subst_table:
                        text = re.sub(f"{key}", f"{subst_table[key]}", text)
                    doc += f"{text} "

                res.append((doc, [r[0] for r in refs[file[:-4]]]))

                print(f"doc number {file[:-4]}")
                # print(doc)
                # print(f'{res[-1][1]} \n')
    return res


def extract_txt(dataset_dir) -> DKPE_LIST:
    res = []

    dir_cont = listdir(dataset_dir)
    for subset in _DATA_SUBSET:
        if subset in dir_cont:
            subset_dir = f"{dataset_dir}/{subset}"

            # total_keywords = 0
            # found_keywords = 0
            # stemmer = PorterStemmer()
            # lemmer = simplemma.load_data("pl")

            with open(f"{dataset_dir}/references/test.json") as inp:
                with open(f"{dataset_dir}/references/test-stem.json") as inp_s:
                    references = json.load(inp)
                    references_s = json.load(inp_s)

                    for file in listdir(subset_dir):
                        if file[:-4] in references_s:
                            doc = open(
                                f"{subset_dir}/{file}", "r", encoding="utf-8"
                            ).read()

                            # kp = [line.rstrip() for line in open(f'{dataset_dir}/references/{file[:-4]}.key', 'r', encoding='utf-8').readlines() if line.strip()]
                            kp = [k[0].rstrip() for k in references[file[:-4]]]

                            # for line in open(f'{dataset_dir}/references/{file[:-4]}.key', 'r', encoding='utf-8').readlines():
                            #    if line.strip():
                            #        total_keywords += 1
                            #        stemmed = stemmer.stem(line.lower().strip())
                            #        lemmed = simplemma.lemmatize(line.lower().strip(), lemmer)
                            #        if stemmed in doc.lower() or lemmed in doc.lower():
                            #            found_keywords += 1
                            #        else:
                            #            print(f"didn't find {line}")
                            #            print(doc)

                            res.append((doc, kp))
                            print(f"doc number {file[:-4]}")

            # print("|Statistics for PL-PAK|")
            # print(f'Found Key-Phrases: {found_keywords}')
            # print(f'Total Key-Phrases: {total_keywords}')
            # print(f'Percentage of unavailable key-phrases: {((1 - found_keywords/total_keywords)*100):.3}%')
    print(len(res))
    return res
