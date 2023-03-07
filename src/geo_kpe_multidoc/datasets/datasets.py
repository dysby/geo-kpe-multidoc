import io
import re
from os import path
from typing import List, Tuple
from zipfile import ZipFile

from torch.utils.data import Dataset

# Datasets from https://github.com/LIAAD/KeywordExtractor-Datasets
DATASETS = {
    "110-PT-BN-KP": {
        "zip_file": "110-PT-BN-KP.zip",
        "language": "pt",
        "tagger": "pt_core_news_lg",
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
        "tagger": "en_core_web_trf",
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


class KPEDataset(Dataset):
    """Suported Evaluation Dataset"""

    def __init__(self, name, zip_file, root_dir, transform=None):
        """
        Parameters
        ----------
            name: str = Name of the Dataset
            zip_file: str = Path to the zip file with docs and annoations.
            root_dir: str = Data path.
            transform: Optional[Callable]: Optional transform to be applied
                on a sample.

        """
        self.name = name
        self.ids, self.documents, self.keys = self._read_zip(
            path.join(root_dir, zip_file)
        )
        # self.root_dir = root_dir
        self.transform = transform

    def _read_zip(self, filename) -> Tuple[List[str], List, List]:
        """
        read documents from docsutf8 dir w/ .txt, and read keys from keys dir w/ .key

        ex:
            "keys/2000_10_11-21_00_00-Jornal2-2-topic-seg.txt-Nr9.key"
        to
            id = "2000_10_11-21_00_00-Jornal2-2-topic-seg.txt-Nr9"
        """
        ids = []
        documents = []
        keys = []

        with ZipFile(filename, mode="r") as zf:
            for file in zf.namelist():
                if file[-4:] == ".key":  # optional filtering by filetype
                    id = file.split("/")[-1][:-4]
                    ids.append(id)
                    with zf.open(file) as f:
                        keys.append(
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
        return ids, documents, keys

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        """
        Returns:
        --------
            name: id of the document
            document: txt content
            keys: gold keyphrases for document
        """
        return self.ids[idx], self.documents[idx], self.keys[idx]