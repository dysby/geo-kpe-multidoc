import csv
import json
import os
import re
from io import StringIO
from os import path
from pathlib import Path
from typing import List

import doi
import metapub
from loguru import logger
from nltk.stem import PorterStemmer
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from scidownl import scihub_download

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH, GEO_KPE_MULTIDOC_OUTPUT_PATH

from ..utils.IO import read_from_file, write_to_file


def map_orig_links(dir_input: str, dir_output: str) -> None:
    id_types = ["DOI", "PMID", "ISSN", "Web", "Other"]
    results = {id: set() for id in id_types}

    regex = {
        "DOI": re.compile(r"^10.\d{4,9}[/=][-._;()/:a-zA-Z0-9]+"),
        "PMID": re.compile(r"[0-9]{6,8}"),
        "ISSN": re.compile(r"ISSN:.*"),
        "Web": re.compile(r"https?://.*"),
        "Other": True,
    }

    # TODO: read text or list path?
    for line in Path(dir_input).read_text(encoding="utf-8"):
        for pat in regex:
            if regex[pat] == True or regex[pat].match(line):
                results[pat].add(line)
                break

        logger.debug("Extraction statistics:")
        for type in results:
            logger.debug(f"{type} = {len(results[type])} entries")

    logger.info(f"Outputing to {dir_output}")
    write_to_file(dir_output, results)


def extract_files_from_link(dir_input: str, dir_output: str) -> None:
    doc_ids = read_from_file(dir_input)
    c_f = 0

    output = path.join(GEO_KPE_MULTIDOC_DATA_PATH, "ResisBank/src/documents/pdfs/")

    with open(
        path.join(
            GEO_KPE_MULTIDOC_DATA_PATH, "ResisBank/src/documents/downloaded_pdfs.txt"
        ),
        "a",
    ) as output_doi:
        for type in doc_ids:
            print(f"{type} = {len(doc_ids[type])} entries")

        paper_type = "doi"
        for entry in sorted(doc_ids["DOI"]):
            name = re.sub("[\./]", "-", entry)
            file_path = f"{output}{name}.pdf"

            if not os.path.isfile(file_path):
                scihub_download(entry, paper_type=paper_type, out=file_path)

                if os.path.isfile(file_path):
                    output_doi.write(f"{entry}\n")

    pmid_fetcher = metapub.PubMedFetcher()
    paper_type = "pmid"
    for entry in sorted(doc_ids["PMID"]):
        src = metapub.FindIt(entry)
        print(src.url)
        c_f += 1 if src != None else 0

    print(f"Found {c_f} files in total")


def pdf_to_txt():
    pdf_source = path.join(GEO_KPE_MULTIDOC_DATA_PATH, "ResisBank/src/documents/pdfs/")
    txt_destiny = path.join(GEO_KPE_MULTIDOC_DATA_PATH, "ResisBank/src/documents/txt/")
    reference_destiny = path.join(
        GEO_KPE_MULTIDOC_DATA_PATH, "ResisBank/src/documents/references/"
    )

    stemmed_dic = {}
    stemmer = PorterStemmer()

    word_count = 0
    kp_count = 0
    n_docs = 0

    with open(f"{reference_destiny}test.json", "rb") as ref_file:
        with open(
            f"{reference_destiny}test-stem.json", "w", encoding="utf-8"
        ) as ref_file_stem:
            ref_dic = json.load(ref_file)

            for file in ref_dic:
                output_string = StringIO()
                with open(f"{pdf_source}{file}.pdf", "rb") as in_file:
                    parser = PDFParser(in_file)
                    doc = PDFDocument(parser)
                    rsrcmgr = PDFResourceManager()
                    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
                    interpreter = PDFPageInterpreter(rsrcmgr, device)

                    for page in PDFPage.create_pages(doc):
                        interpreter.process_page(page)

                    with open(
                        f"{txt_destiny}{file}.txt", "w", encoding="utf-8"
                    ) as out_file:
                        out_file.write(output_string.getvalue())
                        print(f"Converted {file}.pdf to {file}.txt")

                        word_count += len(output_string.getvalue().split())

                stemmed_dic[file] = []
                kp_count += len(ref_dic[file])
                for kp in ref_dic[file]:
                    stemmed_dic[file].append([stemmer.stem(kp[0])])

            n_docs = len(stemmed_dic)
            json.dump(stemmed_dic, ref_file_stem)

    logger.debug(
        f"""Dataset Statistics:
    N_docs = {n_docs}
    Avg word count = {word_count/n_docs:.3}
    Avg kp = {kp_count/n_docs:.1}"""
    )
