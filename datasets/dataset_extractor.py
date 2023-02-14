import csv
import os, os.path
import re
import doi

import metapub
import json
from typing import List
from utils.IO import read_from_file, write_to_file
from io import StringIO
from scidownl import scihub_download

from nltk.stem import PorterStemmer
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

def map_orig_links(dir_input : str, dir_output : str) -> None:
    with open(dir_input, 'r', encoding='utf-8') as file:
        id_types = ["DOI", "PMID", "ISSN", "Web", "Other"]
        results = {id:set() for id in id_types}

        regex = {"DOI" : re.compile("^10.\d{4,9}[/=][-._;()/:a-zA-Z0-9]+"), 
                 "PMID" : re.compile("[0-9]{6,8}"), 
                 "ISSN" : re.compile("ISSN:.*"), 
                 "Web" : re.compile("https?://.*"), 
                 "Other" : True}

        for line in [line.rstrip() for line in file.readlines()]:
            for pat in regex:
                if regex[pat] == True or regex[pat].match(line):
                    results[pat].add(line)
                    break
        
        print("Extraction statistics")
        for type in results:
            print(f'{type} = {len(results[type])} entries')

    print(f'Outputing to {os.getcwd()}{dir_output}')
    write_to_file(f'{os.getcwd()}{dir_output}', results)
    print('Success!')

def extract_files_from_link(dir_input : str, dir_output : str) -> None:
    doc_ids = read_from_file(dir_input)
    c_f = 0

    output = "./raw_data/ResisBank/src/documents/pdfs/"
    
    with open("./raw_data/ResisBank/src/documents/downloaded_pdfs.txt", "a") as output_doi:
        for type in doc_ids:
            print(f'{type} = {len(doc_ids[type])} entries')

        paper_type = "doi"
        for entry in sorted(doc_ids["DOI"]):
            name = re.sub("[\./]", "-", entry)
            file_path = f'{output}{name}.pdf'

            if not os.path.isfile(file_path):
                scihub_download(entry, paper_type=paper_type, out=file_path)

                if os.path.isfile(file_path):
                    output_doi.write(f'{entry}\n')

    pmid_fetcher = metapub.PubMedFetcher()
    paper_type = "pmid"
    for entry in sorted(doc_ids["PMID"]):
        src = metapub.FindIt(entry)
        print(src.url)
        c_f += 1 if src != None else 0
    
    

    print(f'Found {c_f} files in total')

def pdf_to_txt():
    pdf_source = "./raw_data/ResisBank/src/documents/pdfs/"
    txt_destiny = "./raw_data/ResisBank/src/documents/txt/"
    reference_destiny = "./raw_data/ResisBank/src/documents/references/"

    stemmed_dic = {}
    stemmer = PorterStemmer()

    word_count = 0
    kp_count = 0
    n_docs = 0

    with open(f'{reference_destiny}test.json', 'rb') as ref_file:
        with open(f'{reference_destiny}test-stem.json', 'w', encoding='utf-8') as ref_file_stem:
            ref_dic = json.load(ref_file)
            
            
            for file in ref_dic:

                output_string = StringIO()
                with open(f'{pdf_source}{file}.pdf', 'rb') as in_file:
                    parser = PDFParser(in_file)
                    doc = PDFDocument(parser)
                    rsrcmgr = PDFResourceManager()
                    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
                    interpreter = PDFPageInterpreter(rsrcmgr, device)

                    for page in PDFPage.create_pages(doc):
                        interpreter.process_page(page)

                    with open(f'{txt_destiny}{file}.txt', 'w', encoding='utf-8') as out_file:
                        out_file.write(output_string.getvalue())
                        print(f'Converted {file}.pdf to {file}.txt')

                        word_count += len(output_string.getvalue().split())

                stemmed_dic[file] = []
                kp_count += len(ref_dic[file])
                for kp in ref_dic[file]:
                    stemmed_dic[file].append([stemmer.stem(kp[0])])

            n_docs = len(stemmed_dic)
            json.dump(stemmed_dic, ref_file_stem)


    print(f'Dataset Statistics:\nN_docs = {n_docs}\nAvg word count = {word_count/n_docs:.3}\nAvg kp = {kp_count/n_docs:.1}')