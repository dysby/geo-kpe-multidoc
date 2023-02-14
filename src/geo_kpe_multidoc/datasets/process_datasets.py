import json
import os
from os import path
import re
from typing import List, Tuple

import simplemma
from bs4 import BeautifulSoup
from models.pre_processing.pos_tagging import *
from nltk.stem import PorterStemmer
from utils.IO import read_from_file, write_to_file

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_RAWDATA_PATH


class DataSet:
    """
    A class to abstract processing datasets.

    """
    def __init__(self, datasets = ["DUC"]):
        """ 
        Arguments:
            datasets: Names of datasets in list of string form.
                   The following datasets are currently supported
                      * DUC
                      * NUS
                      * Inspec
                      * SemEval
                      * PubMed
                      * PT-KP
                      * ES-CACIC
                      * ES-WICC
                      * FR-WIKI
                      * DE-TeKET
            unsupervised: Requested supervision criteria
        """

        self.dataset_content = {}
        self.supported_datasets = {"DUC"      : "xml", 
                                   "NUS"      : "xml", 
                                   "Inspec"   : "xml",
                                   "SemEval"  : "txt",
                                   "PubMed"   : "xml",
                                   "ResisBank" : "txt",
                                   "MKDUC01"  : "mdkpe",
                                   "PT-KP"    : "xml",
                                   "ES-CACIC" : "txt", 
                                   "ES-WICC"  : "txt", 
                                   "FR-WIKI"  : "txt",
                                   "DE-TeKET" : "txt"}

        self.data_subset = ["train", "dev", "test"]

        for dataset in datasets:
            if dataset not in self.supported_datasets:
                raise ValueError(f'Requested dataset {dataset} is not implemented. \n Set = {self.supported_datasets}')

            else:
                self.dataset_content[dataset] =  self.extract_from_dataset(dataset, self.supported_datasets[dataset])

    def extract_from_dataset(self, dataset_name: str = "DUC", data_t : str = "xml") -> List[Tuple[str,List[str]]]:
        dataset_dir = path.join(GEO_KPE_MULTIDOC_RAWDATA_PATH, dataset_name)
        
        p_data_path = path.join(GEO_KPE_MULTIDOC_RAWDATA_PATH, "processed_data", dataset_name, f"{dataset_name}_processed")

        if path.isfile(f'{p_data_path}.txt'):
            return read_from_file(p_data_path)

        res = None
        if data_t == "xml":
            res = self.extract_xml(dataset_dir)
        elif data_t == "txt":
            res = self.extract_txt(dataset_dir)
        elif data_t == "mdkpe":
            res = self.extract_mdkpe(dataset_dir)

        write_to_file(p_data_path, res)

        return res

    def extract_mdkpe(self, dataset_dir):
        with open(f'{dataset_dir}/MKDUC01.json', 'r') as source_f:
            dataset = json.load(source_f)
            res = []
            for topic in dataset:
                docs = []
                kps = []
                for doc in dataset[topic]["documents"]:
                    docs.append(dataset[topic]["documents"][doc])
                for kp in dataset[topic]["keyphrases"]:
                    kps.append(kp[0])
                res.append((docs, kps))
        return res
 
    def extract_xml(self, dataset_dir):
        res = []

        dir_cont = os.listdir(dataset_dir)
        for subset in self.data_subset:
            if subset in dir_cont:
                subset_dir = f'{dataset_dir}/{subset}'
                
                ref_file = open(f'{dataset_dir}/references/{subset}.json')
                refs = json.load(ref_file)

                #subst_table = { "-LSB-" : "(", "-LRB-" : "(", "-RRB-" : ")", "-RSB-" : ")", "p." : "page", }
                subst_table = {}

                for file in os.listdir(subset_dir):
                    if file[:-4] not in refs:
                        raise RuntimeError(f'Can\'t find key-phrases for file {file}')

                    doc = ""
                    soup = BeautifulSoup(open(f'{subset_dir}/{file}').read(), "xml")

                    #content = soup.find_all('journal-title') 
                    #for word in content:
                    #    doc += "{}. ".format(re.sub(r'Figs?\.*\s*[0-9]*\.*', r'Fig ', word.get_text()))
                        
                    #content = soup.find_all('p') 
                    #for word in content:
                    #    doc += "{} ".format(re.sub(r'Figs?\.*\s*[0-9]*\.*', r'Fig ', word.get_text()))

                    #content = soup.find_all(['article-title ', 'title'])
                    #for word in content:
                    #    doc += "{}. ".format(re.sub(r'Figs?\.*\s*[0-9]*\.*', r'Fig ', word.get_text()))

                    content = soup.find_all('word')
                    for word in content:
                        text = word.get_text()
                        for key in subst_table:
                            text = re.sub(f'{key}', f'{subst_table[key]}', text)
                        doc += f'{text} '

                    res.append((doc, [r[0] for r in refs[file[:-4]]]))

                    print(f'doc number {file[:-4]}')
                    #print(doc)
                    #print(f'{res[-1][1]} \n')
        return res

    def extract_txt(self, dataset_dir):
        res = []

        dir_cont = os.listdir(dataset_dir)
        for subset in self.data_subset:
            if subset in dir_cont:
                subset_dir = f'{dataset_dir}/{subset}'
                
                #total_keywords = 0
                #found_keywords = 0
                #stemmer = PorterStemmer()
                #lemmer = simplemma.load_data("pl")

                with open(f'{dataset_dir}/references/test.json') as inp:
                    with open(f'{dataset_dir}/references/test-stem.json') as inp_s:
                        references = json.load(inp)
                        references_s = json.load(inp_s)

                        for file in os.listdir(subset_dir):
                            if file[:-4] in references_s:
                                doc = open(f'{subset_dir}/{file}', 'r', encoding='utf-8').read()
                                
                                #kp = [line.rstrip() for line in open(f'{dataset_dir}/references/{file[:-4]}.key', 'r', encoding='utf-8').readlines() if line.strip()]
                                kp = [k[0].rstrip() for k in references[file[:-4]]]

                                #for line in open(f'{dataset_dir}/references/{file[:-4]}.key', 'r', encoding='utf-8').readlines():
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
                                print(f'doc number {file[:-4]}')

                #print("|Statistics for PL-PAK|")
                #print(f'Found Key-Phrases: {found_keywords}')
                #print(f'Total Key-Phrases: {total_keywords}')
                #print(f'Percentage of unavailable key-phrases: {((1 - found_keywords/total_keywords)*100):.3}%')
        print(len(res))
        return res