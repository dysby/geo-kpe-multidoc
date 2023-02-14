import numpy as np
import torch
import os

from nltk.stem import PorterStemmer
from typing import List, Tuple, Set

from utils.IO import read_from_file, write_to_file

class EmbeddingsMemory:
    """
    Class to calculate and store embeddings in memory
    """
    def __init__(self, corpus):
        self.corpus = corpus
        self.stemmer = PorterStemmer()

    def write_embeds(self, model, save_dir, doc_sents_words, stemming = False):
        doc_sents_words_embed = []

        for i in range(len(doc_sents_words)):
            sent = [self.stemmer.stem(word) for word in doc_sents_words[i] ] if stemming else doc_sents_words[i]
            doc_sents_words_embed.append(model.embed(sent))
        write_to_file(save_dir, doc_sents_words_embed)

    def save_embeddings(self, dataset_obj, model, embeds, save_dir, tagger, stemming = False, start_index = 0):
        for dataset in dataset_obj.dataset_content:
            dir = f'{save_dir}{dataset}/{embeds}/'

            if not os.path.isdir(dir):
                os.mkdir(dir)

            for i in range(start_index, len(dataset_obj.dataset_content[dataset])):
                torch.cuda.empty_cache()
                result_dir = f'{dir}{i}'
                doc_sents_words = []
                for sent in tagger.tagger(dataset_obj.dataset_content[dataset][i][0]).sents:
                    torch.cuda.empty_cache()
                    if sent.text.strip():
                        doc_sents_words.append([token.text for token in sent])

                torch.cuda.empty_cache()
                self.write_embeds(model, result_dir, doc_sents_words)
                print(f'Doc {i} stored')