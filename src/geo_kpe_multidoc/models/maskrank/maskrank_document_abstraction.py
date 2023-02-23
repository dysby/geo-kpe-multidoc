import re
import time
from typing import Callable, List, Set, Tuple

import numpy as np
import simplemma
from keybert._mmr import mmr
from nltk import RegexpParser
from sklearn.metrics.pairwise import cosine_similarity

from ...utils.IO import read_from_file


class Document:
    """
    Class to encapsulate document representation and functionality
    """

    def __init__(self, raw_text, id):
        """
        Stores the raw text representation of the doc, a pos_tagger and the grammar to
        extract candidates with.
        
        Attributes:
            self.raw_text -> Raw text representation of the document
            self.doc_sents -> Document in list form divided by sentences
            self.punctuation_regex -> regex that covers most punctuation and notation marks

            self.tagged_text -> The entire document divided by sentences with POS tags in each word
            self.candidate_set -> Set of candidates in list form, according to the supplied grammar
            self.candidate_set_embed -> Set of candidates in list form, according to the supplied grammar, in embedding form
        """

        self.raw_text = raw_text
        self.punctuation_regex = "[!\"#\$%&'\(\)\*\+,\.\/:;<=>\?@\[\]\^_`{\|}~\-\–\—\‘\’\“\”]"
        self.doc_sents = []
        self.id = id

    def pos_tag(self, tagger, memory, id):
        """
        Method that handles POS_tagging of an entire document, whilst storing it seperated by sentences
        """
        self.tagged_text, self.doc_sents, self.doc_sents_words = tagger.pos_tag_text_sents_words(self.raw_text, memory, id)
        self.doc_sents = [sent.text for sent in self.doc_sents if sent.text.strip()]

    def embed_doc(self, model, stemmer : Callable = None):
        """
        Method that embeds the document.
        """

        # doc_info = model.embed_full(self.raw_text) # encode(documents, show_progress_bar=False, output_value = None)
        doc_info = model.embedding_model.encode(self.raw_text, show_progress_bar=False, output_value = None)

        self.doc_token_ids = doc_info["input_ids"].squeeze().tolist()
        self.doc_token_embeddings = doc_info["token_embeddings"]
        self.doc_attention_mask = doc_info["attention_mask"]

        return doc_info["sentence_embedding"].detach().numpy()

    def embed_global(self, model):
        raise NotImplemented

    def global_embed_doc(self, model):
        raise NotImplemented

    def embed_candidates(self, model, stemmer : Callable = None, cand_mode: str = "MaskAll", attention : str = ""):
        """
        Method that embeds the current candidate set, having several modes according to usage. 
            cand_mode
            | MaskFirst only masks the first occurence of a candidate;
            | MaskAll masks all occurences of said candidate

            The default value is MaskAll.
        """
        self.candidate_set_embed = []

        if cand_mode == "MaskFirst" or cand_mode == "MaskAll":
            occurences = 1 if cand_mode == "MaskFirst" else 0

            escaped_docs = [re.sub(re.escape(candidate), "<mask>", self.raw_text, occurences) for candidate in self.candidate_set]
            self.candidate_set_embed = model.embed(escaped_docs)

        elif cand_mode == "MaskHighest":
            for candidate in self.candidate_set:
                candidate = re.escape(candidate)
                candidate_embeds = []

                for match in re.finditer(candidate, self.raw_text):
                    masked_text = f'{self.raw_text[:match.span()[0]]}<mask>{self.raw_text[match.span()[1]:]}'
                    if attention == "global_attention":
                        candidate_embeds.append(self.embed_global(masked_text))
                    else:
                        candidate_embeds.append(model.embed(masked_text))
                self.candidate_set_embed.append(candidate_embeds)

        elif cand_mode == "MaskSubset":
            self.candidate_set = sorted(self.candidate_set, reverse=True, key= lambda x : len(x))
            seen_candidates = {}

            for candidate in self.candidate_set:
                prohibited_pos = []
                len_candidate = len(candidate)
                for prev_candidate in seen_candidates:
                    if len_candidate == len(prev_candidate):
                        break

                    elif candidate in prev_candidate:
                        prohibited_pos.extend(seen_candidates[prev_candidate])

                pos = []
                for match in re.finditer(re.escape(candidate), self.raw_text):
                    pos.append((match.span()[0], match.span()[1]))
                
                seen_candidates[candidate] = pos
                subset_pos = []
                for p in pos:
                    subset_flag = True
                    for prob in prohibited_pos:
                        if p[0] >= prob[0] and p[1] <= prob[1]:
                            subset_flag = False
                            break
                    if subset_flag:
                        subset_pos.append(p)

                masked_doc = self.raw_text
                for i in range(len(subset_pos)):
                    masked_doc = f'{masked_doc[:(subset_pos[i][0] + i*(len_candidate - 5))]}<mask>{masked_doc[subset_pos[i][1] + i*(len_candidate - 5):]}'
                self.candidate_set_embed.append(model.embed(masked_doc))
                
    def extract_candidates(self, min_len : int = 5, grammar : str = "", lemmer : Callable = None):
        """
        Method that uses Regex patterns on POS tags to extract unique candidates from a tagged document and 
        stores the sentences each candidate occurs in
        """
        candidate_set = set()

        parser = RegexpParser(grammar)
        np_trees = list(parser.parse_sents(self.tagged_text))

        for i in range(len(np_trees)):
            temp_cand_set = []
            for subtree in np_trees[i].subtrees(filter = lambda t : t.label() == 'NP'):
                temp_cand_set.append(' '.join(word for word, tag in subtree.leaves()))

            for candidate in temp_cand_set:
                if len(candidate) > min_len:
                    candidate_set.add(candidate)

        self.candidate_set = list(candidate_set)

    def embed_n_candidates(self, model, min_len, stemmer, **kwargs) -> List[Tuple]:
        t = time.time()
        self.doc_embed = self.embed_doc(model, stemmer)
        print(f'Embed Doc = {time.time() -  t:.2f}')

        t = time.time()
        self.embed_candidates(model, stemmer, "MaskAll")
        print(f'Embed Candidates = {time.time() -  t:.2f}')

        return self.candidate_set_embed, self.candidate_set

    def evaluate_n_candidates(self, candidate_set_embed, candidate_set) -> List[Tuple]:
        doc_sim = np.absolute(cosine_similarity(candidate_set_embed, self.doc_embed.reshape(1, -1)))
        candidate_score = sorted([(candidate_set[i], 1.0 - doc_sim[i][0]) for i in range(len(doc_sim))], reverse= True, key= lambda x: x[1])

        return candidate_score, [candidate[0] for candidate in candidate_score]

    def top_n_candidates(self, model, top_n: int = 5, min_len : int = 5, stemmer : Callable = None, **kwargs) -> List[Tuple]:

        t = time.time()
        self.doc_embed = self.embed_doc(model, stemmer)
        print(f'Embed Doc = {time.time() -  t:.2f}')

        t = time.time()
        self.embed_candidates(model, stemmer, 
                              "MaskAll" if ("cand_mode" not in kwargs or kwargs["cand_mode"] == "") else kwargs["cand_mode"], 
                              "global_attention" if "global_attention" in kwargs else "")
        print(f'Embed Candidates = {time.time() -  t:.2f}')

        doc_sim = []
        if "cand_mode" not in kwargs or kwargs["cand_mode"] != "MaskHighest":
            doc_sim = np.absolute(cosine_similarity(self.candidate_set_embed, self.doc_embed.reshape(1, -1)))
        
        elif kwargs["cand_mode"] == "MaskHighest":
            doc_embed = self.doc_embed.reshape(1, -1)
            for mask_cand_occur in self.candidate_set_embed:
                if mask_cand_occur != []:
                    doc_sim.append([np.ndarray.min(np.absolute(cosine_similarity(mask_cand_occur, doc_embed)))])
                else:
                    doc_sim.append([1.0])

        candidate_score = sorted([(self.candidate_set[i], 1.0 - doc_sim[i][0]) for i in range(len(doc_sim))], reverse= True, key= lambda x: x[1])

        if top_n == -1:
            return candidate_score, [candidate[0] for candidate in candidate_score]

        return candidate_score[:top_n], [candidate[0] for candidate in candidate_score]