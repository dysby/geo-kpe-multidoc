import re
from itertools import islice
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from nltk.stem import StemmerI
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast

from geo_kpe_multidoc.datasets.datasets import KPEDataset, load_data
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.base_KP_model import BaseKPModel
from geo_kpe_multidoc.models.candidate_extract.candidate_extract_model import (
    KPECandidateExtractionModel,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_punctuation,
    remove_whitespaces,
    select_stemmer,
)
from geo_kpe_multidoc.models.promptrank.data import PromptRankExtractor


def get_PRF(num_c, num_e, num_s):
    F1 = 0.0
    P = float(num_c) / float(num_e) if num_e != 0 else 0.0
    R = float(num_c) / float(num_s) if num_s != 0 else 0.0
    if P + R == 0.0:
        F1 = 0
    else:
        F1 = 2 * P * R / (P + R)
    return P, R, F1


def print_PRF(P, R, F1, N):
    logger.info(f"N={N}")
    logger.info(f"P={P}")
    logger.info(f"R={R}")
    logger.info(f"F1={F1}")


class PromptRank(BaseKPModel):
    """
    PromptRank: Unsupervised Keyphrase Extraction using Prompt (https://arxiv.org/abs/2305.04490)

    from https://github.com/HLT-NLP/PromptRank
    """

    def __init__(
        self, model_name="google/flan-t5-small", tagger="en_core_web_trf", **kwargs
    ):
        self.name = "{}_{}".format(
            self.__class__.__name__, re.sub("[-/]", "_", model_name)
        )

        self.candidate_selection_model = PromptRankExtractor(
            model_name, tagger=tagger, **kwargs
        )

        self.max_len = kwargs.get("max_len", 512)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, model_max_length=self.max_len
        )
        self.model.to(self.device)
        self.model.eval()

        self.temp_en = kwargs.get("temp_en", "Book:")
        self.temp_de = kwargs.get("temp_de", "This book mainly talks about ")
        self.enable_filter = kwargs.get("enable_filter", False)
        self.enable_pos = kwargs.get("enable_pos", True)
        # \gamma in original paper
        self.position_factor = kwargs.get("position_factor", 1.2e8)
        # \alpha in original paper
        self.length_factor = kwargs.get("length_factor", 0.6)

        self.stemmer = select_stemmer(kwargs.get("lang", "en"))
        self.counter = 1

    def extract_candidates(self, doc, min_len, lemmer, **kwargs) -> List[str]:
        self.candidate_selection_model(
            doc=doc, min_len=min_len, lemmer_lang=lemmer, **kwargs
        )

    def top_n_candidates(
        self, doc, candidate_list, top_n, min_len, **kwargs
    ) -> List[Tuple]:
        # input
        # doc_list, labels_stemed, labels,  model, dataloader
        doc_list = None
        labels_stemed = None
        labels = None
        dataloader = None

        cos_similarity_list = {}
        candidate_list = []
        cos_score_list = []
        doc_id_list = []
        pos_list = []

        num_c = num_c_5 = num_c_10 = num_c_15 = 0
        num_e = num_e_5 = num_e_10 = num_e_15 = 0
        num_s = 0

        template_len = (
            self.tokenizer(self.temp_de, return_tensors="pt")["input_ids"].shape[1] - 3
        )  # single space
        # print(template_len)
        # etting_dict["temp_en"] +
        dataset = load_data("DUC2001")
        (
            dataset,
            doc_list,
            labels,
            labels_stemed,
        ) = self.candidate_selection_model.data_process(dataset)
        # dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size)
        dataloader = DataLoader(dataset)

        for id, (en_input_ids, en_input_mask, de_input_ids, dic) in enumerate(
            tqdm(dataloader, desc="Evaluating:")
        ):
            en_input_ids = en_input_ids.to(self.device)
            en_input_mask = en_input_mask.to(self.device)
            de_input_ids = de_input_ids.to(self.device)

            score = np.zeros(en_input_ids.shape[0])

            # print(dic["candidate"])
            # print(dic["de_input_len"])
            # exit(0)

            with torch.no_grad():
                output = self.model(
                    input_ids=en_input_ids,
                    attention_mask=en_input_mask,
                    decoder_input_ids=de_input_ids,
                )[0]
                # print(en_output.shape)
                # x = empty_ids.repeat(en_input_ids.shape[0], 1, 1).to(device)
                # empty_output = model(input_ids=x, decoder_input_ids=de_input_ids)[2]

                for i in range(template_len, de_input_ids.shape[1] - 3):
                    logits = output[:, i, :]
                    logits = logits.softmax(dim=1)
                    logits = logits.cpu().numpy()

                    for j in range(de_input_ids.shape[0]):
                        if i < dic["de_input_len"][j]:
                            score[j] = score[j] + np.log(
                                logits[j, int(de_input_ids[j][i + 1])]
                            )
                        elif i == dic["de_input_len"][j]:
                            score[j] = score[j] / np.power(
                                dic["de_input_len"][j] - template_len,
                                self.length_factor,
                            )
                            # score = score + 0.005 (score - empty_score)

                doc_id_list.extend(dic["idx"])
                candidate_list.extend(dic["candidate"])
                cos_score_list.extend(score)
                pos_list.extend(dic["pos"])

        cos_similarity_list["doc_id"] = doc_id_list
        cos_similarity_list["candidate"] = candidate_list
        cos_similarity_list["score"] = cos_score_list
        cos_similarity_list["pos"] = pos_list
        cosine_similarity_rank = pd.DataFrame(cos_similarity_list)

        for i in range(len(doc_list)):
            doc_len = len(doc_list[i].split())

            # doc_results = cosine_similarity_rank.loc[
            #     cosine_similarity_rank["doc_id"] == i
            # ]
            # if self.enable_pos:
            #     # doc_results.loc[:,"pos"] = torch.Tensor(doc_results["pos"].values.astype(float)) / doc_len + position_factor / (doc_len ** 3)
            #     doc_results["pos"] = doc_results[
            #         "pos"
            #     ] / doc_len + self.position_factor / (doc_len**3)
            #     doc_results["score"] = doc_results["pos"] * doc_results["score"]
            # # * doc_results["score"].values.astype(float)
            # ranked_keyphrases = doc_results.sort_values(by="score", ascending=False)

            if self.enable_pos:
                # doc_results.loc[:,"pos"] = torch.Tensor(doc_results["pos"].values.astype(float)) / doc_len + position_factor / (doc_len ** 3)
                cosine_similarity_rank.loc[
                    cosine_similarity_rank["doc_id"] == i, "pos"
                ] = cosine_similarity_rank.loc[
                    cosine_similarity_rank["doc_id"] == i, "pos"
                ] / doc_len + self.position_factor / (
                    doc_len**3
                )
                cosine_similarity_rank.loc[
                    cosine_similarity_rank["doc_id"] == i, "score"
                ] = (
                    cosine_similarity_rank.loc[
                        cosine_similarity_rank["doc_id"] == i, "pos"
                    ]
                    * cosine_similarity_rank.loc[
                        cosine_similarity_rank["doc_id"] == i, "score"
                    ]
                )

            ranked_keyphrases = cosine_similarity_rank.loc[
                cosine_similarity_rank["doc_id"] == i
            ].sort_values(by="score", ascending=False)
            top_k = ranked_keyphrases.reset_index(drop=True)
            top_k_can = top_k.loc[:, ["candidate"]].values.tolist()
            # print(top_k)
            # exit()
            candidates_set = set()
            candidates_dedup = []
            for temp in top_k_can:
                temp = temp[0].lower()
                if temp in candidates_set:
                    continue
                else:
                    candidates_set.add(temp)
                    candidates_dedup.append(temp)

            # logger.debug("Sorted_Candidate: {} \n".format(top_k_can))
            # logger.debug("Candidates_Dedup: {} \n".format(candidates_dedup))

            j = 0
            Matched = candidates_dedup[:15]
            for id, temp in enumerate(candidates_dedup[0:15]):
                tokens = temp.split()
                tt = " ".join(self.stemmer.stem(t) for t in tokens)
                if tt in labels_stemed[i] or temp in labels[i]:
                    Matched[id] = [temp]
                    if j < 5:
                        num_c_5 += 1
                        num_c_10 += 1
                        num_c_15 += 1
                        num_c += 1
                    elif j < 10 and j >= 5:
                        num_c_10 += 1
                        num_c_15 += 1
                        num_c += 1
                    elif j < 15 and j >= 10:
                        num_c_15 += 1
                        num_c += 1
                    else:
                        num_c += 1
                j += 1

            logger.debug("TOP-K {}: {}".format(i, Matched))
            logger.debug("Reference {}: {}".format(i, labels[i]))

            if len(top_k[0:5]) == 5:
                num_e_5 += 5
            else:
                num_e_5 += len(top_k[0:5])

            if len(top_k[0:10]) == 10:
                num_e_10 += 10
            else:
                num_e_10 += len(top_k[0:10])

            if len(top_k[0:15]) == 15:
                num_e_15 += 15
            else:
                num_e_15 += len(top_k[0:15])

            num_e += len(top_k)
            num_s += len(labels[i])

        p, r, f = get_PRF(num_c_5, num_e_5, num_s)
        print_PRF(p, r, f, 5)
        p, r, f = get_PRF(num_c_10, num_e_10, num_s)
        print_PRF(p, r, f, 10)
        p, r, f = get_PRF(num_c_15, num_e_15, num_s)
        print_PRF(p, r, f, 15)
        p, r, f = get_PRF(num_c, num_e, num_s)
        print_PRF(p, r, f, "All")

    def extract_kp_from_doc(
        self,
        doc: Document,
        top_n,
        min_len,
        stemmer: Optional[StemmerI] = None,
        lemmer: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[List[Tuple], List[str]]:
        """
        Concrete method that extracts key-phrases from a given document, with optional arguments
        relevant to its specific functionality
        """

        self.extract_candidates(doc, min_len, lemmer, **kwargs)

        top_n, candidate_set = self.top_n_candidates(
            doc, top_n, min_len, stemmer, **kwargs
        )

        logger.debug(f"Document #{self.counter} processed")
        self.counter += 1

        return (top_n, candidate_set)
