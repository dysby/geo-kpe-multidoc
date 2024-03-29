import re
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import transformers
from loguru import logger
from nltk.stem import StemmerI
from tabulate import tabulate
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from geo_kpe_multidoc.datasets.promptrank_datasets import PromptRankDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.base_KP_model import BaseKPModel


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

    def __init__(self, model_name, candidate_selection_model, **kwargs):
        self.name = "{}_{}".format(
            self.__class__.__name__, re.sub("[-/]", "_", model_name)
        )

        self.candidate_selection_model = candidate_selection_model
        logger.info(
            f"PromptRank model w/ {self.candidate_selection_model.__class__.__name__}"
        )

        # hack kwargs.max_seq_len is None...
        self.max_len = kwargs.get("max_seq_len") or 512
        # TODO: HACK longmodel
        if "long" in model_name.lower():
            self.max_len = 4096
        self.temp_en = kwargs.get("encoder_prompt") or "Book: "
        self.temp_de = kwargs.get("decoder_prompt")
        if self.temp_de is None:
            self.temp_de = "This book mainly talks about "
        self.enable_filter = kwargs.get("enable_filter", False)
        self.enable_pos = not kwargs.get("no_position_feature", False)
        # \gamma in original paper
        # DUC 0.89
        self.position_factor = kwargs.get("position_factor", 1.2e8)
        # \alpha in original paper
        self.length_factor = kwargs.get("length_factor", 0.6)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        base_model_config = AutoConfig.from_pretrained(model_name)
        conditional_generation_class = next(
            (
                c
                for c in base_model_config.architectures
                if "ForConditionalGeneration" in c
            ),
            None,
        )
        if conditional_generation_class:
            self.model = getattr(
                transformers, conditional_generation_class
            ).from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, model_max_length=self.max_len
            )
        else:
            raise ValueError(
                f"{model_name} is not compatible with a Conditional Generation Architecture."
            )

        self.model.eval()
        self.model.to(self.device)

        self.template_len = (
            self.tokenizer(self.temp_de, return_tensors="pt")["input_ids"].shape[1] - 3
        )
        if self.template_len < 0:
            self.template_len = 0

        # Dataset       \alpha  \gamma
        # Inspec        1         66.08
        # SemEval2010   0.5       17.50
        # SemEval2017   0.2−1     24.42
        # DUC2001       0.4       0.89
        # NUS           0.2       0.89
        # Krapivin      0.5       0.89

        # Dataloader
        self.num_workers = 1  # Not used, because of Tokenizer paralelism warning
        self.batch_size = kwargs.get("batch_size") or 16

        self.counter = 1

    def extract_candidates(
        self, doc, kp_min_len, lemmer, **kwargs
    ) -> Tuple[List[str], List]:
        return self.candidate_selection_model(
            doc=doc, kp_min_len=kp_min_len, lemmer_lang=lemmer, **kwargs
        )

    def top_n_candidates(
        self, doc, candidate_list, positions, top_n, **kwargs
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        # input
        # doc_list, labels_stemed, labels,  model, dataloader
        cos_similarity_list = {}
        candidate_list = []
        cos_score_list = []
        doc_id_list = []
        pos_list = []

        doc_candidate_input_features = self._input_features(doc)
        dataset = PromptRankDataset(doc_candidate_input_features)

        # dataloader = DataLoader(
        #     dataset, num_workers=self.num_workers, batch_size=self.batch_size
        # )
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        for id, (en_input_ids, en_input_mask, de_input_ids, dic) in enumerate(
            dataloader
            # tqdm(dataloader, desc=f"Evaluating {doc.id}")
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

                for i in range(self.template_len, de_input_ids.shape[1] - 3):
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
                                dic["de_input_len"][j] - self.template_len,
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
            doc_len = len(doc.raw_text.split()[: self.max_len])
            # doc_results.loc[:,"pos"] = torch.Tensor(doc_results["pos"].values.astype(float)) / doc_len + position_factor / (doc_len ** 3)
            cosine_similarity_rank.loc[
                cosine_similarity_rank["doc_id"] == doc.id, "pos"
            ] = cosine_similarity_rank.loc[
                cosine_similarity_rank["doc_id"] == doc.id, "pos"
            ] / doc_len + self.position_factor / (
                doc_len**3
            )
            cosine_similarity_rank.loc[
                cosine_similarity_rank["doc_id"] == doc.id, "score"
            ] = (
                cosine_similarity_rank.loc[
                    cosine_similarity_rank["doc_id"] == doc.id, "pos"
                ]
                * cosine_similarity_rank.loc[
                    cosine_similarity_rank["doc_id"] == doc.id, "score"
                ]
            )

        ranked_keyphrases = cosine_similarity_rank.loc[
            cosine_similarity_rank["doc_id"] == doc.id
        ].sort_values(by="score", ascending=False)
        top_k = ranked_keyphrases.reset_index(drop=True)

        # deduplicate candidates
        top_k["score"] = top_k["score"].astype(float)
        top_k = (
            top_k.groupby("candidate")["score"]
            .agg("max")
            .reset_index()
            .sort_values(by="score", ascending=False)
        )

        top_k_can = top_k.loc[:, "candidate"].values.tolist()
        top_k_can_score = list(
            map(tuple, top_k.loc[:, ["candidate", "score"]].values.tolist())
        )

        return top_k_can_score, top_k_can

    def extract_kp_from_doc(
        self,
        doc: Document,
        top_n,
        kp_min_len,
        stemmer: Optional[StemmerI] = None,
        lemmer: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        # Because base KPE extractor does not return positions (it returns mentions)
        self.extract_candidates(doc, kp_min_len, lemmer, **kwargs)
        candidates = doc.candidate_set
        positions = doc.candidate_positions

        top_n, candidate_set = self.top_n_candidates(
            doc, candidates, positions, top_n=top_n, **kwargs
        )

        logger.debug(f"Document #{self.counter} processed")
        self.counter += 1

        return (top_n, candidate_set)

    # def generate_doc_pairs(self, doc):
    def _input_features(self, doc: Document):
        """
        Returns
        -------
        List[en_input_ids, en_input_mask, de_input_ids, dic]:
            A list with transformer for conditional input features and candidate positional features for each keyphrase candidate in the document.
        """

        # TODO: add TLDR at the end of the encoder prompt
        # words = [word for word in doc.raw_text.split(" ")]
        # if len(words) == (self.max_len - 10):
        #     in_text = " ".join(words[: self.max_len - 10]) + "TL;DR:"
        # else:
        #     in_text = doc.raw_text + " TL;DR:"

        en_input = self.tokenizer(
            self.temp_en + doc.raw_text,
            # self.temp_en + in_text,
            max_length=self.max_len,
            # padding="max_length",
            # In original PromptRank was used max length padding
            # It is not usefull for LongT5
            truncation=True,
            return_tensors="pt",
        )
        en_input_ids = en_input["input_ids"]
        en_input_mask = en_input["attention_mask"]

        doc_candidate_input_features = []
        for id, (candidate, position) in enumerate(
            zip(doc.candidate_set, doc.candidate_positions)
        ):
            # (moved to extraction) Remove stopwords in a candidate
            # if remove(candidate):
            #    count += 1
            #    continue

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
                "idx": doc.id,
                "pos": position[0],
            }

            doc_candidate_input_features.append(
                [en_input_ids, en_input_mask, de_input_ids, dic]
            )
        return doc_candidate_input_features

    def _evaluate(self, dataset_top_k, dataset_labels_stemmed, stemmer: StemmerI):
        num_c = num_c_5 = num_c_10 = num_c_15 = 0
        num_e = num_e_5 = num_e_10 = num_e_15 = 0
        num_s = 0

        for top_k, labels_stemmed in zip(dataset_top_k, dataset_labels_stemmed):
            candidates_dedup = list(dict.fromkeys(map(str.lower, top_k)))
            # logger.debug("Sorted_Candidate: {} \n".format(top_k))
            # logger.debug("Candidates_Dedup: {} \n".format(candidates_dedup))

            # Get stemmed labels and document segments
            # labels_stemed = []
            # for label in labels:
            #     tokens = label.split()
            #     if len(tokens) > 0:
            #         labels_stemed.append(" ".join(self.stemmer.stem(t) for t in tokens))

            j = 0
            matched = candidates_dedup[:15]
            for id, temp in enumerate(candidates_dedup[:15]):
                tokens = temp.split()
                tt = " ".join(stemmer.stem(t) for t in tokens)
                # if tt in labels_stemed[i] or temp in labels[i]:
                if tt in labels_stemmed:  # or temp in labels:
                    matched[id] = [temp]
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

            # logger.debug("TOP-K {}: {}".format(i, Matched))
            # logger.debug("Reference {}: {}".format(i, labels[i]))

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
            # num_s += len(labels[i])
            num_s += len(labels_stemmed)

        results = {
            "Precision": 0.0,
            "Recall": 0.0,
            "F1": 0.0,
            "MAP": 0.0,
            "nDCG": 0.0,
            # "P_5": 0.0,
            # "R_5": 0.0,
            "F1_5": 0.0,
            # "P_10": 0.0,
            # "R_10": 0.0,
            "F1_10": 0.0,
            # "P_15": 0.0,
            # "R_15": 0.0,
            "F1_15": 0.0,
        }

        p, r, f = get_PRF(num_c_5, num_e_5, num_s)
        # results["P_5"] = p
        # results["R_5"] = r
        results["F1_5"] = f
        # print_PRF(p, r, f, 5)
        p, r, f = get_PRF(num_c_10, num_e_10, num_s)
        # results["P_10"] = p
        # results["R_10"] = r
        results["F1_10"] = f
        # print_PRF(p, r, f, 10)
        p, r, f = get_PRF(num_c_15, num_e_15, num_s)
        # results["P_15"] = p
        # results["R_15"] = r
        results["F1_15"] = f
        # print_PRF(p, r, f, 15)
        p, r, f = get_PRF(num_c, num_e, num_s)
        results["Precision"] = p
        results["Recall"] = r
        results["F1"] = f
        # print_PRF(p, r, f, "All")
        print("PromptRank Evaluation")
        print(
            tabulate(
                pd.DataFrame.from_records([results]), headers="keys", floatfmt=".2%"
            )
        )
