import re
from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import sled
import torch
import transformers
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

from geo_kpe_multidoc.datasets.promptrank_datasets import PromptRankDataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.base_KP_model import BaseKPModel
from geo_kpe_multidoc.models.candidate_extract.candidate_extract_model import (
    CandidateExtractionModel,
)


class SLEDPromptRank(BaseKPModel):
    """
    Use SLED T5 for PromptRank
    """

    def __init__(
        self, model_name, candidate_selection_model: CandidateExtractionModel, **kwargs
    ):
        self.name = "{}_{}".format(
            self.__class__.__name__, re.sub("[-/]", "_", model_name)
        )

        self.candidate_selection_model = candidate_selection_model

        logger.info(
            f"SLEDPromptRank model w/ {self.candidate_selection_model.__class__.__name__}"
        )

        # hack kwargs.max_seq_len is None...
        self.encoder_prompt = kwargs.get("encoder_prompt") or "Book: "
        self.decoder_prompt = (
            kwargs.get("decoder_prompt") or "This book mainly talks about "
        )
        self.prefix_length = 0

        self.enable_pos = not kwargs.get("no_position_feature", False)
        # \gamma in original paper
        # DUC 0.89
        self.position_factor = kwargs.get("position_factor", 1.2e8)
        # \alpha in original paper
        self.length_factor = kwargs.get("length_factor", 0.6)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        self.template_len = (
            self.tokenizer(self.decoder_prompt, return_tensors="pt")["input_ids"].shape[
                1
            ]
            - 3
        )

        # Dataset       \alpha  \gamma
        # Inspec        1         66.08
        # SemEval2010   0.5       17.50
        # SemEval2017   0.2−1     24.42
        # DUC2001       0.4       0.89
        # NUS           0.2       0.89
        # Krapivin      0.5       0.89

        # Dataloader
        self.num_workers = 1  # Not used, because of Tokenizer paralelism warning
        self.batch_size = 16

        self.counter = 1

    def extract_candidates(
        self, doc, min_len, lemmer, **kwargs
    ) -> Tuple[List[str], List]:
        return self.candidate_selection_model(
            doc=doc, min_len=min_len, lemmer_lang=lemmer, **kwargs
        )

    def top_n_candidates(
        self, doc: Document, candidate_list, positions, top_n, **kwargs
    ) -> List[Tuple]:
        # input
        # doc_list, labels_stemed, labels,  model, dataloader
        cos_similarity_list = {}
        # TODO: check candidate_list ignore from argument in function call
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
            en_input_ids = en_input_ids.clone().to(self.device)
            en_input_mask = en_input_mask.clone().to(self.device)
            de_input_ids = de_input_ids.clone().to(self.device)

            score = np.zeros(en_input_ids.shape[0])

            # print(dic["candidate"])
            # print(dic["de_input_len"])
            # exit(0)

            with torch.no_grad():
                output = self.model(
                    input_ids=en_input_ids,
                    attention_mask=en_input_mask,
                    decoder_input_ids=de_input_ids,
                    prefix_length=self.prefix_length,
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
            doc_len = len(doc.raw_text.split())
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
        min_len,
        lemmer: Optional[Callable] = None,
        **kwargs,
    ) -> Tuple[List[Tuple[str, float]], List[str]]:
        # Because base KPE extractor does not return positions (it returns mentions)
        self.extract_candidates(doc, min_len, lemmer, **kwargs)
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
            A list with transformer for conditional input features and candidate
            positional features for each keyphrase candidate in the document.
        """

        prefix_input_ids = self.tokenizer(
            self.encoder_prompt, return_tensors="pt"
        ).input_ids  # Batch size 1

        en_input = self.tokenizer(
            doc.raw_text,
            return_tensors="pt",
        )
        en_input_ids = en_input["input_ids"]

        # we concatenate them together, but tell SLED where is the prefix by setting the prefix_length tensor
        en_input_ids = torch.cat((prefix_input_ids, en_input_ids), dim=-1)
        attention_mask = torch.ones_like(en_input_ids)
        self.prefix_length = torch.LongTensor([[prefix_input_ids.size(1)]])

        en_input_mask = en_input["attention_mask"]

        doc_candidate_input_features = []
        for id, (candidate, position) in enumerate(
            zip(doc.candidate_set, doc.candidate_positions)
        ):
            # (moved to extraction) Remove stopwords in a candidate
            # if remove(candidate):
            #    count += 1
            #    continue
            de_input = self.decoder_prompt + candidate + " ."
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
