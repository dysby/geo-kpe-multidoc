import re
from enum import Enum
from itertools import repeat
from operator import itemgetter
from typing import Callable, List, Set, Tuple

import numpy as np
from keybert.backend._utils import select_backend

from geo_kpe_multidoc.models.base_KP_model import BaseKPModel


class EnsembleMode(Enum):
    weighted = "weighted"
    harmonic = "harmonic"


class FusionModel:
    """
    Ensemble model to combine results from various models in a single source.

    Averaging mode = ["weighted", "harmonic"]

    """

    def __init__(
        self,
        models: List[BaseKPModel],
        averaging_strategy: EnsembleMode = "weighted",
        models_weights: List[float] = [0.5, 0.5],
    ):
        self.models = models
        if len(models) < 2:
            raise ValueError("Invalid model list")

        temp_name = f"{str(self.__str__).split()[3]}_["
        for model in models:
            temp_name += f"{str(model.__str__).split()[3]}_"
        self.name = f"{temp_name[:-1]}]"

        self.averaging_strategy = averaging_strategy
        self.weights = models_weights

    def extract_kp_from_corpus(
        self,
        corpus,
        dataset,
        top_n=5,
        min_len=0,
        stemming=False,
        lemmatize=False,
        **kwargs,
    ) -> List[List[Tuple]]:
        """
        Concrete method that extracts key-phrases from a list of given documents, with optional arguments
        relevant to its specific functionality. Runs through the list of models contained in self.models when
        initialized
        """

        # kpe extraction results for each topic in corpus by each model in the ensemble
        # res_by_model[model_idx]
        res_by_model = [
            model.extract_kp_from_corpus(
                corpus, dataset, -1, min_len, stemming, **kwargs
            )
            for model in self.models
        ]

        doc_num = len(res_by_model[0])  # number of docs/topics processed by model 0.

        res_docs = [[] for _ in range(doc_num)]
        # res_docs[doc_idx][model_idx]
        for model_res in res_by_model:
            for i in range(doc_num):
                res_docs[i].append(
                    [model_res[i][1], [np.float64(x[1]) for x in model_res[i][0]]]
                )

        #     [ [model_res[i][1], [np.float64(x[1]) for x in model_res[i][0]]] for model_res

        kp_score = {k: {} for k in range(doc_num)}
        for i in range(doc_num):
            # only python 3.11
            # match self.averaging_strategy:
            if self.averaging_strategy == "weighted":
                for j in range(len(self.models)):
                    res_docs[i][j][1] = res_docs[i][j][1] / np.sum(res_docs[i][j][1])

                    for k in range(len(res_docs[i][j][0])):
                        if res_docs[i][j][0][k] not in kp_score[i]:
                            kp_score[i][res_docs[i][j][0][k]] = (
                                self.weights[j] * res_docs[i][j][1][k]
                            )
                        else:
                            kp_score[i][res_docs[i][j][0][k]] += (
                                self.weights[j] * res_docs[i][j][1][k]
                            )
            elif self.averaging_strategy == "harmonic":
                res_docs[i][0][1] = res_docs[i][0][1] / np.sum(res_docs[i][0][1])
                res_docs[i][1][1] = res_docs[i][1][1] / np.sum(res_docs[i][1][1])

                first_m_res = {
                    res_docs[i][0][0][k]: res_docs[i][0][1][k]
                    for k in range(len(res_docs[i][0][0]))
                }
                second_m_res = {
                    res_docs[i][1][0][k]: res_docs[i][1][1][k]
                    for k in range(len(res_docs[i][1][0]))
                }

                # TODO: 1st model extracted keyphrases can be missing in 2nd model results?
                for kp in first_m_res:
                    if kp in second_m_res:
                        kp_score[i][kp] = (
                            2.0
                            * (first_m_res[kp] * second_m_res[kp])
                            / (first_m_res[kp] + second_m_res[kp])
                        )
            else:
                raise ValueError("self.weight is badly initialized")

        kp_score = [
            sorted(
                [(kp, kp_score[doc][kp]) for kp in kp_score[doc]],
                reverse=True,
                key=itemgetter(1),
            )
            for doc in kp_score
        ]

        return [
            (kp_score[i][:top_n], [kp[0] for kp in kp_score[i]])
            for i in range(len(kp_score))
        ]

    def _harmonic_mean(self, res_docs_i):
        ...

    def _weighted_mean(self):
        ...
