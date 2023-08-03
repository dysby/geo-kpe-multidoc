import os
import re
from functools import partial
from operator import itemgetter
from pathlib import Path
from typing import Callable, Union

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import torch
from geopandas import GeoSeries
from libpysal.weights import DistanceBand, W
from nltk.stem import StemmerI
from pysal.explore import esda

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.geo.measures import cached_vincenty, inv_dist
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_hyphens_and_dots,
    stemming,
)


def _moran_i(serie, w):
    moran = esda.moran.Moran(serie, w)
    return moran.I


def _geary_c(serie, w):
    geary = esda.geary.Geary(serie, w)
    return geary.C


def _getisord_g(serie, w):
    gog = esda.getisord.G(serie, w)
    return gog.G


def MoranI(
    scores: np.array,
    weight_matrix: np.array,
    device: torch.device = torch.device("cpu"),
):
    r"""
    .. math::
        I = \frac{z\top W z}{z \top z}
    Return
    ------
    float:
        we follow `pysal` for special cases and `return 1` when all samples have
        the same value regardless of spacial position.
    """
    n = len(scores)

    if n < 2:
        return np.nan

    std = scores.std()
    if std == 0:
        return 1

    z = (scores - scores.mean()) / (scores.std())

    z = torch.Tensor(z.to_numpy()).to(device)
    w = torch.Tensor(weight_matrix).to(device)

    moranI = z.T @ w @ z / (z.T @ z)

    moranI = moranI.cpu().numpy().item()
    return moranI


def GearyC(scores, weight_matrix):
    """The value of Geary's C lies between 0 and some unspecified value greater than 1.
    Values significantly lower than 1 demonstrate increasing positive spatial autocorrelation,
    whilst values significantly higher than 1 illustrate increasing negative spatial autocorrelation.
    [wikipedia]

    Return
    ------
    float:
        we follow `pysal` for special cases and `return 0` when all samples have
        the same value regardless of spacial position.
    """

    N = len(scores)

    if N < 2:
        # logger.warning("GearyC over a single observation. Returning np.NAN.")
        return np.nan

    scores = scores.to_numpy()

    mean = np.mean(scores)
    # sum_adjusted_scores = np.sum([(score - mean) ** 2 for score in scores])
    sum_adjusted_scores = np.sum((scores - mean) ** 2)

    if np.isclose(sum_adjusted_scores, 0):
        # logger.debug("GearyC over a constant surface. Returning 0.")
        return 0.0

    outer_difference_scores = np.subtract.outer(scores, scores)
    outer_difference_scores = outer_difference_scores**2

    sum1 = (
        weight_matrix * outer_difference_scores
    ).sum()  # elementwize product and sum all
    sum2 = weight_matrix.sum()

    gearyC = ((N - 1.0) * sum1) / (2.0 * sum2 * sum_adjusted_scores)
    return gearyC


def GetisOrdG(scores, weight_matrix):
    n = len(scores)

    if n < 2:
        # logger.warning("GetisOrdG over a single value. Returning np.NAN.")
        return np.nan

    scores = scores.to_numpy()

    outer_mul_scores = np.outer(np.asarray(scores), np.asarray(scores))

    sum1 = (weight_matrix * outer_mul_scores).sum()  # elementwize product and sum all
    sum2 = outer_mul_scores.sum()

    getisOrdG = sum1 / sum2
    return getisOrdG


def compute_weights_fully_connected(
    points: GeoSeries,
    weight_function: Callable,
    weight_function_param: Union[int, float],
) -> dict:
    # cord_list (lat, long)
    coordinates = [(y, x) for x, y in zip(points.x, points.y)]
    n = len(coordinates)

    weight_matrix = np.zeros((n, n))
    inds = np.triu_indices_from(weight_matrix, k=1)  # k=1 don't compute diagonal
    for i, j in zip(*inds):
        weight_matrix[i, j] = (
            # To exploit caching the function parameter order must be the same.
            # Fortunatly, distance is simetric so we keep always the smallest value first,
            # by comparison on the sum (lat+long vs lat+long).
            cached_vincenty(coordinates[i], coordinates[j])
            if sum(coordinates[i]) <= sum(coordinates[j])
            else cached_vincenty(coordinates[j], coordinates[i])
        )
    weight_matrix = weight_matrix + weight_matrix.T  # - np.diag(0, n)

    weight_matrix = weight_function(weight_matrix, weight_function_param)

    weights = {i: [weight_matrix[i, j] for j in range(n) if j != i] for i in range(n)}

    return weights


def compute_weights_matrix(
    points: GeoSeries,
    weight_function: Callable,
    weight_function_param: Union[int, float],
) -> np.array:
    # cord_list (lat, long)
    coordinates = [(y, x) for x, y in zip(points.x, points.y)]
    n = len(coordinates)

    weight_matrix = np.zeros((n, n))
    inds = np.triu_indices_from(weight_matrix, k=1)  # k=1 don't compute diagonal
    for i, j in zip(*inds):
        weight_matrix[i, j] = (
            # To exploit caching the function parameter order must be the same.
            # Fortunatly, distance is simetric so we keep always the smallest value first,
            # by comparison on the sum (lat+long vs lat+long).
            cached_vincenty(coordinates[i], coordinates[j])
            if sum(coordinates[i]) <= sum(coordinates[j])
            else cached_vincenty(coordinates[i], coordinates[j])
        )
    weight_matrix = weight_matrix + weight_matrix.T  # - np.diag(0, n)

    weight_matrix = weight_function(weight_matrix, weight_function_param)

    # row standartize
    weight_matrix = weight_matrix / weight_matrix.sum(axis=1)

    return weight_matrix


class MdGeoRank:
    def __init__(
        self,
        experiment: str,
        stemmer: StemmerI,
        d_threshold=1000,
        weight_function: Callable = inv_dist,
        weight_function_param=500,
        **kwargs
    ) -> None:
        self.experiment = experiment
        self.d_threshold = d_threshold
        self.weight_function = weight_function
        self.weight_function_param = weight_function_param
        self.stemmer = stemmer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_files(self, path: str):
        mdkpe_file_name_pattern = re.compile(r"d\d{2}-mdkpe-results\.pkl")
        for file in os.listdir(path):
            if os.path.isfile(
                os.path.join(path, file)
            ) and mdkpe_file_name_pattern.match(file):
                yield file

    def _load_semantic_rank_model_outputs(self):
        # speedup on hlt cluster
        loaded_model_files = {
            filename[:3]: joblib.load(
                Path(GEO_KPE_MULTIDOC_CACHE_PATH).joinpath(self.experiment, filename)
            )
            for filename in self._get_files(
                Path(GEO_KPE_MULTIDOC_CACHE_PATH).joinpath(self.experiment)
            )
        }
        return loaded_model_files

    def _load_md_coordinates(self):
        coordinates_file_path = Path(GEO_KPE_MULTIDOC_CACHE_PATH).joinpath(
            "MKDUC01-topic-doc-coordinates-mordecai3-geopandas-20230615.parquet"
        )

        topic_doc_coordinates = (
            gpd.read_parquet(coordinates_file_path)
            .drop(["location", "x", "y", "n"], axis=1)
            .set_index(["topic_id", "doc"])
        )
        topic_doc_coordinates.columns = ["_geometry"]
        return topic_doc_coordinates

    def geo_association(self):
        topic_doc_coordinates = self._load_md_coordinates()
        semantic_rank_model_outputs = self._load_semantic_rank_model_outputs()

        results = pd.DataFrame()

        for topic_id, topic_model_outputs in semantic_rank_model_outputs.items():
            """
            topic_model_outputs: dict_keys(['dataset',
                                            'topic',
                                            'top_n_scores',
                                            'candidate_document_matrix',
                                            'gold_kp',
                                            'documents_embeddings',
                                            'candidate_embeddings',
                                            'ranking_p_doc'])
            """
            candidate_scores_per_doc = pd.DataFrame(
                [
                    {"doc": doc_id, "candidate": candidate, "score": score}
                    for doc_id, doc_scores in topic_model_outputs[
                        "score_per_document"
                    ].items()
                    for candidate, score in doc_scores[0]
                ]
            ).pivot(index="doc", columns="candidate", values="score")

            candidate_scores = pd.DataFrame(
                candidate_scores_per_doc.mean(), columns=["semantic_score"]
            )

            gold_stem = topic_model_outputs["gold_kp"]

            candidate_scores["in_gold"] = False
            for i, kp in enumerate(candidate_scores.index):
                if stemming(remove_hyphens_and_dots(kp)) in gold_stem:
                    candidate_scores.loc[kp, "in_gold"] = True

            # number of documents where the candidate is present, within the current topic.
            # TEMP
            df = (
                pd.DataFrame()
                .reindex_like(candidate_scores_per_doc)
                .fillna(0)
                .astype(int)
                .transpose()
            )

            for cand, docs in topic_model_outputs["candidate_document_matrix"].items():
                for doc in docs:
                    df.loc[cand, doc] += 1

            candidate_scores["N"] = df.sum(axis=1)
            # END TEMP

            # candidate_scores["N"] = topic_model_outputs["candidate_document_matrix"].sum(axis=1)
            candidate_scores.index.name = "candidate"

            topic_point_scores = gpd.GeoDataFrame(
                candidate_scores_per_doc.join(topic_doc_coordinates.loc[topic_id])
            ).reset_index(drop=True)
            # when the same location apears in multiple documents, the semantic score for that location is the mean of values for that location
            topic_point_scores = (
                topic_point_scores.groupby("_geometry").mean().reset_index()
            )
            topic_point_scores = gpd.GeoDataFrame(topic_point_scores)

            # print(f"{topic_id}: topic_point_scores: {len(topic_point_scores)} observations.")
            topic_point_scores = topic_point_scores[
                ~topic_point_scores["_geometry"].is_empty
            ]
            # print(f"{topic_id}: topic_point_scores: {len(topic_point_scores)} without empty observations")

            # If custom weight function
            # neighbors={'c': ['b'], 'b': ['c', 'a'], 'a': ['b']}
            # # neighbors = { i: neighbors for i in range(n_points)}

            weights = compute_weights_matrix(
                topic_point_scores["_geometry"],
                self.weight_function,
                self.weight_function_param,
            )

            topic_point_scores = topic_point_scores.drop("_geometry", axis=1)

            # n_points = len(topic_point_scores)
            # neighbors = {i: [j for j in range(n_points) if j != i] for i in range(n_points)}
            # weights = compute_weights_fully_connected(scores,
            #                                           weight_function,
            #                                           weight_function_param)
            # w = W(neighbors, weights)
            # w.transform = "R"

            __moran_i = partial(MoranI, weight_matrix=weights)
            # only a few values maybe not usefull to move to GPU
            # __moran_i = partial(MoranI, weight_matrix=weights, device=self.device)
            __geary_c = partial(GearyC, weight_matrix=weights)
            __getisord_g = partial(GetisOrdG, weight_matrix=weights)

            geo_associations = pd.DataFrame()
            geo_associations["moran_i"] = topic_point_scores.agg(__moran_i)
            geo_associations["geary_c"] = topic_point_scores.agg(__geary_c)
            geo_associations["getisord_g"] = topic_point_scores.agg(__getisord_g)

            topic_scores = candidate_scores.join(geo_associations)
            # topic_scores["N"] = N
            # topic_scores.index.name = "candidate"
            topic_scores["topic"] = topic_id
            topic_scores.set_index(["topic", topic_scores.index])
            results = pd.concat([results, topic_scores])

        results = results.set_index(["topic", results.index])
        return results

    def _rank(self, geo_association_results: pd.DataFrame):
        """
        compose final score and transform results to evaluation format



        Parameters
        ----------
        geo_association_results : pd.DataFrame
            _description_

        Returns
        -------
        dict:   {
                    dataset_name: [
                        ( top_n_cand_and_scores: List[Tuple[str, float]] , candidates: List[str]),
                        ...
                        ],
                    ...
                }
        """
        #

        final_score = pd.DataFrame().reindex_like(geo_association_results)

        final_score["score"] = (
            geo_association_results["score"] + geo_association_results["moran_i"]
        )

        rankings = dict()
        for dataset in ["dataset"]:
            for topic in final_score.index.get_level_values(0):
                top_n_scores = sorted(
                    list(
                        final_score.loc[topic]["score"].items(),
                        key=itemgetter(1),
                        reversed=True,
                    )
                )
                candidates = final_score.loc[topic].index.to_list()
                rankings.setdefault(dataset, []).append((top_n_scores, candidates))

        return rankings
