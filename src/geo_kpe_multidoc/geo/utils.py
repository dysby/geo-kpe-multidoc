import itertools
import os
from datetime import datetime
from time import time
from typing import Callable, Dict

import numpy as np
import pandas as pd
from loguru import logger

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.datasets.process_mordecai import load_topic_geo_locations
from geo_kpe_multidoc.geo.measures import (
    GearyC,
    GetisOrdG,
    MoranI,
    cached_vincenty,
    inv_dist,
)


def process_geo_associations_for_topics(
    data: pd.DataFrame,
    docs_data: pd.DataFrame,
    w_function: Callable = inv_dist,
    w_function_param=1,
    save_cache=True,
) -> pd.DataFrame:
    """Process MDKPERank KPE extraction candidates, scores by document, and document geo locations (coordinates).

    Compute Geo Association Measures for each candidate, based on observation pairs $(x_i, c_i)$ where $x_i$ is a
    Keyphrase semantic score regarding a document and $c_i$ is a geo location present in that document.

    Result dataframe is saved in cache.

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    docs_data : pd.DataFrame
        _description_
    w_function : Callable, optional
        distance weighting function, by default inv_dist

    Returns
    -------
    pd.DataFrame
        Updated `data` DataFrame with geo association measures computed for each (topic, candidate) pair.
    """
    t = datetime.now()
    filename = (
        "-".join(
            [
                "geo-measures",
                w_function.__name__,
                str(w_function_param),
                t.strftime(r"%Y%m%d-%H%M%S"),
            ]
        )
        + ".parquet"
    )

    logger.info(
        f"Computing geo associations with distance function:{w_function.__name__} and a={w_function_param}"
    )

    for topic in data.index.get_level_values(0).unique():
        logger.info(f"Computing geo associations for candidates of the topic {topic}.")

        docs_coordinates = load_topic_geo_locations(topic)

        for keyphrase in data.loc[topic].index:
            logger.debug(f"Geo associations for {keyphrase}.")
            kp_scores = docs_data.loc[(topic, slice(None), keyphrase), :].droplevel(2)
            # kp_scores have the semantic scores of the keyphrase in each of the documents it appears.
            moran_i, geary_c, getis_g = geo_associations(
                kp_scores, docs_coordinates, w_function, w_function_param
            )
            data.loc[(topic, keyphrase), ["moran_i", "geary_c", "getis_g"]] = (
                moran_i,
                geary_c,
                getis_g,
            )

        # save data in cache at each loop over topics
        if save_cache:
            data.to_parquet(
                os.path.join(GEO_KPE_MULTIDOC_CACHE_PATH, "MKDUC01", filename)
            )

    if save_cache:
        logger.debug(f"Geo association measures saved in cache dir: {filename}.")

    return data


def geo_associations(
    kp_data: pd.DataFrame,
    coordinates_data: Dict,
    w_function: Callable,
    w_function_param=1,
):
    """Compute 3 measures of geospacial association for one keyphrase

    Parameters
    ----------
    kp_data : pd.DataFrame
        kp_data have the observation `semantic_score` of each keyphrase for each document it appears.

    Returns
    -------
    Tuple[float, float, float]:
        Moran I, Geary C, Getis Ord G
    """
    # topic is level 0 multiindex of the dataframe
    # topic = list(kp_data.index.get_level_values(0))[0]

    scores, w = scores_weight_matrix(
        # drop topic from index level
        kp_data["semantic_score"].droplevel(0).to_dict(),
        coordinates_data,
        w_function,
        w_function_param,
    )

    moran_i = MoranI(scores, w)
    geary_c = GearyC(scores, w)
    getis_g = GetisOrdG(scores, w)

    return (moran_i, geary_c, getis_g)


def _score_w_geo_association_I(df: pd.DataFrame, S, N, I, lambda_=1, gamma=1):
    df["score_w_geo_association_I"] = (
        df[S] * lambda_ * (df[N] - (df[N] * gamma * df[I]))
    )
    return df


def _score_w_geo_association_C(df: pd.DataFrame, S, N, C, lambda_=1, gamma=1):
    df["score_w_geo_association_C"] = df[S] * lambda_ * df[N] / (gamma * df[C])
    return df


def _score_w_geo_association_G(df: pd.DataFrame, S, N, G, lambda_=1, gamma=1):
    df["score_w_geo_association_G"] = df[S] * lambda_ * (df[N] * gamma) * df[G]
    return df


def scores_weight_matrix(
    keyphrase_scores,
    docs_locations,
    weighting_func: Callable = inv_dist,
    weighting_func_param=1,
):
    """Build observation scores array (n) and compute similatity matrix (n, n) with weigthed based distance metric.

    n = C (total candidates) * sum of # Coordenates connected with each candidate

    Candidade -(M:N)- Document -(K:L)- GeoCoordinate

    The weight matrix is row standatized.

    Parameters
    ----------
    keyphrase_scores : _type_
        _description_
    keyphrase_coordinates : _type_
        _description_

    Returns
    -------
    np.array
        Semantic score of each observation (N)
    np.array
        (N, N) array with weighted similatity for all observation pairs.

    """
    scores = []
    coordinates = []
    for doc, keyphrase_score_in_doc in keyphrase_scores.items():
        scores.extend(
            itertools.repeat(keyphrase_score_in_doc, len(docs_locations[doc]))
        )
        coordinates.extend(docs_locations[doc])

    assert len(scores) == len(coordinates)

    n = len(scores)
    scores = np.array(scores)

    start = time()
    logger.debug(f"vincenty dist start n={n}")
    weight_matrix = np.asarray(
        [
            [
                # To exploit caching the function parameter order must be the same.
                # Fortunatly distance is simetric so we keep always the smallest value first.
                # But each coordenate have 2 values threfore we compare on the sum
                # of the coordenates (lat+long vs lat+long).
                cached_vincenty(coordinates[i], coordinates[j])
                if sum(coordinates[i]) <= sum(coordinates[j])
                else cached_vincenty(coordinates[i], coordinates[j])
                for i in range(n)
            ]
            for j in range(n)
        ]
    )
    end = time()
    logger.debug(
        "vincenty distance for n={} points processing time: {:.1f}s".format(
            n, end - start
        )
    )

    # transform distance matrix to similatity measure
    weight_matrix = weighting_func(weight_matrix, weighting_func_param)
    # row standartize
    weight_matrix = weight_matrix / weight_matrix.sum(axis=0)

    end = time()
    logger.debug(
        "vincenty distance 2nd stage processing time: {:.1f}s".format(end - start)
    )

    return scores, weight_matrix
