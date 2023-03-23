import itertools
import math
import os
import time
from datetime import datetime
from typing import Callable, Dict

import numpy as np
import pandas as pd
from loguru import logger
from vincenty import vincenty

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH
from geo_kpe_multidoc.datasets.process_mordecai import load_topic_geo_locations

r"""
Functions
---------


\begin{align}
f(x) = 1 / (a + x) \\
f(x) = exp(- x^a)  \\
f(x) = arccot(ax)
\end{align}
"""


def inv_dist(d: np.ndarray, a=1):
    r"""$f(x) = \frac{1}{a + d}$"""
    return 1 / (a + d)


def exp_dist(d: np.ndarray, a=1):
    r"""$f(x) = exp(- x^a)$"""
    return np.e ** (-(d**a))


def arc_dist(d: np.ndarray, a=1):
    r"""$f(x) = arccot(a x) = arctan(1 / (a x))$
    Abramowitz, M. and Stegun, I. A., Handbook of Mathematical Functions, 10th printing, New York: Dover, 1964, pp. 79.
    https://personal.math.ubc.ca/~cbm/aands/page_79.htm
    """
    return np.arctan(1 / (a * d))


def orig_dist(d: np.ndarray):
    return 1.0 / (np.e**d)


from functools import lru_cache


@lru_cache(maxsize=1000)
def cached_vincenty(c1, c2):
    """Vincenty geo distance formula with caching.

    Parameters
    ----------
    c1 : Tuple[lat: float, long: float]
        Point geo coordenate
    c2 : Tuple[lat: float, long: float]
        Point geo coordenate

    Returns
    -------
    float
        Geo distance in Km.
    """
    return vincenty(c1, c2)


def preprocess_scores_weight_matrix(
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

    start = time.time()
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
    end = time.time()
    logger.debug(
        "vincenty distance for n={} points processing time: {:.1f}s".format(
            n, end - start
        )
    )

    # transform distance matrix to similatity measure
    weight_matrix = weighting_func(weight_matrix, weighting_func_param)
    # row standartize
    weight_matrix = weight_matrix / weight_matrix.sum(axis=0)

    end = time.time()
    logger.debug(
        "vincenty distance 2nd stage processing time: {:.1f}s".format(end - start)
    )

    return scores, weight_matrix


def process_geo_associations_for_topics(
    data: pd.DataFrame,
    docs_data: pd.DataFrame,
    w_function: Callable = inv_dist,
    w_function_param=1,
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
        data.to_parquet(os.path.join(GEO_KPE_MULTIDOC_CACHE_PATH, "MKDUC01", filename))

    return data


def geo_associations(
    kp_data: pd.DataFrame,
    coordinates_data: Dict,
    w_function: Callable,
    w_function_param=1,
):
    # topic is level 0 multiindex of the dataframe
    # topic = list(kp_data.index.get_level_values(0))[0]

    scores, w = preprocess_scores_weight_matrix(
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


def MoranI(scores, weight_matrix):
    """
    Return
    ------
    float:
        we follow `pysal` for special cases and `return 1` when all samples have
        the same value regardless of spacial position.
    """
    n = len(scores)

    if n == 1 or n == 0:
        logger.warning("MoranI over a single observation. Returning np.NAN.")
        return np.nan

    mean = np.mean(scores)
    adjusted_scores = scores - mean

    if all(np.isclose(adjusted_scores, 0)):
        logger.debug("MoranI over a constant surface. Returning 1.")
        return 1

    moranI = n / np.sum(adjusted_scores**2)

    outer_mul_scores = np.outer(adjusted_scores, adjusted_scores)

    sum1 = (weight_matrix * outer_mul_scores).sum()  # elementwize product and sum all
    sum2 = weight_matrix.sum()
    moranI = moranI * (sum1 / sum2)
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

    n = len(scores)

    if n == 1 or n == 0:
        logger.warning("GearyC over a single observation. Returning np.NAN.")
        return np.nan

    mean = np.mean(scores)
    # sum_adjusted_scores = np.sum([(score - mean) ** 2 for score in scores])
    sum_adjusted_scores = np.sum((scores - mean) ** 2)

    if np.isclose(sum_adjusted_scores, 0):
        logger.debug("GearyC over a constant surface. Returning 0.")
        return 0

    outer_difference_scores = np.subtract.outer(scores, scores)
    outer_difference_scores = outer_difference_scores**2

    sum1 = (
        weight_matrix * outer_difference_scores
    ).sum()  # elementwize product and sum all
    sum2 = weight_matrix.sum()

    gearyC = ((n - 1.0) * sum1) / (2.0 * sum2 * sum_adjusted_scores)
    return gearyC


def GetisOrdG(scores, weight_matrix):
    n = len(scores)

    if n == 1 or n == 0:
        logger.warning("GetisOrdG over a single value. Returning np.NAN.")
        return np.nan

    outer_mul_scores = np.outer(np.asarray(scores), np.asarray(scores))

    sum1 = (weight_matrix * outer_mul_scores).sum()  # elementwize product and sum all
    sum2 = outer_mul_scores.sum()

    getisOrdG = sum1 / sum2
    return getisOrdG


# """
# boston = (42.3541165, -71.0693514)
# newyork = (40.7791472, -73.9680804)

# keyphrase_scores = {
#     "candidate 1" : 0.5,
#     "candidate 2" : 0.8
# }

# keyphrase_coordinates = {
#     "candidate 1" : [ boston, newyork ],
#     "candidate 2" : [ boston ]
# }

# MoranI(keyphrase_scores , keyphrase_coordinates)
# GearyC(keyphrase_scores , keyphrase_coordinates)
# GetisOrdG(keyphrase_scores , keyphrase_coordinates)
# """


def MoranI_alt(keyphrase_scores, keyphrase_coordinates):
    """
    TODO: to test the loop must be for keyphrase scores (one per document),
        and the keyphrase_coordinates are in fact doc_coordinates.
    """
    scores = []
    coordinates = []
    for key, value_list in keyphrase_coordinates.items():
        for value in value_list:
            scores.append(keyphrase_scores[key])
            coordinates.append(value)
    n = len(scores)
    scores = np.array(scores)
    mean = np.mean(scores)
    adjusted_scores = [(score - mean) ** 2 for score in scores]
    moranI = n / np.sum(adjusted_scores)
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        for j in range(n):
            # distance = 1.0 / (np.e ** vincenty(coordinates[i], coordinates[j]))
            distance = 1.0 / (1 + vincenty(coordinates[i], coordinates[j]))
            sum1 += distance * (scores[i] - mean) * (scores[j] - mean)
            sum2 += distance
    moranI = moranI * (sum1 / sum2)
    return moranI


def GearyC_alt(keyphrase_scores, keyphrase_coordinates):
    scores = []
    coordinates = []
    for key, value_list in keyphrase_coordinates.items():
        for value in value_list:
            scores.append(keyphrase_scores[key])
            coordinates.append(value)
    n = len(scores)
    scores = np.array(scores)
    mean = np.mean(scores)
    sum_adjusted_scores = np.sum([(score - mean) ** 2 for score in scores])
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        for j in range(n):
            # distance = 1.0 / (np.e ** vincenty(coordinates[i], coordinates[j]))
            distance = 1.0 / (1 + vincenty(coordinates[i], coordinates[j]))
            sum1 += distance * ((scores[i] - scores[j]) ** 2)
            sum2 += distance
    gearyC = ((n - 1.0) * sum1) / (2.0 * sum2 * sum_adjusted_scores)
    return gearyC


def GetisOrdG_alt(keyphrase_scores, keyphrase_coordinates):
    scores = []
    coordinates = []
    for key, value_list in keyphrase_coordinates.items():
        for value in value_list:
            scores.append(keyphrase_scores[key])
            coordinates.append(value)
    n = len(scores)
    scores = np.array(scores)
    sum1 = 0.0
    sum2 = 0.0
    for i in range(n):
        for j in range(n):
            # distance = 1.0 / (np.e ** vincenty(coordinates[i], coordinates[j]))
            distance = 1.0 / (1 + vincenty(coordinates[i], coordinates[j]))
            sum1 += distance * scores[i] * scores[j]
            sum2 += scores[i] * scores[j]
    getisOrdG = sum1 / sum2
    return getisOrdG
