import math
import time
from typing import Callable, Dict

import numpy as np
import pandas as pd
from loguru import logger
from vincenty import vincenty

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
    r"""$f(x) = arccot(a x)$"""
    return math.arccot(a * d)


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
    keyphrase_scores, keyphrase_coordinates, weighting_func
):
    """Build observation scores array (n) and compute similatity matrix (n, n) with weigthed based distance metric.

    n = C (total candidates) * sum of # Coordenates connected with each candidate

    Candidade -(M:N)- Document -(K:L)- GeoCoordinate

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
    for key, value_list in keyphrase_coordinates.items():
        for value in value_list:
            scores.append(keyphrase_scores[key])
            coordinates.append(value)
    n = len(scores)
    scores = np.array(scores)

    start = time.time()
    logger.info(f"vincenty dist start n={n}")
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
    logger.debug("vincenty dist time n={}: {:.1f}".format(n, end - start))

    # transforme distance matrix to similatity measure
    weight_matrix = weighting_func(weight_matrix)

    end = time.time()
    logger.debug("vincenty dist stage 2 time n={}: {:.1f}".format(n, end - start))

    return scores, weight_matrix


def geo_associations(
    kp_data: pd.DataFrame, coordinates_data: Dict, w_function: Callable
):
    # topic is level 0 multiindex of the dataframe
    topic = list(kp_data.index.get_level_values(0))[0]

    coordinates = {
        kp: coordinates_data[topic][kp]
        for kp in kp_data.index.get_level_values(1).unique().to_list()
    }

    scores, w = preprocess_scores_weight_matrix(
        # drop topic from index level
        kp_data["semantic_score"].droplevel(0).to_dict(),
        coordinates,
        w_function,
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
    mean = np.mean(scores)

    if n == 1 or n == 0:
        logger.warning("MoranI over a single observation. Returning np.NAN.")
        return np.nan

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
    mean = np.mean(scores)

    if n == 1 or n == 0:
        logger.warning("GearyC over a single observation. Returning np.NAN.")
        return np.nan
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


# def MoranI(keyphrase_scores, keyphrase_coordinates):
#     scores = []
#     coordinates = []
#     for key, value_list in keyphrase_coordinates.items():
#         for value in value_list:
#             scores.append(keyphrase_scores[key])
#             coordinates.append(value)
#     n = len(scores)
#     scores = np.array(scores)
#     mean = np.mean(scores)
#     adjusted_scores = [(score - mean) ** 2 for score in scores]
#     moranI = n / np.sum(adjusted_scores)
#     sum1 = 0.0
#     sum2 = 0.0
#     for i in range(n):
#         for j in range(n):
#             distance = 1.0 / (np.e ** vincenty(coordinates[i], coordinates[j]))
#             sum1 += distance * (scores[i] - mean) * (scores[j] - mean)
#             sum2 += distance
#     moranI = moranI * (sum1 / sum2)
#     return moranI


# def GearyC(keyphrase_scores, keyphrase_coordinates):
#     scores = []
#     coordinates = []
#     for key, value_list in keyphrase_coordinates.items():
#         for value in value_list:
#             scores.append(keyphrase_scores[key])
#             coordinates.append(value)
#     n = len(scores)
#     scores = np.array(scores)
#     mean = np.mean(scores)
#     sum_adjusted_scores = np.sum([(score - mean) ** 2 for score in scores])
#     sum1 = 0.0
#     sum2 = 0.0
#     for i in range(n):
#         for j in range(n):
#             distance = 1.0 / (np.e ** vincenty(coordinates[i], coordinates[j]))
#             sum1 += distance * ((scores[i] - scores[j]) ** 2)
#             sum2 += distance
#     gearyC = ((n - 1.0) * sum1) / (2.0 * sum2 * sum_adjusted_scores)
#     return gearyC


# def GetisOrdG(keyphrase_scores, keyphrase_coordinates):
#     scores = []
#     coordinates = []
#     for key, value_list in keyphrase_coordinates.items():
#         for value in value_list:
#             scores.append(keyphrase_scores[key])
#             coordinates.append(value)
#     n = len(scores)
#     scores = np.array(scores)
#     sum1 = 0.0
#     sum2 = 0.0
#     for i in range(n):
#         for j in range(n):
#             distance = 1.0 / (np.e ** vincenty(coordinates[i], coordinates[j]))
#             sum1 += distance * scores[i] * scores[j]
#             sum2 += scores[i] * scores[j]
#     getisOrdG = sum1 / sum2
#     return getisOrdG
