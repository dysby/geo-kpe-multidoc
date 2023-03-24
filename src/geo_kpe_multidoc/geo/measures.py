from typing import Dict

import numpy as np
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
    r"""$f(x) = arccot(a x) = arccot(z) = arctan(1 / z)
    Abramowitz, M. and Stegun, I. A., Handbook of Mathematical Functions, 10th printing, New York: Dover, 1964, pp. 79.
    https://personal.math.ubc.ca/~cbm/aands/page_79.htm

    """
    EPSILON = 1e-20
    if d == 0:
        d = EPSILON
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
