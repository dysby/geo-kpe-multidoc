"""
Evaluation Metrics implementations for string predictions and string true labels
"""
import numpy as np


def compare_with_gold_fuzzy(
    detected_keywords, gold_standard_keywords, fuzzy_threshold=3, keyword_separator=";"
):
    """
    from https://github.com/guanjianyu/rakun
    Fuzzy comparison of keyword matches. Given a fuzzy edit distance threshold,
    how many  keywords out of the top 10 are OK?
    input: detected_keywords (list of string).
    input: gold_standard_keywords (list of strings).
    input: fuzzy_threshold (int) -> max acceptable edit distance.
    """

    precision_correct = 0
    precision_overall = 0

    recall_correct = 0
    recall_overall = 0

    for enx, keyword_set in enumerate(detected_keywords):
        gold_standard_set = gold_standard_keywords[enx]
        count = 0
        method_keywords = keyword_set.split(keyword_separator)

        if type(gold_standard_set) is float:  ## this is np.nan -> not defined.
            continue

        gold_standard_set = set(gold_standard_set.split(keyword_separator))

        top_n = len(gold_standard_set)
        if top_n >= len(method_keywords):
            top_n = len(method_keywords)

        ## recall
        parsed_rec = set()
        for el in method_keywords:
            if not el in parsed_rec:
                parsed_rec.add(el)
                if el in gold_standard_set:
                    recall_correct += 1
        recall_overall += top_n

        ## precision
        parsed_prec = set()
        for el in method_keywords:
            if not el in parsed_prec:
                parsed_prec.add(el)
                if el in gold_standard_set:
                    precision_correct += 1
        precision_overall += len(method_keywords)

    precision = float(precision_correct) / (
        precision_overall
    )  ## Number of correctly predicted over all predicted (num gold)

    recall = float(recall_correct) / (
        recall_overall
    )  ## Correct over all detected keywords

    if (precision + recall) > 0:
        F1 = 2 * (precision * recall) / (precision + recall)

    else:
        F1 = 0

    return precision, recall, F1


def get_score_full(candidates, references, maxDepth=15):
    # from https://github.com/xnliang98/uke_ccrank/
    # DOI:10.18653/v1/2021.emnlp-main.14
    precision = []
    recall = []
    reference_set = set(references)
    referencelen = len(reference_set)
    true_positive = 0
    for i in range(maxDepth):
        if len(candidates) > i:
            kp_pred = candidates[i]
            if kp_pred in reference_set:
                true_positive += 1
            precision.append(true_positive / float(i + 1))
            recall.append(true_positive / float(referencelen))
        else:
            precision.append(true_positive / float(len(candidates)))
            recall.append(true_positive / float(referencelen))
    return precision, recall


def precision(predictions, true_label) -> float:
    # & for intersection.
    p = set(predictions)
    t = set(true_label)
    pr = len(p & t) / len(p)
    return pr


def recall(predictions, true_label) -> float:
    p = set(predictions)
    t = set(true_label)
    r = len(p & t) / len(t)
    return r


def f1_score(precision, recall) -> float:
    f1 = 0.0
    if precision != 0 and recall != 0:
        f1 = (2.0 * precision * recall) / (precision + recall)

    return f1


def MAP(top_kp, true_label) -> float:
    ap = [
        len([k for k in top_kp[: i + 1] if k in true_label]) / float(i + 1)
        for i in range(len(top_kp))
        if top_kp[i] in true_label
    ]
    map = np.sum(ap) / float(len(true_label))
    return map


def nDCG(top_kp, true_label) -> float:
    ndcg = np.sum(
        [
            1.0 / np.log2(i + 1)
            for i in range(1, len(top_kp) + 1)
            if top_kp[i - 1] in true_label
        ]
    )
    ndcg = ndcg / np.sum([1.0 / np.log2(i + 1) for i in range(1, len(true_label) + 1)])
    return ndcg
