"""
Evaluation Metrics implementations for string predictions and string true labels
"""
import numpy as np


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


def precision(predictions, true_label):
    p = set(predictions)
    t = set(true_label)
    return min(
        1.0,
        # len([kp for kp in predictions if kp in true_label]) / float(len(predictions)),
        len(p.intersection(t)) / len(p),
    )


def recall(predictions, true_label):
    p = set(predictions)
    t = set(true_label)
    return min(
        1.0,
        # (len([kp for kp in predictions if kp in true_label]) / len(true_label)),
        len(p.intersection(t)) / len(t),
    )


def f1_score(precision, recall):
    f1 = 0.0
    if precision != 0 and recall != 0:
        f1 = (2.0 * precision * recall) / (precision + recall)

    return f1


def MAP(top_kp, true_label):
    ap = [
        len([k for k in top_kp[: i + 1] if k in true_label]) / float(i + 1)
        for i in range(len(top_kp))
        if top_kp[i] in true_label
    ]
    map = np.sum(ap) / float(len(true_label))
    return map


def nDCG(top_kp, true_label):
    ndcg = np.sum(
        [
            1.0 / np.log2(i + 1)
            for i in range(1, len(top_kp) + 1)
            if top_kp[i - 1] in true_label
        ]
    )
    ndcg = ndcg / np.sum([1.0 / np.log2(i + 1) for i in range(1, len(true_label) + 1)])
    return ndcg
