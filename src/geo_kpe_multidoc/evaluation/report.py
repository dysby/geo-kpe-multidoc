import os
import re
from datetime import datetime
from itertools import chain, zip_longest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from tabulate import tabulate

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_OUTPUT_PATH


def output_top_cands(
    model_results: dict[str, list] = {}, true_labels: dict[str, tuple[list, ...]] = {}
):
    """
    Print Top N candidates that are in Gold Candidate list

    Parameters:
    -----------
        model_results: values are a list with results for each document [((doc1_top_n_candidates, doc1_top_n_scores], doc1_candidates), ...]
    """
    top_cand_l = []
    for dataset in model_results:
        for i in range(len(model_results[dataset])):
            top_kp_and_score = model_results[dataset][i][0]
            true_label = true_labels[dataset][i]
            top_cand_l += [
                round(float(score), 2)
                for kp, score in top_kp_and_score
                if kp in true_label
            ]
    print(top_cand_l)

    top_cand_sims = {
        round(float(x), 2): (0 + top_cand_l.count(round(float(x), 2)))
        for x in np.arange(0, 1.01, 0.01)
    }
    print(top_cand_sims)


def output_one_top_cands(
    doc_ids: list[str],
    model_results: dict[str, list] = {},
    true_labels: dict[str, tuple[list]] = {},
    top_n: int = 20,
    doc_id: str = None,
) -> str:
    """
    Print one example Top N candidate and Gold Candidate list

    Parameters:
    -----------
        model_results: values are a list with results for each document [((doc1_top_n_candidates, doc1_top_n_scores], doc1_candidates), ...]
    """

    for dataset in model_results.keys():
        doc_idx = doc_ids.index(doc_id) if doc_id else 0
        # doc_keys = [kp for kp, _ in model_results[dataset][doc_idx][0]]
        doc_keys = model_results[dataset][doc_idx][0]
        gold_keys = true_labels[dataset][doc_idx]
        # print(f"Keyphrase extraction for {doc_id}")
        table = tabulate(
            [
                [dk[0] if len(dk) > 1 else dk, dk[1] if len(dk) > 1 else dk, gk]
                for dk, gk in zip_longest(doc_keys[:top_n], gold_keys, fillvalue="-")
            ],
            headers=["Extracted", "Score", "Gold"],
            floatfmt=".5f",
        )
        # print(table)
    return table


def output_one_top_cands_geo(
    doc_ids: list[str],
    model_results: dict[str, list] = {},
    true_labels: dict[str, tuple[list]] = {},
    top_n: int = 20,
):
    """
    Print one example Top N candidate and Gold Candidate list

    Parameters:
    -----------
        model_results: values are a list with results for each document
            [((doc1_top_n_candidates, doc1_top_n_scores], doc1_candidates), ...]
    """
    for dataset in model_results.keys():
        # print only 1 document example
        top_n_and_scores, candidates = dataset[0]
        gold_keys = true_labels[dataset][0]
        print(doc_ids[0])
        for ranking_type, (candidates_scores, candidades) in top_n_and_scores.items():
            doc_keys = [kp for kp, _ in candidates_scores]
            print(f"Table for {ranking_type}")
            print(
                tabulate(
                    [
                        [dk, gk]
                        for dk, gk in zip_longest(
                            doc_keys[:top_n], gold_keys, fillvalue="-"
                        )
                    ],
                    headers=["Extracted", "Gold"],
                )
            )


def diff_greedy(doc_idx):
    kpe = pd.DataFrame()
    kpe_lemma = pd.DataFrame()
    kpe_greedy = pd.DataFrame()
    print(
        tabulate(
            [
                (c1, c2, c3)
                for c1, c2, c3 in zip_longest(
                    kpe[(kpe.doc == doc_idx) & (kpe.in_gold)]["candidate"],
                    kpe_lemma[(kpe_lemma.doc == doc_idx) & (kpe_lemma.in_gold)][
                        "candidate"
                    ],
                    kpe_greedy[(kpe_greedy.doc == doc_idx) & (kpe_greedy.in_gold)][
                        "candidate"
                    ],
                    fillvalue="-",
                )
            ],
            headers=["base", "lemma", "greedy"],
        )
    )


def plot_score_distribuitions_with_gold(
    results: pd.DataFrame, title: str, xlim=None
) -> plt.Figure:
    plt.rcParams["figure.figsize"] = (6.4, 4.8)

    results = results[np.isfinite(results["score"])]

    fig, ax = plt.subplots()
    ax = results[~results["in_gold"]]["score"].plot.hist(density=True)
    ax = results[results["in_gold"]]["score"].plot.hist(density=True, alpha=0.5)
    if xlim is not None and np.all(np.isfinite(xlim)):
        ax.set_xlim(xlim)
    ax.set_title(title, fontsize=12)
    plt.legend(["non-gold", "gold"])
    return fig


def plot_non_versus_gold_density(
    gold_values: pd.Series,
    not_gold_values: pd.Series,
    title: str = None,
    density=False,
    legend=None,
):
    _, ax = plt.subplots()

    legend = legend if legend else ["gold", "non-gold"]

    gold_values.hist(ax=ax, alpha=0.5, bins=20, density=density, color="orange")
    not_gold_values.hist(ax=ax, alpha=0.5, bins=20, density=density, color="blue")
    ax.set_title(title)
    plt.legend(legend)
    plt.show()


def table_latex(df, caption=None, label=None, percentage=False):
    s = df.style
    if percentage:
        s.format("{:.2%}")
    else:
        s.format(precision=3)

    s.format_index(lambda s: re.sub("_(\d+)", r"_{\1}", s), axis=1)
    return s.to_latex(
        label=label,
        caption=caption,
        clines="skip-last;data",
        # clines = "all;data",
        convert_css=True,
        position_float="centering",
        multicol_align="|c|",
        hrules=True,
    )


def print_fig_latex(filename, caption="Caption", label="label"):
    print(
        f"""\\begin{{figure}}
    \centering
    \includegraphics[width=\\textwidth]{{fig/{filename}}}
    \caption{{{caption}}}
    \label{{fig:{label}}}
    \end{{figure}}
    """
    )


def plot_distributions(df, name=None):
    if not name:
        t = datetime.now().strftime(r"%Y%m%d-%H%M%S")
        name = f"plot-geo-distributions-{t}.pdf"

    # f, axs = plt.subplots(3, 1, figsize=(8, 4), gridspec_kw=dict(width_ratios=[4, 3]))
    f, axs = plt.subplots(1, 3, figsize=(12, 4))
    sb.histplot(df, x="moran_i", hue="gold", bins=40, legend=False, ax=axs[0])
    sb.histplot(df, x="geary_c", hue="gold", bins=40, legend=False, ax=axs[1])
    sb.histplot(df, x="getis_g", hue="gold", bins=40, legend=True, ax=axs[2])
    f.tight_layout()
    plt.savefig(os.path.join(GEO_KPE_MULTIDOC_OUTPUT_PATH, name), dpi=100)
    print_fig_latex(name, caption=name.replace("-", " ").replace("_", " "))


def plot_dist(
    df, idx_filter, title=None, bins=20, column="moran_i", by="gold", ax=None
):
    # p = sb.histplot(
    #     df.loc[idx_filter]
    #       .groupby(by, group_keys=False),
    #       #.apply(lambda x: x.sample(min(len(x), min(np.unique(df.loc[idx_filter][by], return_counts=True)[1]))).sample(frac=1)),
    #     stat='density',
    #     x=column, hue=by, bins=bins, ax=ax
    # )
    # p.set_title(title)

    sb.histplot(
        df[~df.in_gold].loc[idx_filter],
        stat="probability",
        x=column,
        color="steelblue",
        bins=20,
        ax=ax,
    )

    sb.histplot(
        df[df.in_gold].loc[idx_filter],
        stat="probability",
        x=column,
        color="goldenrod",
        bins=20,
        ax=ax,
        alpha=0.5,
    )
    if ax:
        ax.set_title(title)


def save_plot(p, name, caption="Caption", label="label"):
    p.get_figure().savefig(os.path.join(GEO_KPE_MULTIDOC_OUTPUT_PATH, name), dpi=100)
    print_fig_latex(name, caption, label)
    return p


def add_gold_label(df, gold):
    """
    Mutate dataframe `df` adding a label column if candidate is in the gold set.
    """
    gold_idx = pd.MultiIndex.from_tuples(
        chain.from_iterable(
            df.index[
                df.index.isin([topic], level=0) & df.index.isin(gold[topic], level=1)
            ]
            for topic in df.index.get_level_values(0).unique()
        ),
        names=["topic", "keyphrases"],
    )

    not_gold_idx = pd.MultiIndex.from_tuples(
        chain.from_iterable(
            df.index[
                df.index.isin([topic], level=0) & ~df.index.isin(gold[topic], level=1)
            ]
            for topic in df.index.get_level_values(0).unique()
        ),
        names=["topic", "keyphrases"],
    )

    df.loc[gold_idx, "gold"] = True
    df.loc[not_gold_idx, "gold"] = False
