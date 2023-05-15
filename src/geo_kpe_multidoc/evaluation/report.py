import itertools
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_OUTPUT_PATH


def plot_score_distribuitions_with_gold(
    results: pd.DataFrame, title: str, xlim=None
) -> plt.Figure:
    plt.rcParams["figure.figsize"] = (6.4, 4.8)

    fig, ax = plt.subplots()
    ax = results[results["in_gold"] == False]["score"].plot.hist(density=True)
    ax = results[results["in_gold"] == True]["score"].plot.hist(density=True, alpha=0.5)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_title(title, fontsize=12)
    plt.legend(["non-gold", "gold"])
    return fig


def plot_non_versus_gold_density(
    gold_values: pd.Series, not_gold_values: pd.Series, title: str = None, density=False
):
    _, ax = plt.subplots()

    gold_values.hist(ax=ax, alpha=0.5, bins=20, density=density, color="orange")
    not_gold_values.hist(ax=ax, alpha=0.5, bins=20, density=density, color="blue")
    ax.set_title(title)
    plt.legend(["gold", "non-gold"])
    plt.show()


def print_latex(df, caption=None):
    print(
        df.style.format(precision=2, escape="latex")
        .format_index(escape="latex")
        .format_index(lambda s: s.replace("_", " ").capitalize(), axis=1)
        .to_latex(
            hrules=True,
            environment="table",
            position_float="centering",
            # clines = "all;data",
            caption=caption,
        )
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
        df[df.gold == False].loc[idx_filter],
        stat="probability",
        x=column,
        color="steelblue",
        bins=20,
        ax=ax,
    )

    sb.histplot(
        df[df.gold == True].loc[idx_filter],
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
        itertools.chain.from_iterable(
            df.index[
                df.index.isin([topic], level=0) & df.index.isin(gold[topic], level=1)
            ]
            for topic in df.index.get_level_values(0).unique()
        ),
        names=["topic", "keyphrases"],
    )

    not_gold_idx = pd.MultiIndex.from_tuples(
        itertools.chain.from_iterable(
            df.index[
                df.index.isin([topic], level=0) & ~df.index.isin(gold[topic], level=1)
            ]
            for topic in df.index.get_level_values(0).unique()
        ),
        names=["topic", "keyphrases"],
    )

    df.loc[gold_idx, "gold"] = True
    df.loc[not_gold_idx, "gold"] = False
