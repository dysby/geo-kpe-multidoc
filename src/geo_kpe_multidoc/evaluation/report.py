import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_OUTPUT_PATH


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
