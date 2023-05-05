import os

import joblib
import pandas as pd
from matplotlib import pyplot as plt

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_CACHE_PATH, GEO_KPE_MULTIDOC_OUTPUT_PATH
from geo_kpe_multidoc.evaluation.report import add_gold_label, plot_dist
from geo_kpe_multidoc.geo.measures import arc_dist, exp_dist, inv_dist
from geo_kpe_multidoc.geo.utils import process_geo_associations_for_topics

################################################################################################################

experiment = "MKDUC01-MDKPERank-Longformer-4096-128-130"


docs_data = pd.read_parquet(
    os.path.join(
        GEO_KPE_MULTIDOC_CACHE_PATH, experiment, "MKDUC01-docs-data-20230501.parquet"
    )
)
topic_data = pd.read_parquet(
    os.path.join(
        GEO_KPE_MULTIDOC_CACHE_PATH, experiment, "MKDUC01-topic-data-20230501.parquet"
    )
)
topic_docs_coordinates = pd.read_parquet(
    os.path.join(
        GEO_KPE_MULTIDOC_CACHE_PATH, "MKDUC01-topic-doc-coordinates-20230504.parquet"
    )
)
topic_candidate_document_matrix = joblib.load(
    os.path.join(
        GEO_KPE_MULTIDOC_CACHE_PATH,
        experiment,
        "MKDUC01-topic-cand-doc-matrix-20230501.pkl",
    )
)

gold_24 = joblib.load(
    os.path.join(GEO_KPE_MULTIDOC_CACHE_PATH, experiment, "MKDUC01-gold-20230501.pkl")
)
add_gold_label(topic_data, gold_24)

################################################################################################################

d45 = pd.IndexSlice["d45", :]
d57 = pd.IndexSlice["d57", :]
df = topic_data.copy()

f_map = {
    "Inverse": inv_dist,
    "Exponential": exp_dist,
    "Arccot": arc_dist,
}

w_function = f_map["Inverse"]
w_function_param = 10


process_geo_associations_for_topics(
    df,
    docs_data,
    topic_candidate_document_matrix,
    doc_coordinate_data=topic_docs_coordinates,
    w_function=w_function,
    w_function_param=w_function_param,
    save_cache=False,
)


################################################################################################################

plt.rcParams["figure.figsize"] = (15, 10)
fig, ax = plt.subplots(3, 3)

title = f"{w_function.__name__}, a = {w_function_param}"

plot_dist(df, d45, title=title + ", d45", ax=ax[0, 0])
plot_dist(df, d57, title=title + ", d57", ax=ax[0, 1])
plot_dist(
    df,
    pd.IndexSlice[:, :],
    title=title + ", All topics",
    ax=ax[0, 2],
)

plot_dist(df, d45, column="geary_c", ax=ax[1, 0])
plot_dist(df, d57, column="geary_c", ax=ax[1, 1])
plot_dist(df, pd.IndexSlice[:, :], column="geary_c", ax=ax[1, 2])

plot_dist(df, d45, column="getis_g", ax=ax[2, 0])
plot_dist(df, d57, column="getis_g", ax=ax[2, 1])
plot_dist(df, pd.IndexSlice[:, :], column="getis_g", ax=ax[2, 2])


fig.savefig(
    os.path.join(
        GEO_KPE_MULTIDOC_OUTPUT_PATH,
        f"geo_measures_dist-{w_function.__name__}-a{w_function_param}.pdf",
    ),
    dpi=100,
)

plt.show()
