import json
from os import path

from elasticsearch.exceptions import ConnectionTimeout
from loguru import logger

# import plotly.express as px
from mordecai import Geoparser

from geo_kpe_multidoc import (
    GEO_KPE_MULTIDOC_DATA_PATH,
    GEO_KPE_MULTIDOC_MAPBOX_TOKEN,
    GEO_KPE_MULTIDOC_MORDECAI_ES_URL,
)


def process_MKDUC01():
    parser = Geoparser(es_hosts=GEO_KPE_MULTIDOC_MORDECAI_ES_URL, progress=False)

    with open(path.join(GEO_KPE_MULTIDOC_DATA_PATH, "MKDUC01/MKDUC01.json")) as s_json:
        docs = json.load(s_json)

        for doc_group in docs:
            try:
                logger.info(f"Geoparsing group {doc_group}")
                if doc_group in [
                    "d08",
                    "d14",
                    "d22",
                    "d28",
                    "d50",
                    "d53",
                ]:
                    continue

                res = {
                    doc_group: {
                        doc_id: str(parser.geoparse(text))
                        for doc_id, text in docs[doc_group]["documents"].items()
                    }
                }

                with open(
                    path.join(
                        GEO_KPE_MULTIDOC_DATA_PATH,
                        f"MKDUC01/MKDUC01-mordecai-{doc_group}.json",
                    ),
                    mode="w",
                ) as f:
                    json.dump(res, f, indent=4, separators=(",", ": "))
                del res

            except ConnectionTimeout as e:
                logger.error(e)
                with open(
                    path.join(
                        GEO_KPE_MULTIDOC_DATA_PATH,
                        "MKDUC01/MKDUC01-mordecai-errors.txt",
                    ),
                    mode="a",
                ) as f:
                    f.write("\n")
                    f.write(doc_group)

        # assert res_old == res

        # res = {
        #         topic: {
        #             doc_id: str(parser.geoparse(text))
        #             for doc_id, text in topic_docs["documents"].items()
        #         }
        #         for topic, topic_docs in docs.items()
        #     }
        # with open(
        #     path.join(GEO_KPE_MULTIDOC_DATA_PATH, "MKDUC01/MKDUC01-mordecai.json"),
        #     mode="w",
        # ) as f:
        #     json.dump(res, f, indent=4, separators=(",", ": "))

        # with open(
        #     path.join(GEO_KPE_MULTIDOC_DATA_PATH, "MKDUC01/MKDUC01-mordecai-old-3.json"),
        #     mode="w",
        # ) as f:
        #     json.dump(res_old, f, indent=4, separators=(",", ": "))


def build_map():
    with open(
        path.join(GEO_KPE_MULTIDOC_DATA_PATH, "MKDUC01/MKDUC01-mordecai.json"), "r"
    ) as s_json:
        source = json.load(s_json)
        res = {"geo_loc": [], "country_loc": [], "full_p": []}
        geo_locations = []
        country_locations = []

        for t in source:
            for d in source[t]:
                data_list = eval(source[t][d])
                for entry in data_list:
                    if "geo" in entry:
                        res["geo_loc"].append(
                            (float(entry["geo"]["lat"]), float(entry["geo"]["lon"]))
                        )

                        doc_f = {}
                        for e in entry:
                            if e in ["word", "country_predicted", "country_conf"]:
                                doc_f[e] = entry[e]
                        for e in entry["geo"]:
                            if e in ["country_code3", "geonameid", "place_name"]:
                                doc_f[e] = entry["geo"][e]
                            elif e in ["lat", "lon"]:
                                doc_f[e] = float(entry["geo"][e])
                        res["full_p"].append(doc_f)
                    else:
                        res["country_loc"].append(entry["country_predicted"])

        with open(
            path.join(GEO_KPE_MULTIDOC_DATA_PATH, "MKDUC01/MKDUC01-geo_locations.json"),
            "w",
        ) as d_json:
            json.dump(res, d_json, indent=4, separators=(",", ": "))

        px.set_mapbox_access_token(GEO_KPE_MULTIDOC_MAPBOX_TOKEN)
        fig = px.scatter_mapbox(
            res["full_p"],
            lat="lat",
            lon="lon",
            hover_name="word",
            hover_data=[
                "word",
                "country_predicted",
                "country_conf",
                "country_code3",
                "geonameid",
                "place_name",
            ],
            size_max=15,
            zoom=1,
            width=1000,
            height=800,
        )
        # fig.data[0].marker = dict(size = 5, color="red")
        fig.show()


# process_MKDUC01()
# build_map()
