import json
from os import path

from loguru import logger
from mordecai3 import Geoparser

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH


def geo_parse_MKDUC01():
    parser = Geoparser()

    with open(
        path.join(GEO_KPE_MULTIDOC_DATA_PATH, "MKDUC01", "MKDUC01.json")
    ) as s_json:
        docs = json.load(s_json)

        for doc_group in docs:
            logger.debug(f"Geoparsing group {doc_group}")
            try:
                res = {
                    doc_group: {
                        doc_id: str(parser.geoparse(text))
                        for doc_id, text in docs[doc_group]["documents"].items()
                    }
                }

                with open(
                    path.join(
                        GEO_KPE_MULTIDOC_DATA_PATH,
                        "MKDUC01",
                        f"MKDUC01-mordecai3-{doc_group}.json",
                    ),
                    mode="w",
                ) as f:
                    logger.debug(f"Geoparsing save {doc_group}")
                    json.dump(res, f)

            except Exception as e:
                logger.error(e)
                with open(
                    path.join(
                        GEO_KPE_MULTIDOC_DATA_PATH,
                        "MKDUC01",
                        "MKDUC01-mordecai3-errors.txt",
                    ),
                    mode="a",
                ) as f:
                    f.write(doc_group)
                    f.write("\n")
