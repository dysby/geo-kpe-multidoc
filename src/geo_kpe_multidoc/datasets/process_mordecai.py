import json
from os import path

import plotly.express as px
from mordecai import Geoparser

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_RAWDATA_PATH, MAPBOX_KEY, GEO_KPE_MULTIDOC_MORDECAI_ES_URL


def process_MKDUC01():
	parser = Geoparser(es_hosts=GEO_KPE_MULTIDOC_MORDECAI_ES_URL)

	with open(path.join(GEO_KPE_MULTIDOC_RAWDATA_PATH, 'MKDUC-01/MKDUC01.json'), 'r') as s_json:
		docs = json.load(s_json)
		res = {}

		for doc_group in docs:
			res[doc_group] = {}
			for doc in docs[doc_group]["documents"]:
				res[doc_group][doc] = str(parser.geoparse(docs[doc_group]["documents"][doc]))

	with open(path.join(GEO_KPE_MULTIDOC_RAWDATA_PATH, "MKDUC-01/MKDUC01-mordecai.json"), 'w') as d_json:
		json.dump(res, d_json, indent=4, separators=(',', ': '))

def build_map():
	with open(path.join(GEO_KPE_MULTIDOC_RAWDATA_PATH,"MKDUC-01/MKDUC01-mordecai.json"), 'r') as s_json:
		source = json.load(s_json)
		res = {"geo_loc" : [], "country_loc" : [], "full_p" : []}
		geo_locations = []
		country_locations = []

		for t in source:
			for d in source[t]:
				data_list = eval(source[t][d])
				for entry in data_list:
					if "geo" in entry:
						res["geo_loc"].append((float(entry["geo"]["lat"]), float(entry["geo"]["lon"])))

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

		with open(path.join(GEO_KPE_MULTIDOC_RAWDATA_PATH,"MKDUC-01/MKDUC01-geo_locations.json"), 'w') as d_json:
			json.dump(res, d_json, indent=4, separators=(',', ': '))

		px.set_mapbox_access_token(MAPBOX_KEY)	
		fig = px.scatter_mapbox(res["full_p"], lat="lat", lon="lon", hover_name = "word", hover_data = ["word", "country_predicted", "country_conf", "country_code3", "geonameid", "place_name"], size_max=15, zoom=1, width=1000, height=800)
		#fig.data[0].marker = dict(size = 5, color="red")
		fig.show()

#process_MKDUC01()
# build_map()
