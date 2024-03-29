import re
import string
from functools import partial
from os import path

import orjson
import streamlit as st
import torch
from annotated_text.util import get_annotated_html
from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH
from geo_kpe_multidoc.datasets.datasets import DATASETS, load_dataset
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.factory import kpe_model_factory
from sentence_transformers import SentenceTransformer

# from pipelines.keyphrase_extraction_pipeline import KeyphraseExtractionPipeline


# @st.cache(allow_output_mutation=True, show_spinner=False)
@st.cache_resource(show_spinner=False)
def load_pipeline(chosen_model, chosen_language):
    # if "keyphrase-extraction" in chosen_model:
    #     return KeyphraseExtractionPipeline(chosen_model)
    # elif "keyphrase-generation" in chosen_model:
    #     return KeyphraseGenerationPipeline(chosen_model, truncation=True)
    # BACKEND_MODEL_NAME = "longformer-paraphrase-multilingual-mpnet-base-v2"
    BACKEND_MODEL_NAME = "sentence-transformers/sentence-t5-base"
    match chosen_language:
        case "en":
            TAGGER_NAME = "en_core_web_trf"
        case "pt":
            TAGGER_NAME = "pt_core_news_lg"

    args = {
        "experiment_name": "debug",
        "extraction_variant": "base",
        "kp_min_len": 2,
        "kp_max_words": 7,
        "rank_model": "EmbedRank",
        "embed_model": "sentence-t5-base",
        "no_position_feature": False,
        "add_query_prefix": False,
        "candidate_mode": "mentions_no_context",
        "mmr": False,
        "mmr_diversity": None,
    }

    kpe_model = kpe_model_factory(
        BACKEND_MODEL_NAME, TAGGER_NAME, language=chosen_language, **args
    )
    return kpe_model.extract_kp_from_doc


def generate_run_id():
    return f"run_{re.sub('keyphrase-extraction-|keyphrase-generation-', '', st.session_state.chosen_model)}_{st.session_state.current_run_id}"


def extract_keyphrases():
    gold_keyphrases = None

    txt = st.session_state.input_text

    # if st.session_state.chosen_dataset == "--INPUT--":
    #     txt = st.session_state.input_text
    # else:
    #     ds = load_dataset(
    #         st.session_state.chosen_dataset,
    #         GEO_KPE_MULTIDOC_DATA_PATH,
    #     )
    #     d_idx = ds.ids.index(st.session_state.chosen_doc_id)
    #     _, txt, gold_keyphrases = ds[d_idx]

    # txt = remove_punctuation(txt)
    # txt = remove_whitespaces(txt)[1:]

    st.session_state.input_text = txt

    top_n_and_scores, _ = pipe(Document(txt, st.session_state.current_run_id))
    st.session_state.keyphrases = top_n_and_scores
    st.session_state.gold_keyphrases = gold_keyphrases
    st.session_state.history[generate_run_id()] = {
        "run_id": st.session_state.current_run_id,
        "model": st.session_state.chosen_model,
        "text": st.session_state.input_text,
        "keyphrases": st.session_state.keyphrases,
        "gold_keyphrases": st.session_state.gold_keyphrases,
    }
    st.session_state.current_run_id += 1


def get_annotated_text(text, keyphrases, color="#d294ff"):
    for i, (keyphrase, _) in enumerate(keyphrases):
        text = re.sub(
            rf"({keyphrase})([^A-Za-z0-9])",
            rf"$K:{i}\2",
            text,
            flags=re.I,
        )

    result = []
    for i, word in enumerate(text.split(" ")):
        if "$K" in word and re.search(
            "(\d+)$", word.translate(str.maketrans("", "", string.punctuation))
        ):
            result.append(
                (
                    re.sub(
                        r"\$K:\d+",
                        keyphrases[
                            int(
                                re.search(
                                    r"(\d+)$",
                                    word.translate(
                                        str.maketrans("", "", string.punctuation)
                                    ),
                                ).group(1)
                            )
                        ][0],
                        word,
                    ),
                    "KEY",
                    color,
                )
            )
        else:
            if i == len(st.session_state.input_text.split(" ")) - 1:
                result.append(f" {word}")
            elif i == 0:
                result.append(f"{word} ")
            else:
                result.append(f" {word} ")
    return result


def render_output(layout, runs, reverse=False):
    runs = list(runs.values())[::-1] if reverse else list(runs.values())
    for run in runs:
        layout.markdown(
            f"""
            <p style=\"margin-bottom: 0rem\"><strong>Run:</strong> {run.get('run_id')}</p>
            <p style=\"margin-bottom: 0rem\"><strong>Model:</strong> {run.get('model')}</p>
            """,
            unsafe_allow_html=True,
        )

        result = get_annotated_text(
            run.get("text"), list(run.get("keyphrases")[: st.session_state.top_n])
        )
        layout.markdown(
            f"""
            <p style="margin-bottom: 0.5rem"><strong>Text:</strong></p>
            {get_annotated_html(*result)}
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.gold_keyphrases:
            gold_keyphrases = [
                (keyphrase, "KEY", "#FFA500")
                for keyphrase in run.get("gold_keyphrases")
                # if keyphrase.lower() not in run.get("text").lower()
            ]
            for i in range(len(gold_keyphrases)):
                if i % 2 == 0:
                    gold_keyphrases.insert(i + 1, " ")

            layout.markdown(
                f"<p style=\"margin: 1rem 0 0 0\"><strong>Gold keyphrases:</strong> {get_annotated_html(*gold_keyphrases) if gold_keyphrases else 'None' }</p>",
                unsafe_allow_html=True,
            )
        layout.markdown("---")
        candidates_keyphrases = [
            (f"{keyphrase} ({score:.2f})", "KEY", "#d294ff")
            for keyphrase, score in run.get("keyphrases")[: st.session_state.top_n]
            # if keyphrase.lower() not in run.get("text").lower()
        ]
        for i in range(len(candidates_keyphrases)):
            if i % 2 == 0:
                candidates_keyphrases.insert(i + 1, " ")

        layout.markdown(
            f"<p style=\"margin: 1rem 0 0 0\"><strong>Candidate keyphrases:</strong> {get_annotated_html(*candidates_keyphrases) if candidates_keyphrases else 'None' }</p>",
            unsafe_allow_html=True,
        )
        layout.markdown("---")


dir_path = path.dirname(path.realpath(__file__))
if "config" not in st.session_state:
    with open(path.join(dir_path, "config.json"), encoding="utf8") as f:
        content = f.read()
        st.session_state.config = orjson.loads(content)
        st.session_state.history = {}
        st.session_state.keyphrases = []
        st.session_state.current_run_id = 1


st.set_page_config(
    page_icon="🔑",
    page_title="Keyphrase extraction",
    layout="centered",
)

with open(path.join(dir_path, "css/style.css"), encoding="utf8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.header("Keyphrase extraction")

description = """
Try it out yourself! 👇
"""

st.write(description)

with st.form("keyphrase-extraction-form"):
    # st.session_state.chosen_language = st.selectbox("Text Language:", ["en", "pt"])
    st.session_state.chosen_language = "en"

    st.session_state.chosen_model = "EmbedRank"
    # st.session_state.chosen_model = st.selectbox(
    #     "Choose model:", st.session_state.config.get("models")
    # )

    st.session_state.chosen_dataset = "--INPUT--"
    # st.session_state.chosen_dataset = st.selectbox(
    #     "Choose Dataset:", st.session_state.config.get("datasets")
    # )
    # st.session_state.chosen_doc_id = (
    #     st.text_input(
    #         "Select DOC ID from Dataset",
    #         "",
    #     )
    #     .replace("\n", " ")
    #     .strip()
    # )

    # st.markdown(
    #     f"Dataset / document selection has priority over input text\nIf you don't want to select dataset leave it empty."
    # )

    st.session_state.top_n = st.number_input(
        "How many keyphrases to extract?",
        value=int(st.session_state.config.get("top_n")),
        placeholder="Type N...",
        min_value=1,
        max_value=100,
        step=1,
    )

    st.session_state.input_text = (
        st.text_area(
            "✍ Input",
            st.session_state.config.get("example_text"),
            height=350,
            max_chars=2500,
        )
        .replace("\n", " ")
        .strip()
    )

    with st.spinner("Extracting keyphrases..."):
        _, button_container = st.columns([7, 1])
        pressed = button_container.form_submit_button("Extract")

if pressed and st.session_state.input_text != "":
    with st.spinner("Loading pipeline..."):
        pipe = load_pipeline(
            # f"{st.session_state.config.get('model_author')}/{st.session_state.chosen_model}"
            st.session_state.chosen_model,
            st.session_state.chosen_language,
        )
    with st.spinner("Extracting keyphrases"):
        extract_keyphrases()
elif st.session_state.input_text == "":
    st.error("The text input is empty 🙃 Please provide a text in the input field.")

if len(st.session_state.history.keys()) > 0:
    options = st.multiselect(
        "Specify the runs you want to see",
        st.session_state.history.keys(),
        format_func=lambda run_id: f"Run {run_id.split('_')[-1]}: {run_id.split('_')[1]}",
    )
    if options:
        render_output(
            st,
            {key: st.session_state.history[key] for key in options},
        )
    else:
        render_output(st, st.session_state.history, reverse=True)
