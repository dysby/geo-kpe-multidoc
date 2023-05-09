import time
from os import path

import simplemma
from loguru import logger

from geo_kpe_multidoc import GEO_KPE_MULTIDOC_DATA_PATH
from geo_kpe_multidoc.datasets import TextDataset
from geo_kpe_multidoc.datasets.datasets import DATASETS, KPEDataset
from geo_kpe_multidoc.evaluation.evaluation_tools import evaluate_kp_extraction
from geo_kpe_multidoc.evaluation.mkduc01_eval import MKDUC01_Eval
from geo_kpe_multidoc.models import EmbedRank, MaskRank
from geo_kpe_multidoc.models.mdkperank.mdkperank_model import MDKPERank
from geo_kpe_multidoc.models.pre_processing.pos_tagging import POS_tagger_spacy


def test_EmbedRank():
    backend_model = "longformer-paraphrase-multilingual-mpnet-base-v2"
    parser = "en_core_web_trf"
    # kpe_embed = EmbedRank(backend_model, parser)
    kpe_mask = MaskRank(backend_model, parser)

    txt = """INDUSTRIAL SOCIETY AND ITS FUTURE

    Introduction

    1. The Industrial Revolution and its consequences have been a disaster for the human race. They have greatly increased the life-expectancy of those of us who live in "advanced" countries, but they have destabilized society, have made life unfulfilling, have subjected human beings to indignities, have led to widespread psychological suffering (in the Third World to physical suffering as well) and have inflicted severe damage on the natural world. The continued development of technology will worsen the situation. It will certainly subject human beings to greater indignities and inflict greater damage on the natural world, it will probably lead to greater social disruption and psychological suffering, and it may lead to increased physical suffering even in "advanced" countries.

    2. The industrial-technological system may survive or it may break down. If it survives, it MAY eventually achieve a low level of physical and psychological suffering, but only after passing through a long and very painful period of adjustment and only at the cost of permanently reducing human beings and many other living organisms to engineered products and mere cogs in the social machine. Furthermore, if the system survives, the consequences will be inevitable: There is no way of reforming or modifying the system so as to prevent it from depriving people of dignity and autonomy.

    3. If the system breaks down the consequences will still be very painful. But the bigger the system grows the more disastrous the results of its breakdown will be, so if it is to break down it had best break down sooner rather than later.

    4. We therefore advocate a revolution against the industrial system. This revolution may or may not make use of violence; it may be sudden or it may be a relatively gradual process spanning a few decades. We can't predict any of that. But we do outline in a very general way the measures that those who hate the industrial system should take in order to prepare the way for a revolution against that form of society. This is not to be a POLITICAL revolution. Its object will be to overthrow not governments but the economic and technological basis of the present society.

    5. In this article we give attention to only some of the negative developments that have grown out of the industrial-technological system. Other such developments we mention only briefly or ignore altogether. This does not mean that we regard these other developments as unimportant. For practical reasons we have to confine our discussion to areas that have received insufficient public attention or in which we have something new to say. For example, since there are well-developed environmental and wilderness movements, we have written very little about environmental degradation or the destruction of wild nature, even though we consider these to be highly important.
    """

    # top_n, candidate_set = kpe_embed.extract_kp_from_doc(txt=txt, top_n=5, min_len=2)
    # print("Embed Rank")
    # print(top_n)
    # print("===========================================")
    # print(candidate_set)

    top_n, candidate_set = kpe_mask.extract_kp_from_doc(txt=txt, top_n=5, min_len=2)
    print("Mask Rank")
    print(top_n)
    print("===========================================")
    print(candidate_set)


def test_MDKPERank():
    #########################################
    # EXAMPLE code from MkDUC01 #
    #########################################
    BACKEND_MODEL_NAME = "longformer-paraphrase-multilingual-mpnet-base-v2"
    TAGGER_NAME = "en_core_web_trf"
    DATASET_LIST = ["MKDUC01"]
    # TOPN = 3

    start_time = time.time()
    logger.info(f"Initializing - {time.time() - start_time}")
    logger.info(f"Reading Datasets: {DATASET_LIST}")
    datasets = TextDataset.build_datasets(DATASET_LIST)

    kpe_model = MDKPERank(BACKEND_MODEL_NAME, TAGGER_NAME)

    mkde_evaluator = MKDUC01_Eval(
        path.join(GEO_KPE_MULTIDOC_DATA_PATH, "MKDUC01", "MKDUC01.json")
    )
    pred_kps_per_topic = {}
    for corpus in datasets:
        logger.info(f"KPE for {corpus.name} with {len(corpus)} topics")
        for topic, docs, _gold_kpe in corpus:
            print(f"Getting KPs for {topic} - {time.time() - start_time}")
            kpe_scores = kpe_model.extract_kp_from_topic(
                topic=docs,
                top_n=20,
                min_len=5,
                stemming=False,
                lemmatize=False,
            )
            # for topic_kpe_output in kpe_output:
            #    print(f"Top {TOPN} Keyphrases for topic")
            #    for keyphrase, score in topic_kpe_scores[:TOPN]:
            #        print(keyphrase, score)
            #    print("==========================")
            kps = [kp for (kp, _kp_score) in kpe_scores]
            pred_kps_per_topic[topic] = kps
    # Evaluate the predicted KPs over all topics:
    logger.info(f"Evaluating predicted KPs - {time.time() - start_time}")
    final_scores = mkde_evaluator.evaluate(pred_kps_per_topic, clusterLevel=True)
    logger.info(final_scores)


def test_KPERank():
    BACKEND_MODEL_NAME = "longformer-paraphrase-multilingual-mpnet-base-v2"
    TAGGER_NAME = "en_core_web_trf"
    # TAGGER_NAME = "pt_core_news_lg"
    DATASET_LIST = ["fao30"]
    # LANGUAGE = "pt"
    # stemmer = PorterStemmer() if stemming else None

    kpe_model = EmbedRank(BACKEND_MODEL_NAME, TAGGER_NAME)
    # kpe_model = MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME)

    model_results = {}
    true_labels = {}
    for ds_name in DATASET_LIST:
        data = KPEDataset(
            ds_name, DATASETS[ds_name]["zip_file"], GEO_KPE_MULTIDOC_DATA_PATH
        )

        # update SpaCy POS tagging for dataset language
        kpe_model.tagger = POS_tagger_spacy(DATASETS[ds_name]["tagger"])

        model_results[ds_name] = []
        true_labels[ds_name] = []
        for _doc_id, doc, gold_kp in data:
            top_n_and_scores, candidates = kpe_model.extract_kp_from_doc(
                kpe_model.pre_process(doc),
                top_n=20,
                min_len=2,
                lemmer=DATASETS[ds_name]["language"],
            )
            model_results[ds_name].append(
                (
                    top_n_and_scores,
                    candidates,
                )
            )
            true_labels[ds_name].append(gold_kp)

            # model_results["dataset_name"][(doc1_top_n, doc1_candidates), (doc2...)]
    evaluate_kp_extraction(model_results, true_labels)


if __name__ == "__main__":
    # print(os.environ["PYTHONPATH"])
    # import debugpy
    # debugpy.listen(5678)
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()
    # print('break on this line')
    test_KPERank()
