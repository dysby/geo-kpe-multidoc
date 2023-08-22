import argparse
import textwrap
from datetime import datetime

import geo_kpe_multidoc.geo.measures
import pandas as pd
from geo_kpe_multidoc.datasets.datasets import DATASETS, load_dataset
from geo_kpe_multidoc.evaluation.evaluation_tools import (
    evaluate_kp_extraction,
    evaluate_kp_extraction_base,
    model_scores_to_dataframe,
    postprocess_model_outputs,
)
from geo_kpe_multidoc.evaluation.report import (
    output_one_top_cands,
    plot_score_distribuitions_with_gold,
)
from geo_kpe_multidoc.models.mdgeorank.mdgeorank import (
    GeospacialAssociationIndex,
    MdGeoRank,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    remove_new_lines_and_tabs,
    remove_whitespaces,
    select_stemmer,
)
from tabulate import tabulate

import wandb


def write_resume_txt(performance_metrics, args):
    with open("runs.resume.txt", "a") as f:
        stamp = datetime.now().strftime(r"%Y%m%d-%H%M")
        print(f"Date: {stamp}", file=f)
        print(f"Args: {args}", file=f)
        print(
            tabulate(
                performance_metrics[
                    [
                        # "Precision",
                        "Recall",
                        # "F1",
                        # "MAP",
                        "nDCG",
                        "F1_5",
                        "F1_10",
                        "F1_15",
                    ]
                ],
                headers="keys",
                floatfmt=".2%",
            ),
            file=f,
        )


# fmt: off
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Key-phrase extraction
            --------------------------------
                Multi language
                Multi document
                Geospatial association measures
            """
        ),
    )
    parser.add_argument( "--experiment_name", default="run", type=str, help="Name/path to load experiment results",)
    parser.add_argument( "--dataset_name", type=str, default="MKDUC01", help="The dataset name MKDUC01",)
    parser.add_argument( "--geo.weight_function", type=str, required=True, help="Weitgh matrix distance function",)
    parser.add_argument( "--geo.weight_function_param", type=str, required=True, help="Weigth matrix distance function parameter",)
    parser.add_argument( "--geo.geo_association_index", type=str, required=True, help="Geospacial association index used in rerank",)
    parser.add_argument( "--geo.alpha", type=str, required=True, help="Geospacial association index power factor used in rerank",)
    parser.add_argument( "--preprocessing", action="store_true", help="Preprocess text documents by removing pontuation",)
    return parser.parse_args()
# fmt: on


def main():
    args = parse_args()
    args.preprocessing = (
        [remove_new_lines_and_tabs, remove_whitespaces] if args.preprocessing else []
    )
    stemmer = select_stemmer(DATASETS[args.dataset_name].get("language"))
    lemmer = DATASETS[args.dataset_name].get("language")

    dataset = load_dataset(args.dataset_name)

    true_labels = {}
    # TODO: topic id must be alfabeticaly sorted when loading from dataset,
    # because at evaluation time model results dictionary have the topic results list
    # sorted by topic_id
    for topic, docs, gold_kp in dataset:
        true_labels.setdefault("dataset", []).append(gold_kp)

    geo_kpe_model = MdGeoRank(
        args.experiment_name,
        stemmer,
        weight_function=getattr(
            geo_kpe_multidoc.geo.measures, args.geo.weight_function
        ),
        weight_function_param=args.geo.weight_function_param,
        geo_association_index=args.geo.geo_association_index,
    )

    # Re-Rank with Geo Associations
    md_model_output_and_geo_associations = geo_kpe_model.geospacial_association()
    model_output_ranking = geo_kpe_model.rank(
        md_model_output_and_geo_associations,
    )

    # True labels are preprocessed while loading Dataset.
    model_results = postprocess_model_outputs(
        model_output_ranking, stemmer, lemmer, args.preprocessing
    )

    with wandb.init(
        project="geo-kpe-multidoc",
        name=f"geo-{args.experiment_name}",
        config=vars(args),
    ):
        # Print and Save Results
        kpe_for_doc = output_one_top_cands(dataset.ids, model_results, true_labels)

        dataset_kpe = model_scores_to_dataframe(model_results, true_labels)

        xlim = (dataset_kpe["score"].min(), dataset_kpe["score"].max())
        fig = plot_score_distribuitions_with_gold(
            results=dataset_kpe,
            title=f"geo - {args.experiment_name.replace('-', ' ')}",
            xlim=xlim,
        )

        # mlflow.log_figure(fig, artifact_file="score_distribution.png")
        wandb.log({"score_distribution": wandb.Image(fig)})

        performance_metrics = evaluate_kp_extraction_base(model_results, true_labels)
        performance_metrics = pd.concat(
            [performance_metrics, evaluate_kp_extraction(model_results, true_labels)]
        )

        # mlflow.log_text(kpe_for_doc, artifact_file="first-doc-extraction-sample.txt")
        wandb.log({"first-doc-extraction-sample": kpe_for_doc})

        metric_names = [
            "_base_" + value for value in performance_metrics.iloc[0].index.values
        ]

        metrics = performance_metrics.iloc[0]
        metrics.index = metric_names
        all_metrics = metrics.to_dict()

        # mlflow.log_metrics(metrics.to_dict())
        metrics = performance_metrics.iloc[1]
        # mlflow.log_metrics(metrics.to_dict())

        all_metrics.update(metrics.to_dict())
        wandb.log(all_metrics)

    print(
        tabulate(
            performance_metrics[
                # ["Precision", "Recall", "F1", "MAP", "nDCG", "F1_5", "F1_10", "F1_15"]
                ["Recall", "nDCG", "F1_5", "F1_10", "F1_15"]
            ],
            headers="keys",
            floatfmt=".2%",
        )
    )

    write_resume_txt(performance_metrics, args)

    # save(dataset_kpe, performance_metrics, fig, args)


if __name__ == "__main__":
    main()
