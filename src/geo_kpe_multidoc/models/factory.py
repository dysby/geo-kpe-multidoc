import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from geo_kpe_multidoc.models.backend.roberta2longformer.roberta2bigbird import (
    convert_roberta_to_bigbird,
)
from geo_kpe_multidoc.models.backend.roberta2longformer.roberta2longformer import (
    convert_roberta_to_longformer,
)
from geo_kpe_multidoc.models.backend.roberta2longformer.roberta2nyströmformer import (
    convert_roberta_to_nystromformer,
)
from geo_kpe_multidoc.models.base_KP_model import BaseKPModel, ExtractionEvaluator
from geo_kpe_multidoc.models.embedrank.embedrank_model import EmbedRank
from geo_kpe_multidoc.models.embedrank.longembedrank import LongEmbedRank
from geo_kpe_multidoc.models.fusion_model import FusionModel
from geo_kpe_multidoc.models.maskrank.maskrank_manual import LongformerMaskRank
from geo_kpe_multidoc.models.maskrank.maskrank_model import MaskRank
from geo_kpe_multidoc.models.mdkperank.mdkperank_model import MDKPERank
from geo_kpe_multidoc.models.promptrank.promptrank import PromptRank


def generateLongformerRanker(
    ranker_class: BaseKPModel, backend_model_name, tagger_name, args
):
    # Load AllenAi Longformer
    if backend_model_name == "allenai/longformer-base-4096":
        from transformers import AutoTokenizer, LongformerModel

        model = LongformerModel.from_pretrained(backend_model_name)
        tokenizer = AutoTokenizer.from_pretrained(backend_model_name, use_fast=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kpe_model = ranker_class(
            model,
            tokenizer,
            tagger_name,
            device=device,
            name=backend_model_name.replace("/", "-"),
        )
        return kpe_model

    # Generate Longformer from Sentence Transformer
    new_max_pos = args.longformer_max_length
    attention_window = args.longformer_attention_window
    copy_from_position = (
        args.longformer_only_copy_to_max_position
        if args.longformer_only_copy_to_max_position
        else None
    )

    base_name = backend_model_name.split("/")[-1]

    model_name = f"longformer_{base_name[:15]}_{new_max_pos}_attw{attention_window}"
    if copy_from_position:
        model_name += f"_cpmaxpos{copy_from_position}"

    # model, tokenizer = to_longformer_t_v4(
    #     SentenceTransformer(backend_model_name),
    #     max_pos=new_max_pos,
    #     attention_window=attention_window,
    #     copy_from_position=copy_from_position,
    # )
    # # in RAM convertion to longformer needs this.
    # if hasattr(model.embeddings, "token_type_ids"):
    #     del model.embeddings.token_type_ids

    sbert = SentenceTransformer(backend_model_name)

    longformer_model, longformer_tokenizer = convert_roberta_to_longformer(
        sbert._modules["0"].auto_model,
        sbert.tokenizer,
        new_max_pos,
        attention_window,
        copy_from_position,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kpe_model = ranker_class(
        longformer_model,
        longformer_tokenizer,
        tagger=tagger_name,
        device=device,
        name=model_name,
        candidate_embedding_strategy=args.candidate_mode,
    )
    return kpe_model


def generateBigBirdRanker(backend_model_name, tagger_name, args):
    # Generate BigBird from Sentence Transformer
    new_max_pos = args.longformer_max_length

    base_name = backend_model_name.split("/")[-1]

    model_name = f"bigbird_{base_name}_{new_max_pos}"

    sbert = SentenceTransformer(backend_model_name)

    bigbird_model, bigbird_tokenizer = convert_roberta_to_bigbird(
        sbert._modules["0"].auto_model, sbert.tokenizer, new_max_pos
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kpe_model = LongEmbedRank(
        bigbird_model,
        bigbird_tokenizer,
        tagger_name,
        device=device,
        name=model_name,
        candidate_embedding_strategy=args.candidate_mode,
    )
    return kpe_model


def generateNystromformerRanker(backend_model_name, tagger_name, args):
    # Generate Nystromformer from Sentence Transformer
    new_max_pos = args.longformer_max_length

    base_name = backend_model_name.split("/")[-1]

    model_name = f"nystromformer_{base_name}_{new_max_pos}"

    sbert = SentenceTransformer(backend_model_name)

    bigbird_model, bigbird_tokenizer = convert_roberta_to_nystromformer(
        sbert._modules["0"].auto_model, sbert.tokenizer, new_max_pos
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kpe_model = LongEmbedRank(
        bigbird_model,
        bigbird_tokenizer,
        tagger_name,
        device=device,
        name=model_name,
        candidate_embedding_strategy=args.candidate_mode,
    )
    return kpe_model


def kpe_model_factory(args, BACKEND_MODEL_NAME, TAGGER_NAME) -> BaseKPModel:
    if args.rank_model == "EmbedRank":
        if "[longformer]" in BACKEND_MODEL_NAME:
            kpe_model = generateLongformerRanker(
                LongEmbedRank,
                BACKEND_MODEL_NAME.replace("[longformer]", ""),
                TAGGER_NAME,
                args,
            )
        elif "[bigbird]" in BACKEND_MODEL_NAME:
            kpe_model = generateBigBirdRanker(
                BACKEND_MODEL_NAME.replace("[bigbird]", ""), TAGGER_NAME, args
            )
        elif "[nystromformer]" in BACKEND_MODEL_NAME:
            kpe_model = generateNystromformerRanker(
                BACKEND_MODEL_NAME.replace("[nystromformer]", ""), TAGGER_NAME, args
            )
        else:
            kpe_model = EmbedRank(
                BACKEND_MODEL_NAME,
                TAGGER_NAME,
                candidate_embedding_strategy=args.candidate_mode,
            )
    elif args.rank_model == "MaskRank":
        if "[longformer]" in BACKEND_MODEL_NAME:
            kpe_model = generateLongformerRanker(
                LongformerMaskRank,
                BACKEND_MODEL_NAME.replace("[longformer]", ""),
                TAGGER_NAME,
                args,
            )
        else:
            kpe_model = MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME)
    elif args.rank_model == "PromptRank":
        kpe_model = PromptRank(BACKEND_MODEL_NAME, TAGGER_NAME, **vars(args))
    elif args.rank_model == "MDKPERank":
        if "[longformer]" in BACKEND_MODEL_NAME:
            base_name = BACKEND_MODEL_NAME.replace("[longformer]", "")
            kpe_model = MDKPERank(
                generateLongformerRanker(LongEmbedRank, base_name, TAGGER_NAME, args),
                rank_strategy=args.md_strategy,
            )
        else:
            ranker = EmbedRank(
                BACKEND_MODEL_NAME,
                TAGGER_NAME,
                candidate_embedding_strategy=args.candidate_mode,
            )
            kpe_model = MDKPERank(ranker, rank_strategy=args.md_strategy)
    elif args.rank_model == "ExtractionEvaluator":
        kpe_model = ExtractionEvaluator(BACKEND_MODEL_NAME, TAGGER_NAME)
    elif args.rank_model == "FusionRank":
        kpe_model = FusionModel(
            [
                EmbedRank(
                    BACKEND_MODEL_NAME,
                    TAGGER_NAME,
                    candidate_embedding_strategy=args.candidate_mode,
                ),
                MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME),
            ],
            averaging_strategy=args.ensemble_mode,
            # models_weights=args.weights,
        )
    else:
        # raise ValueError("Model selection must be one of [EmbedRank, MaskRank].")
        logger.critical("Unknown KPE Model type.")
        raise ValueError("Unknown KPE Model type.")

    return kpe_model
