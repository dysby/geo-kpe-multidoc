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
from geo_kpe_multidoc.models.maskrank.longmaskrank import LongformerMaskRank
from geo_kpe_multidoc.models.maskrank.maskrank_model import MaskRank
from geo_kpe_multidoc.models.mdkperank.mdkperank_model import MDKPERank
from geo_kpe_multidoc.models.mdkperank.mdpromptrank import MdPromptRank
from geo_kpe_multidoc.models.promptrank.promptrank import PromptRank


def generateLongformerRanker(
    ranker_class: BaseKPModel,
    backend_model_name,
    tagger_model,
    longformer_max_length,
    longformer_attention_window,
    **kwargs,
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
            tagger_model,
            device=device,
            name=backend_model_name.replace("/", "-"),
            **kwargs,
        )
        return kpe_model

    # Generate Longformer from Sentence Transformer

    new_max_pos = longformer_max_length
    attention_window = longformer_attention_window
    copy_from_position = kwargs.get("longformer_only_copy_to_max_position", None)

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
        tagger=tagger_model,
        device=device,
        name=model_name,
        candidate_embedding_strategy=kwargs.get("candidate_mode"),
        **kwargs,
    )
    return kpe_model


def generateBigBirdRanker(
    backend_model_name, tagger_model, longformer_max_length, **kwargs
):
    # Generate BigBird from Sentence Transformer
    new_max_pos = longformer_max_length

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
        tagger_model,
        device=device,
        name=model_name,
        candidate_embedding_strategy=kwargs.get("candidate_mode"),
        **kwargs,
    )
    return kpe_model


def generateNystromformerRanker(
    backend_model_name, tagger_model, longformer_max_length, **kwargs
):
    # Generate Nystromformer from Sentence Transformer
    new_max_pos = longformer_max_length

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
        tagger_model,
        device=device,
        name=model_name,
        candidate_embedding_strategy=kwargs.get("candidate_mode"),
        **kwargs,
    )
    return kpe_model


def kpe_model_factory(BACKEND_MODEL_NAME, TAGGER_NAME, **kwargs) -> BaseKPModel:
    rank_class = kwargs.get("rank_model", "EmbedRank")
    if rank_class == "EmbedRank":
        if "[longformer]" in BACKEND_MODEL_NAME:
            kpe_model = generateLongformerRanker(
                LongEmbedRank,
                BACKEND_MODEL_NAME.replace("[longformer]", ""),
                TAGGER_NAME,
                **kwargs,
            )
        elif "[bigbird]" in BACKEND_MODEL_NAME:
            kpe_model = generateBigBirdRanker(
                BACKEND_MODEL_NAME.replace("[bigbird]", ""),
                TAGGER_NAME,
                **kwargs,
            )
        elif "[nystromformer]" in BACKEND_MODEL_NAME:
            kpe_model = generateNystromformerRanker(
                BACKEND_MODEL_NAME.replace("[nystromformer]", ""),
                TAGGER_NAME,
                **kwargs,
            )
        else:
            kpe_model = EmbedRank(
                BACKEND_MODEL_NAME,
                TAGGER_NAME,
                candidate_embedding_strategy=kwargs["candidate_mode"],
                **kwargs,
            )
    elif rank_class == "MaskRank":
        if "[longformer]" in BACKEND_MODEL_NAME:
            kpe_model = generateLongformerRanker(
                LongformerMaskRank,
                BACKEND_MODEL_NAME.replace("[longformer]", ""),
                TAGGER_NAME,
                **kwargs,
            )
        else:
            kpe_model = MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME, **kwargs)
    elif rank_class == "PromptRank":
        kpe_model = PromptRank(BACKEND_MODEL_NAME, TAGGER_NAME, **kwargs)
    elif rank_class == "MDKPERank":
        if "[longformer]" in BACKEND_MODEL_NAME:
            base_name = BACKEND_MODEL_NAME.replace("[longformer]", "")
            kpe_model = MDKPERank(
                generateLongformerRanker(
                    LongEmbedRank, base_name, TAGGER_NAME, **kwargs
                ),
                rank_strategy=kwargs["md_strategy"],
                **kwargs,
            )
        else:
            ranker = EmbedRank(
                BACKEND_MODEL_NAME,
                TAGGER_NAME,
                candidate_embedding_strategy=kwargs["candidate_mode"],
                **kwargs,
            )
            kpe_model = MDKPERank(ranker, rank_strategy=kwargs["md_strategy"], **kwargs)
    elif rank_class == "MdPromptRank":
        base_model = PromptRank(BACKEND_MODEL_NAME, TAGGER_NAME, **kwargs)
        kpe_model = MdPromptRank(base_model)
    elif rank_class == "ExtractionEvaluator":
        kpe_model = ExtractionEvaluator(BACKEND_MODEL_NAME, TAGGER_NAME, **kwargs)
    elif rank_class == "FusionRank":
        kpe_model = FusionModel(
            [
                EmbedRank(
                    BACKEND_MODEL_NAME,
                    TAGGER_NAME,
                    candidate_embedding_strategy=kwargs["candidate_mode"],
                    **kwargs,
                ),
                MaskRank(BACKEND_MODEL_NAME, TAGGER_NAME, **kwargs),
            ],
            averaging_strategy=kwargs["ensemble_mode"],
            # models_weights=args.weights,
        )
    else:
        # raise ValueError("Model selection must be one of [EmbedRank, MaskRank].")
        logger.critical("Unknown KPE Model type.")
        raise ValueError("Unknown KPE Model type.")

    return kpe_model
