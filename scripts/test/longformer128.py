from sentence_transformers import SentenceTransformer

import geo_kpe_multidoc
from geo_kpe_multidoc.datasets.datasets import load_data
from geo_kpe_multidoc.document import Document
from geo_kpe_multidoc.models.backend._longmodels import to_longformer_t_v4
from geo_kpe_multidoc.models.embedrank.embedrank_longformer_manual import (
    EmbedRankManual,
)
from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    filter_special_tokens,
)
from geo_kpe_multidoc.models.sentence_embedder import SentenceEmbedder

dataset = load_data("DUC2001", geo_kpe_multidoc.GEO_KPE_MULTIDOC_DATA_PATH)

print("doc:", dataset[1][0])
doc = dataset[1][1]
gold = dataset[1][2]


BACKEND_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TAGGER_NAME = "en_core_web_trf"

model, tokenizer = to_longformer_t_v4(
    SentenceTransformer(BACKEND_MODEL_NAME),
    max_pos=512,
    attention_window=128,
    copy_from_position=130,
)
# in RAM convertion to longformer needs this.
del model.embeddings.token_type_ids
kpe_model = EmbedRankManual(model, tokenizer, TAGGER_NAME)


top_n_and_scores, candidates = kpe_model.extract_kp_from_doc(
    Document(raw_text=doc, id=dataset[1][0], topic=dataset[1][0], dataset="DUC2001"),
    # kpe_model.pre_process(doc),
    top_n=-1,
    min_len=5,
    lemmer="en",
)

from typing import List

import torch


def find_occurrences(a: List[int], b: List[int]) -> List[List[int]]:
    occurrences = []
    # TODO: escape search in right padding indexes
    for i in range(len(b) - len(a) + 1):
        if b[i : i + len(a)] == a:
            occurrences.append(list(range(i, i + len(a))))
    return occurrences


def candidade_mean_embedding(
    candidate: List[int], doc_input_ids: List[int], doc_token_embeddings: torch.Tensor
):
    """doc_token_embeddings: Tensor.Size([N, 768])"""
    candidate_occurrences = find_occurrences(candidate, doc_input_ids)
    if len(candidate_occurrences) != 0:
        embds = torch.empty(size=(len(candidate_occurrences), 768))
        for i, occurrence in enumerate(candidate_occurrences):
            embds[i] = torch.mean(doc_token_embeddings[occurrence, :], dim=0)
        return torch.mean(embds, dim=0)
    else:
        return torch.mean(doc_token_embeddings[[], :], dim=0)


from geo_kpe_multidoc.models.pre_processing.pre_processing_utils import (
    filter_special_tokens,
)


def sim(doc, gold):
    doc_embedding_slong = slong.encode(doc)
    doc_embedding_sbert = sbert.encode(doc, output_value=None, convert_to_tensor=True)
    doc_sim = (
        cos_sim(
            doc_embedding_sbert["sentence_embedding"],
            doc_embedding_slong["sentence_embedding"],
        )
        .detach()
        .squeeze()
        .numpy()
        .item()
    )
    results = []
    for candidate in gold:
        cand_embd_slong = slong.encode(candidate)
        cand_embd_sbert = sbert.encode(
            candidate, output_value=None, convert_to_tensor=True
        )

        results.append(
            {
                "doc_len": len(
                    slong.tokenize(doc)["input_ids"].detach().squeeze().tolist()
                ),
                "doc_sim": doc_sim,
                "candidate": candidate,
                "sbert_sim": cos_sim(
                    doc_embedding_sbert["sentence_embedding"],
                    cand_embd_sbert["sentence_embedding"],
                )
                .detach()
                .squeeze()
                .numpy()
                .item(),
                "long_sim": cos_sim(
                    doc_embedding_slong["sentence_embedding"],
                    cand_embd_slong["sentence_embedding"],
                )
                .detach()
                .squeeze()
                .numpy()
                .item(),
                "mean_in_doc_sim": cos_sim(
                    doc_embedding_slong["sentence_embedding"],
                    candidade_mean_embedding(
                        filter_special_tokens(slong.tokenize(candidate)["input_ids"]),
                        doc_embedding_slong["input_ids"].detach().squeeze().tolist(),
                        doc_embedding_slong["token_embeddings"],
                    ),
                )
                .detach()
                .squeeze()
                .numpy()
                .item(),
            }
        )

    return pd.DataFrame.from_records(results)


def sim_docs(doc1, doc2, embedder):
    if isinstance(embedder, SentenceTransformer):
        doc1_embd = embedder.encode(doc1, output_value=None, convert_to_tensor=True)
        doc2_embd = embedder.encode(doc2, output_value=None, convert_to_tensor=True)
    else:
        doc1_embd = embedder.encode(doc1)
        doc2_embd = embedder.encode(doc2)
    return (
        cos_sim(doc1_embd["sentence_embedding"], doc2_embd["sentence_embedding"])
        .detach()
        .squeeze()
        .numpy()
        .item()
    )


"""
In [78]: doc1 = "Humans evolved as hunters and gatherers where we all worked for ourselves. It’s only at the beginning of agriculture we became mor
    ...: e hierarchical. The Industrial Revolution and factories made us extremely hierarchical because one individual couldn’t necessarily own or
    ...: build a factory, but now, thanks to the internet, we’re going back to an age where more and more people can work for themselves. I would r
    ...: ather be a failed entrepreneur than someone who never tried. Because even a failed entrepreneur has the skill set to make it on their own"
    ...:

In [79]: slong.tokenize(doc1)['input_ids'].shape
Out[79]: torch.Size([1, 119])

In [80]: doc2 = "He remembered how once he had been walking down a crowded street when a tremendous shout of hundreds of voices women's voices—had
    ...: burst from a side-street a little way ahead. It was a great formidable cry of anger and despair, a deep, loud 'Oh-o-o-o-oh!' that went hum
    ...: ming on like the reverberation of a bell. His heart had leapt. It's started! he had thought. A riot! The proles are breaking loose at last
    ...: ! When he had reached the spot it was to see a mob of two or three hundred women crowding round the stalls of a street market, with faces
    ...: as tragic as though they had been the doomed passengers on a sinking ship. But at this moment the general despair broke down into a multit
    ...: ude of individual quarrels. It appeared that one of the stalls had been selling tin saucepans. They were wretched, flimsy things, but cook
    ...: ing-pots of any kind were always difficult to get. Now the supply had unexpectedly given out. The successful women, bumped and jostled by
    ...: the rest, were trying to make off with their saucepans while dozens of others clamoured round the stall, accusing the stall-keeper of favo
    ...: uritism and of having more saucepans somewhere in reserve. There was a fresh outburst of yells. Two bloated women, one of them with her ha
    ...: ir coming down, had got hold of the same saucepan and were trying to tear it out of one another's hands. For a moment they were both tuggi
    ...: ng, and then the handle came off. Winston watched them disgustedly. And yet, just for a moment, what almost frightening power had sounded
    ...: in that cry from only a few hundred throats! Why was it that they could never shout like that about anything that mattered?"

In [81]: slong.tokenize(doc2)['input_ids'].shape
Out[81]: torch.Size([1, 425])

In [82]: sim_by_docs(doc1, doc2, slong)
Out[82]: array(0.19883409, dtype=float32)

In [83]: sim_by_docs(doc1, doc2, sbert)
Out[83]: array(0.08245981, dtype=float32)

In [84]: sim_by_docs(doc1+doc2, doc2, slong)
Out[84]: array(0.90485835, dtype=float32)
"""
