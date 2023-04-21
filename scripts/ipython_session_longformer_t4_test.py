# coding: utf-8
from geo_kpe_multidoc.models.sentence_embedder import SentenceEmbedder
from geo_kpe_multidoc.models.backend._longmodels import to_longformer_t_v4
from sentence_transformers import SentenceTransformer

slong = SentenceEmbedder(*to_longformer_t_v4(SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")))
sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
outputs_1 = slong.encode("helder")
outputs_2 = slong.encode(["helder", "margarida"])
import torch
torch.allclose(outputs_1["sentence_embedding"], outputs_2["sentence_embedding"][0, :])

from sentence_transformers.util import cos_sim
cos_sim(outputs_1['sentence_embedding'], outputs_2['sentence_embedding'][0, :])

outputs_1['sentence_embedding'].sum() == outputs_2['sentence_embedding'][0, :].sum()
outputs_2['sentence_embedding'][0, :].mean() == outputs_1['sentence_embedding'].mean()


sembeds_1 = sbert.encode("helder", output_value=None, convert_to_tensor='pt')
sembeds_2 = sbert.encode(["helder", "margarida"], output_value=None, convert_to_tensor='pt')
torch.allclose(sembeds_1['sentence_embedding'], sembeds_2['sentence_embedding'][0, :])
sembeds_2
sembeds_2['sentence_embedding']
sembeds_2.keys()
sembeds_2[0]
torch.allclose(sembeds_1['sentence_embedding'], sembeds_2[0]['sentence_embedding'])
sembeds_1['input_ids']
sembeds_2[0]['input_ids']
sembeds_2[1]['input_ids']
sembeds_2[0]['input_ids']
cos_sim(sembeds_1['sentence_embedding'], outputs_1['sentence_embedding'])
cos_sim(sembeds_2[1]['sentence_embedding'], outputs_2['sentence_embedding'][1, :])
sentence = "pretrained_init_configuration (Dict[str, Dict[str, Any]]) — A dictionary with, as keys, the short-cut-names of the pretrained models, and as associated values, a dictionary of specific arguments to pass to the __init__ method of the tokenizer class for this pretrained model when loading the tokenizer with the from_pretrained() method."

def compare(sentence, sbert, slong):
    sbert_output = sbert.encode(sentence, output_value=None, convert_to_tensor='pt')
    slong_output = slong.encode(sentence)
    return cos_sim(slong_output['sentence_embedding'], sbert_output['sentence_embedding'])
    
compare(sentence, sbert, slong)


sembeds_2 = sbert.encode(["helder", "pretrained_init_configuration"], output_value=None, convert_to_tensor='pt')

torch.allclose(sembeds_1['sentence_embedding'], sembeds_2[0]['sentence_embedding'])


cos_sim(sembeds_1['sentence_embedding'], sembeds_2[0]['sentence_embedding'])
cos_sim(sembeds_1['sentence_embedding'], sembeds_2)

sentence1 = "helder"
sentence2 = ["helder","Once you have sentence embeddings computed, you usually want to compare them to each other. Here, I show you how you can compute the cosine similarity between embeddings, for example, to measure the semantic similarity of two texts."]

sentence1_embeddings_slong = slong.encode(sentence1)['sentence_embedding']
sentence2_embeddings_slong = slong.encode(sentence2)['sentence_embedding']
sentence1_embeddings_sbert = sbert.encode(sentence1, convert_to_tensor=True)
sentence2_embeddings_sbert = sbert.encode(sentence2, convert_to_tensor=True)

cos_sim(sentence1_embeddings_sbert, sentence1_embeddings_slong)
cos_sim(sentence2_embeddings_sbert, sentence2_embeddings_slong)
cos_sim(sentence1_embeddings_sbert, sentence2_embeddings_slong)
cos_sim(sentence2_embeddings_sbert, sentence1_embeddings_slong)
cos_sim(sentence1_embeddings_long, sentence2_embeddings_slong)
cos_sim(sentence1_embeddings_slong, sentence2_embeddings_slong)
cos_sim(sentence1_embeddings_sbert, sentence2_embeddings_sbert)

torch.allclose(sentence1_embeddings_sbert, sentence2_embeddings_sbert[0, :])

sentence3 = "Once you have sentence embeddings computed, you usually want to compare them to each other. Here, I show you how you can compute the cosine similarity between embeddings, for example, to measure the semantic similarity of two texts. " * 3
sentence3_embeddings_sbert = sbert.encode(sentence3, convert_to_tensor=True)
sentence3_embeddings_slong = slong.encode(sentence3)['sentence_embedding']

t = sbert.encode(sentence1, convert_to_tensor=True)
torch.allclose(t, sentence1_embeddings_sbert)

t = slong.encode(sentence1)['sentence_embedding']
torch.allclose(t, sentence1_embeddings_slong)

cos_sim(sentence3_embeddings_sbert, sentence3_embeddings_slong)

import geo_kpe_multidoc
from geo_kpe_multidoc.datasets.datasets import load_data

dataset = load_data("DUC2001", geo_kpe_multidoc.GEO_KPE_MULTIDOC_DATA_PATH)

print("doc:", dataset[1][0])
doc = dataset[1][1]
gold = dataset[1][2]

doc_embedding_slong = slong.encode(doc)

doc_embedding_sbert_full = sbert.encode(doc, output_value=None, convert_to_tensor=True)
doc_embedding_sbert = sbert.encode(doc, convert_to_tensor=True)

gold_embedding_sbert = sbert.encode(gold, convert_to_tensor=True)
gold_embedding_slong_full = slong.encode(gold)


cos_sim(gold_embedding_sbert, gold_embedding_slong_full['sentence_embedding'])

cos_sim(doc_embedding_sbert, gold_embedding_slong_full['sentence_embedding'])

cos_sim(doc_embedding_slong, gold_embedding_slong_full['sentence_embedding'])

cos_sim(doc_embedding_slong['sentence_embedding'], gold_embedding_slong_full['sentence_embedding'])

cos_sim(doc_embedding_slong, gold_embedding_slong_full['sentence_embedding'])

cos_sim(doc_embedding_slong, gold_embedding_slong_full['sentence_embedding'])

cos_sim(doc_embedding_sbert, gold_embedding_slong_full['sentence_embedding'])

cos_sim(doc_embedding_slong['sentence_embedding'], gold_embedding_slong_full['sentence_embedding'])

from typing import List
from itertools import chain
    
def find_occurrences(a: List[int], b: List[int]) -> List[List[int]]:
    occurrences = []
    # TODO: escape search in right padding indexes
    for i in range(len(b) - len(a) + 1):
        if b[i:i+len(a)] == a:
            occurrences.append(list(range(i, i + len(a))))
    return occurrences
    
def cand_mean_embedding(candidate: List[int], doc_input_ids: List[int], doc_token_embeddings ):
    candidate_occurrences = find_occurrences(candidate, doc_input_ids)
    candidate_occurrences = list(chain(*candidate_occurrences))
    return torch.mean(doc_token_embeddings[:, candidate_occurrences, :], dim=1)


candidate_0 = [85325, 66961,     7]
candidate_1 = [11704, 149039,      7]
candidate_2 = [149039,      7,  57495]
candidate_3 = [60680,      6, 199417,  17690]
candidate_4 = [18276,     6, 85729,  1830,   674]
candidate_5 = [15889, 10336, 54529]

candidate_embedding_0 = cand_mean_embedding(candidate_0, doc_embedding_slong['input_ids'].squeeze().tolist(), doc_embedding_slong['token_embeddings'])
candidate_embedding_1 = cand_mean_embedding(candidate_1, doc_embedding_slong['input_ids'].squeeze().tolist(), doc_embedding_slong['token_embeddings'])
candidate_embedding_2 = cand_mean_embedding(candidate_2, doc_embedding_slong['input_ids'].squeeze().tolist(), doc_embedding_slong['token_embeddings'])
candidate_embedding_3 = cand_mean_embedding(candidate_3, doc_embedding_slong['input_ids'].squeeze().tolist(), doc_embedding_slong['token_embeddings'])
candidate_embedding_4 = cand_mean_embedding(candidate_4, doc_embedding_slong['input_ids'].squeeze().tolist(), doc_embedding_slong['token_embeddings'])
candidate_embedding_5 = cand_mean_embedding(candidate_5, doc_embedding_slong['input_ids'].squeeze().tolist(), doc_embedding_slong['token_embeddings'])

for embd_candidate in [candidate_embedding_0, candidate_embedding_1, candidate_embedding_2, candidate_embedding_3, candidate_embedding_4, candidate_embedding_5]:
    print(cos_sim(embd_candidate, doc_embedding_slong['sentence_embedding']))
    
gold_space = [ " " + s for s in gold ]
gold_space_embedding_slong_full = slong.encode(gold_space)

gold_space_embedding_slong_full['input_ids'][:, :8]
gold_embedding_slong_full['input_ids'][:, :8]

for g in gold:
    print(g in doc)
    
