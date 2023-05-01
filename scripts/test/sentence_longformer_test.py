import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer
from transformers import BertForMaskedLM, LongformerSelfAttention


def alternative_mode_test():
    from roberta2longformer import convert_roberta_to_longformer
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    # convertion
    lmodel, lmodel_tokenizer = convert_roberta_to_longformer(
        roberta_model=sbert._modules["0"]._modules["auto_model"],
        roberta_tokenizer=sbert._modules["0"].tokenizer,
    )
    sbert.max_seq_length = 4096
    sbert._modules["0"]._modules["auto_model"] = lmodel
    sbert._modules["0"].tokenizer = lmodel_tokenizer

    original = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")

    from geo_kpe_multidoc.datasets.datasets import load_data

    docs = load_data("DUC2001", "/home/helder/doc/mecd/thesis/data")

    sentence_embedding_original = original.encode([docs[85][1]])
    sentence_embedding_transformed = sbert.encode([docs[85][1]])
    np.all(np.isclose(sentence_embedding_original, sentence_embedding_transformed))

    sentence_embedding_original = original.encode([docs[194][1]])
    sentence_embedding_transformed = sbert.encode([docs[194][1]])
    np.all(np.isclose(sentence_embedding_original, sentence_embedding_transformed))


class BertLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )


# change it into bert-version
class BertLongForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        # Replace self-attention with long-attention
        for i, layer in enumerate(self.bert.encoder.layer):
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)


model_original = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
# model_longformer = SentenceTransformer(
#     "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
# )
# # Load Custom Transformer Model and Build SentenceTransformer w/ MEAN pooling
# word_embedding_model = Longformer.from_pretrained(
#     "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
#     # "/home/helder/doc/mecd/thesis/models/longformer-test"
# )
# #
# pooling_model = Pooling(768)
#
model_custom = SentenceTransformer(
    "/home/helder/doc/mecd/thesis/models/longformer-paraphrase-multilingual-mpnet-base-v2"
)
model_xlmroberta_long = SentenceTransformer(
    "/home/helder/doc/mecd/thesis/models/longformer-xlm-roberta"
)


from geo_kpe_multidoc.datasets.datasets import load_data

docs = load_data("DUC2001", "/home/helder/doc/mecd/thesis/data/")

# Dataset doc token size
# sorted([(n, doc[0], model_longformer.tokenize([doc[1]])['input_ids'].size()) for n, doc in enumerate(docs)], key=itemgetter(2), reverse=True)

small = 85
large = 194


def get_embeddings(docs, idx, model_original, model_alternative):
    doc = [docs[idx][1]]
    sentence_embedding_original = model_original.encode(doc)
    sentence_embedding_alternative = model_alternative.encode(doc)
    return sentence_embedding_original, sentence_embedding_alternative


def equal_embeddings(vector1, vector2):
    return np.all(np.isclose(vector1, vector2))


print("Small doc")
print(
    "Original vs Longformer :",
    # equal_embeddings(*get_embeddings(docs, small, model_original, model_longformer)),
    equal_embeddings(*get_embeddings(docs, small, model_original, model_custom)),
)
print(
    "Original vs XLMroberta-long :",
    equal_embeddings(
        *get_embeddings(docs, small, model_original, model_xlmroberta_long)
    ),
)
print("Large_doc")
print(
    "Original vs Longformer :",
    equal_embeddings(*get_embeddings(docs, large, model_original, model_custom)),
    # equal_embeddings(*get_embeddings(docs, large, model_original, model_longformer)),
)
print(
    "Original vs XLMroberta-long :",
    equal_embeddings(
        *get_embeddings(docs, large, model_original, model_xlmroberta_long)
    ),
)
