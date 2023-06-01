#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate nlp

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export LOGURU_LEVEL=INFO

python scripts/run.py \
	--dataset_name DUC2001 \
	--experiment_name EmbedRank-DUC2001 \
	--embed_model "[longformer]paraphrase-multilingual-mpnet-base-v2" \
	--lemmatization \
