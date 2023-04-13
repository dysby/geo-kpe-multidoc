#!/bin/sh

python scripts/run.py \
	--dataset_name DUC2001 \
	--experiment_name EmbedRank-DUC2001 \
	--rank_model EmbedRank \
	--stemming \
	--lemmatization \
	# --doc_limit 1 \
	# --embedrank_mmr
