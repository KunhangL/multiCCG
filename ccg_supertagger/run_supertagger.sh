#!/bin/bash

MODE="predict"
TOPK=10
BETA=0.0005

python -u supertagger.py \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_treebanks.json \
 --model_dir ../plms/mt5-base \
 --checkpoint_path ./checkpoints_treebanks/fc_mt5-base_drop0.5_epoch_20.pt \
 --device cuda:0 \
 --batch_size 8 \
 --top_k ${TOPK} \
 --beta ${BETA} \
 --mode predict \
 --batch_predicted_path ./predicted_supertags.json \
 2>&1 | tee -a supertagger_${MODE}_${TOPK}_${BETA}.log