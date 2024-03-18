#!/bin/bash

DATA="ccgbank"
MODEL_NAME="mt5base"
TOPK=10
BETA=0.0005

python -u parser.py \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_ccgbank.json \
 --instantiated_unary_rules_path ../data/instantiated_unary_rules_ccgbank.json \
 --instantiated_binary_rules_path ../data/instantiated_binary_rules_ccgbank.json \
 --unary_rules_n 20 \
 --supertagging_model_dir ../plms/mt5-base \
 --supertagging_model_checkpoint_path ../ccg_supertagger/checkpoints_ccgbank/fc_mt5-base_drop0.5_epoch_8.pt \
 --predicted_auto_files_dir ./evaluation \
 --beta ${BETA} \
 --top_k_supertags ${TOPK} \
 --batch_size 10 \
 --possible_roots "S[dcl]|NP|S[wq]|S[q]|S[qem]|S[b]\\NP" \
 --mode predict_batch \
 --batch_data_mode ccgbank_dev \
 2>&1 | tee -a AStarParsing_${DATA}_${MODEL_NAME}_${TOPK}_${BETA}.log