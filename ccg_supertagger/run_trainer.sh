#!/bin/bash

DATA="ccgbank"
python -u trainer.py \
 --batch_size 10 \
 --dropout_p 0.5 \
 --lexical_category2idx_path ../data/lexical_category2idx_cutoff_ccgbank.json \
 --model_dir ../plms/mt5-base \
 --checkpoints_dir ./checkpoints_$DATA \
 --mode train \
 --n_epochs 20 \
 --data_mode ccgbank \
 --load_mode reuse \
 2>&1 | tee -a supertagging_train_$DATA.log