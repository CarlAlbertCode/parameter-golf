#!/usr/bin/env bash
set -euo pipefail

RUN_ID=baseline_4070_local \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=7 \
MODEL_DIM=384 \
NUM_HEADS=6 \
NUM_KV_HEADS=3 \
TRAIN_SEQ_LEN=512 \
TRAIN_BATCH_TOKENS=131072 \
ITERATIONS=200 \
WARMDOWN_ITERS=120 \
VAL_BATCH_SIZE=131072 \
VAL_LOSS_EVERY=0 \
MAX_WALLCLOCK_SECONDS=0 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
