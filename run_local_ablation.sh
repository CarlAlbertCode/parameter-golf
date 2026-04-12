#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 {control|depthcond|stateaccum|parallel7|depthcond_parallel7|depthcond_loops2|depthcond_span245|depthcond_span356}"
  exit 1
fi

ABLATION="$1"

COMMON_ENV=(
  DATA_PATH=./data/datasets/fineweb10B_sp4096
  TOKENIZER_TYPE=sp
  TOKENIZER_PATH=data/tokenizers/fineweb_4096_bpe.model
  VOCAB_SIZE=4096
  NUM_LAYERS=10
  MODEL_DIM=384
  NUM_HEADS=6
  NUM_KV_HEADS=3
  MLP_MULT=4
  TRAIN_SEQ_LEN=512
  TRAIN_BATCH_TOKENS=131072
  VAL_BATCH_SIZE=32768
  VAL_MAX_TOKENS=2097152
  VAL_LOSS_EVERY=0
  TRAIN_LOG_EVERY=50
  ITERATIONS=400
  WARMUP_STEPS=20
  WARMDOWN_ITERS=400
  MAX_WALLCLOCK_SECONDS=0
  MUON_MOMENTUM=0.99
  QK_GAIN=5.0
  BIGRAM_VOCAB_SIZE=0
  BIGRAM_DIM=128
  EMA_DECAY=0.998
  USE_EMA_FOR_EVAL=0
  LEAKY_RELU_SLOPE=0.5
  EVAL_STRIDE=0
  ENABLE_KV_ADAPT=0
  SDCLIP_COEF=2.5
  ATTN_SCALE_INIT=0.5
  MLP_SCALE_INIT=0.5
  SKIP_GATE_INIT=4.0
  RECURRENT_LOOPS=1
  RECURRENT_LAYER_START=3
  RECURRENT_LAYER_END=5
  RECURRENT_DEPTH_SLOTS=4
  TTT_ENABLED=0
  DIAG_ENABLE=0
  SEED=1337
  PARALLEL_RESIDUALS=1
)

RUN_ID=""
EXTRA_ENV=()

case "$ABLATION" in
  control)
    RUN_ID=sp4096_recur_control_400
    EXTRA_ENV=(
      RECURRENT_DEPTH_CONDITIONING=0
      RECURRENT_STATE_ACCUM=0
      PARALLEL_RESIDUAL_START=9
    )
    ;;
  depthcond)
    RUN_ID=sp4096_recur_depthcond_400
    EXTRA_ENV=(
      RECURRENT_DEPTH_CONDITIONING=1
      RECURRENT_STATE_ACCUM=0
      PARALLEL_RESIDUAL_START=9
    )
    ;;
  stateaccum)
    RUN_ID=sp4096_recur_stateaccum_400
    EXTRA_ENV=(
      RECURRENT_DEPTH_CONDITIONING=0
      RECURRENT_STATE_ACCUM=1
      PARALLEL_RESIDUAL_START=9
    )
    ;;
  parallel7)
    RUN_ID=sp4096_recur_parallel7_400
    EXTRA_ENV=(
      RECURRENT_DEPTH_CONDITIONING=0
      RECURRENT_STATE_ACCUM=0
      PARALLEL_RESIDUAL_START=7
    )
    ;;
  depthcond_parallel7)
    RUN_ID=sp4096_recur_depthcond_parallel7_400
    EXTRA_ENV=(
      RECURRENT_DEPTH_CONDITIONING=1
      RECURRENT_STATE_ACCUM=0
      PARALLEL_RESIDUAL_START=7
    )
    ;;
  depthcond_loops2)
    RUN_ID=sp4096_recur_depthcond_loops2_400
    EXTRA_ENV=(
      RECURRENT_DEPTH_CONDITIONING=1
      RECURRENT_STATE_ACCUM=0
      RECURRENT_LOOPS=2
      PARALLEL_RESIDUAL_START=9
    )
    ;;
  depthcond_span245)
    RUN_ID=sp4096_recur_depthcond_span245_400
    EXTRA_ENV=(
      RECURRENT_DEPTH_CONDITIONING=1
      RECURRENT_STATE_ACCUM=0
      RECURRENT_LAYER_START=2
      RECURRENT_LAYER_END=5
      PARALLEL_RESIDUAL_START=9
    )
    ;;
  depthcond_span356)
    echo "depthcond_span356 crosses the encoder/decoder split in this trainer; using the nearest legal 3-layer encoder span 2:5 instead." >&2
    RUN_ID=sp4096_recur_depthcond_span245_from356_400
    EXTRA_ENV=(
      RECURRENT_DEPTH_CONDITIONING=1
      RECURRENT_STATE_ACCUM=0
      RECURRENT_LAYER_START=2
      RECURRENT_LAYER_END=5
      PARALLEL_RESIDUAL_START=9
    )
    ;;
  *)
    echo "Unknown ablation: $ABLATION"
    echo "Valid: control depthcond stateaccum parallel7 depthcond_parallel7 depthcond_loops2 depthcond_span245 depthcond_span356"
    exit 1
    ;;
esac

mkdir -p logs

echo "logs/${RUN_ID}.txt"

env \
  "${COMMON_ENV[@]}" \
  "${EXTRA_ENV[@]}" \
  RUN_ID="$RUN_ID" \
  PYTHONUNBUFFERED=1 \
  python3 train_gpt.py 2>&1 | tee "logs/${RUN_ID}.txt"
