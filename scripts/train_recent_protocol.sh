#!/usr/bin/env bash
set -euo pipefail

# Intent: reproduce one arm of the recent WikiText-103 protocol. This script is
# the container runtime. It launches one local Accelerate process per visible
# ClusterX GPU. All scientific choices remain explicit environment variables.
: "${GPT2_MODEL:?Set GPT2_MODEL}"
: "${WT103_ARTIFACT:?Set WT103_ARTIFACT}"
: "${DFM_RUNS:?Set DFM_RUNS}"
PYTHON_BIN=${PYTHON_BIN:-python}
ARCHITECTURE=${ARCHITECTURE:-transformer_only}
FUSION_TIMING=${FUSION_TIMING:-}
LOSS=${LOSS:-ce}
RUN_NAME=${RUN_NAME:-recent-${ARCHITECTURE}-${LOSS}}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:-$((7308 * EPOCHS))}
NUM_GPUS=${NUM_GPUS:-4}
GLOBAL_BATCH=${GLOBAL_BATCH:-16}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-1}
EFFECTIVE_WORKERS=$((NUM_GPUS * GRADIENT_ACCUMULATION))

if ((NUM_GPUS <= 0 || GRADIENT_ACCUMULATION <= 0 || GLOBAL_BATCH % EFFECTIVE_WORKERS != 0)); then
    echo "GLOBAL_BATCH=${GLOBAL_BATCH} must be divisible by NUM_GPUS*GRADIENT_ACCUMULATION=${EFFECTIVE_WORKERS}." >&2
    exit 2
fi
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-$((GLOBAL_BATCH / EFFECTIVE_WORKERS))}
VISIBLE_GPUS=$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')
if [[ "${VISIBLE_GPUS}" != "${NUM_GPUS}" ]]; then
    echo "Expected exactly ${NUM_GPUS} visible GPUs, got ${VISIBLE_GPUS}." >&2
    exit 2
fi

if [[ -z "${FUSION_TIMING}" ]]; then
    [[ "${ARCHITECTURE}" == traditional ]] && FUSION_TIMING=pre_attn || FUSION_TIMING=post_attn
fi
negative_args=()
if [[ "${LOSS}" == margin ]]; then
    negative_args=(--negative-prepared "${WT103_ARTIFACT}/random-negative-seed42")
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
"${PYTHON_BIN}" -m accelerate.commands.launch \
  --num_processes "${NUM_GPUS}" --mixed_precision bf16 --module dfm.train \
  --model "${GPT2_MODEL}" \
  --datastore "${WT103_ARTIFACT}/datastore" \
  --prepared "${WT103_ARTIFACT}/prepared/exclude-block" \
  --output "${DFM_RUNS}/${RUN_NAME}" \
  --architecture "${ARCHITECTURE}" \
  --fusion-timing "${FUSION_TIMING}" \
  --fusion-layers "${FUSION_LAYERS:-0,2,5,8,10,11}" \
  --loss "${LOSS}" \
  --margin "${MARGIN:-0.05}" --margin-weight "${MARGIN_WEIGHT:-0.1}" \
  --preserve-negative-rng \
  --reader-dim "${READER_DIM:-256}" --reader-layers "${READER_LAYERS:-2}" \
  --reader-heads "${READER_HEADS:-8}" --reader-sharing "${READER_SHARING:-independent}" \
  --reader-topology "${READER_TOPOLOGY:-causal}" --reader-write "${READER_WRITE:-residual}" \
  --reader-ff-multiplier "${READER_FF_MULTIPLIER:-4}" --reader-dropout "${READER_DROPOUT:-0.0}" \
  --learning-rate "${LEARNING_RATE:-0.001}" --weight-decay "${WEIGHT_DECAY:-0.0}" \
  --warmup-steps "${WARMUP_STEPS:-0}" --max-grad-norm "${MAX_GRAD_NORM:-1.0}" \
  --batch-size "${PER_DEVICE_BATCH}" --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
  --epochs "${EPOCHS}" --max-steps "${MAX_STEPS}" --seed "${SEED:-42}" \
  "${negative_args[@]}" "$@"
