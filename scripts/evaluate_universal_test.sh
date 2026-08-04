#!/usr/bin/env bash
set -euo pipefail

# Intent: evaluate an audited final checkpoint on the same 280 cross-article WikiText test
# rows under retrieved, deterministic disjoint random, and true memory bypass.
: "${GPT2_MODEL:?Set GPT2_MODEL}"
: "${WT103_ARTIFACT:?Set WT103_ARTIFACT}"
: "${CHECKPOINT:?Set CHECKPOINT to a step-XXXXXXXX directory}"
: "${EVAL_OUTPUT:?Set EVAL_OUTPUT to a new directory}"
PYTHON_BIN=${PYTHON_BIN:-python}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
"${PYTHON_BIN}" -m dfm.universal_eval \
  --model "${GPT2_MODEL}" \
  --checkpoint "${CHECKPOINT}" \
  --prepared "${UNIVERSAL_PREPARED:-${WT103_ARTIFACT}/prepared/exclude-block}" \
  --random-prepared "${UNIVERSAL_RANDOM_PREPARED:-${WT103_ARTIFACT}/random-negative-seed42}" \
  --datastore "${WT103_ARTIFACT}/datastore" \
  --output "${EVAL_OUTPUT}" \
  --expected-samples 280 --bootstrap-resamples 10000 \
  --bootstrap-seed 42 --random-seed 42 "$@"
