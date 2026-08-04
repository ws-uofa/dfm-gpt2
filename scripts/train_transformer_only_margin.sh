#!/usr/bin/env bash
set -euo pipefail

# Intent: apply the latest retrieved-vs-random margin objective to the default
# performance/parameter tradeoff transformer-only residual reader.
: "${GPT2_MODEL:?Set GPT2_MODEL}"; : "${WT103_ARTIFACT:?Set WT103_ARTIFACT}"; : "${DFM_RUNS:?Set DFM_RUNS}"
PYTHON_BIN=${PYTHON_BIN:-python}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
"${PYTHON_BIN}" -m dfm.train \
  --model "${GPT2_MODEL}" --datastore "${WT103_ARTIFACT}/datastore" \
  --prepared "${WT103_ARTIFACT}/prepared/exclude-block" \
  --negative-prepared "${WT103_ARTIFACT}/random-negative-seed42" \
  --output "${DFM_RUNS}/transformer-only-margin" --architecture transformer_only --loss margin \
  --margin 0.05 --margin-weight 0.1 "$@"
