#!/usr/bin/env bash
set -euo pipefail

# Intent: reproduce the conventional frozen-GPT-2 DFM baseline with LM CE only.
: "${GPT2_MODEL:?Set GPT2_MODEL}"; : "${WT103_ARTIFACT:?Set WT103_ARTIFACT}"; : "${DFM_RUNS:?Set DFM_RUNS}"
PYTHON_BIN=${PYTHON_BIN:-python}
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
"${PYTHON_BIN}" -m dfm.train \
  --model "${GPT2_MODEL}" --datastore "${WT103_ARTIFACT}/datastore" \
  --prepared "${WT103_ARTIFACT}/prepared/exclude-block" \
  --output "${DFM_RUNS}/traditional-ce" --architecture traditional --loss ce "$@"
