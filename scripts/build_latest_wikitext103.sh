#!/usr/bin/env bash
set -euo pipefail

# Intent: reproduce the latest leakage-safe WikiText-103 database and prepared
# retrieval rows. This is a GPU data job, not a training job.
: "${WIKITEXT103:?Set WIKITEXT103}"
: "${GPT2_MODEL:?Set GPT2_MODEL}"
: "${EMBEDDING_MODEL:?Set EMBEDDING_MODEL}"
: "${WT103_ARTIFACT:?Set WT103_ARTIFACT}"
PYTHON_BIN=${PYTHON_BIN:-python}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
"${PYTHON_BIN}" scripts/build_wikitext103.py all \
  --dataset "${WIKITEXT103}" \
  --gpt2-tokenizer "${GPT2_MODEL}" \
  --embedding-model "${EMBEDDING_MODEL}" \
  --output "${WT103_ARTIFACT}" "$@"
