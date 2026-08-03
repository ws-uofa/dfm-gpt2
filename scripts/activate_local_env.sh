#!/usr/bin/env bash
# Source this file to select the shared project environment and local paths.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOCAL_ENV=${DFM_LOCAL_ENV:-${REPO_ROOT}/configs/local.env}
if [[ ! -f "${LOCAL_ENV}" ]]; then
    echo "Missing ${LOCAL_ENV}. Copy configs/paths.env.example to configs/local.env first." >&2
    return 2 2>/dev/null || exit 2
fi

# shellcheck disable=SC1090
source "${LOCAL_ENV}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TMPDIR="${DFM_TMPDIR:?DFM_TMPDIR is not set}"
mkdir -p "${HF_HOME:?HF_HOME is not set}" \
    "${HF_DATASETS_CACHE:?HF_DATASETS_CACHE is not set}" \
    "${TMPDIR}" "${DFM_RUNS:?DFM_RUNS is not set}" || return 2

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    return 2 2>/dev/null || exit 2
fi
echo "dfm-gpt2 environment: ${PYTHON_BIN} (${REPO_ROOT})"
