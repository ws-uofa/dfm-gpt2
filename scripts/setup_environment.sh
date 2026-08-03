#!/usr/bin/env bash
set -euo pipefail

# Prepare or validate the persistent shared environment. By default this is
# read-only; pass --install only when dependencies intentionally need updating.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOCAL_ENV=${DFM_LOCAL_ENV:-${REPO_ROOT}/configs/local.env}
if [[ -f "${LOCAL_ENV}" ]]; then
    # shellcheck disable=SC1090
    source "${LOCAL_ENV}"
fi
PYTHON_BIN=${PYTHON_BIN:-/plm-shared/sunsiyuan/.venvs/dfm/bin/python}
INSTALL=0
PIP_CHECK=0
for argument in "$@"; do
    case "${argument}" in
        --install) INSTALL=1 ;;
        --pip-check) PIP_CHECK=1 ;;
        *) echo "Usage: $0 [--install] [--pip-check]" >&2; exit 2 ;;
    esac
done

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing shared Python environment: ${PYTHON_BIN}" >&2
    exit 2
fi
if ((INSTALL)); then
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u ALL_PROXY -u all_proxy \
        "${PYTHON_BIN}" -m pip install -r "${REPO_ROOT}/requirements-dev.txt"
fi

if ((PIP_CHECK)) && ! PIP_CHECK_OUTPUT=$("${PYTHON_BIN}" -m pip check 2>&1); then
    # The shared environment contains platform tools outside this project. A
    # broken optional package such as autofaiss should be visible, but should
    # not prevent validation of dfm-gpt2's own imports and tests.
    echo "Shared-environment pip check reported unrelated conflicts:" >&2
    echo "${PIP_CHECK_OUTPUT}" >&2
fi
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_environment.py"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m pytest -p no:cacheprovider "${REPO_ROOT}/tests"
bash -n "${REPO_ROOT}"/scripts/*.sh
echo "Environment validation passed."
