#!/usr/bin/env python
"""Check imports, versions, storage paths, and optional CUDA visibility."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import sys


PACKAGES = ("torch", "transformers", "accelerate", "datasets", "numpy", "faiss-cpu")
PATHS = ("GPT2_MODEL", "EMBEDDING_MODEL", "WIKITEXT103", "WT103_ARTIFACT", "DFM_RUNS")


def main() -> None:
    import faiss
    import torch

    missing = [name for name in PATHS if not os.environ.get(name)]
    absent = [name for name in PATHS if os.environ.get(name) and not Path(os.environ[name]).exists()]
    report = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "faiss_runtime": getattr(faiss, "__version__", "unknown"),
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpus": torch.cuda.device_count(),
        "missing_environment_variables": missing,
        "missing_paths": absent,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing or absent:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

