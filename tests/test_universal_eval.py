from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from dfm.config import DFMConfig
from dfm.universal_eval import (
    bootstrap,
    compact_protocol,
    validate_fixed_arguments,
    validate_protocol,
)


def write_meta(root: Path, payload: dict[str, object]) -> None:
    root.mkdir()
    (root / "meta.json").write_text(json.dumps(payload))


def test_bootstrap_is_deterministic() -> None:
    values = np.asarray([-0.2, -0.1, 0.0, 0.1])
    assert bootstrap(values, 100, 42) == bootstrap(values, 100, 42)
    with pytest.raises(ValueError, match="empty"):
        bootstrap(np.asarray([]), 10, 42)


def test_universal_arguments_are_fixed() -> None:
    args = argparse.Namespace(
        expected_samples=280,
        bootstrap_resamples=10_000,
        bootstrap_seed=42,
        random_seed=42,
        amp_dtype="bfloat16",
        device="cuda:0",
    )
    validate_fixed_arguments(args)
    args.expected_samples = 10
    with pytest.raises(ValueError, match="fixed arguments"):
        validate_fixed_arguments(args)


def test_compact_protocol_drops_large_manifest() -> None:
    result = compact_protocol(
        {"packing_mode": "article_partial", "target_packing_mode": "cross_article", "parts": list(range(100))}
    )
    assert result == {
        "packing_mode": "article_partial",
        "target_packing_mode": "cross_article",
    }


def test_protocol_accepts_article_training_with_cross_article_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = {
        "packing_mode": "article_partial",
        "target_packing_mode": "cross_article",
        "block_size": 1024,
        "chunk_size": 16,
        "top_k": 16,
    }
    positive = tmp_path / "positive"
    random = tmp_path / "random"
    write_meta(positive, common)
    positive_sha = hashlib.sha256((positive / "meta.json").read_bytes()).hexdigest()
    write_meta(
        random,
        {
            **common,
            "negative_control_protocol": "uniform-real-datastore-disjoint-v1",
            "negative_control_seed": 42,
            "negative_control_source_meta_sha256": positive_sha,
        },
    )
    monkeypatch.setattr("dfm.universal_eval.validate_data_contract", lambda *args, **kwargs: None)
    args = argparse.Namespace(
        prepared=str(positive),
        random_prepared=str(random),
        datastore=str(tmp_path / "datastore"),
        random_seed=42,
    )
    validate_protocol(args, DFMConfig(architecture="transformer_only"))


def test_protocol_rejects_non_cross_article_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = {
        "packing_mode": "article_only",
        "block_size": 1024,
        "chunk_size": 16,
        "top_k": 16,
    }
    positive = tmp_path / "positive"
    random = tmp_path / "random"
    write_meta(positive, meta)
    positive_sha = hashlib.sha256((positive / "meta.json").read_bytes()).hexdigest()
    write_meta(
        random,
        {
            **meta,
            "negative_control_protocol": "uniform-real-datastore-disjoint-v1",
            "negative_control_seed": 42,
            "negative_control_source_meta_sha256": positive_sha,
        },
    )
    monkeypatch.setattr("dfm.universal_eval.validate_data_contract", lambda *args, **kwargs: None)
    args = argparse.Namespace(
        prepared=str(positive),
        random_prepared=str(random),
        datastore=str(tmp_path / "datastore"),
        random_seed=42,
    )
    with pytest.raises(ValueError, match="target packing"):
        validate_protocol(args, DFMConfig())
