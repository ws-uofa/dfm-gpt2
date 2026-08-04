from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import DFMConfig
from .data import Datastore, MemoryCollator, PairedDataset, load_prepared_split, validate_data_contract
from .model import DFMForCausalLM


CONDITIONS = ("off", "retrieved", "random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal WikiText-103 retrieved/random/off test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True, help="A step-XXXXXXXX directory")
    parser.add_argument("--prepared", required=True, help="Fixed cross-article retrieved test view")
    parser.add_argument("--random-prepared", required=True, help="Seed-42 aligned disjoint random view")
    parser.add_argument("--datastore", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp-dtype", choices=("none", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--expected-samples", type=int, default=280)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    shifted_logits, shifted_labels = logits[:, :-1].float(), labels[:, 1:]
    valid = shifted_labels.ne(-100)
    safe = shifted_labels.masked_fill(~valid, 0)
    nll = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        safe.reshape(-1),
        reduction="none",
    ).reshape_as(safe)
    correct = shifted_logits.argmax(dim=-1).eq(safe)
    return (
        nll[valid].detach().cpu().numpy().astype(np.float64),
        correct[valid].detach().cpu().numpy().astype(np.float64),
    )


def bootstrap(values: np.ndarray, resamples: int, seed: int) -> list[float]:
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    rng = np.random.default_rng(seed)
    means = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(resamples)]
    )
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def validate_protocol(args: argparse.Namespace, config: DFMConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    positive_meta = json.loads((Path(args.prepared) / "meta.json").read_text())
    random_meta = json.loads((Path(args.random_prepared) / "meta.json").read_text())
    expected = {"block_size": 1024, "chunk_size": 16, "top_k": 16}
    for name, meta in (("retrieved", positive_meta), ("random", random_meta)):
        # Training blocks may use article_partial/article_only.  The universal
        # test target is nevertheless the fixed cross-article test stream.
        target_packing = meta.get("target_packing_mode", meta.get("packing_mode"))
        if target_packing != "cross_article":
            raise ValueError(
                f"{name} universal-test target packing must be cross_article; "
                f"got {target_packing!r}"
            )
        mismatches = {key: (meta.get(key), value) for key, value in expected.items() if meta.get(key) != value}
        if mismatches:
            raise ValueError(f"{name} universal-test metadata mismatch: {mismatches}")
    if random_meta.get("negative_control_protocol") != "uniform-real-datastore-disjoint-v1":
        raise ValueError("random view is not the audited disjoint real-datastore control")
    if int(random_meta.get("negative_control_seed", -1)) != args.random_seed:
        raise ValueError("random-control seed mismatch")
    positive_meta_hash = sha256(Path(args.prepared) / "meta.json")
    if random_meta.get("negative_control_source_meta_sha256") != positive_meta_hash:
        raise ValueError("random view is not bound to the selected retrieved metadata")
    validate_data_contract(
        args.prepared,
        args.datastore,
        chunk_size=16,
        top_k=16,
        memory_dim=config.memory_dim,
        value_mode="key_plus_continuation",
    )
    return positive_meta, random_meta


def autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    """Return a fresh context manager for each model call."""

    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def compact_protocol(meta: dict[str, Any]) -> dict[str, Any]:
    """Keep the reproducibility contract without duplicating large manifests."""

    keys = (
        "packing_mode",
        "target_packing_mode",
        "block_size",
        "chunk_size",
        "top_k",
        "negative_control_protocol",
        "negative_control_seed",
        "memory_value_mode",
    )
    return {key: meta[key] for key in keys if key in meta}


def validate_fixed_arguments(args: argparse.Namespace) -> None:
    expected = {
        "expected_samples": 280,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 42,
        "random_seed": 42,
        "amp_dtype": "bfloat16",
    }
    mismatches = {
        key: (getattr(args, key), value)
        for key, value in expected.items()
        if getattr(args, key) != value
    }
    if torch.device(args.device).type != "cuda":
        mismatches["device"] = (args.device, "cuda")
    if mismatches:
        raise ValueError(f"universal-test fixed arguments changed: {mismatches}")


def main() -> None:
    args = parse_args()
    validate_fixed_arguments(args)
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config_payload = json.loads((checkpoint / "dfm-config.json").read_text())
    config_payload["fusion_layers"] = tuple(config_payload["fusion_layers"])
    config = DFMConfig(**config_payload)
    positive_meta, random_meta = validate_protocol(args, config)

    audit_path = checkpoint.parent / "training-audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"strict universal evaluation requires the run audit: {audit_path}"
        )
    training_audit = json.loads(audit_path.read_text())
    if not training_audit.get("passed"):
        raise ValueError(f"training audit did not pass: {audit_path}")
    expected_checkpoint_name = f"step-{int(training_audit.get('step', -1)):08d}"
    if checkpoint.name != expected_checkpoint_name:
        raise ValueError(
            f"strict evaluation requires the audited final checkpoint {expected_checkpoint_name}"
        )

    positive = load_prepared_split(args.prepared, "test")
    random = load_prepared_split(args.random_prepared, "test")
    if len(positive) != args.expected_samples or len(random) != args.expected_samples:
        raise ValueError(
            f"universal test requires exactly {args.expected_samples} aligned rows; "
            f"got {len(positive)} and {len(random)}"
        )
    dataset = PairedDataset(positive, random)
    collator = MemoryCollator(
        Datastore(args.datastore),
        chunk_size=16,
        top_k=16,
        value_mode="key_plus_continuation",
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)

    model = DFMForCausalLM.from_pretrained(args.model, config)
    state = torch.load(checkpoint / "dfm.pt", map_location="cpu", weights_only=True)
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("checkpoint contains a non-finite trainable tensor")
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_nonbase = [name for name in incompatible.missing_keys if not name.startswith("base.")]
    if unexpected or missing_nonbase:
        raise ValueError(f"checkpoint mismatch: unexpected={unexpected}, missing={missing_nonbase}")
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_count != int(training_audit.get("trainable_parameter_count", -1)):
        raise ValueError("checkpoint architecture does not match the training-audit parameter count")
    checkpoint_hash = sha256(checkpoint / "dfm.pt")
    if checkpoint_hash != training_audit.get("final_checkpoint_sha256"):
        raise ValueError("checkpoint SHA256 does not match training-audit.json")
    device = torch.device(args.device)
    model.to(device).eval()
    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(args.amp_dtype)

    condition_nll: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    condition_top1: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    rows: list[dict[str, Any]] = []
    random_hash = hashlib.sha256()
    random_audit = {"retrieved_collisions": 0, "within_query_duplicates": 0, "mask_mismatches": 0}
    with torch.inference_mode():
        for sample_idx, batch in enumerate(loader):
            retrieved_ids = np.asarray(positive[sample_idx]["retrieved_chunk_ids"], dtype=np.int64)
            random_ids = np.asarray(random[sample_idx]["retrieved_chunk_ids"], dtype=np.int64)
            random_hash.update(np.asarray(random_ids, dtype="<i8").tobytes())
            random_audit["mask_mismatches"] += int(np.count_nonzero((retrieved_ids >= 0) != (random_ids >= 0)))
            for retrieved_row, random_row in zip(retrieved_ids, random_ids):
                retrieved_valid = retrieved_row[retrieved_row >= 0]
                random_valid = random_row[random_row >= 0]
                random_audit["retrieved_collisions"] += int(np.isin(random_valid, retrieved_valid).sum())
                random_audit["within_query_duplicates"] += len(random_valid) - len(np.unique(random_valid))

            batch = {name: value.to(device) for name, value in batch.items()}
            calls = {
                "off": (None, None),
                "retrieved": (batch["memory"], batch["memory_mask"]),
                "random": (batch["negative_memory"], batch["negative_memory_mask"]),
            }
            sample_values = {}
            for condition, (memory, mask) in calls.items():
                with autocast_context(device, amp_dtype):
                    logits = model(batch["input_ids"], batch["attention_mask"], memory, mask)
                nll, top1 = token_metrics(logits, batch["labels"])
                condition_nll[condition].append(nll)
                condition_top1[condition].append(top1)
                sample_values[condition] = nll
            if len({len(value) for value in sample_values.values()}) != 1:
                raise ValueError("condition token alignment differs")
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "token_count": len(sample_values["off"]),
                    **{f"{name}_mean_nll": float(value.mean()) for name, value in sample_values.items()},
                    "retrieved_minus_random_mean_nll": float(
                        sample_values["retrieved"].mean() - sample_values["random"].mean()
                    ),
                    "retrieved_minus_off_mean_nll": float(
                        sample_values["retrieved"].mean() - sample_values["off"].mean()
                    ),
                }
            )
    if any(random_audit.values()):
        raise ValueError(f"random-control audit failed: {random_audit}")

    all_nll = {name: np.concatenate(values) for name, values in condition_nll.items()}
    all_top1 = {name: np.concatenate(values) for name, values in condition_top1.items()}
    sample_ret_random = np.asarray([row["retrieved_minus_random_mean_nll"] for row in rows])
    sample_ret_off = np.asarray([row["retrieved_minus_off_mean_nll"] for row in rows])
    summary = {
        "protocol": "wikitext103_universal_test_v1",
        "passed": True,
        "config": config_payload,
        "num_samples": len(rows),
        "num_tokens": int(len(all_nll["off"])),
        "conditions": {
            name: {
                "mean_nll": float(all_nll[name].mean()),
                "perplexity": math.exp(float(all_nll[name].mean())),
                "top1_accuracy": float(all_top1[name].mean()),
            }
            for name in CONDITIONS
        },
        "comparisons": {
            "retrieved_minus_random": {
                "token_weighted_mean_nll": float(all_nll["retrieved"].mean() - all_nll["random"].mean()),
                "sample_equal_mean_nll": float(sample_ret_random.mean()),
                "paired_bootstrap_95_ci": bootstrap(sample_ret_random, args.bootstrap_resamples, args.bootstrap_seed),
                "retrieved_sample_win_rate": float((sample_ret_random < 0).mean()),
            },
            "retrieved_minus_off": {
                "token_weighted_mean_nll": float(all_nll["retrieved"].mean() - all_nll["off"].mean()),
                "sample_equal_mean_nll": float(sample_ret_off.mean()),
                "paired_bootstrap_95_ci": bootstrap(sample_ret_off, args.bootstrap_resamples, args.bootstrap_seed),
                "retrieved_sample_win_rate": float((sample_ret_off < 0).mean()),
            },
        },
        "random_control": {
            "seed": args.random_seed,
            "id_sha256": random_hash.hexdigest(),
            "audit": random_audit,
        },
        "artifacts": {
            "checkpoint_sha256": checkpoint_hash,
            "config_sha256": sha256(checkpoint / "dfm-config.json"),
            "training_audit_sha256": sha256(audit_path),
            "retrieved_meta_sha256": sha256(Path(args.prepared) / "meta.json"),
            "random_meta_sha256": sha256(Path(args.random_prepared) / "meta.json"),
            "datastore_meta_sha256": sha256(Path(args.datastore) / "meta.json"),
        },
        "training_audit": training_audit,
        "prepared_protocol": compact_protocol(positive_meta),
        "random_protocol": compact_protocol(random_meta),
    }
    output.mkdir(parents=True)
    with (output / "sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
