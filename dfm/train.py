from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import DFMConfig, LossConfig
from .data import (
    Datastore,
    MemoryCollator,
    PairedDataset,
    load_prepared_split,
    validate_data_contract,
)
from .losses import margin_loss, token_nll
from .model import DFMForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GPT-2 DFM on prepared WikiText-103 memory")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--datastore", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architecture", choices=("traditional", "transformer_only"), required=True)
    parser.add_argument("--fusion-layers", default="0,2,5,8,10,11")
    parser.add_argument("--fusion-timing", choices=("pre_attn", "post_attn"))
    parser.add_argument("--memory-dim", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--memory-value-mode",
        choices=("key", "continuation", "key_plus_continuation"),
        default="key_plus_continuation",
    )
    parser.add_argument("--projector-hidden", type=int, default=768)
    parser.add_argument("--memory-attention-heads", type=int, default=12)
    parser.add_argument("--gate-type", choices=("none", "per_head", "token_wise_per_head", "token_wise_per_head_concat"), default="token_wise_per_head")
    parser.add_argument("--gate-init", type=float, default=0.0, help="Pre-sigmoid gate logit")
    parser.add_argument("--memory-attention-dropout", type=float, default=0.1)
    parser.add_argument("--reader-dim", type=int, default=256)
    parser.add_argument("--reader-layers", type=int, default=2)
    parser.add_argument("--reader-heads", type=int, default=8)
    parser.add_argument("--reader-ff-multiplier", type=int, default=4)
    parser.add_argument("--reader-dropout", type=float, default=0.0)
    parser.add_argument("--reader-topology", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--reader-write", choices=("residual", "replace"), default="residual")
    parser.add_argument("--reader-sharing", choices=("independent", "shared"), default="independent")
    parser.add_argument("--loss", choices=("ce", "margin"), default="ce")
    parser.add_argument("--negative-prepared")
    parser.add_argument("--negative-control-seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--margin-weight", type=float, default=0.1)
    parser.add_argument(
        "--preserve-negative-rng",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore RNG after the random-memory forward so it cannot perturb later positive paths",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=2000)
    return parser.parse_args()


def save_checkpoint(accelerator: Accelerator, model: DFMForCausalLM, output: Path, step: int) -> None:
    if not accelerator.is_main_process:
        return
    target = output / f"step-{step:08d}"
    if target.exists():
        return
    target.mkdir(parents=True, exist_ok=False)
    unwrapped = accelerator.unwrap_model(model)
    trainable = {name: value.cpu() for name, value in unwrapped.state_dict().items() if not name.startswith("base.")}
    torch.save(trainable, target / "dfm.pt")
    (target / "dfm-config.json").write_text(
        json.dumps(asdict(unwrapped.dfm_config), indent=2, sort_keys=True) + "\n"
    )


def named_tensor_fingerprint(items: Iterable[tuple[str, torch.Tensor]]) -> str:
    """SHA256 over names, dtypes, shapes, and exact tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in items:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_fingerprint(module: torch.nn.Module) -> str:
    return named_tensor_fingerprint(module.state_dict().items())


def dfm_fingerprint(model: DFMForCausalLM) -> str:
    return named_tensor_fingerprint(
        (name, value) for name, value in model.state_dict().items() if not name.startswith("base.")
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def negative_forward_rng_context(reference: torch.Tensor, preserve: bool):
    """Keep the random-memory forward off the later positive RNG trajectory."""

    if not preserve:
        yield
        return
    devices = []
    if reference.is_cuda and reference.device.index is not None:
        devices.append(reference.device.index)
    with torch.random.fork_rng(devices=devices, enabled=True):
        yield


def main() -> None:
    args = parse_args()
    loss_cfg = LossConfig(args.loss, args.margin, args.margin_weight)
    loss_cfg.validate()
    if args.loss == "margin" and not args.negative_prepared:
        raise SystemExit("--negative-prepared is required for margin loss")

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation, mixed_precision="bf16")
    torch.manual_seed(args.seed)
    fusion_layers = tuple(int(value) for value in args.fusion_layers.split(",") if value.strip())
    slots_per_hit = 2 if args.memory_value_mode == "key_plus_continuation" else 1
    cfg = DFMConfig(
        architecture=args.architecture,
        fusion_layers=fusion_layers,
        fusion_timing=args.fusion_timing
        or ("pre_attn" if args.architecture == "traditional" else "post_attn"),
        memory_dim=args.memory_dim,
        memory_slots=args.top_k * slots_per_hit,
        traditional_projector_hidden=args.projector_hidden,
        memory_attention_heads=args.memory_attention_heads,
        gate_type=args.gate_type,
        gate_init=args.gate_init,
        memory_attention_dropout=args.memory_attention_dropout,
        reader_dim=args.reader_dim,
        reader_layers=args.reader_layers,
        reader_heads=args.reader_heads,
        reader_ff_multiplier=args.reader_ff_multiplier,
        reader_dropout=args.reader_dropout,
        reader_topology=args.reader_topology,
        reader_write=args.reader_write,
        reader_sharing=args.reader_sharing,
    )
    validate_data_contract(
        args.prepared,
        args.datastore,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        memory_dim=args.memory_dim,
        value_mode=args.memory_value_mode,
    )
    artifact_meta = {
        "prepared_meta": str((Path(args.prepared).resolve() / "meta.json")),
        "datastore_meta": str((Path(args.datastore).resolve() / "meta.json")),
    }
    if args.loss == "margin":
        validate_data_contract(
            args.negative_prepared,
            args.datastore,
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            memory_dim=args.memory_dim,
            value_mode=args.memory_value_mode,
        )
        negative_meta_path = Path(args.negative_prepared).resolve() / "meta.json"
        negative_meta = json.loads(negative_meta_path.read_text())
        if negative_meta.get("negative_control_protocol") != "uniform-real-datastore-disjoint-v1":
            raise ValueError("margin training requires the audited disjoint real-datastore control")
        if int(negative_meta.get("negative_control_seed", -1)) != args.negative_control_seed:
            raise ValueError("negative-control seed mismatch")
        negative_audit = negative_meta.get("random_negative_audit", {})
        audit_counts = {
            key: int(negative_audit.get(key, negative_meta.get(key, -1)))
            for key in ("retrieved_collision_count", "within_query_duplicate_count")
        }
        if any(value != 0 for value in audit_counts.values()):
            raise ValueError(f"negative-control metadata audit failed: {negative_audit}")
        positive_meta_sha = file_sha256(Path(artifact_meta["prepared_meta"]))
        if negative_meta.get("negative_control_source_meta_sha256") != positive_meta_sha:
            raise ValueError("negative control is not bound to the selected positive prepared metadata")
        artifact_meta["negative_meta"] = str(negative_meta_path)
    model = DFMForCausalLM.from_pretrained(args.model, cfg)
    model.base.eval()  # frozen GPT-2 dropout must not add noise to paired forwards
    initial_base_fingerprint = tensor_fingerprint(model.base)
    initial_dfm_fingerprint = dfm_fingerprint(model)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    seen_gradients: set[str] = set()
    seen_nonzero_gradients: set[str] = set()

    positive = load_prepared_split(args.prepared, "train")
    dataset = positive
    if args.loss == "margin":
        dataset = PairedDataset(positive, load_prepared_split(args.negative_prepared, "train"))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=MemoryCollator(
            Datastore(args.datastore),
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            value_mode=args.memory_value_mode,
        ),
        num_workers=0,
    )
    optimizer = AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    output = Path(args.output)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=False)
        artifacts = {
            name: {"path": path, "sha256": file_sha256(Path(path))}
            for name, path in artifact_meta.items()
        }
        (output / "run.json").write_text(
            json.dumps(
                {
                    "arguments": vars(args),
                    "model": asdict(cfg),
                    "loss": asdict(loss_cfg),
                    "artifacts": artifacts,
                    "initial_dfm_fingerprint": initial_dfm_fingerprint,
                },
                indent=2,
            )
            + "\n"
        )

    total_steps = args.epochs * math.ceil(len(loader) / args.gradient_accumulation)
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps
    )
    step = 0
    model.train()
    accelerator.unwrap_model(model).base.eval()
    for _epoch in range(args.epochs):
        for batch in loader:
            with accelerator.accumulate(model):
                retrieved_logits = model(
                    batch["input_ids"], batch["attention_mask"], batch["memory"], batch["memory_mask"]
                )
                retrieved_nll = token_nll(retrieved_logits, batch["labels"])
                penalty = retrieved_nll.new_zeros(())
                loss = retrieved_nll
                if args.loss == "margin":
                    with negative_forward_rng_context(
                        batch["negative_memory"], args.preserve_negative_rng
                    ):
                        random_logits = model(
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["negative_memory"],
                            batch["negative_memory_mask"],
                        )
                    random_nll = token_nll(random_logits, batch["labels"])
                    loss, penalty = margin_loss(
                        retrieved_nll, random_nll, margin=args.margin, weight=args.margin_weight
                    )
                accelerator.backward(loss)
                unwrapped = accelerator.unwrap_model(model)
                for name, parameter in unwrapped.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        seen_gradients.add(name)
                        if bool(parameter.grad.detach().ne(0).any()):
                            seen_nonzero_gradients.add(name)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                if step % args.log_every == 0 and accelerator.is_main_process:
                    print(
                        json.dumps(
                            {"step": step, "loss": loss.item(), "retrieved_nll": retrieved_nll.item(), "margin_penalty": penalty.item()}
                        ),
                        flush=True,
                    )
                if step % args.save_every == 0:
                    accelerator.wait_for_everyone()
                    save_checkpoint(accelerator, model, output, step)
                if step >= total_steps:
                    break
        if step >= total_steps:
            break
    accelerator.wait_for_everyone()
    save_checkpoint(accelerator, model, output, step)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        final_base_fingerprint = tensor_fingerprint(unwrapped.base)
        final_checkpoint = output / f"step-{step:08d}" / "dfm.pt"
        audit = {
            "passed": (
                initial_base_fingerprint == final_base_fingerprint
                and trainable_names == seen_gradients
                and trainable_names == seen_nonzero_gradients
            ),
            "step": step,
            "base_fingerprint_before": initial_base_fingerprint,
            "base_fingerprint_after": final_base_fingerprint,
            "initial_dfm_fingerprint": initial_dfm_fingerprint,
            "final_dfm_fingerprint": dfm_fingerprint(unwrapped),
            "final_checkpoint": str(final_checkpoint.resolve()),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "frozen_base_unchanged": initial_base_fingerprint == final_base_fingerprint,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in unwrapped.parameters() if parameter.requires_grad
            ),
            "trainable_tensor_count": len(trainable_names),
            "missing_gradient_tensors": sorted(trainable_names - seen_gradients),
            "missing_nonzero_gradient_tensors": sorted(trainable_names - seen_nonzero_gradients),
        }
        (output / "training-audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        if not audit["passed"]:
            raise RuntimeError("training audit failed; see training-audit.json")


if __name__ == "__main__":
    main()
