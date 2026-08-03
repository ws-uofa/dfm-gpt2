from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

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
    parser.add_argument("--memory-attention-dropout", type=float, default=0.0)
    parser.add_argument("--reader-dim", type=int, default=256)
    parser.add_argument("--reader-layers", type=int, default=4)
    parser.add_argument("--reader-heads", type=int, default=8)
    parser.add_argument("--reader-ff-multiplier", type=int, default=4)
    parser.add_argument("--reader-topology", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--reader-write", choices=("residual", "replace"), default="residual")
    parser.add_argument("--loss", choices=("ce", "margin"), default="ce")
    parser.add_argument("--negative-prepared")
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--margin-weight", type=float, default=0.1)
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
        reader_topology=args.reader_topology,
        reader_write=args.reader_write,
    )
    validate_data_contract(
        args.prepared,
        args.datastore,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        memory_dim=args.memory_dim,
        value_mode=args.memory_value_mode,
    )
    model = DFMForCausalLM.from_pretrained(args.model, cfg)
    model.base.eval()  # frozen GPT-2 dropout must not add noise to paired forwards

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
        (output / "run.json").write_text(
            json.dumps({"arguments": vars(args), "model": asdict(cfg), "loss": asdict(loss_cfg)}, indent=2) + "\n"
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
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
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


if __name__ == "__main__":
    main()
