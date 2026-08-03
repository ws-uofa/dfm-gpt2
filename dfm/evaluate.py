from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import DFMConfig
from .data import Datastore, MemoryCollator, PairedDataset, load_prepared_split, validate_data_contract
from .losses import token_nll
from .model import DFMForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Aligned retrieved/random/off DFM evaluation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--random-prepared", required=True)
    parser.add_argument("--datastore", required=True)
    parser.add_argument("--architecture", choices=("traditional", "transformer_only"), required=True)
    parser.add_argument("--fusion-layers", default="0,2,5,8,10,11")
    parser.add_argument("--memory-dim", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--memory-value-mode", choices=("key", "continuation", "key_plus_continuation"), default="key_plus_continuation")
    parser.add_argument("--projector-hidden", type=int, default=768)
    parser.add_argument("--memory-attention-heads", type=int, default=12)
    parser.add_argument("--gate-type", choices=("none", "per_head", "token_wise_per_head", "token_wise_per_head_concat"), default="token_wise_per_head")
    parser.add_argument("--gate-init", type=float, default=0.0)
    parser.add_argument("--memory-attention-dropout", type=float, default=0.0)
    parser.add_argument("--reader-dim", type=int, default=256)
    parser.add_argument("--reader-layers", type=int, default=4)
    parser.add_argument("--reader-heads", type=int, default=8)
    parser.add_argument("--reader-ff-multiplier", type=int, default=4)
    parser.add_argument("--reader-topology", choices=("causal", "bidirectional"), default="causal")
    parser.add_argument("--reader-write", choices=("residual", "replace"), default="residual")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    validate_data_contract(
        args.prepared,
        args.datastore,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        memory_dim=args.memory_dim,
        value_mode=args.memory_value_mode,
    )

    dataset = PairedDataset(
        load_prepared_split(args.prepared, "test"),
        load_prepared_split(args.random_prepared, "test"),
    )
    if args.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=MemoryCollator(
            Datastore(args.datastore),
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            value_mode=args.memory_value_mode,
        ),
    )
    slots_per_hit = 2 if args.memory_value_mode == "key_plus_continuation" else 1
    config = DFMConfig(
        architecture=args.architecture,
        fusion_layers=tuple(int(value) for value in args.fusion_layers.split(",") if value.strip()),
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
    model = DFMForCausalLM.from_pretrained(args.model, config)
    state = torch.load(Path(args.checkpoint) / "dfm.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda")
    model.to(device).eval()

    sums = {name: 0.0 for name in ("retrieved", "random", "off")}
    count = 0
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            calls = {
                "retrieved": (batch["memory"], batch["memory_mask"]),
                "random": (batch["negative_memory"], batch["negative_memory_mask"]),
                "off": (None, None),
            }
            for name, (memory, mask) in calls.items():
                logits = model(batch["input_ids"], batch["attention_mask"], memory, mask)
                sums[name] += token_nll(logits, batch["labels"]).item() * len(batch["input_ids"])
            count += len(batch["input_ids"])
    nll = {name: value / count for name, value in sums.items()}
    payload = {
        "samples": count,
        "nll": nll,
        "retrieved_minus_random": nll["retrieved"] - nll["random"],
        "retrieved_minus_off": nll["retrieved"] - nll["off"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
