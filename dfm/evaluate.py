from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import DFMConfig
from .data import Datastore, MemoryCollator, PairedDataset, load_prepared_split
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    dataset = PairedDataset(
        load_prepared_split(args.prepared, "test"),
        load_prepared_split(args.random_prepared, "test"),
    )
    if args.max_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=MemoryCollator(Datastore(args.datastore)))
    model = DFMForCausalLM.from_pretrained(args.model, DFMConfig(architecture=args.architecture))
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
