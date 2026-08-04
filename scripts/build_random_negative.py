#!/usr/bin/env python
"""Create the fixed, aligned real-datastore negative control for margin loss."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", required=True)
    parser.add_argument("--datastore", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    positive, output = Path(args.positive), Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    size = int(json.loads((Path(args.datastore) / "meta.json").read_text())["num_chunks"])
    rng = np.random.default_rng(args.seed)
    collisions = duplicates = row_count = valid_id_count = 0
    random_id_hash = hashlib.sha256()
    split_parts: dict[str, list[str]] = {}
    for split in ("train", "validation", "test"):
        parts = sorted(positive.glob(f"shards/shard-*/prepared_parts/{split}/part-*"))
        split_parts[split] = []
        for part_index, part_path in enumerate(parts):
            rows = []
            for row in load_from_disk(str(part_path)):
                positive_ids = np.asarray(row["retrieved_chunk_ids"], dtype=np.int64)
                random_ids = np.full_like(positive_ids, -1)
                for query, ids in enumerate(positive_ids):
                    valid = ids >= 0
                    if not valid.any():
                        continue
                    forbidden = set(ids[valid].tolist())
                    chosen: list[int] = []
                    while len(chosen) < int(valid.sum()):
                        candidate = int(rng.integers(0, size))
                        if candidate not in forbidden and candidate not in chosen:
                            chosen.append(candidate)
                    random_ids[query, valid] = chosen
                    collisions += len(forbidden.intersection(chosen))
                    duplicates += len(chosen) - len(set(chosen))
                copy = dict(row)
                copy["retrieved_chunk_ids"] = random_ids.tolist()
                rows.append(copy)
                row_count += 1
                valid_id_count += int((random_ids >= 0).sum())
                random_id_hash.update(np.asarray(random_ids, dtype="<i8").tobytes())
            relative = f"shards/shard-00000-of-00001/prepared_parts/{split}/part-{part_index:06d}"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_list(rows).save_to_disk(str(destination))
            split_parts[split].append(relative)
    positive_meta_path = positive / "meta.json"
    meta = json.loads(positive_meta_path.read_text())
    meta.update({
        "negative_control_protocol": "uniform-real-datastore-disjoint-v1",
        "negative_control_seed": args.seed,
        "negative_control_source": str(positive.resolve()),
        "negative_control_source_meta_sha256": sha256(positive_meta_path),
        "retrieved_collision_count": collisions,
        "within_query_duplicate_count": duplicates,
        "random_id_aggregate_sha256": random_id_hash.hexdigest(),
        "random_negative_audit": {
            "row_count": row_count,
            "valid_id_count": valid_id_count,
            "retrieved_collision_count": collisions,
            "within_query_duplicate_count": duplicates,
        },
        "split_parts": split_parts,
    })
    output.mkdir(parents=True, exist_ok=True)
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
