from __future__ import annotations

import json
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from datasets import Dataset, load_from_disk
from torch import Tensor
from torch.utils.data import Dataset as TorchDataset


def validate_data_contract(
    prepared_root: str | Path,
    datastore_root: str | Path,
    *,
    chunk_size: int,
    top_k: int,
    memory_dim: int,
    value_mode: str,
) -> None:
    """Fail early when an ablation config does not match its artifacts."""

    prepared = json.loads((Path(prepared_root) / "meta.json").read_text())
    datastore = json.loads((Path(datastore_root) / "meta.json").read_text())
    expected_chunk = int(prepared.get("chunk_size", datastore.get("chunk_size", -1)))
    stored_top_k = int(prepared.get("top_k", prepared.get("stored_top_k", -1)))
    if chunk_size != expected_chunk:
        raise ValueError(f"chunk_size={chunk_size} but artifact uses {expected_chunk}")
    if not 0 < top_k <= stored_top_k:
        raise ValueError(f"top_k={top_k} but artifact stores {stored_top_k}")
    if memory_dim != int(datastore["embedding_dim"]):
        raise ValueError("memory_dim does not match datastore embeddings")
    if value_mode != "key" and not datastore.get("future_embeddings_file"):
        raise ValueError(f"value_mode={value_mode} requires continuation embeddings")


class LazyPartsDataset(TorchDataset):
    """Random access over the sharded Hugging Face parts used by the full build."""

    def __init__(self, paths: Sequence[Path], lengths: Sequence[int], cache_size: int = 4) -> None:
        self.paths = list(paths)
        self.ends = np.cumsum(lengths).tolist()
        self.cache_size = cache_size
        self.cache: OrderedDict[int, Dataset] = OrderedDict()

    def __len__(self) -> int:
        return self.ends[-1] if self.ends else 0

    def _part(self, index: int) -> Dataset:
        if index not in self.cache:
            self.cache[index] = load_from_disk(str(self.paths[index]))
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        part = bisect_right(self.ends, index)
        start = 0 if part == 0 else self.ends[part - 1]
        return self._part(part)[index - start]


def _part_length(path: Path) -> int:
    state = json.loads((path / "state.json").read_text())
    # Dataset state does not always record row count, so Arrow metadata is the
    # authoritative fallback. This is done once at startup, never per sample.
    if "_num_rows" in state:
        return int(state["_num_rows"])
    return len(load_from_disk(str(path)))


def load_prepared_split(root: str | Path, split: str) -> LazyPartsDataset:
    root = Path(root)
    parts = sorted(root.glob(f"shards/shard-*/prepared_parts/{split}/part-*"))
    if not parts:
        direct = root / split
        if direct.exists():
            dataset = load_from_disk(str(direct))
            return LazyPartsDataset([direct], [len(dataset)])
        raise FileNotFoundError(f"No prepared {split!r} parts below {root}")
    return LazyPartsDataset(parts, [_part_length(path) for path in parts])


class PairedDataset(TorchDataset):
    """Join aligned retrieved and random-control rows for margin training."""

    def __init__(self, positive: TorchDataset, negative: TorchDataset) -> None:
        if len(positive) != len(negative):
            raise ValueError("positive and negative datasets are not aligned")
        self.positive = positive
        self.negative = negative

    def __len__(self) -> int:
        return len(self.positive)

    def __getitem__(self, index: int) -> dict[str, Any]:
        positive, negative = dict(self.positive[index]), self.negative[index]
        if positive["input_ids"] != negative["input_ids"]:
            raise ValueError(f"unaligned negative row at index {index}")
        positive["negative_retrieved_chunk_ids"] = negative["retrieved_chunk_ids"]
        return positive


class Datastore:
    """Memory-mapped key and continuation embeddings; no large copy is made."""

    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        meta = json.loads((root / "meta.json").read_text())
        self.size = int(meta["num_chunks"])
        self.dim = int(meta["embedding_dim"])
        self.keys = np.load(root / meta["embeddings_file"], mmap_mode="r")
        future_file = meta.get("future_embeddings_file")
        self.future = np.load(root / future_file, mmap_mode="r") if future_file else None
        if self.keys.shape != (self.size, self.dim):
            raise ValueError("datastore embedding shapes do not match meta.json")
        if self.future is not None and self.future.shape != (self.size, self.dim):
            raise ValueError("future embedding shape does not match meta.json")

    def lookup(self, ids: Tensor, value_mode: str) -> tuple[Tensor, Tensor]:
        """Return key, continuation, or interleaved key+continuation slots."""

        valid = ids >= 0
        safe = ids.clamp_min(0).cpu().numpy()
        keys = torch.from_numpy(np.asarray(self.keys[safe]).copy())
        if value_mode == "key":
            return keys, valid
        if self.future is None:
            raise ValueError(f"value_mode={value_mode!r} requires future embeddings")
        future = torch.from_numpy(np.asarray(self.future[safe]).copy())
        if value_mode == "continuation":
            return future, valid
        if value_mode != "key_plus_continuation":
            raise ValueError(f"unsupported memory value mode: {value_mode}")
        memory = torch.stack((keys, future), dim=-2).flatten(-3, -2)
        mask = torch.stack((valid, valid), dim=-1).flatten(-2, -1)
        return memory, mask


class MemoryCollator:
    def __init__(
        self,
        datastore: Datastore,
        *,
        chunk_size: int = 16,
        top_k: int = 16,
        value_mode: str = "key_plus_continuation",
    ) -> None:
        self.datastore = datastore
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.value_mode = value_mode

    def _memory(self, rows: list[dict[str, Any]], key: str) -> tuple[Tensor, Tensor]:
        ids = torch.tensor([row[key] for row in rows], dtype=torch.long)
        if self.top_k <= 0 or self.top_k > ids.shape[-1]:
            raise ValueError(f"top_k must be in [1, {ids.shape[-1]}]")
        ids = ids[..., : self.top_k]
        chunk_memory, chunk_mask = self.datastore.lookup(ids, self.value_mode)
        # Every token in one LM chunk sees the retrieval result attached to that chunk.
        memory = chunk_memory.repeat_interleave(self.chunk_size, dim=1)
        mask = chunk_mask.repeat_interleave(self.chunk_size, dim=1)
        return memory, mask

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Tensor]:
        batch = {
            name: torch.tensor([row[name] for row in rows], dtype=torch.long)
            for name in ("input_ids", "attention_mask", "labels")
        }
        batch["memory"], batch["memory_mask"] = self._memory(rows, "retrieved_chunk_ids")
        if "negative_retrieved_chunk_ids" in rows[0]:
            batch["negative_memory"], batch["negative_memory_mask"] = self._memory(
                rows, "negative_retrieved_chunk_ids"
            )
        return batch
