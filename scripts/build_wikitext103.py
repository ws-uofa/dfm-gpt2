#!/usr/bin/env python
"""Build the leakage-safe WikiText-103 datastore used by this repository.

The script is stage based so the 7.5M-vector full build can be resumed. It uses
cross-article packing and excludes every candidate from the current LM block,
which is the best audited configuration of database-construction-v1.
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator

import faiss
import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer


SPLITS = ("train", "validation", "test")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("tokenize", "encode", "index", "prepare", "all"))
    parser.add_argument("--dataset", required=True, help="Local saved dataset or Salesforce/wikitext")
    parser.add_argument("--gpt2-tokenizer", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--candidate-pool", type=int, default=2048)
    parser.add_argument("--nlist", type=int, default=10942)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--pq-code-size", type=int, default=64)
    parser.add_argument("--part-rows", type=int, default=128)
    parser.add_argument("--max-blocks", type=int, help="Small deterministic smoke build")
    return parser.parse_args()


def load_wikitext(path: str) -> DatasetDict:
    source = Path(path)
    if source.exists():
        value = load_from_disk(str(source))
    else:
        value = load_dataset(path, "wikitext-103-raw-v1")
    if not isinstance(value, DatasetDict):
        raise TypeError("WikiText source must be a DatasetDict")
    return value


def token_stream(split: Dataset, tokenizer, batch_size: int = 1024) -> Iterator[int]:
    """Concatenate rows with EOS; chunks and blocks may cross article boundaries."""

    eos = tokenizer.eos_token_id
    for start in range(0, len(split), batch_size):
        rows = split[start : start + batch_size]["text"]
        for ids in tokenizer(rows, add_special_tokens=False)["input_ids"]:
            yield from ids
            yield eos


def blocks(stream: Iterable[int], size: int = 1024) -> Iterator[np.ndarray]:
    buffer: list[int] = []
    for token in stream:
        buffer.append(token)
        if len(buffer) == size:
            yield np.asarray(buffer, dtype=np.int32)
            buffer.clear()


def stage_tokenize(args: argparse.Namespace) -> None:
    root = Path(args.output)
    tokens_root = root / "tokens"
    tokens_root.mkdir(parents=True, exist_ok=True)
    dataset = load_wikitext(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_tokenizer, use_fast=True)
    counts = {}
    for split in SPLITS:
        rows = blocks(token_stream(dataset[split], tokenizer))
        if args.max_blocks:
            rows = islice(rows, args.max_blocks)
        path = tokens_root / f"{split}-blocks.npy"
        collected = list(rows) if args.max_blocks else None
        if collected is not None:
            array = np.stack(collected) if collected else np.empty((0, 1024), dtype=np.int32)
        else:
            # Count once, then fill a disk-backed NPY without holding the corpus in RAM.
            count = sum(1 for _ in blocks(token_stream(dataset[split], tokenizer)))
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32, shape=(count, 1024))
            for index, row in enumerate(blocks(token_stream(dataset[split], tokenizer))):
                array[index] = row
        if collected is not None:
            np.save(path, array)
        counts[split] = int(array.shape[0])
        if split == "train":
            # The datastore also keeps complete trailing 16-token chunks which
            # do not form a complete 1024-token LM block. This reproduces the
            # audited total of 7,482,592 key/continuation pairs.
            chunk_limit = args.max_blocks * 64 + 1 if args.max_blocks else None
            chunk_count = (
                chunk_limit
                if chunk_limit is not None
                else sum(1 for _ in islice(token_stream(dataset[split], tokenizer), 15, None, 16))
            )
            chunks_path = tokens_root / "train-chunks.npy"
            chunks = np.lib.format.open_memmap(
                chunks_path, mode="w+", dtype=np.int32, shape=(chunk_count, 16)
            )
            stream = iter(token_stream(dataset[split], tokenizer))
            for chunk_index in range(chunk_count):
                chunk = list(islice(stream, 16))
                if len(chunk) != 16:
                    raise RuntimeError("token stream ended before the counted chunk total")
                chunks[chunk_index] = chunk
            chunks.flush()
    (tokens_root / "meta.json").write_text(json.dumps({"packing": "cross_article", "block_size": 1024, "counts": counts}, indent=2) + "\n")


class Embedder:
    def __init__(self, name: str, device: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def __call__(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if is_query:
            instruction = "Instruct: Given a text chunk, retrieve relevant memory chunks\nQuery:"
            texts = [f"{instruction}\n{text}" for text in texts]
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state
        last = batch.attention_mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        return torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy()


def batches(total: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, size):
        yield start, min(start + size, total)


def stage_encode(args: argparse.Namespace) -> None:
    root = Path(args.output)
    train = np.load(root / "tokens/train-chunks.npy", mmap_mode="r")
    pair_count = len(train) - 1
    encoder = Embedder(args.embedding_model, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_tokenizer, use_fast=True)
    datastore = root / "datastore"
    datastore.mkdir(parents=True, exist_ok=True)
    keys = np.lib.format.open_memmap(datastore / "embeddings.npy", mode="w+", dtype=np.float16, shape=(pair_count, encoder.dim))
    future = np.lib.format.open_memmap(datastore / "future_embeddings.npy", mode="w+", dtype=np.float16, shape=(pair_count, encoder.dim))
    for start, end in batches(pair_count, args.batch_size):
        keys[start:end] = encoder(tokenizer.batch_decode(train[start:end], skip_special_tokens=False)).astype(np.float16)
        future[start:end] = encoder(tokenizer.batch_decode(train[start + 1 : end + 1], skip_special_tokens=False)).astype(np.float16)
        keys.flush(); future.flush()
    meta = {
        "protocol": "wikitext103_database_construction_v1",
        "packing_mode": "cross_article",
        "num_chunks": pair_count,
        "embedding_dim": encoder.dim,
        "embeddings_file": "embeddings.npy",
        "future_embeddings_file": "future_embeddings.npy",
        "chunk_size": 16,
    }
    (datastore / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def stage_index(args: argparse.Namespace) -> None:
    root = Path(args.output) / "datastore"
    vectors = np.load(root / "embeddings.npy", mmap_mode="r")
    quantizer = faiss.IndexFlatIP(vectors.shape[1])
    index = faiss.IndexIVFPQ(quantizer, vectors.shape[1], min(args.nlist, len(vectors)), args.pq_code_size, 8, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(len(vectors), size=min(1_000_000, len(vectors)), replace=False)
    index.train(np.asarray(vectors[sample_ids], dtype=np.float32))
    for start, end in batches(len(vectors), 100_000):
        index.add(np.asarray(vectors[start:end], dtype=np.float32))
    index.nprobe = args.nprobe
    faiss.write_index(index, str(root / "faiss.index"))


def _save_part(root: Path, split: str, part: int, rows: list[dict]) -> None:
    path = root / "shards/shard-00000-of-00001/prepared_parts" / split / f"part-{part:06d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(path))


def stage_prepare(args: argparse.Namespace) -> None:
    root = Path(args.output)
    datastore = root / "datastore"
    index = faiss.read_index(str(datastore / "faiss.index"))
    index.nprobe = args.nprobe
    encoder = Embedder(args.embedding_model, args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_tokenizer, use_fast=True)
    prepared = root / "prepared/exclude-block"
    split_parts: dict[str, list[str]] = {}
    for split in SPLITS:
        token_blocks = np.load(root / f"tokens/{split}-blocks.npy", mmap_mode="r")
        split_parts[split] = []
        pending: list[dict] = []
        part = 0
        for block_start, block_end in batches(len(token_blocks), max(1, args.batch_size // 64)):
            current = token_blocks[block_start:block_end]
            chunks = current.reshape(-1, 64, 16)
            query_chunks = chunks[:, :-1].reshape(-1, 16)
            query_vectors = encoder(
                tokenizer.batch_decode(query_chunks, skip_special_tokens=False), is_query=True
            )
            scores, ids = index.search(query_vectors, args.candidate_pool)
            scores = scores.reshape(len(current), 63, -1)
            ids = ids.reshape(len(current), 63, -1)
            for local, block_tokens in enumerate(current):
                selected_ids = np.full((64, 16), -1, dtype=np.int64)
                selected_scores = np.zeros((64, 16), dtype=np.float32)
                for query in range(63):
                    candidates = ids[local, query]
                    candidate_scores = scores[local, query]
                    if split == "train":
                        global_block = block_start + local
                        low, high = global_block * 64, (global_block + 1) * 64
                        key_in_block = (candidates >= low) & (candidates < high)
                        future_in_block = (candidates + 1 >= low) & (candidates + 1 < high)
                        keep = ~(key_in_block | future_in_block)
                        candidates, candidate_scores = candidates[keep], candidate_scores[keep]
                    selected_ids[query + 1] = candidates[:16]
                    selected_scores[query + 1] = candidate_scores[:16]
                pending.append({
                    "input_ids": block_tokens.tolist(),
                    "attention_mask": [1] * 1024,
                    "labels": block_tokens.tolist(),
                    "retrieved_chunk_ids": selected_ids.tolist(),
                    "retrieval_scores": selected_scores.tolist(),
                })
                if len(pending) == args.part_rows:
                    _save_part(prepared, split, part, pending)
                    split_parts[split].append(f"shards/shard-00000-of-00001/prepared_parts/{split}/part-{part:06d}")
                    pending, part = [], part + 1
        if pending:
            _save_part(prepared, split, part, pending)
            split_parts[split].append(f"shards/shard-00000-of-00001/prepared_parts/{split}/part-{part:06d}")
    meta = {
        "protocol": "wikitext103_database_construction_v1",
        "packing_mode": "cross_article",
        "exclude_current_block": True,
        "exclude_current_article": False,
        "block_size": 1024,
        "chunk_size": 16,
        "top_k": 16,
        "memory_slots": 32,
        "memory_value_mode": "key_plus_future",
        "split_parts": split_parts,
    }
    (prepared / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    args = arguments()
    stages = ("tokenize", "encode", "index", "prepare") if args.stage == "all" else (args.stage,)
    functions = {"tokenize": stage_tokenize, "encode": stage_encode, "index": stage_index, "prepare": stage_prepare}
    for stage in stages:
        print(f"== stage: {stage} ==", flush=True)
        functions[stage](args)


if __name__ == "__main__":
    main()
