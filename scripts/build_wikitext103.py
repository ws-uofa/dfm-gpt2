#!/usr/bin/env python
"""Build an ablatable WikiText-103 datastore used by this repository.

The script is stage based so a full build can be resumed. CLI defaults reproduce
the best audited database-construction-v1 setting; every experimental choice is
explicitly overrideable.
"""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
import re
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
    parser.add_argument("--packing-mode", choices=("cross_article", "article_only"), default="cross_article")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--continuation", choices=("none", "append", "only"), default="append")
    parser.add_argument("--exclude-current-block", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-current-article", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-pool", type=int, default=2048)
    parser.add_argument("--index-type", choices=("flat", "ivf_flat", "ivf_pq"), default="ivf_pq")
    parser.add_argument("--metric", choices=("inner_product", "l2"), default="inner_product")
    parser.add_argument("--nlist", type=int, default=10942)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--pq-code-size", type=int, default=64)
    parser.add_argument("--pq-nbits", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=64)
    parser.add_argument("--prepared-name", default="exclude-block")
    parser.add_argument("--part-rows", type=int, default=128)
    parser.add_argument("--max-blocks", type=int, help="Small deterministic smoke build")
    return parser.parse_args()


def load_wikitext(path: str) -> DatasetDict:
    source = Path(path)
    if source.exists():
        if (source / "dataset_dict.json").exists() or (source / "state.json").exists():
            value = load_from_disk(str(source))
        else:
            files = {
                split: [str(item) for item in sorted(source.glob(f"{split}-*.parquet"))]
                for split in SPLITS
            }
            if any(not items for items in files.values()):
                raise FileNotFoundError(f"local WikiText directory lacks parquet splits: {source}")
            value = load_dataset("parquet", data_files=files)
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


TOP_HEADING = re.compile(r"^\s*=\s+[^=].*?\s+=\s*$")


def article_stream(split: Dataset, tokenizer, batch_size: int = 1024) -> Iterator[tuple[int, list[int]]]:
    """Regroup WikiText raw rows into top-level articles, preserving every row EOS."""

    article_id = 0
    started = False
    current: list[int] = []
    eos = tokenizer.eos_token_id
    for start in range(0, len(split), batch_size):
        texts = split[start : start + batch_size]["text"]
        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for text, ids in zip(texts, encoded):
            if TOP_HEADING.match(text):
                if started and current:
                    yield article_id, current
                    current = []
                    article_id += 1
                started = True
            current.extend(ids)
            current.append(eos)
    if current:
        yield article_id, current


def blocks(stream: Iterable[int], size: int = 1024) -> Iterator[np.ndarray]:
    buffer: list[int] = []
    for token in stream:
        buffer.append(token)
        if len(buffer) == size:
            yield np.asarray(buffer, dtype=np.int32)
            buffer.clear()


def block_records(split: Dataset, tokenizer, args: argparse.Namespace):
    """Yield tokens, mask, and per-chunk article IDs for either packing policy."""

    chunks_per_block = args.block_size // args.chunk_size
    if args.packing_mode == "cross_article":
        token_buffer: list[int] = []
        article_buffer: list[int] = []
        for article_id, article_tokens in article_stream(split, tokenizer):
            token_buffer.extend(article_tokens)
            article_buffer.extend([article_id] * len(article_tokens))
            while len(token_buffer) >= args.block_size:
                tokens = np.asarray(token_buffer[: args.block_size], dtype=np.int32)
                articles = np.asarray(article_buffer[: args.block_size], dtype=np.int64)
                del token_buffer[: args.block_size]
                del article_buffer[: args.block_size]
                by_chunk = articles.reshape(chunks_per_block, args.chunk_size)
                chunk_articles = np.stack((by_chunk.min(axis=1), by_chunk.max(axis=1)), axis=-1)
                yield tokens, np.ones(args.block_size, dtype=np.int8), chunk_articles
        return

    eos = tokenizer.eos_token_id
    for article_id, article_tokens in article_stream(split, tokenizer):
        for start in range(0, len(article_tokens), args.block_size):
            content = article_tokens[start : start + args.block_size]
            valid = len(content)
            tokens = np.full(args.block_size, eos, dtype=np.int32)
            mask = np.zeros(args.block_size, dtype=np.int8)
            tokens[:valid] = content
            mask[:valid] = 1
            chunk_articles = np.full((chunks_per_block, 2), -1, dtype=np.int64)
            full_chunks = valid // args.chunk_size
            chunk_articles[:full_chunks, :] = article_id
            yield tokens, mask, chunk_articles


def pair_records(split: Dataset, tokenizer, args: argparse.Namespace):
    """Yield adjacent full chunks plus provenance used by exclusion ablations."""

    chunks_per_block = args.block_size // args.chunk_size
    if args.packing_mode == "cross_article":
        token_buffer: list[int] = []
        article_buffer: list[int] = []
        previous = None
        chunk_index = 0
        for article_id, article_tokens in article_stream(split, tokenizer):
            token_buffer.extend(article_tokens)
            article_buffer.extend([article_id] * len(article_tokens))
            while len(token_buffer) >= args.chunk_size:
                chunk = np.asarray(token_buffer[: args.chunk_size], dtype=np.int32)
                chunk_articles = article_buffer[: args.chunk_size]
                del token_buffer[: args.chunk_size]
                del article_buffer[: args.chunk_size]
                current = (
                    chunk,
                    chunk_index // chunks_per_block,
                    min(chunk_articles),
                    max(chunk_articles),
                )
                if previous is not None:
                    yield (
                        previous[0], chunk, previous[1], current[1],
                        previous[2], previous[3], current[2], current[3],
                    )
                previous = current
                chunk_index += 1
        return

    global_block = 0
    for article_id, article_tokens in article_stream(split, tokenizer):
        previous = None
        full_chunks = len(article_tokens) // args.chunk_size
        for chunk_index in range(full_chunks):
            start = chunk_index * args.chunk_size
            chunk = np.asarray(article_tokens[start : start + args.chunk_size], dtype=np.int32)
            block_id = global_block + chunk_index // chunks_per_block
            current = (chunk, block_id, article_id, article_id)
            if previous is not None:
                yield (
                    previous[0], chunk, previous[1], block_id,
                    previous[2], previous[3], article_id, article_id,
                )
            previous = current
        global_block += (len(article_tokens) + args.block_size - 1) // args.block_size


def stage_tokenize(args: argparse.Namespace) -> None:
    root = Path(args.output)
    tokens_root = root / "tokens"
    if (tokens_root / "meta.json").exists():
        raise FileExistsError(f"tokenize stage already exists: {tokens_root}")
    tokens_root.mkdir(parents=True, exist_ok=True)
    dataset = load_wikitext(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_tokenizer, use_fast=True)
    if args.block_size <= 0 or args.chunk_size <= 0 or args.block_size % args.chunk_size:
        raise ValueError("block_size must be a positive multiple of chunk_size")
    counts = {}
    for split in SPLITS:
        iterator = block_records(dataset[split], tokenizer, args)
        rows = list(islice(iterator, args.max_blocks)) if args.max_blocks else None
        count = len(rows) if rows is not None else sum(1 for _ in block_records(dataset[split], tokenizer, args))
        tokens = np.lib.format.open_memmap(tokens_root / f"{split}-blocks.npy", mode="w+", dtype=np.int32, shape=(count, args.block_size))
        masks = np.lib.format.open_memmap(tokens_root / f"{split}-masks.npy", mode="w+", dtype=np.int8, shape=(count, args.block_size))
        articles = np.lib.format.open_memmap(tokens_root / f"{split}-chunk-articles.npy", mode="w+", dtype=np.int64, shape=(count, args.block_size // args.chunk_size, 2))
        source = rows if rows is not None else block_records(dataset[split], tokenizer, args)
        for index, (row_tokens, row_mask, row_articles) in enumerate(source):
            tokens[index], masks[index], articles[index] = row_tokens, row_mask, row_articles
        tokens.flush(); masks.flush(); articles.flush()
        counts[split] = count

    pairs = pair_records(dataset["train"], tokenizer, args)
    if args.max_blocks:
        pairs = islice(pairs, args.max_blocks * (args.block_size // args.chunk_size))
        collected_pairs = list(pairs)
        pair_count = len(collected_pairs)
    else:
        collected_pairs = None
        pair_count = sum(1 for _ in pair_records(dataset["train"], tokenizer, args))
    fields = {
        "train-key-tokens.npy": (np.int32, (pair_count, args.chunk_size)),
        "train-future-tokens.npy": (np.int32, (pair_count, args.chunk_size)),
        "pair-key-block.npy": (np.int64, (pair_count,)),
        "pair-future-block.npy": (np.int64, (pair_count,)),
        "pair-key-article-first.npy": (np.int64, (pair_count,)),
        "pair-key-article-last.npy": (np.int64, (pair_count,)),
        "pair-future-article-first.npy": (np.int64, (pair_count,)),
        "pair-future-article-last.npy": (np.int64, (pair_count,)),
    }
    arrays = {name: np.lib.format.open_memmap(tokens_root / name, mode="w+", dtype=dtype, shape=shape) for name, (dtype, shape) in fields.items()}
    source_pairs = collected_pairs if collected_pairs is not None else pair_records(dataset["train"], tokenizer, args)
    for index, values in enumerate(source_pairs):
        for name, value in zip(fields, values):
            arrays[name][index] = value
    for array in arrays.values(): array.flush()
    (tokens_root / "meta.json").write_text(json.dumps({"packing_mode": args.packing_mode, "block_size": args.block_size, "chunk_size": args.chunk_size, "pair_count": pair_count, "counts": counts}, indent=2) + "\n")


class Embedder:
    def __init__(self, name: str, device: str, max_length: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.max_length = max_length
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def __call__(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if is_query:
            instruction = "Instruct: Given a text chunk, retrieve relevant memory chunks\nQuery:"
            texts = [f"{instruction}\n{text}" for text in texts]
        batch = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        hidden = self.model(**batch).last_hidden_state
        # Padding is on the left, so the final position is always the last token.
        pooled = hidden[:, -1]
        return torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy()


def batches(total: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, size):
        yield start, min(start + size, total)


def stage_encode(args: argparse.Namespace) -> None:
    root = Path(args.output)
    keys_tokens = np.load(root / "tokens/train-key-tokens.npy", mmap_mode="r")
    future_tokens = np.load(root / "tokens/train-future-tokens.npy", mmap_mode="r")
    pair_count = len(keys_tokens)
    encoder = Embedder(args.embedding_model, args.device, args.embedding_max_length)
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_tokenizer, use_fast=True)
    datastore = root / "datastore"
    if (datastore / "meta.json").exists():
        raise FileExistsError(f"encode stage already exists: {datastore}")
    datastore.mkdir(parents=True, exist_ok=True)
    keys = np.lib.format.open_memmap(datastore / "embeddings.npy", mode="w+", dtype=np.float16, shape=(pair_count, encoder.dim))
    future = None
    if args.continuation != "none":
        future = np.lib.format.open_memmap(datastore / "future_embeddings.npy", mode="w+", dtype=np.float16, shape=(pair_count, encoder.dim))
    for start, end in batches(pair_count, args.batch_size):
        keys[start:end] = encoder(tokenizer.batch_decode(keys_tokens[start:end], skip_special_tokens=False)).astype(np.float16)
        if future is not None:
            future[start:end] = encoder(tokenizer.batch_decode(future_tokens[start:end], skip_special_tokens=False)).astype(np.float16)
        keys.flush()
        if future is not None: future.flush()
    meta = {
        "protocol": "wikitext103_database_construction_v1",
        "packing_mode": args.packing_mode,
        "num_chunks": pair_count,
        "embedding_dim": encoder.dim,
        "embeddings_file": "embeddings.npy",
        "future_embeddings_file": "future_embeddings.npy" if future is not None else None,
        "chunk_size": args.chunk_size,
        "continuation": args.continuation,
        "key_block_ids_file": "../tokens/pair-key-block.npy",
        "future_block_ids_file": "../tokens/pair-future-block.npy",
        "key_article_first_file": "../tokens/pair-key-article-first.npy",
        "key_article_last_file": "../tokens/pair-key-article-last.npy",
        "future_article_first_file": "../tokens/pair-future-article-first.npy",
        "future_article_last_file": "../tokens/pair-future-article-last.npy",
    }
    (datastore / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def stage_index(args: argparse.Namespace) -> None:
    root = Path(args.output) / "datastore"
    if (root / "faiss.index").exists():
        raise FileExistsError(f"index stage already exists: {root / 'faiss.index'}")
    vectors = np.load(root / "embeddings.npy", mmap_mode="r")
    metric = faiss.METRIC_INNER_PRODUCT if args.metric == "inner_product" else faiss.METRIC_L2
    quantizer = faiss.IndexFlatIP(vectors.shape[1]) if args.metric == "inner_product" else faiss.IndexFlatL2(vectors.shape[1])
    if args.index_type == "flat":
        index = quantizer
    elif args.index_type == "ivf_flat":
        index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], min(args.nlist, len(vectors)), metric)
    else:
        index = faiss.IndexIVFPQ(quantizer, vectors.shape[1], min(args.nlist, len(vectors)), args.pq_code_size, args.pq_nbits, metric)
    if not index.is_trained:
        rng = np.random.default_rng(42)
        sample_ids = rng.choice(len(vectors), size=min(1_000_000, len(vectors)), replace=False)
        index.train(np.asarray(vectors[sample_ids], dtype=np.float32))
    for start, end in batches(len(vectors), 100_000):
        index.add(np.asarray(vectors[start:end], dtype=np.float32))
    if hasattr(index, "nprobe"): index.nprobe = args.nprobe
    faiss.write_index(index, str(root / "faiss.index"))
    meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.update({
        "index_file": "faiss.index",
        "index_type": args.index_type,
        "metric": args.metric,
        "nlist": args.nlist if args.index_type != "flat" else None,
        "nprobe": args.nprobe if args.index_type != "flat" else None,
        "pq_code_size": args.pq_code_size if args.index_type == "ivf_pq" else None,
        "pq_nbits": args.pq_nbits if args.index_type == "ivf_pq" else None,
    })
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def _save_part(root: Path, split: str, part: int, rows: list[dict]) -> None:
    path = root / "shards/shard-00000-of-00001/prepared_parts" / split / f"part-{part:06d}"
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).save_to_disk(str(path))


def stage_prepare(args: argparse.Namespace) -> None:
    root = Path(args.output)
    datastore = root / "datastore"
    index = faiss.read_index(str(datastore / "faiss.index"))
    if hasattr(index, "nprobe"): index.nprobe = args.nprobe
    encoder = Embedder(args.embedding_model, args.device, args.embedding_max_length)
    tokenizer = AutoTokenizer.from_pretrained(args.gpt2_tokenizer, use_fast=True)
    prepared = root / f"prepared/{args.prepared_name}"
    if (prepared / "meta.json").exists():
        raise FileExistsError(f"prepare stage already exists: {prepared}")
    chunks_per_block = args.block_size // args.chunk_size
    key_blocks = np.load(root / "tokens/pair-key-block.npy", mmap_mode="r")
    future_blocks = np.load(root / "tokens/pair-future-block.npy", mmap_mode="r")
    key_article_first = np.load(root / "tokens/pair-key-article-first.npy", mmap_mode="r")
    key_article_last = np.load(root / "tokens/pair-key-article-last.npy", mmap_mode="r")
    future_article_first = np.load(root / "tokens/pair-future-article-first.npy", mmap_mode="r")
    future_article_last = np.load(root / "tokens/pair-future-article-last.npy", mmap_mode="r")
    split_parts: dict[str, list[str]] = {}
    for split in SPLITS:
        token_blocks = np.load(root / f"tokens/{split}-blocks.npy", mmap_mode="r")
        block_masks = np.load(root / f"tokens/{split}-masks.npy", mmap_mode="r")
        block_articles = np.load(root / f"tokens/{split}-chunk-articles.npy", mmap_mode="r")
        split_parts[split] = []
        pending: list[dict] = []
        part = 0
        for block_start, block_end in batches(len(token_blocks), max(1, args.batch_size // chunks_per_block)):
            current = token_blocks[block_start:block_end]
            current_masks = block_masks[block_start:block_end]
            current_articles = block_articles[block_start:block_end]
            chunks = current.reshape(-1, chunks_per_block, args.chunk_size)
            query_chunks = chunks[:, :-1].reshape(-1, args.chunk_size)
            query_vectors = encoder(
                tokenizer.batch_decode(query_chunks, skip_special_tokens=False), is_query=True
            )
            scores, ids = index.search(query_vectors, args.candidate_pool)
            scores = scores.reshape(len(current), chunks_per_block - 1, -1)
            ids = ids.reshape(len(current), chunks_per_block - 1, -1)
            for local, block_tokens in enumerate(current):
                selected_ids = np.full((chunks_per_block, args.top_k), -1, dtype=np.int64)
                selected_scores = np.zeros((chunks_per_block, args.top_k), dtype=np.float32)
                valid_chunks = current_masks[local].reshape(chunks_per_block, args.chunk_size).all(axis=1)
                for query in range(chunks_per_block - 1):
                    if not valid_chunks[query] or not valid_chunks[query + 1]:
                        continue
                    candidates = ids[local, query]
                    candidate_scores = scores[local, query]
                    present = candidates >= 0
                    candidates, candidate_scores = candidates[present], candidate_scores[present]
                    if split == "train":
                        global_block = block_start + local
                        excluded = np.zeros_like(candidates, dtype=bool)
                        if args.exclude_current_block:
                            excluded |= key_blocks[candidates] == global_block
                            excluded |= future_blocks[candidates] == global_block
                        if args.exclude_current_article:
                            query_first, query_last = current_articles[local, query]
                            key_overlap = (key_article_first[candidates] <= query_last) & (key_article_last[candidates] >= query_first)
                            future_overlap = (future_article_first[candidates] <= query_last) & (future_article_last[candidates] >= query_first)
                            excluded |= key_overlap | future_overlap
                        keep = ~excluded
                        candidates, candidate_scores = candidates[keep], candidate_scores[keep]
                    if len(candidates) < args.top_k:
                        raise RuntimeError("candidate pool underfilled after exclusions")
                    selected_ids[query + 1] = candidates[: args.top_k]
                    selected_scores[query + 1] = candidate_scores[: args.top_k]
                pending.append({
                    "input_ids": block_tokens.tolist(),
                    "attention_mask": current_masks[local].tolist(),
                    "labels": np.where(current_masks[local], block_tokens, -100).tolist(),
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
        "packing_mode": args.packing_mode,
        "exclude_current_block": args.exclude_current_block,
        "exclude_current_article": args.exclude_current_article,
        "block_size": args.block_size,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "continuation": args.continuation,
        "memory_slots": args.top_k * (2 if args.continuation == "append" else 1),
        "memory_value_mode": {"none": "key", "append": "key_plus_continuation", "only": "continuation"}[args.continuation],
        "split_parts": split_parts,
    }
    (prepared / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def main() -> None:
    args = arguments()
    if args.candidate_pool < args.top_k:
        raise ValueError("candidate_pool must be at least top_k")
    if args.block_size % args.chunk_size:
        raise ValueError("block_size must be divisible by chunk_size")
    stages = ("tokenize", "encode", "index", "prepare") if args.stage == "all" else (args.stage,)
    functions = {"tokenize": stage_tokenize, "encode": stage_encode, "index": stage_index, "prepare": stage_prepare}
    for stage in stages:
        print(f"== stage: {stage} ==", flush=True)
        functions[stage](args)


if __name__ == "__main__":
    main()
