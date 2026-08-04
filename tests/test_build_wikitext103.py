from __future__ import annotations

import argparse

from datasets import Dataset

from scripts.build_wikitext103 import block_records


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, texts: list[str], add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [[index + 1, index + 1] for index, _ in enumerate(texts)]}


def test_article_training_keeps_cross_article_eval_targets() -> None:
    split = Dataset.from_dict(
        {"text": [" = Article A = ", "alpha", " = Article B = ", "beta"]}
    )
    args = argparse.Namespace(block_size=4, chunk_size=2, packing_mode="article_only")
    article_rows = list(block_records(split, FakeTokenizer(), args))
    target_rows = list(
        block_records(split, FakeTokenizer(), args, packing_mode="cross_article")
    )

    assert [int(mask.sum()) for _, mask, _ in article_rows] == [4, 2, 4, 2]
    assert [int(mask.sum()) for _, mask, _ in target_rows] == [4, 4, 4]
