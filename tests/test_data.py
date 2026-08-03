import json

import numpy as np
import pytest
import torch

from dfm.data import Datastore, validate_data_contract


def make_datastore(tmp_path):
    np.save(tmp_path / "embeddings.npy", np.arange(12, dtype=np.float16).reshape(3, 4))
    np.save(tmp_path / "future.npy", np.arange(12, 24, dtype=np.float16).reshape(3, 4))
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "num_chunks": 3,
                "embedding_dim": 4,
                "embeddings_file": "embeddings.npy",
                "future_embeddings_file": "future.npy",
                "chunk_size": 2,
            }
        )
    )
    return Datastore(tmp_path)


@pytest.mark.parametrize(
    ("mode", "slots"),
    [("key", 2), ("continuation", 2), ("key_plus_continuation", 4)],
)
def test_datastore_value_modes(tmp_path, mode: str, slots: int) -> None:
    values, mask = make_datastore(tmp_path).lookup(torch.tensor([[[0, -1]]]), mode)
    assert values.shape == (1, 1, slots, 4)
    assert mask.sum().item() == (1 if mode != "key_plus_continuation" else 2)


def test_data_contract_allows_topk_prefix(tmp_path) -> None:
    datastore_root = tmp_path / "datastore"
    datastore_root.mkdir()
    make_datastore(datastore_root)
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    (prepared_root / "meta.json").write_text(json.dumps({"chunk_size": 2, "top_k": 8}))
    validate_data_contract(
        prepared_root,
        datastore_root,
        chunk_size=2,
        top_k=4,
        memory_dim=4,
        value_mode="key_plus_continuation",
    )
