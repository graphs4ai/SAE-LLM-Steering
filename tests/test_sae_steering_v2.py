from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract_activations import _pair_key_from_row_id
from select_sae_features import (
    _open_activations_parquet,
    _parquet_thrift_size_limit,
    _score_paired_contrastive_chunked,
)
from utils.sae_steering import build_normalized_steering_vector, unit_decoder_direction


def test_pair_key_from_row_id() -> None:
    assert _pair_key_from_row_id("anderson_001_left") == "anderson_001"
    assert _pair_key_from_row_id("anderson_001_right") == "anderson_001"
    assert _pair_key_from_row_id("custom_row") == "custom_row"


def test_open_activations_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"layer_0-feature_0": [1.0], "class": ["left"]})
    parquet_path = tmp_path / "acts.parquet"
    df.to_parquet(parquet_path, index=False)

    assert _parquet_thrift_size_limit(str(parquet_path)) == 64 * 1024 * 1024
    pf = _open_activations_parquet(str(parquet_path))
    assert pf.schema.names == ["layer_0-feature_0", "class"]


def test_paired_contrastive_scoring(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "layer_1-feature_0": [3.0, 1.0, 1.0, 2.0],
            "layer_1-feature_1": [0.0, 0.0, 0.0, 0.0],
            "class": ["left", "right", "left", "right"],
            "pair_key": ["p0", "p0", "p1", "p1"],
        }
    )
    parquet_path = tmp_path / "acts.parquet"
    df.to_parquet(parquet_path, index=False)

    scores = _score_paired_contrastive_chunked(
        str(parquet_path),
        ["layer_1-feature_0", "layer_1-feature_1"],
        chunk_size=2,
        min_active_rate=0.0,
    )

    assert scores["layer_1-feature_0"]["selection_score"] == 1.5
    assert scores["layer_1-feature_0"]["mean_left"] == 2.0
    assert scores["layer_1-feature_0"]["mean_right"] == 1.5
    assert scores["layer_1-feature_0"]["signed_delta"] == 0.5
    assert scores["layer_1-feature_0"]["active_rate"] == 1.0
    assert scores["layer_1-feature_1"]["selection_score"] == 0.0
    assert scores["layer_1-feature_1"]["active_rate"] == 0.0


def test_unit_norm_steering_vector() -> None:
    w_dec = torch.tensor(
        [
            [3.0, 4.0],
            [0.0, 2.0],
        ],
        dtype=torch.float32,
    )
    steering = build_normalized_steering_vector(
        w_dec,
        {
            0: 2.0,
            1: -1.0,
        },
    )
    expected = 2.0 * unit_decoder_direction(w_dec[0]) - unit_decoder_direction(w_dec[1])
    assert torch.allclose(steering, expected)
