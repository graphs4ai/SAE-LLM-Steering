"""Stage 2: SAE feature selection.

Ranks SAE latent columns from the activations W&B artifact and emits a
ranked-feature artifact for ``optimize_sae_steering.py``.
"""

from __future__ import annotations

import json
import os
import re
import struct
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


_FEATURE_COL_RE = re.compile(r"^layer_(\d+)-feature_(\d+)$")


def _parse_feature_column(column: str) -> tuple[int, int] | None:
    match = _FEATURE_COL_RE.match(str(column))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _parquet_thrift_size_limit(parquet_path: str) -> int:
    """Return Thrift decode limits large enough for very wide SAE parquet footers.

    Wide activations tables (e.g. 2 x 262k SAE latents) store one parquet column
    per latent. The serialized schema in the file footer can exceed PyArrow's
    default Thrift string/container limits (~100 MiB).
    """
    with open(parquet_path, "rb") as f:
        f.seek(-8, 2)
        footer_len = struct.unpack("<i", f.read(4))[0]
    if footer_len <= 0:
        raise ValueError(f"Invalid parquet footer length in {parquet_path}")
    # Footer size tracks per-column schema metadata; double it for container overhead.
    return max(64 * 1024 * 1024, footer_len * 2)


def _open_activations_parquet(parquet_path: str) -> pq.ParquetFile:
    thrift_limit = _parquet_thrift_size_limit(parquet_path)
    return pq.ParquetFile(
        parquet_path,
        thrift_string_size_limit=thrift_limit,
        thrift_container_size_limit=thrift_limit,
    )


def _default_column_chunk_size(n_rows: int | None, n_cols: int) -> int:
    """Pick a column chunk size that keeps each read under ~256 MiB."""
    rows = max(int(n_rows or 1), 1)
    # float32 bytes per value; leave headroom for Arrow/pandas temporaries.
    max_values = (256 * 1024 * 1024) // 4
    chunk = max_values // rows
    # Wide SAE tables (e.g. 4 x 65k) benefit from large chunks; tiny chunks
    # re-scan parquet metadata/column chunks far too often.
    return int(min(max(chunk, 1024), n_cols, 65_536))


def _score_mean_activation_chunked(
    parquet_path: str,
    feature_columns: list[str],
    chunk_size: int | None = None,
) -> dict[str, float]:
    """Compute mean activation per feature column without loading all columns.

    Activations are stored as a very wide parquet (one column per SAE latent).
    Reusing a single ``ParquetFile`` handle and vectorizing the row-mean over
    each column chunk is much faster than per-column Python loops with small
    chunks.
    """
    pf = _open_activations_parquet(parquet_path)
    n_rows = pf.metadata.num_rows if pf.metadata is not None else None
    if n_rows == 0:
        raise ValueError(f"Activations parquet is empty: {parquet_path}")

    if chunk_size is None:
        chunk_size = _default_column_chunk_size(n_rows, len(feature_columns))
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    scores: dict[str, float] = {}
    n_chunks = (len(feature_columns) + chunk_size - 1) // chunk_size
    print(
        f"Scoring {len(feature_columns)} features in {n_chunks} column-chunks "
        f"(chunk_size={chunk_size}, n_rows={n_rows})"
    )

    for start in range(0, len(feature_columns), chunk_size):
        cols = feature_columns[start:start + chunk_size]
        # Reuse the open ParquetFile so footer/schema are not re-parsed.
        table = pf.read(columns=cols)
        # (n_rows, n_cols) float array; one reduction instead of per-column means.
        arr = np.column_stack([
            table.column(col).to_numpy(zero_copy_only=False)
            for col in cols
        ])
        means = arr.mean(axis=0)
        scores.update(zip(cols, map(float, means)))
        del table, arr

    return scores


def _score_paired_contrastive_chunked(
    parquet_path: str,
    feature_columns: list[str],
    chunk_size: int | None = None,
    min_active_rate: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Score features by paired left/right activation contrast."""
    pf = _open_activations_parquet(parquet_path)
    n_rows = pf.metadata.num_rows if pf.metadata is not None else None
    if n_rows == 0:
        raise ValueError(f"Activations parquet is empty: {parquet_path}")

    if chunk_size is None:
        chunk_size = _default_column_chunk_size(n_rows, len(feature_columns))
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    metadata_cols = {"class", "pair_key"}
    schema_names = set(pf.schema.names)
    missing = metadata_cols - schema_names
    if missing:
        raise ValueError(
            f"Paired contrastive selection requires parquet metadata columns "
            f"{sorted(metadata_cols)}, missing {sorted(missing)}."
        )

    meta_table = pf.read(columns=["class", "pair_key"])
    meta_df = meta_table.to_pandas()
    meta_df["class"] = meta_df["class"].astype(str)
    meta_df["pair_key"] = meta_df["pair_key"].astype(str)

    left_mask = meta_df["class"].eq("left").to_numpy()
    right_mask = meta_df["class"].eq("right").to_numpy()
    if not bool(left_mask.any()) or not bool(right_mask.any()):
        raise ValueError("Expected both left and right rows in feature-selection parquet.")

    pair_order = sorted(meta_df["pair_key"].unique())
    left_index = {pair: idx for idx, pair in enumerate(meta_df.loc[left_mask, "pair_key"])}
    right_index = {pair: idx for idx, pair in enumerate(meta_df.loc[right_mask, "pair_key"])}
    shared_pairs = [pair for pair in pair_order if pair in left_index and pair in right_index]
    if not shared_pairs:
        raise ValueError("No shared left/right pair_key values found for paired contrastive selection.")

    left_positions = np.array([left_index[pair] for pair in shared_pairs], dtype=np.int64)
    right_positions = np.array([right_index[pair] for pair in shared_pairs], dtype=np.int64)

    scores: dict[str, dict[str, float]] = {}
    n_chunks = (len(feature_columns) + chunk_size - 1) // chunk_size
    print(
        f"Scoring {len(feature_columns)} features with paired contrastive metric "
        f"in {n_chunks} column-chunks (chunk_size={chunk_size}, n_rows={n_rows})"
    )

    for start in range(0, len(feature_columns), chunk_size):
        cols = feature_columns[start:start + chunk_size]
        table = pf.read(columns=cols)
        arr = np.column_stack([
            table.column(col).to_numpy(zero_copy_only=False)
            for col in cols
        ]).astype(np.float32, copy=False)

        left_arr = arr[left_mask][left_positions]
        right_arr = arr[right_mask][right_positions]
        pair_deltas = left_arr - right_arr
        selection_scores = np.mean(np.abs(pair_deltas), axis=0)
        mean_left = np.mean(arr[left_mask], axis=0)
        mean_right = np.mean(arr[right_mask], axis=0)
        signed_delta = np.mean(pair_deltas, axis=0)
        active_rates = np.mean(arr > 0, axis=0)

        for idx, col in enumerate(cols):
            active_rate = float(active_rates[idx])
            if active_rate < min_active_rate:
                continue
            scores[col] = {
                "selection_score": float(selection_scores[idx]),
                "mean_left": float(mean_left[idx]),
                "mean_right": float(mean_right[idx]),
                "signed_delta": float(signed_delta[idx]),
                "active_rate": active_rate,
            }
        del table, arr, left_arr, right_arr, pair_deltas

    return scores


def _find_parquet(artifact_dir: str) -> str:
    for root, _dirs, files in os.walk(artifact_dir):
        for name in files:
            if name.endswith(".parquet"):
                return os.path.join(root, name)
    raise FileNotFoundError(
        f"No parquet file found in activations artifact dir: {artifact_dir}"
    )


def _decoder_norms_for_ranked_features(
    cfg: DictConfig,
    ranked_features: list[dict[str, Any]],
) -> dict[str, float]:
    if not ranked_features:
        return {}

    from model_factory import get_model_wrapper

    layers = sorted({int(entry["layer"]) for entry in ranked_features})
    wrapper = get_model_wrapper(cfg, device="cpu")
    if not hasattr(wrapper, "load_saes") or not hasattr(wrapper, "saes"):
        return {}
    wrapper.load_saes(layers)

    norms: dict[str, float] = {}
    for entry in ranked_features:
        layer = int(entry["layer"])
        feature = int(entry["feature"])
        sae = wrapper.saes[layer]
        norms[str(entry["feature_name"])] = float(
            sae.W_dec[feature].float().norm().item()
        )
    return norms


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    from utils.seeds import (
        log_resolved_seeds,
        resolve_seeds_from_cfg,
        resolved_seeds_to_dict,
    )

    resolved = resolve_seeds_from_cfg(cfg)
    log_resolved_seeds(resolved, prefix="select_sae_features")
    fs_seed = resolved.feature_selection

    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir

    wandb_cfg = cfg.get("wandb", {}) or {}
    fs_cfg = cfg.get("feature_selection", {}) or {}
    method = str(fs_cfg.get("method", "mean_activation"))
    min_active_rate = float(fs_cfg.get("min_active_rate", 0.0))
    ranking_top_n = int(fs_cfg.get("ranking_top_n", 256))
    if ranking_top_n <= 0:
        raise ValueError(
            f"feature_selection.ranking_top_n must be positive, got {ranking_top_n}"
        )

    activations_artifact_name = cfg.data.get("activations_artifact_name", None)
    if not activations_artifact_name:
        raise ValueError(
            "data.activations_artifact_name must be set "
            "(W&B activations artifact from extract_activations)."
        )

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(wandb_config, dict):
        wandb_config["resolved_seeds"] = resolved_seeds_to_dict(resolved)

    wandb.init(
        project=wandb_cfg.get("project", "SAE-LLM-Steering"),
        name=wandb_cfg.get("run_name", None),
        job_type="feature_selection",
        config=wandb_config,
    )

    print(f"Downloading activations artifact: {activations_artifact_name}")
    artifact = wandb.use_artifact(activations_artifact_name, type="dataset")
    artifact_dir = artifact.download()
    parquet_path = _find_parquet(artifact_dir)
    print(f"Using activations parquet: {parquet_path}")

    pf = _open_activations_parquet(parquet_path)
    feature_columns = [
        name for name in pf.schema.names if _parse_feature_column(name) is not None
    ]
    if not feature_columns:
        raise ValueError(
            f"No layer_L-feature_F columns found in {parquet_path}"
        )
    print(f"Found {len(feature_columns)} SAE feature columns")

    if method == "paired_contrastive":
        scored_features = _score_paired_contrastive_chunked(
            parquet_path,
            feature_columns,
            min_active_rate=min_active_rate,
        )
        ranked = sorted(
            scored_features.items(),
            key=lambda item: item[1]["selection_score"],
            reverse=True,
        )
    elif method == "mean_activation":
        scores = _score_mean_activation_chunked(parquet_path, feature_columns)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        scored_features = {
            name: {
                "selection_score": float(score),
                "mean_left": None,
                "mean_right": None,
                "signed_delta": None,
                "active_rate": None,
            }
            for name, score in scores.items()
        }
    else:
        raise ValueError(
            f"Unknown feature_selection.method={method!r}. "
            "Expected 'paired_contrastive' or 'mean_activation'."
        )

    if ranking_top_n > len(ranked):
        print(
            f"Warning: ranking_top_n={ranking_top_n} exceeds available "
            f"features ({len(ranked)}); using all features."
        )
        ranking_top_n_effective = len(ranked)
    else:
        ranking_top_n_effective = ranking_top_n

    top_ranked = ranked[:ranking_top_n_effective]

    ranked_features: list[dict[str, Any]] = []
    for rank, (feature_name, stats) in enumerate(top_ranked, start=1):
        parsed = _parse_feature_column(feature_name)
        assert parsed is not None
        layer_idx, feature_idx = parsed
        ranked_features.append({
            "rank": rank,
            "layer": layer_idx,
            "feature": feature_idx,
            "feature_name": feature_name,
            "score": stats["selection_score"],
            "selection_score": stats["selection_score"],
            "mean_left": stats["mean_left"],
            "mean_right": stats["mean_right"],
            "signed_delta": stats["signed_delta"],
            "active_rate": stats["active_rate"],
            "selection_frequency": None,
            "selection_count": None,
        })

    decoder_norms = _decoder_norms_for_ranked_features(cfg, ranked_features)
    for entry in ranked_features:
        entry["decoder_norm"] = decoder_norms.get(entry["feature_name"])

    model_name = cfg.model.name.split("/")[-1]
    split_id = cfg.data.get("split_id", None)
    feature_selection_dataset = cfg.data.get("feature_selection_dataset", None)
    prompt_mode = str(cfg.get("extraction", {}).get("prompt_mode", "bare_statement"))

    ranking_payload = {
        "model_name": model_name,
        "split_id": split_id,
        "feature_selection_dataset": feature_selection_dataset,
        "activations_artifact_name": activations_artifact_name,
        "method": method,
        "ranking_top_n": ranking_top_n_effective,
        "seed": fs_seed,
        "selection_layers": None,
        "hook_site": "resid_post",
        "prompt_mode": prompt_mode,
        "decoder_normalization": "unit_norm",
        "coefficient_type": "beta",
        "edit_mode": "additive_latent_delta",
        "ranked_features": ranked_features,
    }

    feature_ranking_json_path = os.path.join(run_dir, "feature_ranking.json")
    with open(feature_ranking_json_path, "w", encoding="utf-8") as f:
        json.dump(ranking_payload, f, indent=2, ensure_ascii=False)
    print(f"Feature ranking JSON saved to: {feature_ranking_json_path}")

    feature_ranking_csv_path = os.path.join(run_dir, "feature_ranking.csv")
    pd.DataFrame(ranked_features).to_csv(feature_ranking_csv_path, index=False)
    print(f"Feature ranking CSV saved to: {feature_ranking_csv_path}")

    artifacts_cfg = cfg.get("artifacts", {}) or {}
    feature_ranking_override = artifacts_cfg.get("feature_ranking_name", None)
    if feature_ranking_override:
        feature_artifact_name_out = str(feature_ranking_override)
    else:
        feature_artifact_name_out = (
            f"feature-ranking-{model_name}-"
            f"{split_id or 'nosplit'}-top{ranking_top_n_effective}"
        )

    feature_artifact = wandb.Artifact(
        name=feature_artifact_name_out,
        type="dataset",
        description=(
            "SAE feature ranking on the feature-selection activations artifact"
        ),
        metadata={
            "n_features": ranking_top_n_effective,
            "method": method,
            "model_name": model_name,
            "split_id": split_id,
            "feature_selection_dataset": feature_selection_dataset,
            "activations_artifact_name": activations_artifact_name,
            "ranking_top_n": ranking_top_n_effective,
            "seed": fs_seed,
            "hook_site": "resid_post",
            "prompt_mode": prompt_mode,
            "decoder_normalization": "unit_norm",
            "coefficient_type": "beta",
            "edit_mode": "additive_latent_delta",
        },
    )
    feature_artifact.add_file(feature_ranking_json_path)
    feature_artifact.add_file(feature_ranking_csv_path)
    wandb.log_artifact(feature_artifact)
    print(f"Feature ranking artifact logged: {feature_artifact_name_out}")

    wandb.summary.update({
        "n_features": ranking_top_n_effective,
        "n_candidate_features": len(feature_columns),
        "method": method,
        "top_feature": ranked_features[0]["feature_name"] if ranked_features else None,
        "top_score": ranked_features[0]["selection_score"] if ranked_features else None,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
