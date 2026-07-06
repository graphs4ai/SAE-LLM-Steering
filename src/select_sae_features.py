"""Stage 2: SAE feature selection by mean activation.

Ranks SAE latent columns from the activations W&B artifact by mean activation
over the feature-selection dataset and emits a ranked-feature artifact for
``optimize_sae_steering.py``.
"""

from __future__ import annotations

import json
import os
import re
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
    pf = pq.ParquetFile(parquet_path)
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


def _find_parquet(artifact_dir: str) -> str:
    for root, _dirs, files in os.walk(artifact_dir):
        for name in files:
            if name.endswith(".parquet"):
                return os.path.join(root, name)
    raise FileNotFoundError(
        f"No parquet file found in activations artifact dir: {artifact_dir}"
    )


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

    schema = pq.read_schema(parquet_path)
    feature_columns = [
        name for name in schema.names if _parse_feature_column(name) is not None
    ]
    if not feature_columns:
        raise ValueError(
            f"No layer_L-feature_F columns found in {parquet_path}"
        )
    print(f"Found {len(feature_columns)} SAE feature columns")

    scores = _score_mean_activation_chunked(parquet_path, feature_columns)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
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
    for rank, (feature_name, score) in enumerate(top_ranked, start=1):
        parsed = _parse_feature_column(feature_name)
        assert parsed is not None
        layer_idx, feature_idx = parsed
        ranked_features.append({
            "rank": rank,
            "layer": layer_idx,
            "feature": feature_idx,
            "feature_name": feature_name,
            "score": score,
            "selection_frequency": None,
            "selection_count": None,
        })

    model_name = cfg.model.name.split("/")[-1]
    split_id = cfg.data.get("split_id", None)
    feature_selection_dataset = cfg.data.get("feature_selection_dataset", None)

    ranking_payload = {
        "model_name": model_name,
        "split_id": split_id,
        "feature_selection_dataset": feature_selection_dataset,
        "activations_artifact_name": activations_artifact_name,
        "method": "mean_activation",
        "ranking_top_n": ranking_top_n_effective,
        "seed": fs_seed,
        "selection_layers": None,
        "hook_site": "resid_post",
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
            "SAE feature ranking by mean activation on the "
            "feature-selection activations artifact"
        ),
        metadata={
            "n_features": ranking_top_n_effective,
            "method": "mean_activation",
            "model_name": model_name,
            "split_id": split_id,
            "feature_selection_dataset": feature_selection_dataset,
            "activations_artifact_name": activations_artifact_name,
            "ranking_top_n": ranking_top_n_effective,
            "seed": fs_seed,
            "hook_site": "resid_post",
        },
    )
    feature_artifact.add_file(feature_ranking_json_path)
    feature_artifact.add_file(feature_ranking_csv_path)
    wandb.log_artifact(feature_artifact)
    print(f"Feature ranking artifact logged: {feature_artifact_name_out}")

    wandb.summary.update({
        "n_features": ranking_top_n_effective,
        "n_candidate_features": len(feature_columns),
        "method": "mean_activation",
        "top_feature": ranked_features[0]["feature_name"] if ranked_features else None,
        "top_score": ranked_features[0]["score"] if ranked_features else None,
    })
    wandb.finish()


if __name__ == "__main__":
    main()
