from __future__ import annotations

from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.weight_cache import (  # noqa: E402
    WeightAsset,
    check_model_repo,
    check_sae_folder,
    collect_pipeline_weight_assets,
    format_weight_cache_report,
)


def test_weight_asset_status_and_download_command() -> None:
    cached = WeightAsset(
        kind="model",
        repo_id="google/gemma-3-4b-it",
        label="base model google/gemma-3-4b-it",
        files=("config.json", "model.safetensors"),
        cached_files=("config.json", "model.safetensors"),
        missing_files=(),
    )
    assert cached.status == "cached"
    assert cached.download_command() == "huggingface-cli download google/gemma-3-4b-it"

    sae = WeightAsset(
        kind="sae",
        repo_id="google/gemma-scope-2-4b-it",
        label="SAE layer_17",
        files=(
            "resid_post/layer_17_width_65k_l0_medium/params.safetensors",
            "resid_post/layer_17_width_65k_l0_medium/config.json",
        ),
        cached_files=(),
        missing_files=(
            "resid_post/layer_17_width_65k_l0_medium/params.safetensors",
            "resid_post/layer_17_width_65k_l0_medium/config.json",
        ),
    )
    assert sae.status == "missing"
    cmd = sae.download_command()
    assert "huggingface-cli download google/gemma-scope-2-4b-it" in cmd
    assert '--include "resid_post/layer_17_width_65k_l0_medium/*"' in cmd


def test_check_sae_folder_reports_partial(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_cached(repo_id: str, filename: str) -> bool:
        return filename.endswith("config.json")

    monkeypatch.setattr("utils.weight_cache._is_cached_file", fake_cached)
    asset = check_sae_folder(
        "google/gemma-scope-2-4b-it",
        "resid_post/layer_17_width_65k_l0_medium",
        sae_id="layer_17_width_65k_l0_medium",
    )
    assert asset.status == "partial"
    assert asset.missing_files == (
        "resid_post/layer_17_width_65k_l0_medium/params.safetensors",
    )


def test_check_model_repo_uses_index_shards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "utils.weight_cache._model_weight_filenames",
        lambda repo_id: ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
    )
    present = {
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    monkeypatch.setattr(
        "utils.weight_cache._is_cached_file",
        lambda repo_id, filename: filename in present,
    )
    asset = check_model_repo("google/gemma-3-4b-it")
    assert asset.status == "cached"
    assert asset.missing_files == ()


def test_check_model_repo_unknown_shards_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("utils.weight_cache._model_weight_filenames", lambda repo_id: None)
    monkeypatch.setattr("utils.weight_cache._is_cached_file", lambda repo_id, filename: False)
    asset = check_model_repo("google/gemma-3-27b-it")
    assert asset.status == "missing"
    assert asset.missing_files == ("(full model snapshot)",)
    assert any("unknown offline" in note for note in asset.notes)


def test_format_report_includes_download_commands() -> None:
    assets = [
        WeightAsset(
            kind="model",
            repo_id="google/gemma-3-27b-it",
            label="base model google/gemma-3-27b-it",
            files=("(full model snapshot)",),
            cached_files=(),
            missing_files=("(full model snapshot)",),
        )
    ]
    text = format_weight_cache_report(assets)
    assert "[missing] base model google/gemma-3-27b-it" in text
    assert "huggingface-cli download google/gemma-3-27b-it" in text
    assert "HF_HOME" in text


def test_collect_pipeline_weight_assets_dedupes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "config" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "gemma-3-4b.yaml").write_text(
        "\n".join(
            [
                "# @package _global_",
                "model:",
                '  name: "google/gemma-3-4b-it"',
                '  wrapper: "gemma"',
                "extraction:",
                "  layers: [17, 22]",
                '  sae_release: "gemma-scope-2-4b-it-res"',
                '  sae_l0: "medium"',
            ]
        ),
        encoding="utf-8",
    )

    cfg = OmegaConf.create(
        {
            "extraction": {
                "sae_width": "65k",
                "sae_l0": "medium",
                "sae_release": "gemma-scope-2-4b-it-res",
            }
        }
    )
    experiment = OmegaConf.create({"models": ["gemma-3-4b"]})

    def fake_sae_repo_and_folder(release: str, sae_id: str) -> tuple[str, str]:
        folder = f"resid_post/{sae_id}"
        return "google/gemma-scope-2-4b-it", folder

    monkeypatch.setattr("utils.weight_cache._sae_repo_and_folder", fake_sae_repo_and_folder)
    monkeypatch.setattr(
        "utils.weight_cache.check_model_repo",
        lambda repo_id: WeightAsset(
            kind="model",
            repo_id=repo_id,
            label=f"base model {repo_id}",
            files=("config.json",),
            cached_files=("config.json",),
            missing_files=(),
        ),
    )

    seen_sae: list[str] = []

    def fake_check_sae(repo_id: str, folder_name: str, *, sae_id: str) -> WeightAsset:
        seen_sae.append(sae_id)
        return WeightAsset(
            kind="sae",
            repo_id=repo_id,
            label=f"SAE {sae_id}",
            files=(f"{folder_name}/params.safetensors",),
            cached_files=(),
            missing_files=(f"{folder_name}/params.safetensors",),
        )

    monkeypatch.setattr("utils.weight_cache.check_sae_folder", fake_check_sae)

    assets = collect_pipeline_weight_assets(
        cfg=cfg,
        experiment=experiment,
        sae_widths=["65k", "65k"],  # duplicate width should not duplicate SAEs
        project_root=tmp_path,
    )
    assert sum(1 for a in assets if a.kind == "model") == 1
    assert sorted(seen_sae) == [
        "layer_17_width_65k_l0_medium",
        "layer_22_width_65k_l0_medium",
    ]
    assert sum(1 for a in assets if a.kind == "sae") == 2
