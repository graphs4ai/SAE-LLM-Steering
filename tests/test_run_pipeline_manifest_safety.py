from __future__ import annotations

from pathlib import Path
import sys

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_pipeline import (  # noqa: E402
    _build_commands,
    _should_preserve_existing_manifest,
    _should_schedule_shared_stage,
    _should_skip_existing,
)
from utils.local_artifacts import write_artifact  # noqa: E402
from utils.seeds import ResolvedSeeds  # noqa: E402


@pytest.mark.parametrize(
    ("previous_status", "resume", "force", "skip_existing", "expected"),
    [
        ("completed", True, False, True, True),
        ("completed", True, False, False, True),
        ("completed", False, False, True, True),
        ("completed", False, False, False, False),
        ("completed", True, True, True, False),
        ("planned", True, False, True, False),
        ("failed", True, False, True, False),
        (None, True, False, True, False),
    ],
)
def test_should_skip_existing(
    previous_status: str | None,
    resume: bool,
    force: bool,
    skip_existing: bool,
    expected: bool,
) -> None:
    assert (
        _should_skip_existing(previous_status, resume, force, skip_existing)
        is expected
    )


@pytest.mark.parametrize(
    ("previous_status", "dry_run", "expected"),
    [
        ("completed", True, True),
        ("completed", False, False),
        ("planned", True, False),
        ("failed", True, False),
        (None, True, False),
    ],
)
def test_should_preserve_existing_manifest(
    previous_status: str | None,
    dry_run: bool,
    expected: bool,
) -> None:
    assert _should_preserve_existing_manifest(previous_status, dry_run) is expected


def test_dry_run_force_preserves_completed_manifest() -> None:
    """Regression: dry_run + force must not clobber completed manifests."""
    assert _should_skip_existing("completed", resume=True, force=True, skip_existing=True) is False
    assert _should_preserve_existing_manifest("completed", dry_run=True) is True


def test_force_without_dry_run_does_not_preserve_completed_manifest() -> None:
    assert _should_preserve_existing_manifest("completed", dry_run=False) is False


def test_shared_stage_scheduled_once_per_name() -> None:
    scheduled: set[str] = set()
    assert _should_schedule_shared_stage("act-a", scheduled, force=False, exists=False) is True
    assert _should_schedule_shared_stage("act-a", scheduled, force=False, exists=False) is False
    assert _should_schedule_shared_stage("act-b", scheduled, force=False, exists=False) is True


def test_shared_stage_skips_when_artifact_exists() -> None:
    scheduled: set[str] = set()
    assert _should_schedule_shared_stage("act-a", scheduled, force=False, exists=True) is False
    assert "act-a" in scheduled
    # Already recorded as cached; still skipped.
    assert _should_schedule_shared_stage("act-a", scheduled, force=False, exists=True) is False


def test_shared_stage_force_reschedules_existing(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    payload = tmp_path / "activations.parquet"
    payload.write_bytes(b"parquet")
    write_artifact(
        "act-a",
        {"activations.parquet": payload},
        {"stage": "extract"},
        root=root,
    )
    scheduled: set[str] = set()
    assert _should_schedule_shared_stage("act-a", scheduled, force=True, exists=True) is True
    assert _should_schedule_shared_stage("act-a", scheduled, force=True, exists=True) is False


def _dummy_resolved() -> ResolvedSeeds:
    return ResolvedSeeds(
        global_seed=1,
        feature_selection=1,
        extraction=1,
        optimization=1,
        optimization_fast_sample=1,
        optimization_split=1,
        ipi=1,
        poeta=1,
    )


def test_build_commands_dedupes_extract_and_rank() -> None:
    stages = OmegaConf.create(
        {
            "extract_activations": True,
            "feature_selection": True,
            "optimization": True,
            "ipi_baseline": False,
            "ipi_intervened": False,
            "poeta": False,
        }
    )
    artifact_names = {
        "activations": "activations-demo",
        "feature_ranking": "feature-ranking-demo",
        "multipliers": "multipliers-demo",
        "ipi_baseline": "ipi-baseline-demo",
        "ipi_intervened": "ipi-intervened-demo",
    }
    cfg = OmegaConf.create({"ipi": {"seed_dependent_option_scores": False}})
    first = _build_commands(
        model_cfg_name="gemma-3-4b",
        direction="minimize",
        top_k=60,
        n_trials=100,
        stages=stages,
        include_baseline_ipi=False,
        include_extract=True,
        include_feature_selection=True,
        artifact_names=artifact_names,
        intervention_scope="prompt_all",
        intervention_last_k=3,
        sae_width="65k",
        bounds_multiplier=3.0,
        resolved=_dummy_resolved(),
        cfg=cfg,
    )
    second = _build_commands(
        model_cfg_name="gemma-3-4b",
        direction="maximize",
        top_k=60,
        n_trials=100,
        stages=stages,
        include_baseline_ipi=False,
        include_extract=False,
        include_feature_selection=False,
        artifact_names=artifact_names,
        intervention_scope="prompt_all",
        intervention_last_k=3,
        sae_width="65k",
        bounds_multiplier=3.0,
        resolved=_dummy_resolved(),
        cfg=cfg,
    )
    assert any("extract_activations.py" in cmd for cmd in first)
    assert any("select_sae_features.py" in cmd for cmd in first)
    assert not any("extract_activations.py" in cmd for cmd in second)
    assert not any("select_sae_features.py" in cmd for cmd in second)
    assert any("optimize_sae_steering.py" in cmd for cmd in second)
    assert all(":latest" not in cmd for cmd in first + second)


def test_build_commands_threads_force_override() -> None:
    stages = OmegaConf.create(
        {
            "extract_activations": True,
            "feature_selection": False,
            "optimization": False,
            "ipi_baseline": False,
            "ipi_intervened": False,
            "poeta": False,
        }
    )
    artifact_names = {
        "activations": "activations-demo",
        "feature_ranking": "feature-ranking-demo",
        "multipliers": "multipliers-demo",
        "ipi_baseline": "ipi-baseline-demo",
        "ipi_intervened": "ipi-intervened-demo",
    }
    cfg = OmegaConf.create({"ipi": {"seed_dependent_option_scores": False}})
    cmds = _build_commands(
        model_cfg_name="gemma-3-4b",
        direction="minimize",
        top_k=60,
        n_trials=100,
        stages=stages,
        include_baseline_ipi=False,
        include_extract=True,
        include_feature_selection=False,
        artifact_names=artifact_names,
        intervention_scope="prompt_all",
        intervention_last_k=3,
        sae_width="65k",
        bounds_multiplier=3.0,
        resolved=_dummy_resolved(),
        cfg=cfg,
        force=True,
    )
    assert len(cmds) == 1
    assert "pipeline.force=true" in cmds[0]
