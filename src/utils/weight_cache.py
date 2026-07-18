"""Report which HuggingFace model / SAE weights a pipeline plan needs.

Used by ``run_pipeline`` dry-runs so HPC jobs can pre-download assets on a
machine with Hub access before moving to a cluster with limited internet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from omegaconf import DictConfig, OmegaConf


# Tokenizer / config files transformers commonly needs alongside weights.
_MODEL_AUX_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.json",
    "preprocessor_config.json",
    "processor_config.json",
)


@dataclass(frozen=True)
class WeightAsset:
    """One downloadable Hub asset (base model snapshot or SAE folder)."""

    kind: str  # "model" | "sae"
    repo_id: str
    label: str
    files: tuple[str, ...]
    cached_files: tuple[str, ...] = ()
    missing_files: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if not self.files:
            return "unknown"
        if not self.missing_files:
            return "cached"
        if self.cached_files:
            return "partial"
        return "missing"

    def download_command(self) -> str:
        if self.kind == "model":
            return f"hf download {self.repo_id}"
        if len(self.files) == 1:
            return f"hf download {self.repo_id} {self.files[0]}"
        # SAE folders: include both params + config via a prefix glob.
        prefixes = sorted({str(Path(f).parent).replace("\\", "/") for f in self.files})
        includes = " ".join(f'--include "{prefix}/*"' for prefix in prefixes)
        return f"hf download {self.repo_id} {includes}"


@dataclass
class WeightCacheReport:
    assets: list[WeightAsset] = field(default_factory=list)

    @property
    def missing_or_partial(self) -> list[WeightAsset]:
        return [a for a in self.assets if a.status != "cached"]

    @property
    def all_cached(self) -> bool:
        return bool(self.assets) and not self.missing_or_partial


def _is_cached_file(repo_id: str, filename: str) -> bool:
    """Return True when ``filename`` resolves to an existing local Hub file."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False

    path = try_to_load_from_cache(repo_id, filename)
    return isinstance(path, str) and Path(path).is_file()


def _read_cached_json(repo_id: str, filename: str) -> dict[str, Any] | None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None

    path = try_to_load_from_cache(repo_id, filename)
    if not isinstance(path, str) or not Path(path).is_file():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def _model_weight_filenames(repo_id: str) -> tuple[str, ...] | None:
    """Infer weight shard names from a locally cached index, if present."""
    index = _read_cached_json(repo_id, "model.safetensors.index.json")
    if index is not None:
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict) and weight_map:
            return tuple(sorted({str(v) for v in weight_map.values()}))

    pt_index = _read_cached_json(repo_id, "pytorch_model.bin.index.json")
    if pt_index is not None:
        weight_map = pt_index.get("weight_map")
        if isinstance(weight_map, dict) and weight_map:
            return tuple(sorted({str(v) for v in weight_map.values()}))

    for single in ("model.safetensors", "pytorch_model.bin"):
        if _is_cached_file(repo_id, single):
            return (single,)

    # Index / single-file not in cache: we cannot enumerate shards offline.
    return None


def check_model_repo(repo_id: str) -> WeightAsset:
    """Check whether a base LLM repo looks complete in the local HF cache."""
    notes: list[str] = []
    weight_files = _model_weight_filenames(repo_id)

    # Required aux files: config + a tokenizer the HF loaders can use.
    required_aux = ("config.json", "tokenizer.json", "tokenizer_config.json")

    if weight_files is None:
        # Offline we cannot enumerate shards without an index in cache.
        notes.append(
            "weight shard list unknown offline (no index/single checkpoint in cache); "
            "download the full model snapshot"
        )
        cached_aux = tuple(f for f in required_aux if _is_cached_file(repo_id, f))
        if cached_aux:
            notes.append("partial aux files already cached: " + ", ".join(cached_aux))
        return WeightAsset(
            kind="model",
            repo_id=repo_id,
            label=f"base model {repo_id}",
            files=("(full model snapshot)",),
            cached_files=(),
            missing_files=("(full model snapshot)",),
            notes=tuple(notes),
        )

    expected = tuple(list(weight_files) + [f for f in required_aux if f not in weight_files])
    cached = tuple(f for f in expected if _is_cached_file(repo_id, f))
    missing = tuple(f for f in expected if f not in cached)

    optional_missing = [
        aux
        for aux in _MODEL_AUX_FILES
        if aux not in expected and not _is_cached_file(repo_id, aux)
    ]
    if optional_missing and not missing:
        preview = ", ".join(optional_missing[:6])
        suffix = "..." if len(optional_missing) > 6 else ""
        notes.append(f"optional tokenizer/aux files not cached: {preview}{suffix}")

    return WeightAsset(
        kind="model",
        repo_id=repo_id,
        label=f"base model {repo_id}",
        files=expected,
        cached_files=cached,
        missing_files=missing,
        notes=tuple(notes),
    )


def check_sae_folder(repo_id: str, folder_name: str, *, sae_id: str) -> WeightAsset:
    """Check Gemma Scope 2 SAE folder files (``params.safetensors`` + ``config.json``)."""
    folder = folder_name.strip("/").replace("\\", "/")
    files = (
        f"{folder}/params.safetensors",
        f"{folder}/config.json",
    )
    cached = tuple(f for f in files if _is_cached_file(repo_id, f))
    missing = tuple(f for f in files if f not in cached)
    return WeightAsset(
        kind="sae",
        repo_id=repo_id,
        label=f"SAE {sae_id} ({repo_id}/{folder})",
        files=files,
        cached_files=cached,
        missing_files=missing,
    )


def _load_model_yaml(project_root: Path, model_cfg_name: str) -> DictConfig:
    path = project_root / "config" / "model" / f"{model_cfg_name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Model config not found: {path}")
    return OmegaConf.load(path)


def _resolve_layers(
    layers_cfg: Any,
    sae_release: str,
) -> list[int]:
    if layers_cfg is None or (
        isinstance(layers_cfg, str) and layers_cfg.lower() == "all"
    ):
        from gemma_3_wrapper import infer_allowed_sae_layers

        return list(infer_allowed_sae_layers(sae_release))
    return [int(layer) for layer in layers_cfg]


def _sae_repo_and_folder(release: str, sae_id: str) -> tuple[str, str]:
    from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory

    directory = get_pretrained_saes_directory()
    if release not in directory:
        raise KeyError(f"Unknown SAE release {release!r} (not in sae_lens directory)")
    info = directory[release]
    if sae_id not in info.saes_map:
        raise KeyError(
            f"SAE id {sae_id!r} not found in release {release!r}. "
            f"Known ids include: {sorted(info.saes_map)[:8]}..."
        )
    return str(info.repo_id), str(info.saes_map[sae_id])


def collect_pipeline_weight_assets(
    *,
    cfg: DictConfig,
    experiment: DictConfig,
    sae_widths: Sequence[str],
    project_root: Path,
) -> list[WeightAsset]:
    """Deduplicate model + SAE assets required by the experiment matrix."""
    extraction_defaults = cfg.get("extraction", {}) or {}
    default_l0 = str(extraction_defaults.get("sae_l0", "medium"))
    default_release = str(
        extraction_defaults.get("sae_release", "gemma-scope-2-4b-it-res")
    )

    assets_by_key: dict[tuple[str, str], WeightAsset] = {}

    for model_cfg_name in experiment.models:
        model_cfg_name = str(model_cfg_name)
        model_yaml = _load_model_yaml(project_root, model_cfg_name)
        model_block = model_yaml.get("model", {}) or {}
        extraction_block = model_yaml.get("extraction", {}) or {}

        repo_id = str(model_block.get("name") or "").strip()
        if not repo_id:
            raise ValueError(
                f"config/model/{model_cfg_name}.yaml is missing model.name"
            )

        model_key = ("model", repo_id)
        if model_key not in assets_by_key:
            assets_by_key[model_key] = check_model_repo(repo_id)

        wrapper = str(model_block.get("wrapper", "")).lower()
        if wrapper != "gemma":
            # Non-Gemma wrappers only need the base checkpoint for this pipeline.
            continue

        sae_release = str(extraction_block.get("sae_release", default_release))
        sae_l0 = str(extraction_block.get("sae_l0", default_l0))
        layers = _resolve_layers(extraction_block.get("layers", "all"), sae_release)

        for sae_width in sae_widths:
            width = str(sae_width)
            for layer in layers:
                sae_id = f"layer_{layer}_width_{width}_l0_{sae_l0}"
                sae_repo, folder = _sae_repo_and_folder(sae_release, sae_id)
                key = ("sae", f"{sae_repo}/{folder}")
                if key not in assets_by_key:
                    assets_by_key[key] = check_sae_folder(
                        sae_repo, folder, sae_id=sae_id
                    )

    # Stable order: models first, then SAEs by label.
    models = [a for a in assets_by_key.values() if a.kind == "model"]
    saes = [a for a in assets_by_key.values() if a.kind == "sae"]
    models.sort(key=lambda a: a.repo_id)
    saes.sort(key=lambda a: a.label)
    return models + saes


def format_weight_cache_report(assets: Iterable[WeightAsset]) -> str:
    """Human-readable dry-run section for terminal output."""
    assets_list = list(assets)
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("WEIGHT / CHECKPOINT DOWNLOAD CHECK (dry-run)")
    lines.append("=" * 70)

    if not assets_list:
        lines.append("No model/SAE assets resolved from the experiment matrix.")
        return "\n".join(lines)

    status_counts = {"cached": 0, "partial": 0, "missing": 0, "unknown": 0}
    for asset in assets_list:
        status_counts[asset.status] = status_counts.get(asset.status, 0) + 1
        marker = {
            "cached": "[cached]",
            "partial": "[partial]",
            "missing": "[missing]",
            "unknown": "[unknown]",
        }.get(asset.status, f"[{asset.status}]")
        lines.append(f"{marker} {asset.label}")
        if asset.missing_files:
            for filename in asset.missing_files:
                lines.append(f"         missing: {filename}")
        for note in asset.notes:
            lines.append(f"         note: {note}")

    lines.append("-" * 70)
    lines.append(
        "summary: "
        f"cached={status_counts.get('cached', 0)} "
        f"partial={status_counts.get('partial', 0)} "
        f"missing={status_counts.get('missing', 0)} "
        f"unknown={status_counts.get('unknown', 0)}"
    )

    todo = [a for a in assets_list if a.status != "cached"]
    if not todo:
        lines.append("All required weights appear present in the local HuggingFace cache.")
    else:
        lines.append("Pre-download on a networked machine (then copy HF cache / set HF_HOME):")
        seen_cmds: set[str] = set()
        for asset in todo:
            cmd = asset.download_command()
            if cmd in seen_cmds:
                continue
            seen_cmds.add(cmd)
            lines.append(f"  {cmd}")
        lines.append(
            "Tip: after download, point the cluster job at the same cache via "
            "HF_HOME or HUGGINGFACE_HUB_CACHE."
        )

    lines.append("=" * 70)
    return "\n".join(lines)


def report_pipeline_weight_cache(
    *,
    cfg: DictConfig,
    experiment: DictConfig,
    sae_widths: Sequence[str],
    project_root: Path,
) -> WeightCacheReport:
    """Collect assets and print the dry-run download report."""
    assets = collect_pipeline_weight_assets(
        cfg=cfg,
        experiment=experiment,
        sae_widths=sae_widths,
        project_root=project_root,
    )
    print(format_weight_cache_report(assets))
    return WeightCacheReport(assets=assets)
