"""Local filesystem artifact store (replaces W&B artifact I/O).

Layout::

    artifacts/<artifact_name>/
      metadata.json
      <payload files>

Artifact names may include a W&B-style ``:alias`` suffix (e.g. ``:latest``);
aliases are stripped before resolving paths.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping


_DEFAULT_ROOT = "artifacts"
_METADATA_FILENAME = "metadata.json"


def normalize_artifact_name(name: str) -> str:
    """Strip optional ``:alias`` suffix from an artifact reference."""
    text = str(name).strip()
    if not text:
        raise ValueError("Artifact name must be a non-empty string.")
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    if not text:
        raise ValueError(f"Artifact name is empty after stripping alias: {name!r}")
    return text


def resolve_artifacts_root(
    root: str | Path | None = None,
    *,
    cfg: Any | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Return the absolute artifacts root directory.

    Precedence: explicit ``root`` > ``cfg.artifacts.root`` > ``artifacts``.
    Paths are resolved relative to ``project_root`` (or cwd when unset).
    """
    if root is None and cfg is not None and hasattr(cfg, "get"):
        artifacts_cfg = cfg.get("artifacts") or {}
        if hasattr(artifacts_cfg, "get"):
            root = artifacts_cfg.get("root")
        elif isinstance(artifacts_cfg, Mapping):
            root = artifacts_cfg.get("root")
    if root is None:
        root = _DEFAULT_ROOT

    path = Path(root)
    if not path.is_absolute():
        base = Path(project_root) if project_root is not None else Path.cwd()
        path = base / path
    return path.resolve()


def artifact_dir(
    name: str,
    *,
    root: str | Path | None = None,
    cfg: Any | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Return ``<root>/<normalized_name>/`` (does not create the directory)."""
    artifacts_root = resolve_artifacts_root(root, cfg=cfg, project_root=project_root)
    return artifacts_root / normalize_artifact_name(name)


def metadata_path(
    name: str,
    *,
    root: str | Path | None = None,
    cfg: Any | None = None,
    project_root: str | Path | None = None,
) -> Path:
    return artifact_dir(name, root=root, cfg=cfg, project_root=project_root) / _METADATA_FILENAME


def artifact_exists(
    name: str,
    required_files: Iterable[str] | None = None,
    *,
    root: str | Path | None = None,
    cfg: Any | None = None,
    project_root: str | Path | None = None,
) -> bool:
    """Return True when the artifact dir exists and all required files are present."""
    directory = artifact_dir(name, root=root, cfg=cfg, project_root=project_root)
    if not directory.is_dir():
        return False
    if not (directory / _METADATA_FILENAME).is_file():
        return False
    if required_files is None:
        return True
    for filename in required_files:
        if not (directory / filename).is_file():
            return False
    return True


def load_metadata(
    name: str,
    *,
    root: str | Path | None = None,
    cfg: Any | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    path = metadata_path(name, root=root, cfg=cfg, project_root=project_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact metadata not found for {normalize_artifact_name(name)!r}: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact metadata must be a JSON object: {path}")
    return payload


def resolve_path(
    name: str,
    filename: str,
    *,
    root: str | Path | None = None,
    cfg: Any | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Return the path to a required payload file inside an artifact directory."""
    path = artifact_dir(name, root=root, cfg=cfg, project_root=project_root) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Artifact file {filename!r} not found for "
            f"{normalize_artifact_name(name)!r}: {path}"
        )
    return path


def find_file(
    name: str,
    pattern: str,
    *,
    root: str | Path | None = None,
    cfg: Any | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Return the first file matching ``pattern`` under the artifact directory."""
    directory = artifact_dir(name, root=root, cfg=cfg, project_root=project_root)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Artifact directory not found for {normalize_artifact_name(name)!r}: "
            f"{directory}"
        )
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No files matching {pattern!r} in artifact "
            f"{normalize_artifact_name(name)!r}: {directory}"
        )
    return matches[0]


def write_artifact(
    name: str,
    files: Mapping[str, str | Path] | Iterable[str | Path],
    metadata: Mapping[str, Any],
    *,
    root: str | Path | None = None,
    cfg: Any | None = None,
    project_root: str | Path | None = None,
    force: bool = False,
    extra_dirs: Mapping[str, str | Path] | None = None,
) -> Path:
    """
    Persist payload files and ``metadata.json`` under ``artifacts/<name>/``.

    ``files`` may be a mapping of ``dest_name -> source_path`` or an iterable of
    source paths (basename preserved). ``extra_dirs`` copies directory trees
    into named subdirectories of the artifact (e.g. transcripts).
    """
    if not metadata:
        raise ValueError("Artifact metadata must not be empty.")

    directory = artifact_dir(name, root=root, cfg=cfg, project_root=project_root)
    required_names: list[str] = []
    if isinstance(files, Mapping):
        required_names = list(files.keys())
    else:
        required_names = [Path(src).name for src in files]

    if (
        not force
        and artifact_exists(
            name,
            required_files=required_names or None,
            root=root,
            cfg=cfg,
            project_root=project_root,
        )
    ):
        raise FileExistsError(
            f"Artifact {normalize_artifact_name(name)!r} already exists at "
            f"{directory}. Pass force=True to overwrite."
        )

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    if isinstance(files, Mapping):
        items = list(files.items())
    else:
        items = [(Path(src).name, src) for src in files]

    if not items and not extra_dirs:
        raise ValueError("At least one file or extra_dirs entry is required.")

    for dest_name, source in items:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Artifact source file does not exist: {source_path}")
        dest_path = directory / dest_name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)

    if extra_dirs:
        for subdir_name, source_dir in extra_dirs.items():
            source_path = Path(source_dir)
            if not source_path.is_dir():
                continue
            dest_subdir = directory / subdir_name
            dest_subdir.mkdir(parents=True, exist_ok=True)
            for path in sorted(source_path.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(source_path)
                    target = dest_subdir / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)

    meta_path = directory / _METADATA_FILENAME
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(dict(metadata), f, indent=2, ensure_ascii=False, default=str)

    return directory


def should_force(cfg: Any | None = None) -> bool:
    """Return True when ``pipeline.force`` is set on the config."""
    if cfg is None or not hasattr(cfg, "get"):
        return False
    pipeline_cfg = cfg.get("pipeline") or {}
    if hasattr(pipeline_cfg, "get"):
        return bool(pipeline_cfg.get("force", False))
    if isinstance(pipeline_cfg, Mapping):
        return bool(pipeline_cfg.get("force", False))
    return False
