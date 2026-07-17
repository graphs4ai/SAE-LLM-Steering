from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.local_artifacts import (  # noqa: E402
    artifact_dir,
    artifact_exists,
    find_file,
    load_metadata,
    normalize_artifact_name,
    resolve_path,
    write_artifact,
)


def test_normalize_artifact_name_strips_alias() -> None:
    assert normalize_artifact_name("activations-foo:latest") == "activations-foo"
    assert normalize_artifact_name("activations-foo") == "activations-foo"


def test_write_and_resolve_artifact(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    source = tmp_path / "payload.txt"
    source.write_text("hello", encoding="utf-8")

    written = write_artifact(
        "demo-artifact",
        {"payload.txt": source},
        {"stage": "test", "n": 1},
        root=root,
    )
    assert written == root / "demo-artifact"
    assert artifact_exists("demo-artifact", required_files=["payload.txt"], root=root)
    assert resolve_path("demo-artifact", "payload.txt", root=root).read_text() == "hello"
    assert load_metadata("demo-artifact", root=root)["stage"] == "test"
    assert find_file("demo-artifact", "payload.*", root=root).name == "payload.txt"


def test_write_artifact_refuses_overwrite_without_force(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    source = tmp_path / "payload.txt"
    source.write_text("v1", encoding="utf-8")
    write_artifact("demo", {"payload.txt": source}, {"v": 1}, root=root)

    source.write_text("v2", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_artifact("demo", {"payload.txt": source}, {"v": 2}, root=root)

    write_artifact("demo", {"payload.txt": source}, {"v": 2}, root=root, force=True)
    assert load_metadata("demo", root=root)["v"] == 2
    assert resolve_path("demo", "payload.txt", root=root).read_text() == "v2"


def test_artifact_dir_uses_normalized_name(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    assert artifact_dir("name:latest", root=root) == root / "name"
