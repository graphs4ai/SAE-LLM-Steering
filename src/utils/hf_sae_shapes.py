"""Work around sae_lens HTTP Range probes failing on HuggingFace Xet CDN.

``sae_lens.loading.pretrained_sae_loaders.get_safetensors_tensor_shapes`` fetches
safetensors header bytes via raw ``requests`` Range GETs. For Xet-backed Hub
files those redirects land on ``cas-bridge.xethub.hf.co`` and currently return
403 AccessDenied, even when the file is already in the local HF cache.

Prefer reading the header from a local cached (or ``hf_hub_download``-resolved)
file, and only fall back to the original network probe if needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

_PATCHED = False


def _shapes_from_local_safetensors(path: str | Path) -> dict[str, list[int]]:
    with open(path, "rb") as f:
        meta_size = int.from_bytes(f.read(8), byteorder="little")
        metadata_json = f.read(meta_size).decode("utf-8").strip()
    metadata = json.loads(metadata_json)
    return {
        name: info["shape"]
        for name, info in metadata.items()
        if name != "__metadata__"
    }


def _resolve_local_path(repo_id: str, filename: str) -> str | None:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, filename)
    if isinstance(cached, str) and Path(cached).is_file():
        return cached

    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception:
        return None
    if isinstance(path, str) and Path(path).is_file():
        return path
    return None


def install_local_safetensors_shape_patch() -> None:
    """Monkeypatch sae_lens shape probing to prefer local Hub cache files."""
    global _PATCHED
    if _PATCHED:
        return

    import sae_lens.loading.pretrained_sae_loaders as loaders

    original: Callable[[str, str], dict[str, list[int]]] = (
        loaders.get_safetensors_tensor_shapes
    )

    def get_safetensors_tensor_shapes(
        repo_id: str, filename: str
    ) -> dict[str, list[int]]:
        local_path = _resolve_local_path(repo_id, filename)
        if local_path is not None:
            return _shapes_from_local_safetensors(local_path)
        return original(repo_id, filename)

    loaders.get_safetensors_tensor_shapes = get_safetensors_tensor_shapes
    _PATCHED = True
