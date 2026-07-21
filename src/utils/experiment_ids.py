from __future__ import annotations

from typing import Any

"""Deterministic experiment/run and artifact name builders."""


# Kept in sync with `src/utils/intervention_hooks.py`. Hard-coding the default
# here (instead of importing) keeps this module dependency-free, as documented.
_DEFAULT_SCOPE = "prompt_without_buffer"


def _normalize(value: str) -> str:
    """Normalize free-text identifiers into stable slug-like fragments."""
    return value.strip().replace("/", "-").replace(" ", "-")


def format_layers_slug(layers_cfg: Any) -> str:
    """Return a Hydra-safe slug for SAE layer selection (no ``[``/``]``).

    Examples:
        ``"all"`` -> ``"all"``
        ``[9, 17, 22, 29]`` -> ``"L9-17-22-29"``
    """
    if isinstance(layers_cfg, str):
        if layers_cfg.lower() == "all":
            return "all"
        return _normalize(layers_cfg)
    return "L" + "-".join(str(int(layer)) for layer in layers_cfg)


def format_bounds_multiplier_slug(value: float) -> str:
    """Stable slug fragment for optimization.bounds_multiplier (e.g. 3.0 -> boundsx3)."""
    numeric = float(value)
    if numeric.is_integer():
        return f"boundsx{int(numeric)}"
    return f"boundsx{numeric:g}"


def format_sae_width_slug(value: str) -> str:
    """Stable slug fragment for extraction.sae_width (e.g. 65k -> sae65k)."""
    return f"sae{_normalize(value)}"


def _sae_bounds_suffix(
    sae_width: str | None = None,
    bounds_multiplier: float | None = None,
) -> str:
    """Optional `__sae{width}__boundsx{mult}` fragment for artifact names."""
    parts: list[str] = []
    if sae_width is not None:
        parts.append(format_sae_width_slug(sae_width))
    if bounds_multiplier is not None:
        parts.append(format_bounds_multiplier_slug(bounds_multiplier))
    if not parts:
        return ""
    return "__" + "__".join(parts)


def scope_identity_suffix(scope: str | None, last_k: int | None) -> str:
    """Return the optional `__{scope}__lastk{last_k}` fragment.

    Returns an empty string only when scope is None so every concrete scope
    (including the legacy default `prompt_without_buffer`) is encoded in
    run IDs and artifact names.
    """
    if scope is None:
        return ""
    suffix = f"__{_normalize(scope)}"
    if last_k is not None:
        suffix = f"{suffix}__lastk{int(last_k)}"
    return suffix


def make_run_id(
    model_name: str,
    dataset_tag: str,
    direction: str | None,
    top_k: int | None,
    n_trials: int | None,
    seed: int,
    condition: str | None = None,
    scope: str | None = None,
    last_k: int | None = None,
    sae_width: str | None = None,
    bounds_multiplier: float | None = None,
) -> str:
    parts: list[str] = [_normalize(model_name), _normalize(dataset_tag)]
    if condition:
        parts.append(_normalize(condition))
    if direction:
        parts.append(_normalize(direction))
    if top_k is not None:
        parts.append(f"k{int(top_k)}")
    if n_trials is not None:
        parts.append(f"trials{int(n_trials)}")
    if sae_width is not None:
        parts.append(format_sae_width_slug(sae_width))
    if bounds_multiplier is not None:
        parts.append(format_bounds_multiplier_slug(bounds_multiplier))
    parts.append(f"seed{int(seed)}")
    base = "__".join(parts)
    return f"{base}{scope_identity_suffix(scope, last_k)}"


def make_activation_artifact_name(
    model_name: str,
    dataset_tag: str,
    layers: str,
    sae_width: str | None = None,
) -> str:
    base = (
        f"activations-{_normalize(model_name)}-{_normalize(dataset_tag)}-"
        f"{_normalize(layers)}"
    )
    return f"{base}{_sae_bounds_suffix(sae_width=sae_width)}"


def make_feature_ranking_artifact_name(
    model_name: str,
    dataset_tag: str,
    ranking_top_n: int,
    sae_width: str | None = None,
) -> str:
    base = (
        f"feature-ranking-{_normalize(model_name)}-{_normalize(dataset_tag)}-"
        f"top{int(ranking_top_n)}"
    )
    return f"{base}{_sae_bounds_suffix(sae_width=sae_width)}"


def make_multiplier_artifact_name(
    model_name: str,
    dataset_tag: str,
    direction: str,
    top_k: int,
    n_trials: int,
    seed: int,
    scope: str | None = None,
    last_k: int | None = None,
    sae_width: str | None = None,
    bounds_multiplier: float | None = None,
) -> str:
    return (
        f"multipliers-{_normalize(model_name)}-{_normalize(dataset_tag)}-"
        f"{_normalize(direction)}-k{int(top_k)}-trials{int(n_trials)}-seed{int(seed)}"
        f"{_sae_bounds_suffix(sae_width=sae_width, bounds_multiplier=bounds_multiplier)}"
        f"{scope_identity_suffix(scope, last_k)}"
    )


def make_ipi_artifact_name(
    model_name: str,
    dataset_tag: str,
    condition: str,
    seed: int,
    direction: str | None = None,
    top_k: int | None = None,
    n_trials: int | None = None,
    scope: str | None = None,
    last_k: int | None = None,
    sae_width: str | None = None,
    bounds_multiplier: float | None = None,
) -> str:
    condition_norm = _normalize(condition)
    if condition_norm == "baseline":
        return f"ipi-baseline-{_normalize(model_name)}-{_normalize(dataset_tag)}-seed{int(seed)}"
    if condition_norm == "intervened":
        if direction is None or top_k is None or n_trials is None:
            raise ValueError(
                "Intervened IPI artifact name requires direction, top_k, and n_trials."
            )
        return (
            f"ipi-intervened-{_normalize(model_name)}-{_normalize(dataset_tag)}-"
            f"{_normalize(direction)}-k{int(top_k)}-trials{int(n_trials)}-seed{int(seed)}"
            f"{_sae_bounds_suffix(sae_width=sae_width, bounds_multiplier=bounds_multiplier)}"
            f"{scope_identity_suffix(scope, last_k)}"
        )
    raise ValueError(
        f"Unsupported condition={condition!r}. Expected 'baseline' or 'intervened'."
    )


if __name__ == "__main__":
    demo = {
        "run_id": make_run_id(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            direction="minimize",
            top_k=80,
            n_trials=3000,
            seed=42,
            sae_width="65k",
            bounds_multiplier=3.0,
        ),
        "run_id_default_scope": make_run_id(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            direction="minimize",
            top_k=16,
            n_trials=1000,
            seed=42,
            scope="prompt_without_buffer",
            last_k=3,
            sae_width="65k",
            bounds_multiplier=3.0,
        ),
        "run_id_with_scope": make_run_id(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            direction="minimize",
            top_k=16,
            n_trials=1000,
            seed=42,
            scope="prompt_last_token",
            last_k=3,
            sae_width="262k",
            bounds_multiplier=1.5,
        ),
        "activations": make_activation_artifact_name(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            layers=format_layers_slug("all"),
        ),
        "activations_layers_list": make_activation_artifact_name(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            layers=format_layers_slug([9, 17, 22, 29]),
        ),
        "feature_ranking": make_feature_ranking_artifact_name(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            ranking_top_n=256,
        ),
        "multipliers": make_multiplier_artifact_name(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            direction="minimize",
            top_k=80,
            n_trials=3000,
            seed=42,
        ),
        "multipliers_with_scope": make_multiplier_artifact_name(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            direction="minimize",
            top_k=16,
            n_trials=1000,
            seed=42,
            scope="prompt_last_token",
            last_k=3,
        ),
        "ipi_baseline": make_ipi_artifact_name(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            condition="baseline",
            seed=42,
        ),
        "ipi_intervened": make_ipi_artifact_name(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            condition="intervened",
            direction="minimize",
            top_k=80,
            n_trials=3000,
            seed=42,
        ),
        "ipi_intervened_with_scope": make_ipi_artifact_name(
            model_name="gemma-3-4b",
            dataset_tag="three_way_split_v1",
            condition="intervened",
            direction="minimize",
            top_k=16,
            n_trials=1000,
            seed=42,
            scope="prompt_last_token",
            last_k=3,
        ),
    }
    for key, value in demo.items():
        print(f"{key}={value}")
