from __future__ import annotations

from typing import Any, Mapping, Optional

import torch


def unit_decoder_direction(w_dec_row: torch.Tensor) -> torch.Tensor:
    """Return a unit-norm decoder direction, leaving zero rows unchanged."""
    norm = w_dec_row.float().norm()
    if float(norm.item()) == 0.0:
        return w_dec_row
    return w_dec_row / norm.to(device=w_dec_row.device, dtype=w_dec_row.dtype)


def build_normalized_steering_vector(
    w_dec: torch.Tensor,
    feature_coefficients: dict[int, float],
) -> torch.Tensor:
    """Build sum_j beta_j * d_hat_j from decoder rows."""
    steering = torch.zeros(
        w_dec.shape[1],
        device=w_dec.device,
        dtype=w_dec.dtype,
    )
    for feature_idx, beta in feature_coefficients.items():
        if beta == 0.0:
            continue
        steering = steering + float(beta) * unit_decoder_direction(w_dec[feature_idx])
    return steering


def latent_clamp_delta(
    sae: Any,
    x: torch.Tensor,
    feature_alphas: Mapping[int, float],
) -> torch.Tensor:
    """Per-position residual-stream delta for latent-space SAE steering.

    Computes ``z = sae.encode(x)`` (the real, input-dependent SAE code), then
    additively clamps target features: ``z'_j = ReLU(z_j + alpha_j)`` for
    ``j`` in ``feature_alphas``, leaving all other latents untouched. Since
    ``sae.decode(f) = f @ W_dec + b_dec`` is affine, the reconstruction
    difference ``decode(z') - decode(z)`` simplifies to ``(z' - z) @ W_dec``
    (the decoder bias cancels), so only ``sae.encode`` and ``sae.W_dec`` are
    needed here.

    Args:
        sae: A ``sae_lens.SAE`` instance (or compatible), providing
            ``encode(x) -> z`` and ``W_dec`` of shape ``[d_sae, d_model]``.
        x: Real residual-stream activations at the masked positions, shape
            ``[..., d_model]``.
        feature_alphas: Mapping of SAE feature index -> additive shift alpha_j
            (Optuna-optimized, applied to the real per-token latent value).

    Returns:
        Delta tensor with the same leading shape as ``x`` and last dim
        ``d_model``, to be added directly to the real residual stream
        (``x' = x + delta``). This is only the *effect* of the intervention;
        the SAE's own reconstruction error on ``x`` is never touched.
    """
    with torch.no_grad():
        z = sae.encode(x)
        z_delta = torch.zeros_like(z)
        for feature_idx, alpha in feature_alphas.items():
            if alpha == 0.0:
                continue
            z_j = z[..., feature_idx]
            z_delta[..., feature_idx] = torch.clamp(z_j + float(alpha), min=0.0) - z_j
        return z_delta @ sae.W_dec


def compute_latent_clamp_bounds(
    ranked_feature_stats: Mapping[str, Mapping[str, Optional[float]]],
    multiplier: float,
    fallback_bounds: tuple[float, float],
) -> dict[str, tuple[float, float]]:
    """Per-feature Optuna bounds for ``latent_clamp_additive``, derived from
    real per-feature activation statistics collected during feature selection
    (``mean_left`` / ``mean_right`` from ``paired_contrastive`` ranking).

    For each feature: ``scale = max(|mean_left|, |mean_right|)`` and
    ``bounds = (-multiplier * scale, +multiplier * scale)``. A multiplier of
    1.0 already lets alpha fully ablate the feature from either class's
    typical activation level (``ReLU(scale - scale) == 0``); values above 1.0
    give headroom to also amplify beyond the larger class's mean.

    Features with missing or zero statistics (e.g. the ``mean_activation``
    selection method, which does not compute ``mean_left``/``mean_right``)
    fall back to ``fallback_bounds`` with a printed warning, rather than
    failing the whole run.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for feature_name, stats in ranked_feature_stats.items():
        mean_left = stats.get("mean_left")
        mean_right = stats.get("mean_right")
        if mean_left is None or mean_right is None:
            print(
                f"WARNING: auto_bounds could not find mean_left/mean_right for "
                f"{feature_name!r} (likely feature_selection.method="
                f"'mean_activation'); falling back to bounds={fallback_bounds}."
            )
            bounds[feature_name] = fallback_bounds
            continue

        scale = max(abs(float(mean_left)), abs(float(mean_right)))
        if scale == 0.0:
            print(
                f"WARNING: auto_bounds found scale=0.0 for {feature_name!r} "
                f"(mean_left={mean_left}, mean_right={mean_right}); falling "
                f"back to bounds={fallback_bounds}."
            )
            bounds[feature_name] = fallback_bounds
            continue

        bounds[feature_name] = (-multiplier * scale, multiplier * scale)

    return bounds
