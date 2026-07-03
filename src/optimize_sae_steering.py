"""Stage 3: SAE steering optimization.

Replaces llm-lobotomy's `optimize_intervention.py` (Optuna + residual multipliers).
Must consume the feature-ranking W&B artifact and emit multiplier artifacts
compatible with `ipi_eval.py` and `src/utils/metrics_backfill.py`.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError(
        "SAE steering optimization is not implemented yet. "
        "Expected inputs: cfg.optimization.feature_artifact_name, "
        "cfg.artifacts.multiplier_name. "
        "Expected output: W&B multipliers artifact with intervention metadata."
    )


if __name__ == "__main__":
    main()
