"""Stage 2: SAE-based feature selection and ranking.

Replaces llm-lobotomy's `train_eval_svc.py` (SVM + mRMR). Must consume the
activations W&B artifact and emit a ranked feature artifact compatible with
`optimize_sae_steering.py` and the shared artifact naming in
`src/utils/experiment_ids.py`.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError(
        "SAE feature selection is not implemented yet. "
        "Expected inputs: cfg.data.activations_artifact_name, "
        "cfg.artifacts.feature_ranking_name. "
        "Expected output: W&B feature-ranking artifact with ranked_features JSON."
    )


if __name__ == "__main__":
    main()
