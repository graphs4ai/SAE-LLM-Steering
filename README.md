# SAE-LLM-Steering

Sibling project to [llm-lobotomy](../llm-lobotomy). It reuses the same end-to-end pipeline for political stance steering (activation extraction → feature selection → optimization → IPI evaluation → PoETa), but replaces **feature selection** and **optimization** with SAE-based methods instead of SVM + Optuna.

## Pipeline

```
1. extract_activations.py      → activations artifact
2. select_sae_features.py      → feature-ranking artifact   (SAE; to implement)
3. optimize_sae_steering.py    → multipliers artifact       (SAE; to implement)
4. ipi_eval.py                 → baseline / intervened IPI artifacts
5. poeta_evaluator.py          → capability benchmark
```

Orchestration, artifact naming, and manifest tracking follow the same Hydra + W&B pattern as llm-lobotomy via `src/run_pipeline.py`.

See `notebooks/gemma_scope2_playground.ipynb` for an interactive Gemma Scope 2 sandbox on `google/gemma-3-4b-it`.

## Copied from llm-lobotomy

Shared infrastructure was copied from the sibling repo. **Not** copied (replaced by SAE stages):

- `src/train_eval_svc.py`
- `src/optimize_intervention.py`
- `src/compile_target_neurons.py`
- `scripts/clear_optuna_run.py`
- `visualizations/multipliers.py`

## Quick start

```bash
uv sync
wandb login
huggingface-cli login  # gated Gemma weights

# Dry-run the full sweep matrix
uv run python src/run_pipeline.py experiment=k80_trials pipeline.dry_run=true

# Gemma Scope playground
uv run jupyter lab notebooks/gemma_scope2_playground.ipynb

# Individual stages (once SAE scripts are implemented)
uv run python src/extract_activations.py model=gemma-3-4b
uv run python src/select_sae_features.py model=gemma-3-4b data.activations_artifact_name=...
uv run python src/optimize_sae_steering.py model=gemma-3-4b optimization.feature_artifact_name=...
uv run python src/ipi_eval.py model=gemma-3-4b ipi.condition=baseline
uv run python src/ipi_eval.py model=gemma-3-4b ipi.condition=intervened ipi.multiplier_artifact_name=...
```

### Environment

- One **[uv](https://docs.astral.sh/uv/)** environment (`pyproject.toml` + `uv.lock`), Python 3.11+.
- **PoETaV2** comes from the sibling checkout at `../PoETaV2`, not PyPI.
