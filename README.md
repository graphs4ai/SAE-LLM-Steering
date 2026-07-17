# SAE-LLM-Steering

Sibling project to [llm-lobotomy](../llm-lobotomy). It reuses the same end-to-end pipeline for political stance steering (activation extraction → feature selection → optimization → IPI evaluation → PoETa), but replaces **feature selection** and **optimization** with SAE-based methods instead of SVM + Optuna-on-neurons.

## Pipeline

```
1. extract_activations.py      → activations artifact
2. select_sae_features.py      → feature-ranking artifact
3. optimize_sae_steering.py    → multipliers artifact
4. ipi_eval.py                 → baseline / intervened IPI artifacts
5. poeta_evaluator.py          → capability benchmark
```

Orchestration, artifact naming, and manifest tracking use Hydra + W&B via `src/run_pipeline.py`.

See `notebooks/gemma_scope2_playground.ipynb` for an interactive Gemma Scope 2 sandbox on `google/gemma-3-4b-it`.

## Quick start

```bash
uv sync
wandb login
huggingface-cli login  # gated Gemma weights

# Dry-run the full sweep matrix
uv run python src/run_pipeline.py experiment=k80_trials pipeline.dry_run=true

# Gemma Scope playground
uv run jupyter lab notebooks/gemma_scope2_playground.ipynb

# Individual stages
uv run python src/extract_activations.py model=gemma-3-4b
uv run python src/select_sae_features.py model=gemma-3-4b artifacts.activations=...
uv run python src/optimize_sae_steering.py model=gemma-3-4b artifacts.feature_ranking=...
uv run python src/ipi_eval.py model=gemma-3-4b ipi.condition=baseline
uv run python src/ipi_eval.py model=gemma-3-4b ipi.condition=intervened artifacts.multipliers=...
```

Shared intervention knobs live under `intervention.*` (edit mode, decoder normalization, scope, last_k). Stage I/O uses `artifacts.*` only.

### Environment

- One **[uv](https://docs.astral.sh/uv/)** environment (`pyproject.toml` + `uv.lock`), Python 3.11+.
- **PoETaV2** comes from the sibling checkout at `../PoETaV2`, not PyPI.
