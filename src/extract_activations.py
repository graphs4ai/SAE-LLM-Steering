import pandas as pd
import torch
from tqdm import tqdm
from model_factory import get_model_wrapper
from activation_df import ActivationDataFrame
import os
import tempfile
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from utils.ipi_prompts import build_ipi_chat_prompt
from utils.ipi_surrogate import resolve_option_scores
from utils.experiment_ids import format_layers_slug
from utils.local_artifacts import (
    artifact_exists,
    normalize_artifact_name,
    resolve_artifacts_root,
    should_force,
    write_artifact,
)


def get_last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Finds the index of the last non-padding token for each sequence in the batch.
    Assuming attention_mask is 1 for tokens and 0 for padding.
    """
    return attention_mask.sum(dim=1) - 1


def _pair_key_from_row_id(row_id: str) -> str:
    text = str(row_id)
    if text.endswith("_left"):
        return text[:-5]
    if text.endswith("_right"):
        return text[:-6]
    return text


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    from utils.seeds import (
        apply_torch_seed,
        log_resolved_seeds,
        resolve_seeds_from_cfg,
        resolved_seeds_to_dict,
    )

    resolved = resolve_seeds_from_cfg(cfg)
    apply_torch_seed(resolved.extraction)
    log_resolved_seeds(resolved, prefix="extract_activations")

    # Configuration from Hydra
    batch_size = cfg.extraction.batch_size
    prompt_mode = str(cfg.extraction.get("prompt_mode", "bare_statement"))
    device = cfg.extraction.device if torch.cuda.is_available(
    ) and cfg.extraction.device == "cuda" else "cpu"

    # Get Hydra output directory
    hydra_cfg = HydraConfig.get()
    run_dir = hydra_cfg.runtime.output_dir

    input_path = hydra.utils.to_absolute_path(cfg.data.feature_selection_dataset)

    # Determine artifact name based on dataset and model
    dataset_name = os.path.basename(input_path).replace(
        '.csv', '').replace('_propositions', '')
    model_name = cfg.model.name.split('/')[-1]
    layers = cfg.extraction.layers
    layers_str = format_layers_slug(layers)

    # Orchestrator-driven name takes precedence over the dataset-derived default
    # so a sweep can pin a deterministic identity (see src/utils/experiment_ids.py).
    artifacts_cfg = cfg.get('artifacts', {}) or {}
    override_name = artifacts_cfg.get('activations', None)
    artifact_name = normalize_artifact_name(
        str(override_name)
        if override_name
        else f"activations-{dataset_name}-{model_name}-{layers_str}"
    )
    force = should_force(cfg)
    project_root = hydra.utils.get_original_cwd()
    artifacts_root = resolve_artifacts_root(cfg=cfg, project_root=project_root)

    if (
        not force
        and artifact_exists(
            artifact_name,
            required_files=["activations.parquet"],
            root=artifacts_root,
        )
    ):
        print(
            f"[skip] activations artifact already exists: {artifact_name} "
            f"(under {artifacts_root})"
        )
        return

    print(f"Loading data from {input_path}...")
    if not os.path.exists(input_path):
        print("Input file not found. Creating dummy data for demonstration.")
        df = pd.DataFrame({
            'statement': ["This is a test sentence.", "Another political statement.", "Short one."] * 10,
            'pol_label_human': ["neutral", "political", "neutral"] * 10
        })
    else:
        df = pd.read_csv(input_path)
        if 'statement' not in df.columns or 'pol_label_human' not in df.columns:
            raise ValueError(
                "Input DataFrame must contain 'statement' and 'pol_label_human' columns.")

    print(f"Loaded {len(df)} samples.")

    print(f"Initializing model...")
    wrapper = get_model_wrapper(cfg)
    if wrapper.model.tokenizer is None:
        raise ValueError("The model wrapper must have a tokenizer.")
    loaded_name = getattr(wrapper.model.cfg, "model_name", None) or cfg.model.name
    print(f"Loaded model: {loaded_name}")

    if wrapper.model.tokenizer.pad_token is None:
        wrapper.model.tokenizer.pad_token = wrapper.model.tokenizer.eos_token

    option_scores = resolve_option_scores(cfg)

    layers_cfg = cfg.extraction.layers
    is_gemma_sae = hasattr(wrapper, "resolve_sae_layers")
    if is_gemma_sae:
        layers = wrapper.resolve_sae_layers(layers_cfg)
        d_features = wrapper.d_sae
    elif isinstance(layers_cfg, str):
        layers = list(range(wrapper.n_layers))
        d_features = wrapper.model.cfg.d_model
    else:
        layers = list(layers_cfg)
        d_features = wrapper.model.cfg.d_model

    activation_df = ActivationDataFrame(layers=layers, d_features=d_features)

    print("Starting extraction loop...")

    total_samples = len(df)

    for start_idx in tqdm(range(0, total_samples, batch_size), desc="Processing Batches"):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_df = df.iloc[start_idx:end_idx]

        statements = batch_df['statement'].astype(str).tolist()
        labels = batch_df['pol_label_human'].tolist()
        row_ids = batch_df["id"].astype(str).tolist() if "id" in batch_df.columns else [
            f"row_{idx}" for idx in batch_df.index
        ]
        pair_keys = [_pair_key_from_row_id(row_id) for row_id in row_ids]

        if prompt_mode == "ipi_chat":
            texts = [
                build_ipi_chat_prompt(
                    wrapper.model.tokenizer,
                    statement=statement,
                    language=str(cfg.ipi.get("language", "pt")),
                    option_scores=option_scores,
                )
                for statement in statements
            ]
        elif prompt_mode == "bare_statement":
            texts = statements
        else:
            raise ValueError(
                f"Unknown extraction.prompt_mode={prompt_mode!r}. "
                "Expected 'ipi_chat' or 'bare_statement'."
            )

        encoding = wrapper.model.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=cfg.extraction.max_length
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        try:
            layer_activations = wrapper.get_layer_activations(
                input_ids, layers=layers)
        except Exception as e:
            print(f"Error processing batch {start_idx}-{end_idx}: {e}")
            continue

        padding_side = wrapper.model.tokenizer.padding_side

        batch_indices = torch.arange(input_ids.shape[0])

        if padding_side == 'left':
            final_activations = layer_activations[:, -1, :]
        else:
            last_token_indices = attention_mask.sum(dim=1).long() - 1
            final_activations = layer_activations[batch_indices,
                                                  last_token_indices, :]

        activation_df.add_batch(
            final_activations,
            labels,
            metadata={
                "pair_key": pair_keys,
                "row_id": row_ids,
                "statement": statements,
            },
        )

    with tempfile.TemporaryDirectory(dir=run_dir) as tmp_dir:
        tmp_parquet = os.path.join(tmp_dir, "activations.parquet")
        print(f"Saving results to {tmp_parquet}...")
        activation_df.save(tmp_parquet)

        artifact_metadata = {
            'model_name': cfg.model.name,
            'model_wrapper': cfg.model.wrapper,
            'feature_selection_dataset': input_path,
            'n_samples': total_samples,
            'n_layers': len(layers),
            'layers': layers,
            'd_features': d_features,
            'batch_size': batch_size,
            'max_length': cfg.extraction.max_length,
            'prompt_mode': prompt_mode,
            'token_position': 'last_prompt',
            'extraction_seed': resolved.extraction,
            'resolved_seeds': resolved_seeds_to_dict(resolved),
            'n_features': len(layers) * d_features,
        }
        if is_gemma_sae:
            artifact_metadata.update({
                'd_sae': d_features,
                'sae_width': getattr(wrapper, 'sae_width', None),
                'sae_l0': getattr(wrapper, 'sae_l0', None),
                'sae_release': getattr(wrapper, 'sae_release', None),
                'hook': 'resid_post',
            })
        else:
            artifact_metadata['d_model'] = d_features

        artifact_path = write_artifact(
            artifact_name,
            {"activations.parquet": tmp_parquet},
            artifact_metadata,
            root=artifacts_root,
            force=force,
        )
        print(f"Activations artifact written: {artifact_path}")


if __name__ == "__main__":
    main()
