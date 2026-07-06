import pandas as pd
import torch
from tqdm import tqdm
from model_factory import get_model_wrapper
from activation_df import ActivationDataFrame
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import wandb

from utils.ipi_prompts import build_ipi_chat_prompt
from utils.ipi_surrogate import resolve_option_scores


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

    # W&B configuration
    wandb_cfg = cfg.get('wandb', {})

    # Resolve input path: prefer W&B artifact if provided
    dataset_artifact_name = cfg.data.get('dataset_artifact_name', None)
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(wandb_config, dict):
        wandb_config["resolved_seeds"] = resolved_seeds_to_dict(resolved)
    if dataset_artifact_name:
        # Initialize W&B early to download artifact
        wandb.init(
            project=wandb_cfg.get('project', 'activation-bias-classifier'),
            name=wandb_cfg.get('run_name', None),
            job_type="extraction",
            config=wandb_config,
        )
        print(f"Downloading dataset artifact: {dataset_artifact_name}")
        artifact = wandb.use_artifact(dataset_artifact_name, type='dataset')
        artifact_dir = artifact.download()
        # Find CSV file in artifact
        csv_files = [f for f in os.listdir(artifact_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError(
                f"No CSV file found in artifact {dataset_artifact_name}")
        input_path = os.path.join(artifact_dir, csv_files[0])
        print(f"Using dataset from artifact: {input_path}")
        wandb_initialized = True
    else:
        input_path = hydra.utils.to_absolute_path(cfg.data.feature_selection_dataset)
        wandb_initialized = False

    # Determine artifact name based on dataset and model
    dataset_name = os.path.basename(input_path).replace(
        '.csv', '').replace('_propositions', '')
    model_name = cfg.model.name.split('/')[-1]
    layers = cfg.extraction.layers
    layers_str = 'all' if layers == 'all' else f"L{str(layers)}"

    # Orchestrator-driven name takes precedence over the dataset-derived default
    # so a sweep can pin a deterministic identity (see src/utils/experiment_ids.py).
    artifacts_cfg = cfg.get('artifacts', {}) or {}
    override_name = artifacts_cfg.get('activations_name', None)
    artifact_name = (
        str(override_name)
        if override_name
        else f"activations-{dataset_name}-{model_name}-{layers_str}"
    )
    output_path = f"data/{artifact_name}.parquet"

    # Initialize W&B with job_type="extraction" (if not already initialized for artifact download)
    if not wandb_initialized:
        wandb.init(
            project=wandb_cfg.get('project', 'activation-bias-classifier'),
            name=wandb_cfg.get('run_name', None),
            job_type="extraction",
        )

    # Update W&B config
    wandb_run_config = {
        'feature_selection_dataset': input_path,
        'dataset_artifact': dataset_artifact_name,
        'output_file': output_path,
        'batch_size': batch_size,
        'device': device,
        'layers': layers_str,
        'max_length': cfg.extraction.max_length,
        'model_name': cfg.model.name,
        'model_wrapper': cfg.model.wrapper,
        'extraction_seed': resolved.extraction,
        'resolved_seeds': resolved_seeds_to_dict(resolved),
        'prompt_mode': prompt_mode,
    }
    if cfg.model.wrapper == "gemma":
        wandb_run_config.update({
            'sae_width': cfg.extraction.get('sae_width', '65k'),
            'sae_l0': cfg.extraction.get('sae_l0', 'medium'),
            'sae_release': cfg.extraction.get(
                'sae_release', 'gemma-scope-2-4b-it-res'),
            'hook': 'resid_post',
        })
    wandb.config.update(wandb_run_config)

    # 1. Load Data
    print(f"Loading data from {input_path}...")
    # For demonstration, creating a dummy dataframe if file doesn't exist
    if not os.path.exists(input_path):
        print("Input file not found. Creating dummy data for demonstration.")
        df = pd.DataFrame({
            'statement': ["This is a test sentence.", "Another political statement.", "Short one."] * 10,
            'pol_label_human': ["neutral", "political", "neutral"] * 10
        })
    else:
        df = pd.read_csv(input_path)
        # Ensure columns exist
        if 'statement' not in df.columns or 'pol_label_human' not in df.columns:
            raise ValueError(
                "Input DataFrame must contain 'statement' and 'pol_label_human' columns.")

    print(f"Loaded {len(df)} samples.")

    # 2. Initialize Model using factory
    print(f"Initializing model...")
    wrapper = get_model_wrapper(cfg)
    if wrapper.model.tokenizer is None:
        raise ValueError("The model wrapper must have a tokenizer.")
    loaded_name = getattr(wrapper.model.cfg, "model_name", None) or cfg.model.name
    print(f"Loaded model: {loaded_name}")

    # Ensure tokenizer has a pad token
    if wrapper.model.tokenizer.pad_token is None:
        wrapper.model.tokenizer.pad_token = wrapper.model.tokenizer.eos_token

    option_scores = resolve_option_scores(cfg)

    # Resolve layers list (handle 'all' option). Gemma uses SAE-allowed layers.
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

    # 3. Initialize Accumulator with layer info
    activation_df = ActivationDataFrame(layers=layers, d_features=d_features)

    # 4. Processing Loop
    print("Starting extraction loop...")

    # Create batches
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

        # Tokenize
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
            # Wrapper handles device placement internally;
            # activations are returned on CPU (hooks use .detach().cpu())
            layer_activations = wrapper.get_layer_activations(
                input_ids, layers=layers)
        except Exception as e:
            print(f"Error processing batch {start_idx}-{end_idx}: {e}")
            continue

        padding_side = wrapper.model.tokenizer.padding_side

        batch_indices = torch.arange(input_ids.shape[0])

        if padding_side == 'left':
            last_token_indices = -1
            final_activations = layer_activations[:, -1, :]
        else:
            last_token_indices = attention_mask.sum(dim=1).long() - 1
            final_activations = layer_activations[batch_indices,
                                                  last_token_indices, :]

        # Add to accumulator
        activation_df.add_batch(
            final_activations,
            labels,
            metadata={
                "pair_key": pair_keys,
                "row_id": row_ids,
                "statement": statements,
            },
        )

    # 5. Save Results
    print(f"Saving results to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    activation_df.save(output_path)
    full_output_path = os.path.join(os.getcwd(), output_path)
    print(f"Done. Saved to {full_output_path}")

    # --- ARTIFACT: Log activations as versioned dataset artifact ---

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

    activations_artifact = wandb.Artifact(
        name=artifact_name,
        type="dataset",
        description=f"Extracted activations from {cfg.model.name} on {dataset_name} dataset",
        metadata=artifact_metadata,
    )
    activations_artifact.add_file(full_output_path)
    wandb.log_artifact(activations_artifact)
    print(f"Activations artifact logged: {artifact_name}")

    # Log summary metrics
    summary = {
        'n_samples': total_samples,
        'n_layers': len(layers),
        'd_features': d_features,
        'n_features': len(layers) * d_features,
    }
    if is_gemma_sae:
        summary['d_sae'] = d_features
    else:
        summary['d_model'] = d_features
    wandb.summary.update(summary)

    # Finish W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
