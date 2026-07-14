from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from sae_lens import SAE
from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

from utils.hf_sae_shapes import install_local_safetensors_shape_patch
from utils.intervention_hooks import (
    DEFAULT_LAST_K,
    DEFAULT_SCOPE,
    _build_mask,
    assert_scope,
)
from utils.sae_steering import build_normalized_steering_vector, latent_clamp_delta


# Prefer local HF cache for sae_lens safetensors header probes (Xet CDN 403).
install_local_safetensors_shape_patch()

# Gemma Scope 2 resid_post SAEs are only published for these layers.
ALLOWED_SAE_LAYERS: tuple[int, ...] = (9, 17, 22, 29)
DEFAULT_SAE_WIDTH = "65k"
DEFAULT_SAE_L0 = "medium"
DEFAULT_SAE_RELEASE = "gemma-scope-2-4b-it-res"
INTERVENTION_MODE = "sae_decoded_delta_additive"
DEFAULT_DECODER_NORMALIZATION = "raw"

# Two mutually exclusive intervention mechanisms (see `_build_delta_module_hooks`):
#   decoder_delta_additive - static per-layer vector sum_j alpha_j * W_dec[j],
#                             identical at every masked position.
#   latent_clamp_additive  - per-token: z = sae.encode(x), z'_j = ReLU(z_j + alpha_j),
#                             delta = (z' - z) @ W_dec, added at each masked position.
EDIT_MODE_DECODER_DELTA = "decoder_delta_additive"
EDIT_MODE_LATENT_CLAMP = "latent_clamp_additive"
VALID_EDIT_MODES: tuple[str, ...] = (EDIT_MODE_DECODER_DELTA, EDIT_MODE_LATENT_CLAMP)
DEFAULT_EDIT_MODE = EDIT_MODE_DECODER_DELTA

# SAE metadata uses classic TL aliases; on SAETransformerBridge those live at
# the BlockBridge hook names (see notebooks/gemma_scope2_playground.ipynb).
_TL_TO_BRIDGE = {
    "hook_resid_pre": "hook_in",
    "hook_resid_mid": "ln2.hook_in",
    "hook_resid_post": "hook_out",
}


def parse_feature_coefficient_name(name: str) -> Tuple[int, int]:
    """Parse ``layer_{L}-feature_{F}`` (also accepts ``neuron`` for compatibility)."""
    parts = name.split("-")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid feature coefficient name {name!r}. "
            "Expected 'layer_{L}-feature_{F}'."
        )
    layer = int(parts[0].split("_")[1])
    feature = int(parts[1].split("_")[1])
    return layer, feature


class Gemma3Wrapper:
    """
    Wrapper for Gemma-3 models using SAETransformerBridge + Gemma Scope 2 SAEs.

    Extracts SAE latent features at resid_post for layers in
    ``ALLOWED_SAE_LAYERS``. Two intervention mechanisms are supported (see
    ``edit_mode``), both applied on resid_post / hook_out:

    - ``decoder_delta_additive`` (default): static per-layer vector
      ``x' = x + sum_j alpha_j * W_dec[j]``, identical at every masked
      position.
    - ``latent_clamp_additive``: per-token, ``z = sae.encode(x)``,
      ``z'_j = ReLU(z_j + alpha_j)`` for target features, then
      ``x' = x + (z' - z) @ W_dec``.
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        n_devices: int = 1,
        sae_layers: Optional[Union[List[int], str]] = None,
        sae_width: str = DEFAULT_SAE_WIDTH,
        sae_l0: str = DEFAULT_SAE_L0,
        sae_release: str = DEFAULT_SAE_RELEASE,
    ):
        """
        Args:
            model_name: Gemma-3 checkpoint (e.g. "google/gemma-3-4b-it")
            device: "cuda" or "cpu"
            dtype: torch.bfloat16 recommended for Gemma Scope 2
            n_devices: Ignored; SAETransformerBridge path is single-device.
            sae_layers: Layers to load SAEs for (subset of ALLOWED_SAE_LAYERS,
                        or "all"). If None, SAEs are loaded lazily on first
                        ``get_layer_activations`` call.
            sae_width: SAE width id (e.g. "65k")
            sae_l0: SAE L0 id (e.g. "medium")
            sae_release: HuggingFace SAE release name
        """
        if n_devices > 1:
            print(
                "Warning: Gemma3Wrapper ignores n_devices>1; "
                "SAETransformerBridge uses a single device."
            )

        self.device = device
        self.n_devices = 1
        self.dtype = dtype
        self.sae_width = sae_width
        self.sae_l0 = sae_l0
        self.sae_release = sae_release

        self.model = SAETransformerBridge.boot_transformers(
            model_name,
            device=device,
            dtype=dtype,
        )

        # Single-device path: inputs go to the model device.
        if getattr(self.model.cfg, "device", None) is not None:
            self.input_device = str(self.model.cfg.device)
        else:
            self.input_device = device

        if self.model.tokenizer.pad_token_id is None:
            self.model.tokenizer.pad_token_id = self.model.tokenizer.eos_token_id

        self.n_layers = self.model.cfg.n_layers

        self.saes: Dict[int, SAE] = {}
        # Cache hook name used by run_with_cache_with_saes.
        # Do not use this for causal intervention; use self.model.blocks[layer].
        self.sae_hook_names: Dict[int, str] = {}
        self.residual_sites: Dict[int, nn.Module] = {}
        self._d_sae: Optional[int] = None

        if sae_layers is not None:
            self.load_saes(sae_layers, width=sae_width, l0=sae_l0)

    @property
    def d_sae(self) -> int:
        if self._d_sae is None:
            raise RuntimeError(
                "No SAEs loaded yet. Call load_saes(...) or get_layer_activations(...)."
            )
        return self._d_sae

    def resolve_sae_layers(
        self, layers: Union[List[int], str]
    ) -> List[int]:
        """Expand and validate SAE layer indices against ALLOWED_SAE_LAYERS."""
        if isinstance(layers, str):
            if layers.lower() == "all":
                return list(ALLOWED_SAE_LAYERS)
            raise ValueError("layers must be a list or 'all'")

        if not layers:
            raise ValueError("layers list cannot be empty")

        resolved = [int(layer) for layer in layers]
        invalid = [layer for layer in resolved if layer not in ALLOWED_SAE_LAYERS]
        if invalid:
            raise ValueError(
                f"Unsupported SAE layers: {invalid}. "
                f"Allowed: {list(ALLOWED_SAE_LAYERS)}"
            )
        return resolved

    def _resolve_bridge_hook_name(self, alias: str) -> str:
        resolved = self.model._resolve_hook_name(alias)
        prefix, _, leaf = resolved.rpartition(".")
        if leaf in _TL_TO_BRIDGE:
            return f"{prefix}.{_TL_TO_BRIDGE[leaf]}"
        return resolved

    def load_saes(
        self,
        layers: Union[List[int], str],
        width: Optional[str] = None,
        l0: Optional[str] = None,
        release: Optional[str] = None,
    ) -> None:
        """Load Gemma Scope 2 resid_post SAEs for the requested layers."""
        width = width or self.sae_width
        l0 = l0 or self.sae_l0
        release = release or self.sae_release
        self.sae_width = width
        self.sae_l0 = l0
        self.sae_release = release

        resolved_layers = self.resolve_sae_layers(layers)
        sae_device = self.input_device

        for layer in resolved_layers:
            if layer in self.saes:
                continue

            sae_id = f"layer_{layer}_width_{width}_l0_{l0}"
            sae, _cfg_dict, _sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
                release=release,
                sae_id=sae_id,
                device=sae_device,
                dtype=self.dtype,
            )
            hook_alias = sae.cfg.metadata.hook_name
            hook_name = self._resolve_bridge_hook_name(hook_alias)

            self.saes[layer] = sae
            self.sae_hook_names[layer] = hook_name

            if self._d_sae is None:
                self._d_sae = int(sae.cfg.d_sae)
            elif int(sae.cfg.d_sae) != self._d_sae:
                raise RuntimeError(
                    f"SAE d_sae mismatch at layer {layer}: "
                    f"got {sae.cfg.d_sae}, expected {self._d_sae}"
                )

    def _parse_layer_feature_alphas(
        self,
        activation_multipliers: Dict[str, float],
    ) -> Dict[int, Dict[int, float]]:
        """Group additive coefficients alpha_j by layer."""
        layer_feature_alphas: Dict[int, Dict[int, float]] = {}
        for name, alpha in activation_multipliers.items():
            layer, feature = parse_feature_coefficient_name(name)
            if layer not in ALLOWED_SAE_LAYERS:
                raise ValueError(
                    f"Feature {name!r} uses unsupported SAE layer {layer}. "
                    f"Allowed: {list(ALLOWED_SAE_LAYERS)}"
                )
            layer_feature_alphas.setdefault(layer, {})[feature] = float(alpha)
        return layer_feature_alphas

    def _ensure_saes_for_coefficients(
        self,
        layer_feature_alphas: Dict[int, Dict[int, float]],
    ) -> None:
        missing = [
            layer for layer in layer_feature_alphas if layer not in self.saes
        ]
        if missing:
            self.load_saes(missing)

    def _build_steering_vector(
        self,
        layer: int,
        feature_alphas: Dict[int, float],
        decoder_normalization: str = DEFAULT_DECODER_NORMALIZATION,
    ) -> torch.Tensor:
        """Linear-decoder Option-2 delta: sum_j alpha_j * W_dec[j]."""
        sae = self.saes[layer]
        w_dec = sae.W_dec
        d_sae = int(sae.cfg.d_sae)
        for feature_idx, alpha in feature_alphas.items():
            if feature_idx < 0 or feature_idx >= d_sae:
                raise ValueError(
                    f"Feature index {feature_idx} out of range for layer {layer} "
                    f"(d_sae={d_sae})."
                )
        if decoder_normalization == "unit_norm":
            return build_normalized_steering_vector(
                w_dec=w_dec,
                feature_coefficients=feature_alphas,
            )
        if decoder_normalization != "raw":
            raise ValueError(
                f"Unknown decoder_normalization={decoder_normalization!r}. "
                "Expected 'raw' or 'unit_norm'."
            )
        steering = torch.zeros(
            w_dec.shape[1],
            device=w_dec.device,
            dtype=w_dec.dtype,
        )
        for feature_idx, alpha in feature_alphas.items():
            if alpha == 0.0:
                continue
            steering = steering + alpha * w_dec[feature_idx]
        return steering

    def _get_residual_site(self, layer: int) -> nn.Module:
        if layer in self.residual_sites:
            return self.residual_sites[layer]

        if not hasattr(self.model, "blocks"):
            raise AttributeError(
                "Expected SAETransformerBridge model to expose `model.blocks`, "
                "but it was not found."
            )
        if layer < 0 or layer >= len(self.model.blocks):
            raise ValueError(
                f"Layer {layer} out of range for model.blocks length "
                f"{len(self.model.blocks)}."
            )

        site = self.model.blocks[layer]
        self.residual_sites[layer] = site
        return site

    @staticmethod
    def _split_module_output(
        output: Any,
    ) -> Tuple[Any, Optional[Tuple[Any, ...]]]:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
            return hidden, rest
        return output, None

    @staticmethod
    def _rebuild_module_output(
        hidden: Any,
        rest: Optional[Tuple[Any, ...]],
    ) -> Any:
        if rest is None:
            return hidden
        return (hidden, *rest)

    def _make_residual_steering_module_hook(
        self,
        steering_vector: torch.Tensor,
        input_len: int,
        scope: str,
        last_k: int,
        debug_seq_lens: Optional[List[int]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        state: Dict[str, Any] = {"calls": 0, "seq_lens": []}
        steering_vector = steering_vector.detach()

        def hook_fn(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            state["calls"] += 1

            hidden, rest = self._split_module_output(output)

            if not torch.is_tensor(hidden):
                raise TypeError(f"Expected tensor hidden state, got {type(hidden)}")

            if hidden.ndim != 3:
                raise ValueError(
                    f"Expected hidden [batch, seq_len, d_model], got {tuple(hidden.shape)}"
                )

            batch, seq_len, d_model = hidden.shape
            del batch
            state["seq_lens"].append(int(seq_len))
            if debug_seq_lens is not None and len(debug_seq_lens) < 20:
                debug_seq_lens.append(int(seq_len))

            delta = steering_vector.to(device=hidden.device, dtype=hidden.dtype)

            if delta.ndim != 1:
                raise ValueError(f"Expected steering vector 1D, got {tuple(delta.shape)}")

            if delta.shape[0] != d_model:
                raise ValueError(
                    f"Steering vector dim {delta.shape[0]} does not match d_model {d_model}"
                )

            out = hidden.clone()

            if scope == "generated_only":
                # Skip prefill; apply on later decode calls (KV-cache safe).
                is_prefill = state["calls"] == 1 and seq_len >= input_len
                if is_prefill:
                    return output
                out[:, -1:, :] += delta
            else:
                mask = _build_mask(
                    seq_len=seq_len,
                    input_len=input_len,
                    scope=scope,
                    last_k=last_k,
                    device=hidden.device,
                )
                if not bool(mask.any()):
                    return output
                out[:, mask, :] = out[:, mask, :] + delta

            return self._rebuild_module_output(out, rest)

        return hook_fn, state

    def _make_latent_clamp_module_hook(
        self,
        sae: SAE,
        feature_alphas: Dict[int, float],
        input_len: int,
        scope: str,
        last_k: int,
        debug_seq_lens: Optional[List[int]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Per-token latent-space intervention hook.

        Unlike `_make_residual_steering_module_hook` (a fixed vector added
        identically everywhere), this encodes the *real* residual stream at
        each masked position, additively clamps target latents, and injects
        only the resulting reconstruction delta. See `latent_clamp_delta`.
        """
        state: Dict[str, Any] = {"calls": 0, "seq_lens": [], "delta_norms": []}

        def hook_fn(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            state["calls"] += 1

            hidden, rest = self._split_module_output(output)

            if not torch.is_tensor(hidden):
                raise TypeError(f"Expected tensor hidden state, got {type(hidden)}")

            if hidden.ndim != 3:
                raise ValueError(
                    f"Expected hidden [batch, seq_len, d_model], got {tuple(hidden.shape)}"
                )

            batch, seq_len, d_model = hidden.shape
            del batch, d_model
            state["seq_lens"].append(int(seq_len))
            if debug_seq_lens is not None and len(debug_seq_lens) < 20:
                debug_seq_lens.append(int(seq_len))

            out = hidden.clone()

            def _apply_delta(x_masked: torch.Tensor) -> torch.Tensor:
                delta = latent_clamp_delta(sae, x_masked, feature_alphas)
                delta = delta.to(device=hidden.device, dtype=hidden.dtype)
                state["delta_norms"].append(
                    float(delta.float().norm(dim=-1).mean().item())
                )
                return x_masked + delta

            if scope == "generated_only":
                is_prefill = state["calls"] == 1 and seq_len >= input_len
                if is_prefill:
                    return output
                out[:, -1:, :] = _apply_delta(out[:, -1:, :])
            else:
                mask = _build_mask(
                    seq_len=seq_len,
                    input_len=input_len,
                    scope=scope,
                    last_k=last_k,
                    device=hidden.device,
                )
                if not bool(mask.any()):
                    return output
                out[:, mask, :] = _apply_delta(out[:, mask, :])

            return self._rebuild_module_output(out, rest)

        return hook_fn, state

    def _build_delta_module_hooks(
        self,
        activation_multipliers: Dict[str, float],
        input_len: int,
        intervention_scope: str,
        last_k: int,
        decoder_normalization: str = DEFAULT_DECODER_NORMALIZATION,
        edit_mode: str = DEFAULT_EDIT_MODE,
        debug_seq_lens: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        assert_scope(intervention_scope)
        if edit_mode not in VALID_EDIT_MODES:
            raise ValueError(
                f"Unknown edit_mode={edit_mode!r}. Expected one of {VALID_EDIT_MODES}."
            )
        if edit_mode == EDIT_MODE_LATENT_CLAMP and decoder_normalization != "raw":
            raise ValueError(
                f"decoder_normalization={decoder_normalization!r} is not "
                f"meaningful with edit_mode={EDIT_MODE_LATENT_CLAMP!r}: latent-"
                "space clamping always decodes through the SAE's raw W_dec. "
                "Normalization only applies to edit_mode="
                f"{EDIT_MODE_DECODER_DELTA!r}."
            )

        layer_feature_alphas = self._parse_layer_feature_alphas(
            activation_multipliers
        )
        self._ensure_saes_for_coefficients(layer_feature_alphas)

        hook_records: List[Dict[str, Any]] = []

        for layer, feature_alphas in layer_feature_alphas.items():
            residual_site = self._get_residual_site(layer)

            if edit_mode == EDIT_MODE_LATENT_CLAMP:
                sae = self.saes[layer]
                hook_fn, state = self._make_latent_clamp_module_hook(
                    sae=sae,
                    feature_alphas=feature_alphas,
                    input_len=input_len,
                    scope=intervention_scope,
                    last_k=last_k,
                    debug_seq_lens=debug_seq_lens,
                )
                hook_records.append(
                    {
                        "layer": layer,
                        "residual_site": residual_site,
                        "hook_fn": hook_fn,
                        "state": state,
                        "steering_norm": None,
                        "n_features": len(feature_alphas),
                        "edit_mode": edit_mode,
                    }
                )
                continue

            steering_vector = self._build_steering_vector(
                layer,
                feature_alphas,
                decoder_normalization=decoder_normalization,
            )

            hook_fn, state = self._make_residual_steering_module_hook(
                steering_vector=steering_vector,
                input_len=input_len,
                scope=intervention_scope,
                last_k=last_k,
                debug_seq_lens=debug_seq_lens,
            )

            hook_records.append(
                {
                    "layer": layer,
                    "residual_site": residual_site,
                    "hook_fn": hook_fn,
                    "state": state,
                    "steering_norm": float(steering_vector.float().norm().item()),
                    "n_features": len(feature_alphas),
                    "edit_mode": edit_mode,
                }
            )

        return hook_records

    @contextmanager
    def _registered_module_hooks(
        self,
        hook_records: List[Dict[str, Any]],
    ) -> Iterator[None]:
        handles: List[Any] = []
        try:
            for record in hook_records:
                handle = record["residual_site"].register_forward_hook(
                    record["hook_fn"]
                )
                handles.append(handle)
            yield
        finally:
            for handle in handles:
                handle.remove()

        for record in hook_records:
            if record["state"]["calls"] == 0:
                raise RuntimeError(
                    f"Residual steering hook for layer {record['layer']} did not fire. "
                    "Check residual_site resolution."
                )

    def _debug_print_intervention(
        self,
        hook_records: List[Dict[str, Any]],
    ) -> None:
        for record in hook_records:
            if record.get("edit_mode") == EDIT_MODE_LATENT_CLAMP:
                delta_norms = record["state"].get("delta_norms", [])
                mean_norm = sum(delta_norms) / len(delta_norms) if delta_norms else 0.0
                max_norm = max(delta_norms) if delta_norms else 0.0
                print(
                    f"[intervention] layer={record['layer']} "
                    f"site={type(record['residual_site']).__name__} "
                    f"edit_mode={record['edit_mode']} "
                    f"mean_delta_norm={mean_norm:.4f} max_delta_norm={max_norm:.4f} "
                    f"n_features={record['n_features']} "
                    f"hook_calls={record['state']['calls']} "
                    f"seq_lens={record['state']['seq_lens']}"
                )
            else:
                print(
                    f"[intervention] layer={record['layer']} "
                    f"site={type(record['residual_site']).__name__} "
                    f"steering_norm={record['steering_norm']:.4f} "
                    f"n_features={record['n_features']} "
                    f"hook_calls={record['state']['calls']} "
                    f"seq_lens={record['state']['seq_lens']}"
                )

    def get_layer_activations(
        self,
        tokens: torch.Tensor,
        layers: Union[List[int], str] = "all",
    ) -> torch.Tensor:
        """
        Returns SAE latent activations for selected layers, concatenated
        along the feature dimension.

        Shape: [batch, seq_len, n_layers * d_sae], layers in sorted order.
        """
        resolved_layers = self.resolve_sae_layers(layers)
        missing = [layer for layer in resolved_layers if layer not in self.saes]
        if missing:
            self.load_saes(missing)

        sae_list = [self.saes[layer] for layer in sorted(resolved_layers)]

        with torch.no_grad():
            _logits, cache = self.model.run_with_cache_with_saes(
                tokens.to(self.input_device),
                saes=sae_list,
            )

        activations: List[torch.Tensor] = []
        for layer in sorted(resolved_layers):
            hook_name = self.sae_hook_names[layer]
            cache_key = f"{hook_name}.hook_sae_acts_post"
            if cache_key not in cache:
                raise RuntimeError(
                    f"Missing SAE feature activations for layer {layer} "
                    f"(cache key {cache_key!r})"
                )
            activations.append(cache[cache_key].detach().cpu())

        return torch.cat(activations, dim=-1)

    def generate_with_intervention(
        self,
        input_ids: torch.Tensor,
        activation_multipliers: Optional[Dict[str, float]] = None,
        max_new_tokens: int = 10,
        temperature: Optional[float] = None,
        do_sample: bool = False,
        stop_at_eos: bool = True,
        eos_token_id: Optional[int] = None,
        verbose: bool = False,
        intervention_scope: str = DEFAULT_SCOPE,
        last_k: int = DEFAULT_LAST_K,
        decoder_normalization: str = DEFAULT_DECODER_NORMALIZATION,
        edit_mode: str = DEFAULT_EDIT_MODE,
        debug_seq_lens: Optional[List[int]] = None,
        debug_intervention: bool = False,
        **generate_kwargs,
    ) -> torch.Tensor:
        if eos_token_id is None:
            eos_token_id = self.model.tokenizer.eos_token_id

        input_ids = input_ids.to(self.input_device)
        generate_kwargs_common = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature is not None else 1.0,
            do_sample=do_sample,
            stop_at_eos=stop_at_eos,
            eos_token_id=eos_token_id,
            verbose=verbose,
            **generate_kwargs,
        )

        if not activation_multipliers:
            return self.model.generate(input_ids, **generate_kwargs_common)

        input_len = int(input_ids.shape[-1])
        hook_records = self._build_delta_module_hooks(
            activation_multipliers=activation_multipliers,
            input_len=input_len,
            intervention_scope=intervention_scope,
            last_k=last_k,
            decoder_normalization=decoder_normalization,
            edit_mode=edit_mode,
            debug_seq_lens=debug_seq_lens,
        )

        with torch.no_grad():
            with self._registered_module_hooks(hook_records):
                output = self.model.generate(input_ids, **generate_kwargs_common)

        if debug_intervention:
            self._debug_print_intervention(hook_records)

        return output

    def get_stance_token_ids(self, language: str = "pt") -> Tuple[int, int]:
        """
        Gets the token IDs for positive (Agree) and negative (Disagree) stance words.
        Gemma is space-sensitive; keep leading space.
        """
        if language == "pt":
            pos = "Con"
            neg = "Dis"
        else:
            pos = "Agree"
            neg = "Disagree"

        pos_id = self.model.tokenizer.encode(pos, add_special_tokens=False)[0]
        neg_id = self.model.tokenizer.encode(neg, add_special_tokens=False)[0]
        return pos_id, neg_id

    def _forward_logits(
        self,
        input_ids: torch.Tensor,
        activation_multipliers: Optional[Dict[str, float]] = None,
        intervention_scope: str = DEFAULT_SCOPE,
        last_k: int = DEFAULT_LAST_K,
        decoder_normalization: str = DEFAULT_DECODER_NORMALIZATION,
        edit_mode: str = DEFAULT_EDIT_MODE,
        debug_intervention: bool = False,
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.input_device)

        if not activation_multipliers:
            with torch.no_grad():
                return self.model(input_ids)

        input_len = int(input_ids.shape[1])
        hook_records = self._build_delta_module_hooks(
            activation_multipliers=activation_multipliers,
            input_len=input_len,
            intervention_scope=intervention_scope,
            last_k=last_k,
            decoder_normalization=decoder_normalization,
            edit_mode=edit_mode,
        )

        with torch.no_grad():
            with self._registered_module_hooks(hook_records):
                logits = self.model(input_ids)

        if debug_intervention:
            self._debug_print_intervention(hook_records)

        return logits

    def get_soft_stance_score(
        self,
        input_ids: torch.Tensor,
        activation_multipliers: Optional[Dict[str, float]] = None,
        positive_token_id: Optional[int] = None,
        negative_token_id: Optional[int] = None,
        language: str = "en",
        intervention_scope: str = DEFAULT_SCOPE,
        last_k: int = DEFAULT_LAST_K,
        decoder_normalization: str = DEFAULT_DECODER_NORMALIZATION,
        edit_mode: str = DEFAULT_EDIT_MODE,
    ) -> Tuple[float, float]:
        if positive_token_id is None or negative_token_id is None:
            pos, neg = self.get_stance_token_ids(language)
            positive_token_id = positive_token_id or pos
            negative_token_id = negative_token_id or neg

        logits = self._forward_logits(
            input_ids=input_ids,
            activation_multipliers=activation_multipliers,
            intervention_scope=intervention_scope,
            last_k=last_k,
            decoder_normalization=decoder_normalization,
            edit_mode=edit_mode,
        )
        last_logits = logits[0, -1]
        probs = F.softmax(last_logits, dim=-1)

        p_pos = probs[positive_token_id].item()
        p_neg = probs[negative_token_id].item()

        return p_pos - p_neg, p_pos + p_neg

    def get_expected_ipi_score(
        self,
        input_ids: torch.Tensor,
        option_token_ids: dict[int, list[int]],
        activation_multipliers: Optional[Dict[str, float]] = None,
        intervention_scope: str = DEFAULT_SCOPE,
        last_k: int = DEFAULT_LAST_K,
        decoder_normalization: str = DEFAULT_DECODER_NORMALIZATION,
        edit_mode: str = DEFAULT_EDIT_MODE,
    ) -> float:
        from utils.ipi_surrogate import expected_ipi_from_logits

        logits = self._forward_logits(
            input_ids=input_ids,
            activation_multipliers=activation_multipliers,
            intervention_scope=intervention_scope,
            last_k=last_k,
            decoder_normalization=decoder_normalization,
            edit_mode=edit_mode,
        )
        return expected_ipi_from_logits(logits[0, -1, :], option_token_ids)
