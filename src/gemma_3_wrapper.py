import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from sae_lens import SAE
from sae_lens.analysis.sae_transformer_bridge import SAETransformerBridge

from utils.intervention_hooks import (
    DEFAULT_LAST_K,
    DEFAULT_SCOPE,
    assert_scope,
    make_delta_steering_hook,
)


# Gemma Scope 2 resid_post SAEs are only published for these layers.
ALLOWED_SAE_LAYERS: tuple[int, ...] = (9, 17, 22, 29)
DEFAULT_SAE_WIDTH = "65k"
DEFAULT_SAE_L0 = "medium"
DEFAULT_SAE_RELEASE = "gemma-scope-2-4b-it-res"
INTERVENTION_MODE = "sae_decoded_delta_additive"

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
    ``ALLOWED_SAE_LAYERS``. Interventions use Option-2 additive decoded-delta:
    ``x' = x + sum_j alpha_j * W_dec[j]`` on resid_post / hook_out.
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
        self.sae_hook_names: Dict[int, str] = {}
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
    ) -> torch.Tensor:
        """Linear-decoder Option-2 delta: sum_j alpha_j * W_dec[j]."""
        sae = self.saes[layer]
        w_dec = sae.W_dec
        d_sae = int(sae.cfg.d_sae)
        steering = torch.zeros(
            w_dec.shape[1],
            device=w_dec.device,
            dtype=w_dec.dtype,
        )
        for feature_idx, alpha in feature_alphas.items():
            if feature_idx < 0 or feature_idx >= d_sae:
                raise ValueError(
                    f"Feature index {feature_idx} out of range for layer {layer} "
                    f"(d_sae={d_sae})."
                )
            if alpha == 0.0:
                continue
            steering = steering + alpha * w_dec[feature_idx]
        return steering

    def _build_delta_hooks(
        self,
        activation_multipliers: Dict[str, float],
        input_len: int,
        intervention_scope: str,
        last_k: int,
        debug_seq_lens: Optional[List[int]] = None,
    ) -> List[Tuple[str, object]]:
        assert_scope(intervention_scope)
        layer_feature_alphas = self._parse_layer_feature_alphas(
            activation_multipliers
        )
        self._ensure_saes_for_coefficients(layer_feature_alphas)

        hooks: List[Tuple[str, object]] = []
        for layer, feature_alphas in layer_feature_alphas.items():
            steering_vector = self._build_steering_vector(layer, feature_alphas)
            hook_name = self.sae_hook_names[layer]
            hooks.append(
                (
                    hook_name,
                    make_delta_steering_hook(
                        steering_vector=steering_vector,
                        input_len=input_len,
                        scope=intervention_scope,
                        last_k=last_k,
                        debug_seq_lens=debug_seq_lens,
                    ),
                )
            )
        return hooks

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
        debug_seq_lens: Optional[List[int]] = None,
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
        hooks = self._build_delta_hooks(
            activation_multipliers=activation_multipliers,
            input_len=input_len,
            intervention_scope=intervention_scope,
            last_k=last_k,
            debug_seq_lens=debug_seq_lens,
        )

        with torch.no_grad():
            with self.model.hooks(fwd_hooks=hooks):
                return self.model.generate(input_ids, **generate_kwargs_common)

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
    ) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.input_device)

        if not activation_multipliers:
            with torch.no_grad():
                return self.model(input_ids)

        input_len = int(input_ids.shape[1])
        hooks = self._build_delta_hooks(
            activation_multipliers=activation_multipliers,
            input_len=input_len,
            intervention_scope=intervention_scope,
            last_k=last_k,
        )
        with torch.no_grad():
            return self.model.run_with_hooks(input_ids, fwd_hooks=hooks)

    def get_soft_stance_score(
        self,
        input_ids: torch.Tensor,
        activation_multipliers: Optional[Dict[str, float]] = None,
        positive_token_id: Optional[int] = None,
        negative_token_id: Optional[int] = None,
        language: str = "en",
        intervention_scope: str = DEFAULT_SCOPE,
        last_k: int = DEFAULT_LAST_K,
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
    ) -> float:
        from utils.ipi_surrogate import expected_ipi_from_logits

        logits = self._forward_logits(
            input_ids=input_ids,
            activation_multipliers=activation_multipliers,
            intervention_scope=intervention_scope,
            last_k=last_k,
        )
        return expected_ipi_from_logits(logits[0, -1, :], option_token_ids)
