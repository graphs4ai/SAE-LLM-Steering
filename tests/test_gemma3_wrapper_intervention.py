from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import torch
import torch.nn as nn

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Avoid pulling sae_lens / transformer_lens during lightweight unit tests.
if "sae_lens" not in sys.modules:
    sae_lens = ModuleType("sae_lens")
    sae_lens.__path__ = []  # mark as package
    sae_lens.SAE = type("SAE", (), {})

    bridge_mod = ModuleType("sae_lens.analysis.sae_transformer_bridge")
    bridge_mod.SAETransformerBridge = MagicMock()

    analysis_pkg = ModuleType("sae_lens.analysis")
    analysis_pkg.__path__ = []

    loading_pkg = ModuleType("sae_lens.loading")
    loading_pkg.__path__ = []
    loaders_mod = ModuleType("sae_lens.loading.pretrained_sae_loaders")
    loaders_mod.get_safetensors_tensor_shapes = lambda *a, **k: {}

    sys.modules["sae_lens"] = sae_lens
    sys.modules["sae_lens.analysis"] = analysis_pkg
    sys.modules["sae_lens.analysis.sae_transformer_bridge"] = bridge_mod
    sys.modules["sae_lens.loading"] = loading_pkg
    sys.modules["sae_lens.loading.pretrained_sae_loaders"] = loaders_mod

from gemma_3_wrapper import Gemma3Wrapper, multi_gpu_boot_kwargs


D_MODEL = 8
N_LAYERS = 26
VOCAB = 32


def test_multi_gpu_boot_kwargs_emits_integer_budgets() -> None:
    """Regression: max_memory values must be ints, never the string 'auto'."""
    import gemma_3_wrapper as gw

    class _Props:
        total_memory = 40 * (1024**3)

    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 3

        @staticmethod
        def get_device_properties(i: int) -> _Props:
            del i
            return _Props()

    original_cuda = gw.torch.cuda
    gw.torch.cuda = _FakeCuda  # type: ignore[assignment]
    try:
        kwargs = multi_gpu_boot_kwargs(2)
    finally:
        gw.torch.cuda = original_cuda

    assert kwargs["device_map"] == "balanced"
    assert kwargs["max_memory"] == {
        0: 40 * (1024**3),
        1: 40 * (1024**3),
        2: 0,
    }
    assert all(isinstance(v, int) for v in kwargs["max_memory"].values())
    assert "auto" not in kwargs["max_memory"].values()


class MockBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 1
        self.pad_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [hash(text) % VOCAB]


class MockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(
            [MockBlock(D_MODEL) for _ in range(N_LAYERS)]
        )
        self.unembed = nn.Linear(D_MODEL, VOCAB, bias=False)
        self.tokenizer = MockTokenizer()
        self.cfg = SimpleNamespace(n_layers=N_LAYERS, device="cpu")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 10,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        for _ in range(max_new_tokens):
            logits = self(input_ids)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids


class MockSAE:
    def __init__(self, d_sae: int = 4) -> None:
        self.W_dec = torch.randn(d_sae, D_MODEL)
        self.cfg = SimpleNamespace(
            d_sae=d_sae,
            metadata=SimpleNamespace(hook_name="blocks.17.hook_resid_post"),
        )


def make_test_wrapper() -> Gemma3Wrapper:
    wrapper = Gemma3Wrapper.__new__(Gemma3Wrapper)
    wrapper.input_device = "cpu"
    wrapper.device = "cpu"
    wrapper.n_devices = 1
    wrapper.dtype = torch.float32
    wrapper.sae_release = "gemma-scope-2-4b-it-res"
    wrapper.allowed_sae_layers = (9, 17, 22, 29)
    wrapper.model = MockModel()
    wrapper.saes = {
        17: MockSAE(),
        22: MockSAE(),
    }
    wrapper.sae_hook_names = {
        17: "blocks.17.hook_out",
        22: "blocks.22.hook_out",
    }
    wrapper.residual_sites = {}
    wrapper._d_sae = 4
    return wrapper


def test_device_for_layer_matches_block_device(wrapper: Gemma3Wrapper) -> None:
    # Place layers on distinct logical devices via meta-less CPU tensors with
    # different .device tags by moving one block's params (still CPU).
    site_17 = wrapper._get_residual_site(17)
    assert wrapper.device_for_layer(17) == next(site_17.parameters()).device
    assert wrapper.device_for_layer(22) == next(
        wrapper._get_residual_site(22).parameters()
    ).device


def test_boot_kwargs_pass_n_devices() -> None:
    captured: dict[str, Any] = {}

    class FakeBridge:
        @staticmethod
        def boot_transformers(model_name: str, **kwargs: Any) -> MockModel:
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs
            model = MockModel()
            model.cfg = SimpleNamespace(
                n_layers=N_LAYERS,
                device="cuda:0",
                n_devices=2,
            )
            return model

    import gemma_3_wrapper as gw

    original = gw.SAETransformerBridge
    original_multi = gw.multi_gpu_boot_kwargs
    gw.SAETransformerBridge = FakeBridge
    # Avoid requiring 2 real GPUs in CI; still assert the intended kwargs shape.
    gw.multi_gpu_boot_kwargs = lambda n: {
        "device_map": "balanced",
        "max_memory": {i: 40 * (1024**3) for i in range(n)},
    }
    try:
        wrapper = Gemma3Wrapper(
            model_name="google/gemma-3-4b-it",
            device="cuda",
            dtype=torch.bfloat16,
            n_devices=2,
            sae_layers=None,
        )
    finally:
        gw.SAETransformerBridge = original
        gw.multi_gpu_boot_kwargs = original_multi

    assert "n_devices" not in captured["kwargs"]
    assert captured["kwargs"].get("device_map") == "balanced"
    assert isinstance(captured["kwargs"].get("max_memory"), dict)
    assert all(
        isinstance(v, int) for v in captured["kwargs"]["max_memory"].values()
    )
    assert "device" not in captured["kwargs"]
    assert wrapper.n_devices == 2
    assert wrapper.input_device == "cuda:0"


def test_boot_kwargs_single_device_passes_device() -> None:
    captured: dict[str, Any] = {}

    class FakeBridge:
        @staticmethod
        def boot_transformers(model_name: str, **kwargs: Any) -> MockModel:
            captured["kwargs"] = kwargs
            model = MockModel()
            model.cfg = SimpleNamespace(n_layers=N_LAYERS, device="cpu", n_devices=1)
            return model

    import gemma_3_wrapper as gw

    original = gw.SAETransformerBridge
    gw.SAETransformerBridge = FakeBridge
    try:
        wrapper = Gemma3Wrapper(
            model_name="google/gemma-3-4b-it",
            device="cpu",
            dtype=torch.float32,
            n_devices=1,
            sae_layers=None,
        )
    finally:
        gw.SAETransformerBridge = original

    assert captured["kwargs"].get("device") == "cpu"
    assert "n_devices" not in captured["kwargs"]
    assert wrapper.n_devices == 1
    assert wrapper.input_device == "cpu"


def _count_forward_hooks(module: nn.Module) -> int:
    return len(module._forward_hooks)


def test_no_intervention_matches_baseline(wrapper: Gemma3Wrapper) -> None:
    input_ids = torch.tensor([[3, 5, 7]])
    baseline = wrapper._forward_logits(input_ids, None)
    again = wrapper._forward_logits(input_ids, None)
    assert torch.equal(baseline, again)


def test_alpha_zero_intervention_fires_and_matches_baseline(
    wrapper: Gemma3Wrapper,
) -> None:
    input_ids = torch.tensor([[3, 5, 7]])
    baseline = wrapper._forward_logits(input_ids, None)
    multipliers = {"layer_17-feature_0": 0.0}

    hook_records = wrapper._build_delta_module_hooks(
        activation_multipliers=multipliers,
        input_len=int(input_ids.shape[1]),
        intervention_scope="all_tokens",
        last_k=3,
    )

    with torch.no_grad():
        with wrapper._registered_module_hooks(hook_records):
            logits = wrapper.model(input_ids)

    assert hook_records[0]["state"]["calls"] > 0
    assert torch.allclose(baseline, logits)


def test_nonzero_intervention_changes_logits(wrapper: Gemma3Wrapper) -> None:
    input_ids = torch.tensor([[3, 5, 7]])
    baseline = wrapper._forward_logits(input_ids, None)
    multipliers = {"layer_17-feature_0": 100.0}

    hook_records = wrapper._build_delta_module_hooks(
        activation_multipliers=multipliers,
        input_len=int(input_ids.shape[1]),
        intervention_scope="all_tokens",
        last_k=3,
    )

    with torch.no_grad():
        with wrapper._registered_module_hooks(hook_records):
            logits = wrapper.model(input_ids)

    assert hook_records[0]["state"]["calls"] > 0
    max_diff = (logits - baseline).abs().max().item()
    assert max_diff > 0.0


def test_multi_layer_intervention_fires_all_hooks(
    wrapper: Gemma3Wrapper,
) -> None:
    input_ids = torch.tensor([[3, 5, 7]])
    multipliers = {
        "layer_17-feature_0": 50.0,
        "layer_22-feature_1": 50.0,
    }

    hook_records = wrapper._build_delta_module_hooks(
        activation_multipliers=multipliers,
        input_len=int(input_ids.shape[1]),
        intervention_scope="all_tokens",
        last_k=3,
    )

    with torch.no_grad():
        with wrapper._registered_module_hooks(hook_records):
            wrapper.model(input_ids)

    assert len(hook_records) == 2
    assert all(record["state"]["calls"] > 0 for record in hook_records)


def test_generation_intervention_fires_hooks_and_debug_seq_lens(
    wrapper: Gemma3Wrapper,
) -> None:
    input_ids = torch.tensor([[3, 5, 7]])
    debug_seq_lens: list[int] = []
    multipliers = {"layer_17-feature_0": 25.0}

    hook_records = wrapper._build_delta_module_hooks(
        activation_multipliers=multipliers,
        input_len=int(input_ids.shape[1]),
        intervention_scope="generated_only",
        last_k=3,
        debug_seq_lens=debug_seq_lens,
    )

    with torch.no_grad():
        with wrapper._registered_module_hooks(hook_records):
            wrapper.model.generate(input_ids, max_new_tokens=3)

    assert hook_records[0]["state"]["calls"] > 0
    assert len(debug_seq_lens) > 0


def test_hook_cleanup_on_success_and_failure(wrapper: Gemma3Wrapper) -> None:
    input_ids = torch.tensor([[3, 5, 7]])
    multipliers = {"layer_17-feature_0": 10.0}
    site = wrapper._get_residual_site(17)
    baseline_hooks = _count_forward_hooks(site)

    hook_records = wrapper._build_delta_module_hooks(
        activation_multipliers=multipliers,
        input_len=int(input_ids.shape[1]),
        intervention_scope="all_tokens",
        last_k=3,
    )

    with torch.no_grad():
        with wrapper._registered_module_hooks(hook_records):
            wrapper._forward_logits(
                input_ids,
                multipliers,
                intervention_scope="all_tokens",
            )

    assert _count_forward_hooks(site) == baseline_hooks

    broken_records = wrapper._build_delta_module_hooks(
        activation_multipliers=multipliers,
        input_len=int(input_ids.shape[1]),
        intervention_scope="all_tokens",
        last_k=3,
    )

    def broken_forward(_input_ids: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("simulated failure")

    original_forward = wrapper.model.forward
    wrapper.model.forward = broken_forward
    try:
        raised = False
        try:
            with torch.no_grad():
                with wrapper._registered_module_hooks(broken_records):
                    wrapper.model(input_ids)
        except RuntimeError as exc:
            raised = exc.args[0] == "simulated failure"
        assert raised
    finally:
        wrapper.model.forward = original_forward

    assert _count_forward_hooks(site) == baseline_hooks

    baseline_a = wrapper._forward_logits(input_ids, None)
    baseline_b = wrapper._forward_logits(input_ids, None)
    assert torch.equal(baseline_a, baseline_b)


def test_forward_logits_intervention_path(wrapper: Gemma3Wrapper) -> None:
    input_ids = torch.tensor([[3, 5, 7]])
    baseline = wrapper._forward_logits(input_ids, None)
    multipliers = {"layer_17-feature_0": 100.0}
    logits = wrapper._forward_logits(
        input_ids,
        multipliers,
        intervention_scope="all_tokens",
    )
    assert (logits - baseline).abs().max().item() > 0.0


def _run_all() -> None:
    wrapper = make_test_wrapper()
    tests = [
        test_no_intervention_matches_baseline,
        test_alpha_zero_intervention_fires_and_matches_baseline,
        test_nonzero_intervention_changes_logits,
        test_multi_layer_intervention_fires_all_hooks,
        test_generation_intervention_fires_hooks_and_debug_seq_lens,
        test_hook_cleanup_on_success_and_failure,
        test_forward_logits_intervention_path,
        test_device_for_layer_matches_block_device,
    ]
    for test in tests:
        test(wrapper)
        print(f"passed: {test.__name__}")

    test_multi_gpu_boot_kwargs_emits_integer_budgets()
    print("passed: test_multi_gpu_boot_kwargs_emits_integer_budgets")
    test_boot_kwargs_pass_n_devices()
    print("passed: test_boot_kwargs_pass_n_devices")
    test_boot_kwargs_single_device_passes_device()
    print("passed: test_boot_kwargs_single_device_passes_device")


if __name__ == "__main__":
    _run_all()
    print("All smoke tests passed.")
