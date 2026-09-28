import pytest
from vllm.transformers_utils.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

from vllm_gguf_plugin.weights_adapter import (
    GGUFModelFiles,
    Qwen35GGUFAdapter,
    get_adapter_architecture,
)
from vllm_gguf_plugin.weights_adapter import qwen3_5 as qwen3_5_adapter


@pytest.fixture
def no_gguf_patching(monkeypatch):
    # patch_hf_config reads GGUF metadata; the architecture choice does not depend on it
    monkeypatch.setattr(
        qwen3_5_adapter,
        "maybe_patch_hf_config_from_gguf",
        lambda path, config, mmproj_path=None: config,
    )


def text_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(num_hidden_layers=4)


def test_text_only_config_resolves_to_causal_lm():
    assert get_adapter_architecture(text_config()) == "Qwen3_5ForCausalLM"


def test_multimodal_config_resolves_to_conditional_generation():
    config = Qwen3_5Config(text_config=text_config().to_dict())
    assert get_adapter_architecture(config) == "Qwen3_5ForConditionalGeneration"


def test_text_only_gguf_without_mmproj_loads_as_causal_lm(no_gguf_patching):
    files = GGUFModelFiles(backbone=("model.gguf",))
    patched = Qwen35GGUFAdapter().patch_hf_config(files, text_config())

    assert patched.model_type == "qwen3_5_text"
    assert patched.architectures == ["Qwen3_5ForCausalLM"]


def test_text_config_with_mmproj_is_wrapped_as_multimodal(no_gguf_patching):
    files = GGUFModelFiles(backbone=("model.gguf",), mm_proj="mmproj.gguf")
    patched = Qwen35GGUFAdapter().patch_hf_config(files, text_config())

    assert isinstance(patched, Qwen3_5Config)
    assert patched.architectures == ["Qwen3_5ForConditionalGeneration"]
