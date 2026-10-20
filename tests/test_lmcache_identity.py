# SPDX-License-Identifier: Apache-2.0

from types import ModuleType, SimpleNamespace

import vllm_gguf_plugin.lmcache_identity as identity


def config(model, weights, dtype):
    return SimpleNamespace(
        model_config=SimpleNamespace(model=model, model_weights=weights),
        cache_config=SimpleNamespace(cache_dtype=dtype),
    )


def test_name_tells_gguf_models_and_kv_dtypes_apart(tmp_path):
    a, b = tmp_path / "swift.gguf", tmp_path / "base@1.gguf"
    a.write_bytes(b"x" * 10)
    b.write_bytes(b"x" * 11)
    swift = identity.lmcache_model_name(config("/hf-config", str(a), "fp8"))
    base = identity.lmcache_model_name(config("/hf-config", str(b), "fp8"))
    swift_nvfp4 = identity.lmcache_model_name(config("/hf-config", str(a), "nvfp4"))
    assert swift == "/hf-config/swift.gguf.a/kv-fp8"
    assert len({swift, base, swift_nvfp4}) == 3
    assert "@" not in base


def test_hf_model_keeps_its_name_plus_kv_dtype():
    name = identity.lmcache_model_name(config("Qwen/Qwen3.5-4B", "", "auto"))
    assert name == "Qwen/Qwen3.5-4B/kv-auto"


def test_adapter_factories_see_the_new_name_only(tmp_path):
    seen = []
    module = ModuleType("fake_connector")
    for fn in ("create_scheduler_adapter", "create_worker_adapter"):
        setattr(module, fn, lambda url, ctx, cfg, *a: seen.append(cfg.model_config.model))
    identity._patch_module(module)
    identity._patch_module(module)  # idempotent

    weights = tmp_path / "m.gguf"
    weights.write_bytes(b"x")
    cfg = config("/hf-config", str(weights), "fp8")
    module.create_scheduler_adapter("tcp://x:1", None, cfg, 1.0, 2.0)
    module.create_worker_adapter("tcp://x:1", None, cfg, 1.0, 2.0)
    assert seen == ["/hf-config/m.gguf.1/kv-fp8"] * 2
    assert cfg.model_config.model == "/hf-config"  # vLLM's own config untouched


def test_lmcache_connector_name_hook(tmp_path):
    module = ModuleType("lmcache_connector")
    module.get_dcp_decorated_model_name = lambda cfg, kv: cfg.model_config.model + "#dcp"
    identity._patch_module(module)
    weights = tmp_path / "m.gguf"
    weights.write_bytes(b"x")
    cfg = config("/hf-config", str(weights), "nvfp4")
    assert module.get_dcp_decorated_model_name(cfg, None) == "/hf-config/m.gguf.1/kv-nvfp4#dcp"
