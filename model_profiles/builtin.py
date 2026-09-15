"""Conservative candidate profiles for common small-model roles."""

from __future__ import annotations

from .schema import CapabilityState, ModelProfile


def _unknown_capabilities() -> dict[str, CapabilityState]:
    return {name: CapabilityState() for name in (
        "json_output",
        "tool_call_generation",
        "tool_result_reinjection",
        "multimodal",
        "thinking_control",
    )}


def builtin_profiles() -> tuple[ModelProfile, ...]:
    """Return profiles without binding to a local asset or claiming support."""

    common = {
        "context": {"n_ctx": 4096, "input_budget": 3072, "max_new_tokens": 768},
        "generation": {"temperature": 0.7, "top_p": 0.9, "thinking": "unknown"},
        "resources": {"kv_cache": "unknown", "gpu_layers": "auto"},
        "status": "candidate",
        "production_eligible": False,
        "evidence": {
            "fixture_set": "small-model-core-v1",
            "source": "builtin_candidate",
            "weights_loaded": False,
            "network_used": False,
        },
    }
    qwen3_common = {
        **common,
        "generation": {"temperature": 0.6, "top_p": 0.95, "thinking": "declared"},
        "resources": {
            "kv_cache": "unknown",
            "gpu_layers": "auto",
            "min_ram_gb": 4.0,
            "min_vram_gb": 2.0,
            "min_disk_gb": 3.0,
            "edge_compatible": True,
        },
    }
    return (
        ModelProfile(
            model_id="QW1.8B",
            revision="builtin-qw1-v1",
            backend="llama_server",
            adaptation={
                "prompt_family": "qwen_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            aliases=("qwen-1_8b",),
            capabilities=_unknown_capabilities(),
            **common,
        ),
        ModelProfile(
            model_id="Qwen3-0.6B",
            revision="builtin-qwen3-0.6b-v2",
            backend="transformers_sidecar",
            format="both",
            artifact_sha256="a455aef79b391edba6d07a05f0bacb09523395edf18f561aabb27cff72d73fa1",
            tokenizer_digest="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            chat_template_digest="d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
            adaptation={
                "prompt_family": "qwen3_chat_v1",
                "tool_mode": "sidecar_candidate",
                "structured_output": "grammar_first",
                "summary_mode": "state_schema_v1",
            },
            roles=("tool_router", "summarizer"),
            aliases=("qwen3-0.6b",),
            capabilities={
                **_unknown_capabilities(),
                "thinking_control": CapabilityState(
                    "declared", ("b1_template_probe_isolated_tokenizer",)
                ),
            },
            **qwen3_common,
        ),
        ModelProfile(
            model_id="Qwen2.5-0.5B",
            revision="builtin-qwen25-0.5b-v2",
            backend="llama_server",
            format="both",
            artifact_sha256="cae472e554dcf487f5116f2c84450844c36b4a704580e0983df78cd213aa8d3f",
            tokenizer_digest="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
            chat_template_digest="5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
            context={"n_ctx": 32768, "input_budget": 3072, "max_new_tokens": 512},
            generation={"temperature": 0.7, "top_p": 0.8, "thinking": "unknown"},
            adaptation={
                "prompt_family": "qwen_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            aliases=("qwen2.5-0.5b",),
            capabilities=_unknown_capabilities(),
            resources={
                "kv_cache": "unknown",
                "gpu_layers": "auto",
                "min_ram_gb": 3.0,
                "min_vram_gb": 1.5,
                "min_disk_gb": 2.0,
                "edge_compatible": True,
            },
            evidence={**common["evidence"], "core_model_id": "qwen2.5-0.5b", "probe_ticket": "M-SM-B1", "artifact_digest_mode": "manifest", "manifest_sha256": "40133469bc80b60b3e00680e221998a16809ee176c1e1a12c951798d6125a2b9"},
            status="candidate",
            production_eligible=False,
        ),
        ModelProfile(
            model_id="MiniCPM4-0.5B",
            revision="builtin-minicpm4-0.5b-v2",
            backend="llama_server",
            format="both",
            artifact_sha256="00cf2fd1bbd59ab1ea51c1f2623b63506e7db82316501688c4bfff99717b42a5",
            tokenizer_digest="adf7208af154a5ca065d2eda4e5419e02aac58c2c00627874748b75ec6769094",
            chat_template_digest="8bef75d59004eb0e62db89e97090d4736b14fd6c4a0282370c28833be1e83051",
            context={"n_ctx": 32768, "input_budget": 3072, "max_new_tokens": 512},
            generation={"temperature": 0.8, "top_p": 0.8, "thinking": "unknown"},
            adaptation={
                "prompt_family": "minicpm4_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            aliases=("minicpm4-0.5b",),
            capabilities=_unknown_capabilities(),
            resources={
                "kv_cache": "unknown",
                "gpu_layers": "auto",
                "min_ram_gb": 3.0,
                "min_vram_gb": 1.5,
                "min_disk_gb": 3.0,
                "edge_compatible": True,
            },
            evidence={**common["evidence"], "core_model_id": "minicpm4-0.5b", "probe_ticket": "M-SM-B1", "artifact_digest_mode": "manifest", "manifest_sha256": "d76a2498c3133d5340de32bb2b33d11f64e6a9fa703f6bb2a72e69020eae9407"},
            status="candidate",
            production_eligible=False,
        ),
        ModelProfile(
            model_id="DistilQwen2.5-DS3-0324-7B",
            revision="builtin-distilqwen-ds3-0324-v2",
            backend="llama_server",
            format="both",
            artifact_sha256="e885b819a519535bad1a3b5c0734c8963ce92d975a3302cdd4d3760c587e7b40",
            tokenizer_digest="9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa",
            chat_template_digest="8a5e6a9efdf4606caa92aab4ea9d2b36ce2af1148b00a6db7af33eacfddb98e2",
            context={"n_ctx": 32768, "input_budget": 8192, "max_new_tokens": 1024},
            generation={"temperature": 0.7, "top_p": 0.8, "thinking": "unknown"},
            adaptation={
                "prompt_family": "qwen_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            aliases=("distilqwen25-ds3-0324-7b",),
            capabilities=_unknown_capabilities(),
            resources={
                "kv_cache": "unknown",
                "gpu_layers": "auto",
                "min_ram_gb": 12.0,
                "min_vram_gb": 8.0,
                "min_disk_gb": 20.0,
                "edge_compatible": False,
            },
            evidence={**common["evidence"], "core_model_id": "distilqwen25-ds3-0324-7b", "probe_ticket": "DSW-D1", "artifact_digest_mode": "manifest", "manifest_sha256": "c428acdfeae0c890a809ef95b59edf74846ff5af25a8735e211a7a08fe97c438"},
            status="candidate",
            production_eligible=False,
        ),
        ModelProfile(
            model_id="Qwen3-4B",
            revision="builtin-qwen3-4b-v1",
            backend="llama_server",
            context={"n_ctx": 8192, "input_budget": 3072, "max_new_tokens": 512},
            generation={"temperature": 0.7, "top_p": 0.8, "thinking": "declared"},
            adaptation={
                "prompt_family": "qwen3_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer", "draft", "verify"),
            aliases=("qwen3-4b", "qwen3-4b-gguf"),
            capabilities=_unknown_capabilities(),
            evidence={
                **common["evidence"],
                "core_model_id": "qwen3-4b",
                "probe_ticket": "EX-QW3V2-01",
                "calibration": "v2-loose-512-3rounds",
                "artifact_revision": "bc640142c66e1fdd12af0bd68f40445458f3869b",
            },
            status="candidate",
            production_eligible=False,
        ),
        ModelProfile(
            model_id="Gemma-small",
            revision="builtin-gemma-v1",
            backend="llama_server",
            adaptation={
                "prompt_family": "gemma_chat_v1",
                "tool_mode": "host_router",
                "structured_output": "json_repair",
                "summary_mode": "state_schema_v1",
            },
            roles=("answer", "summarizer"),
            capabilities=_unknown_capabilities(),
            **common,
        ),
    )
