"""SIDE-KOAKU-01 fleet contract tests: main-repo model asset manifest -> harness profile.

The main repo owns the artifact manifest contract (``.qlh-model-asset.json`` or the
compat ``model.manifest.json``).  The harness must be able to mirror its model id,
format, revision, sha256 digests and budget fields without importing main-repo
code, without copying its loader and without inventing capability evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_workbench.model_profiles import (
    DEFAULT_BACKEND,
    KNOWN_FORMATS,
    MANIFEST_FILENAMES,
    MANIFEST_SCHEMA,
    UNVERSIONED_REVISION,
    ManifestBridgeError,
    ModelProfile,
    bridge_report,
    find_manifest,
    load_manifest,
    model_id_from_source,
    profile_from_manifest,
)

# Mirrors build/exp-p8-real/qwen2.5-0.5b/model.manifest.json from the main repo.
MANIFEST: dict[str, object] = {
    "artifact_sha256": "a" * 64,
    "files": [
        {"path": "config.json", "sha256": "b" * 64, "size_bytes": 681},
        {"path": "generation_config.json", "sha256": "c" * 64, "size_bytes": 138},
        {"path": "model.safetensors", "sha256": "d" * 64, "size_bytes": 988097824},
        {"path": "tokenizer.json", "sha256": "e" * 64, "size_bytes": 7031645},
        {"path": "tokenizer_config.json", "sha256": "f" * 64, "size_bytes": 7228},
    ],
    "manifest_sha256": "1" * 64,
    "model_type": "safetensors",
    "revision": "",
    "schema": MANIFEST_SCHEMA,
    "source": "Qwen/Qwen2.5-0.5B",
}


def _write_asset(root: Path, manifest: dict[str, object], *, name: str = "model.manifest.json") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _asset(tmp_path: Path, manifest: dict[str, object] | None = None) -> Path:
    return _write_asset(tmp_path / "qwen2.5-0.5b", dict(manifest or MANIFEST))


def test_manifest_is_discovered_by_both_contract_names(tmp_path: Path) -> None:
    assert MANIFEST_FILENAMES[0] == ".qlh-model-asset.json"
    canonical = _write_asset(tmp_path / "a", dict(MANIFEST), name=MANIFEST_FILENAMES[0])
    compat = _write_asset(tmp_path / "b", dict(MANIFEST))
    assert find_manifest(canonical).name == MANIFEST_FILENAMES[0]
    assert find_manifest(compat).name == "model.manifest.json"
    assert find_manifest(tmp_path / "missing") is None
    with pytest.raises(ManifestBridgeError):
        load_manifest(tmp_path / "missing")


def test_bridge_maps_manifest_onto_profile_fields(tmp_path: Path) -> None:
    profile = profile_from_manifest(_asset(tmp_path))

    # 票面要求的共享字段：model id / engine / format / revision / sha256。
    assert profile.model_id == "qwen/qwen2.5-0.5b"
    assert profile.backend == DEFAULT_BACKEND == "manifest_only"
    assert profile.format == "safetensors"
    assert profile.format in KNOWN_FORMATS
    assert profile.revision == UNVERSIONED_REVISION  # manifest revision was empty
    assert profile.artifact_sha256 == "a" * 64
    assert profile.tokenizer_digest == "e" * 64
    assert profile.chat_template_digest == "f" * 64
    # 别名让旧的短名仍可检索。
    assert "qwen2.5-0.5b" in profile.aliases
    # 资源与证据来自 manifest，不含本机绝对路径。
    assert profile.resources["file_count"] == 5
    assert profile.resources["total_bytes"] == 995137516
    assert profile.evidence["manifest_sha256"] == "1" * 64
    assert profile.evidence["source"] == "qlh.main_repo_model_asset_manifest"


def test_bridge_keeps_capabilities_unknown_and_not_production_eligible(tmp_path: Path) -> None:
    profile = profile_from_manifest(_asset(tmp_path))

    assert {name: state.status for name, state in profile.capabilities.items()} == {
        "json_output": "unknown",
        "tool_call_generation": "unknown",
        "tool_result_reinjection": "unknown",
        "multimodal": "unknown",
        "thinking_control": "unknown",
    }
    assert profile.status == "unknown"
    assert profile.production_eligible is False


def test_bridge_round_trips_through_canonical_profile_json(tmp_path: Path) -> None:
    profile = profile_from_manifest(_asset(tmp_path))

    restored = ModelProfile.from_dict(profile.as_dict())
    assert restored.format == "safetensors"
    assert restored.digest == profile.digest
    assert restored.as_dict()["format"] == profile.format


def test_format_is_independent_of_backend(tmp_path: Path) -> None:
    directory = _asset(tmp_path)
    gguf_manifest = dict(MANIFEST, model_type="gguf")
    gguf_dir = _write_asset(tmp_path / "gguf-asset", gguf_manifest)

    transformer_profile = profile_from_manifest(directory, backend="llama_server")
    gguf_profile = profile_from_manifest(gguf_dir, backend="llama_server")

    # 同 engine、同模型 → profile_id 相同，但 format 不同 ⇒ digest 必须不同。
    assert transformer_profile.backend == gguf_profile.backend
    assert transformer_profile.format != gguf_profile.format
    assert transformer_profile.digest != gguf_profile.digest


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda m: m.update(artifact_sha256="A" * 64), "artifact_sha256"),
        (lambda m: m.pop("artifact_sha256"), "artifact_sha256"),
        (lambda m: m.update(model_type="unknown-container"), "unsupported model format"),
        (lambda m: m.update(model_type=""), "unsupported model format"),
        (lambda m: m.update(source="/etc/passwd"), "source"),
        (lambda m: m.update(source="Qwen/../../etc"), "source"),
        (lambda m: m.update(source=""), "source"),
        (lambda m: m.update(schema=2), "unsupported model asset manifest schema"),
        (lambda m: m.update(files=[]), "files must be a non-empty array"),
        (lambda m: m.update(files=[{"path": "C:/abs.json", "sha256": "b" * 64, "size_bytes": 1}]), "files[].path"),
        (lambda m: m.update(files=[{"path": "x.json", "sha256": "nope", "size_bytes": 1}]), "files[].sha256"),
        (lambda m: m.update(files=[{"path": "x.json", "sha256": "b" * 64, "size_bytes": -1}]), "size_bytes"),
    ],
)
def test_bridge_fails_closed_on_untrusted_manifests(tmp_path: Path, mutate, message: str) -> None:
    manifest = json.loads(json.dumps(MANIFEST))
    mutate(manifest)

    with pytest.raises(ManifestBridgeError) as excinfo:
        profile_from_manifest(_asset(tmp_path, manifest))

    assert message in str(excinfo.value)


def test_bridge_rejects_oversized_and_malformed_manifests(tmp_path: Path) -> None:
    oversized = _write_asset(tmp_path / "big", dict(MANIFEST))
    (oversized / "model.manifest.json").write_text(
        "{" + " " * (2 * 1024 * 1024 + 1) + "}", encoding="utf-8"
    )
    with pytest.raises(ManifestBridgeError, match="size gate"):
        load_manifest(oversized)

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "model.manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestBridgeError, match="not valid JSON"):
        load_manifest(malformed)


def test_bridge_requires_a_real_engine_name(tmp_path: Path) -> None:
    directory = _asset(tmp_path)

    with pytest.raises(ManifestBridgeError, match="engine"):
        profile_from_manifest(directory, backend="")
    with pytest.raises(ManifestBridgeError, match="engine"):
        profile_from_manifest(directory, backend="   ")


def test_bridge_report_maps_good_assets_and_reports_bad_ones(tmp_path: Path) -> None:
    good = _asset(tmp_path / "good")
    broken = _write_asset(tmp_path / "broken", dict(MANIFEST, artifact_sha256=""))
    empty = tmp_path / "empty"
    empty.mkdir()

    report = bridge_report([good, broken, empty])

    assert report["mapped_count"] == 1
    assert report["error_count"] == 2
    assert report["profiles"][0]["format"] == "safetensors"
    assert report["profiles"][0]["production_eligible"] is False
    assert {error["asset"] for error in report["errors"]} == {"broken", "empty"}
    assert all(error["code"] == "manifest_unusable" for error in report["errors"])


def test_model_id_from_source_normalises_repo_ids() -> None:
    assert model_id_from_source("Qwen/Qwen2.5-0.5B") == ("qwen/qwen2.5-0.5b", ("qwen2.5-0.5b",))
    with pytest.raises(ManifestBridgeError):
        model_id_from_source("")
    with pytest.raises(ManifestBridgeError):
        model_id_from_source("D:\\models\\qwen")
