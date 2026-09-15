"""Read QLH main-repo model asset manifests into harness model profiles.

The bridge is deliberately one-way and read-only: it parses the JSON contract the
main repo writes next to local model assets (``.qlh-model-asset.json`` or the
compat name ``model.manifest.json``) and maps the shared fields -- model id,
engine/backend, **format**, revision, sha256 digests, capability and budget
policy -- onto :class:`~model_profiles.schema.ModelProfile`.

Guarantees:

* never imports main-repo code and never copies its loader;
* never starts a sidecar, never materialises weights, never touches the network;
* never invents capability evidence -- everything the manifest cannot prove stays
  ``unknown`` and the profile stays ``production_eligible=False`` (fail-closed);
* rejects absolute paths and oversized manifests before parsing them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import ModelProfile, ProfileValidationError


BRIDGE_SCHEMA = "qlh.harness.model_fleet_bridge.v1"
MANIFEST_SCHEMA = 1
MANIFEST_FILENAMES = (".qlh-model-asset.json", "model.manifest.json")
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
UNVERSIONED_REVISION = "unversioned"
DEFAULT_BACKEND = "manifest_only"

#: Formats the main-repo manifest is allowed to advertise.
KNOWN_FORMATS = ("safetensors", "gguf", "onnx", "openvino")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
_TOKENIZER_NAMES = ("tokenizer.json", "tokenizer.model", "spm.model")
_CHAT_TEMPLATE_NAMES = ("chat_template.jinja", "chat_template.json", "tokenizer_config.json")


class ManifestBridgeError(ValueError):
    """Raised when a main-repo manifest cannot be trusted or mapped."""


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _reject_absolute(value: str, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ManifestBridgeError(f"{field} is empty")
    if re.match(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2})", text) or "\\" in text:
        raise ManifestBridgeError(f"{field} must be a repo-relative identifier")
    if ".." in text.split("/"):
        raise ManifestBridgeError(f"{field} must not traverse upwards")
    return text


def find_manifest(directory: str | Path) -> Path | None:
    """Return the manifest path for one asset directory, or ``None``."""

    root = Path(directory)
    for name in MANIFEST_FILENAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def load_manifest(directory: str | Path) -> tuple[dict[str, Any], Path]:
    """Read an asset manifest with a hard size gate; never follow huge files."""

    root = Path(directory)
    manifest_path = find_manifest(root)
    if manifest_path is None:
        raise ManifestBridgeError("no model asset manifest was found")
    try:
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ManifestBridgeError("model asset manifest exceeds the size gate")
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - filesystem dependent
        raise ManifestBridgeError("model asset manifest could not be read") from exc
    except json.JSONDecodeError as exc:
        raise ManifestBridgeError("model asset manifest is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ManifestBridgeError("model asset manifest must be a JSON object")
    if value.get("schema") != MANIFEST_SCHEMA:
        raise ManifestBridgeError("unsupported model asset manifest schema")
    return value, manifest_path


def model_id_from_source(source: Any) -> tuple[str, tuple[str, ...]]:
    """Map the manifest ``source`` repo id onto a profile id plus aliases."""

    text = _reject_absolute(str(source or ""), "source")
    if not _SAFE_ID.fullmatch(text):
        raise ManifestBridgeError("source must be a safe repo identifier")
    model_id = text.lower()
    aliases = (model_id, model_id.rsplit("/", 1)[-1])
    return model_id, tuple(dict.fromkeys(alias for alias in aliases if alias != model_id)) or (model_id,)


def _format_from_manifest(manifest: Mapping[str, Any]) -> str:
    raw = str(manifest.get("model_type") or "").strip().lower()
    if raw not in KNOWN_FORMATS:
        raise ManifestBridgeError(f"unsupported model format: {raw or '(missing)'}")
    return raw


def _digest_for(names: Iterable[str], files: Mapping[str, str]) -> str | None:
    for name in names:
        digest = files.get(name)
        if digest:
            return digest
    return None


def _file_digests(manifest: Mapping[str, Any]) -> tuple[dict[str, str], int]:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ManifestBridgeError("manifest files must be a non-empty array")
    digests: dict[str, str] = {}
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ManifestBridgeError("manifest file entries must be objects")
        path = _reject_absolute(str(entry.get("path") or ""), "files[].path")
        digest = str(entry.get("sha256") or "")
        if not _SHA256.fullmatch(digest):
            raise ManifestBridgeError("files[].sha256 must be a lowercase SHA-256 digest")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ManifestBridgeError("files[].size_bytes must be a non-negative integer")
        digests[path.rsplit("/", 1)[-1]] = digest
        total_bytes += size
    return digests, total_bytes


def profile_from_manifest(
    directory: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    roles: Iterable[str] = ("answer",),
    context: Mapping[str, Any] | None = None,
    generation: Mapping[str, Any] | None = None,
) -> ModelProfile:
    """Map one main-repo asset directory onto an unverified harness profile.

    ``backend`` (engine) and ``format`` stay independent fields: the manifest
    proves the artifact *format*, while the engine that will run it must be
    declared by the caller and verified later.
    """

    if not isinstance(backend, str) or not backend.strip():
        raise ManifestBridgeError("backend must be a non-empty engine name")
    manifest, manifest_path = load_manifest(directory)
    model_id, aliases = model_id_from_source(manifest.get("source"))
    model_format = _format_from_manifest(manifest)

    artifact_sha256 = str(manifest.get("artifact_sha256") or "")
    if not _SHA256.fullmatch(artifact_sha256):
        raise ManifestBridgeError("artifact_sha256 must be a lowercase SHA-256 digest")

    digests, total_bytes = _file_digests(manifest)
    revision = str(manifest.get("revision") or "").strip() or UNVERSIONED_REVISION

    evidence: dict[str, Any] = {
        "source": "qlh.main_repo_model_asset_manifest",
        "bridge_schema": BRIDGE_SCHEMA,
        "manifest_file": manifest_path.name,
        "manifest_sha256": str(manifest.get("manifest_sha256") or ""),
        "declared_model_type": model_format,
    }
    resources: dict[str, Any] = {"file_count": len(digests), "total_bytes": total_bytes}
    budget = dict(context or {})
    if budget.get("n_ctx") is None:
        # The manifest only proves file digests; context policy stays explicit.
        budget.setdefault("n_ctx", 4096)

    try:
        return ModelProfile(
            model_id=model_id,
            revision=revision,
            backend=backend,
            format=model_format,
            artifact_sha256=artifact_sha256,
            tokenizer_digest=_digest_for(_TOKENIZER_NAMES, digests),
            chat_template_digest=_digest_for(_CHAT_TEMPLATE_NAMES, digests),
            context=budget,
            generation=dict(generation or {}),
            aliases=aliases,
            resources=resources,
            # Capabilities are NOT inferred from a manifest: unknown ⇒ fail-closed.
            status="unknown",
            production_eligible=False,
            evidence=evidence,
        )
    except ProfileValidationError as exc:  # pragma: no cover - defensive
        raise ManifestBridgeError(f"manifest produced an invalid profile: {exc}") from exc


def bridge_report(directories: Iterable[str | Path]) -> dict[str, Any]:
    """Map many asset directories, reporting per-directory failures."""

    mapped: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for directory in directories:
        path = Path(directory)
        try:
            profile = profile_from_manifest(path)
        except ManifestBridgeError as exc:
            errors.append({"asset": path.name, "code": "manifest_unusable", "message": str(exc)})
            continue
        mapped.append(
            {
                "asset": path.name,
                "profile_id": profile.profile_id,
                "model_id": profile.model_id,
                "format": profile.format,
                "backend": profile.backend,
                "revision": profile.revision,
                "artifact_sha256": profile.artifact_sha256,
                "profile_digest": profile.digest,
                "production_eligible": profile.production_eligible,
            }
        )
    return {
        "bridge_schema": BRIDGE_SCHEMA,
        "mapped_count": len(mapped),
        "error_count": len(errors),
        "profiles": sorted(mapped, key=lambda item: item["profile_id"]),
        "errors": errors,
    }
