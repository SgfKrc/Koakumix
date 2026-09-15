"""Small-model harness image workbench.

The package is intentionally independent from ``src/``.  It is Koakumix's
image-generation boundary and delegates execution to an injected local
executor.
"""

from .assets import ImageAssetRecord, ImageAssetStore
from .contracts import (
    GeneratedImage,
    ImageAdapter,
    ImageAdapterCapabilities,
    ImageAdapterError,
    ImageRequest,
    ImageRequestError,
)
from .editing import (
    CONTRACT_ONLY,
    EDIT_MODES,
    EDIT_MODE_SPECS,
    EDIT_REQUEST_SCHEMA,
    UNAVAILABLE,
    EditAvailability,
    EditModeSpec,
    ImageEditRequest,
    edit_capability_matrix,
    edit_mode_spec,
    resolve_edit_availability,
)
from .local_engine import LocalImageEngine, LocalImageEngineConfig
from .manifest import AssetManifestReport, validate_asset_manifest
from .refs import (
    IMAGE_REF_SCHEMA,
    MAX_CARD_CHARS,
    MAX_PROMPT_CHARS,
    MAX_REFS_PER_TURN,
    MULTIMODAL_CONTEXT_SCHEMA,
    SUPPORTED_MIME_TYPES,
    ImageAssetRef,
    ImageRefError,
    MultimodalContext,
    ThumbnailCard,
    build_multimodal_context,
    render_thumbnail_card,
)

__all__ = [
    "CONTRACT_ONLY",
    "EDIT_MODES",
    "EDIT_MODE_SPECS",
    "EDIT_REQUEST_SCHEMA",
    "IMAGE_REF_SCHEMA",
    "MAX_CARD_CHARS",
    "MAX_PROMPT_CHARS",
    "MAX_REFS_PER_TURN",
    "MULTIMODAL_CONTEXT_SCHEMA",
    "SUPPORTED_MIME_TYPES",
    "UNAVAILABLE",
    "AssetManifestReport",
    "EditAvailability",
    "EditModeSpec",
    "GeneratedImage",
    "ImageAdapter",
    "ImageAdapterCapabilities",
    "ImageAdapterError",
    "ImageAssetRecord",
    "ImageAssetRef",
    "ImageAssetStore",
    "ImageEditRequest",
    "ImageRefError",
    "ImageRequest",
    "ImageRequestError",
    "LocalImageEngine",
    "LocalImageEngineConfig",
    "MultimodalContext",
    "ThumbnailCard",
    "build_multimodal_context",
    "edit_capability_matrix",
    "edit_mode_spec",
    "render_thumbnail_card",
    "resolve_edit_availability",
    "validate_asset_manifest",
]
