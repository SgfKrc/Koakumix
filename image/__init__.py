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
    "IMAGE_REF_SCHEMA",
    "MAX_CARD_CHARS",
    "MAX_PROMPT_CHARS",
    "MAX_REFS_PER_TURN",
    "MULTIMODAL_CONTEXT_SCHEMA",
    "SUPPORTED_MIME_TYPES",
    "AssetManifestReport",
    "GeneratedImage",
    "ImageAdapter",
    "ImageAdapterCapabilities",
    "ImageAdapterError",
    "ImageAssetRecord",
    "ImageAssetRef",
    "ImageAssetStore",
    "ImageRefError",
    "ImageRequest",
    "ImageRequestError",
    "LocalImageEngine",
    "LocalImageEngineConfig",
    "MultimodalContext",
    "ThumbnailCard",
    "build_multimodal_context",
    "render_thumbnail_card",
    "validate_asset_manifest",
]
