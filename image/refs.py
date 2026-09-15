"""Multimodal follow-up contract: image asset refs, thumbnail cards, budgeted context.

S3.2-MM-01: a generated (or uploaded) image must be referenceable from the chat
context -- by an *asset reference* plus a small *thumbnail card*, never by inlining
raw bytes -- and the card must be paid for out of the input budget explicitly.

Design boundaries:

* an :class:`ImageAssetRef` carries identifiers and digests only; there is no
  path, no base64 payload and no file handle, so a ref is safe to log, store and
  send to a model provider;
* rendering a card is bounded in characters and counted in tokens with the same
  counter the context engine uses; when a tokenizer is not supplied the count is
  marked ``heuristic_estimate`` instead of being presented as exact;
* budget pressure is reported, not hidden: dropped cards increase
  ``omitted_count`` and set ``truncated`` so the caller can tell the user that an
  image was not shown;
* the real vision adapter is out of scope here (hardware/gated): this module only
  produces the contract the adapter will consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .assets import ImageAssetRecord


IMAGE_REF_SCHEMA = "qlh.harness.image_ref.v1"
MULTIMODAL_CONTEXT_SCHEMA = "qlh.harness.multimodal_context.v1"
SUPPORTED_MIME_TYPES = ("image/png", "image/jpeg", "image/webp")
IMAGE_SOURCES = ("generated", "user_upload")
MAX_REFS_PER_TURN = 4
MAX_CARD_CHARS = 240
MAX_PROMPT_CHARS = 8000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ASSET_ID = re.compile(r"^img_[0-9a-f]{8,64}$")
_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/]{1,2})")


class ImageRefError(ValueError):
    """Raised when an image reference or card would be unsafe or unusable."""


class TokenEstimator:
    """Minimal counter protocol shared with ``context_engine.tokenizer``."""

    def count(self, text: str) -> int:  # pragma: no cover - structural only
        raise NotImplementedError


def _heuristic_count(text: str) -> int:
    # Same shape as context_engine.tokenizer.HeuristicTokenizer: CJK characters
    # and punctuation count individually, runs of word characters count once.
    return len(re.findall(r"[\u4e00-\u9fff]|\w+|[^\w\s]", text, re.UNICODE))


def _resolve_counter(tokenizer: Any | None) -> tuple[Any, str]:
    if tokenizer is None:
        return None, "heuristic_estimate"
    counter = getattr(tokenizer, "count", None)
    if not callable(counter):
        raise ImageRefError("tokenizer must expose a callable count(text)")
    return tokenizer, "provided"


@dataclass(frozen=True, slots=True)
class ImageAssetRef:
    """A safe pointer to a stored image: identifiers and digests, never bytes."""

    asset_id: str
    sha256: str
    mime_type: str
    owner_scope: str = "local"
    width: int | None = None
    height: int | None = None
    size_bytes: int = 0
    source: str = "generated"
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.asset_id, str) or not _ASSET_ID.fullmatch(self.asset_id):
            raise ImageRefError("asset_id must look like the store's img_<hex> identifier")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ImageRefError("sha256 must be a lowercase SHA-256 digest")
        if self.mime_type not in SUPPORTED_MIME_TYPES:
            raise ImageRefError(f"unsupported image mime type: {self.mime_type}")
        if self.source not in IMAGE_SOURCES:
            raise ImageRefError(f"unsupported image source: {self.source}")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ImageRefError(f"{name} must be a positive integer or null")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ImageRefError("size_bytes must be a non-negative integer")
        for value in (self.owner_scope, self.label):
            if not isinstance(value, str):
                raise ImageRefError("owner_scope and label must be strings")
            if _ABSOLUTE_PATH.match(value):
                raise ImageRefError("image references cannot contain absolute paths")

    @classmethod
    def from_record(
        cls,
        record: ImageAssetRecord,
        *,
        source: str = "generated",
        label: str = "",
        owner_scope: str | None = None,
    ) -> "ImageAssetRef":
        """Build a reference from a stored asset record."""

        if not isinstance(record, ImageAssetRecord):
            raise ImageRefError("from_record expects an ImageAssetRecord")
        return cls(
            asset_id=record.asset_id,
            sha256=record.sha256,
            mime_type=record.mime_type,
            owner_scope=owner_scope if owner_scope is not None else record.owner_scope,
            width=record.width,
            height=record.height,
            size_bytes=record.size_bytes,
            source=source,
            label=label,
        )

    @property
    def dimensions(self) -> str:
        if self.width is None or self.height is None:
            return "unknown"
        return f"{self.width}x{self.height}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ref_schema": IMAGE_REF_SCHEMA,
            "asset_id": self.asset_id,
            "sha256": self.sha256,
            "mime_type": self.mime_type,
            "owner_scope": self.owner_scope,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "source": self.source,
            "label": self.label,
            "dimensions": self.dimensions,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImageAssetRef":
        if not isinstance(value, Mapping):
            raise ImageRefError("image reference must be an object")
        return cls(
            asset_id=str(value.get("asset_id", "")),
            sha256=str(value.get("sha256", "")),
            mime_type=str(value.get("mime_type", "")),
            owner_scope=str(value.get("owner_scope", "local")),
            width=value.get("width"),
            height=value.get("height"),
            size_bytes=value.get("size_bytes", 0),
            source=str(value.get("source", "generated")),
            label=str(value.get("label", "")),
        )


@dataclass(frozen=True, slots=True)
class ThumbnailCard:
    """One bounded text card describing an image reference to a small model."""

    asset_id: str
    text: str
    token_count: int
    truncated: bool
    token_count_mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "text": self.text,
            "token_count": self.token_count,
            "truncated": self.truncated,
            "token_count_mode": self.token_count_mode,
        }


def render_thumbnail_card(
    ref: ImageAssetRef,
    *,
    tokenizer: Any | None = None,
    label: str | None = None,
    max_chars: int = MAX_CARD_CHARS,
) -> ThumbnailCard:
    """Render one card; the text is bounded and carries no local path."""

    if not isinstance(ref, ImageAssetRef):
        raise ImageRefError("render_thumbnail_card expects an ImageAssetRef")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 32:
        raise ImageRefError("max_chars must be an integer of at least 32")

    counter, mode = _resolve_counter(tokenizer)
    resolved_label = ref.label if label is None else label
    parts = [
        f"asset={ref.asset_id}",
        f"mime={ref.mime_type}",
        f"dimensions={ref.dimensions}",
        f"sha256={ref.sha256[:12]}",
        f"origin={ref.source}",
    ]
    if resolved_label:
        parts.insert(0, f"label={resolved_label[:48]}")
    text = "image: " + " ".join(parts)
    truncated = len(text) > max_chars
    if truncated:
        text = text[: max_chars - 1] + "\u2026"
    count = counter.count(text) if counter is not None else _heuristic_count(text)
    return ThumbnailCard(
        asset_id=ref.asset_id,
        text=text,
        token_count=int(count),
        truncated=truncated,
        token_count_mode=mode,
    )


@dataclass(frozen=True, slots=True)
class MultimodalContext:
    """The follow-up turn: prompt plus the cards that actually fit the budget."""

    prompt: str
    input_budget: int
    prompt_tokens: int
    card_tokens: int
    cards: tuple[ThumbnailCard, ...] = ()
    refs: tuple[ImageAssetRef, ...] = ()
    omitted_count: int = 0
    truncated: bool = False
    notices: tuple[str, ...] = ()
    token_count_mode: str = "heuristic_estimate"

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.card_tokens

    @property
    def fits(self) -> bool:
        return self.total_tokens <= self.input_budget

    def as_dict(self) -> dict[str, Any]:
        return {
            "context_schema": MULTIMODAL_CONTEXT_SCHEMA,
            "prompt": self.prompt,
            "input_budget": self.input_budget,
            "prompt_tokens": self.prompt_tokens,
            "card_tokens": self.card_tokens,
            "total_tokens": self.total_tokens,
            "fits": self.fits,
            "omitted_count": self.omitted_count,
            "truncated": self.truncated,
            "notices": list(self.notices),
            "token_count_mode": self.token_count_mode,
            "cards": [card.as_dict() for card in self.cards],
            "refs": [ref.as_dict() for ref in self.refs],
        }


def build_multimodal_context(
    prompt: str,
    refs: Iterable[ImageAssetRef],
    *,
    input_budget: int,
    tokenizer: Any | None = None,
    max_refs: int = MAX_REFS_PER_TURN,
) -> MultimodalContext:
    """Build a follow-up turn whose cards are paid for out of ``input_budget``.

    Cards are added in order until the budget would be exceeded; the remainder is
    reported through ``omitted_count``/``truncated`` rather than silently dropped.
    An over-long prompt is trimmed and says so.
    """

    if not isinstance(prompt, str):
        raise ImageRefError("prompt must be a string")
    if isinstance(input_budget, bool) or not isinstance(input_budget, int) or input_budget <= 0:
        raise ImageRefError("input_budget must be a positive integer")
    if isinstance(max_refs, bool) or not isinstance(max_refs, int) or max_refs <= 0:
        raise ImageRefError("max_refs must be a positive integer")

    counter, mode = _resolve_counter(tokenizer)
    notices: list[str] = []

    trimmed_prompt = prompt
    if len(trimmed_prompt) > MAX_PROMPT_CHARS:
        trimmed_prompt = trimmed_prompt[: MAX_PROMPT_CHARS - 1] + "\u2026"
        notices.append("prompt_truncated")
    prompt_tokens = int(counter.count(trimmed_prompt)) if counter is not None else _heuristic_count(trimmed_prompt)
    if prompt_tokens > input_budget:
        # The text itself already overflows: no card can be afforded.
        return MultimodalContext(
            prompt=trimmed_prompt,
            input_budget=input_budget,
            prompt_tokens=prompt_tokens,
            card_tokens=0,
            cards=(),
            refs=(),
            omitted_count=len(tuple(refs)),
            truncated=True,
            notices=tuple(notices + ["prompt_exceeds_input_budget", "image_context_dropped"]),
            token_count_mode=mode,
        )

    admitted_refs: list[ImageAssetRef] = []
    admitted_cards: list[ThumbnailCard] = []
    used = 0
    considered = 0
    omitted = 0
    for ref in refs:
        if not isinstance(ref, ImageAssetRef):
            raise ImageRefError("refs must contain ImageAssetRef values")
        considered += 1
        if len(admitted_refs) >= max_refs:
            omitted += 1
            continue
        card = render_thumbnail_card(ref, tokenizer=tokenizer)
        if used + card.token_count > input_budget - prompt_tokens:
            omitted += 1
            continue
        admitted_refs.append(ref)
        admitted_cards.append(card)
        used += card.token_count

    if considered and not admitted_refs:
        notices.append("image_context_dropped")
    if omitted:
        notices.append("image_cards_omitted")

    return MultimodalContext(
        prompt=trimmed_prompt,
        input_budget=input_budget,
        prompt_tokens=prompt_tokens,
        card_tokens=used,
        cards=tuple(admitted_cards),
        refs=tuple(admitted_refs),
        omitted_count=omitted,
        truncated=bool(omitted),
        notices=tuple(notices),
        token_count_mode=mode,
    )
