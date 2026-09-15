"""S3.2-MM-01 tests: image asset refs, thumbnail cards and budgeted follow-up context.

The image side of the harness must let a chat turn point at a stored image by
reference, describe it with a small bounded card, and pay for that card out of the
input budget -- without inlining bytes and without hiding a dropped image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness_workbench.context_engine.budget import (
    ContextBudget,
    ContextBudgetError,
    reserve_for_asset_cards,
)
from harness_workbench.image import (
    MAX_CARD_CHARS,
    MAX_REFS_PER_TURN,
    MULTIMODAL_CONTEXT_SCHEMA,
    GeneratedImage,
    ImageAssetRef,
    ImageAssetStore,
    ImageRefError,
    build_multimodal_context,
    render_thumbnail_card,
)


class _OneTokenCounter:
    """Deterministic counter so budget arithmetic is trivial to assert."""

    def count(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return 1


class _CharCounter:
    def count(self, text: str) -> int:
        return len(text)


def _store(tmp_path: Path) -> ImageAssetStore:
    return ImageAssetStore(tmp_path / "images")


def _record(store: ImageAssetStore, *, size: int = 64, mime: str = "image/png"):
    image = GeneratedImage(
        data=b"\x89PNG\r\n\x1a\n" + b"payload" * 32,
        mime_type=mime,
        width=size,
        height=size,
        seed=7,
    )
    return store.put(image, prompt="a red cube on a table")


def _ref(store: ImageAssetStore, *, label: str = "candidate-1", size: int = 64) -> ImageAssetRef:
    return ImageAssetRef.from_record(_record(store, size=size), label=label)


def test_ref_from_a_stored_asset_keeps_ids_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = _record(store)
    ref = ImageAssetRef.from_record(record, label="candidate-1")

    assert ref.asset_id == record.asset_id
    assert ref.sha256 == record.sha256
    assert ref.mime_type == "image/png"
    assert ref.dimensions == "64x64"
    assert ref.source == "generated"

    payload = ref.as_dict()
    assert payload["ref_schema"].startswith("qlh.harness.image_ref")
    # No path, no bytes anywhere in the reference contract.
    assert not any("\\\\" in str(value) or ":" in str(value)[:2] for value in payload.values())
    assert "data" not in payload


def test_end_to_end_fixture_turn_uses_the_stored_asset(tmp_path: Path) -> None:
    """生图(fixture) → 引用 → 缩略卡 → 追问上下文 的一轮闭环，不含真实推理。"""

    store = _store(tmp_path)
    ref = _ref(store)
    budget = ContextBudget(n_ctx=4096, max_new_tokens=256, overhead=512)
    context = build_multimodal_context("这张图里有什么？", [ref], input_budget=budget.input_budget)

    assert context.fits is True
    assert context.omitted_count == 0
    assert context.truncated is False
    assert len(context.cards) == 1
    assert context.refs == (ref,)
    assert context.cards[0].asset_id == ref.asset_id
    assert ref.asset_id in context.cards[0].text
    assert context.card_tokens > 0

    payload = context.as_dict()
    assert payload["context_schema"] == MULTIMODAL_CONTEXT_SCHEMA
    assert payload["cards"][0]["asset_id"] == ref.asset_id


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"asset_id": "/etc/passwd"}, "asset_id"),
        ({"asset_id": "deadbeef"}, "asset_id"),
        ({"sha256": "A" * 64}, "sha256"),
        ({"sha256": "abc"}, "sha256"),
        ({"mime_type": "image/tiff"}, "mime"),
        ({"source": "downloaded"}, "source"),
        ({"width": 0}, "width"),
        ({"height": -4}, "height"),
        ({"size_bytes": -1}, "size_bytes"),
        ({"label": "C:\\\\models\\\\img.png"}, "absolute path"),
        ({"owner_scope": "D:\\\\shared"}, "absolute path"),
    ],
)
def test_ref_rejects_unsafe_values(kwargs: dict, message: str) -> None:
    base = {
        "asset_id": "img_" + "a" * 24,
        "sha256": "b" * 64,
        "mime_type": "image/png",
    }
    with pytest.raises(ImageRefError, match=message):
        ImageAssetRef(**{**base, **kwargs})


def test_card_text_is_bounded_and_the_label_is_clamped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    long_label = "标注" * 120
    ref = ImageAssetRef.from_record(_record(store), label=long_label)

    card = render_thumbnail_card(ref)

    # Inside the hard character bound -- which holds because the label is clamped
    # to 48 characters, not because the bound was hit.
    assert len(card.text) <= MAX_CARD_CHARS
    assert card.text.startswith("image: label=")
    assert long_label not in card.text
    # The digest is summarised, not dumped, and no local path leaks in.
    assert ref.sha256[:12] in card.text
    assert str(store.root) not in card.text


def test_card_text_is_clamped_when_the_bound_is_low(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ref = _ref(store)

    card = render_thumbnail_card(ref, max_chars=32)

    assert len(card.text) <= 32
    assert card.truncated is True
    assert card.text.endswith("\u2026")


def test_cards_are_charged_to_the_input_budget(tmp_path: Path) -> None:
    store = _store(tmp_path)
    refs = [_ref(store, label=f"candidate-{index}") for index in range(3)]

    generous = build_multimodal_context("看图", refs, input_budget=200, tokenizer=_OneTokenCounter())
    assert len(generous.cards) == 3
    assert generous.card_tokens == 3
    assert generous.prompt_tokens == 1
    assert generous.total_tokens == 4
    assert generous.fits is True
    assert generous.token_count_mode == "provided"


def test_over_budget_cards_are_reported_not_hidden(tmp_path: Path) -> None:
    store = _store(tmp_path)
    refs = [_ref(store, label=f"c-{index}") for index in range(3)]

    # Budget admits the prompt plus at most one card (1 token each).
    tight = build_multimodal_context("看图", refs, input_budget=2, tokenizer=_OneTokenCounter())

    assert len(tight.cards) == 1
    assert tight.omitted_count == 2
    assert tight.truncated is True
    assert "image_cards_omitted" in tight.notices
    # The dropped references are not silently presented as understood.
    assert tight.refs == (refs[0],)


def test_a_turn_without_room_drops_images_with_a_notice(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ref = _ref(store)

    dropped = build_multimodal_context("看图", [ref], input_budget=1, tokenizer=_OneTokenCounter())

    assert dropped.cards == ()
    assert dropped.refs == ()
    assert dropped.truncated is True
    assert "image_context_dropped" in dropped.notices


def test_a_prompt_that_already_overflows_drops_everything(tmp_path: Path) -> None:
    store = _store(tmp_path)
    refs = [_ref(store)]

    overflowed = build_multimodal_context("很长的提示" * 50, refs, input_budget=5, tokenizer=_CharCounter())

    assert overflowed.cards == ()
    assert overflowed.card_tokens == 0
    assert "prompt_exceeds_input_budget" in overflowed.notices
    assert "image_context_dropped" in overflowed.notices
    assert overflowed.truncated is True


def test_max_refs_caps_the_turn(tmp_path: Path) -> None:
    store = _store(tmp_path)
    refs = [_ref(store, label=f"c-{index}") for index in range(MAX_REFS_PER_TURN + 3)]

    context = build_multimodal_context("看图", refs, input_budget=500, tokenizer=_OneTokenCounter())

    assert len(context.cards) == MAX_REFS_PER_TURN
    assert context.omitted_count == 3


def test_heuristic_estimate_is_labelled_as_an_estimate(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ref = _ref(store)

    context = build_multimodal_context("看图", [ref], input_budget=2000)
    card = render_thumbnail_card(ref)

    assert context.token_count_mode == "heuristic_estimate"
    assert card.token_count_mode == "heuristic_estimate"
    assert card.token_count > 0


def test_context_and_cards_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ref = _ref(store)
    context = build_multimodal_context("看图", [ref], input_budget=2000)

    restored = ImageAssetRef.from_dict(ref.as_dict())
    assert restored == ref
    assert context.as_dict()["cards"][0]["text"] == context.cards[0].text
    assert context.as_dict()["refs"][0] == ref.as_dict()


def test_reserve_for_asset_cards_updates_the_budget_formula() -> None:
    base = ContextBudget(n_ctx=4096, max_new_tokens=256, overhead=512, alignment=1)
    reserved = reserve_for_asset_cards(base, 96)

    assert reserved.overhead == base.overhead + 96
    assert reserved.input_budget == base.input_budget - 96
    assert reserved.as_dict()["input_budget"] == reserved.input_budget
    # Zero cards must not change the engine's existing budget.
    assert reserve_for_asset_cards(base, 0) is base


def test_reserve_for_asset_cards_fails_closed() -> None:
    base = ContextBudget(n_ctx=1024, max_new_tokens=512, overhead=256)

    with pytest.raises(ContextBudgetError):
        reserve_for_asset_cards(base, -1)
    with pytest.raises(ContextBudgetError):
        reserve_for_asset_cards(base, True)
    # Cards must not be allowed to eat the last of the input budget silently.
    with pytest.raises(ContextBudgetError):
        reserve_for_asset_cards(base, 512)


def test_build_rejects_bad_arguments(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ref = _ref(store)

    with pytest.raises(ImageRefError, match="prompt"):
        build_multimodal_context(None, [ref], input_budget=100)  # type: ignore[arg-type]
    with pytest.raises(ImageRefError, match="input_budget"):
        build_multimodal_context("看图", [ref], input_budget=0)
    with pytest.raises(ImageRefError, match="max_refs"):
        build_multimodal_context("看图", [ref], input_budget=100, max_refs=0)
    with pytest.raises(ImageRefError, match="ImageAssetRef"):
        build_multimodal_context("看图", [{"asset_id": "x"}], input_budget=100)  # type: ignore[list-item]
    with pytest.raises(ImageRefError, match="max_chars"):
        render_thumbnail_card(ref, max_chars=8)
