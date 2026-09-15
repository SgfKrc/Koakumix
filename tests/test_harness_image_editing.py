"""S3.2-EDIT-01 tests: edit contracts, remote mapping and honest availability.

Four editing paths (img2img, inpaint, IP-Adapter, instruct) need one backend-neutral
contract, a remote mapping shape that carries references instead of bytes, and a
truthful statement of where each path can run today (nowhere yet, locally or
remotely -- the local executor is known-unavailable and no remote endpoint is
verified).
"""

from __future__ import annotations

import pytest

from harness_workbench.image import (
    EDIT_MODES,
    GeneratedImage,
    ImageAdapterCapabilities,
    ImageAssetRef,
    ImageAssetStore,
    ImageEditRequest,
    ImageRequestError,
    edit_capability_matrix,
    edit_mode_spec,
    resolve_edit_availability,
)


def _ref(*, asset_id: str = "img_" + "a" * 24, size: int = 512) -> ImageAssetRef:
    return ImageAssetRef(
        asset_id=asset_id,
        sha256="b" * 64,
        mime_type="image/png",
        width=size,
        height=size,
        size_bytes=1024,
        source="generated",
        label="fixture",
    )


def _mask(*, size: int = 512) -> ImageAssetRef:
    return ImageAssetRef(
        asset_id="img_" + "c" * 24,
        sha256="d" * 64,
        mime_type="image/png",
        width=size,
        height=size,
        size_bytes=64,
        source="generated",
        label="mask",
    )


def test_every_edit_mode_has_a_declared_spec() -> None:
    assert EDIT_MODES == ("img2img", "inpaint", "ip_adapter", "instruct")
    for mode in EDIT_MODES:
        spec = edit_mode_spec(mode)
        assert spec.mode == mode
        assert spec.requires
        # Honest runtime status: nothing is wired in this ticket.
        assert spec.local_status == "unavailable"
        assert spec.remote_status == "contract_only"


def test_img2img_requires_a_source_reference() -> None:
    with pytest.raises(ImageRequestError, match="source"):
        ImageEditRequest(mode="img2img", prompt="make it snowy")

    request = ImageEditRequest(mode="img2img", prompt="make it snowy", source=_ref(), strength=0.45)
    assert request.source is not None
    assert request.strength == 0.45


def test_inpaint_requires_a_mask_matching_the_source() -> None:
    with pytest.raises(ImageRequestError, match="mask"):
        ImageEditRequest(mode="inpaint", prompt="remove the sign", source=_ref())

    with pytest.raises(ImageRequestError, match="dimensions"):
        ImageEditRequest(
            mode="inpaint", prompt="remove the sign", source=_ref(size=512), mask=_mask(size=256)
        )

    ok = ImageEditRequest(mode="inpaint", prompt="remove the sign", source=_ref(), mask=_mask())
    assert ok.mask is not None


def test_ip_adapter_requires_a_reference() -> None:
    with pytest.raises(ImageRequestError, match="reference"):
        ImageEditRequest(mode="ip_adapter", prompt="in that style")

    request = ImageEditRequest(mode="ip_adapter", prompt="in that style", reference=_ref())
    assert request.reference is not None


def test_instruct_requires_an_instruction() -> None:
    with pytest.raises(ImageRequestError, match="instruction"):
        ImageEditRequest(mode="instruct", source=_ref())

    request = ImageEditRequest(mode="instruct", source=_ref(), instruction="turn the sky purple")
    assert request.instruction == "turn the sky purple"


def test_remote_payload_carries_references_never_bytes() -> None:
    request = ImageEditRequest(mode="inpaint", prompt="clean", source=_ref(), mask=_mask())
    payload = request.as_remote_payload()

    assert payload["mode"] == "inpaint"
    assert payload["init_image_ref"]["asset_id"] == _ref().asset_id
    assert payload["mask_ref"]["asset_id"] == _mask().asset_id
    assert "ip_adapter_ref" not in payload  # optional component omitted when absent
    serialized = repr(payload)
    assert "data" not in payload
    assert "\\\\" not in serialized
    assert "base64" not in serialized


def test_from_mapping_rejects_unsafe_reference_objects() -> None:
    good = {"mode": "img2img", "prompt": "x", "source": _ref().as_dict()}
    assert ImageEditRequest.from_mapping(good).source == _ref()

    with pytest.raises(ImageRequestError, match="reference"):
        ImageEditRequest.from_mapping(
            {"mode": "img2img", "prompt": "x", "source": {"asset_id": "C:\\\\img.png"}}
        )
    with pytest.raises(ImageRequestError, match="reference"):
        ImageEditRequest.from_mapping({"mode": "img2img", "prompt": "x", "source": {"asset_id": "x"}})
    with pytest.raises(ImageRequestError, match="reference"):
        ImageEditRequest.from_mapping({"mode": "img2img", "prompt": "x", "source": "img_ab.png"})


def test_numeric_bounds_are_enforced() -> None:
    with pytest.raises(ImageRequestError, match="strength"):
        ImageEditRequest(mode="img2img", source=_ref(), strength=1.5)
    with pytest.raises(ImageRequestError, match="width and height"):
        ImageEditRequest(mode="img2img", source=_ref(), width=100)
    with pytest.raises(ImageRequestError, match="steps"):
        ImageEditRequest(mode="img2img", source=_ref(), steps=0)
    with pytest.raises(ImageRequestError, match="instruction"):
        ImageEditRequest(mode="instruct", source=_ref(), instruction="x" * 2000)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ImageRequestError, match="mode must be one of"):
        ImageEditRequest(mode="superres", prompt="x")
    with pytest.raises(ImageRequestError, match="mode must be one of"):
        edit_mode_spec("")


def test_manifest_hint_records_the_lineage() -> None:
    request = ImageEditRequest(mode="inpaint", prompt="clean", source=_ref(), mask=_mask())
    hint = request.manifest_hint()

    assert hint["edit_mode"] == "inpaint"
    assert hint["source_asset_id"] == _ref().asset_id
    assert hint["mask_asset_id"] == _mask().asset_id
    assert hint["reference_asset_id"] is None


def test_edited_asset_metadata_can_carry_the_hint(tmp_path) -> None:
    """The edited result stays traceable through the existing asset store."""

    store = ImageAssetStore(tmp_path / "images")
    request = ImageEditRequest(mode="img2img", prompt="snow", source=_ref(), strength=0.5)
    image = GeneratedImage(
        data=b"\x89PNG\r\n\x1a\n" + b"payload" * 16,
        mime_type="image/png",
        width=64,
        height=64,
        metadata=request.manifest_hint(),
    )
    record = store.put(image, prompt=request.prompt)

    assert record.asset_id.startswith("img_")
    assert record.prompt == "snow"
    assert image.metadata["edit_mode"] == "img2img"
    assert image.metadata["source_asset_id"] == _ref().asset_id


def test_availability_is_contract_only_without_a_live_adapter() -> None:
    availability = resolve_edit_availability("img2img")

    assert availability.available is False
    assert availability.local_status == "unavailable"
    assert availability.remote_status == "contract_only"
    assert "local_edit_executor_unavailable" in availability.reasons
    assert "remote_edit_endpoint_not_verified" in availability.reasons


def test_declared_edit_support_without_a_live_runtime_is_not_available() -> None:
    capabilities = ImageAdapterCapabilities(
        backend="remote-qlh",
        model_ids=("sd15-fp16",),
        supports_edit=True,
        runtime_available=False,
    )
    availability = resolve_edit_availability("inpaint", capabilities=capabilities)

    assert availability.available is False
    assert "adapter_runtime_not_live" in availability.reasons


def test_availability_only_flips_with_adapter_evidence() -> None:
    capabilities = ImageAdapterCapabilities(
        backend="remote-qlh",
        model_ids=("sd15-fp16",),
        supports_edit=True,
        runtime_available=True,
        evidence={"verified_at": "fixture"},
    )
    availability = resolve_edit_availability("img2img", capabilities=capabilities)

    assert availability.available is True
    assert availability.remote_status == "available"
    assert availability.local_status == "unavailable"
    assert "local_edit_executor_unavailable" in availability.reasons


def test_capability_matrix_covers_every_mode_in_a_stable_order() -> None:
    matrix = edit_capability_matrix()

    assert tuple(item.mode for item in matrix) == EDIT_MODES
    assert all(item.available is False for item in matrix)
    assert all(item.as_dict()["spec"]["mode"] == item.mode for item in matrix)


def test_request_serialises_to_a_stable_contract() -> None:
    request = ImageEditRequest(mode="ip_adapter", prompt="style", reference=_ref())
    payload = request.as_dict()

    assert payload["request_schema"].startswith("qlh.harness.image_edit_request")
    assert payload["mode"] == "ip_adapter"
    assert payload["remote_payload"]["ip_adapter_ref"]["asset_id"] == _ref().asset_id
    assert payload["manifest_hint"]["reference_asset_id"] == _ref().asset_id
