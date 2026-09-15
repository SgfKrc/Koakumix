"""Versioned model profiles and capability gates for the harness."""

from .builtin import builtin_profiles
from .capability_gate import CapabilityGate, GateDecision
from .fleet_select import (
    ACCELERATORS,
    PLATFORMS,
    SELECTION_ROLES,
    SELECTION_SCHEMA,
    DeviceProfile,
    FleetCandidate,
    FleetSelection,
    FleetSelector,
    select_fleet_model,
)
from .manifest_bridge import (
    BRIDGE_SCHEMA,
    DEFAULT_BACKEND,
    KNOWN_FORMATS,
    MANIFEST_FILENAMES,
    MANIFEST_SCHEMA,
    UNVERSIONED_REVISION,
    ManifestBridgeError,
    bridge_report,
    find_manifest,
    load_manifest,
    model_id_from_source,
    profile_from_manifest,
)
from .probe import ProbeResult, probe_local_model
from .registry import ProfileAdmissionError, ProfileRegistry, profile_diff
from .schema import (
    CAPABILITY_NAMES,
    PROFILE_SCHEMA,
    PROFILE_STATUSES,
    CapabilityState,
    ModelProfile,
    ProfileValidationError,
)

__all__ = [
    "ACCELERATORS",
    "BRIDGE_SCHEMA",
    "CAPABILITY_NAMES",
    "DEFAULT_BACKEND",
    "KNOWN_FORMATS",
    "MANIFEST_FILENAMES",
    "MANIFEST_SCHEMA",
    "PLATFORMS",
    "PROFILE_SCHEMA",
    "PROFILE_STATUSES",
    "SELECTION_ROLES",
    "SELECTION_SCHEMA",
    "UNVERSIONED_REVISION",
    "CapabilityGate",
    "CapabilityState",
    "DeviceProfile",
    "FleetCandidate",
    "FleetSelection",
    "FleetSelector",
    "GateDecision",
    "ManifestBridgeError",
    "ModelProfile",
    "ProbeResult",
    "ProfileAdmissionError",
    "ProfileRegistry",
    "ProfileValidationError",
    "bridge_report",
    "builtin_profiles",
    "find_manifest",
    "load_manifest",
    "model_id_from_source",
    "probe_local_model",
    "profile_diff",
    "profile_from_manifest",
    "select_fleet_model",
]
