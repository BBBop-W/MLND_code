"""Freeze a qualified, balanced MLND benchmark before method comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import isclose, isfinite
import re
from typing import Any, Iterable, Mapping

from .pool import CandidateCase, CandidatePool, payload_sha256
from .qualification import QualificationDecision, SCALES, service_tolerance


SELECTION_RULE = "qualification_only_then_preregistered_candidate_order"
OVERCOMPLETE_SELECTION_RULE = (
    "first_qualified_by_preregistered_replacement_rank_within_block_scale"
)
_ALLOWED_SELECTION_RULES = {SELECTION_RULE, OVERCOMPLETE_SELECTION_RULE}
FORMAL_QUALIFICATION_POOL_LABEL_V5 = "mlnd-real-data-overcomplete-qualification-v5"
QUALIFICATION_MODEL_IDENTITY_SCHEMA = "mlnd-qualification-model-identity-v1"
_QUALIFICATION_MODEL_IDENTITY_KEYS = frozenset(
    {
        "schema",
        "effective_model_config_sha256",
        "qualification_case_sha256",
        "expected_screening_is_exact",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _parse_utc_timestamp(value: str, *, field_name: str) -> datetime:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    normalised = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field_name} must explicitly use UTC")
    return parsed


@dataclass(frozen=True, slots=True)
class FreezeMetadata:
    """Audit assertions proving selection was locked before comparison."""

    qualification_completed_at_utc: str
    manifest_frozen_at_utc: str
    comparison_started_at_utc: str | None = None
    performance_results_observed_during_selection: bool = False
    selection_rule: str = SELECTION_RULE

    def __post_init__(self) -> None:
        qualified_at = _parse_utc_timestamp(
            self.qualification_completed_at_utc,
            field_name="qualification_completed_at_utc",
        )
        frozen_at = _parse_utc_timestamp(
            self.manifest_frozen_at_utc,
            field_name="manifest_frozen_at_utc",
        )
        if qualified_at > frozen_at:
            raise ValueError("qualification must be completed before manifest freeze")
        if self.comparison_started_at_utc is not None:
            raise ValueError(
                "a benchmark manifest must be frozen before comparison starts"
            )
        if self.performance_results_observed_during_selection:
            raise ValueError(
                "performance results may not be observed during case selection"
            )
        if self.selection_rule not in _ALLOWED_SELECTION_RULES:
            raise ValueError(
                "selection_rule must be one of the pre-registered qualification "
                f"rules: {sorted(_ALLOWED_SELECTION_RULES)}"
            )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplacementSelectionProtocol:
    """Performance-blind rule for an overcomplete qualification pool."""

    target_cases_per_stratum: int
    expected_stratum_count: int
    selection_rule: str = OVERCOMPLETE_SELECTION_RULE
    requires_contiguous_qualification_prefix: bool = True
    selection_uses_algorithm_performance: bool = False
    forced_opening_constraints_added: bool = False

    def __post_init__(self) -> None:
        if self.selection_rule != OVERCOMPLETE_SELECTION_RULE:
            raise ValueError("unsupported overcomplete selection rule")
        if self.target_cases_per_stratum < 1:
            raise ValueError("target_cases_per_stratum must be positive")
        if self.expected_stratum_count < 1:
            raise ValueError("expected_stratum_count must be positive")
        if not self.requires_contiguous_qualification_prefix:
            raise ValueError(
                "qualification must follow a contiguous replacement prefix"
            )
        if self.selection_uses_algorithm_performance:
            raise ValueError("algorithm performance may not influence case selection")
        if self.forced_opening_constraints_added:
            raise ValueError("qualification may not force facilities open")

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectionLedgerEntry:
    """Qualification/selection status for every pre-registered replacement."""

    case_id: str
    stratum_id: str
    replacement_rank_within_stratum: int
    qualification_status: str
    selected: bool
    rejection_reasons: tuple[str, ...] = ()
    qualification_run_id: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id or not self.stratum_id:
            raise ValueError("selection-ledger IDs must be non-empty")
        if self.replacement_rank_within_stratum < 0:
            raise ValueError("replacement rank must be non-negative")
        if self.qualification_status not in {"accepted", "rejected", "not_attempted"}:
            raise ValueError("invalid qualification_status")
        if self.selected and self.qualification_status != "accepted":
            raise ValueError("only an accepted qualification may be selected")
        if self.qualification_status == "rejected" and not self.rejection_reasons:
            raise ValueError("a rejected ledger entry must retain rejection reasons")
        if self.qualification_status != "rejected" and self.rejection_reasons:
            raise ValueError("only rejected entries may contain rejection reasons")
        attempted = self.qualification_status != "not_attempted"
        if attempted != bool(self.qualification_run_id):
            raise ValueError(
                "qualification_run_id must be present exactly for attempted entries"
            )

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rejection_reasons"] = list(self.rejection_reasons)
        return payload


@dataclass(frozen=True, slots=True)
class FrozenCase:
    candidate: CandidateCase
    qualification: QualificationDecision
    selection_rank_within_scale: int

    def __post_init__(self) -> None:
        if not self.qualification.accepted:
            raise ValueError("only accepted qualifications may be frozen")
        if self.candidate.case_id != self.qualification.case_id:
            raise ValueError("candidate and qualification case IDs do not match")
        if self.candidate.scale != self.qualification.scale:
            raise ValueError("candidate and qualification scales do not match")
        if self.candidate.instance_sha256 != self.qualification.instance_sha256:
            raise ValueError("candidate and qualification instance hashes do not match")
        if self.selection_rank_within_scale < 0:
            raise ValueError("selection_rank_within_scale must be non-negative")

    def to_payload(self, *, candidate_seed_rank: int) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_payload(seed_rank=candidate_seed_rank),
            "qualification": self.qualification.to_payload(),
            "selection_rank_within_scale": self.selection_rank_within_scale,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """Immutable logical manifest; ``sha256`` is its content address."""

    cases: tuple[FrozenCase, ...]
    qualification_ledger: tuple[QualificationDecision, ...]
    candidate_seed_order: tuple[int, ...]
    source_pool_sha256: str
    source_pool_label: str
    freeze_metadata: FreezeMetadata
    minimum_case_count: int = 36
    minimum_cases_per_scale: int = 12
    schema_version: int = 1
    replacement_selection_protocol: ReplacementSelectionProtocol | None = None
    selection_ledger: tuple[SelectionLedgerEntry, ...] = ()

    def __post_init__(self) -> None:
        validate_manifest(self)

    def to_payload(self) -> dict[str, Any]:
        seed_ranks = {seed: rank for rank, seed in enumerate(self.candidate_seed_order)}
        return {
            "schema_version": self.schema_version,
            "source_pool_sha256": self.source_pool_sha256,
            "source_pool_label": self.source_pool_label,
            "candidate_seed_order": list(self.candidate_seed_order),
            "minimum_case_count": self.minimum_case_count,
            "minimum_cases_per_scale": self.minimum_cases_per_scale,
            "freeze_metadata": self.freeze_metadata.to_payload(),
            "cases": [
                case.to_payload(
                    candidate_seed_rank=seed_ranks[case.candidate.layout_seed]
                )
                for case in self.cases
            ],
            "qualification_ledger": [
                decision.to_payload() for decision in self.qualification_ledger
            ],
            "replacement_selection_protocol": (
                self.replacement_selection_protocol.to_payload()
                if self.replacement_selection_protocol is not None
                else None
            ),
            "selection_ledger": [entry.to_payload() for entry in self.selection_ledger],
            "rejection_ledger": [
                entry.to_payload()
                for entry in self.selection_ledger
                if entry.qualification_status == "rejected"
            ],
        }

    @property
    def sha256(self) -> str:
        return payload_sha256(self.to_payload())

    def artifact_payload(self) -> dict[str, Any]:
        payload = self.to_payload()
        payload["manifest_sha256"] = self.sha256
        return payload


def _decision_map(
    decisions: Iterable[QualificationDecision],
) -> dict[str, QualificationDecision]:
    by_case: dict[str, QualificationDecision] = {}
    for decision in decisions:
        if decision.case_id in by_case:
            raise ValueError(f"duplicate qualification for case {decision.case_id!r}")
        by_case[decision.case_id] = decision
    return by_case


@dataclass(frozen=True, slots=True)
class _CandidateReplacement:
    stratum_id: str
    replacement_rank: int
    target_cases_per_stratum: int
    expected_stratum_count: int


@dataclass(frozen=True, slots=True)
class _CandidateQualificationModelIdentity:
    qualification_case_sha256: str
    effective_model_config_sha256: str
    expected_screening_is_exact: bool


def _candidate_declares_model_identity(candidate: CandidateCase) -> bool:
    return "qualification_model_identity" in candidate.generator_parameters


def _candidate_qualification_model_identity(
    candidate: CandidateCase,
    *,
    required: bool,
    enforce_locked_replacement_binding: bool = False,
) -> _CandidateQualificationModelIdentity | None:
    raw_identity = candidate.generator_parameters.get("qualification_model_identity")
    if raw_identity is None:
        if required:
            raise ValueError(
                f"formal v5 case {candidate.case_id!r} is missing "
                "qualification_model_identity"
            )
        return None
    if not isinstance(raw_identity, Mapping):
        raise ValueError(
            f"case {candidate.case_id!r} has invalid qualification_model_identity"
        )
    if set(raw_identity) != _QUALIFICATION_MODEL_IDENTITY_KEYS:
        raise ValueError(
            f"case {candidate.case_id!r} qualification_model_identity must contain "
            "exactly the locked four fields"
        )
    if raw_identity["schema"] != QUALIFICATION_MODEL_IDENTITY_SCHEMA:
        raise ValueError(
            f"case {candidate.case_id!r} has an unsupported qualification model "
            "identity schema"
        )
    declared_qualification_case_sha256 = raw_identity["qualification_case_sha256"]
    declared_effective_model_config_sha256 = raw_identity[
        "effective_model_config_sha256"
    ]
    if (
        not isinstance(declared_qualification_case_sha256, str)
        or _SHA256_RE.fullmatch(declared_qualification_case_sha256) is None
    ):
        raise ValueError(
            f"case {candidate.case_id!r} has an invalid qualification case hash"
        )
    if (
        not isinstance(declared_effective_model_config_sha256, str)
        or _SHA256_RE.fullmatch(declared_effective_model_config_sha256) is None
    ):
        raise ValueError(
            f"case {candidate.case_id!r} has an invalid effective model config hash"
        )
    expected_screening_is_exact = raw_identity["expected_screening_is_exact"]
    if type(expected_screening_is_exact) is not bool:
        raise ValueError(
            f"case {candidate.case_id!r} expected_screening_is_exact must be boolean"
        )
    raw_profile = candidate.generator_parameters.get("qualification_solve_profile")
    if not isinstance(raw_profile, Mapping):
        raise ValueError(
            f"formal v5 case {candidate.case_id!r} is missing its qualification "
            "solve profile"
        )
    # ``case_builder`` imports this module for the selection-rule constants, so
    # keep the inverse dependency local to runtime after both modules are fully
    # initialized.
    from .case_builder import (
        LOCKED_QUALIFICATION_PROFILES,
        QualificationSolveProfile,
        locked_replacement_order,
        qualification_case_sha256,
    )

    try:
        profile = QualificationSolveProfile(**dict(raw_profile))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"case {candidate.case_id!r} has an invalid qualification solve profile"
        ) from error
    if set(raw_profile) != set(profile.to_payload()):
        raise ValueError(
            f"case {candidate.case_id!r} qualification solve profile must contain "
            "the complete locked payload"
        )
    if profile not in LOCKED_QUALIFICATION_PROFILES:
        raise ValueError(
            f"case {candidate.case_id!r} does not use a locked qualification profile"
        )
    if enforce_locked_replacement_binding:
        if not candidate.case_id.endswith(f"__profile-{profile.profile_id}"):
            raise ValueError(
                f"case {candidate.case_id!r} ID does not bind its qualification profile"
            )
        selection = candidate.generator_parameters.get("pre_registered_selection")
        if not isinstance(selection, Mapping):
            raise ValueError(
                f"formal v5 case {candidate.case_id!r} is missing replacement metadata"
            )
        replacement_rank = selection.get("replacement_rank_within_stratum")
        replacement_order = locked_replacement_order()
        if (
            isinstance(replacement_rank, bool)
            or not isinstance(replacement_rank, int)
            or not 0 <= replacement_rank < len(replacement_order)
        ):
            raise ValueError(
                f"formal v5 case {candidate.case_id!r} has an invalid replacement rank"
            )
        expected_profile_id, expected_seed = replacement_order[replacement_rank]
        if (
            profile.profile_id != expected_profile_id
            or candidate.layout_seed != expected_seed
        ):
            raise ValueError(
                f"formal v5 case {candidate.case_id!r} profile/seed does not match its "
                "locked replacement rank"
            )
    recomputed_effective_config_sha256 = profile.effective_model_config_sha256()
    if declared_effective_model_config_sha256 != recomputed_effective_config_sha256:
        raise ValueError(
            f"case {candidate.case_id!r} effective model config hash does not match "
            "its qualification profile"
        )
    recomputed_case_sha256 = qualification_case_sha256(
        candidate.instance_sha256,
        profile,
    )
    if declared_qualification_case_sha256 != recomputed_case_sha256:
        raise ValueError(
            f"case {candidate.case_id!r} qualification case hash does not match its "
            "physical instance and profile"
        )
    if expected_screening_is_exact is not profile.expected_screening_is_exact:
        raise ValueError(
            f"case {candidate.case_id!r} screening exactness does not match its "
            "qualification profile"
        )
    return _CandidateQualificationModelIdentity(
        qualification_case_sha256=declared_qualification_case_sha256,
        effective_model_config_sha256=declared_effective_model_config_sha256,
        expected_screening_is_exact=expected_screening_is_exact,
    )


def _pool_requires_model_identity(pool: CandidatePool) -> bool:
    return pool.pool_label == FORMAL_QUALIFICATION_POOL_LABEL_V5 or any(
        _candidate_declares_model_identity(case) for case in pool.candidates
    )


def _manifest_requires_model_identity(manifest: BenchmarkManifest) -> bool:
    return (
        manifest.source_pool_label == FORMAL_QUALIFICATION_POOL_LABEL_V5
        or any(
            _candidate_declares_model_identity(case.candidate)
            for case in manifest.cases
        )
        or any(
            decision.qualification_case_sha256 is not None
            for decision in manifest.qualification_ledger
        )
    )


def _required_int(value: Any, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _candidate_replacement(
    candidate: CandidateCase,
) -> _CandidateReplacement | None:
    metadata = candidate.generator_parameters.get("pre_registered_selection")
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"case {candidate.case_id!r} has invalid pre_registered_selection"
        )
    if metadata.get("selection_rule") != OVERCOMPLETE_SELECTION_RULE:
        raise ValueError(
            f"case {candidate.case_id!r} has an unsupported selection rule"
        )
    if metadata.get("selection_uses_algorithm_performance") is not False:
        raise ValueError(
            f"case {candidate.case_id!r} does not certify performance-blind selection"
        )
    if metadata.get("forced_opening_constraints_added") is not False:
        raise ValueError(f"case {candidate.case_id!r} may not force facilities open")
    expected_large_target = [50, 60] if candidate.scale == "large" else None
    if metadata.get("large_scale_open_h_qualification_target") != expected_large_target:
        raise ValueError(
            f"case {candidate.case_id!r} has the wrong large-scale H-open target"
        )
    solve_profile = candidate.generator_parameters.get("qualification_solve_profile")
    if not isinstance(solve_profile, Mapping):
        raise ValueError(
            f"case {candidate.case_id!r} is missing qualification_solve_profile"
        )
    if solve_profile.get("adds_forced_opening_constraint") is not False:
        raise ValueError(
            f"case {candidate.case_id!r} solve profile may not force facilities open"
        )
    stratum_id = metadata.get("stratum_id")
    expected_stratum_id = f"{candidate.demand_block_id}__{candidate.scale}"
    if stratum_id != expected_stratum_id:
        raise ValueError(f"case {candidate.case_id!r} has an invalid selection stratum")
    return _CandidateReplacement(
        stratum_id=stratum_id,
        replacement_rank=_required_int(
            metadata.get("replacement_rank_within_stratum"),
            field_name="replacement_rank_within_stratum",
            minimum=0,
        ),
        target_cases_per_stratum=_required_int(
            metadata.get("target_cases_per_stratum"),
            field_name="target_cases_per_stratum",
            minimum=1,
        ),
        expected_stratum_count=_required_int(
            metadata.get("expected_stratum_count"),
            field_name="expected_stratum_count",
            minimum=1,
        ),
    )


def _validate_decision_candidate(
    decision: QualificationDecision,
    candidate: CandidateCase,
    *,
    require_model_identity: bool = False,
    enforce_locked_replacement_binding: bool = False,
) -> None:
    if decision.scale != candidate.scale:
        raise ValueError(f"qualification scale mismatch for case {candidate.case_id!r}")
    if decision.instance_sha256 != candidate.instance_sha256:
        raise ValueError(
            f"qualification instance hash mismatch for case {candidate.case_id!r}"
        )
    model_identity = _candidate_qualification_model_identity(
        candidate,
        required=require_model_identity,
        enforce_locked_replacement_binding=enforce_locked_replacement_binding,
    )
    if model_identity is None:
        if decision.qualification_case_sha256 is not None:
            raise ValueError(
                f"qualification decision for legacy case {candidate.case_id!r} "
                "has no candidate-owned model identity"
            )
    elif (
        decision.qualification_case_sha256 != model_identity.qualification_case_sha256
        or decision.effective_model_config_sha256
        != model_identity.effective_model_config_sha256
        or decision.screening_is_exact is not model_identity.expected_screening_is_exact
    ):
        raise ValueError(
            f"qualification model identity mismatch for case {candidate.case_id!r}"
        )
    if decision.active_h is not None and (
        decision.active_h < 0 or decision.active_h > decision.open_h
    ):
        raise ValueError(
            f"qualification active-H count is invalid for {candidate.case_id!r}"
        )
    if (
        candidate.scale == "large"
        and decision.accepted
        and decision.policy.large_scale_open_h_target is not None
    ):
        target_lower, target_upper = decision.policy.large_scale_open_h_target
        if decision.active_h is None or not (
            target_lower <= decision.active_h <= target_upper
        ):
            raise ValueError(
                "an accepted large qualification must meet its active-H target"
            )
        if (
            decision.total_demand_per_hour is None
            or not isfinite(decision.total_demand_per_hour)
            or decision.total_demand_per_hour <= 0.0
        ):
            raise ValueError(
                "an accepted large qualification must record positive finite "
                "total demand"
            )
        expected_threshold = service_tolerance(decision.total_demand_per_hour)
        if decision.active_h_threshold_per_hour is None or not isclose(
            decision.active_h_threshold_per_hour,
            expected_threshold,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "an accepted large qualification must record its auditable "
                "active-H threshold"
            )
    replacement = _candidate_replacement(candidate)
    if replacement is None:
        return
    candidate_counts = candidate.generator_parameters.get("candidate_counts")
    if not isinstance(candidate_counts, Mapping):
        raise ValueError(
            f"case {candidate.case_id!r} is missing locked candidate counts"
        )
    expected_h = _required_int(
        candidate_counts.get("H"),
        field_name="candidate_counts.H",
        minimum=1,
    )
    if decision.candidate_h != expected_h:
        raise ValueError(
            f"qualification H-candidate count mismatch for {candidate.case_id!r}"
        )
    if candidate.scale == "large":
        if expected_h != 90:
            raise ValueError(
                "large qualification cases must have exactly 90 H candidates"
            )
        if decision.policy.large_scale_open_h_target != (50, 60):
            raise ValueError(
                "large qualification decisions must use the locked 50--60 H target"
            )


def _freeze_overcomplete(
    pool: CandidatePool,
    by_case: Mapping[str, QualificationDecision],
    *,
    freeze_metadata: FreezeMetadata,
    minimum_case_count: int,
    minimum_cases_per_scale: int,
    require_model_identity: bool,
    enforce_locked_replacement_binding: bool,
) -> BenchmarkManifest:
    candidates_by_stratum: dict[str, list[tuple[int, CandidateCase]]] = {}
    target_values: set[int] = set()
    expected_stratum_values: set[int] = set()
    for candidate in pool.candidates:
        replacement = _candidate_replacement(candidate)
        if replacement is None:
            raise ValueError(
                "overcomplete pools require selection metadata on every candidate"
            )
        candidates_by_stratum.setdefault(replacement.stratum_id, []).append(
            (replacement.replacement_rank, candidate)
        )
        target_values.add(replacement.target_cases_per_stratum)
        expected_stratum_values.add(replacement.expected_stratum_count)

    if len(target_values) != 1 or len(expected_stratum_values) != 1:
        raise ValueError("overcomplete selection metadata is inconsistent across cases")
    target = next(iter(target_values))
    expected_strata = next(iter(expected_stratum_values))
    if len(candidates_by_stratum) != expected_strata:
        raise ValueError(
            f"pool has {len(candidates_by_stratum)} selection strata; "
            f"expected {expected_strata}"
        )
    if freeze_metadata.selection_rule != OVERCOMPLETE_SELECTION_RULE:
        raise ValueError(
            "freeze metadata must lock the overcomplete replacement selection rule"
        )

    scale_rank = {scale: index for index, scale in enumerate(SCALES)}
    ordered_strata = sorted(
        candidates_by_stratum,
        key=lambda stratum: (
            scale_rank[candidates_by_stratum[stratum][0][1].scale],
            candidates_by_stratum[stratum][0][1].demand_block_id,
            stratum,
        ),
    )
    selected_candidates: list[tuple[CandidateCase, QualificationDecision]] = []
    qualification_ledger: list[QualificationDecision] = []
    selection_ledger: list[SelectionLedgerEntry] = []
    for stratum in ordered_strata:
        ranked = sorted(candidates_by_stratum[stratum], key=lambda item: item[0])
        ranks = [rank for rank, _candidate in ranked]
        if ranks != list(range(len(ranked))):
            raise ValueError(
                f"replacement ranks for stratum {stratum!r} must be contiguous from zero"
            )
        attempted_ranks = [
            rank for rank, candidate in ranked if candidate.case_id in by_case
        ]
        if attempted_ranks != list(range(len(attempted_ranks))):
            raise ValueError(
                f"qualification decisions for stratum {stratum!r} must be a "
                "contiguous prefix of the pre-registered replacement order"
            )

        selected_ids: set[str] = set()
        for rank, candidate in ranked:
            decision = by_case.get(candidate.case_id)
            if decision is None:
                continue
            _validate_decision_candidate(
                decision,
                candidate,
                require_model_identity=require_model_identity,
                enforce_locked_replacement_binding=(enforce_locked_replacement_binding),
            )
            qualification_ledger.append(decision)
            if decision.accepted and len(selected_ids) < target:
                selected_ids.add(candidate.case_id)
                selected_candidates.append((candidate, decision))
        if len(selected_ids) < target:
            next_candidate = (
                ranked[len(attempted_ranks)][1].case_id
                if len(attempted_ranks) < len(ranked)
                else None
            )
            raise ValueError(
                f"stratum {stratum!r} has only {len(selected_ids)} qualified cases; "
                f"requires {target}; next pre-registered candidate={next_candidate!r}"
            )

        for rank, candidate in ranked:
            decision = by_case.get(candidate.case_id)
            if decision is None:
                selection_ledger.append(
                    SelectionLedgerEntry(
                        case_id=candidate.case_id,
                        stratum_id=stratum,
                        replacement_rank_within_stratum=rank,
                        qualification_status="not_attempted",
                        selected=False,
                    )
                )
            elif decision.accepted:
                selection_ledger.append(
                    SelectionLedgerEntry(
                        case_id=candidate.case_id,
                        stratum_id=stratum,
                        replacement_rank_within_stratum=rank,
                        qualification_status="accepted",
                        selected=candidate.case_id in selected_ids,
                        qualification_run_id=decision.qualification_run_id,
                    )
                )
            else:
                selection_ledger.append(
                    SelectionLedgerEntry(
                        case_id=candidate.case_id,
                        stratum_id=stratum,
                        replacement_rank_within_stratum=rank,
                        qualification_status="rejected",
                        selected=False,
                        rejection_reasons=decision.rejection_reasons,
                        qualification_run_id=decision.qualification_run_id,
                    )
                )

    ranks_within_scale = {scale: 0 for scale in SCALES}
    frozen_cases: list[FrozenCase] = []
    for candidate, decision in selected_candidates:
        frozen_cases.append(
            FrozenCase(
                candidate=candidate,
                qualification=decision,
                selection_rank_within_scale=ranks_within_scale[candidate.scale],
            )
        )
        ranks_within_scale[candidate.scale] += 1

    return BenchmarkManifest(
        cases=tuple(frozen_cases),
        qualification_ledger=tuple(qualification_ledger),
        candidate_seed_order=pool.candidate_seed_order,
        source_pool_sha256=pool.sha256,
        source_pool_label=pool.pool_label,
        freeze_metadata=freeze_metadata,
        minimum_case_count=minimum_case_count,
        minimum_cases_per_scale=minimum_cases_per_scale,
        replacement_selection_protocol=ReplacementSelectionProtocol(
            target_cases_per_stratum=target,
            expected_stratum_count=expected_strata,
        ),
        selection_ledger=tuple(selection_ledger),
    )


def freeze_manifest(
    pool: CandidatePool,
    decisions: Iterable[QualificationDecision],
    *,
    freeze_metadata: FreezeMetadata,
    minimum_case_count: int = 36,
    minimum_cases_per_scale: int = 12,
) -> BenchmarkManifest:
    """Freeze cases using only qualification evidence and locked pool order.

    A legacy closed pool requires a decision for every candidate.  An
    overcomplete pool permits a contiguous qualification prefix within each
    block-scale stratum and selects the first qualifying replacements.  Both
    modes retain every attempted rejection and never inspect A--E performance.
    """

    if minimum_case_count < 1:
        raise ValueError("minimum_case_count must be positive")
    if minimum_cases_per_scale < 1:
        raise ValueError("minimum_cases_per_scale must be positive")

    by_case = _decision_map(decisions)
    pool_ids = {candidate.case_id for candidate in pool.candidates}
    extra = sorted(by_case.keys() - pool_ids)
    if extra:
        raise ValueError(f"qualification decisions reference unknown cases: {extra}")

    require_model_identity = _pool_requires_model_identity(pool)
    enforce_locked_replacement_binding = (
        pool.pool_label == FORMAL_QUALIFICATION_POOL_LABEL_V5
    )
    for candidate in pool.candidates:
        _candidate_qualification_model_identity(
            candidate,
            required=require_model_identity,
            enforce_locked_replacement_binding=enforce_locked_replacement_binding,
        )

    replacement_metadata = [
        _candidate_replacement(candidate) for candidate in pool.candidates
    ]
    if any(item is not None for item in replacement_metadata):
        if not all(item is not None for item in replacement_metadata):
            raise ValueError(
                "a pool may not mix legacy and overcomplete candidate contracts"
            )
        return _freeze_overcomplete(
            pool,
            by_case,
            freeze_metadata=freeze_metadata,
            minimum_case_count=minimum_case_count,
            minimum_cases_per_scale=minimum_cases_per_scale,
            require_model_identity=require_model_identity,
            enforce_locked_replacement_binding=enforce_locked_replacement_binding,
        )

    missing = sorted(pool_ids - by_case.keys())
    if missing:
        raise ValueError(f"qualification decisions missing for cases: {missing}")

    selected: list[FrozenCase] = []
    ranks = {scale: 0 for scale in SCALES}
    ledger: list[QualificationDecision] = []
    for candidate in pool.ordered_candidates():
        decision = by_case[candidate.case_id]
        _validate_decision_candidate(
            decision,
            candidate,
            require_model_identity=require_model_identity,
            enforce_locked_replacement_binding=enforce_locked_replacement_binding,
        )
        ledger.append(decision)
        if decision.accepted:
            selected.append(
                FrozenCase(
                    candidate=candidate,
                    qualification=decision,
                    selection_rank_within_scale=ranks[candidate.scale],
                )
            )
            ranks[candidate.scale] += 1

    return BenchmarkManifest(
        cases=tuple(selected),
        qualification_ledger=tuple(ledger),
        candidate_seed_order=pool.candidate_seed_order,
        source_pool_sha256=pool.sha256,
        source_pool_label=pool.pool_label,
        freeze_metadata=freeze_metadata,
        minimum_case_count=minimum_case_count,
        minimum_cases_per_scale=minimum_cases_per_scale,
    )


def _validate_replacement_selection_ledger(
    manifest: BenchmarkManifest,
) -> dict[str, SelectionLedgerEntry]:
    protocol = manifest.replacement_selection_protocol
    if protocol is None:
        if manifest.selection_ledger:
            raise ValueError(
                "selection_ledger requires a replacement_selection_protocol"
            )
        if manifest.freeze_metadata.selection_rule != SELECTION_RULE:
            raise ValueError("legacy manifests must use the legacy selection rule")
        return {}
    if manifest.freeze_metadata.selection_rule != protocol.selection_rule:
        raise ValueError("freeze metadata and replacement selection rule differ")
    if (
        protocol.target_cases_per_stratum * protocol.expected_stratum_count
        < manifest.minimum_case_count
    ):
        raise ValueError("replacement protocol cannot supply the minimum case count")

    entries_by_case: dict[str, SelectionLedgerEntry] = {}
    entries_by_stratum: dict[str, list[SelectionLedgerEntry]] = {}
    for entry in manifest.selection_ledger:
        if entry.case_id in entries_by_case:
            raise ValueError("selection-ledger case IDs must be unique")
        entries_by_case[entry.case_id] = entry
        entries_by_stratum.setdefault(entry.stratum_id, []).append(entry)
    if len(entries_by_stratum) != protocol.expected_stratum_count:
        raise ValueError(
            "selection ledger does not contain the expected number of strata"
        )

    ledger_decisions = {
        decision.case_id: decision for decision in manifest.qualification_ledger
    }
    attempted_ids: set[str] = set()
    selected_ids: set[str] = set()
    for stratum, entries in entries_by_stratum.items():
        ordered = sorted(
            entries,
            key=lambda entry: entry.replacement_rank_within_stratum,
        )
        if [entry.replacement_rank_within_stratum for entry in ordered] != list(
            range(len(ordered))
        ):
            raise ValueError(
                f"selection-ledger ranks for {stratum!r} are not contiguous"
            )
        attempted = [
            entry for entry in ordered if entry.qualification_status != "not_attempted"
        ]
        if attempted != ordered[: len(attempted)]:
            raise ValueError(
                f"selection ledger for {stratum!r} is not a qualification prefix"
            )
        accepted = [
            entry for entry in attempted if entry.qualification_status == "accepted"
        ]
        expected_selected = accepted[: protocol.target_cases_per_stratum]
        actual_selected = [entry for entry in ordered if entry.selected]
        if actual_selected != expected_selected:
            raise ValueError(
                f"stratum {stratum!r} did not select the first qualified replacements"
            )
        if len(actual_selected) != protocol.target_cases_per_stratum:
            raise ValueError(f"stratum {stratum!r} is underfilled")

        for entry in attempted:
            decision = ledger_decisions.get(entry.case_id)
            if decision is None:
                raise ValueError(
                    f"attempted selection entry {entry.case_id!r} lacks a decision"
                )
            expected_status = "accepted" if decision.accepted else "rejected"
            if entry.qualification_status != expected_status:
                raise ValueError("selection and qualification ledger statuses differ")
            if entry.qualification_run_id != decision.qualification_run_id:
                raise ValueError("selection and qualification run IDs differ")
            if entry.rejection_reasons != decision.rejection_reasons:
                raise ValueError(
                    "selection ledger lost qualification rejection reasons"
                )
            attempted_ids.add(entry.case_id)
        selected_ids.update(entry.case_id for entry in actual_selected)

    if attempted_ids != set(ledger_decisions):
        raise ValueError("qualification ledger and attempted selection entries differ")
    if selected_ids != {case.candidate.case_id for case in manifest.cases}:
        raise ValueError("selection ledger and frozen cases differ")
    return entries_by_case


def validate_manifest(manifest: BenchmarkManifest) -> None:
    """Validate balancing, identity, ordering, and selection audit invariants."""

    if manifest.schema_version != 1:
        raise ValueError("unsupported benchmark-manifest schema_version")
    if manifest.minimum_case_count < 36:
        raise ValueError("formal benchmark must require at least 36 cases")
    if manifest.minimum_cases_per_scale < 12:
        raise ValueError("formal benchmark must require at least 12 cases per scale")
    if len(manifest.cases) < manifest.minimum_case_count:
        raise ValueError(
            f"manifest has {len(manifest.cases)} cases; "
            f"requires at least {manifest.minimum_case_count}"
        )
    if not manifest.source_pool_sha256:
        raise ValueError("source_pool_sha256 must be non-empty")
    if not manifest.source_pool_label:
        raise ValueError("source_pool_label must be non-empty")
    if not manifest.candidate_seed_order:
        raise ValueError("candidate_seed_order must be locked in the manifest")
    if len(set(manifest.candidate_seed_order)) != len(manifest.candidate_seed_order):
        raise ValueError("candidate_seed_order must contain unique seeds")

    case_ids = [case.candidate.case_id for case in manifest.cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("frozen case IDs must be unique")
    ledger_ids = [decision.case_id for decision in manifest.qualification_ledger]
    if len(set(ledger_ids)) != len(ledger_ids):
        raise ValueError("qualification-ledger case IDs must be unique")
    ledger_by_case: Mapping[str, QualificationDecision] = {
        decision.case_id: decision for decision in manifest.qualification_ledger
    }
    require_model_identity = _manifest_requires_model_identity(manifest)
    enforce_locked_replacement_binding = (
        manifest.source_pool_label == FORMAL_QUALIFICATION_POOL_LABEL_V5
    )
    if require_model_identity:
        missing_identity = sorted(
            decision.case_id
            for decision in manifest.qualification_ledger
            if decision.qualification_case_sha256 is None
        )
        if missing_identity:
            raise ValueError(
                "formal v5 qualification ledger is missing model identity for "
                f"cases: {missing_identity}"
            )
    selection_by_case = _validate_replacement_selection_ledger(manifest)

    scale_counts = {scale: 0 for scale in SCALES}
    expected_ranks = {scale: 0 for scale in SCALES}
    previous_order_key: tuple[Any, ...] | None = None
    seed_rank = {seed: rank for rank, seed in enumerate(manifest.candidate_seed_order)}
    scale_rank = {scale: index for index, scale in enumerate(SCALES)}
    for frozen_case in manifest.cases:
        candidate = frozen_case.candidate
        if candidate.layout_seed not in seed_rank:
            raise ValueError(
                f"case {candidate.case_id!r} uses a seed absent from locked order"
            )
        selection_entry = selection_by_case.get(candidate.case_id)
        if selection_entry is None:
            order_key = (
                scale_rank[candidate.scale],
                candidate.demand_block_id,
                seed_rank[candidate.layout_seed],
                candidate.case_id,
            )
        else:
            replacement = _candidate_replacement(candidate)
            if replacement is None:
                raise ValueError("frozen replacement case lost its selection metadata")
            if (
                replacement.stratum_id != selection_entry.stratum_id
                or replacement.replacement_rank
                != selection_entry.replacement_rank_within_stratum
            ):
                raise ValueError("frozen case and selection-ledger rank differ")
            order_key = (
                scale_rank[candidate.scale],
                candidate.demand_block_id,
                selection_entry.stratum_id,
                selection_entry.replacement_rank_within_stratum,
            )
        if previous_order_key is not None and order_key <= previous_order_key:
            raise ValueError("frozen cases are not in pre-registered candidate order")
        previous_order_key = order_key
        if frozen_case.selection_rank_within_scale != expected_ranks[candidate.scale]:
            raise ValueError("selection ranks within a scale must be contiguous")
        expected_ranks[candidate.scale] += 1
        scale_counts[candidate.scale] += 1

        ledger_decision = ledger_by_case.get(candidate.case_id)
        if ledger_decision != frozen_case.qualification:
            raise ValueError(
                f"frozen qualification is absent/mismatched in ledger for "
                f"{candidate.case_id!r}"
            )
        _validate_decision_candidate(
            frozen_case.qualification,
            candidate,
            require_model_identity=require_model_identity,
            enforce_locked_replacement_binding=enforce_locked_replacement_binding,
        )

    underfilled = {
        scale: count
        for scale, count in scale_counts.items()
        if count < manifest.minimum_cases_per_scale
    }
    if underfilled:
        raise ValueError(
            "manifest must contain the minimum number of cases at every scale: "
            f"{underfilled}"
        )
