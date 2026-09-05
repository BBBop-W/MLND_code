"""Deterministic, performance-blind candidate pools for B&C experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable, Mapping

from .hashing import canonical_json, canonical_sha256
from .qualification import SCALES


_SCALE_RANK = {scale: index for index, scale in enumerate(SCALES)}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These fields reveal downstream solver performance and therefore may not be
# smuggled into instance-generation metadata used to order/select candidates.
_PROHIBITED_SELECTION_FIELDS = frozenset(
    {
        "bnc_gap",
        "bnc_runtime_seconds",
        "comparison_result",
        "comparison_results",
        "incumbent_objective",
        "method_performance",
        "mip_gap",
        "native_gap",
        "native_runtime_seconds",
        "node_count",
        "objective_bound",
        "profile_results",
        "runtime_seconds",
        "solve_time",
        "speedup",
        "winner",
    }
)


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode JSON deterministically for immutable artifact identities."""

    return canonical_json(payload).encode("utf-8")


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(payload)


def _normalise_field_name(name: object) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def assert_performance_blind_metadata(value: Any, *, path: str = "metadata") -> None:
    """Reject comparison-outcome fields recursively from selection metadata."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalised = _normalise_field_name(key)
            if normalised in _PROHIBITED_SELECTION_FIELDS:
                raise ValueError(
                    f"performance-based selection field is prohibited: {path}.{key}"
                )
            assert_performance_blind_metadata(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_performance_blind_metadata(child, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class CandidateCase:
    """One spatial crop, demand block, and pre-registered layout seed."""

    case_id: str
    scale: str
    demand_block_id: str
    layout_seed: int
    instance_ref: str
    instance_sha256: str
    generator_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.scale not in SCALES:
            raise ValueError(f"scale must be one of {SCALES}")
        if not self.demand_block_id:
            raise ValueError("demand_block_id must be non-empty")
        if isinstance(self.layout_seed, bool) or not isinstance(self.layout_seed, int):
            raise ValueError("layout_seed must be an integer")
        if not self.instance_ref:
            raise ValueError("instance_ref must be non-empty")
        if _SHA256_RE.fullmatch(self.instance_sha256) is None:
            raise ValueError(
                "instance_sha256 must be a lowercase 64-character SHA256 digest"
            )
        assert_performance_blind_metadata(
            self.generator_parameters,
            path=f"candidate[{self.case_id}].generator_parameters",
        )

    def to_payload(self, *, seed_rank: int | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["generator_parameters"] = dict(self.generator_parameters)
        if seed_rank is not None:
            payload["candidate_seed_rank"] = seed_rank
        return payload


def candidate_order_key(
    candidate: CandidateCase,
    candidate_seed_order: tuple[int, ...],
) -> tuple[int, str, int, str]:
    """Stable, pre-performance ordering used by qualification and freezing."""

    try:
        seed_rank = candidate_seed_order.index(candidate.layout_seed)
    except ValueError as error:
        raise ValueError(
            f"candidate {candidate.case_id!r} uses unregistered seed "
            f"{candidate.layout_seed}"
        ) from error
    order_rank = seed_rank
    selection = candidate.generator_parameters.get("pre_registered_selection")
    if (
        isinstance(selection, Mapping)
        and "replacement_rank_within_stratum" in selection
    ):
        replacement_rank = selection["replacement_rank_within_stratum"]
        if (
            isinstance(replacement_rank, bool)
            or not isinstance(replacement_rank, int)
            or replacement_rank < 0
        ):
            raise ValueError(
                f"candidate {candidate.case_id!r} has an invalid replacement rank"
            )
        order_rank = replacement_rank
    return (
        _SCALE_RANK[candidate.scale],
        candidate.demand_block_id,
        order_rank,
        candidate.case_id,
    )


@dataclass(frozen=True, slots=True)
class CandidatePool:
    """A closed candidate set whose ordering cannot use solver outcomes."""

    candidates: tuple[CandidateCase, ...]
    candidate_seed_order: tuple[int, ...]
    pool_label: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported candidate-pool schema_version")
        if not self.pool_label:
            raise ValueError("pool_label must be non-empty")
        if not self.candidates:
            raise ValueError("candidate pool must not be empty")
        if not self.candidate_seed_order:
            raise ValueError("candidate_seed_order must not be empty")
        if len(set(self.candidate_seed_order)) != len(self.candidate_seed_order):
            raise ValueError("candidate_seed_order must contain unique seeds")
        case_ids = [candidate.case_id for candidate in self.candidates]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("candidate case IDs must be unique")
        for candidate in self.candidates:
            candidate_order_key(candidate, self.candidate_seed_order)

    def ordered_candidates(self) -> tuple[CandidateCase, ...]:
        return tuple(
            sorted(
                self.candidates,
                key=lambda candidate: candidate_order_key(
                    candidate, self.candidate_seed_order
                ),
            )
        )

    def to_payload(self) -> dict[str, Any]:
        seed_ranks = {seed: rank for rank, seed in enumerate(self.candidate_seed_order)}
        return {
            "schema_version": self.schema_version,
            "pool_label": self.pool_label,
            "candidate_seed_order": list(self.candidate_seed_order),
            "candidates": [
                candidate.to_payload(seed_rank=seed_ranks[candidate.layout_seed])
                for candidate in self.ordered_candidates()
            ],
        }

    @property
    def sha256(self) -> str:
        return payload_sha256(self.to_payload())


def build_candidate_pool(
    candidates: Iterable[CandidateCase],
    *,
    candidate_seed_order: Iterable[int],
    pool_label: str,
) -> CandidatePool:
    """Build a pool while normalising iterable inputs to immutable tuples."""

    return CandidatePool(
        candidates=tuple(candidates),
        candidate_seed_order=tuple(candidate_seed_order),
        pool_label=pool_label,
    )
