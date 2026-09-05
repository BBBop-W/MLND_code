from __future__ import annotations

import re
from dataclasses import replace

import pytest

from src.benchmark.branch_cut.formal_environment import (
    ABSENT_VERSION,
    FormalEnvironmentCapture,
    FormalEnvironmentError,
    GurobiVersions,
    ModuleDistributionVersions,
    OperatingSystemIdentity,
    PythonIdentity,
    canonical_environment_json,
    capture_formal_environment,
    collect_formal_environment_contract,
    collect_session_snapshot,
    hash_stable_host_identifier,
    normalize_cpu_model,
)
from src.benchmark.branch_cut.hashing import canonical_json, canonical_sha256


class _FakeProbe:
    def __init__(
        self,
        *,
        host_identifier: tuple[str, str] | None = (
            "macos-ioplatformuuid",
            "ABCDEF01-2345-6789-ABCD-EF0123456789",
        ),
        pid: int = 1200,
        available_memory_bytes: int = 16_000_000_000,
        load_average: tuple[float, ...] = (0.5, 0.75, 1.0),
        missing_versions: frozenset[tuple[str, str]] = frozenset(),
    ) -> None:
        self._host_identifier = host_identifier
        self._pid = pid
        self._available_memory_bytes = available_memory_bytes
        self._load_average = load_average
        self._missing_versions = missing_versions

    def operating_system(self) -> tuple[str, str, str, str]:
        return "Darwin", "15.6.1", "24.6.0", "Darwin Kernel Version 24.6.0"

    def machine(self) -> str:
        return "arm64"

    def pointer_bits(self) -> int:
        return 64

    def byteorder(self) -> str:
        return "little"

    def cpu_model(self) -> str:
        return "  Apple\tM4   MAX  "

    def physical_cores(self) -> int:
        return 16

    def logical_cores(self) -> int:
        return 16

    def installed_ram_bytes(self) -> int:
        return 68_719_476_736

    def python_identity(self) -> tuple[str, str]:
        return "CPython", "3.13.5"

    def module_version(self, module_name: str) -> str | None:
        if ("module", module_name) in self._missing_versions:
            return None
        return {
            "gurobipy": "13.0.1",
            "numpy": "2.3.2",
            "scipy": "1.16.1",
        }[module_name]

    def distribution_version(self, distribution_name: str) -> str | None:
        if ("distribution", distribution_name) in self._missing_versions:
            return None
        return {
            "gurobipy": "13.0.1",
            "numpy": "2.3.2",
            "scipy": "1.16.1",
        }[distribution_name]

    def gurobi_native_version(self) -> tuple[int, ...] | None:
        if ("native", "gurobipy") in self._missing_versions:
            return None
        return 13, 0, 1

    def stable_host_identifier(self) -> tuple[str, str] | None:
        return self._host_identifier

    def pid(self) -> int:
        return self._pid

    def executable(self) -> str:
        return "/opt/python/bin/python3"

    def cwd(self) -> str:
        return "/work/formal-benchmark"

    def affinity_cpu_ids(self) -> tuple[int, ...]:
        return 3, 1, 1, 2

    def blas_environment(self) -> tuple[tuple[str, str], ...]:
        return (
            ("OPENBLAS_NUM_THREADS", "1"),
            ("OMP_NUM_THREADS", "1"),
            ("MKL_NUM_THREADS", "unset"),
        )

    def available_memory_bytes(self) -> int:
        return self._available_memory_bytes

    def load_average(self) -> tuple[float, ...]:
        return self._load_average


def test_stable_contract_contains_complete_normalized_identity() -> None:
    contract = collect_formal_environment_contract(_FakeProbe())
    payload = contract.to_payload()

    assert payload == {
        "byteorder": "little",
        "cpu_model": "apple m4 max",
        "gurobipy": {
            "distribution_version": "13.0.1",
            "module_version": "13.0.1",
            "native_version": "13.0.1",
        },
        "installed_ram_bytes": 68_719_476_736,
        "logical_cores": 16,
        "machine": "arm64",
        "numpy": {
            "distribution_version": "2.3.2",
            "module_version": "2.3.2",
        },
        "operating_system": {
            "formal_version": "15.6.1",
            "kernel_release": "24.6.0",
            "kernel_version": "Darwin Kernel Version 24.6.0",
            "name": "Darwin",
        },
        "physical_cores": 16,
        "pointer_bits": 64,
        "python": {"implementation": "CPython", "version": "3.13.5"},
        "schema_version": 1,
        "scipy": {
            "distribution_version": "1.16.1",
            "module_version": "1.16.1",
        },
    }
    assert re.fullmatch(r"[0-9a-f]{64}", contract.sha256)
    assert contract.environment_id == f"formal-environment-{contract.sha256}"
    assert canonical_environment_json(contract) == canonical_json(payload)
    assert contract.sha256 == canonical_sha256(
        {
            "contract": payload,
            "domain": "mlnd-branch-cut-formal-environment-v1",
        }
    )


def test_host_identifier_is_domain_separated_and_raw_value_is_never_stored() -> None:
    raw_identifier = "ABCDEF01-2345-6789-ABCD-EF0123456789"
    contract = collect_formal_environment_contract(_FakeProbe())
    capture = capture_formal_environment(_FakeProbe())

    assert "host_identity" not in canonical_json(contract.to_payload())
    assert raw_identifier not in canonical_json(contract.to_payload())
    assert raw_identifier not in canonical_json(capture.to_payload())
    assert capture.host_identity_sha256 == hash_stable_host_identifier(
        "macos-ioplatformuuid",
        raw_identifier,
    )
    assert capture.to_payload()["host_binding"] == {
        "host_identity_sha256": capture.host_identity_sha256,
        "included_in_contract_hash": False,
    }
    assert hash_stable_host_identifier("source-a", raw_identifier) != (
        hash_stable_host_identifier("source-b", raw_identifier)
    )
    assert hash_stable_host_identifier("source-a", raw_identifier) != (
        canonical_sha256(raw_identifier)
    )


def test_missing_host_identifier_allows_contract_but_formal_capture_fails_closed() -> None:
    contract = collect_formal_environment_contract(_FakeProbe(host_identifier=None))
    assert "host_identity" not in canonical_json(contract.to_payload())

    with pytest.raises(FormalEnvironmentError, match="host identifier is unavailable"):
        capture_formal_environment(_FakeProbe(host_identifier=None))


def test_missing_dependency_versions_are_explicit_absent() -> None:
    missing = frozenset(
        {
            ("module", "numpy"),
            ("distribution", "scipy"),
            ("module", "gurobipy"),
            ("distribution", "gurobipy"),
            ("native", "gurobipy"),
        }
    )
    contract = collect_formal_environment_contract(
        _FakeProbe(missing_versions=missing)
    )

    assert contract.numpy.module_version == ABSENT_VERSION
    assert contract.numpy.distribution_version == "2.3.2"
    assert contract.scipy.module_version == "1.16.1"
    assert contract.scipy.distribution_version == ABSENT_VERSION
    assert contract.gurobipy == GurobiVersions(
        module_version=ABSENT_VERSION,
        distribution_version=ABSENT_VERSION,
        native_version=ABSENT_VERSION,
    )


def test_volatile_snapshot_is_explicitly_excluded_from_stable_hash() -> None:
    first = capture_formal_environment(
        _FakeProbe(
            pid=1200,
            available_memory_bytes=16_000_000_000,
            load_average=(0.5, 0.75, 1.0),
        )
    )
    second = capture_formal_environment(
        _FakeProbe(
            pid=9800,
            available_memory_bytes=2_000_000_000,
            load_average=(8.0, 6.0, 4.0),
        )
    )

    assert first.session != second.session
    assert first.contract == second.contract
    assert first.stable_environment_sha256 == second.stable_environment_sha256
    assert first.to_payload()["session_snapshot_in_contract_hash"] is False
    assert first.to_payload()["session_snapshot"] != second.to_payload()[
        "session_snapshot"
    ]


def test_same_specification_on_different_hosts_shares_contract_hash() -> None:
    first = capture_formal_environment(
        _FakeProbe(host_identifier=("linux-machine-id", "host-alpha"))
    )
    second = capture_formal_environment(
        _FakeProbe(host_identifier=("linux-machine-id", "host-beta"))
    )

    assert first.contract == second.contract
    assert first.stable_environment_sha256 == second.stable_environment_sha256
    assert first.host_identity_sha256 != second.host_identity_sha256


def test_session_snapshot_normalizes_affinity_and_blas_entries() -> None:
    snapshot = collect_session_snapshot(_FakeProbe())

    assert snapshot.affinity_cpu_ids == (1, 2, 3)
    assert snapshot.blas_environment == (
        ("MKL_NUM_THREADS", "unset"),
        ("OMP_NUM_THREADS", "1"),
        ("OPENBLAS_NUM_THREADS", "1"),
    )
    assert snapshot.to_payload()["volatile"] is True


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        (
            "operating_system",
            OperatingSystemIdentity(
                "Darwin",
                "15.6.2",
                "24.6.0",
                "Darwin Kernel Version 24.6.0",
            ),
        ),
        ("machine", "x86_64"),
        ("pointer_bits", 32),
        ("byteorder", "big"),
        ("cpu_model", "apple m3 max"),
        ("physical_cores", 15),
        ("logical_cores", 20),
        ("installed_ram_bytes", 34_359_738_368),
        ("python", PythonIdentity("CPython", "3.13.6")),
        (
            "numpy",
            ModuleDistributionVersions("2.3.3", "2.3.3"),
        ),
        (
            "scipy",
            ModuleDistributionVersions("1.16.2", "1.16.2"),
        ),
        (
            "gurobipy",
            GurobiVersions("13.0.2", "13.0.2", "13.0.2"),
        ),
    ),
)
def test_every_stable_identity_family_changes_contract_hash(
    field_name: str,
    changed_value: object,
) -> None:
    contract = collect_formal_environment_contract(_FakeProbe())
    changed = replace(contract, **{field_name: changed_value})

    assert changed.sha256 != contract.sha256


def test_cpu_normalization_is_deterministic() -> None:
    assert normalize_cpu_model(" Apple\tM4   MAX ") == "apple m4 max"
    assert normalize_cpu_model("APPLE M4 MAX") == "apple m4 max"


def test_capture_type_cannot_mix_session_into_contract_identity() -> None:
    capture = capture_formal_environment(_FakeProbe())

    assert isinstance(capture, FormalEnvironmentCapture)
    assert "session" not in capture.contract.to_payload()
    assert "pid" not in canonical_environment_json(capture.contract)
