"""Stable formal-environment identity and non-identifying session telemetry.

The stable contract is suitable for pre-registration and exact resume checks.
Volatile process state is captured separately and is deliberately excluded
from :attr:`FormalEnvironmentContract.sha256`.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import re
import struct
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .hashing import canonical_json, canonical_sha256


FORMAL_ENVIRONMENT_SCHEMA_VERSION = 1
FORMAL_ENVIRONMENT_HASH_DOMAIN = "mlnd-branch-cut-formal-environment-v1"
FORMAL_HOST_ID_HASH_DOMAIN = "mlnd-branch-cut-formal-host-id-v1"
ABSENT_VERSION = "absent"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BLAS_ENVIRONMENT_NAMES = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class FormalEnvironmentError(RuntimeError):
    """The formal environment cannot be observed without ambiguity."""


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FormalEnvironmentError(f"{name} must be a nonempty string")
    return value.strip()


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise FormalEnvironmentError(f"{name} must be a positive integer")
    return value


def normalize_cpu_model(value: str) -> str:
    """Normalize harmless formatting and case differences in CPU model text."""

    text = unicodedata.normalize("NFKC", _require_text(value, "cpu_model"))
    return " ".join(text.replace("\x00", " ").split()).casefold()


def hash_stable_host_identifier(source: str, raw_identifier: str) -> str:
    """Hash a host identifier without retaining its sensitive raw value."""

    normalized_source = _require_text(source, "host identifier source").casefold()
    normalized_identifier = unicodedata.normalize(
        "NFKC",
        _require_text(raw_identifier, "host identifier"),
    ).strip().casefold()
    return canonical_sha256(
        {
            "domain": FORMAL_HOST_ID_HASH_DOMAIN,
            "source": normalized_source,
            "value": normalized_identifier,
        }
    )


@dataclass(frozen=True, slots=True)
class OperatingSystemIdentity:
    name: str
    formal_version: str
    kernel_release: str
    kernel_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "formal_version",
            "kernel_release",
            "kernel_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, slots=True)
class ModuleDistributionVersions:
    module_version: str
    distribution_version: str

    def __post_init__(self) -> None:
        for field_name in ("module_version", "distribution_version"):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, slots=True)
class GurobiVersions:
    module_version: str
    distribution_version: str
    native_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "module_version",
            "distribution_version",
            "native_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_text(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True, slots=True)
class PythonIdentity:
    implementation: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "implementation",
            _require_text(self.implementation, "python implementation"),
        )
        object.__setattr__(
            self,
            "version",
            _require_text(self.version, "python version"),
        )


@dataclass(frozen=True, slots=True)
class FormalEnvironmentContract:
    """Shareable hardware, runtime, and numerical-stack specification."""

    operating_system: OperatingSystemIdentity
    machine: str
    pointer_bits: int
    byteorder: str
    cpu_model: str
    physical_cores: int
    logical_cores: int
    installed_ram_bytes: int
    python: PythonIdentity
    numpy: ModuleDistributionVersions
    scipy: ModuleDistributionVersions
    gurobipy: GurobiVersions
    schema_version: int = FORMAL_ENVIRONMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.operating_system, OperatingSystemIdentity):
            raise FormalEnvironmentError("operating_system identity is invalid")
        object.__setattr__(self, "machine", _require_text(self.machine, "machine"))
        if self.pointer_bits not in {32, 64, 128}:
            raise FormalEnvironmentError("pointer_bits must be 32, 64, or 128")
        if self.byteorder not in {"little", "big"}:
            raise FormalEnvironmentError("byteorder must be little or big")
        object.__setattr__(self, "cpu_model", normalize_cpu_model(self.cpu_model))
        for field_name in (
            "physical_cores",
            "logical_cores",
            "installed_ram_bytes",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_positive_integer(getattr(self, field_name), field_name),
            )
        if self.logical_cores < self.physical_cores:
            raise FormalEnvironmentError("logical_cores cannot be below physical_cores")
        if not isinstance(self.python, PythonIdentity):
            raise FormalEnvironmentError("python identity is invalid")
        if not isinstance(self.numpy, ModuleDistributionVersions):
            raise FormalEnvironmentError("numpy version identity is invalid")
        if not isinstance(self.scipy, ModuleDistributionVersions):
            raise FormalEnvironmentError("scipy version identity is invalid")
        if not isinstance(self.gurobipy, GurobiVersions):
            raise FormalEnvironmentError("gurobipy version identity is invalid")
        if self.schema_version != FORMAL_ENVIRONMENT_SCHEMA_VERSION:
            raise FormalEnvironmentError("unsupported formal environment schema")

    def to_payload(self) -> dict[str, Any]:
        return {
            "byteorder": self.byteorder,
            "cpu_model": self.cpu_model,
            "gurobipy": {
                "distribution_version": self.gurobipy.distribution_version,
                "module_version": self.gurobipy.module_version,
                "native_version": self.gurobipy.native_version,
            },
            "installed_ram_bytes": self.installed_ram_bytes,
            "logical_cores": self.logical_cores,
            "machine": self.machine,
            "numpy": {
                "distribution_version": self.numpy.distribution_version,
                "module_version": self.numpy.module_version,
            },
            "operating_system": {
                "formal_version": self.operating_system.formal_version,
                "kernel_release": self.operating_system.kernel_release,
                "kernel_version": self.operating_system.kernel_version,
                "name": self.operating_system.name,
            },
            "physical_cores": self.physical_cores,
            "pointer_bits": self.pointer_bits,
            "python": {
                "implementation": self.python.implementation,
                "version": self.python.version,
            },
            "schema_version": self.schema_version,
            "scipy": {
                "distribution_version": self.scipy.distribution_version,
                "module_version": self.scipy.module_version,
            },
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(
            {
                "contract": self.to_payload(),
                "domain": FORMAL_ENVIRONMENT_HASH_DOMAIN,
            }
        )

    @property
    def environment_id(self) -> str:
        return f"formal-environment-{self.sha256}"


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Volatile diagnostics intentionally excluded from the contract hash."""

    pid: int
    executable: str
    cwd: str
    affinity_cpu_ids: tuple[int, ...]
    blas_environment: tuple[tuple[str, str], ...]
    available_memory_bytes: int | None
    load_average: tuple[float, ...]

    def __post_init__(self) -> None:
        _require_positive_integer(self.pid, "pid")
        _require_text(self.executable, "executable")
        _require_text(self.cwd, "cwd")
        if any(
            isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 0
            for cpu in self.affinity_cpu_ids
        ):
            raise FormalEnvironmentError("affinity CPU IDs must be nonnegative integers")
        if tuple(sorted(set(self.affinity_cpu_ids))) != self.affinity_cpu_ids:
            raise FormalEnvironmentError("affinity CPU IDs must be sorted and unique")
        if tuple(sorted(self.blas_environment)) != self.blas_environment:
            raise FormalEnvironmentError("BLAS environment entries must be sorted")
        if self.available_memory_bytes is not None:
            _require_positive_integer(
                self.available_memory_bytes,
                "available_memory_bytes",
            )
        if len(self.load_average) not in {0, 3}:
            raise FormalEnvironmentError("load_average must be empty or contain 3 values")

    def to_payload(self) -> dict[str, Any]:
        return {
            "affinity_cpu_ids": list(self.affinity_cpu_ids),
            "available_memory_bytes": self.available_memory_bytes,
            "blas_environment": dict(self.blas_environment),
            "cwd": self.cwd,
            "executable": self.executable,
            "load_average": list(self.load_average),
            "pid": self.pid,
            "volatile": True,
        }


@dataclass(frozen=True, slots=True)
class FormalEnvironmentCapture:
    """Shareable contract, host binding, and a non-hashed session snapshot."""

    contract: FormalEnvironmentContract
    host_identity_sha256: str
    session: SessionSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.contract, FormalEnvironmentContract):
            raise FormalEnvironmentError("capture contract is invalid")
        if (
            not isinstance(self.host_identity_sha256, str)
            or _SHA256_RE.fullmatch(self.host_identity_sha256) is None
        ):
            raise FormalEnvironmentError(
                "host_identity_sha256 must be a lowercase SHA256 digest"
            )
        if not isinstance(self.session, SessionSnapshot):
            raise FormalEnvironmentError("capture session snapshot is invalid")

    @property
    def stable_environment_sha256(self) -> str:
        return self.contract.sha256

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract": self.contract.to_payload(),
            "contract_sha256": self.contract.sha256,
            "host_binding": {
                "host_identity_sha256": self.host_identity_sha256,
                "included_in_contract_hash": False,
            },
            "session_snapshot": self.session.to_payload(),
            "session_snapshot_in_contract_hash": False,
        }


class EnvironmentProbe(Protocol):
    """Injectable observation boundary used by formal-environment collection."""

    def operating_system(self) -> tuple[str, str, str, str]: ...

    def machine(self) -> str: ...

    def pointer_bits(self) -> int: ...

    def byteorder(self) -> str: ...

    def cpu_model(self) -> str: ...

    def physical_cores(self) -> int | None: ...

    def logical_cores(self) -> int | None: ...

    def installed_ram_bytes(self) -> int | None: ...

    def python_identity(self) -> tuple[str, str]: ...

    def module_version(self, module_name: str) -> str | None: ...

    def distribution_version(self, distribution_name: str) -> str | None: ...

    def gurobi_native_version(self) -> str | tuple[int, ...] | None: ...

    def stable_host_identifier(self) -> tuple[str, str] | None: ...

    def pid(self) -> int: ...

    def executable(self) -> str: ...

    def cwd(self) -> str: ...

    def affinity_cpu_ids(self) -> tuple[int, ...]: ...

    def blas_environment(self) -> tuple[tuple[str, str], ...]: ...

    def available_memory_bytes(self) -> int | None: ...

    def load_average(self) -> tuple[float, ...]: ...


def _version_or_absent(value: str | None) -> str:
    return ABSENT_VERSION if value is None else _require_text(value, "version")


def _native_version_or_absent(value: str | tuple[int, ...] | None) -> str:
    if value is None:
        return ABSENT_VERSION
    if isinstance(value, tuple):
        if not value or any(isinstance(part, bool) or not isinstance(part, int) for part in value):
            raise FormalEnvironmentError("native Gurobi version tuple is invalid")
        return ".".join(str(part) for part in value)
    return _require_text(value, "native Gurobi version")


def collect_formal_environment_contract(
    probe: EnvironmentProbe | None = None,
) -> FormalEnvironmentContract:
    """Collect the shareable stable environment specification."""

    observer = probe or SystemEnvironmentProbe()
    os_values = observer.operating_system()
    if not isinstance(os_values, tuple) or len(os_values) != 4:
        raise FormalEnvironmentError("operating system probe is invalid")
    physical_cores = observer.physical_cores()
    logical_cores = observer.logical_cores()
    installed_ram_bytes = observer.installed_ram_bytes()
    python_values = observer.python_identity()
    if not isinstance(python_values, tuple) or len(python_values) != 2:
        raise FormalEnvironmentError("Python identity probe is invalid")
    return FormalEnvironmentContract(
        operating_system=OperatingSystemIdentity(*os_values),
        machine=observer.machine(),
        pointer_bits=observer.pointer_bits(),
        byteorder=observer.byteorder(),
        cpu_model=observer.cpu_model(),
        physical_cores=_require_positive_integer(physical_cores, "physical_cores"),
        logical_cores=_require_positive_integer(logical_cores, "logical_cores"),
        installed_ram_bytes=_require_positive_integer(
            installed_ram_bytes,
            "installed_ram_bytes",
        ),
        python=PythonIdentity(*python_values),
        numpy=ModuleDistributionVersions(
            module_version=_version_or_absent(observer.module_version("numpy")),
            distribution_version=_version_or_absent(
                observer.distribution_version("numpy")
            ),
        ),
        scipy=ModuleDistributionVersions(
            module_version=_version_or_absent(observer.module_version("scipy")),
            distribution_version=_version_or_absent(
                observer.distribution_version("scipy")
            ),
        ),
        gurobipy=GurobiVersions(
            module_version=_version_or_absent(observer.module_version("gurobipy")),
            distribution_version=_version_or_absent(
                observer.distribution_version("gurobipy")
            ),
            native_version=_native_version_or_absent(
                observer.gurobi_native_version()
            ),
        ),
    )


def collect_session_snapshot(
    probe: EnvironmentProbe | None = None,
) -> SessionSnapshot:
    """Collect volatile diagnostics without adding them to the stable hash."""

    observer = probe or SystemEnvironmentProbe()
    return SessionSnapshot(
        pid=observer.pid(),
        executable=observer.executable(),
        cwd=observer.cwd(),
        affinity_cpu_ids=tuple(sorted(set(observer.affinity_cpu_ids()))),
        blas_environment=tuple(sorted(observer.blas_environment())),
        available_memory_bytes=observer.available_memory_bytes(),
        load_average=tuple(observer.load_average()),
    )


def capture_formal_environment(
    probe: EnvironmentProbe | None = None,
) -> FormalEnvironmentCapture:
    """Capture a contract plus host binding, failing closed without the latter."""

    observer = probe or SystemEnvironmentProbe()
    host_identifier = observer.stable_host_identifier()
    if host_identifier is None:
        raise FormalEnvironmentError(
            "stable host identifier is unavailable in formal mode"
        )
    if not isinstance(host_identifier, tuple) or len(host_identifier) != 2:
        raise FormalEnvironmentError("stable host identifier probe is invalid")
    return FormalEnvironmentCapture(
        contract=collect_formal_environment_contract(observer),
        host_identity_sha256=hash_stable_host_identifier(*host_identifier),
        session=collect_session_snapshot(observer),
    )


class SystemEnvironmentProbe:
    """Read-only platform probe; it never creates a Gurobi environment."""

    def operating_system(self) -> tuple[str, str, str, str]:
        system = platform.system()
        if system == "Darwin":
            formal_version = platform.mac_ver()[0]
        elif system == "Linux":
            try:
                release = platform.freedesktop_os_release()
            except OSError:
                release = {}
            distribution = release.get("ID", "linux")
            version = release.get("VERSION_ID", "")
            formal_version = f"{distribution}-{version}".rstrip("-")
        elif system == "Windows":
            formal_version = platform.win32_ver()[0] or platform.version()
        else:
            formal_version = platform.version()
        return (
            system,
            formal_version,
            platform.release(),
            platform.version(),
        )

    def machine(self) -> str:
        return platform.machine()

    def pointer_bits(self) -> int:
        return struct.calcsize("P") * 8

    def byteorder(self) -> str:
        return sys.byteorder

    @staticmethod
    def _sysctl(name: str) -> str | None:
        try:
            completed = subprocess.run(
                ("sysctl", "-n", name),
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        value = completed.stdout.strip()
        return value or None

    def cpu_model(self) -> str:
        system = platform.system()
        if system == "Darwin":
            value = self._sysctl("machdep.cpu.brand_string") or self._sysctl(
                "hw.model"
            )
            if value:
                return value
        elif system == "Linux":
            try:
                lines = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                lines = []
            for key in ("model name", "hardware", "processor"):
                for line in lines:
                    label, separator, value = line.partition(":")
                    if separator and label.strip().casefold() == key:
                        if value.strip():
                            return value.strip()
        value = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
        return value or platform.machine()

    def physical_cores(self) -> int | None:
        try:
            psutil = importlib.import_module("psutil")
            value = psutil.cpu_count(logical=False)
            if value:
                return int(value)
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        if platform.system() == "Darwin":
            value = self._sysctl("hw.physicalcpu")
            return int(value) if value and value.isdigit() else None
        if platform.system() == "Linux":
            try:
                text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                return None
            packages: set[tuple[str, str]] = set()
            for block in text.split("\n\n"):
                fields = dict(
                    line.split(":", 1)
                    for line in block.splitlines()
                    if ":" in line
                )
                physical = fields.get("physical id", "").strip()
                core = fields.get("core id", "").strip()
                if physical and core:
                    packages.add((physical, core))
            return len(packages) or None
        return None

    def logical_cores(self) -> int | None:
        return os.cpu_count()

    def installed_ram_bytes(self) -> int | None:
        try:
            psutil = importlib.import_module("psutil")
            return int(psutil.virtual_memory().total)
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        if platform.system() == "Darwin":
            value = self._sysctl("hw.memsize")
            return int(value) if value and value.isdigit() else None
        try:
            return int(os.sysconf("SC_PHYS_PAGES")) * int(
                os.sysconf("SC_PAGE_SIZE")
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    def python_identity(self) -> tuple[str, str]:
        return platform.python_implementation(), platform.python_version()

    def module_version(self, module_name: str) -> str | None:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name == module_name:
                return None
            raise FormalEnvironmentError(
                f"cannot import {module_name} for version inspection: {error}"
            ) from error
        except ImportError as error:
            raise FormalEnvironmentError(
                f"cannot import {module_name} for version inspection: {error}"
            ) from error
        version = getattr(module, "__version__", None)
        return str(version) if version is not None else None

    def distribution_version(self, distribution_name: str) -> str | None:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def gurobi_native_version(self) -> str | tuple[int, ...] | None:
        try:
            module = importlib.import_module("gurobipy")
        except ModuleNotFoundError as error:
            if error.name == "gurobipy":
                return None
            raise FormalEnvironmentError(
                f"cannot import gurobipy for native version inspection: {error}"
            ) from error
        except ImportError as error:
            raise FormalEnvironmentError(
                f"cannot import gurobipy for native version inspection: {error}"
            ) from error
        gurobi = getattr(module, "gurobi", None)
        version = getattr(gurobi, "version", None)
        return version() if callable(version) else None

    def stable_host_identifier(self) -> tuple[str, str] | None:
        system = platform.system()
        if system == "Darwin":
            try:
                completed = subprocess.run(
                    ("ioreg", "-rd1", "-c", "IOPlatformExpertDevice"),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            match = re.search(
                r'"IOPlatformUUID"\s*=\s*"([^"]+)"',
                completed.stdout,
            )
            return ("macos-ioplatformuuid", match.group(1)) if match else None
        if system == "Linux":
            for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
                try:
                    value = path.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeError):
                    continue
                if value:
                    return "linux-machine-id", value
            return None
        if system == "Windows":
            try:
                import winreg

                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Cryptography",
                ) as key:
                    value, _kind = winreg.QueryValueEx(key, "MachineGuid")
            except (ImportError, OSError):
                return None
            return "windows-machineguid", str(value)
        return None

    def pid(self) -> int:
        return os.getpid()

    def executable(self) -> str:
        return sys.executable or "unavailable"

    def cwd(self) -> str:
        try:
            return Path.cwd().as_posix()
        except OSError:
            return "unavailable"

    def affinity_cpu_ids(self) -> tuple[int, ...]:
        get_affinity = getattr(os, "sched_getaffinity", None)
        if callable(get_affinity):
            try:
                return tuple(sorted(int(cpu) for cpu in get_affinity(0)))
            except (OSError, TypeError, ValueError):
                pass
        try:
            psutil = importlib.import_module("psutil")
            return tuple(sorted(int(cpu) for cpu in psutil.Process().cpu_affinity()))
        except (ImportError, AttributeError, OSError, TypeError, ValueError):
            return ()

    def blas_environment(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, os.environ.get(name, "unset"))
            for name in _BLAS_ENVIRONMENT_NAMES
        )

    def available_memory_bytes(self) -> int | None:
        try:
            psutil = importlib.import_module("psutil")
            return int(psutil.virtual_memory().available)
        except (ImportError, AttributeError, TypeError, ValueError):
            pass
        try:
            return int(os.sysconf("SC_AVPHYS_PAGES")) * int(
                os.sysconf("SC_PAGE_SIZE")
            )
        except (AttributeError, OSError, TypeError, ValueError):
            return None

    def load_average(self) -> tuple[float, ...]:
        try:
            return tuple(float(value) for value in os.getloadavg())
        except (AttributeError, OSError):
            return ()


def canonical_environment_json(contract: FormalEnvironmentContract) -> str:
    """Return the canonical JSON representation of the stable contract."""

    if not isinstance(contract, FormalEnvironmentContract):
        raise TypeError("contract must be a FormalEnvironmentContract")
    return canonical_json(contract.to_payload())


__all__ = [
    "ABSENT_VERSION",
    "EnvironmentProbe",
    "FormalEnvironmentCapture",
    "FormalEnvironmentContract",
    "FormalEnvironmentError",
    "GurobiVersions",
    "ModuleDistributionVersions",
    "OperatingSystemIdentity",
    "PythonIdentity",
    "SessionSnapshot",
    "SystemEnvironmentProbe",
    "canonical_environment_json",
    "capture_formal_environment",
    "collect_formal_environment_contract",
    "collect_session_snapshot",
    "hash_stable_host_identifier",
    "normalize_cpu_model",
]
