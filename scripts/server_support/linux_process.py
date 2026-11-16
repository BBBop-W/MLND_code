"""Linux own-descendant inspection; no global PID scan and no cgroup claims."""
from __future__ import annotations
import ctypes
import errno
import os
import operator
import platform
import signal
import sys

_PY_PIDFD_OPEN = getattr(os, "pidfd_open", None)
_PY_PIDFD_SEND_SIGNAL = getattr(signal, "pidfd_send_signal", None)
_close_fd = os.close


def _platform_details():
    return {"platform": sys.platform, "machine": platform.machine().lower(),
            "kernel_release": platform.release(), "pointer_bytes": ctypes.sizeof(ctypes.c_void_p),
            "long_bytes": ctypes.sizeof(ctypes.c_long), "maxsize": sys.maxsize}


def _wait_capabilities():
    return {"os.waitid": callable(getattr(os, "waitid", None)),
            "os.WNOWAIT": hasattr(os, "WNOWAIT")}


def _integer(value, minimum, maximum, name):
    value = operator.index(value)
    if not minimum <= value <= maximum:
        raise OverflowError(f"{name} is outside its Linux integer range")
    return value


def _load_syscall():
    call = ctypes.CDLL(None, use_errno=True).syscall
    # syscall is variadic: only its fixed syscall-number argument is declared.
    # Every remaining argument is explicitly sized at the call site below.
    call.argtypes = [ctypes.c_long]
    call.restype = ctypes.c_long
    return call


def _syscall(number, operation, *arguments):
    details = _platform_details()
    if (not details["platform"].startswith("linux") or details["machine"] not in ("x86_64", "amd64")
            or details["pointer_bytes"] != 8 or details["long_bytes"] != 8 or details["maxsize"] < 2**32):
        raise RuntimeError("pidfd syscall fallback supports Linux x86_64 LP64 only; x32/other architectures are refused")
    call = _load_syscall()
    ctypes.set_errno(0)
    result = call(ctypes.c_long(number), *arguments)
    if result == -1:
        code = ctypes.get_errno() or errno.EIO
        raise OSError(code, f"{operation}: {os.strerror(code)}")
    return int(result)


def pidfd_open(pid, flags=0):
    """Use a real kernel pidfd even if Python was built with older headers."""
    pid = _integer(pid, -(2**31), 2**31-1, "pid")
    flags = _integer(flags, 0, 2**32-1, "flags")
    if _PY_PIDFD_OPEN is not None:
        return _PY_PIDFD_OPEN(pid, flags)
    # Linux 5.15 arch/x86/entry/syscalls/syscall_64.tbl: pidfd_open = 434.
    return _syscall(434, "pidfd_open", ctypes.c_long(pid), ctypes.c_ulong(flags))


def pidfd_send_signal(descriptor, signal_number, siginfo=None, flags=0):
    if siginfo is not None:
        raise TypeError("siginfo must be None")
    descriptor = _integer(descriptor, -(2**31), 2**31-1, "pidfd")
    signal_number = _integer(signal_number, -(2**31), 2**31-1, "signal")
    flags = _integer(flags, 0, 2**32-1, "flags")
    if _PY_PIDFD_SEND_SIGNAL is not None:
        return _PY_PIDFD_SEND_SIGNAL(descriptor, signal_number, None, flags)
    # NULL is a pointer-sized siginfo_t*, not an integer argument.
    _syscall(424, "pidfd_send_signal", ctypes.c_long(descriptor), ctypes.c_long(signal_number),
             ctypes.c_void_p(None), ctypes.c_ulong(flags))


def require_linux():
    details = _platform_details()
    if not details["platform"].startswith("linux"):
        raise RuntimeError("Actual execution requires Linux; --verify/--dry-run and portable tests work elsewhere")
    if details["maxsize"] < 2**32 or details["pointer_bytes"] != 8:
        raise RuntimeError("Actual execution requires 64-bit Python for the 3/6/10/16/24 GiB address-space limits")
    waits = _wait_capabilities()
    missing = [name for name, present in waits.items() if not present]
    if missing:
        raise RuntimeError("Required unreaped-child observation interface missing: " + ", ".join(missing))
    report = dict(details, wait_interfaces=waits,
                  pidfd_open_backend="python" if _PY_PIDFD_OPEN is not None else "ctypes_syscall_x86_64",
                  pidfd_send_signal_backend="python" if _PY_PIDFD_SEND_SIGNAL is not None else "ctypes_syscall_x86_64")
    descriptor = None
    try:
        descriptor = pidfd_open(os.getpid())
        pidfd_send_signal(descriptor, 0)
        report["pidfd_self_probe"] = "passed"
    except OSError as error:
        # Preserve errno/cause. A real ENOSYS/EPERM is not an absent Python
        # binding and must not fall back to PID-based signaling.
        raise RuntimeError(f"pidfd runtime probe failed: errno={error.errno}; "
                           f"open={report['pidfd_open_backend']}, signal={report['pidfd_send_signal_backend']}; {error}") from error
    finally:
        if descriptor is not None:
            _close_fd(descriptor)
    return report


def become_subreaper():
    require_linux()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)


def read_at(directory_fd, name):
    descriptor = os.open(name, os.O_RDONLY, dir_fd=directory_fd)
    with os.fdopen(descriptor, "r", encoding="ascii") as stream:
        return stream.read()


def process_info(pid):
    """Read stat/children through a bound proc directory, resistant to PID reuse."""
    descriptor = os.open(f"/proc/{int(pid)}", os.O_RDONLY | os.O_DIRECTORY)
    try:
        fields = read_at(descriptor, "stat").rsplit(")", 1)[1].split()
        children = set()
        tasks_fd = os.open("task", os.O_RDONLY | os.O_DIRECTORY, dir_fd=descriptor)
        try:
            for tid in os.listdir(tasks_fd):
                try:
                    children.update(int(value) for value in read_at(tasks_fd, f"{tid}/children").split())
                except FileNotFoundError:
                    if os.path.exists(f"/proc/{int(pid)}/task/{tid}"):
                        raise RuntimeError("A live owned thread has no readable proc children interface")
        finally:
            os.close(tasks_fd)
        return {"pid": int(pid), "state": fields[0], "ppid": int(fields[1]), "pgid": int(fields[2]),
                "sid": int(fields[3]), "start_ticks": int(fields[19]), "virtual_bytes": int(fields[20]),
                "rss_bytes": int(fields[21]) * os.sysconf("SC_PAGE_SIZE"), "children": sorted(children)}
    finally:
        os.close(descriptor)


def same_process(pid, start_ticks):
    try:
        info = process_info(pid)
        return info["start_ticks"] == start_ticks and info["state"] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        return False


def descendants(guardian_pid, expected_pgid):
    root = process_info(guardian_pid)
    queue, visited, members, escaped = list(root["children"]), set(), [], []
    while queue:
        pid = queue.pop()
        if pid in visited:
            continue
        visited.add(pid)
        try:
            info = process_info(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        queue.extend(info["children"])
        if info["state"] == "Z":
            continue
        (members if info["pgid"] == expected_pgid and info["sid"] == expected_pgid else escaped).append(info)
    return members, escaped


def reap_adopted():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
