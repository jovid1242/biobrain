"""macOS CPU energy ESTIMATE for the current process — a kernel power-model value, not a measurement.

Source: proc_pid_rusage(RUSAGE_INFO_V6) field ri_energy_nj ("energy estimate", populated by Apple's closed
CPU power controller), CPU cores of this process only (no GPU, memory, display, other processes). Repeat
spread on this machine was 0.6–3.5 % in the owner's probe; windows shorter than ~1 s are not meaningful.
Never report it as measured energy, and never compare it across machines or OS versions.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys

LABEL = "OS CPU energy estimate (macOS kernel model via proc_pid_rusage ri_energy_nj; CPU cores of this process only; NOT a measurement)"

_V6_FIELDS = """ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins
ri_wired_size ri_resident_size ri_phys_footprint ri_proc_start_abstime ri_proc_exit_abstime
ri_child_user_time ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups
ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread ri_diskio_byteswritten
ri_cpu_time_qos_default ri_cpu_time_qos_maintenance ri_cpu_time_qos_background
ri_cpu_time_qos_utility ri_cpu_time_qos_legacy ri_cpu_time_qos_user_initiated
ri_cpu_time_qos_user_interactive ri_billed_system_time ri_serviced_system_time
ri_logical_writes ri_lifetime_max_phys_footprint ri_instructions ri_cycles
ri_billed_energy ri_serviced_energy ri_interval_max_phys_footprint ri_runnable_time
ri_flags ri_user_ptime ri_system_ptime ri_pinstructions ri_pcycles ri_energy_nj
ri_penergy_nj ri_secure_time_in_system ri_secure_ptime_in_system ri_neural_footprint
ri_lifetime_max_neural_footprint ri_interval_max_neural_footprint ri_conclave_footprint
ri_page_wait_time_mach ri_page_cache_hits""".split()


class _RusageInfoV6(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _V6_FIELDS] + [("ri_reserved", ctypes.c_uint64 * 6)]


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


_lib = None
_ns_per_tick = None


def _init() -> bool:
    global _lib, _ns_per_tick
    if _lib is not None:
        return True
    if sys.platform != "darwin" or ctypes.sizeof(_RusageInfoV6) != 464:
        return False
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.proc_pid_rusage.restype = ctypes.c_int
        tb = _Timebase()
        lib.mach_timebase_info(ctypes.byref(tb))
        _ns_per_tick = tb.numer / tb.denom
        _lib = lib
        return True
    except (OSError, AttributeError):
        return False


def snapshot() -> dict | None:
    if not _init():
        return None
    info = _RusageInfoV6()
    if _lib.proc_pid_rusage(os.getpid(), 6, ctypes.byref(info)) != 0:
        return None
    return {"energy_nj": int(info.ri_energy_nj), "p_core_energy_nj": int(info.ri_penergy_nj),
            "cpu_time_ns": (info.ri_user_time + info.ri_system_time) * _ns_per_tick,
            "p_core_cpu_time_ns": (info.ri_user_ptime + info.ri_system_ptime) * _ns_per_tick}


def delta(before: dict | None, after: dict | None, wall_s: float) -> dict | None:
    if before is None or after is None:
        return None
    d = {k: after[k] - before[k] for k in before}
    return {"label": LABEL, "energy_nj": d["energy_nj"], "p_core_energy_nj": d["p_core_energy_nj"],
            "cpu_time_s": d["cpu_time_ns"] / 1e9, "p_core_cpu_time_s": d["p_core_cpu_time_ns"] / 1e9,
            "window_s": wall_s, "reliable_window": wall_s >= 1.0}


def power_state() -> dict:
    """Power source and Low Power Mode (macOS pmset); performance and the estimate depend on both."""
    if sys.platform != "darwin":
        return {}
    try:
        batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        cfg = subprocess.run(["pmset", "-g"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    low = [line.split()[-1] for line in cfg.splitlines() if "lowpowermode" in line]
    source = "battery" if "Battery Power" in batt else "ac" if "AC Power" in batt else "unknown"
    level = re.search(r"(\d+)%", batt)
    return {"power_source": source, "low_power_mode": low[0] == "1" if low else None,
            "battery_percent": int(level.group(1)) if level else None, **thermal_state()}


def thermal_state() -> dict:
    """OS thermal/performance warnings from `pmset -g therm` (notifications only; no temperatures without root)."""
    if sys.platform != "darwin":
        return {}
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    warnings = [line.strip() for line in out.splitlines() if line.strip() and "No " not in line]
    return {"thermal_warnings": warnings}
