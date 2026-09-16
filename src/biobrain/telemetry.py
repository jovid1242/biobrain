"""Memory telemetry and the project-wide memory budget (`--memory-budget`, default 8GB).

Sizes use binary units: "8GB" means 8 GiB. The budget is checked *before* large allocations,
so an over-budget step fails with an explanation (or warns, when not strict) instead of
swapping or being killed.
"""

from __future__ import annotations

import os
import re
import resource
import sys
import time
from dataclasses import dataclass, field

import psutil

DEFAULT_BUDGET = "8GB"
_UNITS = {"": 1, "B": 1, "K": 1 << 10, "KB": 1 << 10, "KIB": 1 << 10, "M": 1 << 20, "MB": 1 << 20,
          "MIB": 1 << 20, "G": 1 << 30, "GB": 1 << 30, "GIB": 1 << 30, "T": 1 << 40, "TB": 1 << 40, "TIB": 1 << 40}


def parse_bytes(text: str | int) -> int:
    if isinstance(text, int):
        return text
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*", text)
    if not m or m.group(2).upper() not in _UNITS:
        raise ValueError(f"cannot parse a size from {text!r} (examples: 512MB, 8GB)")
    return int(float(m.group(1)) * _UNITS[m.group(2).upper()])


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


def peak_rss_bytes() -> int:
    """Lifetime peak RSS of this process (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


class MemoryBudgetExceeded(MemoryError):
    pass


@dataclass
class MemoryBudget:
    limit: int
    strict: bool = True
    events: list[dict] = field(default_factory=list)

    @classmethod
    def from_env(cls, override: str | None = None, strict: bool = True) -> "MemoryBudget":
        return cls(parse_bytes(override or os.environ.get("BIOBRAIN_MEMORY_BUDGET", DEFAULT_BUDGET)), strict)

    def check(self, what: str, estimate: int) -> bool:
        """Would current RSS + `estimate` exceed the budget? Raise (strict) or warn."""
        rss = rss_bytes()
        ok = rss + estimate <= self.limit
        self.events.append({"t": time.time(), "what": what, "estimate": int(estimate), "rss": rss, "ok": ok})
        if not ok:
            msg = (f"memory budget: '{what}' needs ~{fmt_bytes(estimate)} on top of RSS {fmt_bytes(rss)}, "
                   f"budget is {fmt_bytes(self.limit)}")
            if self.strict:
                raise MemoryBudgetExceeded(msg)
            print(f"WARNING: {msg}", file=sys.stderr)
        return ok

    def snapshot(self, label: str) -> dict:
        event = {"t": time.time(), "what": label, "rss": rss_bytes(), "peak_rss": peak_rss_bytes()}
        self.events.append(event)
        return event

    def report(self) -> dict:
        return {"limit": self.limit, "strict": self.strict, "peak_rss": peak_rss_bytes(), "events": self.events}
