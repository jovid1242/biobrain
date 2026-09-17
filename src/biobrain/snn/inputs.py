"""Synthetic external spike input, pre-generated as a CSR over time steps.

Pre-generation keeps random-number generation out of the timed simulation loop and guarantees that both
execution modes receive exactly the same events. Events within a step are unique neurons in ascending order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import InputSpec


@dataclass
class InputSchedule:
    indptr: np.ndarray   # int64 (steps + 1)
    neurons: np.ndarray  # int32
    weight: float
    info: dict

    @property
    def events(self) -> int:
        return int(self.neurons.size)

    @property
    def nbytes(self) -> int:
        return int(self.indptr.nbytes + self.neurons.nbytes)

    def at(self, t: int) -> np.ndarray:
        return self.neurons[self.indptr[t]:self.indptr[t + 1]]


def generate(spec: InputSpec, n: int, steps: int, seed: int) -> InputSchedule:
    rng = np.random.default_rng([seed, 0x1A7])
    eligible = np.sort(rng.permutation(n)[: max(1, int(round(spec.fraction * n)))]) if n else np.zeros(0, np.int64)
    per_step: list[np.ndarray] = []
    empty = np.zeros(0, dtype=np.int32)
    if spec.pattern == "sparse_pattern":
        members = np.sort(rng.choice(eligible, size=min(spec.pattern_neurons, eligible.size), replace=False))
        offsets = rng.integers(0, spec.pattern_length, size=members.size)
    pulse_members = None
    for t in range(steps):
        on = spec.on_steps is None or t < spec.on_steps
        if not on or spec.pattern == "none" or n == 0:
            per_step.append(empty)
        elif spec.pattern == "poisson":
            per_step.append(eligible[rng.random(eligible.size) < spec.rate].astype(np.int32))
        elif spec.pattern == "burst":
            active = (t % spec.burst_period) < spec.burst_on
            per_step.append(eligible[rng.random(eligible.size) < spec.rate].astype(np.int32) if active else empty)
        elif spec.pattern == "pulse":
            if t % spec.pulse_every == 0:
                k = int(round(spec.pulse_fraction * eligible.size))
                pulse_members = np.sort(rng.choice(eligible, size=k, replace=False)).astype(np.int32)
                per_step.append(pulse_members)
            else:
                per_step.append(empty)
        else:  # sparse_pattern: the same spatiotemporal pattern, repeated every pattern_every steps
            phase = t % spec.pattern_every
            per_step.append(members[offsets == phase].astype(np.int32) if phase < spec.pattern_length else empty)
    counts = np.array([a.size for a in per_step], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    neurons = np.concatenate(per_step).astype(np.int32) if steps else empty
    info = {"pattern": spec.pattern, "events": int(neurons.size), "eligible_neurons": int(eligible.size),
            "mean_events_per_step": float(neurons.size / max(steps, 1)),
            "mean_fraction_per_step": float(neurons.size / max(steps * max(n, 1), 1))}
    return InputSchedule(indptr, neurons, float(spec.weight), info)
