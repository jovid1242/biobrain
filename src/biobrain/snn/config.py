"""Simulation configuration.

Every field is classified in `PROVENANCE` as OBSERVED (read from FlyWire), DERIVED (computed from observed
data by a fixed rule), ASSUMED (our modelling choice; FlyWire does not determine it) or TUNED (set by the
documented calibration procedure). docs/ASSUMPTIONS.md explains each choice.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

PROVENANCE = {
    "neuron.dt_ms": "ASSUMED",
    "neuron.tau_m_ms": "ASSUMED",
    "neuron.v_threshold": "ASSUMED (unit of every potential and weight)",
    "neuron.v_reset": "ASSUMED",
    "neuron.refractory_ms": "ASSUMED",
    "neuron.delay_ms": "ASSUMED (FlyWire has no conduction delays)",
    "neuron.dtype": "ASSUMED (numerical choice)",
    "weights.synapse_count": "OBSERVED",
    "weights.transform": "ASSUMED",
    "weights.normalization": "DERIVED (connectome-wide mean of the transform)",
    "weights.gain": "TUNED (calibration procedure) or ASSUMED where stated",
    "weights.clip": "ASSUMED",
    "weights.min_synapses": "ASSUMED",
    "signs.predictions": "OBSERVED (Eckstein et al. 2024 predictions; not ground truth)",
    "signs.mapping": "ASSUMED (transmitter -> fast sign; receptors are not in the data)",
    "signs.source": "ASSUMED (one transmitter per neuron vs. per-edge prediction)",
    "inputs.*": "ASSUMED (synthetic drive, not sensory data)",
    "run.*": "experimental setting",
}

SIGN_KEYS = ("acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine")


@dataclass(frozen=True)
class NeuronParams:
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory_ms: float = 2.0
    delay_ms: float = 1.0
    dtype: str = "float32"

    def __post_init__(self):
        if self.dt_ms <= 0 or self.tau_m_ms <= 0 or self.refractory_ms < 0:
            raise ValueError("dt and tau must be positive, refractory period non-negative")
        if self.v_reset >= self.v_threshold:
            raise ValueError("v_reset must be below v_threshold (a reset neuron must not spike without input)")
        if self.delay_ms < self.dt_ms:
            raise ValueError("delay must be at least one time step")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")

    @property
    def alpha(self) -> float:
        return math.exp(-self.dt_ms / self.tau_m_ms)

    @property
    def ref_steps(self) -> int:
        return int(round(self.refractory_ms / self.dt_ms))

    @property
    def delay_steps(self) -> int:
        return max(1, int(round(self.delay_ms / self.dt_ms)))


@dataclass(frozen=True)
class WeightSpec:
    transform: str = "linear"  # linear | sqrt | log1p | clipped
    gain: float = 0.1          # mean |weight| of a connectome edge, in threshold units
    clip: float = 20.0         # synapse-count ceiling for "clipped"
    min_synapses: int = 1

    def __post_init__(self):
        if self.transform not in ("linear", "sqrt", "log1p", "clipped"):
            raise ValueError(f"unknown weight transform {self.transform!r}")
        if self.gain < 0 or self.clip <= 0 or self.min_synapses < 1:
            raise ValueError("gain >= 0, clip > 0, min_synapses >= 1")


@dataclass(frozen=True)
class SignSpec:
    source: str = "neuron"  # neuron: annotated top_nt of the presynaptic neuron | edge: dominant predicted class
    acetylcholine: float = 1.0
    gaba: float = -1.0
    glutamate: float = -1.0
    dopamine: float = 0.0
    serotonin: float = 0.0
    octopamine: float = 0.0
    missing: float = 0.0

    def __post_init__(self):
        if self.source not in ("neuron", "edge"):
            raise ValueError("sign source must be 'neuron' or 'edge'")
        for key in (*SIGN_KEYS, "missing"):
            if getattr(self, key) not in (-1.0, 0.0, 1.0):
                raise ValueError(f"sign for {key} must be -1, 0 or +1")

    def value(self, transmitter: str) -> float:
        return getattr(self, transmitter) if transmitter in SIGN_KEYS else self.missing


@dataclass(frozen=True)
class InputSpec:
    pattern: str = "poisson"  # none | poisson | burst | pulse | sparse_pattern
    rate: float = 0.01        # per eligible neuron per step (within bursts for "burst")
    weight: float = 2.0       # external event size in threshold units (>= 1 forces a spike unless refractory)
    fraction: float = 1.0     # share of neurons eligible for external input
    on_steps: int | None = None  # input only during [0, on_steps)
    burst_period: int = 200
    burst_on: int = 50
    pulse_every: int = 100
    pulse_fraction: float = 0.1
    pattern_neurons: int = 20
    pattern_length: int = 50
    pattern_every: int = 100

    def __post_init__(self):
        if self.pattern not in ("none", "poisson", "burst", "pulse", "sparse_pattern"):
            raise ValueError(f"unknown input pattern {self.pattern!r}")
        if not (0 <= self.rate <= 1 and 0 < self.fraction <= 1 and 0 <= self.pulse_fraction <= 1):
            raise ValueError("rates and fractions must lie in [0, 1]")


@dataclass(frozen=True)
class RunSpec:
    steps: int = 1000
    mode: str = "time_step"  # time_step | event_driven
    aggregation: str = "sparse"  # event_driven only: sparse | auto
    seed: int = 0
    record_spikes: bool = False
    record_voltage_every: int = 0
    profile: bool = False

    def __post_init__(self):
        if self.mode not in ("time_step", "event_driven"):
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.aggregation not in ("sparse", "auto"):
            raise ValueError(f"unknown aggregation {self.aggregation!r}")
        if self.steps < 1:
            raise ValueError("steps >= 1")


@dataclass(frozen=True)
class SimConfig:
    neuron: NeuronParams = field(default_factory=NeuronParams)
    weights: WeightSpec = field(default_factory=WeightSpec)
    signs: SignSpec = field(default_factory=SignSpec)
    inputs: InputSpec = field(default_factory=InputSpec)
    run: RunSpec = field(default_factory=RunSpec)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfig":
        return cls(NeuronParams(**data.get("neuron", {})), WeightSpec(**data.get("weights", {})),
                   SignSpec(**data.get("signs", {})), InputSpec(**data.get("inputs", {})), RunSpec(**data.get("run", {})))

    def replace(self, **sections) -> "SimConfig":
        """replace(run={"mode": "event_driven"}, weights={"gain": 0.2}) — nested partial updates."""
        updated = {name: dataclasses.replace(getattr(self, name), **changes) for name, changes in sections.items()}
        return dataclasses.replace(self, **updated)

    def digest(self, *sections: str) -> str:
        """sha256 of the canonical JSON of the given sections (all by default)."""
        data = self.to_dict()
        chosen = {k: data[k] for k in (sections or data)}
        return hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest()

    @property
    def model_hash(self) -> str:
        """Identical for both modes of the same model: neuron, weights, signs."""
        return self.digest("neuron", "weights", "signs")

    @property
    def stimulus_hash(self) -> str:
        return hashlib.sha256(json.dumps([self.to_dict()["inputs"], self.run.seed, self.run.steps], sort_keys=True).encode()).hexdigest()
