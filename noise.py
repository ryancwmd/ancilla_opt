"""Noise parameters and Kraus operators (workstream A).

Markovian model: k-qubit depolarizing noise after every gate (p1, p2, p3 by gate
size), amplitude damping plus phase damping on every idle qubit in every ASAP
layer. Channel application lives in sim.py; this module only holds parameters
and Kraus matrices (plain numpy, so they can be checked independently).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

import numpy as np

PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@dataclass(frozen=True)
class NoiseParams:
    p1: float = 1e-3          # depolarizing, single-qubit gates
    p2: float = 1e-2          # depolarizing, cx
    p3: float | None = None   # depolarizing, ccx; None -> 1 - (1 - p2)**6, about 0.059
    t_layer_us: float = 3.0   # one layer's duration, ~ one Toffoli
    T1_us: float = 200.0
    T2_us: float = 150.0      # must satisfy T2 <= 2 * T1
    source: str = "defaults"  # record where numbers came from

    def __post_init__(self):
        if self.T2_us > 2 * self.T1_us * (1 + 1e-12):
            raise ValueError("need T2 <= 2*T1, got T1=%r, T2=%r" % (self.T1_us, self.T2_us))
        for name in ("p1", "p2"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError("%s must be in [0, 1], got %r" % (name, v))
        if self.p3 is not None and not 0.0 <= self.p3 <= 1.0:
            raise ValueError("p3 must be in [0, 1], got %r" % (self.p3,))

    # -- constructors -------------------------------------------------------
    @classmethod
    def none(cls) -> "NoiseParams":
        """Exactly noiseless: zero gate error, infinite T1 and T2."""
        return cls(p1=0.0, p2=0.0, p3=0.0, T1_us=math.inf, T2_us=math.inf, source="none")

    def scaled(self, s: float) -> "NoiseParams":
        """Multiply gate errors by s and divide T1, T2 by s. scaled(0) is none()."""
        if s == 0:
            return NoiseParams.none()
        if s < 0:
            raise ValueError("noise scale must be >= 0, got %r" % (s,))
        return replace(
            self,
            p1=min(1.0, self.p1 * s),
            p2=min(1.0, self.p2 * s),
            p3=None if self.p3 is None else min(1.0, self.p3 * s),
            T1_us=self.T1_us / s,
            T2_us=self.T2_us / s,
            source="%s x%g" % (self.source, s),
        )

    # -- derived quantities -------------------------------------------------
    @property
    def p3_eff(self) -> float:
        return 1.0 - (1.0 - self.p2) ** 6 if self.p3 is None else self.p3

    def p_gate(self, k: int) -> float:
        """Depolarizing strength for a k-qubit gate."""
        if k == 1:
            return self.p1
        if k == 2:
            return self.p2
        if k == 3:
            return self.p3_eff
        raise ValueError("no depolarizing strength for %d-qubit gates" % k)

    @property
    def gamma(self) -> float:
        """Per-layer amplitude damping: 1 - exp(-t_layer / T1)."""
        if math.isinf(self.T1_us):
            return 0.0
        return 1.0 - math.exp(-self.t_layer_us / self.T1_us)

    @property
    def T_phi_us(self) -> float:
        """Pure dephasing time from 1/T_phi = 1/T2 - 1/(2 T1)."""
        inv = (0.0 if math.isinf(self.T2_us) else 1.0 / self.T2_us) - (
            0.0 if math.isinf(self.T1_us) else 1.0 / (2.0 * self.T1_us))
        if inv <= 1e-300:
            return math.inf
        return 1.0 / inv

    @property
    def lam(self) -> float:
        """Per-layer phase damping: 1 - exp(-2 t_layer / T_phi)."""
        tphi = self.T_phi_us
        if math.isinf(tphi):
            return 0.0
        return 1.0 - math.exp(-2.0 * self.t_layer_us / tphi)

    def is_noiseless(self) -> bool:
        return self.p1 == 0 and self.p2 == 0 and self.p3_eff == 0 and self.gamma == 0 and self.lam == 0

    def to_dict(self) -> dict:
        return {"p1": self.p1, "p2": self.p2, "p3": self.p3_eff, "t_layer_us": self.t_layer_us,
                "T1_us": self.T1_us, "T2_us": self.T2_us, "gamma": self.gamma, "lambda": self.lam,
                "source": self.source}


# ---------------------------------------------------------------------------
# Kraus operators
# ---------------------------------------------------------------------------

def amplitude_damping_kraus(gamma: float):
    return [np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=complex),
            np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=complex)]


def phase_damping_kraus(lam: float):
    return [np.array([[1, 0], [0, math.sqrt(1 - lam)]], dtype=complex),
            np.array([[0, 0], [0, math.sqrt(lam)]], dtype=complex)]


def pauli_strings(k: int):
    """All 4^k k-qubit Pauli matrices, identity first (qubit 0 most significant)."""
    out = []
    for labels in itertools.product("IXYZ", repeat=k):
        m = np.array([[1]], dtype=complex)
        for l in labels:
            m = np.kron(m, PAULIS[l])
        out.append(m)
    return out


def depolarizing_kraus(p: float, k: int):
    """Kraus form of the k-qubit depolarizing channel (1-p) rho + p I/2^k Tr rho.
    sim.apply_depolarizing uses the equivalent trace form; this is for tests."""
    d2 = 4 ** k
    ps = pauli_strings(k)
    return [math.sqrt(1 - p + p / d2) * ps[0]] + [math.sqrt(p / d2) * P for P in ps[1:]]


# ---------------------------------------------------------------------------
# Optional: parameters from a Qiskit fake backend (never needed by tests)
# ---------------------------------------------------------------------------

def from_fake_backend(name: str = "FakeSherbrooke") -> NoiseParams:
    """Median gate errors, T1, T2 and 2q gate duration from a
    qiskit_ibm_runtime.fake_provider backend. IBM reports average gate error r;
    the depolarizing p is r * 2^k / (2^k - 1)."""
    try:
        from qiskit_ibm_runtime import fake_provider
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError("from_fake_backend needs qiskit-ibm-runtime: %s" % e)
    backend = getattr(fake_provider, name)()
    target = backend.target
    one_q, two_q, dur2 = [], [], []
    for op_name in target.operation_names:
        for qargs, props in target[op_name].items():
            if props is None or props.error is None or qargs is None:
                continue
            if len(qargs) == 1 and op_name in ("sx", "x"):
                one_q.append(props.error)
            elif len(qargs) == 2:
                two_q.append(props.error)
                if props.duration:
                    dur2.append(props.duration)
    t1 = [target.qubit_properties[i].t1 for i in range(target.num_qubits)]
    t2 = [target.qubit_properties[i].t2 for i in range(target.num_qubits)]
    med = lambda xs: float(np.median([x for x in xs if x is not None]))
    r1, r2 = med(one_q), med(two_q)
    T1, T2 = med(t1) * 1e6, med(t2) * 1e6
    T2 = min(T2, 2 * T1)
    p2 = min(1.0, 4 * r2 / 3)
    # one layer ~ one Toffoli ~ 6 two-qubit gates in sequence
    t_layer = 6 * med(dur2) * 1e6 if dur2 else 3.0
    return NoiseParams(p1=min(1.0, 2 * r1), p2=p2, p3=None, t_layer_us=t_layer,
                       T1_us=T1, T2_us=T2, source="fake_backend:%s" % name)
