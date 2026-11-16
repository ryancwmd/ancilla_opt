"""Benchmark circuits and numpy reference simulators (owned by workstream B).

The statevector simulator here is independent of A's JAX simulator and is used
to compute every Problem's target_state, so it doubles as a cross-check.

`reference_run_discrete` is a small numpy density-matrix simulator that follows
the noise model in instructions.md (depolarizing after each gate, amplitude
plus phase damping on idle qubits in every layer). It has the same signature as
A's `sim.run_discrete`, so it can be passed as `simulator=` to evaluate.py and
baselines.py as a cross-check or before A's simulator lands. A's simulator is
the one the project reports.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from problem import (Ancilla, Decision, Problem, compute_gates, inverse_block,
                     layers, physical_schedule, validate)

# ---------------------------------------------------------------------------
# Gate matrices (most-significant-bit convention: qubit order as listed)
# ---------------------------------------------------------------------------

_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_I2 = np.eye(2, dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)


def _perm_matrix(dim: int, swap: tuple[int, int]) -> np.ndarray:
    m = np.eye(dim, dtype=np.complex128)
    i, j = swap
    m[[i, j]] = m[[j, i]]
    return m


GATES: dict[str, np.ndarray] = {
    "x": _X,
    "h": _H,
    "cx": _perm_matrix(4, (2, 3)),     # basis |control target>
    "ccx": _perm_matrix(8, (6, 7)),    # basis |c1 c2 target>
}


# ---------------------------------------------------------------------------
# Statevector reference
# ---------------------------------------------------------------------------

def apply_gate_sv(psi: np.ndarray, U: np.ndarray, qs, n: int) -> np.ndarray:
    k = len(qs)
    t = psi.reshape((2,) * n)
    t = np.tensordot(U.reshape((2,) * (2 * k)), t, axes=(list(range(k, 2 * k)), list(qs)))
    t = np.moveaxis(t, list(range(k)), list(qs))
    return t.reshape(2 ** n)


def run_statevector(n: int, gates, psi: np.ndarray | None = None) -> np.ndarray:
    """Apply gates to |0...0> (or to psi) and return the final statevector."""
    if psi is None:
        psi = np.zeros(2 ** n, dtype=np.complex128)
        psi[0] = 1.0
    for name, qs in gates:
        psi = apply_gate_sv(psi, GATES[name], qs, n)
    return psi


def reference_state(problem: Problem, d=None, physical: bool = True) -> np.ndarray:
    """Noiseless output of prep + the discrete schedule for d (default all-hold)."""
    from problem import build_schedule
    if d is None:
        d = (0,) * len(problem.decisions)
    sched = physical_schedule(problem, d) if physical else build_schedule(problem, d)
    return run_statevector(problem.n_qubits, list(problem.prep) + list(sched))


def fill_target(problem: Problem) -> Problem:
    """Return the Problem with target_state = noiseless output of prep + gates."""
    psi = run_statevector(problem.n_qubits, list(problem.prep) + list(problem.gates))
    return dataclasses.replace(problem, target_state=tuple(complex(c) for c in psi))


# ---------------------------------------------------------------------------
# Noisy density-matrix reference (cross-check for A's simulator)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RefNoise:
    """Stand-in with the same fields and defaults as A's noise.NoiseParams, so
    B's tests and scripts run without importing A's code. Use A's NoiseParams
    for reported results; any object with these fields works everywhere."""
    p1: float = 1e-3
    p2: float = 1e-2
    p3: float | None = None
    t_layer_us: float = 3.0
    T1_us: float = 200.0
    T2_us: float = 150.0
    source: str = "defaults"

    @classmethod
    def none(cls) -> "RefNoise":
        return cls(p1=0.0, p2=0.0, p3=0.0, T1_us=math.inf, T2_us=math.inf, source="none")

    def scaled(self, s: float) -> "RefNoise":
        if s == 0:
            return RefNoise.none()
        return dataclasses.replace(self, p1=self.p1 * s, p2=self.p2 * s,
                                   p3=None if self.p3 is None else self.p3 * s,
                                   T1_us=self.T1_us / s, T2_us=self.T2_us / s,
                                   source="%s x%g" % (self.source, s))


def noise_rates(noise) -> dict:
    """Per-gate depolarizing strengths and per-layer idle strengths from any
    object with A's NoiseParams fields (p1, p2, p3, t_layer_us, T1_us, T2_us)."""
    p1, p2 = float(noise.p1), float(noise.p2)
    p3 = 1.0 - (1.0 - p2) ** 6 if noise.p3 is None else float(noise.p3)
    t, T1, T2 = float(noise.t_layer_us), float(noise.T1_us), float(noise.T2_us)
    gamma = 0.0 if math.isinf(T1) else 1.0 - math.exp(-t / T1)
    inv_tphi = (0.0 if math.isinf(T2) else 1.0 / T2) - (0.0 if math.isinf(T1) else 1.0 / (2.0 * T1))
    if inv_tphi < -1e-15:
        raise ValueError("need T2 <= 2*T1, got T1=%r, T2=%r" % (T1, T2))
    lam = 1.0 - math.exp(-2.0 * t * max(inv_tphi, 0.0))
    return {"p": {1: p1, 2: p2, 3: p3}, "gamma": gamma, "lambda": lam}


def _apply_op_dm(rho: np.ndarray, U: np.ndarray, qs, n: int) -> np.ndarray:
    """U rho U^dagger on qubits qs; rho is a (2,)*2n tensor."""
    k = len(qs)
    Ut = U.reshape((2,) * (2 * k))
    rho = np.tensordot(Ut, rho, axes=(list(range(k, 2 * k)), list(qs)))
    rho = np.moveaxis(rho, list(range(k)), list(qs))
    cols = [n + q for q in qs]
    rho = np.tensordot(rho, Ut.conj(), axes=(cols, list(range(k, 2 * k))))
    rho = np.moveaxis(rho, list(range(2 * n - k, 2 * n)), cols)
    return rho


def _full_depolarize(rho: np.ndarray, qs, n: int) -> np.ndarray:
    """Tr_qs(rho) (x) I/2^k, applied one qubit at a time: on qubit q, the 2x2
    block (row q, col q) is replaced by its trace times I/2."""
    for q in qs:
        r = np.moveaxis(rho, (q, n + q), (0, 1))
        half_tr = 0.5 * (r[0, 0] + r[1, 1])
        out = np.empty_like(r)
        out[0, 0] = half_tr
        out[1, 1] = half_tr
        out[0, 1] = 0
        out[1, 0] = 0
        rho = np.moveaxis(out, (0, 1), (q, n + q))
    return rho


def _idle(rho: np.ndarray, q: int, n: int, gamma: float, lam: float) -> np.ndarray:
    """Amplitude damping (gamma) then phase damping (lam) on qubit q, written as
    the closed-form block update of their Kraus sums (tests check the match):
    rho00 += gamma*rho11, rho11 *= 1-gamma, off-diagonals *= sqrt((1-gamma)(1-lam))."""
    r = np.moveaxis(rho, (q, n + q), (0, 1)).copy()
    r[0, 0] += gamma * r[1, 1]
    r[1, 1] *= 1.0 - gamma
    c = math.sqrt((1.0 - gamma) * (1.0 - lam))
    r[0, 1] *= c
    r[1, 0] *= c
    return np.moveaxis(r, (0, 1), (q, n + q))


def kraus_ops(gamma: float, lam: float) -> dict:
    """Kraus operators for the idle channels (for tests and reference)."""
    return {
        "amp_damp": [np.array([[1, 0], [0, math.sqrt(1 - gamma)]], dtype=np.complex128),
                     np.array([[0, math.sqrt(gamma)], [0, 0]], dtype=np.complex128)],
        "phase_damp": [np.array([[1, 0], [0, math.sqrt(1 - lam)]], dtype=np.complex128),
                       np.array([[0, 0], [0, math.sqrt(lam)]], dtype=np.complex128)],
    }


def apply_kraus_dm(rho: np.ndarray, ops, q: int, n: int) -> np.ndarray:
    return sum(_apply_op_dm(rho, K, [q], n) for K in ops)


def reference_run_discrete(problem: Problem, d, noise=None) -> np.ndarray:
    """Numpy density matrix for the physical schedule of d under `noise`
    (None = noiseless). Same procedure as A's run_discrete: prep without noise,
    then per ASAP layer: each gate followed by its depolarizing channel, then
    amplitude and phase damping on every qubit the layer does not touch."""
    n = problem.n_qubits
    psi = run_statevector(n, problem.prep)
    rho = np.outer(psi, psi.conj()).reshape((2,) * (2 * n))
    sched = physical_schedule(problem, d)
    rates = noise_rates(noise) if noise is not None else None
    for layer in layers(sched):
        touched = set()
        for gi in layer:
            name, qs = sched[gi]
            rho = _apply_op_dm(rho, GATES[name], qs, n)
            touched.update(qs)
            if rates is not None:
                p = rates["p"][len(qs)]
                if p > 0:
                    rho = (1 - p) * rho + p * _full_depolarize(rho, qs, n)
        if rates is not None and (rates["gamma"] > 0 or rates["lambda"] > 0):
            for q in range(n):
                if q not in touched:
                    rho = _idle(rho, q, n, rates["gamma"], rates["lambda"])
    return rho.reshape(2 ** n, 2 ** n)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def _decision(problem_gates, ancilla: Ancilla, id: str, free_after: int, recompute_before: int) -> Decision:
    tmp = Problem("tmp", 0, (), (ancilla,), (), tuple(problem_gates), (), ())
    comp = compute_gates(tmp, ancilla)
    return Decision(id=id, ancilla=ancilla.qubit, free_after=free_after,
                    recompute_before=recompute_before,
                    uncompute=inverse_block(comp), recompute=comp)


def toy() -> Problem:
    """Benchmark 1 (instructions.md). Qubits: 0=c1 1=c2 2=c3 3=c4 4=a 5=b 6=t1 7=t2."""
    gates = (
        ("ccx", (0, 1, 4)),   # 0 compute a = c1 AND c2
        ("ccx", (4, 2, 6)),   # 1 first use of a
        ("ccx", (2, 3, 5)),   # 2 compute b = c3 AND c4
        ("cx", (5, 7)),       # 3 use b
        ("ccx", (2, 3, 5)),   # 4 uncompute b
        ("ccx", (4, 3, 7)),   # 5 second use of a
        ("ccx", (0, 1, 4)),   # 6 final uncompute of a
    )
    a = Ancilla(4, alloc_step=0, release_step=6)
    b = Ancilla(5, alloc_step=2, release_step=4)
    p = Problem(
        name="toy",
        n_qubits=8,
        data_qubits=(0, 1, 2, 3, 6, 7),
        ancillas=(a, b),
        prep=tuple(("h", (q,)) for q in (0, 1, 2, 3)),
        gates=gates,
        decisions=(_decision(gates, a, "a_gap", free_after=1, recompute_before=5),),
        target_state=(),
    )
    p = fill_target(p)
    validate(p)
    return p


def bench2() -> Problem:
    """Benchmark 2: a = c1 AND c2 is used three times, with a different ancilla
    in each of the two gaps, giving two decisions on a.

    Qubits (9): 0=c1 1=c2 2=c3 3=c4 4=a 5=b 6=c 7=t1 8=t2.
    b = c3 AND c4 lives in gap 1; c = c2 AND c3 lives in gap 2. b and c never
    overlap, so reassignment alone puts them on one qubit even when a is held.
    Freeing only one gap leaves the peak at 2; only freeing both reaches 1.
    """
    gates = (
        ("ccx", (0, 1, 4)),   # 0  compute a = c1 AND c2
        ("ccx", (4, 2, 7)),   # 1  use 1 of a
        ("ccx", (2, 3, 5)),   # 2  compute b = c3 AND c4
        ("cx", (5, 8)),       # 3  use b
        ("ccx", (2, 3, 5)),   # 4  uncompute b
        ("ccx", (4, 3, 8)),   # 5  use 2 of a
        ("ccx", (1, 2, 6)),   # 6  compute c = c2 AND c3
        ("cx", (6, 7)),       # 7  use c
        ("ccx", (1, 2, 6)),   # 8  uncompute c
        ("ccx", (4, 3, 7)),   # 9  use 3 of a
        ("ccx", (0, 1, 4)),   # 10 final uncompute of a
    )
    a = Ancilla(4, alloc_step=0, release_step=10)
    b = Ancilla(5, alloc_step=2, release_step=4)
    c = Ancilla(6, alloc_step=6, release_step=8)
    p = Problem(
        name="bench2",
        n_qubits=9,
        data_qubits=(0, 1, 2, 3, 7, 8),
        ancillas=(a, b, c),
        prep=tuple(("h", (q,)) for q in (0, 1, 2, 3)),
        gates=gates,
        decisions=(
            _decision(gates, a, "a_gap1", free_after=1, recompute_before=5),
            _decision(gates, a, "a_gap2", free_after=5, recompute_before=9),
        ),
        target_state=(),
    )
    p = fill_target(p)
    validate(p)
    return p


BENCHMARKS = {"toy": toy, "bench2": bench2}
