"""Rounding and exact evaluation of discrete schedules (owned by workstream B).

Every number the project reports is measured here, on a discrete schedule,
with exact quantities (true max, not SmoothMax). Depth and gate counts are taken
on the physical schedule, the same gate list A's simulator runs.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np

import problem as P
from problem import Problem

TWO_Q_EQUIV = {"x": 0, "h": 0, "cx": 1, "ccx": 6}   # Toffoli = 6 CNOT


def round_decisions(r) -> tuple[int, ...]:
    """d_i = 1 if r_i > 0.5, else 0."""
    return tuple(int(float(x) > 0.5) for x in np.asarray(r, dtype=float).reshape(-1))


def zeros(problem: Problem) -> tuple[int, ...]:
    return (0,) * len(problem.decisions)


def one_hot(problem: Problem, i: int) -> tuple[int, ...]:
    d = [0] * len(problem.decisions)
    d[i] = 1
    return tuple(d)


def all_schedules(problem: Problem):
    """Every decision vector, in binary counting order (all-hold first)."""
    import itertools
    return [tuple(d) for d in itertools.product((0, 1), repeat=len(problem.decisions))]


def live_intervals(problem: Problem, d) -> list[tuple[int, int, int]]:
    """(ancilla, start, end) occupied intervals in base steps, inclusive."""
    return P.ancilla_intervals(problem, d)


def live_counts(problem: Problem, d) -> list[int]:
    """Exact x_t: number of live intervals covering each base step t."""
    x = [0] * len(problem.gates)
    for _, s, e in live_intervals(problem, d):
        for t in range(s, e + 1):
            x[t] += 1
    return x


def peak_ancillas(problem: Problem, d) -> int:
    return max(live_counts(problem, d), default=0)


def assign_qubits(problem: Problem, d) -> dict:
    """Greedy interval colouring (problem.assign_intervals).

    Returns {"mapping": {(ancilla, start, end): physical qubit},
             "ancilla_qubits_used": k, "physical_qubits": len(data_qubits) + k}.
    """
    mapping = P.assign_intervals(problem, d)
    used = len(set(mapping.values()))
    return {"mapping": mapping, "ancilla_qubits_used": used,
            "physical_qubits": len(problem.data_qubits) + used}


def depth(problem: Problem, d) -> int:
    return len(P.layers(P.physical_schedule(problem, d)))


def gate_two_qubit_equiv(gates) -> int:
    return sum(TWO_Q_EQUIV[name] for name, _ in gates)


def two_qubit_equiv(problem: Problem, d) -> int:
    """Total two-qubit-equivalent gates of the schedule: cx 1, ccx 6, 1q gates 0.
    Prep is excluded, as for depth and live counts."""
    return gate_two_qubit_equiv(P.build_schedule(problem, d))


def fill_costs(problem: Problem) -> Problem:
    """Fill each decision's extra_depth and extra_2q from its one-hot schedule,
    relative to the all-hold physical schedule."""
    z = zeros(problem)
    d0, q0 = depth(problem, z), two_qubit_equiv(problem, z)
    decs = tuple(
        dataclasses.replace(dec, extra_depth=depth(problem, one_hot(problem, i)) - d0,
                            extra_2q=two_qubit_equiv(problem, one_hot(problem, i)) - q0)
        for i, dec in enumerate(problem.decisions))
    return dataclasses.replace(problem, decisions=decs)


def target_vector(problem: Problem) -> np.ndarray:
    if len(problem.target_state) == 0:
        from circuits import fill_target
        problem = fill_target(problem)
    return np.asarray(problem.target_state, dtype=np.complex128)


def fidelity_of_rho(rho, psi: np.ndarray) -> float:
    rho = np.asarray(rho)
    return float(np.real(psi.conj() @ rho @ psi))


def is_noiseless(noise) -> bool:
    if noise is None:
        return True
    from circuits import noise_rates
    r = noise_rates(noise)
    return r["gamma"] == 0 and r["lambda"] == 0 and all(p == 0 for p in r["p"].values())


def fidelity(problem: Problem, d, noise=None, simulator: Optional[Callable] = None) -> float:
    """Fidelity of the physical schedule's output to target_state.

    With `simulator` (A's sim.run_discrete, or circuits.reference_run_discrete),
    runs simulator(problem, d, noise) and returns <psi|rho|psi>. Without it, the
    noiseless fidelity from the numpy statevector reference; then noise must be
    None or all-zero (e.g. NoiseParams.none()), so a noisy request is never
    silently answered with a noiseless number."""
    psi = target_vector(problem)
    if simulator is not None:
        return fidelity_of_rho(simulator(problem, tuple(d), noise), psi)
    if not is_noiseless(noise):
        raise ValueError("noisy fidelity needs a simulator (sim.run_discrete or "
                         "circuits.reference_run_discrete)")
    from circuits import reference_state
    phi = reference_state(problem, d, physical=True)
    return float(abs(np.vdot(psi, phi)) ** 2)


def evaluate(problem: Problem, d, noise=None, simulator: Optional[Callable] = None) -> dict:
    """All exact metrics of the discrete schedule d, plus fidelity."""
    d = tuple(int(x) for x in d)
    z = zeros(problem)
    a = assign_qubits(problem, d)
    dep, dep0 = depth(problem, d), depth(problem, z)
    q2, q20 = two_qubit_equiv(problem, d), two_qubit_equiv(problem, z)
    return {
        "d": list(d),
        "live_counts": live_counts(problem, d),
        "peak_ancillas": peak_ancillas(problem, d),
        "ancilla_qubits_used": a["ancilla_qubits_used"],
        "physical_qubits": a["physical_qubits"],
        "assignment": [[q, s, e, phys] for (q, s, e), phys in sorted(a["mapping"].items())],
        "depth": dep,
        "extra_depth": dep - dep0,
        "two_qubit_equiv": q2,
        "extra_2q": q2 - q20,
        "n_gates": len(P.build_schedule(problem, d)),
        "fidelity": fidelity(problem, d, noise, simulator),
    }
