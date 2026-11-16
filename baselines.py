"""Baselines: no reuse, hold with reassignment, brute force (owned by workstream B)."""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import problem as P
from problem import Problem
import evaluate as E

MAX_BRUTE_FORCE_DECISIONS = 12


def _strip_ancillas(problem: Problem) -> Problem:
    """The same circuit with no ancilla bookkeeping. physical_schedule then
    returns the base gate list unchanged, so a simulator that consumes
    physical_schedule runs the circuit with every ancilla on its own qubit."""
    return dataclasses.replace(problem, ancillas=(), decisions=())


def no_reuse(problem: Problem, noise=None, simulator: Optional[Callable] = None) -> dict:
    """d = all zeros and every ancilla keeps its own physical qubit (no reassignment)."""
    z = E.zeros(problem)
    gates = P.build_schedule(problem, z)
    stripped = _strip_ancillas(problem)
    return {
        "schedule": "no_reuse",
        "d": list(z),
        "live_counts": E.live_counts(problem, z),
        "peak_ancillas": E.peak_ancillas(problem, z),
        "ancilla_qubits_used": len(problem.ancillas),
        "physical_qubits": len(problem.data_qubits) + len(problem.ancillas),
        "assignment": [[a.qubit, a.alloc_step, a.release_step, a.qubit] for a in problem.ancillas],
        "depth": len(P.layers(gates)),
        "extra_depth": len(P.layers(gates)) - E.depth(problem, z),
        "two_qubit_equiv": E.gate_two_qubit_equiv(gates),
        "extra_2q": 0,
        "n_gates": len(gates),
        "fidelity": E.fidelity(stripped, (), noise, simulator),
    }


def hold_all_reassigned(problem: Problem, noise=None, simulator: Optional[Callable] = None) -> dict:
    """d = all zeros, with greedy reassignment, so ancillas whose lifetimes never
    overlap still share a qubit."""
    out = E.evaluate(problem, E.zeros(problem), noise, simulator)
    return {"schedule": "hold_all_reassigned", **out}


def schedule_table(problem: Problem, noise=None, simulator: Optional[Callable] = None) -> list[dict]:
    """evaluate() for every one of the 2^N schedules (all-hold first)."""
    n = len(problem.decisions)
    if n > MAX_BRUTE_FORCE_DECISIONS:
        raise ValueError("brute force refuses N = %d decisions (limit %d): 2^N schedules"
                         % (n, MAX_BRUTE_FORCE_DECISIONS))
    return [E.evaluate(problem, d, noise, simulator) for d in E.all_schedules(problem)]


def objective(row: dict, depth0: int, lam_a: float, lam_d: float) -> float:
    """J(d) = (1 - F_d) + lam_a * peak_d + lam_d * (depth_d - depth_0)."""
    return (1.0 - row["fidelity"]) + lam_a * row["peak_ancillas"] + lam_d * (row["depth"] - depth0)


def argmin_table(table: list[dict], lam_a: float, lam_d: float) -> dict:
    """Argmin of J over a schedule_table. Exact ties go to the earlier row
    (binary counting order, so holding wins a tie)."""
    depth0 = table[0]["depth"]
    assert not any(table[0]["d"]), "table[0] must be the all-hold schedule"
    Js = [objective(row, depth0, lam_a, lam_d) for row in table]
    best = min(range(len(table)), key=lambda i: (Js[i], i))
    return {"d": tuple(table[best]["d"]), "J": Js[best], "J_all": Js, "row": table[best]}


def brute_force(problem: Problem, noise, lam_a: float, lam_d: float = 0.01,
                simulator: Optional[Callable] = None, table: Optional[list] = None) -> dict:
    """Evaluate all 2^N schedules and return the table plus the argmin of
    J(d) = (1 - F_d) + lam_a * peak_d + lam_d * (depth_d - depth_0).
    Pass a precomputed `table` (from schedule_table) to reuse it across lam_a."""
    if table is None:
        table = schedule_table(problem, noise, simulator)
    best = argmin_table(table, lam_a, lam_d)
    return {"problem": problem.name, "lam_a": lam_a, "lam_d": lam_d,
            "argmin": best["d"], "J": best["J"],
            "table": [{**row, "J": J} for row, J in zip(table, best["J_all"])]}


def flip_thresholds(table: list[dict], lam_d: float = 0.01) -> list[dict]:
    """Exact lam_a values (lam_a >= 0) where brute force's argmin changes.

    J_d(lam_a) = c_d + lam_a * peak_d is linear in lam_a, so the argmin follows
    the lower envelope of the lines. Returns [{"lam_a", "from", "to"}, ...]
    in increasing lam_a."""
    depth0 = table[0]["depth"]
    c = [objective(row, depth0, 0.0, lam_d) for row in table]
    m = [row["peak_ancillas"] for row in table]
    cur = min(range(len(table)), key=lambda i: (c[i], i))
    lam = 0.0
    out = []
    while True:
        best = None
        for j in range(len(table)):
            if m[j] < m[cur]:
                x = (c[j] - c[cur]) / (m[cur] - m[j])
                if x >= lam - 1e-15:
                    key = (x, m[j], j)
                    if best is None or key < best[0]:
                        best = (key, j)
        if best is None:
            return out
        lam, nxt = best[0][0], best[1]
        out.append({"lam_a": max(lam, 0.0), "from": tuple(table[cur]["d"]), "to": tuple(table[nxt]["d"])})
        cur = nxt
