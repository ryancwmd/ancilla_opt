"""Toy Problem for A's tests (workstream A).

Uses B's circuits.toy() when it is available; otherwise builds the same Problem
from the spec in instructions.md (Benchmark 1), with the target state from
sim.statevector.
"""
from __future__ import annotations

from problem import Ancilla, Decision, Problem

TOY_PREP = (("h", (0,)), ("h", (1,)), ("h", (2,)), ("h", (3,)))
TOY_GATES = (
    ("ccx", (0, 1, 4)),  # 0 compute a
    ("ccx", (4, 2, 6)),  # 1 first use of a
    ("ccx", (2, 3, 5)),  # 2 compute b
    ("cx", (5, 7)),      # 3 use b
    ("ccx", (2, 3, 5)),  # 4 uncompute b
    ("ccx", (4, 3, 7)),  # 5 second use of a
    ("ccx", (0, 1, 4)),  # 6 final uncompute of a
)


def toy_from_spec() -> Problem:
    import sim
    psi = sim.statevector(list(TOY_PREP) + list(TOY_GATES), 8)
    return Problem(
        name="toy",
        n_qubits=8,
        data_qubits=(0, 1, 2, 3, 6, 7),
        ancillas=(Ancilla(4, 0, 6), Ancilla(5, 2, 4)),
        prep=TOY_PREP,
        gates=TOY_GATES,
        decisions=(Decision(id="a_gap", ancilla=4, free_after=1, recompute_before=5,
                            uncompute=(("ccx", (0, 1, 4)),), recompute=(("ccx", (0, 1, 4)),)),),
        target_state=tuple(complex(c) for c in psi),
    )


def toy_problem() -> Problem:
    try:
        from circuits import toy
    except ImportError:
        return toy_from_spec()
    return toy()


def two_gap_problem() -> Problem:
    """A's own 9-qubit, 2-decision test Problem (not B's benchmark 2): a = c1 AND
    c2 used three times, ancilla b in the first gap and ancilla c in the second.
    Exercises the N = 2 mixture and live-count code before benchmark 2 lands."""
    import sim
    prep = TOY_PREP
    gates = (
        ("ccx", (0, 1, 4)),  # 0 compute a
        ("ccx", (4, 2, 6)),  # 1 use a
        ("ccx", (2, 3, 5)),  # 2 compute b
        ("cx", (5, 7)),      # 3 use b
        ("ccx", (2, 3, 5)),  # 4 uncompute b
        ("ccx", (4, 3, 7)),  # 5 use a
        ("cx", (2, 8)),      # 6 compute c
        ("cx", (8, 6)),      # 7 use c
        ("cx", (2, 8)),      # 8 uncompute c
        ("ccx", (4, 2, 6)),  # 9 use a
        ("ccx", (0, 1, 4)),  # 10 uncompute a
    )
    psi = sim.statevector(list(prep) + list(gates), 9)
    blk = (("ccx", (0, 1, 4)),)
    return Problem(
        name="a_two_gap",
        n_qubits=9,
        data_qubits=(0, 1, 2, 3, 6, 7),
        ancillas=(Ancilla(4, 0, 10), Ancilla(5, 2, 4), Ancilla(8, 6, 8)),
        prep=prep,
        gates=gates,
        decisions=(Decision("gap1", 4, 1, 5, blk, blk), Decision("gap2", 4, 5, 9, blk, blk)),
        target_state=tuple(complex(c) for c in psi),
    )
