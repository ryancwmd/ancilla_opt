"""Shared data model for differentiable ancilla reuse (owned by workstream B).

This file is the contract between workstreams A, B and C. It uses only the
Python standard library and must import on Python 3.8 or later, so that C's
separate Reqomp environment can import it too.

Conventions (see instructions.md):
  * Qubit 0 is the most significant bit of a basis index.
  * Gates are (name, qubits) with controls first and target last:
      ("x", (q,)), ("h", (q,)), ("cx", (c, t)), ("ccx", (c1, c2, t)).
  * "Step t" means while base gate t runs. An ancilla is live at both its
    alloc_step and its release_step (intervals are inclusive).
  * The base gate list is the hold-everything schedule. Each Decision is one
    gap in which its ancilla may be freed (uncompute after base gate
    free_after, recompute before base gate recompute_before).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Gate = Tuple[str, Tuple[int, ...]]        # ("ccx", (0, 1, 4)): controls first, target last

# Number of qubits each known gate acts on.
GATE_ARITY = {"x": 1, "h": 1, "cx": 2, "ccx": 3}
# Gates that are their own inverse (all current gates). A new gate that is not
# self-inverse needs an entry in INVERSE_NAME.
SELF_INVERSE = frozenset({"x", "h", "cx", "ccx"})
INVERSE_NAME: Dict[str, str] = {}

# An interval of occupation for one ancilla, in base-step indices, inclusive:
# (ancilla qubit, start step, end step).
Interval = Tuple[int, int, int]


@dataclass(frozen=True)
class Ancilla:
    qubit: int
    alloc_step: int       # index of the base gate that first computes it
    release_step: int     # index of the base gate that finally uncomputes it


@dataclass(frozen=True)
class Decision:
    id: str
    ancilla: int                  # qubit index
    free_after: int               # uncompute block runs right after base gate free_after
    recompute_before: int         # recompute block runs right before base gate recompute_before
    uncompute: tuple[Gate, ...]   # inverse of the ancilla's compute gates, in reverse order
    recompute: tuple[Gate, ...]   # the ancilla's compute gates, in order
    extra_depth: int = -1         # filled by B's evaluate.py (-1 = not computed yet)
    extra_2q: int = -1            # extra two-qubit-equivalent gates, filled by B


@dataclass(frozen=True)
class Problem:
    name: str
    n_qubits: int
    data_qubits: tuple[int, ...]
    ancillas: tuple[Ancilla, ...]
    prep: tuple[Gate, ...]        # state preparation, run first; excluded from depth and live counts
    gates: tuple[Gate, ...]       # base (hold-everything) schedule
    decisions: tuple[Decision, ...]
    target_state: tuple[complex, ...]   # noiseless output of prep + gates, length 2**n_qubits


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _gate_to_json(g: Gate) -> list:
    return [g[0], list(g[1])]


def _gate_from_json(obj) -> Gate:
    name, qs = obj
    return (str(name), tuple(int(q) for q in qs))


def to_dict(problem: Problem) -> dict:
    """Plain-JSON-compatible dict. Complex numbers become [re, im] pairs."""
    return {
        "name": problem.name,
        "n_qubits": problem.n_qubits,
        "data_qubits": list(problem.data_qubits),
        "ancillas": [
            {"qubit": a.qubit, "alloc_step": a.alloc_step, "release_step": a.release_step}
            for a in problem.ancillas
        ],
        "prep": [_gate_to_json(g) for g in problem.prep],
        "gates": [_gate_to_json(g) for g in problem.gates],
        "decisions": [
            {
                "id": d.id,
                "ancilla": d.ancilla,
                "free_after": d.free_after,
                "recompute_before": d.recompute_before,
                "uncompute": [_gate_to_json(g) for g in d.uncompute],
                "recompute": [_gate_to_json(g) for g in d.recompute],
                "extra_depth": d.extra_depth,
                "extra_2q": d.extra_2q,
            }
            for d in problem.decisions
        ],
        "target_state": [[complex(c).real, complex(c).imag] for c in problem.target_state],
    }


def from_dict(obj: dict) -> Problem:
    return Problem(
        name=str(obj["name"]),
        n_qubits=int(obj["n_qubits"]),
        data_qubits=tuple(int(q) for q in obj["data_qubits"]),
        ancillas=tuple(
            Ancilla(int(a["qubit"]), int(a["alloc_step"]), int(a["release_step"]))
            for a in obj["ancillas"]
        ),
        prep=tuple(_gate_from_json(g) for g in obj.get("prep", [])),
        gates=tuple(_gate_from_json(g) for g in obj["gates"]),
        decisions=tuple(
            Decision(
                id=str(d["id"]),
                ancilla=int(d["ancilla"]),
                free_after=int(d["free_after"]),
                recompute_before=int(d["recompute_before"]),
                uncompute=tuple(_gate_from_json(g) for g in d["uncompute"]),
                recompute=tuple(_gate_from_json(g) for g in d["recompute"]),
                extra_depth=int(d.get("extra_depth", -1)),
                extra_2q=int(d.get("extra_2q", -1)),
            )
            for d in obj["decisions"]
        ),
        target_state=tuple(complex(re, im) for re, im in obj.get("target_state", [])),
    )


def to_json(problem: Problem, path) -> None:
    with open(path, "w") as f:
        json.dump(to_dict(problem), f, indent=1)


def from_json(path) -> Problem:
    with open(path) as f:
        return from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

def controls(g: Gate) -> Tuple[int, ...]:
    return tuple(g[1][:-1])


def target(g: Gate) -> int:
    return g[1][-1]


def inverse_gate(g: Gate) -> Gate:
    name, qs = g
    if name in SELF_INVERSE:
        return g
    if name in INVERSE_NAME:
        return (INVERSE_NAME[name], qs)
    raise ValueError("no inverse known for gate %r; add it to problem.INVERSE_NAME" % (name,))


def inverse_block(gates: Sequence[Gate]) -> Tuple[Gate, ...]:
    """Inverse of a gate sequence: each gate inverted, in reverse order."""
    return tuple(inverse_gate(g) for g in reversed(list(gates)))


def ancilla_by_qubit(problem: Problem) -> Dict[int, Ancilla]:
    return {a.qubit: a for a in problem.ancillas}


def compute_gates(problem: Problem, ancilla: Ancilla) -> Tuple[Gate, ...]:
    """The ancilla's compute gates: base gates that target it, from alloc_step
    up to (not including) the first base gate that uses it as a control."""
    q = ancilla.qubit
    out = []
    for t in range(ancilla.alloc_step, len(problem.gates)):
        g = problem.gates[t]
        if q in controls(g):
            break
        if target(g) == q:
            out.append(g)
    return tuple(out)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_gate(g, n_qubits: int, where: str) -> None:
    if not (isinstance(g, tuple) and len(g) == 2 and isinstance(g[0], str) and isinstance(g[1], tuple)):
        raise ValueError("%s: gate %r is not a (name, qubits) tuple" % (where, g))
    name, qs = g
    if name not in GATE_ARITY:
        raise ValueError("%s: unknown gate name %r (known: %s)" % (where, name, sorted(GATE_ARITY)))
    if len(qs) != GATE_ARITY[name]:
        raise ValueError("%s: gate %r needs %d qubits" % (where, g, GATE_ARITY[name]))
    if len(set(qs)) != len(qs):
        raise ValueError("%s: gate %r repeats a qubit" % (where, g))
    for q in qs:
        if not (isinstance(q, int) and 0 <= q < n_qubits):
            raise ValueError("%s: gate %r uses qubit %r outside 0..%d" % (where, g, q, n_qubits - 1))


def _gaps_overlap(a: Decision, b: Decision) -> bool:
    # Non-overlapping means one gap ends at or before the other starts:
    # a.recompute_before <= b.free_after or the reverse. A shared endpoint is
    # allowed, since one use of the ancilla can end one gap and start the next.
    return not (a.recompute_before <= b.free_after or b.recompute_before <= a.free_after)


def validate(problem: Problem) -> None:
    """Raise ValueError with a clear message if the Problem breaks any rule."""
    p = problem
    n = p.n_qubits
    if n < 1:
        raise ValueError("n_qubits must be positive, got %r" % (n,))

    # Structure: qubit sets.
    anc_qubits = [a.qubit for a in p.ancillas]
    if len(set(anc_qubits)) != len(anc_qubits):
        raise ValueError("two ancillas share a qubit: %r" % (anc_qubits,))
    if len(set(p.data_qubits)) != len(p.data_qubits):
        raise ValueError("data_qubits has duplicates: %r" % (p.data_qubits,))
    overlap = set(anc_qubits) & set(p.data_qubits)
    if overlap:
        raise ValueError("qubits %r are both data and ancilla" % (sorted(overlap),))
    for q in list(anc_qubits) + list(p.data_qubits):
        if not (0 <= q < n):
            raise ValueError("qubit %r outside 0..%d" % (q, n - 1))

    # Structure: gates.
    for i, g in enumerate(p.prep):
        _check_gate(g, n, "prep[%d]" % i)
    for i, g in enumerate(p.gates):
        _check_gate(g, n, "gates[%d]" % i)
    T = len(p.gates)

    # Structure: ancilla lifetimes.
    for a in p.ancillas:
        if not (0 <= a.alloc_step < a.release_step < T):
            raise ValueError("ancilla %d: need 0 <= alloc_step < release_step < %d, got %d, %d"
                             % (a.qubit, T, a.alloc_step, a.release_step))
        if target(p.gates[a.alloc_step]) != a.qubit:
            raise ValueError("ancilla %d: base gate %d (alloc_step) does not target it"
                             % (a.qubit, a.alloc_step))
        if target(p.gates[a.release_step]) != a.qubit:
            raise ValueError("ancilla %d: base gate %d (release_step) does not target it"
                             % (a.qubit, a.release_step))
        for t, g in enumerate(p.gates):
            if a.qubit in g[1] and not (a.alloc_step <= t <= a.release_step):
                raise ValueError("ancilla %d: base gate %d touches it outside its lifetime [%d, %d]"
                                 % (a.qubit, t, a.alloc_step, a.release_step))
        for i, g in enumerate(p.prep):
            if a.qubit in g[1]:
                raise ValueError("ancilla %d: prep[%d] touches it; ancillas must start in |0>"
                                 % (a.qubit, i))

    # Decisions.
    ids = [d.id for d in p.decisions]
    if len(set(ids)) != len(ids):
        raise ValueError("decision ids are not unique: %r" % (ids,))
    by_q = ancilla_by_qubit(p)
    for d in p.decisions:
        tag = "decision %r" % (d.id,)
        if d.ancilla not in by_q:
            raise ValueError("%s: qubit %d is not an ancilla" % (tag, d.ancilla))
        a = by_q[d.ancilla]
        for i, g in enumerate(d.uncompute):
            _check_gate(g, n, "%s uncompute[%d]" % (tag, i))
        for i, g in enumerate(d.recompute):
            _check_gate(g, n, "%s recompute[%d]" % (tag, i))

        # Rule 1.
        if not (a.alloc_step < d.free_after < d.recompute_before < a.release_step):
            raise ValueError("%s: rule 1 needs alloc_step < free_after < recompute_before < "
                             "release_step, got %d < %d < %d < %d"
                             % (tag, a.alloc_step, d.free_after, d.recompute_before, a.release_step))
        # Rule 2.
        for t, which in ((d.free_after, "free_after"), (d.recompute_before, "recompute_before")):
            if d.ancilla not in controls(p.gates[t]):
                raise ValueError("%s: rule 2: base gate %d (%s) = %r does not use ancilla %d as a control"
                                 % (tag, t, which, p.gates[t], d.ancilla))
        for t in gap_gates(p, d):
            if d.ancilla in p.gates[t][1]:
                raise ValueError("%s: rule 2: base gate %d = %r touches ancilla %d inside the gap"
                                 % (tag, t, p.gates[t], d.ancilla))
        # Rule 3.
        comp = compute_gates(p, a)
        if tuple(d.recompute) != comp:
            raise ValueError("%s: rule 3: recompute %r != the ancilla's compute gates %r"
                             % (tag, d.recompute, comp))
        if tuple(d.uncompute) != inverse_block(comp):
            raise ValueError("%s: rule 3: uncompute %r != inverse of compute gates %r"
                             % (tag, d.uncompute, inverse_block(comp)))
        # Rule 4.
        ctrl = set()
        for g in comp:
            ctrl.update(controls(g))
        for t in range(a.alloc_step + 1, d.recompute_before):
            if target(p.gates[t]) in ctrl:
                raise ValueError("%s: rule 4: base gate %d = %r targets qubit %d, a control of the "
                                 "ancilla's compute gates, before recompute_before"
                                 % (tag, t, p.gates[t], target(p.gates[t])))
    # Rule 5.
    for i, d1 in enumerate(p.decisions):
        for d2 in p.decisions[i + 1:]:
            if d1.ancilla == d2.ancilla and _gaps_overlap(d1, d2):
                raise ValueError("rule 5: decisions %r and %r on ancilla %d have overlapping gaps"
                                 % (d1.id, d2.id, d1.ancilla))
    # Rule 6.
    if len(p.target_state) != 0:
        if len(p.target_state) != 2 ** n:
            raise ValueError("rule 6: target_state has length %d, expected %d or 0"
                             % (len(p.target_state), 2 ** n))
        norm = math.sqrt(sum(abs(complex(c)) ** 2 for c in p.target_state))
        if abs(norm - 1.0) > 1e-8:
            raise ValueError("rule 6: target_state has norm %.12g, expected 1" % (norm,))


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def gap_gates(problem: Problem, decision: Decision) -> range:
    """Base-gate indices strictly inside the decision's gap."""
    return range(decision.free_after + 1, decision.recompute_before)


def _check_d(problem: Problem, d) -> Tuple[int, ...]:
    d = tuple(int(x) for x in d)
    if len(d) != len(problem.decisions):
        raise ValueError("decision vector has length %d, problem has %d decisions"
                         % (len(d), len(problem.decisions)))
    if any(x not in (0, 1) for x in d):
        raise ValueError("decision vector must be 0/1, got %r" % (d,))
    return d


def _tagged_schedule(problem: Problem, d) -> List[Tuple[Gate, int]]:
    """Discrete schedule as (gate, anchor_step) pairs. anchor_step is the base
    step whose live intervals the gate belongs to: the base gate's own index, or
    free_after for an inserted uncompute, recompute_before for a recompute.

    At a boundary between base gates t-1 and t, inserted uncomputes (free_after
    = t-1) come before inserted recomputes (recompute_before = t); several
    blocks at the same point keep the order of problem.decisions.
    """
    d = _check_d(problem, d)
    after: Dict[int, list] = {}
    before: Dict[int, list] = {}
    for di, dec in zip(d, problem.decisions):
        if di:
            after.setdefault(dec.free_after, []).append(dec)
            before.setdefault(dec.recompute_before, []).append(dec)
    out: List[Tuple[Gate, int]] = []
    for t, g in enumerate(problem.gates):
        for dec in before.get(t, []):
            out.extend((rg, t) for rg in dec.recompute)
        out.append((g, t))
        for dec in after.get(t, []):
            out.extend((ug, t) for ug in dec.uncompute)
    return out


def build_schedule(problem: Problem, d) -> List[Gate]:
    """Discrete gate list (logical qubits) for decision vector d: both blocks
    of every decision with d_i = 1 are inserted."""
    return [g for g, _ in _tagged_schedule(problem, d)]


def ancilla_intervals(problem: Problem, d) -> List[Interval]:
    """Each ancilla's occupied intervals (ancilla, start, end) in base steps,
    inclusive, sorted by (start, ancilla). A held ancilla is one interval
    [alloc_step, release_step]; freeing a decision splits it into
    [.., free_after] and [recompute_before, ..]."""
    d = _check_d(problem, d)
    out: List[Interval] = []
    for a in problem.ancillas:
        cuts = sorted((dec.free_after, dec.recompute_before)
                      for di, dec in zip(d, problem.decisions) if di and dec.ancilla == a.qubit)
        start = a.alloc_step
        for f, r in cuts:
            out.append((a.qubit, start, f))
            start = r
        out.append((a.qubit, start, a.release_step))
    out.sort(key=lambda iv: (iv[1], iv[0]))
    return out


def assign_intervals(problem: Problem, d) -> Dict[Interval, int]:
    """Greedy interval colouring: intervals sorted by (start, ancilla qubit),
    each takes the lowest-numbered free physical qubit from the pool of ancilla
    qubits (ascending). A qubit is free for an interval starting at s if its
    previous interval ended at e < s. Greedy is optimal for intervals, so the
    number of physical ancilla qubits used equals the peak live count."""
    pool = sorted(a.qubit for a in problem.ancillas)
    busy_until = {q: -1 for q in pool}       # last step (inclusive) each physical qubit is busy
    out: Dict[Interval, int] = {}
    for iv in ancilla_intervals(problem, d):
        _, s, e = iv
        for q in pool:
            if busy_until[q] < s:
                out[iv] = q
                busy_until[q] = e
                break
        else:  # cannot happen: each ancilla has at most one live interval at a time
            raise RuntimeError("no free physical qubit for interval %r" % (iv,))
    return out


def physical_schedule(problem: Problem, d) -> List[Gate]:
    """build_schedule(problem, d) with every ancilla qubit moved onto its
    assigned physical qubit. Data qubits are unchanged. Physical ancilla qubits
    are drawn from the problem's ancilla qubits, so n_qubits is unchanged and an
    unused ancilla qubit simply stays in |0>."""
    d = _check_d(problem, d)
    assignment = assign_intervals(problem, d)
    by_anc: Dict[int, List[Tuple[int, int, int]]] = {}
    for (q, s, e), phys in assignment.items():
        by_anc.setdefault(q, []).append((s, e, phys))

    def phys_of(q: int, t: int) -> int:
        if q not in by_anc:
            return q
        for s, e, phys in by_anc[q]:
            if s <= t <= e:
                return phys
        raise ValueError("ancilla %d is touched at base step %d, outside its live intervals %r"
                         % (q, t, by_anc[q]))

    return [(name, tuple(phys_of(q, t) for q in qs)) for (name, qs), t in _tagged_schedule(problem, d)]


def layers(gates: Sequence[Gate]) -> List[List[int]]:
    """ASAP layering: each gate goes in the earliest layer after the last layer
    that used any of its qubits. Returns, per layer, the indices into `gates`."""
    last: Dict[int, int] = {}
    out: List[List[int]] = []
    for i, (_, qs) in enumerate(gates):
        layer = 1 + max((last.get(q, -1) for q in qs), default=-1)
        if layer == len(out):
            out.append([])
        out[layer].append(i)
        for q in qs:
            last[q] = layer
    return out
