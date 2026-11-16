"""Workstream B: rounding, exact metrics, baselines, brute force."""
import dataclasses

import numpy as np
import pytest

import baselines as B
import circuits
import evaluate as E
import problem as P

TOL = 1e-10


@pytest.fixture(scope="module")
def toy():
    return circuits.toy()


@pytest.fixture(scope="module")
def bench2():
    return circuits.bench2()


def test_round_decisions():
    assert E.round_decisions([0.2, 0.5, 0.51, 0.99]) == (0, 0, 1, 1)
    assert E.round_decisions(np.array([0.7])) == (1,)


# --- toy: the hand-computed expected-values table -----------------------------

TOY_TABLE = {
    (0,): dict(live_counts=[1, 1, 2, 2, 2, 1, 1], peak_ancillas=2, physical_qubits=8,
               depth=7, extra_2q=0),
    (1,): dict(live_counts=[1, 1, 1, 1, 1, 1, 1], peak_ancillas=1, physical_qubits=7,
               depth=9, extra_2q=12),
}


@pytest.mark.parametrize("d", [(0,), (1,)])
def test_toy_table(toy, d):
    row = E.evaluate(toy, d)
    for key, want in TOY_TABLE[d].items():
        assert row[key] == want, (key, row[key], want)
    assert abs(row["fidelity"] - 1.0) < TOL
    assert E.live_counts(toy, d) == TOY_TABLE[d]["live_counts"]
    assert E.peak_ancillas(toy, d) == TOY_TABLE[d]["peak_ancillas"]
    assert E.depth(toy, d) == TOY_TABLE[d]["depth"]


def test_toy_two_qubit_equiv_totals(toy):
    # 6 Toffolis + 1 CNOT in the base list; freeing adds 2 Toffolis.
    assert E.two_qubit_equiv(toy, (0,)) == 37
    assert E.two_qubit_equiv(toy, (1,)) == 49


def test_toy_live_intervals_and_assignment(toy):
    assert E.live_intervals(toy, (0,)) == [(4, 0, 6), (5, 2, 4)]
    assert E.live_intervals(toy, (1,)) == [(4, 0, 1), (5, 2, 4), (4, 5, 6)]
    hold = E.assign_qubits(toy, (0,))
    assert hold["mapping"] == {(4, 0, 6): 4, (5, 2, 4): 5}
    free = E.assign_qubits(toy, (1,))
    # b takes a's physical qubit when a is freed.
    assert free["mapping"] == {(4, 0, 1): 4, (5, 2, 4): 4, (4, 5, 6): 4}
    assert free["ancilla_qubits_used"] == 1 == E.peak_ancillas(toy, (1,))


def test_toy_fill_costs(toy):
    assert toy.decisions[0].extra_depth == -1
    filled = E.fill_costs(toy)
    assert filled.decisions[0].extra_depth == 2
    assert filled.decisions[0].extra_2q == 12
    P.validate(filled)


def test_toy_fidelity_with_simulator(toy):
    for d in [(0,), (1,)]:
        F = E.fidelity(toy, d, None, simulator=circuits.reference_run_discrete)
        assert abs(F - 1) < TOL


def test_noisy_fidelity_needs_simulator(toy):
    with pytest.raises(ValueError, match="simulator"):
        E.fidelity(toy, (0,), circuits.RefNoise())
    assert abs(E.fidelity(toy, (0,), circuits.RefNoise.none()) - 1) < TOL


# --- baselines --------------------------------------------------------------------

def test_toy_baselines(toy):
    nr = B.no_reuse(toy)
    hr = B.hold_all_reassigned(toy)
    for row in (nr, hr):
        assert row["physical_qubits"] == 8 and row["depth"] == 7 and row["peak_ancillas"] == 2
        assert abs(row["fidelity"] - 1) < TOL


def test_brute_force_toy_noiseless(toy):
    assert B.brute_force(toy, None, lam_a=0.1, lam_d=0.01)["argmin"] == (1,)
    # The depth penalty 0.01 * 2 = 0.02 outweighs one qubit at lam_a = 0.01.
    assert B.brute_force(toy, None, lam_a=0.01, lam_d=0.01)["argmin"] == (0,)
    flips = B.flip_thresholds(B.schedule_table(toy), lam_d=0.01)
    assert len(flips) == 1 and flips[0]["from"] == (0,) and flips[0]["to"] == (1,)
    assert abs(flips[0]["lam_a"] - 0.02) < TOL


def test_brute_force_toy_noisy_thresholds(toy):
    """Sanity check against instructions.md: flips near 0.069, 0.088, 0.088."""
    for s, want in [(0.5, 0.069), (1, 0.088), (2, 0.088)]:
        table = B.schedule_table(toy, circuits.RefNoise().scaled(s), circuits.reference_run_discrete)
        flips = B.flip_thresholds(table, lam_d=0.01)
        assert len(flips) == 1
        assert abs(flips[0]["lam_a"] - want) < 1.5e-3, (s, flips)
        lo = B.brute_force(toy, None, flips[0]["lam_a"] - 1e-3, 0.01, table=table)
        hi = B.brute_force(toy, None, flips[0]["lam_a"] + 1e-3, 0.01, table=table)
        assert lo["argmin"] == (0,) and hi["argmin"] == (1,)


def test_brute_force_refuses_large_n(toy):
    decs = tuple(dataclasses.replace(toy.decisions[0], id="d%d" % i) for i in range(13))
    big = dataclasses.replace(toy, decisions=decs)
    with pytest.raises(ValueError, match="refuses"):
        B.brute_force(big, None, 0.1, 0.01)


# --- benchmark 2: hand-computed table, all four schedules ------------------------

# Base gates: 9 Toffolis + 2 CNOTs = 56 two-qubit equivalents.
# Hand-computed ASAP depths of the physical schedules: 9, 11, 13, 15. Gap 2 costs
# more layers than gap 1 because c's compute shares control c2 with a's compute.
BENCH2_TABLE = {
    (0, 0): dict(live_counts=[1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 1], peak_ancillas=2,
                 physical_qubits=8, depth=9, extra_depth=0, extra_2q=0,
                 assignment=[[4, 0, 10, 4], [5, 2, 4, 5], [6, 6, 8, 5]]),
    (1, 0): dict(live_counts=[1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1], peak_ancillas=2,
                 physical_qubits=8, depth=11, extra_depth=2, extra_2q=12,
                 assignment=[[4, 0, 1, 4], [4, 5, 10, 4], [5, 2, 4, 4], [6, 6, 8, 5]]),
    (0, 1): dict(live_counts=[1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1], peak_ancillas=2,
                 physical_qubits=8, depth=13, extra_depth=4, extra_2q=12,
                 assignment=[[4, 0, 5, 4], [4, 9, 10, 4], [5, 2, 4, 5], [6, 6, 8, 4]]),
    (1, 1): dict(live_counts=[1] * 11, peak_ancillas=1,
                 physical_qubits=7, depth=15, extra_depth=6, extra_2q=24,
                 assignment=[[4, 0, 1, 4], [4, 5, 5, 4], [4, 9, 10, 4], [5, 2, 4, 4], [6, 6, 8, 4]]),
}


@pytest.mark.parametrize("d", list(BENCH2_TABLE))
def test_bench2_table(bench2, d):
    row = E.evaluate(bench2, d)
    for key, want in BENCH2_TABLE[d].items():
        assert row[key] == want, (d, key, row[key], want)
    assert row["two_qubit_equiv"] == 56 + BENCH2_TABLE[d]["extra_2q"]
    assert abs(row["fidelity"] - 1.0) < TOL


def test_bench2_logical_depths(bench2):
    # Logical (unreassigned) schedules: gap 1's uncompute of a runs beside b's
    # compute (as in the toy), so it costs no layers; gap 2's blocks share
    # control c2 with c's compute and serialise, costing 4 layers either way.
    got = [len(P.layers(P.build_schedule(bench2, d))) for d in BENCH2_TABLE]
    assert got == [9, 9, 13, 13]


def test_bench2_fill_costs(bench2):
    filled = E.fill_costs(bench2)
    assert [(dc.extra_depth, dc.extra_2q) for dc in filled.decisions] == [(2, 12), (4, 12)]


def test_bench2_baselines(bench2):
    nr, hr = B.no_reuse(bench2), B.hold_all_reassigned(bench2)
    assert nr["physical_qubits"] == 9 and hr["physical_qubits"] == 8
    assert nr["depth"] == hr["depth"] == 9
    assert abs(nr["fidelity"] - 1) < TOL and abs(hr["fidelity"] - 1) < TOL
    # With the reference simulator, no_reuse runs the base list with b and c on
    # separate qubits; noiseless it is still exact.
    F = B.no_reuse(bench2, None, circuits.reference_run_discrete)["fidelity"]
    assert abs(F - 1) < TOL


def test_bench2_brute_force_noiseless(bench2):
    # Freeing only one gap never lowers the peak, so the only flip is straight
    # from (0, 0) to (1, 1), at lam_a = lam_d * 6 = 0.06.
    table = B.schedule_table(bench2)
    flips = B.flip_thresholds(table, lam_d=0.01)
    assert [(f["from"], f["to"]) for f in flips] == [((0, 0), (1, 1))]
    assert abs(flips[0]["lam_a"] - 0.06) < TOL
    assert B.brute_force(bench2, None, 0.05, 0.01, table=table)["argmin"] == (0, 0)
    assert B.brute_force(bench2, None, 0.07, 0.01, table=table)["argmin"] == (1, 1)
