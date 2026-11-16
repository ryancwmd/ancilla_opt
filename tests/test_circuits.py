"""Workstream B: data model, benchmark circuits, numpy reference simulators."""
import ast
import dataclasses
import math
import pathlib

import numpy as np
import pytest

import circuits
import problem as P
from problem import Ancilla, Decision

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOL = 1e-10


@pytest.fixture(scope="module")
def toy():
    return circuits.toy()


@pytest.fixture(scope="module")
def bench2():
    return circuits.bench2()


# --- problem.py contract ------------------------------------------------------

def test_problem_py_is_stdlib_only():
    tree = ast.parse((ROOT / "problem.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module.split(".")[0])
    assert mods <= {"__future__", "json", "math", "dataclasses", "typing"}, mods
    assert (ROOT / "problem.py").read_text().lstrip().split("\n")[0].startswith('"""')
    assert "from __future__ import annotations" in (ROOT / "problem.py").read_text()


@pytest.mark.parametrize("name", ["toy", "bench2"])
def test_validates_and_json_round_trip(name, tmp_path):
    p = circuits.BENCHMARKS[name]()
    P.validate(p)
    path = tmp_path / ("%s.json" % name)
    P.to_json(p, path)
    q = P.from_json(path)
    assert q == p
    P.validate(q)


def test_toy_matches_spec(toy):
    assert toy.n_qubits == 8
    assert toy.data_qubits == (0, 1, 2, 3, 6, 7)
    assert toy.ancillas == (Ancilla(4, 0, 6), Ancilla(5, 2, 4))
    assert toy.prep == tuple(("h", (q,)) for q in range(4))
    assert toy.decisions == (Decision(id="a_gap", ancilla=4, free_after=1, recompute_before=5,
                                      uncompute=(("ccx", (0, 1, 4)),),
                                      recompute=(("ccx", (0, 1, 4)),)),)
    assert list(P.gap_gates(toy, toy.decisions[0])) == [2, 3, 4]


def test_empty_target_state_is_valid(toy):
    P.validate(dataclasses.replace(toy, target_state=()))


# --- one broken Problem per validity rule ---------------------------------

def _with_decision(p, **kw):
    return dataclasses.replace(p, decisions=(dataclasses.replace(p.decisions[0], **kw),))


def _with_gate(p, i, g):
    gates = list(p.gates)
    gates[i] = g
    return dataclasses.replace(p, gates=tuple(gates))


def test_rule1_ordering(toy):
    with pytest.raises(ValueError, match="rule 1"):
        P.validate(_with_decision(toy, free_after=0))
    with pytest.raises(ValueError, match="rule 1"):
        P.validate(_with_decision(toy, free_after=5, recompute_before=5))
    with pytest.raises(ValueError, match="rule 1"):
        P.validate(_with_decision(toy, recompute_before=6))


def test_rule2_not_a_control_at_boundary(toy):
    with pytest.raises(ValueError, match="rule 2"):
        P.validate(_with_decision(toy, free_after=2))


def test_rule2_touched_inside_gap(toy):
    with pytest.raises(ValueError, match="rule 2"):
        P.validate(_with_gate(toy, 3, ("cx", (4, 7))))


def test_rule3_wrong_blocks(toy):
    with pytest.raises(ValueError, match="rule 3"):
        P.validate(_with_decision(toy, recompute=(("ccx", (0, 2, 4)),)))
    with pytest.raises(ValueError, match="rule 3"):
        P.validate(_with_decision(toy, uncompute=(("ccx", (1, 0, 4)),)))
    with pytest.raises(ValueError, match="rule 3"):
        P.validate(_with_decision(toy, uncompute=()))


def test_rule3_multi_gate_block_order():
    # Compute = x(4) then ccx(0,1,4): uncompute must be the reverse order.
    gates = (("x", (4,)), ("ccx", (0, 1, 4)), ("cx", (4, 2)), ("x", (3,)), ("cx", (4, 3)),
             ("ccx", (0, 1, 4)), ("x", (4,)))
    comp = (("x", (4,)), ("ccx", (0, 1, 4)))
    dec = Decision("g", 4, 2, 4, uncompute=tuple(reversed(comp)), recompute=comp)
    p = P.Problem("multi", 5, (0, 1, 2, 3), (Ancilla(4, 0, 6),), (), gates, (dec,), ())
    P.validate(p)
    with pytest.raises(ValueError, match="rule 3"):
        P.validate(_with_decision(p, uncompute=comp))


def test_rule4_control_overwritten(toy):
    with pytest.raises(ValueError, match="rule 4"):
        P.validate(_with_gate(toy, 3, ("cx", (5, 0))))


def test_rule5_overlapping_gaps(toy):
    d2 = dataclasses.replace(toy.decisions[0], id="a_gap_again")
    with pytest.raises(ValueError, match="rule 5"):
        P.validate(dataclasses.replace(toy, decisions=toy.decisions + (d2,)))


def test_rule5_shared_endpoint_is_allowed(bench2):
    d1, d2 = bench2.decisions
    assert d1.recompute_before == d2.free_after == 5
    P.validate(bench2)


def test_rule6_target_norm(toy):
    bad = tuple(2 * c for c in toy.target_state)
    with pytest.raises(ValueError, match="rule 6"):
        P.validate(dataclasses.replace(toy, target_state=bad))
    with pytest.raises(ValueError, match="rule 6"):
        P.validate(dataclasses.replace(toy, target_state=toy.target_state[:-1]))


def test_structural_checks(toy):
    with pytest.raises(ValueError, match="unknown gate"):
        P.validate(_with_gate(toy, 3, ("swap", (5, 7))))
    with pytest.raises(ValueError, match="needs 2 qubits"):
        P.validate(_with_gate(toy, 3, ("cx", (5, 7, 6))))
    with pytest.raises(ValueError, match="outside"):
        P.validate(_with_gate(toy, 3, ("cx", (5, 8))))
    with pytest.raises(ValueError, match="does not target it"):
        P.validate(dataclasses.replace(toy, ancillas=(Ancilla(4, 0, 6), Ancilla(5, 2, 3))))
    with pytest.raises(ValueError, match="outside its lifetime"):
        P.validate(_with_gate(toy, 1, ("ccx", (4, 5, 6))))
    with pytest.raises(ValueError, match="not an ancilla"):
        P.validate(_with_decision(toy, ancilla=6))


# --- reference simulator and target -------------------------------------------

def test_gate_matrices_msb_convention():
    # x on qubit 0 of 2 qubits maps |00> (index 0) to |10> (index 2).
    psi = circuits.run_statevector(2, [("x", (0,))])
    assert np.argmax(abs(psi)) == 2
    # cx(0, 1) on |10> gives |11>; cx(1, 0) on |10> leaves it.
    assert np.argmax(abs(circuits.run_statevector(2, [("x", (0,)), ("cx", (0, 1))]))) == 3
    assert np.argmax(abs(circuits.run_statevector(2, [("x", (0,)), ("cx", (1, 0))]))) == 2
    # ccx(2, 0, 1) on 3 qubits fires only when qubits 2 and 0 are set.
    psi = circuits.run_statevector(3, [("x", (0,)), ("x", (2,)), ("ccx", (2, 0, 1))])
    assert np.argmax(abs(psi)) == 0b111
    for name, U in circuits.GATES.items():
        assert np.allclose(U @ U.conj().T, np.eye(U.shape[0]), atol=TOL), name


@pytest.mark.parametrize("name", ["toy", "bench2"])
def test_target_state_has_unit_norm(name):
    p = circuits.BENCHMARKS[name]()
    psi = np.array(p.target_state)
    assert psi.shape == (2 ** p.n_qubits,)
    assert abs(np.linalg.norm(psi) - 1) < TOL


def test_toy_target_state_content(toy):
    # Ancillas end in |0>; t1 = c3 AND c1 AND c2, t2 = (c3 AND c4) XOR (c1 AND c2 AND c4).
    psi = np.array(toy.target_state)
    nz = np.flatnonzero(abs(psi) > 1e-12)
    assert len(nz) == 16
    for idx in nz:
        bits = [(idx >> (7 - q)) & 1 for q in range(8)]
        c1, c2, c3, c4, a, b, t1, t2 = bits
        assert a == 0 and b == 0
        assert t1 == (c1 & c2 & c3)
        assert t2 == ((c3 & c4) ^ (c1 & c2 & c4))
        assert abs(psi[idx] - 0.25) < TOL


def test_fill_target_matches_benchmark(toy):
    assert circuits.fill_target(dataclasses.replace(toy, target_state=())) == toy


@pytest.mark.parametrize("name", ["toy", "bench2"])
def test_every_schedule_is_noiseless_exact(name):
    import evaluate as E
    p = circuits.BENCHMARKS[name]()
    psi = np.array(p.target_state)
    for d in E.all_schedules(p):
        for physical in (False, True):
            phi = circuits.reference_state(p, d, physical=physical)
            assert abs(abs(np.vdot(psi, phi)) ** 2 - 1) < TOL
        rho = circuits.reference_run_discrete(p, d, None)
        assert abs(np.real(psi.conj() @ rho @ psi) - 1) < TOL
        rho = circuits.reference_run_discrete(p, d, circuits.RefNoise.none())
        assert abs(np.real(psi.conj() @ rho @ psi) - 1) < TOL


# --- layering and physical schedule (toy) -----------------------------------------

def test_toy_layers(toy):
    hold, free = P.build_schedule(toy, (0,)), P.build_schedule(toy, (1,))
    assert len(P.layers(hold)) == 7
    assert len(P.layers(free)) == 7
    assert free[2] == ("ccx", (0, 1, 4)) and free[6] == ("ccx", (0, 1, 4))
    lay = P.layers(free)
    layer_of = {g: i for i, L in enumerate(lay) for g in L}
    assert layer_of[2] == layer_of[3]           # inserted uncompute beside base step 2
    assert len(P.layers(P.physical_schedule(toy, (0,)))) == 7
    assert len(P.layers(P.physical_schedule(toy, (1,)))) == 9


def test_toy_physical_schedule(toy):
    assert P.physical_schedule(toy, (0,)) == list(toy.gates)
    assert P.physical_schedule(toy, (1,)) == [
        ("ccx", (0, 1, 4)), ("ccx", (4, 2, 6)), ("ccx", (0, 1, 4)),
        ("ccx", (2, 3, 4)), ("cx", (4, 7)), ("ccx", (2, 3, 4)),
        ("ccx", (0, 1, 4)), ("ccx", (4, 3, 7)), ("ccx", (0, 1, 4))]


def test_layers_basic():
    assert P.layers([]) == []
    assert P.layers([("x", (0,)), ("x", (1,)), ("cx", (0, 1)), ("x", (2,))]) == [[0, 1, 3], [2]]


def test_physical_schedule_ignores_non_ancillas(toy):
    stripped = dataclasses.replace(toy, ancillas=(), decisions=())
    assert P.physical_schedule(stripped, ()) == list(toy.gates)


def test_bad_decision_vector(toy):
    with pytest.raises(ValueError):
        P.build_schedule(toy, (0, 1))
    with pytest.raises(ValueError):
        P.build_schedule(toy, (2,))


# --- noisy density-matrix reference ------------------------------------------

def _random_rho(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2 ** n, 2 ** n)) + 1j * rng.normal(size=(2 ** n, 2 ** n))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def test_idle_update_equals_kraus():
    n, g, lam = 3, 0.13, 0.29
    rho = _random_rho(n, seed=0).reshape((2,) * 6)
    ops = circuits.kraus_ops(g, lam)
    for q in range(n):
        want = circuits.apply_kraus_dm(circuits.apply_kraus_dm(rho, ops["amp_damp"], q, n),
                                       ops["phase_damp"], q, n)
        got = circuits._idle(rho, q, n, g, lam)
        assert np.allclose(got, want, atol=TOL)
    for name, K in ops.items():
        assert np.allclose(sum(k.conj().T @ k for k in K), np.eye(2), atol=TOL), name


def test_full_depolarize_equals_pauli_twirl():
    n = 3
    rho = _random_rho(n, seed=1).reshape((2,) * 6)
    paulis = [circuits._I2, circuits._X, circuits._Y, circuits._Z]
    for qs in [(1,), (2, 0), (0, 1, 2)]:
        want = np.zeros_like(rho)
        import itertools
        for ps in itertools.product(paulis, repeat=len(qs)):
            U = ps[0]
            for p in ps[1:]:
                U = np.kron(U, p)
            want = want + circuits._apply_op_dm(rho, U, list(qs), n)
        want = want / 4 ** len(qs)
        assert np.allclose(circuits._full_depolarize(rho, qs, n), want, atol=TOL)
    # p = 1 on one qubit of a 1-qubit state gives I/2.
    one = np.array([[1, 0], [0, 0]], dtype=complex)
    assert np.allclose(circuits._full_depolarize(one, (0,), 1), np.eye(2) / 2, atol=TOL)


def test_noise_rates_defaults():
    r = circuits.noise_rates(circuits.RefNoise())
    assert abs(r["p"][3] - (1 - 0.99 ** 6)) < TOL
    assert abs(r["gamma"] - (1 - math.exp(-3 / 200))) < TOL
    assert abs(r["lambda"] - (1 - math.exp(-6 / 240))) < TOL     # T_phi = 240 us
    z = circuits.noise_rates(circuits.RefNoise.none())
    assert z["gamma"] == 0 and z["lambda"] == 0 and z["p"] == {1: 0, 2: 0, 3: 0}
    assert circuits.RefNoise().scaled(0) == circuits.RefNoise.none()


def test_noisy_rho_is_physical(toy):
    rho = circuits.reference_run_discrete(toy, (1,), circuits.RefNoise().scaled(2))
    assert abs(np.trace(rho) - 1) < TOL
    assert np.allclose(rho, rho.conj().T, atol=TOL)
    assert np.linalg.eigvalsh(rho).min() > -1e-10


def test_toy_reference_numbers(toy):
    """Sanity check against instructions.md's reference numbers (3 decimals)."""
    psi = np.array(toy.target_state)
    want = {0.5: (0.734, 0.685), 1: (0.543, 0.475), 2: (0.306, 0.238)}
    for s, (fh, ff) in want.items():
        noise = circuits.RefNoise().scaled(s)
        for d, f in (((0,), fh), ((1,), ff)):
            F = np.real(psi.conj() @ circuits.reference_run_discrete(toy, d, noise) @ psi)
            assert abs(F - f) < 1e-3, (s, d, F)
