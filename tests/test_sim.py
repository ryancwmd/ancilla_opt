"""Tests for sim.py (workstream A)."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import problem as P
import sim
from noise import NoiseParams
from toy_fixture import toy_problem, toy_from_spec, two_gap_problem

TOL = 1e-10


def random_unitary(dim, rng):
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(a)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def random_rho(n, rng, rank=3):
    a = rng.normal(size=(2 ** n, rank)) + 1j * rng.normal(size=(2 ** n, rank))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def embed(U, qs, n):
    """Full 2^n x 2^n matrix of U on qubits qs, by Kronecker product + axis permutation."""
    k = len(qs)
    order = list(qs) + [q for q in range(n) if q not in qs]
    M = np.kron(U, np.eye(2 ** (n - k))).reshape((2,) * (2 * n))
    perm = [order.index(j) for j in range(n)]
    M = M.transpose(perm + [n + p for p in perm])
    return M.reshape(2 ** n, 2 ** n)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qs", [(0,), (3,), (1, 2), (3, 0), (2, 1), (0, 2, 3), (3, 1, 0), (2, 0, 1)])
def test_apply_op_matches_full_matrix(qs):
    rng = np.random.default_rng(hash(qs) % 2 ** 32)
    n = 4
    U = random_unitary(2 ** len(qs), rng)
    rho = random_rho(n, rng)
    full = embed(U, qs, n)
    expect = full @ rho @ full.conj().T
    got = np.asarray(sim.apply_op(jnp.asarray(rho), U, qs, n))
    assert np.max(np.abs(got - expect)) < TOL


def test_gate_matrices_msb_convention():
    # cx on (0, 1): |10> -> |11>; with qubit 0 most significant, index 2 -> 3
    psi = sim.statevector([("x", (0,)), ("cx", (0, 1))], 2)
    assert abs(psi[3] - 1) < TOL
    psi = sim.statevector([("x", (0,)), ("x", (1,)), ("ccx", (0, 1, 2))], 3)
    assert abs(psi[7] - 1) < TOL
    # reversed control/target
    psi = sim.statevector([("x", (2,)), ("cx", (2, 0))], 3)
    assert abs(psi[0b101] - 1) < TOL
    # density-matrix sim agrees with statevector
    rho = sim.zero_state(3)
    for g in [("x", (2,)), ("h", (1,)), ("cx", (2, 0)), ("ccx", (0, 1, 2))]:
        rho = sim.apply_gate(rho, g, 3)
    psi = sim.statevector([("x", (2,)), ("h", (1,)), ("cx", (2, 0)), ("ccx", (0, 1, 2))], 3)
    assert np.max(np.abs(np.asarray(rho) - np.outer(psi, psi.conj()))) < TOL


def test_random_circuit_with_noise_is_valid_density_matrix():
    rng = np.random.default_rng(1)
    n = 4
    rho = sim.zero_state(n)
    nz = NoiseParams(p1=0.05, p2=0.1, p3=0.2, T1_us=20.0, T2_us=15.0)
    idle = sim._idle_kraus(nz)
    for _ in range(30):
        k = int(rng.integers(1, 4))
        qs = tuple(int(q) for q in rng.choice(n, size=k, replace=False))
        rho = sim.apply_op(rho, random_unitary(2 ** k, rng), qs, n)
        rho = sim.apply_depolarizing(rho, nz.p_gate(k), qs, n)
        rho = sim.apply_kraus(rho, idle, (int(rng.integers(n)),), n)
    rho = np.asarray(rho)
    assert abs(np.trace(rho) - 1) < TOL
    assert np.max(np.abs(rho - rho.conj().T)) < TOL
    assert np.linalg.eigvalsh(rho).min() > -1e-10


def test_matches_pennylane_default_mixed():
    qml = pytest.importorskip("pennylane")
    rng = np.random.default_rng(7)
    n = 4
    p_dep, gamma, lam = 0.08, 0.12, 0.2
    ops = []
    for _ in range(14):
        kind = rng.choice(["u1", "u2", "cx", "ccx", "h", "dep", "ad", "pd"])
        if kind in ("u1", "u2"):
            k = 1 if kind == "u1" else 2
            qs = tuple(int(q) for q in rng.choice(n, size=k, replace=False))
            ops.append(("U", random_unitary(2 ** k, rng), qs))
        elif kind == "cx":
            ops.append(("G", "cx", tuple(int(q) for q in rng.choice(n, 2, replace=False))))
        elif kind == "ccx":
            ops.append(("G", "ccx", tuple(int(q) for q in rng.choice(n, 3, replace=False))))
        elif kind == "h":
            ops.append(("G", "h", (int(rng.integers(n)),)))
        else:
            ops.append((kind, None, (int(rng.integers(n)),)))

    dev = qml.device("default.mixed", wires=n)

    @qml.qnode(dev)
    def circuit():
        for kind, a, qs in ops:
            if kind == "U":
                qml.QubitUnitary(a, wires=list(qs))
            elif kind == "G":
                {"cx": qml.CNOT, "ccx": qml.Toffoli, "h": qml.Hadamard}[a](wires=list(qs))
            elif kind == "dep":
                qml.DepolarizingChannel(3 * p_dep / 4, wires=qs[0])  # PL puts p/3 on each Pauli
            elif kind == "ad":
                qml.AmplitudeDamping(gamma, wires=qs[0])
            elif kind == "pd":
                qml.PhaseDamping(lam, wires=qs[0])
        return qml.density_matrix(wires=list(range(n)))

    expect = np.asarray(circuit())

    import noise as N
    rho = sim.zero_state(n)
    for kind, a, qs in ops:
        if kind == "U":
            rho = sim.apply_op(rho, a, qs, n)
        elif kind == "G":
            rho = sim.apply_gate(rho, (a, qs), n)
        elif kind == "dep":
            rho = sim.apply_depolarizing(rho, p_dep, qs, n)
        elif kind == "ad":
            rho = sim.apply_kraus(rho, N.amplitude_damping_kraus(gamma), qs, n)
        elif kind == "pd":
            rho = sim.apply_kraus(rho, N.phase_damping_kraus(lam), qs, n)
    assert np.max(np.abs(np.asarray(rho) - expect)) < 1e-10


# ---------------------------------------------------------------------------
# Toy circuit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy():
    p = toy_problem()
    P.validate(p)
    return p


@pytest.mark.parametrize("d", [(0,), (1,)])
def test_toy_noiseless_fidelity_one(toy, d):
    rho = sim.run_discrete(toy, d, NoiseParams.none())
    psi = sim.target_vector(toy)
    assert abs(float(sim.fidelity(rho, psi)) - 1.0) < TOL
    assert np.max(np.abs(np.asarray(rho) - np.outer(psi, psi.conj()))) < TOL


def test_toy_target_matches_spec_reference(toy):
    # B's target_state (from circuits.py's numpy reference when available)
    # equals A's independent statevector of prep + base gates.
    ref = np.array(toy_from_spec().target_state)
    assert np.max(np.abs(np.array(toy.target_state) - ref)) < TOL


def test_toy_matches_b_reference_state():
    circuits = pytest.importorskip("circuits")
    p = circuits.toy()
    psi = np.array(p.target_state)
    for d in [(0,), (1,)]:
        rho = np.asarray(sim.run_discrete(p, d, NoiseParams.none()))
        assert np.max(np.abs(rho - np.outer(psi, psi.conj()))) < TOL


@pytest.mark.parametrize("s,f_hold,f_free", [(0.5, 0.734, 0.685), (1.0, 0.543, 0.475), (2.0, 0.306, 0.238)])
def test_toy_reference_numbers(toy, s, f_hold, f_free):
    # Sanity check against the spec's reference run (3 digits), not exact values.
    F = sim.schedule_fidelities(toy, NoiseParams().scaled(s))
    assert abs(F[0] - f_hold) < 1e-3 and abs(F[1] - f_free) < 1e-3


@pytest.mark.parametrize("r", [0.2, 0.5, 0.8])
def test_toy_soft_fidelity_linear_in_r(toy, r):
    nz = NoiseParams()
    psi = sim.target_vector(toy)
    F_hold = float(sim.fidelity(sim.run_discrete(toy, (0,), nz), psi))
    F_free = float(sim.fidelity(sim.run_discrete(toy, (1,), nz), psi))
    F_soft = float(sim.fidelity(sim.simulate_mixture(toy, jnp.array([r]), nz), psi))
    assert abs(F_soft - (r * F_free + (1 - r) * F_hold)) < TOL
    assert F_hold > F_free  # noisy toy: holding is better at equal qubit cost


def test_mixture_is_differentiable(toy):
    nz = NoiseParams()
    psi = sim.target_vector(toy)
    g = jax.grad(lambda r: sim.fidelity(sim.simulate_mixture_exact(toy, r, nz), psi))(jnp.array([0.3]))
    F = sim.schedule_fidelities(toy, nz)
    assert abs(float(g[0]) - (F[1] - F[0])) < TOL


def test_schedule_weights_two_decisions():
    w = np.asarray(sim.schedule_weights(jnp.array([0.2, 0.7])))
    # order (0,0), (0,1), (1,0), (1,1)
    np.testing.assert_allclose(w, [0.8 * 0.3, 0.8 * 0.7, 0.2 * 0.3, 0.2 * 0.7], atol=TOL)


def test_two_decision_mixture_and_noiseless_fidelity():
    p = two_gap_problem()
    P.validate(p)
    F0 = sim.schedule_fidelities(p, NoiseParams.none())
    np.testing.assert_allclose(F0, 1.0, atol=TOL)
    nz = NoiseParams()
    F = sim.schedule_fidelities(p, nz)
    r = jnp.array([0.3, 0.6])
    w = np.asarray(sim.schedule_weights(r))
    F_soft = float(sim.fidelity(sim.simulate_mixture(p, r, nz), sim.target_vector(p)))
    assert abs(F_soft - w @ F) < TOL


def test_run_discrete_without_ancillas(toy):
    # B's no_reuse baseline runs a Problem with no ancillas and no decisions.
    bare = P.Problem(name="bare", n_qubits=toy.n_qubits, data_qubits=tuple(range(toy.n_qubits)),
                     ancillas=(), prep=toy.prep, gates=toy.gates, decisions=(),
                     target_state=toy.target_state)
    rho = sim.run_discrete(bare, (), NoiseParams())
    F_bare = float(sim.fidelity(rho, sim.target_vector(bare)))
    F_hold = float(sim.fidelity(sim.run_discrete(toy, (0,), NoiseParams()), sim.target_vector(toy)))
    assert abs(F_bare - F_hold) < TOL  # toy hold schedule needs no reassignment
    assert abs(float(sim.fidelity(sim.run_discrete(bare, ()), sim.target_vector(bare))) - 1) < TOL


def test_target_vector_fills_empty_target(toy):
    empty = P.Problem(**{**toy.__dict__, "target_state": ()})
    np.testing.assert_allclose(np.asarray(sim.target_vector(empty)), np.array(toy.target_state), atol=TOL)
