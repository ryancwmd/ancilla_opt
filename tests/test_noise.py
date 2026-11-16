"""Tests for noise.py and the channel application in sim.py (workstream A)."""
import math

import numpy as np
import pytest
import jax.numpy as jnp

import noise as N
import sim
from noise import NoiseParams

TOL = 1e-10


def _kraus_sum(ops):
    d = ops[0].shape[0]
    return sum(K.conj().T @ K for K in ops), np.eye(d)


@pytest.mark.parametrize("ops", [
    N.amplitude_damping_kraus(0.3), N.phase_damping_kraus(0.4),
    N.amplitude_damping_kraus(1.0), N.phase_damping_kraus(0.0),
    N.depolarizing_kraus(0.1, 1), N.depolarizing_kraus(0.2, 2), N.depolarizing_kraus(0.059, 3),
    N.depolarizing_kraus(1.0, 1),
    sim._idle_kraus(NoiseParams()), sim._idle_kraus(NoiseParams().scaled(5)),
], ids=["ad", "pd", "ad1", "pd0", "dep1", "dep2", "dep3", "dep1full", "idle", "idle5"])
def test_trace_preserving(ops):
    s, eye = _kraus_sum(ops)
    assert np.max(np.abs(s - eye)) < TOL


def test_depolarizing_kraus_count():
    assert len(N.depolarizing_kraus(0.1, 3)) == 64


@pytest.mark.parametrize("qs", [(1,), (2, 0), (3, 1, 2)])
def test_depolarizing_trace_form_matches_kraus(qs):
    rng = np.random.default_rng(3)
    n = 4
    a = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    p = 0.37
    fast = np.asarray(sim.apply_depolarizing(jnp.asarray(rho), p, qs, n))
    slow = np.asarray(sim.apply_kraus(jnp.asarray(rho), N.depolarizing_kraus(p, len(qs)), qs, n))
    assert np.max(np.abs(fast - slow)) < TOL


def test_damping_leaves_ground_state():
    rho0 = jnp.asarray(np.diag([1.0, 0.0]).astype(complex))
    for ops in (N.amplitude_damping_kraus(0.4), N.phase_damping_kraus(0.7),
                sim._idle_kraus(NoiseParams().scaled(3))):
        out = np.asarray(sim.apply_kraus(rho0, ops, (0,), 1))
        assert np.max(np.abs(out - np.asarray(rho0))) < TOL


def test_full_depolarizing_gives_maximally_mixed():
    rho = jnp.asarray(np.array([[0.3, 0.2 - 0.1j], [0.2 + 0.1j, 0.7]]))
    out = np.asarray(sim.apply_depolarizing(rho, 1.0, (0,), 1))
    assert np.max(np.abs(out - np.eye(2) / 2)) < TOL
    out = np.asarray(sim.apply_kraus(rho, N.depolarizing_kraus(1.0, 1), (0,), 1))
    assert np.max(np.abs(out - np.eye(2) / 2)) < TOL


def test_none_is_noiseless():
    nz = NoiseParams.none()
    assert nz.is_noiseless()
    assert nz.gamma == 0 and nz.lam == 0 and nz.p_gate(1) == nz.p_gate(2) == nz.p_gate(3) == 0
    assert sim._idle_kraus(nz) == []
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    rho = jnp.asarray(a @ a.conj().T / np.trace(a @ a.conj().T))
    out = rho
    for k, qs in ((1, (0,)), (2, (1, 2)), (3, (2, 0, 1))):
        out = sim.apply_depolarizing(out, nz.p_gate(k), qs, 3)
    assert np.max(np.abs(np.asarray(out - rho))) == 0
    # and through a full schedule: noiseless run == ideal unitary run
    from problem import Problem
    gates = (("h", (0,)), ("cx", (0, 1)), ("ccx", (0, 1, 2)), ("x", (1,)))
    p = Problem("t", 3, (0, 1, 2), (), (), gates, (), ())
    psi = sim.statevector(gates, 3)
    assert np.max(np.abs(np.asarray(sim.run_discrete(p, (), nz)) - np.outer(psi, psi.conj()))) < TOL


def test_default_strengths():
    nz = NoiseParams()
    assert abs(nz.gamma - (1 - math.exp(-3 / 200))) < 1e-15
    assert abs(nz.gamma - 0.015) < 2e-4
    assert abs(nz.T_phi_us - 240.0) < 1e-9
    assert abs(nz.lam - (1 - math.exp(-6 / 240))) < 1e-15
    assert abs(nz.lam - 0.025) < 5e-4
    assert abs(nz.p3_eff - (1 - 0.99 ** 6)) < 1e-15
    assert abs(nz.p3_eff - 0.0585) < 1e-3


def test_scaled():
    nz = NoiseParams()
    s2 = nz.scaled(2)
    assert s2.p1 == 2e-3 and s2.p2 == 2e-2 and s2.T1_us == 100 and s2.T2_us == 75
    assert s2.p3 is None  # p3 derived from the scaled p2
    assert NoiseParams(p3=0.05).scaled(2).p3 == 0.1
    assert nz.scaled(0) == NoiseParams.none()
    assert nz.scaled(1).gamma == nz.gamma


def test_t2_bound():
    with pytest.raises(ValueError):
        NoiseParams(T1_us=100, T2_us=250)
    NoiseParams(T1_us=100, T2_us=200)  # boundary ok
