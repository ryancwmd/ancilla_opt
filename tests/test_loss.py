"""Tests for loss.py and train.py (workstream A)."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import loss as L
import sim
import train as T
from noise import NoiseParams
from circuits import bench2, toy as toy_problem

TOL = 1e-10


@pytest.fixture(scope="module")
def toy():
    return toy_problem()


@pytest.mark.parametrize("r", [0.0, 0.3, 1.0])
def test_toy_live_counts(toy, r):
    x = np.asarray(L.live_counts(toy, jnp.array([r])))
    np.testing.assert_allclose(x, [1, 1, 2 - r, 2 - r, 2 - r, 1, 1], atol=TOL)


def test_bench2_live_counts():
    p = bench2()
    x = np.asarray(L.live_counts(p, jnp.array([0.25, 0.5])))
    # a: 0..10; b: 2..4 (gap1 = 2..4); c: 6..8 (gap2 = 6..8)
    np.testing.assert_allclose(x, [1, 1, 1.75, 1.75, 1.75, 1, 1.5, 1.5, 1.5, 1, 1], atol=TOL)


def test_smoothmax_bounds():
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = jnp.asarray(rng.uniform(0, 3, size=int(rng.integers(1, 12))))
        beta = float(rng.uniform(0.5, 50))
        sm = float(L.smooth_max(x, beta))
        assert float(x.max()) - 1e-12 <= sm <= float(x.max()) + np.log(x.shape[0]) / beta + 1e-12


def test_extra_depth_fallback(toy):
    # B's extra_depth not filled (-1) -> computed from physical schedules: 9 - 7 = 2
    from dataclasses import replace
    unfilled = replace(toy, decisions=tuple(replace(d, extra_depth=-1) for d in toy.decisions))
    np.testing.assert_allclose(L.extra_depths(unfilled), [2.0])
    np.testing.assert_allclose(L.extra_depths(toy), [2.0])


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("budget", [None, 1.5])
def test_grad_matches_finite_differences_toy(toy, noisy, budget):
    nz = NoiseParams() if noisy else NoiseParams.none()
    data = L.make_loss_data(toy, nz)
    f = lambda z: L.loss_terms(z, data, 0.1, 0.01, 5.0, budget)[0]
    g = jax.grad(f)
    h = 1e-5
    for z0 in (-1.3, 0.0, 0.7, 2.5):
        z = jnp.array([z0])
        fd = (float(f(z + h)) - float(f(z - h))) / (2 * h)
        assert abs(float(g(z)[0]) - fd) < 1e-6


def test_grad_matches_finite_differences_two_decisions():
    p = bench2()
    data = L.make_loss_data(p, NoiseParams())
    f = lambda z: L.loss_terms(z, data, 0.05, 0.01, 8.0)[0]
    z = jnp.array([0.4, -0.9])
    g = np.asarray(jax.grad(f)(z))
    h = 1e-5
    for i in range(2):
        e = jnp.zeros(2).at[i].set(h)
        fd = (float(f(z + e)) - float(f(z - e))) / (2 * h)
        assert abs(g[i] - fd) < 1e-6


def test_total_loss_wrapper_matches(toy):
    nz = NoiseParams()
    z = jnp.array([0.3])
    r = jax.nn.sigmoid(z)
    F = float(sim.fidelity(sim.simulate_mixture(toy, r, nz), sim.target_vector(toy)))
    x = L.live_counts(toy, r)
    expect = (1 - F) + 0.1 * float(L.smooth_max(x, 7.0)) + 0.01 * float(r[0]) * 2
    assert abs(float(L.total_loss(z, toy, nz, 0.1, 0.01, 7.0)) - expect) < TOL


def test_train_noiseless_frees_at_large_lambda(toy):
    res = T.train(toy, NoiseParams.none(), lam_a=0.1, out=False)
    assert res.r[0] > 0.9
    assert len(res.history["loss"]) == 300
    assert res.history["r"].shape == (300, 1)
    np.testing.assert_allclose(res.history["beta"][[0, -1]], [2.0, 20.0])


def test_train_noiseless_holds_at_small_lambda(toy):
    res = T.train(toy, NoiseParams.none(), lam_a=0.01, out=False)
    assert res.r[0] < 0.1


def test_train_budget_mode_holds_when_under_budget(toy):
    # Peak 2 is under a budget of 3, so only the depth term acts: hold.
    res = T.train(toy, NoiseParams.none(), lam_a=0.3, budget=3.0, out=False)
    assert res.r[0] < 0.1


def test_train_saves_json(toy, tmp_path):
    out = tmp_path / "t.json"
    T.train(toy, NoiseParams(), lam_a=0.1, steps=20, out=str(out))
    import json
    d = json.loads(out.read_text())
    assert d["config"]["lam_a"] == 0.1 and len(d["history"]["F"]) == 20


def test_sweep_matches_single_runs(toy):
    lams = [0.01, 0.05, 0.2]
    sw = T.sweep_lambda(toy, scales=(0.0, 1.0), lam_as=lams, steps=100, out=False)
    assert sw["lam_a"] == lams and set(sw["runs"]) == {"0.0", "1.0"}
    for s in (0.0, 1.0):
        for i, la in enumerate(lams):
            single = T.train(toy, NoiseParams().scaled(s), la, steps=100, out=False)
            assert abs(sw["runs"][str(s)]["r"][i][0] - single.r[0]) < 1e-8
