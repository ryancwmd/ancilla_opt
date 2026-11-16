"""Live counts, SmoothMax, depth term and total loss (workstream A).

L(z) = (1 - F(r)) + lam_a * SmoothMax_beta(x(r)) + lam_d * sum_i r_i Delta_i,  r = sigmoid(z)

F(r) is the fidelity of the soft circuit simulate_mixture(problem, r, noise) to
problem.target_state. The per-schedule density matrices rho_d do not depend on
r, so they are computed once (sim.schedule_rhos) and passed in as data; the
mixture sum_d w(d) rho_d is formed inside the differentiated function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

import sim
from noise import NoiseParams
from problem import Problem, gap_gates, layers, physical_schedule

KAPPA = 10.0  # budget-mode softplus sharpness


# ---------------------------------------------------------------------------
# Live counts
# ---------------------------------------------------------------------------

def live_count_matrices(problem: Problem):
    """(base, G) with x(r) = base + G @ r. base[t] = number of ancillas alive at
    base step t (inclusive lifetimes); G[t, i] = -1 if t is strictly inside the
    gap of decision i."""
    T = len(problem.gates)
    base = np.zeros(T)
    for a in problem.ancillas:
        base[a.alloc_step:a.release_step + 1] += 1
    G = np.zeros((T, len(problem.decisions)))
    for i, dec in enumerate(problem.decisions):
        for t in gap_gates(problem, dec):
            G[t, i] -= 1
    return base, G


def live_counts(problem: Problem, r):
    """Expected live ancillas x_t at each base step: 1 - r_i inside decision i's
    gap, 1 elsewhere in the lifetime, 0 outside it. Toy: [1, 1, 2-r, 2-r, 2-r, 1, 1]."""
    base, G = live_count_matrices(problem)
    r = jnp.atleast_1d(jnp.asarray(r, dtype=jnp.float64))
    return jnp.asarray(base) + jnp.asarray(G) @ r


def smooth_max(x, beta):
    """(1/beta) log sum_t exp(beta x_t); max(x) <= SmoothMax <= max(x) + log(T)/beta."""
    return logsumexp(beta * x) / beta


# ---------------------------------------------------------------------------
# Depth term
# ---------------------------------------------------------------------------

def extra_depths(problem: Problem):
    """Delta_i per decision: decision.extra_depth if B filled it, else the
    physical one-hot depth minus the physical all-hold depth."""
    N = len(problem.decisions)
    zeros = (0,) * N
    d0 = None
    out = []
    for i, dec in enumerate(problem.decisions):
        if dec.extra_depth >= 0:
            out.append(float(dec.extra_depth))
            continue
        if d0 is None:
            d0 = len(layers(physical_schedule(problem, zeros)))
        e = tuple(1 if j == i else 0 for j in range(N))
        out.append(float(len(layers(physical_schedule(problem, e))) - d0))
    return np.array(out)


# ---------------------------------------------------------------------------
# Total loss
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LossData:
    """Everything the loss needs that does not depend on z, as arrays (so the
    same compiled function serves every noise scale and lam_a)."""
    rhos: Any      # (2^N, D, D) complex, rho_d in sim.all_schedules order
    psi: Any       # (D,) target state
    base: Any      # (T,) live-count offset
    G: Any         # (T, N) live-count gap matrix
    delta: Any     # (N,) extra depth per decision


def make_loss_data(problem: Problem, noise: NoiseParams | None = None) -> LossData:
    noise = NoiseParams.none() if noise is None else noise
    base, G = live_count_matrices(problem)
    return LossData(
        rhos=sim.schedule_rhos(problem, noise),
        psi=sim.target_vector(problem),
        base=jnp.asarray(base),
        G=jnp.asarray(G),
        delta=jnp.asarray(extra_depths(problem)),
    )


jax.tree_util.register_pytree_node(
    LossData,
    lambda ld: ((ld.rhos, ld.psi, ld.base, ld.G, ld.delta), None),
    lambda _, xs: LossData(*xs),
)


def loss_terms(z, data: LossData, lam_a, lam_d, beta, budget=None):
    """Return (loss, aux) with aux = {F, smoothmax, peak_term, depth_term, r}."""
    r = jax.nn.sigmoid(z)
    w = sim.schedule_weights(r).astype(data.rhos.dtype)
    rho = jnp.tensordot(w, data.rhos, axes=1)          # simulate_mixture(problem, r, noise)
    F = sim.fidelity(rho, data.psi)
    x = data.base + data.G @ r
    sm = smooth_max(x, beta)
    if budget is None:
        peak_term = lam_a * sm
    else:
        peak_term = lam_a * jax.nn.softplus(KAPPA * (sm - budget)) / KAPPA
    depth_term = lam_d * jnp.sum(r * data.delta)
    loss = (1.0 - F) + peak_term + depth_term
    return loss, {"F": F, "smoothmax": sm, "peak_term": peak_term, "depth_term": depth_term, "r": r}


def total_loss(z, problem: Problem, noise: NoiseParams | None, lam_a, lam_d=0.01, beta=20.0,
               budget=None, data: LossData | None = None):
    """Scalar loss L(z). Convenience wrapper; training uses loss_terms with a
    precomputed LossData."""
    data = make_loss_data(problem, noise) if data is None else data
    return loss_terms(jnp.atleast_1d(jnp.asarray(z, dtype=jnp.float64)), data, lam_a, lam_d, beta,
                      budget)[0]
