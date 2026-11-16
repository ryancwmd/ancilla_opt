"""Density-matrix simulator (workstream A).

Operators are applied by tensor contraction on the (2,)*2n reshaped density
matrix, never by building a 2^n x 2^n gate matrix. Qubit 0 is the most
significant bit of a basis index.

Main entry points:
  run_discrete(problem, d, noise)            -> rho for one 0/1 schedule
  simulate_mixture_exact(problem, r, noise)  -> sum_d w(d) rho_d  (v1, reference)
  simulate_mixture(problem, r, noise)        -> same; v2 (gap-local) is not built,
                                                so this dispatches to v1
"""
from __future__ import annotations

import functools
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import noise as noise_mod
from noise import NoiseParams
from problem import Problem, layers, physical_schedule

# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _perm_matrix(dim, swap):
    m = np.eye(dim, dtype=complex)
    i, j = swap
    m[[i, j]] = m[[j, i]]
    return m


GATES = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
    "cx": _perm_matrix(4, (2, 3)),     # basis |control target>
    "ccx": _perm_matrix(8, (6, 7)),    # basis |c1 c2 target>
}


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def apply_op(rho, U, qs, n):
    """Return U rho U^dagger for a 2^k x 2^k operator U on qubits qs (in U's own order)."""
    k = len(qs)
    qs = list(qs)
    U = jnp.asarray(U, dtype=jnp.complex128).reshape((2,) * (2 * k))
    rho = rho.reshape((2,) * (2 * n))
    rho = jnp.tensordot(U, rho, axes=(list(range(k, 2 * k)), qs))
    rho = jnp.moveaxis(rho, list(range(k)), qs)
    cols = [n + q for q in qs]
    rho = jnp.tensordot(rho, U.conj(), axes=(cols, list(range(k, 2 * k))))
    rho = jnp.moveaxis(rho, list(range(2 * n - k, 2 * n)), cols)
    return rho.reshape(2 ** n, 2 ** n)


def apply_kraus(rho, kraus_ops, qs, n):
    """Return sum_K K rho K^dagger. Every noise channel except depolarizing goes through this."""
    return sum(apply_op(rho, K, qs, n) for K in kraus_ops)


def _replace_qubit(rho, q, n):
    """rho -> Tr_q(rho) (x) I/2 on qubit q (the fully depolarizing 1-qubit map)."""
    r = rho.reshape((2,) * (2 * n))
    r = jnp.moveaxis(r, (q, n + q), (0, 1))
    tr = r[0, 0] + r[1, 1]
    eye = jnp.eye(2, dtype=rho.dtype)
    r = eye[:, :, None] * (tr.reshape(-1)[None, None, :] / 2)
    r = r.reshape((2, 2) + tr.shape)
    r = jnp.moveaxis(r, (0, 1), (q, n + q))
    return r.reshape(2 ** n, 2 ** n)


def apply_depolarizing(rho, p, qs, n):
    """k-qubit depolarizing: (1 - p) rho + p Tr_qs(rho) (x) I/2^k.
    Equivalent to the 4^k-term Kraus form in noise.depolarizing_kraus, but cheap:
    the full replacement map on qs is the composition of 1-qubit replacements."""
    if p == 0:
        return rho
    rep = rho
    for q in qs:
        rep = _replace_qubit(rep, q, n)
    return (1 - p) * rho + p * rep


def fidelity(rho, psi):
    psi = jnp.asarray(psi, dtype=jnp.complex128)
    return jnp.real(psi.conj() @ rho @ psi)


def zero_state(n):
    return jnp.zeros((2 ** n, 2 ** n), dtype=jnp.complex128).at[0, 0].set(1.0)


def apply_gate(rho, gate, n):
    name, qs = gate
    if name not in GATES:
        raise ValueError("sim.py has no matrix for gate %r (known: %s)" % (name, sorted(GATES)))
    return apply_op(rho, GATES[name], qs, n)


# ---------------------------------------------------------------------------
# Statevector helper (noiseless target when problem.target_state is empty)
# ---------------------------------------------------------------------------

def statevector(gates, n):
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    for name, qs in gates:
        k = len(qs)
        U = GATES[name].reshape((2,) * (2 * k))
        psi = np.tensordot(U, psi, axes=(list(range(k, 2 * k)), list(qs)))
        psi = np.moveaxis(psi, list(range(k)), list(qs))
    return psi.reshape(-1)


def target_vector(problem: Problem):
    """problem.target_state as a complex128 array, or the noiseless output of
    prep + gates if the field is empty (C's adapter may leave it empty)."""
    if len(problem.target_state):
        return jnp.asarray(np.array(problem.target_state, dtype=complex))
    return jnp.asarray(statevector(list(problem.prep) + list(problem.gates), problem.n_qubits))


# ---------------------------------------------------------------------------
# One discrete schedule
# ---------------------------------------------------------------------------

def _idle_kraus(noise: NoiseParams):
    """Amplitude damping followed by phase damping, merged into one Kraus set
    (zero operators dropped). Empty list if both are off."""
    g, lam = noise.gamma, noise.lam
    if g == 0 and lam == 0:
        return []
    ops = [P @ A for A in noise_mod.amplitude_damping_kraus(g) for P in noise_mod.phase_damping_kraus(lam)]
    return [K for K in ops if np.abs(K).max() > 0]


def _run_schedule(prep, gates, n, noise: NoiseParams):
    rho = zero_state(n)
    for g in prep:
        rho = apply_gate(rho, g, n)
    idle = _idle_kraus(noise)
    for layer in layers(gates):
        touched = set()
        for i in layer:
            g = gates[i]
            rho = apply_gate(rho, g, n)
            rho = apply_depolarizing(rho, noise.p_gate(len(g[1])), g[1], n)
            touched.update(g[1])
        if idle:
            for q in range(n):
                if q not in touched:
                    rho = apply_kraus(rho, idle, (q,), n)
    return rho


@functools.partial(jax.jit, static_argnums=(0, 1, 2))
def _run_discrete_jit(problem, d, noise):
    gates = physical_schedule(problem, d)
    return _run_schedule(tuple(problem.prep), gates, problem.n_qubits, noise)


def run_discrete(problem: Problem, d, noise: NoiseParams | None = None):
    """Density matrix after prep (noiseless) and the physical schedule for 0/1
    vector d, with gate noise after every gate and idle noise on every qubit not
    touched in each ASAP layer. Works for Problems with no decisions (d = ())."""
    noise = NoiseParams.none() if noise is None else noise
    d = tuple(int(x) for x in d)
    return _run_discrete_jit(problem, d, noise)


# ---------------------------------------------------------------------------
# Soft circuit: average over schedules
# ---------------------------------------------------------------------------

def all_schedules(n_decisions):
    """All 0/1 vectors, in itertools.product order (d = (0,..,0) first)."""
    return list(itertools.product((0, 1), repeat=n_decisions))


def schedule_weights(r):
    """w(d) = prod_i r_i^d_i (1 - r_i)^(1 - d_i) for every d in all_schedules order."""
    r = jnp.atleast_1d(jnp.asarray(r, dtype=jnp.float64))
    D = jnp.asarray(np.array(all_schedules(r.shape[0]), dtype=float).reshape(-1, r.shape[0]))
    return jnp.prod(D * r[None, :] + (1 - D) * (1 - r[None, :]), axis=1)


@functools.lru_cache(maxsize=64)
def schedule_rhos(problem: Problem, noise: NoiseParams):
    """Stacked rho_d for every d in all_schedules order. The rho_d do not
    depend on r, so they are computed once per (problem, noise) and cached."""
    N = len(problem.decisions)
    if N > 12:
        raise ValueError("simulate_mixture_exact enumerates 2^N schedules; N = %d is too many" % N)
    return jnp.stack([run_discrete(problem, d, noise) for d in all_schedules(N)])


def simulate_mixture_exact(problem: Problem, r, noise: NoiseParams | None = None):
    """v1 reference: sum_d w(d) rho_d over all 2^N decision vectors. Each
    decision's uncompute and recompute blocks are one coin flip, so the mixture
    never contains a branch that uncomputes without recomputing.
    Differentiable in r."""
    noise = NoiseParams.none() if noise is None else noise
    rhos = schedule_rhos(problem, noise)
    w = schedule_weights(r).astype(rhos.dtype)
    return jnp.tensordot(w, rhos, axes=1)


def simulate_mixture(problem: Problem, r, noise: NoiseParams | None = None):
    """Soft circuit. v2 (gap-local mixing) is a stretch goal and not built, so
    this is v1 (exact enumeration)."""
    return simulate_mixture_exact(problem, r, noise)


def schedule_fidelities(problem: Problem, noise: NoiseParams | None = None):
    """F_d = <psi|rho_d|psi> for every d in all_schedules order (numpy array)."""
    noise = NoiseParams.none() if noise is None else noise
    psi = target_vector(problem)
    rhos = schedule_rhos(problem, noise)
    return np.asarray(jnp.real(jnp.einsum("i,dij,j->d", psi.conj(), rhos, psi)))
