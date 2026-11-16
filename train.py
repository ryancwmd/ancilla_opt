"""Training loop and lambda sweep (workstream A).

train(problem, noise, lam_a, ...)   -> TrainResult (final z, final r, history)
sweep_lambda(problem, ...)          -> final r for 21 log-spaced lam_a x noise scales

One logit z_i per decision, r_i = sigmoid(z_i) = probability of freeing,
initialised at z = 0. Adam (optax) on z; beta rises linearly from beta[0] to
beta[1]. The whole loop is a jax.lax.scan, jitted once; the sweep vmaps it
over lam_a, and the per-noise data (schedule density matrices) is an argument,
so one compiled trainer serves every lam_a and every noise scale.
"""
from __future__ import annotations

import functools
import json
import os
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

import loss as loss_mod
from noise import NoiseParams
from problem import Problem

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DEFAULT_SCALES = (0.0, 0.5, 1.0, 2.0)


def default_lam_grid():
    return np.geomspace(0.01, 0.3, 21)


@dataclass
class TrainResult:
    z: np.ndarray
    r: np.ndarray
    history: dict = field(default_factory=dict)   # per-step arrays: loss, F, smoothmax, ...
    config: dict = field(default_factory=dict)

    def to_dict(self):
        return {"config": self.config, "z": self.z.tolist(), "r": self.r.tolist(),
                "history": {k: np.asarray(v).tolist() for k, v in self.history.items()}}


# ---------------------------------------------------------------------------
# Core (pure, jittable, vmappable over lam_a)
# ---------------------------------------------------------------------------

@functools.partial(jax.jit, static_argnames=("steps", "lr", "use_budget"))
def _train_core(data, z0, lam_a, lam_d, beta0, beta1, budget, *, steps, lr, use_budget):
    opt = optax.adam(lr)
    vg = jax.value_and_grad(
        lambda z, beta: loss_mod.loss_terms(z, data, lam_a, lam_d, beta,
                                            budget if use_budget else None),
        has_aux=True)
    denom = max(steps - 1, 1)

    def step(carry, t):
        z, state = carry
        beta = beta0 + (beta1 - beta0) * t / denom
        (L, aux), g = vg(z, beta)
        upd, state = opt.update(g, state, z)
        z = optax.apply_updates(z, upd)
        rec = {"loss": L, "F": aux["F"], "smoothmax": aux["smoothmax"],
               "peak_term": aux["peak_term"], "depth_term": aux["depth_term"],
               "r": aux["r"], "beta": beta, "grad_norm": jnp.linalg.norm(g)}
        return (z, state), rec

    (z, _), hist = jax.lax.scan(step, (z0, opt.init(z0)), jnp.arange(steps, dtype=jnp.float64))
    return z, jax.nn.sigmoid(z), hist


def _config(problem, noise, **kw):
    cfg = {"problem": problem.name, "n_decisions": len(problem.decisions),
           "decision_ids": [d.id for d in problem.decisions], "noise": noise.to_dict()}
    cfg.update(kw)
    return cfg


def _save(obj, out):
    if not out:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(obj, f, indent=1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(problem: Problem, noise: NoiseParams | None, lam_a: float, lam_d: float = 0.01,
          steps: int = 300, lr: float = 0.1, beta=(2.0, 20.0), seed: int = 0,
          budget: float | None = None, out: str | None | bool = None) -> TrainResult:
    """Train the logits z for one lam_a. z starts at 0 (r = 0.5); `seed` is
    recorded but unused, since nothing here is random. `out`: JSON path,
    None for results/train_<name>_la<lam_a>_<noise>.json, False to skip saving."""
    noise = NoiseParams.none() if noise is None else noise
    data = loss_mod.make_loss_data(problem, noise)
    N = len(problem.decisions)
    z0 = jnp.zeros(N)
    t0 = time.time()
    z, r, hist = _train_core(data, z0, float(lam_a), float(lam_d), float(beta[0]), float(beta[1]),
                             float(budget if budget is not None else 0.0),
                             steps=int(steps), lr=float(lr), use_budget=budget is not None)
    res = TrainResult(
        z=np.asarray(z), r=np.asarray(r),
        history={k: np.asarray(v) for k, v in hist.items()},
        config=_config(problem, noise, lam_a=float(lam_a), lam_d=float(lam_d), steps=int(steps),
                       lr=float(lr), beta=list(map(float, beta)), seed=int(seed), budget=budget,
                       seconds=time.time() - t0),
    )
    if out is None:
        out = os.path.join(RESULTS_DIR, "train_%s_la%.4g_%s.json"
                           % (problem.name, lam_a, noise.source.replace(" ", "")))
    _save(res.to_dict(), out)
    return res


def sweep_lambda(problem: Problem, base_noise: NoiseParams | None = None, scales=DEFAULT_SCALES,
                 lam_as=None, lam_d: float = 0.01, steps: int = 300, lr: float = 0.1,
                 beta=(2.0, 20.0), seed: int = 0, budget: float | None = None,
                 keep_history: bool = False, out: str | None | bool = None) -> dict:
    """Train for every lam_a (default 21 log-spaced values in [0.01, 0.3]) at
    each noise scale s (scale 0 = noiseless). lam_a is batched with jax.vmap.

    Returns {"lam_a": [...], "scales": [...], "runs": {str(s): {"r": [[r_i] per lam_a],
    "z": ..., "noise": {...}, "final_loss": [...], "final_F": [...]}}}. With
    keep_history=True each run also carries the per-step history."""
    base_noise = NoiseParams() if base_noise is None else base_noise
    lam_as = default_lam_grid() if lam_as is None else np.asarray(lam_as, dtype=float)
    N = len(problem.decisions)
    lam_vec = jnp.asarray(lam_as)
    t0 = time.time()

    batched = jax.vmap(
        lambda data, la: _train_core(data, jnp.zeros(N), la, float(lam_d), float(beta[0]),
                                     float(beta[1]), float(budget if budget is not None else 0.0),
                                     steps=int(steps), lr=float(lr), use_budget=budget is not None),
        in_axes=(None, 0))
    batched = jax.jit(batched)

    runs = {}
    for s in scales:
        noise = base_noise.scaled(s)
        data = loss_mod.make_loss_data(problem, noise)
        z, r, hist = batched(data, lam_vec)
        run = {"scale": float(s), "noise": noise.to_dict(), "z": np.asarray(z).tolist(),
               "r": np.asarray(r).tolist(),
               "final_loss": np.asarray(hist["loss"][:, -1]).tolist(),
               "final_F": np.asarray(hist["F"][:, -1]).tolist()}
        if keep_history:
            run["history"] = {k: np.asarray(v).tolist() for k, v in hist.items()}
        runs[str(float(s))] = run

    result = {"config": {"problem": problem.name, "decision_ids": [d.id for d in problem.decisions],
                         "base_noise": base_noise.to_dict(), "lam_d": float(lam_d),
                         "steps": int(steps), "lr": float(lr), "beta": list(map(float, beta)),
                         "seed": int(seed), "budget": budget, "seconds": time.time() - t0},
              "lam_a": lam_as.tolist(), "scales": [float(s) for s in scales], "runs": runs}
    if out is None:
        out = os.path.join(RESULTS_DIR, "sweep_%s.json" % problem.name)
    _save(result, out)
    return result


if __name__ == "__main__":  # quick smoke run on the toy
    from circuits import toy
    p = toy()
    for la in (0.01, 0.1):
        res = train(p, NoiseParams.none(), la, out=False)
        print("noiseless lam_a=%g: r=%s" % (la, res.r))
    sw = sweep_lambda(p, out=False)
    for s, run in sw["runs"].items():
        print("s=%s r=%s" % (s, np.round(np.array(run["r"])[:, 0], 3)))
