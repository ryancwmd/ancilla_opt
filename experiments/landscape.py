"""Soft-loss landscape over (r1, r2) with training paths, for two-decision Problems.

    uv run python experiments/landscape.py [--problem bench2] [--scale 1] [--lams 0.1 0.22 0.3]
                                           [--grid 121] [--steps 300] [--lam-d 0.01] [--replot]

Evaluates A's loss.loss_terms at the final training beta over a grid in r, trains each lam_a with
A's train.train, and compares with B's brute force. Writes results/<p>_landscape.json and
figures/<p>_landscape.png (the figure is drawn from the saved results).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common import (FIGURES, INK, INK2, RESULTS, SEQUENTIAL, SERIES, SURFACE, circuits, display_name,
                    get_backend, git_rev, noise_caption, noise_dict, write_json)

import baselines as B
import evaluate as E

BETA = (2.0, 20.0)
CORNER_OFFSETS = {(0, 0): (6, 6, "left", "bottom"), (1, 0): (-6, 6, "right", "bottom"),
                  (0, 1): (6, -6, "left", "top"), (1, 1): (-6, -6, "right", "top")}


def loss_grid(p, noise, lam_a, lam_d, beta, n):
    import jax
    import jax.numpy as jnp
    import loss as loss_mod
    data = loss_mod.make_loss_data(p, noise)
    r = np.linspace(0.0, 1.0, n)
    rc = np.clip(r, 1e-9, 1 - 1e-9)
    z = np.log(rc) - np.log1p(-rc)
    z1, z2 = np.meshgrid(z, z, indexing="xy")
    zs = jnp.asarray(np.stack([z1.ravel(), z2.ravel()], axis=1))
    evaluate = jax.jit(lambda batch: jax.lax.map(
        lambda zz: loss_mod.loss_terms(zz, data, lam_a, lam_d, beta)[0], batch, batch_size=8))
    return r, np.asarray(evaluate(zs)).reshape(n, n)


def run(args) -> dict:
    import train as train_mod
    make_noise, simulator, label = get_backend("a")
    p = E.fill_costs(circuits.BENCHMARKS[args.problem]())
    if len(p.decisions) != 2:
        raise SystemExit("landscape.py plots two-decision Problems; %s has %d" % (p.name, len(p.decisions)))
    noise = make_noise(args.scale)
    table = B.schedule_table(p, noise, simulator)
    flips = B.flip_thresholds(table, args.lam_d)
    cfg = {"script": "experiments/landscape.py", "problem": p.name, "decision_ids": [d.id for d in p.decisions],
           "backend": label, "scale": args.scale, "lam_d": args.lam_d, "lams": list(args.lams),
           "steps": args.steps, "beta": list(BETA), "grid": args.grid, "git": git_rev(),
           "noise": {str(1.0): noise_dict(make_noise(1.0)), str(args.scale): noise_dict(noise)}}
    out = {"config": cfg, "flip_thresholds": flips, "panels": []}
    for lam in args.lams:
        t0 = time.time()
        r, L = loss_grid(p, noise, lam, args.lam_d, BETA[1], args.grid)
        res = train_mod.train(p, noise, lam, lam_d=args.lam_d, steps=args.steps, beta=BETA, out=False)
        traj = np.asarray(res.history["r"], dtype=float).reshape(args.steps, -1)
        panel = {"lam_a": lam, "loss": L.tolist(), "trajectory": traj.tolist(),
                 "r_final": np.asarray(res.r, dtype=float).tolist(), "rounded": list(E.round_decisions(res.r)),
                 "brute_force": list(B.argmin_table(table, lam, args.lam_d)["d"])}
        out["panels"].append(panel)
        out["r_grid"] = r.tolist()
        print("lam_a=%g: trained r=%s rounded %s brute force %s (%.1fs)"
              % (lam, np.round(panel["r_final"], 4), panel["rounded"], panel["brute_force"], time.time() - t0))
    return out


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe
    from matplotlib.colors import ListedColormap

    cfg = res["config"]
    ids = cfg["decision_ids"]
    flip = res["flip_thresholds"][0]["lam_a"] if res["flip_thresholds"] else None
    r = np.asarray(res["r_grid"])
    cmap = ListedColormap(SEQUENTIAL)
    k = len(res["panels"])
    fig, axes = plt.subplots(1, k, figsize=(5.2 * k, 5.4), squeeze=False)
    for i, (ax, pan) in enumerate(zip(axes[0], res["panels"])):
        L = np.asarray(pan["loss"])
        levels = np.linspace(L.min(), L.max(), len(SEQUENTIAL) + 1)
        filled = ax.contourf(r, r, L, levels=levels, cmap=cmap)
        ax.contour(r, r, L, levels=levels, colors=SURFACE, linewidths=0.5)
        traj = np.asarray(pan["trajectory"])
        ax.plot(traj[:, 0], traj[:, 1], color=SERIES[1], linewidth=2, solid_capstyle="round",
                path_effects=[pe.Stroke(linewidth=5, foreground=SURFACE), pe.Normal()],
                label="training path (%d Adam steps)" % len(traj))
        ax.plot(*traj[0], marker="o", markersize=9, markerfacecolor=SURFACE, markeredgecolor=SERIES[1],
                markeredgewidth=2, linestyle="none", label="start, r = (0.5, 0.5)")
        ax.plot(*traj[-1], marker="o", markersize=9, color=SERIES[1], markeredgecolor=SURFACE,
                markeredgewidth=1.5, linestyle="none", label="end of training")
        bf = pan["brute_force"]
        ax.plot(bf[0], bf[1], marker="s", markersize=17, markerfacecolor="none", markeredgecolor=INK,
                markeredgewidth=1.8, linestyle="none", clip_on=False, label="brute-force optimum")
        for (x, y), (dx, dy, ha, va) in CORNER_OFFSETS.items():
            ax.annotate("d = %d%d" % (x, y), (x, y), xytext=(dx + (12 if dx > 0 else -12), dy + (12 if dy > 0 else -12)),
                        textcoords="offset points", ha=ha, va=va, fontsize=9, color=INK,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none", alpha=0.9))
        side = "" if flip is None else (" (below flip %.3f)" % flip if pan["lam_a"] < flip else " (above flip %.3f)" % flip)
        ax.set_title("λa = %g%s" % (pan["lam_a"], side), color=INK, fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("r(%s), probability of freeing (unitless)" % ids[0])
        if i == 0:
            ax.set_ylabel("r(%s), probability of freeing (unitless)" % ids[1])
        for side_name in ("top", "right"):
            ax.spines[side_name].set_visible(False)
        for side_name in ("left", "bottom"):
            ax.spines[side_name].set_color(INK2)
        ax.tick_params(colors=INK2)
        cb = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.04, format="%.3f")
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK2, labelsize=8)
        cb.set_label("soft loss L at β = %g (light = lower)" % cfg["beta"][1], color=INK2, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, 0.94))
    s = cfg["scale"]
    fig.suptitle("%s: soft-loss landscape and training paths, %s"
                 % (display_name(cfg), "noiseless" if s == 0 else "noise scale s = %g" % s), color=INK, fontsize=12)
    fig.text(0.5, 0.012, "Contours: L(r) at the final β; training anneals β from %g to %g, so early steps see a "
             "smoother landscape. " % tuple(cfg["beta"]) + noise_caption(cfg),
             ha="center", fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.075, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="bench2", choices=sorted(circuits.BENCHMARKS))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 0.22, 0.3])
    ap.add_argument("--grid", type=int, default=121)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lam-d", type=float, default=0.01)
    ap.add_argument("--replot", action="store_true", help="redraw the figure from results/<p>_landscape.json")
    args = ap.parse_args()
    figure = FIGURES / ("%s_landscape.png" % args.problem)
    FIGURES.mkdir(exist_ok=True)
    if args.replot:
        plot(json.loads((RESULTS / ("%s_landscape.json" % args.problem)).read_text()), figure)
        print("wrote", figure)
        return
    t0 = time.time()
    res = run(args)
    res["seconds"] = time.time() - t0
    path = write_json("%s_landscape" % args.problem, res)
    plot(res, figure)
    print("wrote", path, "and", figure, "in %.1fs" % res["seconds"])


if __name__ == "__main__":
    main()
