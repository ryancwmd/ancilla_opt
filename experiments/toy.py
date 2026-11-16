"""Integration: train on the toy Problem, round, evaluate exactly, compare.

    uv run python experiments/toy.py [--backend auto|a|reference] [--steps 300]
                                     [--no-train] [--quick] [--lam-d 0.01]

Uses A's train.train / train.sweep_lambda / sim.run_discrete / noise.NoiseParams,
B's evaluate.py and baselines.py, and C's reqomp_adapter/results/toy.json if it
exists. Produces (instructions.md, "Outputs and reporting back"):

  figures/toy_training.png      r over training steps, lam_a in {0.01, 0.1, 1}, s in {0, 1}
  figures/toy_lambda_sweep.png  final r vs lam_a, one line per noise scale, brute-force flips marked
  results/toy_table.md, .json   one row per schedule, metrics + fidelity at each noise scale
  results/toy.json              full config and results, including the sweep and mismatch report

--no-train skips everything that needs A's trainer (table and brute force only).
"""
from __future__ import annotations

import argparse
import inspect
import json
import time

import numpy as np

from common import (FIGURES, NOISE_SCALES, RESULTS, ROOT, circuits, get_backend, git_rev, md_table,
                    noise_dict, write_json)

import baselines as B
import evaluate as E

TRAIN_LAMS = (0.01, 0.1, 1.0)
TRAIN_SCALES = (0.0, 1.0)
SWEEP_LAMS = tuple(float(x) for x in np.logspace(np.log10(0.01), np.log10(0.3), 21))
BAND_REL = 0.15          # mismatches within +-15% of a brute-force flip count as "near threshold"

# Reference palette, categorical slots 1-4 in fixed order (light mode, static figures).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# Table: one row per schedule
# ---------------------------------------------------------------------------

def reqomp_rows() -> list[dict]:
    """Rows from C's reqomp_adapter/results/toy.json, if present. Field names
    are read defensively; fidelity is filled only if C provides it."""
    path = ROOT / "reqomp_adapter" / "results" / "toy.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    runs = data.get("runs", data.get("budgets", data if isinstance(data, list) else []))
    out = []
    for run in runs if isinstance(runs, list) else []:
        def get(*names):
            for n in names:
                if n in run:
                    return run[n]
            return None
        out.append({"schedule": "reqomp (budget %s)" % get("budget", "qubit_budget", "n_ancillas"),
                    "peak_ancillas": get("peak_ancillas", "ancillas", "n_ancillas"),
                    "physical_qubits": get("qubits", "qubit_count", "n_qubits", "physical_qubits"),
                    "depth": get("depth"),
                    "two_qubit_equiv": get("two_qubit_equiv", "2q_equiv", "two_q_equiv"),
                    "fidelity": {str(s): get("fidelity_%s" % s) for s in NOISE_SCALES},
                    "source": str(path.relative_to(ROOT))})
    return out


def schedule_rows(p, make_noise, simulator) -> list[dict]:
    rows = []
    for label, fn in [("no reuse", lambda nz: B.no_reuse(p, nz, simulator)),
                      ("hold, reassigned", lambda nz: B.hold_all_reassigned(p, nz, simulator)),
                      ("free (d = 1)", lambda nz: E.evaluate(p, (1,), nz, simulator))]:
        per = {str(s): fn(make_noise(s)) for s in NOISE_SCALES}
        base = per[str(NOISE_SCALES[0])]
        rows.append({"schedule": label, "d": base["d"], "peak_ancillas": base["peak_ancillas"],
                     "physical_qubits": base["physical_qubits"], "depth": base["depth"],
                     "two_qubit_equiv": base["two_qubit_equiv"],
                     "fidelity": {s: r["fidelity"] for s, r in per.items()}})
    return rows + reqomp_rows()


def table_markdown(rows, cfg) -> str:
    def f(x):
        return "n/a" if x is None else ("%.4f" % x if isinstance(x, float) else str(x))
    hdr = ["schedule", "peak ancillas", "physical qubits", "depth (layers)", "2q-equivalent gates"] + \
          ["F, noiseless" if s == 0 else "F, s = %g" % s for s in NOISE_SCALES]
    body = [[r["schedule"], f(r["peak_ancillas"]), f(r["physical_qubits"]), f(r["depth"]),
             f(r["two_qubit_equiv"])] + [f(r["fidelity"].get(str(s))) for s in NOISE_SCALES] for r in rows]
    n1 = cfg["noise"][str(1.0)]
    return ("# Toy circuit: schedules compared\n\n" + md_table(hdr, body) +
            "\nFidelity to the noiseless target, on the physical schedule. Noise at s = 1: p1 = %g, p2 = %g, "
            "p3 = %s (None = 1-(1-p2)^6), t_layer = %g us, T1 = %g us, T2 = %g us; scale s multiplies "
            "error rates and divides T1, T2. Backend: %s. 2q-equivalent: cx = 1, ccx = 6.\n"
            % (n1["p1"], n1["p2"], n1["p3"], n1["t_layer_us"], n1["T1_us"], n1["T2_us"], cfg["backend"]))


# ---------------------------------------------------------------------------
# Training (A's train.py)
# ---------------------------------------------------------------------------

def r_history(hist) -> np.ndarray:
    """Per-step r as an array (steps, N), from a list of per-step dicts or a dict of lists."""
    if isinstance(hist, dict):
        r = hist.get("r")
        if r is None:
            keys = sorted(k for k in hist if k.startswith("r_") or k.startswith("r["))
            r = np.stack([np.asarray(hist[k]) for k in keys], axis=1)
        r = np.asarray(r, dtype=float)
        return r.reshape(len(r), -1)
    return np.asarray([np.asarray(h["r"], dtype=float).reshape(-1) for h in hist])


def train_one(train_mod, p, noise, lam_a, lam_d, steps, seed=0):
    res = train_mod.train(p, noise, lam_a, lam_d=lam_d, steps=steps, seed=seed)
    return {"r_final": np.asarray(res.r, dtype=float).reshape(-1).tolist(),
            "r_history": r_history(res.history).tolist()}


def sweep(train_mod, p, noise, lams, lam_d, steps) -> tuple[np.ndarray, str]:
    """Final r per lam_a, shape (len(lams), N). Uses A's sweep_lambda when its
    signature is recognisable, otherwise one train() call per lam_a."""
    fn = getattr(train_mod, "sweep_lambda", None)
    if fn is not None:
        params = inspect.signature(fn).parameters
        lam_kw = next((k for k in ("lam_as", "lam_a", "lams", "lambdas", "lam_values") if k in params), None)
        if lam_kw is not None:
            kw = {lam_kw: np.asarray(lams)}
            if "lam_d" in params:
                kw["lam_d"] = lam_d
            if "steps" in params:
                kw["steps"] = steps
            if "noise" in params:
                kw["noise"] = noise
                out = fn(p, **kw)
            else:
                out = fn(p, noise, **kw)
            if isinstance(out, dict):
                out = out.get("r", out.get("r_final"))
            if hasattr(out, "r"):
                out = out.r
            arr = np.asarray(out, dtype=float)
            return arr.reshape(len(lams), -1), "train.sweep_lambda"
    rs = [train_mod.train(p, noise, lam, lam_d=lam_d, steps=steps).r for lam in lams]
    return np.asarray(rs, dtype=float).reshape(len(lams), -1), "train.train loop"


def mismatch_report(lams, r_final, table, flips, lam_d) -> list[dict]:
    out = []
    for lam, r in zip(lams, r_final):
        d = E.round_decisions(r)
        bf = B.argmin_table(table, lam, lam_d)["d"]
        if d != bf:
            near = any(abs(lam - f["lam_a"]) <= BAND_REL * f["lam_a"] for f in flips)
            out.append({"lam_a": lam, "trained": d, "brute_force": bf, "r": list(map(float, r)),
                        "near_threshold": near})
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2)


def _noise_caption(cfg) -> str:
    n = cfg["noise"][str(1.0)]
    return ("Noise at s = 1: p1 = %g, p2 = %g, p3 = 1-(1-p2)^6, t_layer = %g µs, T1 = %g µs, T2 = %g µs; "
            "s scales error rates by s and T1, T2 by 1/s. lam_d = %g. %s"
            % (n["p1"], n["p2"], n["t_layer_us"], n["T1_us"], n["T2_us"], cfg["lam_d"], cfg["backend"]))


def plot_training(training, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(TRAIN_SCALES), figsize=(11, 4.2), sharey=True)
    for ax, s in zip(np.atleast_1d(axes), TRAIN_SCALES):
        ends = []
        for i, lam in enumerate(TRAIN_LAMS):
            h = np.asarray(training[str(s)][str(lam)]["r_history"])[:, 0]
            ax.plot(np.arange(len(h)), h, color=SERIES[i], linewidth=2, label="λa = %g" % lam)
            ends.append((h[-1], len(h) - 1, lam))
        # Direct end labels; lines that end together share one label.
        ends.sort()
        groups = []
        for y, x, lam in ends:
            if groups and abs(y - groups[-1][0]) < 0.06:
                groups[-1][2].append(lam)
            else:
                groups.append([y, x, [lam]])
        for y, x, lams in groups:
            ax.annotate("λa = " + ", ".join("%g" % l for l in sorted(lams)), (x, y), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_title("noiseless" if s == 0 else "noise scale s = %g" % s, color=INK, fontsize=11)
        ax.set_xlabel("training step (Adam iteration)")
        ax.set_ylim(-0.03, 1.03)
        ax.margins(x=0.15)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("r = σ(z), probability of freeing a (unitless)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(TRAIN_LAMS), fontsize=9,
               bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Toy circuit: soft decision r over training", color=INK, fontsize=12)
    fig.text(0.5, 0.005, _noise_caption(cfg), ha="center", fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sweep(sweeps, flips_by_scale, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, s in enumerate(NOISE_SCALES):
        r = np.asarray(sweeps[str(s)]["r_final"])[:, 0]
        label = "noiseless" if s == 0 else "s = %g" % s
        ax.plot(SWEEP_LAMS, r, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=6,
                markeredgecolor="white", markeredgewidth=1, label=label)
        for f in flips_by_scale[str(s)]:
            ax.axvline(f["lam_a"], color=SERIES[i], linewidth=1.2, linestyle="--")
    # One annotation per group of flips that land within 2% of each other.
    marks = sorted((f["lam_a"], s) for s in NOISE_SCALES for f in flips_by_scale[str(s)])
    groups = []
    for lam, s in marks:
        if groups and lam <= groups[-1][0] * 1.02:
            groups[-1][1].append(s)
        else:
            groups.append([lam, [s]])
    for k, (lam, ss) in enumerate(groups):
        names = ", ".join("noiseless" if s == 0 else "s = %g" % s for s in sorted(ss))
        ax.annotate("%s flip\nλa ≈ %.3f" % (names, lam), (lam, 0.12 + 0.22 * k), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("λa, weight on peak live ancillas (loss units per ancilla, log scale)")
    ax.set_ylabel("final r, probability of freeing a (unitless)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, loc="upper left", fontsize=9, title="trained r (line), brute-force flip (dashed)",
              title_fontsize=8)
    _style(ax)
    ax.set_title("Toy circuit: trained decision vs λa", color=INK, fontsize=12)
    fig.text(0.5, 0.005, _noise_caption(cfg), ha="center", fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "a", "reference"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lam-d", type=float, default=0.01)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer steps, for smoke tests")
    args = ap.parse_args()
    steps = 40 if args.quick else args.steps
    t_start = time.time()

    make_noise, simulator, label = get_backend(args.backend)
    p = E.fill_costs(circuits.toy())
    cfg = {"script": "experiments/toy.py", "backend": label, "lam_d": args.lam_d, "steps": steps,
           "train_lams": TRAIN_LAMS, "train_scales": TRAIN_SCALES, "sweep_lams": SWEEP_LAMS,
           "noise_scales": NOISE_SCALES, "band_rel": BAND_REL, "seed": 0, "git": git_rev(),
           "noise": {str(s): noise_dict(make_noise(s)) for s in NOISE_SCALES}}
    out = {"config": cfg}

    # Table and brute force.
    rows = schedule_rows(p, make_noise, simulator)
    tables = {str(s): B.schedule_table(p, make_noise(s), simulator) for s in NOISE_SCALES}
    flips = {s: B.flip_thresholds(t, args.lam_d) for s, t in tables.items()}
    out.update(table=rows, brute_force={"tables": tables, "flip_thresholds": flips})
    RESULTS.mkdir(exist_ok=True)
    write_json("toy_table", {"config": cfg, "rows": rows})
    (RESULTS / "toy_table.md").write_text(table_markdown(rows, cfg))
    print(table_markdown(rows, cfg))
    for s, fl in flips.items():
        print("brute-force flip, s = %s: %s" % (s, ["%.4f" % f["lam_a"] for f in fl]))

    if not args.no_train:
        import train as train_mod
        FIGURES.mkdir(exist_ok=True)
        training = {}
        for s in TRAIN_SCALES:
            training[str(s)] = {}
            for lam in TRAIN_LAMS:
                t0 = time.time()
                training[str(s)][str(lam)] = train_one(train_mod, p, make_noise(s), lam, args.lam_d, steps)
                print("train s=%g lam_a=%g: r=%s (%.1fs)" % (s, lam, training[str(s)][str(lam)]["r_final"],
                                                            time.time() - t0))
        plot_training(training, cfg, FIGURES / "toy_training.png")

        sweeps, mismatches = {}, {}
        for s in NOISE_SCALES:
            t0 = time.time()
            r_final, how = sweep(train_mod, p, make_noise(s), SWEEP_LAMS, args.lam_d, steps)
            sweeps[str(s)] = {"r_final": r_final.tolist(), "method": how,
                              "rounded": [E.round_decisions(r) for r in r_final]}
            mismatches[str(s)] = mismatch_report(SWEEP_LAMS, r_final, tables[str(s)], flips[str(s)], args.lam_d)
            print("sweep s=%g via %s (%.1fs): %d mismatches with brute force, %d outside the band"
                  % (s, how, time.time() - t0, len(mismatches[str(s)]),
                     sum(not m["near_threshold"] for m in mismatches[str(s)])))
        plot_sweep(sweeps, flips, cfg, FIGURES / "toy_lambda_sweep.png")
        out.update(training=training, sweep=sweeps, mismatches=mismatches)
        bad = [(s, m) for s, ms in mismatches.items() for m in ms if not m["near_threshold"]]
        if bad:
            print("WARNING: trained-and-rounded schedules disagree with brute force outside the band "
                  "(instructions.md says stop and report):")
            for s, m in bad:
                print("  s=%s lam_a=%.4f trained %s brute force %s r=%s" % (s, m["lam_a"], m["trained"],
                                                                          m["brute_force"], m["r"]))
    out["seconds"] = time.time() - t_start
    print("wrote", write_json("toy", out), "in %.1fs" % out["seconds"])


if __name__ == "__main__":
    main()
