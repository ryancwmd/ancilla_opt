"""Integration: train on a benchmark Problem, round, evaluate exactly, compare.

    uv run python experiments/integration.py [--problem toy|bench2] [--backend auto|a|reference]
                                             [--steps 300] [--no-train] [--quick] [--lam-d 0.01]

Uses A's train.train / train.sweep_lambda / sim.run_discrete / noise.NoiseParams,
B's evaluate.py and baselines.py, and C's reqomp_adapter/results/<problem>.json if
it exists. Produces, with <p> the problem name:

  figures/<p>_training.png      r over training steps, lam_a in {0.01, 0.1, 1}, s in {0, 1}, one row per decision
  figures/<p>_lambda_sweep.png  final r vs lam_a, one panel per decision, one line per noise scale, brute-force flips marked
  results/<p>_table.md, .json   one row per schedule, metrics + fidelity at each noise scale
  results/<p>.json              full config and results, including the sweep and mismatch report

--no-train skips everything that needs A's trainer (table and brute force only).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from common import (FIGURES, INK, INK2, MARKERS, NOISE_SCALES, RESULTS, ROOT, SERIES, circuits,
                    display_name, get_backend, git_rev, md_table, noise_caption, noise_dict, style_axes,
                    write_json)

import baselines as B
import evaluate as E

TRAIN_LAMS = (0.01, 0.1, 1.0)
TRAIN_SCALES = (0.0, 1.0)
SWEEP_LAMS = tuple(float(x) for x in np.logspace(np.log10(0.01), np.log10(0.3), 21))
BAND_REL = 0.15          # mismatches within +-15% of a brute-force flip count as "near threshold"


# ---------------------------------------------------------------------------
# Table: one row per schedule
# ---------------------------------------------------------------------------

def reqomp_rows(name: str) -> list[dict]:
    """Rows from C's reqomp_adapter/results/<name>.json, if present. Field names
    are read defensively; fidelity is filled only if C provides it."""
    path = ROOT / "reqomp_adapter" / "results" / ("%s.json" % name)
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
    """No reuse, hold with reassignment, then every schedule that frees at least one gap."""
    freeing = [d for d in itertools.product((0, 1), repeat=len(p.decisions)) if any(d)]
    entries = [("no reuse", lambda nz: B.no_reuse(p, nz, simulator)),
               ("hold, reassigned", lambda nz: B.hold_all_reassigned(p, nz, simulator))]
    entries += [("d = " + "".join(map(str, d)), lambda nz, d=d: E.evaluate(p, d, nz, simulator))
                for d in freeing]
    rows = []
    for label, fn in entries:
        per = {str(s): fn(make_noise(s)) for s in NOISE_SCALES}
        base = per[str(NOISE_SCALES[0])]
        rows.append({"schedule": label, "d": base["d"], "peak_ancillas": base["peak_ancillas"],
                     "physical_qubits": base["physical_qubits"], "depth": base["depth"],
                     "two_qubit_equiv": base["two_qubit_equiv"],
                     "fidelity": {s: r["fidelity"] for s, r in per.items()}})
    return rows + reqomp_rows(p.name)


def table_markdown(rows, cfg) -> str:
    def f(x):
        return "n/a" if x is None else ("%.4f" % x if isinstance(x, float) else str(x))
    hdr = ["schedule", "peak ancillas", "physical qubits", "depth (layers)", "2q-equivalent gates"] + \
          ["F, noiseless" if s == 0 else "F, s = %g" % s for s in NOISE_SCALES]
    body = [[r["schedule"], f(r["peak_ancillas"]), f(r["physical_qubits"]), f(r["depth"]),
             f(r["two_qubit_equiv"])] + [f(r["fidelity"].get(str(s))) for s in NOISE_SCALES] for r in rows]
    n1 = cfg["noise"][str(1.0)]
    return ("# %s: schedules compared\n\n" % display_name(cfg) + md_table(hdr, body) +
            "\nDecisions, in d order: %s. Fidelity to the noiseless target, on the physical schedule. "
            "Noise at s = 1: p1 = %g, p2 = %g, p3 = %s (None = 1-(1-p2)^6), t_layer = %g us, T1 = %g us, "
            "T2 = %g us; scale s multiplies error rates and divides T1, T2. Backend: %s. "
            "2q-equivalent: cx = 1, ccx = 6.\n"
            % (", ".join(cfg["decision_ids"]), n1["p1"], n1["p2"], n1["p3"], n1["t_layer_us"], n1["T1_us"],
               n1["T2_us"], cfg["backend"]))


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
    res = train_mod.train(p, noise, lam_a, lam_d=lam_d, steps=steps, seed=seed, out=False)
    return {"r_final": np.asarray(res.r, dtype=float).reshape(-1).tolist(),
            "r_history": r_history(res.history).tolist()}


def sweep(train_mod, p, scales, lams, lam_d, steps) -> dict[str, np.ndarray]:
    """Final r per noise scale, each of shape (len(lams), N), from A's train.sweep_lambda:
    one jitted trainer vmapped over lam_a. A builds the noise as NoiseParams().scaled(s)
    (s = 0 -> NoiseParams.none()), the same noise get_backend("a") gives. Results are
    kept in results/<problem>.json, so sweep_lambda's own JSON is not written."""
    res = train_mod.sweep_lambda(p, scales=tuple(scales), lam_as=np.asarray(lams), lam_d=lam_d,
                                 steps=steps, out=False)
    return {str(float(s)): np.asarray(res["runs"][str(float(s))]["r"], dtype=float).reshape(len(lams), -1)
            for s in scales}


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

def _r_label(cfg, i) -> str:
    ids = cfg["decision_ids"]
    return ("r = σ(z), probability of freeing (unitless)" if len(ids) == 1
            else "r(%s), unitless" % ids[i])


def plot_training(training, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cfg["decision_ids"])
    fig, axes = plt.subplots(n, len(TRAIN_SCALES), figsize=(11, 4.2 + 3.2 * (n - 1)), sharey=True,
                             sharex=True, squeeze=False)
    for i in range(n):
        for j, s in enumerate(TRAIN_SCALES):
            ax = axes[i, j]
            ends = []
            for k, lam in enumerate(TRAIN_LAMS):
                h = np.asarray(training[str(s)][str(lam)]["r_history"])[:, i]
                ax.plot(np.arange(len(h)), h, color=SERIES[k], linewidth=2, label="λa = %g" % lam)
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
            scale = "noiseless" if s == 0 else "noise scale s = %g" % s
            ax.set_title(scale if n == 1 else "%s, %s" % (cfg["decision_ids"][i], scale), color=INK, fontsize=11)
            if i == n - 1:
                ax.set_xlabel("training step (Adam iteration)")
            ax.set_ylim(-0.03, 1.03)
            ax.margins(x=0.15)
            style_axes(ax)
        axes[i, 0].set_ylabel(_r_label(cfg, i))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    top = 0.9 if n == 1 else 0.93
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(TRAIN_LAMS), fontsize=9,
               bbox_to_anchor=(0.5, top + 0.03))
    fig.suptitle("%s: soft decisions over training" % display_name(cfg), color=INK, fontsize=12)
    fig.text(0.5, 0.005, noise_caption(cfg), ha="center", fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05 if n == 1 else 0.03, 1, top))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sweep(sweeps, flips_by_scale, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(cfg["decision_ids"])
    fig, axes = plt.subplots(1, n, figsize=(8.5 if n == 1 else 6.5 * n, 4.8), sharey=True, squeeze=False)
    marks = sorted((f["lam_a"], s) for s in NOISE_SCALES for f in flips_by_scale[str(s)])
    groups = []
    for lam, s in marks:
        if groups and lam <= groups[-1][0] * 1.02:
            groups[-1][1].append(s)
        else:
            groups.append([lam, [s]])
    for i in range(n):
        ax = axes[0, i]
        for k, s in enumerate(NOISE_SCALES):
            r = np.asarray(sweeps[str(s)]["r_final"])[:, i]
            ax.plot(SWEEP_LAMS, r, color=SERIES[k], linewidth=2, marker=MARKERS[k], markersize=6,
                    markeredgecolor="white", markeredgewidth=1, label="noiseless" if s == 0 else "s = %g" % s)
            for f in flips_by_scale[str(s)]:
                ax.axvline(f["lam_a"], color=SERIES[k], linewidth=1.2, linestyle="--")
        # One annotation per group of flips that land within 2% of each other.
        for g, (lam, ss) in enumerate(groups):
            names = ", ".join("noiseless" if s == 0 else "s = %g" % s for s in sorted(ss))
            ax.annotate("%s flip\nλa ≈ %.3f" % (names, lam), (lam, 0.12 + 0.22 * g), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK)
        ax.axhline(0.5, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xscale("log")
        ax.set_xlabel("λa, weight on peak live ancillas (loss units per ancilla, log scale)")
        ax.set_ylim(-0.03, 1.03)
        if n > 1:
            ax.set_title(cfg["decision_ids"][i], color=INK, fontsize=11)
        style_axes(ax)
    axes[0, 0].set_ylabel("final " + _r_label(cfg, 0) if n == 1 else "final r, probability of freeing (unitless)")
    axes[0, 0].legend(frameon=False, loc="upper left", fontsize=9,
                      title="trained r (line), brute-force flip (dashed)", title_fontsize=8)
    fig.suptitle("%s: trained decisions vs λa" % display_name(cfg), color=INK, fontsize=12)
    fig.text(0.5, 0.005, noise_caption(cfg), ha="center", fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="toy", choices=sorted(circuits.BENCHMARKS))
    ap.add_argument("--backend", default="auto", choices=["auto", "a", "reference"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lam-d", type=float, default=0.01)
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer steps, for smoke tests")
    args = ap.parse_args()
    steps = 40 if args.quick else args.steps
    t_start = time.time()

    make_noise, simulator, label = get_backend(args.backend)
    if not args.no_train and not label.startswith("A:"):
        ap.error("training uses A's simulator; use --backend a (or auto) or pass --no-train")
    p = E.fill_costs(circuits.BENCHMARKS[args.problem]())
    name = p.name
    cfg = {"script": "experiments/integration.py", "problem": name,
           "decision_ids": [d.id for d in p.decisions], "backend": label, "lam_d": args.lam_d,
           "steps": steps, "train_lams": TRAIN_LAMS, "train_scales": TRAIN_SCALES, "sweep_lams": SWEEP_LAMS,
           "noise_scales": NOISE_SCALES, "band_rel": BAND_REL, "seed": 0, "git": git_rev(),
           "noise": {str(s): noise_dict(make_noise(s)) for s in NOISE_SCALES}}
    out = {"config": cfg}

    # Table and brute force.
    rows = schedule_rows(p, make_noise, simulator)
    tables = {str(s): B.schedule_table(p, make_noise(s), simulator) for s in NOISE_SCALES}
    flips = {s: B.flip_thresholds(t, args.lam_d) for s, t in tables.items()}
    out.update(table=rows, brute_force={"tables": tables, "flip_thresholds": flips})
    RESULTS.mkdir(exist_ok=True)
    write_json("%s_table" % name, {"config": cfg, "rows": rows})
    (RESULTS / ("%s_table.md" % name)).write_text(table_markdown(rows, cfg))
    print(table_markdown(rows, cfg))
    for s, fl in flips.items():
        print("brute-force flip, s = %s: %s" % (s, ["%.4f (%s)" % (f["lam_a"], "".join(map(str, f["to"])))
                                                    for f in fl]))

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
        plot_training(training, cfg, FIGURES / ("%s_training.png" % name))

        sweeps, mismatches = {}, {}
        t0 = time.time()
        r_by_scale = sweep(train_mod, p, NOISE_SCALES, SWEEP_LAMS, args.lam_d, steps)
        print("sweep: %d noise scales x %d lam_a via train.sweep_lambda (%.1fs)"
              % (len(NOISE_SCALES), len(SWEEP_LAMS), time.time() - t0))
        for s in NOISE_SCALES:
            r_final = r_by_scale[str(s)]
            sweeps[str(s)] = {"r_final": r_final.tolist(), "method": "train.sweep_lambda",
                              "rounded": [E.round_decisions(r) for r in r_final]}
            mismatches[str(s)] = mismatch_report(SWEEP_LAMS, r_final, tables[str(s)], flips[str(s)], args.lam_d)
            print("sweep s=%g: %d mismatches with brute force, %d outside the band"
                  % (s, len(mismatches[str(s)]), sum(not m["near_threshold"] for m in mismatches[str(s)])))
        plot_sweep(sweeps, flips, cfg, FIGURES / ("%s_lambda_sweep.png" % name))
        out.update(training=training, sweep=sweeps, mismatches=mismatches)
        bad = [(s, m) for s, ms in mismatches.items() for m in ms if not m["near_threshold"]]
        if bad:
            print("WARNING: trained-and-rounded schedules disagree with brute force outside the band "
                  "(instructions.md says stop and report):")
            for s, m in bad:
                print("  s=%s lam_a=%.4f trained %s brute force %s r=%s" % (s, m["lam_a"], m["trained"],
                                                                          m["brute_force"], m["r"]))
    out["seconds"] = time.time() - t_start
    print("wrote", write_json(name, out), "in %.1fs" % out["seconds"])


if __name__ == "__main__":
    main()
