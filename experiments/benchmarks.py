"""B3 report: baselines and brute force on every benchmark at every noise scale.

    uv run python experiments/benchmarks.py [--backend auto|a|reference] [--lam-d 0.01]

Writes results/benchmarks.json and results/benchmarks.md, and each benchmark
Problem (with fill_costs applied) to results/problems/<name>.json for A and C.
"""
from __future__ import annotations

import argparse
import time

from common import (NOISE_SCALES, RESULTS, circuits, get_backend, git_rev, md_table, noise_dict,
                    write_json)

import baselines as B
import evaluate as E
import problem as P


def run(backend: str, lam_d: float) -> dict:
    make_noise, simulator, label = get_backend(backend)
    out = {"config": {"script": "experiments/benchmarks.py", "backend": label, "lam_d": lam_d,
                      "noise_scales": list(NOISE_SCALES), "git": git_rev(),
                      "noise": {str(s): noise_dict(make_noise(s)) for s in NOISE_SCALES}},
           "benchmarks": {}}
    (RESULTS / "problems").mkdir(parents=True, exist_ok=True)
    for name, build in circuits.BENCHMARKS.items():
        p = E.fill_costs(build())
        P.to_json(p, RESULTS / "problems" / ("%s.json" % name))
        rec = {"n_qubits": p.n_qubits, "decisions": [d.id for d in p.decisions],
               "extra_depth": [d.extra_depth for d in p.decisions],
               "extra_2q": [d.extra_2q for d in p.decisions], "scales": {}}
        for s in NOISE_SCALES:
            t0 = time.time()
            noise = make_noise(s)
            table = B.schedule_table(p, noise, simulator)
            rec["scales"][str(s)] = {
                "no_reuse": B.no_reuse(p, noise, simulator),
                "hold_all_reassigned": B.hold_all_reassigned(p, noise, simulator),
                "table": table,
                "flip_thresholds": B.flip_thresholds(table, lam_d),
                "seconds": time.time() - t0,
            }
        out["benchmarks"][name] = rec
    return out


def to_markdown(res: dict) -> str:
    cfg = res["config"]
    txt = ["# Benchmarks: baselines and brute force\n",
           "Backend: %s. lam_d = %g. Git %s.\n" % (cfg["backend"], cfg["lam_d"], cfg["git"])]
    for name, rec in res["benchmarks"].items():
        scales = list(rec["scales"])
        txt.append("\n## %s (%d qubits, decisions %s, extra_depth %s, extra_2q %s)\n\n"
                   % (name, rec["n_qubits"], rec["decisions"], rec["extra_depth"], rec["extra_2q"]))
        first = rec["scales"][scales[0]]
        rows = []
        for label, r0 in [("no reuse", first["no_reuse"]), ("hold, reassigned", first["hold_all_reassigned"])] + \
                [("d = %s" % "".join(map(str, row["d"])), row) for row in first["table"]]:
            key = "no_reuse" if label == "no reuse" else "hold_all_reassigned" if label.startswith("hold") else None
            fids = []
            for s in scales:
                sc = rec["scales"][s]
                row = sc[key] if key else next(x for x in sc["table"] if x["d"] == r0["d"])
                fids.append("%.4f" % row["fidelity"])
            rows.append([label, r0["peak_ancillas"], r0["physical_qubits"], r0["depth"],
                         r0["two_qubit_equiv"]] + fids)
        txt.append(md_table(["schedule", "peak ancillas", "physical qubits", "depth (layers)", "2q-equiv"]
                            + ["F (s=%s)" % s for s in scales], rows))
        txt.append("\nBrute-force flip thresholds in lam_a (J linear in lam_a):\n\n")
        for s in scales:
            fl = rec["scales"][s]["flip_thresholds"]
            txt.append("- s = %s: %s\n" % (s, ", ".join("%.4f (%s to %s)" % (f["lam_a"], "".join(map(str, f["from"])),
                                                                          "".join(map(str, f["to"]))) for f in fl) or "none"))
    return "".join(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "a", "reference"])
    ap.add_argument("--lam-d", type=float, default=0.01)
    args = ap.parse_args()
    res = run(args.backend, args.lam_d)
    path = write_json("benchmarks", res)
    md = to_markdown(res)
    (RESULTS / "benchmarks.md").write_text(md)
    print(md)
    print("wrote", path, "and", RESULTS / "benchmarks.md")
