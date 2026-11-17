"""Build the standalone progress report.

    uv run python website/build.py

Fills progress_source.html's {{tokens}} from results/*.json, inlines every ../figures/*.png as a data URI,
and writes website/progress.html.
"""
from __future__ import annotations

import base64
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SITE = ROOT / "website"
SCALES = ("0.0", "0.5", "1.0", "2.0")


def load(name: str) -> dict:
    return json.loads((RESULTS / ("%s.json" % name)).read_text())


def scale_label(s: str) -> str:
    return "noiseless" if float(s) == 0 else "s = %g" % float(s)


def schedule_label(row: dict, ids: list[str]) -> str:
    if not row["schedule"].startswith("d = "):
        return row["schedule"]
    freed = [i for i, bit in zip(ids, row["d"]) if bit]
    what = "free both" if len(freed) == len(ids) > 1 else "free " + ", ".join(freed)
    return "%s (%s)" % (row["schedule"], what)


def schedule_table(name: str) -> str:
    data = load("%s_table" % name)
    ids = data["config"]["decision_ids"]
    head = ["schedule", "peak<br>ancillas", "physical<br>qubits", "depth<br>(layers)", "2q-equiv.<br>gates"] + \
           ["F<br>%s" % scale_label(s) for s in SCALES]
    body = []
    for r in data["rows"]:
        cells = [schedule_label(r, ids), r["peak_ancillas"], r["physical_qubits"], r["depth"], r["two_qubit_equiv"]] + \
                ["%.4f" % r["fidelity"][s] for s in SCALES]
        body.append("<tr><td>%s</td>%s</tr>" % (cells[0], "".join('<td class="num">%s</td>' % c for c in cells[1:])))
    return ('<div class="table-wrap"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>'
            % ("".join("<th>%s</th>" % h for h in head), "".join(body)))


def flip_rows(name: str):
    res = load(name)
    rows = load("%s_table" % name)["rows"]
    lam_d = res["config"]["lam_d"]
    n = len(res["config"]["decision_ids"])
    hold = next(r for r in rows if r["schedule"] == "hold, reassigned")
    free = next(r for r in rows if list(r["d"]) == [1] * n)
    d_peak = hold["peak_ancillas"] - free["peak_ancillas"]
    for s in SCALES:
        gap = hold["fidelity"][s] - free["fidelity"][s]
        predicted = (gap + lam_d * (free["depth"] - hold["depth"])) / d_peak
        brute = res["brute_force"]["flip_thresholds"][s][0]["lam_a"]
        yield s, gap, free["depth"] - hold["depth"], predicted, brute


def flip_check() -> str:
    head = ["benchmark", "noise", "F<sub>hold</sub> − F<sub>free</sub>", "extra depth", "predicted λ<sub>a</sub>*",
            "brute-force flip"]
    body = []
    for name, label in (("toy", "toy"), ("bench2", "benchmark 2")):
        for s, gap, dd, pred, brute in flip_rows(name):
            body.append("<tr><td>%s</td><td>%s</td>%s</tr>" % (
                label, scale_label(s),
                "".join('<td class="num">%s</td>' % c for c in ("%.4f" % gap, dd, "%.4f" % pred, "%.4f" % brute))))
    return ('<div class="table-wrap"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>'
            % ("".join("<th>%s</th>" % h for h in head), "".join(body)))


def match_counts(name: str) -> tuple[int, int]:
    res = load(name)
    total = len(res["config"]["sweep_lams"]) * len(res["config"]["noise_scales"])
    wrong = sum(len(v) for v in res["mismatches"].values())
    return total - wrong, total


def matches(*names: str) -> str:
    counts = [match_counts(n) for n in names]
    return "%d/%d" % (sum(c[0] for c in counts), sum(c[1] for c in counts))


def flips(name: str) -> str:
    res = load(name)
    return ", ".join("%.4f" % res["brute_force"]["flip_thresholds"][s][0]["lam_a"] for s in SCALES)


def noise_text() -> str:
    n = load("toy")["config"]["noise"]["1.0"]
    p3 = 1 - (1 - n["p2"]) ** 6
    return ("p<sub>1</sub> = %g, p<sub>2</sub> = %g, p<sub>3</sub> = 1 − (1 − p<sub>2</sub>)<sup>6</sup> ≈ %.4f, "
            "t<sub>layer</sub> = %g µs, T<sub>1</sub> = %g µs, T<sub>2</sub> = %g µs"
            % (n["p1"], n["p2"], p3, n["t_layer_us"], n["T1_us"], n["T2_us"]))


def inline_figures(html: str) -> str:
    def repl(m):
        data = base64.b64encode((FIGURES / m.group(1)).read_bytes()).decode("ascii")
        return 'src="data:image/png;base64,%s"' % data
    return re.sub(r'src="\.\./figures/([\w.-]+\.png)"', repl, html)


def main():
    toy = load("toy")["config"]
    tokens = {
        "toy_table": schedule_table("toy"),
        "bench2_table": schedule_table("bench2"),
        "flip_check": flip_check(),
        "toy_matches": matches("toy"),
        "bench2_matches": matches("bench2"),
        "all_matches": matches("toy", "bench2"),
        "toy_flips": flips("toy"),
        "bench2_flips": flips("bench2"),
        "noise": noise_text(),
        "lam_d": "%g" % toy["lam_d"],
        "steps": str(toy["steps"]),
    }
    html = (SITE / "progress_source.html").read_text(encoding="utf-8")
    for key, value in tokens.items():
        html = html.replace("{{%s}}" % key, value)
    left = re.findall(r"\{\{\w+\}\}", html)
    if left:
        raise SystemExit("unfilled tokens: %s" % sorted(set(left)))
    html = inline_figures(html)
    if "../figures/" in html:
        raise SystemExit("a figure reference was not inlined")
    out = SITE / "progress.html"
    out.write_text(html, encoding="utf-8")
    print("wrote %s (%.0f KB)" % (out, out.stat().st_size / 1024))


if __name__ == "__main__":
    main()
