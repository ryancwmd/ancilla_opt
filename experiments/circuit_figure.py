"""Circuit diagrams of the benchmark Problems with ancilla lifetimes, decision gaps and live counts.

    uv run python experiments/circuit_figure.py

Draws every Problem in circuits.BENCHMARKS from its gate list, ancillas and decisions, and writes
figures/circuits.png.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle

from common import FIGURES, GRID, INK, INK2, SEQUENTIAL, SERIES, SURFACE, circuits, display_name

import evaluate as E
import problem as P

QUBIT_NAMES = {
    "toy": ["c1", "c2", "c3", "c4", "a", "b", "t1", "t2"],
    "bench2": ["c1", "c2", "c3", "c4", "a", "b", "c", "t1", "t2"],
}
HELD = SEQUENTIAL[1]
GAP = SERIES[1]
X_MIN, X_MAX = -3.6, 10.7


def draw_gate(ax, name, qs, x):
    ys = [-q for q in qs]
    if name == "h":
        ax.add_patch(Rectangle((x - 0.24, ys[0] - 0.24), 0.48, 0.48, facecolor=SURFACE, edgecolor=INK, linewidth=1.2,
                               zorder=4))
        ax.text(x, ys[0], "H", ha="center", va="center", fontsize=8, color=INK, zorder=5)
        return
    ax.plot([x, x], [min(ys), max(ys)], color=INK, linewidth=1.2, zorder=3)
    for y in ys[:-1]:
        ax.add_patch(Circle((x, y), 0.1, color=INK, zorder=4))
    ax.add_patch(Circle((x, ys[-1]), 0.2, facecolor=SURFACE, edgecolor=INK, linewidth=1.2, zorder=4))
    ax.plot([x - 0.2, x + 0.2], [ys[-1], ys[-1]], color=INK, linewidth=1.2, zorder=5)
    ax.plot([x, x], [ys[-1] - 0.2, ys[-1] + 0.2], color=INK, linewidth=1.2, zorder=5)


def draw_problem(ax, p):
    names = QUBIT_NAMES[p.name]
    T = len(p.gates)
    n = p.n_qubits
    ancilla_qubits = {a.qubit for a in p.ancillas}
    for q in range(n):
        ax.plot([-1.7, T - 0.4], [-q, -q], color=INK2, linewidth=0.9, zorder=1)
        label = names[q] + (" (ancilla)" if q in ancilla_qubits else "")
        ax.text(-1.9, -q, label, ha="right", va="center", fontsize=9, color=INK)
    for a in p.ancillas:
        ax.add_patch(Rectangle((a.alloc_step - 0.45, -a.qubit - 0.34), a.release_step - a.alloc_step + 0.9, 0.68,
                               facecolor=HELD, edgecolor="none", zorder=2))
    for dec in p.decisions:
        gap = P.gap_gates(p, dec)
        x0, x1 = gap.start - 0.45, gap.stop - 1 + 0.45
        ax.add_patch(Rectangle((x0, -dec.ancilla - 0.34), x1 - x0, 0.68, facecolor="none", edgecolor=GAP,
                               hatch="////", linewidth=1.4, zorder=2))
        ax.text((x0 + x1) / 2, -dec.ancilla + 0.42, dec.id, ha="center", va="bottom", fontsize=8.5, color=INK,
                zorder=6, bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))
    for g in p.prep:
        draw_gate(ax, g[0], g[1], -1.0)
    for t, g in enumerate(p.gates):
        draw_gate(ax, g[0], g[1], t)
    top = 0.95
    ax.text(-1.0, top, "prep", ha="center", va="bottom", fontsize=8, color=INK2)
    for t in range(T):
        ax.text(t, top, str(t), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.text(-1.9, top, "step", ha="right", va="bottom", fontsize=8, color=INK2)
    hold = E.live_counts(p, E.zeros(p))
    free = E.live_counts(p, (1,) * len(p.decisions))
    rows = [("live ancillas, hold all (d = %s)" % ("0" * len(p.decisions)), hold),
            ("live ancillas, free all (d = %s)" % ("1" * len(p.decisions)), free)]
    for k, (label, counts) in enumerate(rows):
        y = -n - 0.35 - 0.8 * k
        ax.text(-1.9, y, label, ha="right", va="center", fontsize=8.5, color=INK)
        peak = max(counts)
        for t, c in enumerate(counts):
            ax.text(t, y, str(c), ha="center", va="center", fontsize=9, color=INK,
                    fontweight="bold" if c == peak > min(counts) else "normal")
        ax.text(T - 0.1, y, "peak %d" % peak, ha="left", va="center", fontsize=8.5, color=INK2)
    ax.plot([-1.7, T - 0.4], [-n + 0.2, -n + 0.2], color=GRID, linewidth=0.8)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-n - 1.3, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("%s: %d qubits, %d decision%s" % (display_name({"problem": p.name}), n, len(p.decisions),
                                                   "" if len(p.decisions) == 1 else "s"),
                 loc="left", color=INK, fontsize=11)


def main():
    problems = [circuits.BENCHMARKS[name]() for name in ("toy", "bench2")]
    heights = [p.n_qubits + 2.8 for p in problems]
    fig, axes = plt.subplots(len(problems), 1, figsize=(10, 0.62 * sum(heights)),
                             gridspec_kw={"height_ratios": heights})
    for ax, p in zip(axes, problems):
        draw_problem(ax, p)
    handles = [Patch(facecolor=HELD, edgecolor="none", label="ancilla holds a value\n(hold schedule)"),
               Patch(facecolor="none", edgecolor=GAP, hatch="////",
                     label="decision gap: if freed, the\nancilla is uncomputed after the\nearlier use and recomputed\nbefore the later use")]
    axes[0].legend(handles=handles, loc="upper left", frameon=False, fontsize=8.5, labelspacing=1.2,
                   bbox_to_anchor=((7.3 - X_MIN) / (X_MAX - X_MIN), 0.82))
    fig.text(0.5, 0.006, "Gates: ● control, ⊕ target, H Hadamard (prep). Bold counts mark steps at the peak.\n"
             "Freeing every gap lowers the peak by one; on benchmark 2, freeing only one gap leaves it at 2.",
             ha="center", fontsize=8, color=INK2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / "circuits.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
