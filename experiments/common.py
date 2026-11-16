"""Shared helpers for experiment scripts (workstream B).

Scripts run as `uv run python experiments/<name>.py` from the repo root. They
use A's noise.NoiseParams and sim.run_discrete when those modules import, and
otherwise fall back to B's numpy stand-ins (circuits.RefNoise and
circuits.reference_run_discrete), recording which one ran in the results.
"""
from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import circuits  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
NOISE_SCALES = (0.0, 0.5, 1.0, 2.0)


def have_a() -> bool:
    try:
        import noise  # noqa: F401
        import sim  # noqa: F401
        return hasattr(sim, "run_discrete") and hasattr(noise, "NoiseParams")
    except ImportError:
        return False


def get_backend(which: str = "auto"):
    """Return (noise_factory(s), simulator, label). which: auto | a | reference."""
    if which == "auto":
        which = "a" if have_a() else "reference"
    if which == "a":
        import noise
        import sim
        base = noise.NoiseParams()
        return (lambda s: noise.NoiseParams.none() if s == 0 else base.scaled(s)), sim.run_discrete, "A: sim.run_discrete"
    if which == "reference":
        base = circuits.RefNoise()
        return (lambda s: circuits.RefNoise.none() if s == 0 else base.scaled(s)), \
            circuits.reference_run_discrete, "B: circuits.reference_run_discrete (numpy cross-check)"
    raise ValueError("unknown backend %r" % (which,))


def noise_dict(noise) -> dict:
    out = {}
    for f in ("p1", "p2", "p3", "t_layer_us", "T1_us", "T2_us", "source"):
        v = getattr(noise, f, None)
        out[f] = None if v is None else (str(v) if isinstance(v, float) and math.isinf(v) else v)
    return out


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _jsonable(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _jsonable(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, complex):
        return [x.real, x.imag]
    if isinstance(x, float) and math.isinf(x):
        return str(x)
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return _jsonable(x.item())
        if isinstance(x, np.ndarray):
            return _jsonable(x.tolist())
    except ImportError:
        pass
    if hasattr(x, "tolist"):          # jax arrays
        return _jsonable(x.tolist())
    return x


def write_json(name: str, obj) -> pathlib.Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / ("%s.json" % name)
    path.write_text(json.dumps(_jsonable(obj), indent=1))
    return path


def md_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"
