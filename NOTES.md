# NOTES: cross-agent notes, decisions, deviations

Each entry is tagged with its workstream letter. Append; don't rewrite others' entries.

## [B] Environment and repo (2026-11-16)

- Repo root is this directory (`ancilla_opt/`); it plays the role of `qdiff_ancilla/` in instructions.md. The layout below it is flat, as specified.
- Python environment is managed with **uv**. Python is pinned to 3.11 (`.python-version`, `requires-python = ">=3.11,<3.12"`).
  - Setup: `uv sync` (installs main deps plus the `dev` group, which holds pennylane).
  - Run anything with `uv run ...`, e.g. `uv run pytest`.
  - Add a dependency with `uv add <pkg>` (or `uv add --dev <pkg>` for test-only), then refresh pins:
    `uv export --no-hashes --format requirements-txt --all-groups --no-emit-project > requirements.txt`.
  - `requirements.txt` holds exact pins (jax 0.10.2, jaxlib 0.10.2, optax 0.2.9, numpy 2.4.6, matplotlib 3.11.2, pytest 9.2.0, pennylane 0.45.1). `uv.lock` is the source of truth.
- pytest config lives in `pyproject.toml` (`pythonpath = ["."]`, `testpaths = ["tests"]`), so tests can `import problem`, `import sim`, etc.
- Git: `main` holds the shared contract. Branches: `agent-a`, `agent-b`, `agent-c`, each branched from `main`.
  If several agents share one machine, use a separate worktree per agent so checkouts don't collide, e.g.
  `git worktree add ../ancilla_opt-A -b agent-a main` and then `uv sync` inside it.
- C's Reqomp venv lives at `reqomp_adapter/.venv` and is git-ignored.

## [B] problem.py decisions not fixed by the spec

- **Extra helpers** beyond the required functions (all stdlib): `to_dict`/`from_dict`, `controls(g)`, `target(g)`,
  `inverse_gate`, `inverse_block`, `ancilla_by_qubit`, `compute_gates(problem, ancilla)`,
  `ancilla_intervals(problem, d)`, `assign_intervals(problem, d)`, constants `GATE_ARITY`, `SELF_INVERSE`, `INVERSE_NAME`.
- **"The ancilla's compute gates"** (rule 3) = the base gates that target the ancilla from `alloc_step` up to,
  but not including, the first base gate that uses it as a control. For the toy that is just base gate 0.
- **Extra structural checks in `validate`** (beyond rules 1–6): known gate names with correct arity, distinct
  in-range qubits, data and ancilla qubits disjoint, unique decision ids, the gate at `alloc_step` and at
  `release_step` targets the ancilla, no base gate touches an ancilla outside `[alloc_step, release_step]`,
  and prep never touches an ancilla.
- **Rule 5** ("non-overlapping gaps"): for two decisions on one ancilla, require
  `d1.recompute_before <= d2.free_after` or the reverse. A shared endpoint is allowed (benchmark 2 uses it:
  the middle use of `a` ends gap 1 and starts gap 2).
- **Insertion order in `build_schedule`** at a boundary between base gates t-1 and t: uncomputes with
  `free_after = t-1` come first, then recomputes with `recompute_before = t`. Several blocks at one point keep
  the order of `problem.decisions`.
- **Physical assignment** (`assign_intervals`): intervals sorted by (start, ancilla qubit). Each takes the
  lowest free qubit from the pool `sorted(ancilla qubits)`. A qubit whose previous interval ended at step e is
  free for an interval starting at s if e < s. Inserted gates map by an anchor step: base gate t uses step t,
  an uncompute uses `free_after`, a recompute uses `recompute_before`. Physical qubits keep the original
  numbering, so `n_qubits` is unchanged and unused ancilla qubits just sit in |0>.
- `physical_schedule` leaves every non-ancilla qubit alone. So a Problem with `ancillas=()` and `decisions=()`
  gives back its base gate list unchanged. B's `baselines.no_reuse` relies on this to run the
  no-reassignment circuit through A's `run_discrete(problem, d=(), noise)`. **A: please keep `run_discrete`
  working for a Problem with no ancillas and no decisions.**
- `target_state` tolerance in rule 6: |norm − 1| ≤ 1e-8.
