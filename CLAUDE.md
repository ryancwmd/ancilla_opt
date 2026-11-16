# CLAUDE.md: project guide for agents

Differentiable ancilla reuse, a JAX prototype. The full spec is `instructions.md`; read its shared sections plus
your workstream's. Cross-agent decisions and deviations go in `NOTES.md`. This file is the quick reference:
how to work in the repo, plus each workstream's status.

## Working in this repo

- Environment: `uv sync`, then `uv run pytest` / `uv run python <script>`. Python 3.11 (pinned).
- Branch from `main`: `agent-a`, `agent-b`, `agent-c`. Prefer a separate git worktree per agent.
- Only edit the files your workstream owns (table in instructions.md). `problem.py` changes go through B.
- `jax.config.update("jax_enable_x64", True)` at import; complex128. Qubit 0 is the most significant bit.

## Shared contract: problem.py (B)

Status: committed to `main`. Stdlib only, imports on Python 3.8+ (checked with 3.8.20).

| Function | Purpose |
| --- | --- |
| `validate(p)` | Raises `ValueError` for rules 1–6 plus structural checks |
| `to_json` / `from_json` | JSON round trip, complex as `[re, im]` |
| `gap_gates(p, dec)` | `range(free_after + 1, recompute_before)` |
| `build_schedule(p, d)` | Logical gate list for 0/1 vector d |
| `physical_schedule(p, d)` | Same, with ancillas reassigned by greedy interval colouring |
| `layers(gates)` | ASAP layering, list of lists of gate indices |
| `ancilla_intervals`, `assign_intervals` | Interval logic used by `physical_schedule` and `evaluate.py` |
| `compute_gates(p, anc)` | The ancilla's compute gates (rule 3 definition) |

Checked values on the toy: logical layers 7 and 7, physical layers 7 and 9, and b moves onto qubit 4 when freed.

## Workstream B status

- [x] Environment (uv, Python 3.11, pinned `requirements.txt`) and git repo
- [x] `problem.py` on main
