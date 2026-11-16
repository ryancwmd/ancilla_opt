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

## Workstream B status (branch `agent-b`)

- [x] Environment (uv, Python 3.11, pinned `requirements.txt`) and git repo
- [x] `problem.py` on main
- [x] B1: `circuits.toy()`, statevector reference, `fill_target`, validation tests (one per rule)
- [x] B2: `evaluate.py`: rounding, live intervals and counts, peak, assignment, depth, 2q-equiv, `fill_costs`, `evaluate`
- [x] B3: `baselines.py` (`no_reuse`, `hold_all_reassigned`, `brute_force`, `flip_thresholds`), `circuits.bench2()`
  with its hand-computed table, and `experiments/benchmarks.py` writing `results/benchmarks.{json,md}`
- [~] I: `experiments/toy.py` written and smoke-tested with a fake trainer; waits on A's merged `train.py`/`sim.py`/`noise.py`
- Tests: `uv run pytest tests/test_circuits.py tests/test_evaluate.py`, 52 passed

### Things other agents should know (details in NOTES.md, section [B] Report)

- **A**: `circuits.toy()` and `circuits.bench2()` give validated Problems with target states.
  `circuits.reference_state(p, d)` is the noiseless numpy state of the physical schedule.
  `circuits.reference_run_discrete(p, d, noise)` is a numpy noisy cross-check that reproduces the doc's
  reference fidelities. `run_discrete` must accept a Problem with `ancillas=()`, `decisions=()` and `d=()`,
  which `baselines.no_reuse` uses. `toy.py` expects `train(...)` to return `.r` and `.history`, with per-step `r`.
- **C**: the hand-written Problems are in `results/problems/{toy,bench2}.json` (`fill_costs` applied) for the
  C3 acceptance check. `toy.py` reads `reqomp_adapter/results/toy.json` as a list under `"runs"` with
  `budget`, `qubits`, `depth`, `two_qubit_equiv`.
- Noise objects are duck-typed. Anything with `p1`, `p2`, `p3`, `t_layer_us`, `T1_us` and `T2_us` works in B's code.
- Brute force: `B.brute_force(p, noise, lam_a, lam_d, simulator=sim.run_discrete)`. Use
  `B.schedule_table` with `B.flip_thresholds` for exact λa flip points.
