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

## Integration status (`main`)

`agent-a` and `agent-b` are merged into `main` (checkpoint after A3 and B2/B3). `uv run pytest`: 115 passed.
`agent-c` has not started. New work branches from `main`.

- [x] A1–A3: `sim.py`, `noise.py`, `loss.py`, `train.py` (v1 exact mixture; v2 gap-local not built)
- [x] B1–B3: `problem.py`, `circuits.py`, `evaluate.py`, `baselines.py`, `circuits.bench2()`, `experiments/benchmarks.py`
- [x] I: `uv run python experiments/toy.py` (~15 s) writes `figures/toy_{training,lambda_sweep}.png`,
  `results/toy_table.{md,json}` and `results/toy.json`. Trained-and-rounded r matches brute force at all
  84 sweep points (4 noise scales × 21 λa).
- [ ] C1–C3: Reqomp. When C2 lands, rerun `toy.py`; it adds Reqomp rows to the table automatically.
- [ ] Optional: v2 `simulate_mixture`, budget-mode figures, benchmark 2 through the trainer.

## Module quick reference

- **A**: `sim.run_discrete(p, d, noise)` (jitted; works for `ancillas=(), decisions=(), d=()`),
  `sim.simulate_mixture(p, r, noise)` (= v1 `simulate_mixture_exact`), `sim.schedule_fidelities(p, noise)`,
  `sim.fidelity`, `sim.target_vector`. `noise.NoiseParams` with `.none()`, `.scaled(s)`.
  `train.train(p, noise, lam_a, lam_d=0.01, steps=300, ..., out=None)` → `TrainResult(z, r, history, config)`,
  `history` a dict of per-step arrays (`r` is steps × N). Pass `out=False` to skip its JSON file.
  `train.sweep_lambda(p, base_noise=NoiseParams(), scales=(0, .5, 1, 2), lam_as=..., out=None)` →
  `{"runs": {"<s>": {"r": [[r_i] per λa], ...}}}`; noise per run is `base_noise.scaled(s)`.
- **B**: `circuits.toy()` and `circuits.bench2()` give validated Problems with target states (A's tests use
  them directly). `circuits.reference_state` / `reference_run_discrete` are numpy cross-checks.
  `evaluate.evaluate(p, d, noise, simulator=sim.run_discrete)` gives exact metrics.
  `baselines.brute_force(p, noise, lam_a, lam_d, simulator=sim.run_discrete)`; use `baselines.schedule_table`
  with `baselines.flip_thresholds` for exact λa flip points.
- **C**: the hand-written Problems are in `results/problems/{toy,bench2}.json` (`fill_costs` applied) for the
  C3 acceptance check. `toy.py` reads `reqomp_adapter/results/toy.json` as a list under `"runs"` with
  `budget`, `qubits`, `depth`, `two_qubit_equiv`.
- Noise objects are duck-typed in B's code: anything with `p1`, `p2`, `p3`, `t_layer_us`, `T1_us`, `T2_us`.
