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

## [B] Benchmark 2 (`circuits.bench2()`)

9 qubits: 0=c1 1=c2 2=c3 3=c4 4=a 5=b 6=c 7=t1 8=t2. Prep: h on 0–3.

| Step | Gate | Role |
| --- | --- | --- |
| 0 | ccx(0,1,4) | compute a = c1 AND c2 |
| 1 | ccx(4,2,7) | use 1 of a |
| 2 | ccx(2,3,5) | compute b = c3 AND c4 |
| 3 | cx(5,8) | use b |
| 4 | ccx(2,3,5) | uncompute b |
| 5 | ccx(4,3,8) | use 2 of a |
| 6 | ccx(1,2,6) | compute c = c2 AND c3 |
| 7 | cx(6,7) | use c |
| 8 | ccx(1,2,6) | uncompute c |
| 9 | ccx(4,3,7) | use 3 of a |
| 10 | ccx(0,1,4) | final uncompute of a |

Ancillas: a = (4, 0, 10), b = (5, 2, 4), c = (6, 6, 8). Decisions: `a_gap1` (free_after 1, recompute_before 5)
and `a_gap2` (free_after 5, recompute_before 9). Hand-computed table, physical schedules (also in
`tests/test_evaluate.py`):

| d | live counts, steps 0–10 | peak | physical qubits | depth | extra 2q-equiv | noiseless F |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | 1 1 2 2 2 1 2 2 2 1 1 | 2 | 8 (b and c share qubit 5) | 9 | 0 | 1 |
| 10 | 1 1 1 1 1 1 2 2 2 1 1 | 2 | 8 | 11 | 12 | 1 |
| 01 | 1 1 2 2 2 1 1 1 1 1 1 | 2 | 8 | 13 | 12 | 1 |
| 11 | all 1 | 1 | 7 | 15 | 24 | 1 |

No-reuse baseline: 9 physical qubits, depth 9. Design points: freeing only one gap never lowers the peak, so
the decisions are not separable (the only brute-force flip goes from 00 straight to 11). Gap 2 costs more
depth than gap 1 (4 layers against 2) because c's compute shares control c2 with a's compute and serialises.
Without reassignment the logical depths are 9, 9, 13, 13: gap 1's uncompute runs beside b's compute. And b
and c never overlap, so reassignment merges them even when everything is held. That is the case in the loss
section where the Δᵢ baseline must be the physical hold schedule.

## [B] Report (2026-11-16)

**Milestones:** B1, B2 and B3 are done. The integration script (I) is written and smoke-tested, but it waits
on A's `train.py`, `sim.py` and `noise.py`. pytest on `agent-b`: `52 passed`.

**Files:** `problem.py` (on main), `circuits.py`, `evaluate.py`, `baselines.py`, `experiments/common.py`,
`experiments/benchmarks.py`, `experiments/toy.py`, `tests/test_circuits.py`, `tests/test_evaluate.py`,
`results/benchmarks.{json,md}`, `results/toy_table.{json,md}`, `results/toy.json`,
`results/problems/{toy,bench2}.json` (fill_costs applied, for A and C).

**Checks against the doc:** every value in the toy's hand-computed table matches. The numpy noisy reference
reproduces the doc's reference numbers to 3 decimals: F_hold 0.7337, 0.5430, 0.3060 and F_free 0.6846,
0.4745, 0.2381 at s = 0.5, 1, 2. Exact brute-force flips are 0.0200 noiseless, then 0.0691, 0.0884, 0.0879.
No hand-computed value in the doc looked wrong. One editorial issue: the toy's "Reference numbers" paragraph
appears twice in instructions.md.

**Findings so far (reference backend):** the toy flip falls slightly from s = 1 to s = 2 (0.0884 to 0.0879),
because both fidelities drop by about the same amount. Bench2's flip (00 to 11) is 0.060 noiseless, then
0.185, 0.2015 and 0.155 at s = 0.5, 1, 2. That is non-monotone in noise and lies inside the planned sweep range.

**Deviations and additions:**
- `evaluate.fidelity`/`evaluate` raise `ValueError` if given *noisy* parameters without a simulator, instead
  of silently returning a noiseless number. `None` or all-zero noise such as `NoiseParams.none()` is fine.
- `circuits.reference_run_discrete(problem, d, noise)` is a numpy density-matrix stand-in with the same
  signature as A's `run_discrete`. `circuits.RefNoise` mirrors `NoiseParams`' fields, defaults, `none()` and
  `scaled()`. Both are for cross-checks and for running before A lands; reported numbers should use A's.
  The reference applies depolarizing noise as a product of one-qubit twirls and idle noise as a closed-form
  block update. Tests check both against the Kraus and Pauli-sum forms.
- `baselines.brute_force(problem, noise, lam_a, lam_d=0.01, simulator=None, table=None)`: the optional
  `table` reuses one `schedule_table` across many λa. Exact ties go to the earlier schedule in binary
  counting order, so holding wins a tie. Added `flip_thresholds(table, lam_d)`: J is linear in λa, so the
  exact flip points come from the lower envelope.
- `two_qubit_equiv` counts the logical schedule's gates. That is the same multiset as the physical schedule.
- `baselines.no_reuse` runs `simulator(replace(problem, ancillas=(), decisions=()), (), noise)`. See the
  problem.py note: **A's run_discrete must accept a Problem with no ancillas or decisions.**
- `experiments/toy.py`, flags `--no-train`, `--quick`, `--backend`, `--steps`:
  - A trained-and-rounded schedule that disagrees with brute force counts as "near threshold" if λa is within
    ±15% of a brute-force flip (`BAND_REL`). Anything outside the band is printed as a WARNING.
  - Expected from A: `train.train(problem, noise, lam_a, lam_d=..., steps=..., seed=...)` returns an object
    with `.r` (shape (N,)) and `.history`. `.history` is a list of per-step dicts with key `"r"`, or a dict of
    lists with key `"r"`.
  - `train.sweep_lambda` is used if its λ argument is named one of `lam_as`, `lam_a`, `lams`, `lambdas`,
    `lam_values`. It should return final r with shape (n_lambda, N), or a dict or object with `r`. Otherwise
    the script loops over `train.train`.

**Blocking other workstreams:** nothing. A can use `circuits.toy()`/`bench2()` and `circuits.reference_state`
or `reference_run_discrete` for cross-checks.

**Waiting on:**
- A: `train.py`, `sim.py` and `noise.py` merged, for `experiments/toy.py` and the figures.
- C: `reqomp_adapter/results/toy.json`. `toy.py` reads a list under `"runs"` (or `"budgets"`). Each entry
  should have `budget`, `qubits`, `depth`, `two_qubit_equiv`, and optionally `peak_ancillas` and
  `fidelity_<s>`. Other key names are tried defensively.

**Open questions for the humans:** whether ±15% is the right "narrow band" for trained vs brute-force
disagreement. The SmoothMax offset at β = 20 shifts the toy's effective peak gap from 1 to about 0.96.

## [A] Simulator, noise, loss, training (2026-11-16)

**Milestones:** A1, A2, A3 done. `uv run pytest` → `62 passed, 1 skipped` (skip = cross-check against B's
`circuits.toy()`, which runs automatically once `circuits.py` is merged).

**Files:** `sim.py`, `noise.py`, `loss.py`, `train.py`, `tests/test_sim.py`, `tests/test_noise.py`,
`tests/test_loss.py`, plus `tests/toy_fixture.py` (see deviations).

**Checks against the spec:** toy noiseless F = 1.0 for d = 0 and 1. Noisy toy F_hold/F_free =
0.7337/0.6846, 0.5430/0.4745, 0.3060/0.2381 at s = 0.5, 1, 2 (matches the reference numbers). γ = 0.01489,
λ = 0.02469, Tφ = 240 µs, p3 = 0.0585. Sweep flips (grid points): noiseless between λa 0.0197 and 0.0234;
s = 0.5 between 0.065 and 0.077; s = 1 and 2 between 0.077 and 0.091, consistent with brute-force 0.02, 0.069, 0.088, 0.088.

**API for B (integration):**
- `sim.run_discrete(problem, d, noise=None) -> rho` (jitted; problem/d/noise are static, hashable).
  Works for `ancillas=(), decisions=(), d=()` (tested).
- `sim.fidelity(rho, psi)`, `sim.target_vector(problem)` (uses `target_state`, or computes it if empty).
- `sim.simulate_mixture(problem, r, noise)` / `simulate_mixture_exact` (v1). `sim.schedule_fidelities(problem, noise)`
  gives F_d for all d in `itertools.product((0,1), repeat=N)` order.
- `noise.NoiseParams` (`.none()`, `.scaled(s)`, `.gamma`, `.lam`, `.to_dict()`), `noise.from_fake_backend(name)` (optional, untested; qiskit not installed).
- `train.train(problem, noise, lam_a, lam_d=0.01, steps=300, lr=0.1, beta=(2, 20), seed=0, budget=None, out=None)`
  → `TrainResult(z, r, history, config)`. `history` keys: loss, F, smoothmax, peak_term, depth_term, r (steps×N), beta, grad_norm.
  `out=None` saves to `results/train_<name>_la<λ>_<noise>.json`, `out=False` skips saving.
- `train.sweep_lambda(problem, base_noise=NoiseParams(), scales=(0, .5, 1, 2), lam_as=geomspace(.01, .3, 21), ..., keep_history=False, out=None)`
  → `{"lam_a", "scales", "runs": {"0.0": {"r": [[r_i] per λ], "z", "final_F", "final_loss", "noise"}, ...}}`,
  saved to `results/sweep_<name>.json`. Whole toy sweep (4 scales × 21 λ) takes ~10 s.
- For figure 1 (r over training): call `train(...)` per λa and plot `history["r"]`.

**Decisions / deviations:**
- **v2 (`simulate_mixture` gap-local) not built** (stretch goal). `simulate_mixture` dispatches to v1.
- **ρ_d caching:** the per-schedule density matrices don't depend on r, so `sim.schedule_rhos(problem, noise)`
  computes them once (lru_cache), and the loss forms Σ_d w(d) ρ_d inside the differentiated function. Same math as v1,
  but training never re-simulates the circuit. Gradients flow only through w(d), which is exact.
- **Idle noise** = amplitude damping then phase damping, merged into one 3-operator Kraus set (the 4th product is 0).
  Applied to every qubit untouched in a layer, data and ancilla alike, including unused ancilla qubits (they sit in |0⟩, unaffected).
- **Depolarizing** applied in the trace form (1−p)ρ + p·Tr_qs ρ ⊗ I/2ᵏ, built as a composition of 1-qubit replacement maps;
  tested equal to the 4ᵏ-term Kraus form for k = 1, 2, 3.
- Zero-strength channels are skipped (so `NoiseParams.none()` leaves ρ bit-for-bit unchanged).
- `scaled(s)` caps p at 1; with `p3=None`, p3 stays derived from the scaled p2 (1 − (1 − s·p2)⁶ rather than s·p3).
  Pass an explicit `p3` to scale it linearly instead.
- `train`'s `seed` is recorded but unused: z starts at 0 as specified and nothing else is random.
- **`tests/toy_fixture.py`** (new file under tests/, not in A's list): `toy_problem()` returns B's `circuits.toy()` when importable,
  else builds the toy from the spec. Also `two_gap_problem()`, A's own 9-qubit 2-decision Problem for testing N = 2 code paths
  (not benchmark 2). B may delete it once `circuits.py` lands, if tests are switched to import `circuits` directly.

**Blocking others:** nothing. **Open questions:** none beyond the doc's.

## [I] Integration on main (2026-11-16)

- Merged `agent-b` (fast-forward) and `agent-a` into `main`. Only conflict: NOTES.md, where both sections were kept.
- **Fix in `experiments/toy.py`:** its signature-sniffing call to `train.sweep_lambda` passed already-scaled
  noise as `base_noise` while A's default `scales` rescaled it, and it read a top-level `"r"` that A's result
  doesn't have (A nests r under `runs[<s>]`). Now it calls `sweep_lambda(p, scales=NOISE_SCALES, lam_as=..., out=False)`
  once and reads `runs[str(s)]["r"]`. That is the same noise as `get_backend("a")`. `train(...)` is called
  with `out=False` so per-run JSON files don't pile up in `results/`; everything is in `results/toy.json`.
  Training with `--backend reference` is now an error, since the trainer always uses A's simulator.
- **Removed `tests/toy_fixture.py`** (A's suggestion): A's tests use `circuits.toy()` and `circuits.bench2()`.
  bench2 replaces A's stand-in `two_gap_problem()`; its lifetimes are identical (a 0–10, b 2–4, c 6–8), so the
  live-count expectations didn't change. A's spec-built-toy target check became
  `test_target_matches_independent_statevector` (A's `sim.statevector` vs B's target, toy and bench2). The
  redundant `test_toy_matches_b_reference_state` was dropped. `train.py`'s `__main__` imports `circuits` directly.
- **toy.py results (300 steps, A's simulator):** brute-force flips 0.0200, 0.0691, 0.0884, 0.0879 at
  s = 0, 0.5, 1, 2. Trained-and-rounded r matches brute force at all 84 sweep points, with 0 mismatches. The run
  takes ~15 s. Training curve at s = 1, λa = 0.1: r first drops to ~0.17, then rises to 0.88. Early on, β = 2
  makes SmoothMax barely separate a peak of 2 from a peak of 1, so infidelity dominates until β grows.

## [I] Benchmark 2 through the trainer (2026-11-17)

- `experiments/toy.py` is now `experiments/integration.py --problem <toy|bench2>` (default toy, same output
  file names as before). Figures have one panel or row per decision; the table lists every schedule that frees
  at least one gap; Reqomp rows are read from `reqomp_adapter/results/<p>.json`.
- `experiments/benchmarks.py --backend a` rerun on A's simulator: every value matches the numpy reference to
  4 decimals (toy and bench2), so `results/benchmarks.{json,md}` now cite A's backend.
- **bench2 results (300 steps, A's simulator):** brute-force flips 0.0600, 0.1850, 0.2015, 0.1551 at
  s = 0, 0.5, 1, 2, each from 00 straight to 11. Trained-and-rounded r matches brute force at all 84 sweep
  points with 0 mismatches. Both decisions flip together at every noise scale; a_gap2 (extra depth 4) ends
  slightly lower than a_gap1 (extra depth 2) just above each flip. Runs in ~50 s.
- **`experiments/landscape.py`** (two-decision Problems): evaluates `loss.loss_terms` at the final β = 20 on a
  121 x 121 grid over (r1, r2), trains each λa with `train.train`, and draws `figures/<p>_landscape.png` from
  `results/<p>_landscape.json` (`--replot` redraws without recomputing). bench2 at s = 1, λa = 0.1, 0.22,
  0.3: trained-and-rounded 00, 11, 11, equal to brute force. The 01 and 10 corners are the high-loss corners in
  every panel, and above the flip a diagonal valley runs toward 11. At λa = 0.22 and 0.3 the path first heads
  toward 00 while β is small, then turns up the valley to 11. Shared figure palette and helpers moved to
  `experiments/common.py`.
- **`experiments/circuit_figure.py`** draws `figures/circuits.png` (both benchmarks) from each Problem's gates,
  ancillas and decisions: lifetimes shaded, decision gaps hatched, live counts for hold-all and free-all.
- **Progress report:** `website/build.py` fills `website/progress_source.html` from `results/*.json` (tables,
  flip-point check, match counts) and inlines the figures, writing the standalone `website/progress.html`.
  Rebuild after rerunning experiments: `uv run python website/build.py`.
