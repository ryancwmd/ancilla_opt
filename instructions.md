# Agent Handoff: Differentiable Ancilla Reuse

Nov 16, 2026 · @Ryan

## Goal and context

Build a JAX prototype that uses gradient descent to decide when a quantum circuit should uncompute and recompute a scratch (ancilla) qubit, and show it working end to end on one small circuit. This is a class project for a differentiable programming course; everything runs on a laptop, with no quantum hardware.

**The problem in brief.** An ancilla holds an intermediate value. When that value is used twice, the gap between uses offers a choice: *hold* the ancilla (costs a qubit during the gap and exposes it to idle decay) or *free* it by uncomputing after the first use and recomputing before the second (frees the qubit but adds gates and gate error).

**The method in brief.** Each such choice is a probability rᵢ = σ(zᵢ) of freeing. The circuit is simulated as a density matrix, where each choice is a probabilistic mix of the two branches, so one simulation gives the average fidelity over every schedule. The loss combines infidelity, a smooth estimate of peak live ancillas, and a small depth term. After training, each rᵢ is rounded to 0 or 1 and the resulting real circuit is measured exactly, after ancillas are reassigned to shared physical qubits, so one circuit supplies every reported number.

**How this doc is used:** three agents work in parallel, one per workstream (A, B, C). Every agent reads the shared sections (this one, Workstreams, Environment, Data model, Benchmark 1, Required tests, When to stop) plus its own workstream's sections. Agents only edit the files their workstream owns; the shared data model is the contract between them; each agent works on its own git branch, and the two humans merge branches at overseen integration checkpoints.

## Workstreams and milestones

Three agents run at once: A builds the differentiable optimizer, B builds circuits and evaluation, and C integrates Reqomp. They meet at one shared file, `problem.py`, and at a final integration step.

| Workstream | Owns these files | Builds | Reads sections |
| --- | --- | --- | --- |
| **A: Simulator and optimizer** | `sim.py`, `noise.py`, `loss.py`, `train.py`, `tests/test_sim.py`, `tests/test_noise.py`, `tests/test_loss.py` | Density-matrix simulator, noise channels, soft decisions, loss, training loop | Simulator, Noise model, Loss and training |
| **B: Circuits, evaluation, baselines** | `problem.py`, `circuits.py`, `evaluate.py`, `baselines.py`, `experiments/`, `tests/test_circuits.py`, `tests/test_evaluate.py` | The data model, benchmark circuits, rounding and exact metrics, no-reuse and brute-force baselines, experiment scripts | Rounding and exact evaluation, Baselines |
| **C: Reqomp integration** | `reqomp_adapter/`, `tests/test_reqomp_adapter.py` | Run Reqomp on our benchmarks as a baseline; convert its circuit graph into our Problem format | Reqomp integration |

### Order of work

1. **B writes `problem.py` first** (within its first hour), exactly as in the Data model section, and commits it to main before the branches start. A and C code against it; until it lands, they work from the spec in this doc.
2. A, B and C then work fully in parallel on their milestones.
3. **Integration (owned by B, run at a human-overseen checkpoint; A and C flag readiness in** **`NOTES.md`):** `experiments/toy.py` runs B's toy Problem through A's trainer, evaluates with B's code, and compares with B's brute force and C's Reqomp output.

### Milestones

| # | Owner | Milestone | Done when |
| --- | --- | --- | --- |
| A1 | A | Noiseless density-matrix simulator | Simulator tests pass, including the PennyLane cross-check |
| A2 | A | Noise channels and soft decisions | Noise tests pass; soft fidelity is linear in r |
| A3 | A | Loss and training loop | Gradient check passes; training runs on the toy Problem |
| B1 | B | Data model and toy benchmark | `problem.py` committed; toy Problem validates; both schedules give fidelity 1 noiselessly; physical\_schedule puts b on a's qubit in the free schedule |
| B2 | B | Rounding and exact evaluation | Peak qubits, depth and gate counts match the hand-computed values for the toy |
| B3 | B | Baselines and benchmark 2 | Brute force runs on both benchmarks; no-reuse baseline reported |
| C1 | C | Reqomp runs | Reqomp's own examples run in a pinned environment |
| C2 | C | Reqomp baseline on our circuits | Reqomp output for both benchmarks, with qubit and gate counts recorded |
| C3 | C | Graph-to-Problem adapter | Adapter's decisions for the toy match B's hand-written toy Problem |
| I | B | Integration | `experiments/toy.py` produces the progress-report figures |

**Priority if time is short:** A1–A3, B1–B2, brute\_force and hold\_all\_reassigned (from B3), and I are what the progress report (due Nov 17) needs. Benchmark 2, C1–C3 and v2 of the mixture simulation can land later. Budget about 10 agent-hours in total; the humans merge at three checkpoints: after B1, after A3 and B2, and before the final experiments/toy.py run.

## Environment, repo layout and conventions

**Main environment (A and B):** Python 3.11, `jax[cpu]`, `optax`, `numpy`, `matplotlib`, `pytest`. Test-only: `pennylane` (for the simulator cross-check). Optional for B: `qiskit` and `qiskit-ibm-runtime`, used only to read noise parameters from a fake backend. Pin exact versions in `requirements.txt`.

**Reqomp environment (C only):** a separate virtual environment, `reqomp_adapter/.venv`, because Reqomp likely needs an older Qiskit (and possibly an older Python; use whatever its requirements need). Never install its dependencies into the main environment. The adapter communicates with the main code through JSON files only, never by import, except problem.py, which is stdlib-only and must import on Python 3.8 or later.

```
qdiff_ancilla/
  problem.py          # B: shared data model (the contract)
  circuits.py         # B: benchmark builders, numpy statevector reference simulator
  evaluate.py         # B: rounding, scheduling, peak qubits, exact metrics
  baselines.py        # B: no-reuse, brute force
  sim.py              # A: density-matrix simulator
  noise.py            # A: noise channels and parameters
  loss.py             # A: live counts, SmoothMax, depth, total loss
  train.py            # A: training loop
  reqomp_adapter/     # C: own venv, scripts, README, JSON outputs
  experiments/toy.py  # integration script
  figures/  results/  # outputs
  tests/
  NOTES.md            # cross-agent notes, decisions, deviations
```

**Conventions, all agents:**

- Call `jax.config.update("jax_enable_x64", True)` at import; use complex128.
- Qubit 0 is the **most significant** bit of the basis index (matches PennyLane's default). State |q0 q1 …⟩ has index Σ qᵢ · 2^(n−1−i).
- Gates are named strings with a fixed qubit order: `"x" (q)`, `"h" (q)`, `"cx" (control, target)`, `"ccx" (control1, control2, target)`. A`  new gate name needs a matrix in both sim.py (A) and the reference simulator in circuits.py (B); log it in NOTES.md `.
- Time steps are indices into the base gate list. "Step t" means while gate t runs, so an ancilla counts as live at both its alloc\_step and its release\_step.
- Keep default circuits at 10 qubits or fewer; 12 is the hard limit (density matrices are 4ⁿ in size).
- Every random choice takes an explicit seed. Every script writes its config and results to `results/<name>.json`.
- Log anything you decide that this doc doesn't specify in `NOTES.md`, with your workstream letter.

## Data model (`problem.py`, owned by B)

The base gate list is the **hold-everything schedule**: every ancilla stays alive from its compute gate to its final uncompute gate. Each Decision describes one gap where the ancilla could instead be freed. A, B and C must all use exactly this format; changes go through B and get logged in `NOTES.md`. problem.py must use only the Python standard library (no numpy, no JAX), so C can import it from its separate environment. Start it with from \_\_future\_\_ import annotations and spell the Gate alias with typing.Tuple, so it imports on Python 3.8 and later.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple

Gate = Tuple[str, Tuple[int, ...]]        # ("ccx", (0, 1, 4)): controls first, target last

@dataclass(frozen=True)
class Ancilla:
    qubit: int
    alloc_step: int       # index of the base gate that first computes it
    release_step: int     # index of the base gate that finally uncomputes it

@dataclass(frozen=True)
class Decision:
    id: str
    ancilla: int                  # qubit index
    free_after: int               # uncompute block runs right after base gate free_after
    recompute_before: int         # recompute block runs right before base gate recompute_before
    uncompute: tuple[Gate, ...]   # inverse of the ancilla's compute gates, in reverse order
    recompute: tuple[Gate, ...]   # the ancilla's compute gates, in order
    extra_depth: int = -1         # filled by B's evaluate.py (-1 = not computed yet)
    extra_2q: int = -1            # extra two-qubit-equivalent gates, filled by B

@dataclass(frozen=True)
class Problem:
    name: str
    n_qubits: int
    data_qubits: tuple[int, ...]
    ancillas: tuple[Ancilla, ...]
    prep: tuple[Gate, ...]        # state preparation, run first; excluded from depth and live counts
    gates: tuple[Gate, ...]       # base (hold-everything) schedule
    decisions: tuple[Decision, ...]
    target_state: tuple[complex, ...]   # noiseless output of prep + gates, length 2**n_qubits
```

**Required functions in `problem.py`:**

- `to_json(problem, path)` and `from_json(path) -> Problem`. Complex numbers stored as `[re, im]` pairs. This is how C hands Problems to A and B.
- `validate(problem)`, raising `ValueError` with a clear message if any rule below fails.
- `gap_gates(problem, decision) -> range`, the base-gate indices strictly inside the gap: `free_after + 1` to `recompute_before - 1`.
- `build_schedule(problem, d) -> list[Gate]`, the discrete gate list for decision vector d (insert both blocks for each dᵢ = 1).
- `physical_schedule(problem, d) -> list[Gate]`: `build_schedule(problem, d)` with each ancilla gate moved onto its assigned physical qubit. Assignment is greedy interval colouring over the ancilla qubits in ascending order (toy free schedule: a, then b, then a again, all on qubit 4). An inserted uncompute belongs to the interval before its gap, a recompute to the interval after it. The interval and greedy logic lives here; `evaluate.live_intervals` and `assign_qubits` call it. A's simulator and B's metrics both run this list, so qubits, depth and noise describe one circuit.
- `layers(gates) -> list[list[int]]`, ASAP layering: each gate goes in the earliest layer after the last layer that used any of its qubits. A uses it for idle noise; B uses it for depth. Write it in the first hour, together with the dataclasses.

**Validity rules** (`validate` checks all of them):

1. ` alloc_step < free_after < recompute_before  ``< release_step` for the decision's ancilla.
2. The ancilla is used as a control at base gate `free_after` and at base gate `recompute_before`, and is **not** touched by any gate strictly inside the gap.
3. `recompute` equals the ancilla's compute gates; `uncompute` is their inverse in reverse order (for X/CX/CCX, the same gates reversed).
4. No gate after alloc\_step and before recompute\_before targets any control qubit of the ancilla's compute gates. This guarantees the inserted uncompute returns the ancilla to |0⟩ and the recompute reproduces the same value.
5. Two decisions on the same ancilla must have non-overlapping gaps.
6. `target_state` has unit norm, or is empty (C's adapter may emit it empty; circuits.fill\_target(problem) computes it).

**Meaning of a schedule.** A decision vector d ∈ {0, 1}ᴺ selects, for each decision, hold (0) or free (1). Freeing inserts `uncompute` after base gate `free_after` and `recompute` before base gate `recompute_before`. Both blocks belong to **one** choice: they are always inserted together or not at all.

## Benchmark 1: the toy circuit (`circuits.toy()`, built by B)

One ancilla value, a = c1 AND c2, is used twice, and a second ancilla b is needed in between. That gives exactly one decision with a real tradeoff. Every agent uses this as the first test case.

**Qubits (8):** 0 = c1, 1 = c2, 2 = c3, 3 = c4, 4 = a, 5 = b, 6 = t1, 7 = t2. Data qubits are 0–3 and 6–7; ancillas are 4 and 5.

**Prep:** `h` on qubits 0, 1, 2, 3, so the controls are in superposition and a is entangled with the data.

| Base step | Gate | Role |
| --- | --- | --- |
| 0 | `ccx (0, 1, 4)` | compute a |
| 1 | `ccx (4, 2, 6)` | first use of a |
| 2 | `ccx (2, 3, 5)` | compute b |
| 3 | `cx (5, 7)` | use b |
| 4 | `ccx (2, 3, 5)` | uncompute b |
| 5 | `ccx (4, 3, 7)` | second use of a |
| 6 | `ccx (0, 1, 4)` | final uncompute of a |

**Ancillas:** a = `Ancilla(4, alloc_step=0, release_step=6)`, b = `Ancilla(5, alloc_step=2, release_step=4)`.

**The one decision:** `Decision(id="a_gap", ancilla=4, free_after=1, recompute_before=5, uncompute=(("ccx", (0, 1, 4)),), recompute=(("ccx", (0, 1, 4)),))`.

**Target state:** the noiseless output of prep + base gates, computed with a plain numpy statevector simulation in `circuits.py` (independent of A's simulator, so it doubles as a cross-check).

**Hand-computed expected values** (B's and A's tests must reproduce these):

| Quantity | Hold (d = 0) | Free (d = 1) |
| --- | --- | --- |
| Live ancilla counts xₜ, steps 0–6 | \[1, 1, 2, 2, 2, 1, 1\] | \[1, 1, 1, 1, 1, 1, 1\] |
| Peak live ancillas | 2 | 1 |
| Physical qubits after reassignment | 8 | 7 (b takes a's qubit) |
| Depth (ASAP layers of the physical schedule, one layer per gate of any type) | 7 | 9 |
| Extra two-qubit-equivalent gates (Toffoli = 6 CNOT) | 0 | 12 |
| Noiseless fidelity to target | 1.0 | 1.0 |

Freeing costs 2 extra layers on the physical schedule. On the logical schedule the inserted uncompute (qubits 0, 1, 4) runs beside base step 2 (qubits 2, 3, 5), giving depth 7. Once b moves onto a's qubit, the uncompute of a, the compute, use and uncompute of b, and the recompute of a all share qubit 4 and must run in sequence, giving depth 9. So the toy trades one qubit against two noisy Toffolis plus two layers of idle exposure, and extra\_depth for this decision is 2.

**Expected behavior:** noiselessly, free wins once λa exceeds the depth penalty λd · 2 = 0.02. With noise, two extra noisy Toffolis and two extra layers cost more fidelity than a few idle steps on a, so the decision should hold at small λa and flip to free above some threshold. Finding that threshold is the progress-report result.

**Reference numbers** (a numpy run of this spec on the physical schedule; a sanity check, not test values): F\_hold ≈ 0.734, 0.543, 0.306 and F\_free ≈ 0.685, 0.475, 0.238 at s = 0.5, 1, 2. Brute force should flip near λa ≈ 0.069, 0.088 and 0.088. The threshold barely moves from s = 1 to s = 2 because both fidelities fall together; report that as a finding.

**Reference numbers** (a numpy run of this spec on the physical schedule; a sanity check, not test values): F\_hold ≈ 0.734, 0.543, 0.306 and F\_free ≈ 0.685, 0.475, 0.238 at s = 0.5, 1, 2. Brute force should flip near λa ≈ 0.069, 0.088 and 0.088. The threshold barely moves from s = 1 to s = 2 because both fidelities fall together; report that as a finding.

### Benchmark 2 (B, milestone B3)

Build a 9–10 qubit circuit where a = c1 AND c2 is used three times, with a different ancilla needed in each of the two gaps between uses, so there are two decisions on a. Hand-compute the same table for all four schedules and add it to the tests. Keep it brute-forceable.

## Workstream A: density-matrix simulator (`sim.py`)

### Core operations

Apply k-qubit operators to an n-qubit density matrix by tensor contraction, never by building a 2ⁿ × 2ⁿ gate matrix. Starter code (verify it with the tests before building on it):

```python
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def apply_op(rho, U, qs, n):
    """Return U rho U† for a 2^k x 2^k operator U on qubits qs (in U's own order)."""
    k = len(qs)
    U = U.reshape((2,) * (2 * k))
    rho = rho.reshape((2,) * (2 * n))
    rho = jnp.tensordot(U, rho, axes=(list(range(k, 2 * k)), list(qs)))
    rho = jnp.moveaxis(rho, list(range(k)), list(qs))
    cols = [n + q for q in qs]
    rho = jnp.tensordot(rho, U.conj(), axes=(cols, list(range(k, 2 * k))))
    rho = jnp.moveaxis(rho, list(range(2 * n - k, 2 * n)), cols)
    return rho.reshape(2 ** n, 2 ** n)

def apply_kraus(rho, kraus_ops, qs, n):
    """Return sum_K K rho K†. Every noise channel goes through this."""
    return sum(apply_op(rho, K, qs, n) for K in kraus_ops)

def fidelity(rho, psi):
    return jnp.real(psi.conj() @ rho @ psi)
```

**Gate matrices** (`GATES` dict in `sim.py`), with the most-significant-bit convention: `x`, `h`, `cx` (basis |control target⟩, swaps indices 2 and 3), `ccx` (basis |c1 c2 target⟩, swaps indices 6 and 7).

### Running one discrete schedule

`run_discrete(problem, d, noise) -> rho`:

1. Start from |0…0⟩⟨0…0|, apply `problem.prep` without noise.
2. Build the gate list with `physical_schedule(problem, d)` and group it with `layers(...)`.
3. For each layer: apply each gate, then its gate-noise channel on that gate's qubits; then apply the idle-noise channel to every qubit not touched in that layer.

A freed ancilla sits in |0⟩, which amplitude damping and dephasing leave unchanged, so idle noise automatically penalizes only qubits that hold a value (in the toy's free schedule, qubit 5 is never used). Do not special-case this.

### The soft circuit: average over schedules

**Important:** each decision's uncompute and recompute blocks are one coin flip. Mixing them independently (one mix at the uncompute point, another at the recompute point) is wrong: it creates a branch that uncomputes but never recomputes. Implement two versions:

- **v1, reference (`simulate_mixture_exact`):** enumerate all 2ᴺ decision vectors d, run each with `run_discrete`, and return Σ\_d w(d) · ρ\_d, with w(d) = Πᵢ rᵢ^dᵢ (1 − rᵢ)^(1−dᵢ). Exact and simple; fine for N ≤ 6.
- **v2, gap-local (`simulate_mixture`):** for a decision whose gap overlaps no other decision's gap, simulate the gap segment in both branches (free: uncompute block + gap gates + recompute block; hold: gap gates only) and mix once, at the end of the gap: ρ ← r · ρ\_free + (1 − r) · ρ\_hold. For groups of overlapping gaps, enumerate that group's branches jointly. Cost grows with the largest overlapping group, not with N.

v1 is the required path: with at most 2 decisions per benchmark it costs at most 4 discrete runs and is still differentiable in r, which is all this project needs. v2 is a stretch goal. It matches v1 to 1e-10 only if both use a layering barrier at each gap boundary, since ASAP lets gates from outside a gap drift into its layers and the two branches can differ in length. If v2 is attempted, add the barrier as a flag and log it in `NOTES.md`.

v2 has a second precondition: the physical qubit layout outside each gap must be the same in both branches. Once ancillas are reassigned, freeing one ancilla can move a later ancilla onto a different physical qubit, and then the two branches can no longer be merged into one density matrix that later gates act on. Check this per gap and fall back to v1 wherever it fails.

Both versions take `r` as a JAX array and must be differentiable with `jax.grad`. The circuit structure is static, so plain Python loops over gates are fine; add `jax.jit` once the tests pass.

## Workstream A: noise model (`noise.py`)

Use the standard Markovian model: depolarizing noise after every gate, amplitude damping plus dephasing on every idle qubit in every layer. All channels go through `apply_kraus`.

### Parameters

```python
@dataclass(frozen=True)
class NoiseParams:
    p1: float = 1e-3          # depolarizing, single-qubit gates
    p2: float = 1e-2          # depolarizing, cx
    p3: float | None = None   # depolarizing, ccx; None -> 1 - (1 - p2)**6 (a Toffoli is ~6 CNOTs), about 0.059
    t_layer_us: float = 3.0   # one layer's duration, ~ one Toffoli
    T1_us: float = 200.0
    T2_us: float = 150.0      # must satisfy T2 <= 2 * T1
    source: str = "defaults"  # record where numbers came from
```

- `NoiseParams.none()`: all zero / infinite times, so the simulator is exactly noiseless.
- `params.scaled(s)`: multiply p1, p2 (and p3 if set) by s; divide T1 and T2 by s. Used for the noise sweep (s ∈ {0.5, 1, 2}); scaled(0) returns NoiseParams.none() instead of dividing by zero.
- Optional, never needed by tests: `from_fake_backend(name)`, reading median gate errors, T1, T2 and gate durations from a `qiskit_ibm_runtime.fake_provider` backend, with the backend name stored in `source`. IBM reports average gate error r, so convert with p = r · 2ᵏ / (2ᵏ − 1): p = 2r for 1-qubit gates and 4r/3 for 2-qubit gates. Wrap the Qiskit import in `try`.

### Channels

| Channel | Applied | Kraus operators |
| --- | --- | --- |
| k-qubit depolarizing, strength p | after each gate, on its qubits (p1, p2 or p3 by gate size) | √(1 − p + p/4ᵏ) · I, and √(p/4ᵏ) · P for each of the other 4ᵏ − 1 Pauli strings P |
| Amplitude damping, γ | each idle qubit, each layer | K0 = \[\[1, 0\], \[0, √(1 − γ)\]\], K1 = \[\[0, √γ\], \[0, 0\]\] |
| Phase damping, λ | each idle qubit, each layer | K0 = \[\[1, 0\], \[0, √(1 − λ)\]\], K1 = \[\[0, 0\], \[0, √λ\]\] |

Per-layer strengths from the times:

- γ = 1 − exp(−t\_layer / T1)
- Pure dephasing time Tφ, from 1/Tφ = 1/T2 − 1/(2·T1); then λ = 1 − exp(−2·t\_layer / Tφ)

With the defaults, γ ≈ 0.015 and λ ≈ 0.025 (Tφ = 240 µs). Apply depolarizing noise as (1 − p) ρ + p · (Tr\_qs ρ) ⊗ I/2ᵏ: the same channel without summing 4ᵏ Kraus terms, which keeps the 64-term Toffoli case and jit compile time cheap. This is the one exception to routing channels through apply\_kraus; keep the Kraus form for the trace-preservation test.

## Workstream A: loss and training (`loss.py`, `train.py`)

### Parameters and relaxation

One unconstrained logit zᵢ per decision, with rᵢ = σ(zᵢ), the probability of freeing. Initialize z = 0 (r = 0.5, undecided). Train z, never r directly.

### Live counts

For base steps t = 0, …, T − 1 (T = number of base gates), each ancilla contributes:

- 0 outside its lifetime (t < alloc\_step or t > release\_step)
- 1 − rᵢ if t is inside the gap of decision i (`gap_gates(problem, decision_i)`)
- 1 otherwise

xₜ is the sum over ancillas. For the toy: x = \[1, 1, 2 − r, 2 − r, 2 − r, 1, 1\].

### Loss

```latex
L(z) = \big(1 - F(r)\big) + \lambda_a \cdot \frac{1}{\beta} \log \sum_t e^{\beta x_t} + \lambda_d \sum_i r_i \, \Delta_i, \qquad r = \sigma(z)
```

- F(r): fidelity of `simulate_mixture(problem, r, noise)` to `problem.target_state`.
- SmoothMax: use `jax.scipy.special.logsumexp(beta * x) / beta`.
- Δᵢ = `decision.extra_depth`. If it is −1 (B's costs not filled yet), compute it as `len(layers(``physical_schedule(problem, e_i)`` ))  ``- len(layers(physical_schedule(problem, zeros)))`, with eᵢ the one-hot vector for decision i and zeros the all-hold vector. Subtract the physical hold schedule, not problem.gates: reassignment can merge ancillas whose lifetimes never overlap even when everything is held, which changes the baseline depth.
- Optional budget mode, flag `budget=B`: replace the SmoothMax term with λa · softplus(κ · (SmoothMax − B)) / κ, κ = 10, so only peaks above B are penalized.

### Training (`train.py`)

`train(problem, noise, lam_a, lam_d=0.01, steps=300, lr=0.1, beta=(2.0, 20.0), seed=0) -> TrainResult`

- Adam via `optax`, gradients by `jax.value_and_grad` with respect to z.
- β rises linearly from beta\[0\] to beta\[1\] over training.
- Record every step in `history`: loss, F, SmoothMax, depth term, and every rᵢ. Save to `results/` as JSON.
- `TrainResult` holds final z, final r, and history.

### Sweep (`train.sweep_lambda`)

Train for each λa in 21 log-spaced values from 0.01 to 0.3 (the toy flips near 0.02 noiseless and 0.07–0.09 with noise), at noise scales s ∈ {0 (noiseless), 0.5, 1, 2}. Return final r per run. Batch the λa values with jax.vmap so one compiled trainer serves the whole sweep. B's evaluation turns these into the progress-report figures.

## Workstream B: rounding and exact evaluation (`evaluate.py`)

Everything the project reports is measured here, on discrete schedules with exact quantities (true max, not SmoothMax).

### Functions

- `round_decisions(r) -> d`: dᵢ = 1 if rᵢ > 0.5, else 0.
- `live_intervals(problem, d) -> list[(ancilla, start, end)]`: each ancilla's occupied intervals in base-step indices, inclusive. A held ancilla is one interval \[alloc\_step, release\_step\]. Freeing decision i splits it into \[alloc\_step, free\_after\] and \[recompute\_before, release\_step\].
- `peak_ancillas(problem, d) -> int`: maximum number of intervals covering any base step.
- `assign_qubits(problem, d) -> dict`: greedy interval coloring (sort intervals by start, reuse the lowest free physical qubit). Each interval of a freed ancilla is assigned independently, since a recomputed value can live on any clean qubit. Return the mapping and the physical qubit count = len(data\_qubits) + number of ancilla qubits used. For intervals, greedy is optimal, so ancilla qubits used = peak.
- `depth(problem, d) -> int`: `len(layers(``physical_schedule(problem, d)``))`.
- `two_qubit_equiv(problem, d) -> int`: cx counts 1, ccx counts 6, single-qubit gates 0.
- `fill_costs(problem) -> Problem`: fill every decision's `extra_depth` and `extra_2q` from one-hot schedules.
- `evaluate(problem, d, noise=None, simulator=None) -> dict`: all metrics above, plus fidelity. If `simulator` is given (A's `run_discrete`), use it with `noise`; otherwise compute noiseless fidelity with the numpy reference in `circuits.py`. This keeps B unblocked until A's simulator lands.

**Tests:** reproduce every value in the toy table, both schedules, including that b is assigned a's physical qubit in the free schedule.

## Workstream B: baselines (`baselines.py`)

- `no_reuse(problem)`: d = all zeros, and every ancilla keeps its own physical qubit (no reassignment). This is the deck's baseline.
- `hold_all_reassigned(problem)`: d = all zeros, but with `assign_qubits`, so ancillas whose lifetimes never overlap still share qubits. Shows how much comes from reassignment alone.
- `brute_force(problem, noise, lam_a, lam_d, simulator=None)`: evaluate all 2ᴺ schedules and return the table plus the argmin of the exact discrete objective J(d) = (1 − F\_d) + λa · peak\_d + λd · (depth\_d − depth\_0), both measured with depth() on the physical schedule (depth\_0 = the all-hold schedule), and lam\_d passed explicitly in every test (default 0.01). Refuse N > 12 with a clear error.

The trained-and-rounded schedule should match brute force's argmin on every toy run of the λ sweep, except within a narrow band around the flip threshold (SmoothMax's offset can shift it slightly). Report any mismatch outside that band.

## Workstream C: Reqomp integration (`reqomp_adapter/`)

Goal: Reqomp as a baseline on our benchmarks, and Reqomp's value-tracking analysis as the source of Decisions for circuits we don't hand-write. **Run and wrap Reqomp's code; do not reimplement its search** (dynamic programming and heuristics). Reqomp's paper: Paradis, Bichsel & Vechev, *Quantum* 8, 1258 (2024); code at https://github.com/eth-sri/Reqomp.

### C1: get Reqomp running

1. Clone the repo into `reqomp_adapter/vendor/` and create `reqomp_adapter/.venv`.
2. Install the versions its README or requirements specify (expect an older Qiskit). Record exact versions in `reqomp_adapter/README.md`.
3. Run the repo's own examples or tests and record the results.

If it won't install after 3 serious attempts or about 2 hours, **stop C1 and report** what failed, then move to the fallback under C3. Don't patch Reqomp's source beyond trivial import fixes, and log any patch.

### C2: Reqomp baseline on our circuits

1. Write the toy (and later benchmark 2) as Qiskit circuits in Reqomp's input form, with ancillas in an `AncillaRegister`. Reqomp synthesizes uncomputation itself, so leave out the uncompute gates (the toy's base steps 4 and 6) and record which base step each remaining gate came from. Read Reqomp's code and examples to find the expected API.
2. Run Reqomp across every qubit budget it accepts for each circuit.
3. For each output, save the circuit as QASM and record qubit count, gate count, two-qubit-equivalent count and depth in `reqomp_adapter/results/<benchmark>.json`.
4. Convert each output circuit to our Gate format. The paper mentions substituting a relative-phase Toffoli (RCCX); disable that substitution if Reqomp allows it. Otherwise map it to a new gate name `rccx` and log in `NOTES.md` that A and B need its matrix.

### C3: graph-to-Problem adapter

1. From Reqomp's internals, extract its circuit graph with value indices for a given input circuit.
2. For each ancilla value used two or more times, emit one Decision per gap between consecutive uses, keeping only gaps where the graph shows the value can be recomputed (same controls at the same value indices).
3. Write the result with `problem.to_json` (stdlib-only, importable from C's environment), leaving `target_state` empty.
4. **Acceptance:** for the toy, the adapter's Decision matches B's hand-written one exactly once its gate indices are mapped back to B's base steps (same `free_after`, `recompute_before`, blocks).

**Fallback if Reqomp's internals aren't usable:** write `reqomp_adapter/analyze.py`, a stdlib-only value-index analysis over our own Gate lists. Track (qubit, value index) per gate for x/cx/ccx; a gate that repeats an earlier gate with the same control values reproduces that value; an exact inverse returns the target to its previous index. Then find uses, gaps and recomputable gaps as above. Mark it clearly in `NOTES.md` as a simplified stand-in, not Reqomp.

## Required tests

A milestone is done only when its tests pass with `pytest`. Tolerance is 1e-10 for exact identities and 1e-6 for gradient checks unless stated.

### A: simulator, noise, loss

- [ ] After random gates and noise channels, ρ has trace 1, is Hermitian, and has eigenvalues ≥ −1e-10.
- [ ] `apply_op` matches an explicit full-matrix construction (Kronecker products) on random 1–3 qubit gates in a 4-qubit system, including non-adjacent and reversed qubit orders.
- [ ] A random 4-qubit circuit with depolarizing and amplitude damping matches PennyLane's `default.mixed` (its DepolarizingChannel(p) puts p/3 on each of X, Y and Z, so pass it 3p/4 for this doc's 1-qubit p).
- [ ] Every channel is trace-preserving: Σ K†K = I.
- [ ] Amplitude damping and phase damping leave |0⟩⟨0| unchanged; 1-qubit depolarizing with p = 1 gives I/2; `NoiseParams.none()` leaves ρ unchanged.
- [ ] Toy, noiseless: `run_discrete` gives fidelity 1.0 for both d = 0 and d = 1, and matches B's numpy reference state.
- [ ] Toy, noisy: soft fidelity equals r · F(free) + (1 − r) · F(hold) at r ∈ {0.2, 0.5, 0.8}.
- [ ] Stretch, only if v2 is built: `simulate_mixture` (v2) equals `simulate_mixture_exact` (v1) on the toy and on benchmark 2 at random r.
- [ ] Toy live counts equal \[1, 1, 2 − r, 2 − r, 2 − r, 1, 1\].
- [ ] SmoothMax satisfies max(x) ≤ SmoothMax ≤ max(x) + log(T)/β.
- [ ] `jax.grad` of the full loss matches central finite differences in z (step 1e-5).
- [ ] Toy, noiseless, λa = 0.1: training ends with r > 0.9.

### B: data model, circuits, evaluation, baselines

- [ ] The toy Problem passes `validate`; JSON round-trip returns an equal Problem.
- [ ] One deliberately broken Problem per validity rule, each rejected with `ValueError`.
- [ ] The reference target state has norm 1.
- [ ] `layers` gives 7 layers for both logical toy schedules (inserted uncompute in the same layer as base step 2), and 7 and 9 layers for the physical hold and free schedules.
- [ ] Every value in the toy's expected-values table, both schedules, including b assigned to a's physical qubit when freed.
- [ ] `brute_force` on the toy, noiseless, λa = 0.1, returns d = (1,).
- [ ] `brute_force` on the toy, noiseless, λa = 0.01, returns d = (0,), since the depth penalty of 0.02 outweighs one qubit.
- [ ] Benchmark 2: its hand-computed table, all four schedules.

### C: Reqomp

- [ ] `reqomp_adapter/results/toy.json` exists with qubit count, gate count, two-qubit-equivalent count and depth for each budget.
- [ ] The adapter's toy Decision equals B's hand-written one.

### Integration

- [ ] `python experiments/toy.py` runs end to end in under 30 minutes on a laptop CPU and writes the figures and table below.

## Outputs and reporting back

### For the progress report (integration step)

`experiments/toy.py` produces:

1. **`figures/toy_training.png`:** r over training steps for λa ∈ {0.01, 0.1, 1}, noiseless and at noise scale s = 1.
2. **`figures/toy_lambda_sweep.png`:** final r versus λa (log x-axis), one line per noise scale (noiseless, 0.5, 1, 2), with brute force's flip threshold marked for each.
3. **`results/toy_table.md` and `.json`:** one row per schedule (no reuse, hold with reassignment, free, and Reqomp's output if C2 is done) with peak ancillas, physical qubits, depth, two-qubit-equivalent gates, and fidelity at each noise scale.

Label every axis with units and every figure with the noise parameters used.

### How each agent reports back

When you stop, append a section to `NOTES.md` headed with your workstream letter:

- Milestones completed, and the `pytest` summary line
- Files created or changed
- Every deviation from this doc, and why
- Anything blocking another workstream
- Open questions for the humans

## Out of scope, and when to stop and ask

**Out of scope for this run:** choosing *when* to uncompute (softmax over timing slots), Grover and adder benchmarks, a smooth critical-path depth estimate, reimplementing Reqomp's search, circuits above 12 qubits, and anything that needs quantum hardware or network access beyond package installs and cloning Reqomp.

**Stop and report instead of working around it if:**

- The noiseless toy gives different fidelities for hold and free. That means a spec or simulator bug, not something to tune away.
- A hand-computed value in this doc looks wrong. Report it with your reasoning; don't silently change the test to match your code.
- v2, if attempted, can't be made to match v1 with layering barriers in place.
- Reqomp won't install within the C1 limit.
- You need to change a field or function in `problem.py` after it's committed.
- Trained-and-rounded schedules disagree with brute force outside the band near the flip threshold.

**Open questions for the humans** (don't block on these; use the defaults):

- Which fake backend's noise numbers to cite in the report (defaults are in the noise section).
- The weight of the depth term (default λd = 0.01; on the toy it adds 0.02 against freeing, which shifts the flip threshold by that much).
- Whether to present budget mode alongside the λa sweep.
