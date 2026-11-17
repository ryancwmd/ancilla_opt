# Benchmark 2: schedules compared

| schedule | peak ancillas | physical qubits | depth (layers) | 2q-equivalent gates | F, noiseless | F, s = 0.5 | F, s = 1 | F, s = 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no reuse | 2 | 9 | 9 | 56 | 1.0000 | 0.6484 | 0.4259 | 0.1924 |
| hold, reassigned | 2 | 8 | 9 | 56 | 1.0000 | 0.6484 | 0.4258 | 0.1922 |
| d = 01 | 2 | 8 | 13 | 68 | 1.0000 | 0.5599 | 0.3228 | 0.1199 |
| d = 10 | 2 | 8 | 11 | 68 | 1.0000 | 0.6052 | 0.3729 | 0.1512 |
| d = 11 | 1 | 7 | 15 | 80 | 1.0000 | 0.5233 | 0.2843 | 0.0971 |

Decisions, in d order: a_gap1, a_gap2. Fidelity to the noiseless target, on the physical schedule. Noise at s = 1: p1 = 0.001, p2 = 0.01, p3 = None (None = 1-(1-p2)^6), t_layer = 3 us, T1 = 200 us, T2 = 150 us; scale s multiplies error rates and divides T1, T2. Backend: A: sim.run_discrete. 2q-equivalent: cx = 1, ccx = 6.
