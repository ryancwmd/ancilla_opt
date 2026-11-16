# Toy circuit: schedules compared

| schedule | peak ancillas | physical qubits | depth (layers) | 2q-equivalent gates | F, noiseless | F, s = 0.5 | F, s = 1 | F, s = 2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no reuse | 2 | 8 | 7 | 37 | 1.0000 | 0.7337 | 0.5430 | 0.3060 |
| hold, reassigned | 2 | 8 | 7 | 37 | 1.0000 | 0.7337 | 0.5430 | 0.3060 |
| free (d = 1) | 1 | 7 | 9 | 49 | 1.0000 | 0.6846 | 0.4745 | 0.2381 |

Fidelity to the noiseless target, on the physical schedule. Noise at s = 1: p1 = 0.001, p2 = 0.01, p3 = None (None = 1-(1-p2)^6), t_layer = 3 us, T1 = 200 us, T2 = 150 us; scale s multiplies error rates and divides T1, T2. Backend: A: sim.run_discrete. 2q-equivalent: cx = 1, ccx = 6.
