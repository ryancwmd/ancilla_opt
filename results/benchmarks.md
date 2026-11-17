# Benchmarks: baselines and brute force
Backend: A: sim.run_discrete. lam_d = 0.01. Git 59fa5a4.

## toy (8 qubits, decisions ['a_gap'], extra_depth [2], extra_2q [12])

| schedule | peak ancillas | physical qubits | depth (layers) | 2q-equiv | F (s=0.0) | F (s=0.5) | F (s=1.0) | F (s=2.0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no reuse | 2 | 8 | 7 | 37 | 1.0000 | 0.7337 | 0.5430 | 0.3060 |
| hold, reassigned | 2 | 8 | 7 | 37 | 1.0000 | 0.7337 | 0.5430 | 0.3060 |
| d = 0 | 2 | 8 | 7 | 37 | 1.0000 | 0.7337 | 0.5430 | 0.3060 |
| d = 1 | 1 | 7 | 9 | 49 | 1.0000 | 0.6846 | 0.4745 | 0.2381 |

Brute-force flip thresholds in lam_a (J linear in lam_a):

- s = 0.0: 0.0200 (0 to 1)
- s = 0.5: 0.0691 (0 to 1)
- s = 1.0: 0.0884 (0 to 1)
- s = 2.0: 0.0879 (0 to 1)

## bench2 (9 qubits, decisions ['a_gap1', 'a_gap2'], extra_depth [2, 4], extra_2q [12, 12])

| schedule | peak ancillas | physical qubits | depth (layers) | 2q-equiv | F (s=0.0) | F (s=0.5) | F (s=1.0) | F (s=2.0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no reuse | 2 | 9 | 9 | 56 | 1.0000 | 0.6484 | 0.4259 | 0.1924 |
| hold, reassigned | 2 | 8 | 9 | 56 | 1.0000 | 0.6484 | 0.4258 | 0.1922 |
| d = 00 | 2 | 8 | 9 | 56 | 1.0000 | 0.6484 | 0.4258 | 0.1922 |
| d = 01 | 2 | 8 | 13 | 68 | 1.0000 | 0.5599 | 0.3228 | 0.1199 |
| d = 10 | 2 | 8 | 11 | 68 | 1.0000 | 0.6052 | 0.3729 | 0.1512 |
| d = 11 | 1 | 7 | 15 | 80 | 1.0000 | 0.5233 | 0.2843 | 0.0971 |

Brute-force flip thresholds in lam_a (J linear in lam_a):

- s = 0.0: 0.0600 (00 to 11)
- s = 0.5: 0.1850 (00 to 11)
- s = 1.0: 0.2015 (00 to 11)
- s = 2.0: 0.1551 (00 to 11)
