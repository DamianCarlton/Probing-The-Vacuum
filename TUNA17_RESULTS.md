# Hardware results: QuTech Tuna-17 (17 November 2026)

Average teleportation fidelity over six input states (0.500 = nothing gets through, 0.667 = classical limit).

| Circuit | Shots per circuit | Coupling on | Opposite sign | No coupling |
| --- | --- | --- | --- | --- |
| Deep (7 qubits, mu = -4.712, 7 steps x 0.25) | 2,000 | 0.502 ± 0.005 | 0.500 ± 0.005 | 0.493 ± 0.005 |
| Shallow (5 qubits, mu = -0.785, 4 steps x 0.75) | 8,000 | 0.493 ± 0.002 | 0.494 ± 0.002 | 0.497 ± 0.002 |

Predictions for comparison (coupling on vs no coupling):

| Circuit | Perfect machine | IBM Heron noise model (FakeFez) |
| --- | --- | --- |
| Deep | 0.594 vs 0.500 | 0.559 vs 0.498 |
| Shallow | 0.559 vs 0.500 | 0.553 vs 0.496 |

Conclusion: on Tuna-17 the gap between coupled and uncoupled runs was 0.009 ± 0.007 (deep) and
-0.004 ± 0.003 (shallow). Hardware noise erased the signal in both circuits.

Reproduce: run `wormhole_quantum_inspire.ipynb` (deep) or `wormhole_shallow_tuna17.ipynb` (shallow)
in Google Colab with a free Quantum Inspire account.
