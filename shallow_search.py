import warnings; warnings.filterwarnings('ignore')
import numpy as np, itertools, json, sys
import wormhole_ibm as w
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import DensityMatrix, partial_trace

p2 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
nm = NoiseModel()
nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ['cz'])
nm.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ['sx', 'x'])
sim = AerSimulator(method='density_matrix', noise_model=nm)
BASIS = ['cz', 'rz', 'sx', 'x']

def noisy_choi(steps, mu):
    qc, _, r1 = w.protocol(steps, mu, with_reference=True)
    t = transpile(qc, basis_gates=BASIS, optimization_level=3, seed_transpiler=1)
    n2 = t.count_ops().get('cz', 0)
    t.save_density_matrix(qubits=[0, r1])
    dm = sim.run(t).result().data()['density_matrix']
    rho = np.asarray(dm).reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)
    return rho, n2

res = []
for N, dt in itertools.product([2, 3], [0.25, 0.5, 0.75, 1.0]):
    w.N, w.DT = N, dt
    for steps in range(1, 5):
        best = None
        for mu in np.linspace(-2*np.pi, 2*np.pi, 25):
            if abs(mu) < 1e-9: continue
            I = w.mutual_info(w.choi_state(steps, mu))
            if best is None or I > best[1]: best = (mu, I)
        mu = best[0]
        Fi, _ = w.best_decoder(w.choi_state(steps, mu))
        rho, n2 = noisy_choi(steps, mu)
        Fn, _ = w.best_decoder(rho)
        rho0, _ = noisy_choi(steps, 0.0)
        F0, _ = w.best_decoder(rho0)
        res.append(dict(N=N, dt=dt, steps=steps, mu=float(mu), F_ideal=Fi, F_noisy=Fn, F0_noisy=F0, cz=n2))
        print(f"N={N} dt={dt} steps={steps} mu={mu:+.2f} cz={n2:3d}  ideal={Fi:.3f} noisy={Fn:.3f} control={F0:.3f}  gap={Fn-F0:+.3f}", flush=True)
json.dump(res, open(f'shallow_p{p2}.json', 'w'))
