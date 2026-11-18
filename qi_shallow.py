"""Shallow wormhole-teleportation test for Quantum Inspire's Tuna-17.
2 qubits per side (5 qubits in all), 4 time steps of 0.75, coupling mu = -pi/4:
about 13 two-qubit gates before routing, versus ~89 for the deep version.
Noise-model prediction: average fidelity ~0.53 with coupling vs 0.50 without."""
import json, sys, time
import numpy as np
from qiskit import transpile
import wormhole_ibm as w

w.MODEL, w.N, w.DT = 'chaotic', 2, 0.75
STEPS, MU, SHOTS = 4, -np.pi / 4, 8000
WANT = sys.argv[1] if len(sys.argv) > 1 else 'Tuna-17'


def bit(key):
    k = str(key).replace(' ', '')
    return int(k, 16) & 1 if k.startswith('0x') else int(k[-1])


def tidy(counts):
    out = {'0': 0, '1': 0}
    for k, v in counts.items():
        out[str(bit(k))] += v
    return out


def main(backend=None):
    F_ideal, dec = w.best_decoder(w.choi_state(STEPS, MU))
    print(f'Perfect-machine prediction with coupling: {F_ideal:.3f} (0.500 without)')
    if backend is None:
        from qiskit_quantuminspire.qi_provider import QIProvider
        provider = QIProvider()
        names = [b.name for b in provider.backends()]
        name = next((n for n in names if WANT.lower() in n.lower()), None)
        if name is None:
            sys.exit(f'{WANT} not found; machines available: {names}')
        backend = provider.get_backend(name)
    print('Using:', backend.name)
    s = {'steps': STEPS, 'mu': MU, 'decoder': dec.tolist()}
    circs, meta = w.build_batch(s)
    tc = transpile(circs, backend, optimization_level=3, seed_transpiler=7)
    print(f'{len(tc)} circuits, at most {max(sum(1 for i in c.data if i.operation.num_qubits == 2) for c in tc)} two-qubit gates each; {SHOTS} shots each')
    counts = []
    for i in range(0, len(tc), 5):
        while True:
            try:
                job = backend.run(tc[i:i + 5], shots=SHOTS)
                break
            except Exception as e:
                if '429' in str(e) or 'maximum' in str(e):
                    print('  queue full (earlier jobs still waiting), retrying in 60 s...')
                    time.sleep(60)
                else:
                    raise
        res = job.result(timeout=None)
        counts += [tidy(res.get_counts(k)) for k in range(len(res.results))]
        print(f'  batch {i // 5 + 1} of {(len(tc) + 4) // 5} done')
    summary = w.summarise(meta, counts, f'\n{backend.name} (real hardware), shallow circuit:')
    json.dump({'backend': backend.name, 'circuit': {'N': w.N, 'dt': w.DT, 'steps': STEPS, 'mu': MU},
               'prediction_ideal': F_ideal, 'summary': summary, 'counts': counts},
              open('hardware_result_shallow.json', 'w'), indent=1)
    print('Saved hardware_result_shallow.json')


if __name__ == '__main__':
    main()
