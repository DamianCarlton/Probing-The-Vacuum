"""Run the wormhole teleportation test on a Quantum Inspire machine (default: Tuna-17).
Needs: `qi login` done first, and wormhole_ibm.py in the same folder (it holds the circuits)."""
import json, sys
import numpy as np
from qiskit import transpile
import wormhole_ibm as w

w.MODEL = 'chaotic'
WANT = sys.argv[1] if len(sys.argv) > 1 else 'Tuna-17'
SHOTS = 2000


def bit(key):
    """Counts keys can be '0'/'1', '0x0'/'0x1' or longer bitstrings: the measured bit is the lowest one."""
    k = str(key).replace(' ', '')
    return int(k, 16) & 1 if k.startswith('0x') else int(k[-1])


def tidy(counts):
    out = {'0': 0, '1': 0}
    for k, v in counts.items():
        out[str(bit(k))] += v
    return out


def pick_backend(provider):
    names = [b.name for b in provider.backends()]
    print('Machines available to you:', names)
    for cand in [WANT, 'Tuna-17', 'Tuna-9', 'Tuna-5', 'Starmon-7']:
        hit = [n for n in names if cand.lower() in n.lower()]
        if hit:
            return provider.get_backend(hit[0])
    sys.exit('None of the expected machines is listed; send Claude the list above.')


def main(provider=None):
    if provider is None:
        from qiskit_quantuminspire.qi_provider import QIProvider
        provider = QIProvider()
    backend = provider if hasattr(provider, 'run') else pick_backend(provider)
    print('Using:', backend.name)
    s = w.load_settings()
    circs, meta = w.build_batch(s)
    tc = transpile(circs, backend, optimization_level=3, seed_transpiler=7)
    print(f'{len(tc)} circuits, about {max(sum(1 for i in c.data if i.operation.num_qubits == 2) for c in tc)} two-qubit gates each; {SHOTS} shots each')
    # Tuna machines take at most 5 circuits per batch and 5 batches queued per user, and results can
    # take a while: send one batch at a time, wait for it without a time limit, retry if the queue is full.
    import time
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
    summary = w.summarise(meta, counts, f'\n{backend.name} (real hardware):')
    json.dump({'backend': backend.name, 'model': 'chaotic', 'summary': summary, 'counts': counts},
              open('hardware_result_chaotic.json', 'w'), indent=1)
    print('Saved hardware_result_chaotic.json')


if __name__ == '__main__':
    main()
