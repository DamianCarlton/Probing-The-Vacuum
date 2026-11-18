"""
Wormhole-inspired teleportation on a quantum computer (toy model, 7 qubits).

What it does
------------
Two small "black holes" L and R (3 qubits each) start entangled in Bell pairs
(the thermofield-double state at infinite temperature). A message qubit Q is
thrown into L at time -t, the two sides are coupled once with strength mu
(the analogue of the negative-energy pulse that holds a wormhole open), and the
message is read off R's first qubit at time +t.

Dynamics: a chaotic kicked Ising chain, one step = ZZ bonds + X and Z fields.
The circuit itself IS the model, so the local simulator is exact.

Success measure: average teleportation fidelity over the six Pauli eigenstates.
    0.50 = no information gets through (random guess)
    0.667 = best any classical (measure-and-resend) channel can do
    1.00 = perfect teleportation
Controls: mu = 0 (no coupling: should give 0.50) and mu of the opposite sign.

Usage
-----
    pip install qiskit qiskit-aer qiskit-ibm-runtime scipy
    python wormhole_ibm.py scan        # find the best (steps, mu) on the exact simulator
    python wormhole_ibm.py simulate    # exact + noisy (IBM Heron noise model) prediction
    python wormhole_ibm.py hardware    # real IBM QPU; needs your token (see below)

For hardware, create a free account at quantum.cloud.ibm.com, copy your API key, then
    export IBM_QUANTUM_TOKEN="your-key"          (Windows: set IBM_QUANTUM_TOKEN=your-key)
    python wormhole_ibm.py hardware --shots 2000
If the session closes while the job queues, collect it later with
    python wormhole_ibm.py fetch --job <job id>
The default run is 18 circuits x 2000 shots: well inside the free 10 minutes.
Optional: --backend ibm_xxx to pick a machine; otherwise the least busy one is used.
"""
import argparse
import json
import os
import sys

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
from scipy.linalg import expm
from scipy.optimize import minimize

# ---------------- model parameters (chosen by `scan`; see README) ----------------
N = 3            # qubits per side
HX, HZ = 1.05, 0.5
DT = 0.25        # time per Floquet step (chaotic model)
MODEL = 'chaotic'
DEFAULTS = {"steps": None, "mu": None}  # filled from scan_result.json if present

PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------- circuit construction ----------------
def floquet_step(n):
    """One Trotter step of the mixed-field Ising chain H = sum ZZ + HX sum X + HZ sum Z."""
    qc = QuantumCircuit(n)
    for i in range(n - 1):
        qc.rzz(2 * DT, i, i + 1)
    for i in range(n):
        qc.rx(2 * HX * DT, i)
        qc.rz(2 * HZ * DT, i)
    return qc


def evolution(n, steps):
    qc = QuantumCircuit(n)
    step = floquet_step(n)
    for _ in range(steps):
        qc.compose(step, inplace=True)
    return qc


def protocol(steps, mu, with_reference=False):
    """Qubits: [P] Q, L1..LN, R1..RN.  Returns circuit (no measurement).
    The message is inserted by SWAP(Q, L1); we skip the SWAP gate by letting Q
    *become* L1 for the forward evolution (old L1 then sits idle)."""
    off = 1 if with_reference else 0
    q = off
    L = list(range(off + 1, off + 1 + N))
    R = list(range(off + 1 + N, off + 1 + 2 * N))
    qc = QuantumCircuit(off + 1 + 2 * N)
    if with_reference:
        qc.h(0); qc.cx(0, q)                      # P-Q Bell pair (for exact analysis)
    for l, r in zip(L, R):                        # TFD at infinite temperature
        qc.h(l); qc.cx(l, r)
    U = evolution(N, steps)
    qc.compose(U.inverse(), qubits=L, inplace=True)          # e^{+iH_L t}
    Lnew = [q] + L[1:]                                       # message now plays L1
    qc.compose(U, qubits=Lnew, inplace=True)                 # e^{-iH_L t}
    for l, r in zip(Lnew[1:], R[1:]):                        # coupling e^{i mu V}
        qc.rzz(-2 * mu / (N - 1), l, r)
    qc.compose(U, qubits=R, inplace=True)                    # e^{-iH_R t}
    return qc, q, R[0]


# ---------------- exact analysis ----------------
def choi_state(steps, mu):
    qc, _, r1 = protocol(steps, mu, with_reference=True)
    sv = Statevector(qc)
    keep = [0, r1]
    rho = partial_trace(sv, [i for i in range(qc.num_qubits) if i not in keep])
    # qiskit orders qubits little-endian: reorder to (P, R1)
    return DensityMatrix(rho).data.reshape(2, 2, 2, 2).transpose(1, 0, 3, 2).reshape(4, 4)


def mutual_info(rho):
    S = lambda m: -sum(x * np.log2(x) for x in np.linalg.eigvalsh(m) if x > 1e-12)
    r = rho.reshape(2, 2, 2, 2)
    return S(np.einsum('ajbj->ab', r)) + S(np.einsum('iaib->ab', r)) - S(rho)


def su2(p):
    return expm(-1j * (p[0] * PX + p[1] * PY + p[2] * PZ))


def best_decoder(rho):
    phi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    def neg(p):
        W = np.kron(np.eye(2), su2(p))
        return -np.real(phi.conj() @ W @ rho @ W.conj().T @ phi)
    res = min((minimize(neg, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-9})
               for x0 in np.random.default_rng(1).normal(size=(10, 3))), key=lambda r: r.fun)
    return (2 * (-res.fun) + 1) / 3, res.x     # average fidelity, decoder params


# ---------------- measurement circuits ----------------
INPUTS = ['+z', '-z', '+x', '-x', '+y', '-y']


def prep(qc, q, label):
    if label == '-z': qc.x(q)
    elif label == '+x': qc.h(q)
    elif label == '-x': qc.x(q); qc.h(q)
    elif label == '+y': qc.h(q); qc.s(q)
    elif label == '-y': qc.x(q); qc.h(q); qc.s(q)


def rotate_to_z(qc, q, label):
    """Map the input basis state back to |0>/|1> so a Z measurement checks it."""
    if label[1] == 'x': qc.h(q)
    elif label[1] == 'y': qc.sdg(q); qc.h(q)


def measurement_circuit(steps, mu, label, decoder):
    core, q, r1 = protocol(steps, mu)
    qc = QuantumCircuit(core.num_qubits, 1)
    prep(qc, q, label)
    qc.compose(core, inplace=True)
    qc.unitary(su2(decoder), [r1], label='decode')
    rotate_to_z(qc, r1, label)
    qc.measure(r1, 0)
    qc.name = f"mu={mu:+.3f} in={label}"
    return qc


def fidelity_from_counts(counts, label):
    shots = sum(counts.values())
    want = '0' if label[0] == '+' else '1'
    return counts.get(want, 0) / shots


# ---------------- commands ----------------
def cmd_scan(args):
    best = None
    rows = []
    for steps in range(1, 13):
        for mu in np.linspace(-2 * np.pi, 2 * np.pi, 49):
            I = mutual_info(choi_state(steps, mu))
            rows.append((steps, float(mu), float(I)))
            if best is None or I > best[2]:
                best = (steps, float(mu), float(I))
    steps, mu, I = best
    F, dec = best_decoder(choi_state(steps, mu))
    Fneg, _ = best_decoder(choi_state(steps, -mu))
    F0, _ = best_decoder(choi_state(steps, 0.0))
    out = {"steps": steps, "mu": mu, "I_bits": I, "F_best": F, "F_minus_mu": Fneg,
           "F_mu0": F0, "decoder": dec.tolist(), "grid": rows}
    json.dump(out, open(f'scan_result_{MODEL}.json', 'w'), indent=1)
    print(f"[{MODEL}] best: steps={steps}, mu={mu:+.3f}, I(P:R1)={I:.3f} bits (max 2)")
    print(f"exact average fidelity: best mu {F:.3f} | opposite sign {Fneg:.3f} | mu=0 {F0:.3f}  (classical limit 0.667)")


def load_settings():
    f = f'scan_result_{MODEL}.json'
    if not os.path.exists(f):
        sys.exit(f"run `python wormhole_ibm.py scan --model {MODEL}` first")
    return json.load(open(f))


def build_batch(s):
    circs, meta = [], []
    for mu in (s['mu'], -s['mu'], 0.0):
        dec = best_decoder(choi_state(s['steps'], mu))[1] if mu != 0 else np.zeros(3)
        if mu == s['mu']:
            dec = np.array(s['decoder'])
        for lab in INPUTS:
            circs.append(measurement_circuit(s['steps'], mu, lab, dec))
            meta.append((mu, lab))
    return circs, meta


def summarise(meta, all_counts, title):
    res = {}
    for (mu, lab), c in zip(meta, all_counts):
        res.setdefault(mu, []).append(fidelity_from_counts(c, lab))
    best_mu = meta[0][0]
    print(title)
    summary = {}
    for mu, fs in res.items():
        F = float(np.mean(fs))
        shots = sum(all_counts[0].values()) * len(fs)
        err = np.sqrt(F * (1 - F) / shots)
        summary[f"{mu:+.3f}"] = {"F": F, "stderr": float(err)}
        tag = ("coupling ON (best setting)" if mu == best_mu else
               "no coupling (control)" if mu == 0 else "coupling, opposite sign")
        print(f"  mu={mu:+.3f}  F_avg = {F:.3f} ± {err:.3f}   [{tag}]")
    print("  (0.500 = nothing gets through, 0.667 = classical limit)")
    return summary


def cmd_simulate(args):
    from qiskit_aer import AerSimulator
    s = load_settings()
    circs, meta = build_batch(s)
    ideal = AerSimulator()
    counts = ideal.run(transpile(circs, ideal), shots=args.shots, seed_simulator=7).result().get_counts()
    out = {"ideal": summarise(meta, counts, "\nNoiseless simulator:")}
    try:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        fake = FakeFez()
        noisy = AerSimulator.from_backend(fake)
        tc = transpile(circs, fake, optimization_level=3, seed_transpiler=7)
        print(f"\n(transpiled for {fake.name}: ~{max(c.count_ops().get('cz', 0) for c in tc)} two-qubit gates per circuit)")
        counts = noisy.run(tc, shots=args.shots, seed_simulator=7).result().get_counts()
        out["noisy_" + fake.name] = summarise(meta, counts, f"With {fake.name} noise model:")
    except Exception as e:
        print("noisy simulation skipped:", e)
    json.dump(out, open(f'simulate_result_{MODEL}.json', 'w'), indent=1)


def cmd_hardware(args):
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    token = os.environ.get('IBM_QUANTUM_TOKEN')
    if not token:
        sys.exit("set IBM_QUANTUM_TOKEN to your IBM Quantum API key first")
    # Only ever use a free Open Plan instance, so the run can never be billed.
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token, plans_preference=['open'])
    print('instance:', service.active_instance() if hasattr(service, 'active_instance') else 'open plan')
    backend = service.backend(args.backend) if args.backend else service.least_busy(operational=True, simulator=False)
    print("using", backend.name)
    s = load_settings()
    circs, meta = build_batch(s)
    tc = transpile(circs, backend, optimization_level=3, seed_transpiler=7)
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.twirling.enable_gates = True
    job = sampler.run(tc, shots=args.shots)
    print("job id:", job.job_id(), "- waiting in queue (can take a while)...")
    result = job.result()
    counts = [r.data.c.get_counts() for r in result]
    summary = summarise(meta, counts, f"\n{backend.name} (real hardware):")
    json.dump({"backend": backend.name, "job": job.job_id(), "model": MODEL, "summary": summary,
               "counts": counts}, open(f'hardware_result_{MODEL}.json', 'w'), indent=1)


def cmd_fetch(args):
    """Collect a finished hardware job later, e.g. if the session closed while it queued."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get('IBM_QUANTUM_TOKEN')
    if not token or not args.job:
        sys.exit("set IBM_QUANTUM_TOKEN and pass --job <job id>")
    service = QiskitRuntimeService(channel='ibm_quantum_platform', token=token, plans_preference=['open'])
    job = service.job(args.job)
    print("job status:", job.status())
    result = job.result()
    s = load_settings()
    _, meta = build_batch(s)
    counts = [r.data.c.get_counts() for r in result]
    name = job.backend().name
    summary = summarise(meta, counts, f"\n{name} (real hardware):")
    json.dump({"backend": name, "job": args.job, "model": MODEL, "summary": summary,
               "counts": counts}, open(f'hardware_result_{MODEL}.json', 'w'), indent=1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['scan', 'simulate', 'hardware', 'fetch'])
    ap.add_argument('--shots', type=int, default=2000)
    ap.add_argument('--backend', default=None)
    ap.add_argument('--job', default=None, help='job id, for the fetch command')
    ap.add_argument('--model', choices=['chaotic'], default='chaotic')
    a = ap.parse_args()
    MODEL = a.model
    {'scan': cmd_scan, 'simulate': cmd_simulate, 'hardware': cmd_hardware, 'fetch': cmd_fetch}[a.command](a)
