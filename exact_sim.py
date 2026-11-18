"""
Exact (state-vector) simulation of the traversable-wormhole teleportation protocol
(Gao-Jafferis-Wall / Maldacena-Stanford-Yang; Brown et al. 2019 "Quantum gravity in the lab").

Systems: P (reference), Q (message), L (N qubits), R (N qubits).
    |Psi> = e^{-iH_R t} e^{i mu V} e^{-iH_L t} SWAP(Q, L1) e^{iH_L t} |TFD>_LR |Phi+>_PQ
Signal: mutual information I(P : R1). I = 2 bits means perfect teleportation, 0 means none.

H is a chaotic mixed-field Ising chain (real symmetric, so H^T = H):
    H = sum_i Z_i Z_{i+1} + hx sum_i X_i + hz sum_i Z_i
V = (1/(N-1)) sum_{i>=2} Z_Li Z_Ri  (the "negative-energy" coupling; sign of mu matters)
TFD(beta) ∝ (e^{-beta H/2} ⊗ I) |EPR>_LR ; beta=0 is plain Bell pairs.
"""
import numpy as np
from scipy.linalg import expm

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def kron_list(ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def chain_H(N, hx=1.05, hz=0.5):
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        ops = [I2] * N; ops[i] = Z; ops[i + 1] = Z
        H += kron_list(ops)
    for i in range(N):
        ops = [I2] * N; ops[i] = X
        H += hx * kron_list(ops)
        ops = [I2] * N; ops[i] = Z
        H += hz * kron_list(ops)
    return H


class Sim:
    """Qubit layout: index 0 = P, 1 = Q, 2..N+1 = L, N+2..2N+1 = R."""

    def __init__(self, N, beta=0.0, hx=1.05, hz=0.5):
        self.N = N
        self.n = 2 * N + 2
        self.H = chain_H(N, hx, hz)
        w, U = np.linalg.eigh(self.H)
        self.w, self.U = w, U
        self.L = list(range(2, N + 2))
        self.R = list(range(N + 2, 2 * N + 2))
        # TFD on L,R
        d = 2**N
        epr = np.eye(d, dtype=complex).reshape(-1) / np.sqrt(d)  # sum_i |i>_L|i>_R
        A = U @ np.diag(np.exp(-beta * w / 2)) @ U.conj().T
        tfd = (A @ epr.reshape(d, d)).reshape(-1)
        tfd /= np.linalg.norm(tfd)
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)  # P,Q
        self.psi0 = np.kron(bell, tfd).reshape([2] * self.n)
        # diagonal coupling V on (L_i, R_i), i>=2
        zdiag = np.array([1, -1])
        Vdiag = np.zeros([2] * self.n)
        for k in range(1, N):
            shape = [1] * self.n
            sl = zdiag.reshape([2 if q in (self.L[k],) else 1 for q in range(self.n)])
            sr = zdiag.reshape([2 if q in (self.R[k],) else 1 for q in range(self.n)])
            Vdiag = Vdiag + sl * sr
        self.Vdiag = Vdiag / (N - 1)

    def U_t(self, t):
        return self.U @ np.diag(np.exp(-1j * self.w * t)) @ self.U.conj().T

    def apply(self, psi, op, qubits):
        k = len(qubits)
        op = op.reshape([2] * (2 * k))
        psi = np.tensordot(op, psi, axes=(list(range(k, 2 * k)), qubits))
        # tensordot puts new axes first; move them back
        rest = [q for q in range(self.n) if q not in qubits]
        order = list(qubits) + rest
        inv = np.argsort(order)
        return np.transpose(psi, inv)

    def swap(self, psi, a, b):
        return np.swapaxes(psi, a, b)

    def run(self, t, mu):
        psi = self.psi0
        psi = self.apply(psi, self.U_t(-t), self.L)   # e^{+iH_L t}
        psi = self.swap(psi, 1, self.L[0])            # insert message
        psi = self.apply(psi, self.U_t(t), self.L)    # e^{-iH_L t}
        psi = psi * np.exp(1j * mu * self.Vdiag)      # coupling
        psi = self.apply(psi, self.U_t(t), self.R)    # e^{-iH_R t}
        return psi

    def rho_P_R1(self, psi):
        keep = [0, self.R[0]]
        rest = [q for q in range(self.n) if q not in keep]
        m = np.transpose(psi, keep + rest).reshape(4, -1)
        return m @ m.conj().T

    def mutual_info(self, psi):
        rho = self.rho_P_R1(psi)
        r = rho.reshape(2, 2, 2, 2)
        rP = np.einsum('ajbj->ab', r)
        rR = np.einsum('iaib->ab', r)
        S = lambda m: -sum(x * np.log2(x) for x in np.linalg.eigvalsh(m) if x > 1e-12)
        return S(rP) + S(rR) - S(rho)


def best_decoded_fidelity(rho_PR):
    """Max over single-qubit unitaries U on R1 of <Phi+|(I⊗U) rho (I⊗U†)|Phi+>,
    returned as average state fidelity F_avg = (2 F_e + 1)/3. Classical limit: 2/3."""
    from scipy.optimize import minimize
    phi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    def U3(p):
        a, b, c = p
        return expm(-1j * (a * X + b * np.array([[0, -1j], [1j, 0]]) + c * Z))
    def neg(p):
        W = np.kron(I2, U3(p))
        return -np.real(phi.conj() @ W @ rho_PR @ W.conj().T @ phi)
    best = min((minimize(neg, x0, method='Nelder-Mead') for x0 in np.random.default_rng(0).normal(size=(8, 3))),
               key=lambda r: r.fun)
    Fe = -best.fun
    return (2 * Fe + 1) / 3, U3(best.x)


if __name__ == '__main__':
    import json, sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    ts = np.linspace(0, 8, 33)
    mus = np.linspace(-6, 6, 49)
    out = {}
    for beta in [0.0, 1.0]:
        sim = Sim(N, beta=beta)
        grid = np.array([[sim.mutual_info(sim.run(t, mu)) for mu in mus] for t in ts])
        out[str(beta)] = grid.tolist()
        i, j = np.unravel_index(grid.argmax(), grid.shape)
        print(f'beta={beta}: max I={grid.max():.3f} at t={ts[i]:.2f}, mu={mus[j]:.2f}; mu=0 max={grid[:, 24].max():.3f}')
    json.dump({'ts': ts.tolist(), 'mus': mus.tolist(), 'grids': out, 'N': N}, open(f'grid_N{N}.json', 'w'))
