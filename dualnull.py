"""
Dual-null evolution of the Ellis / Morris-Thorne wormhole, after
Shinkai & Hayward, PRD 66, 044005 (2002), arXiv:gr-qc/0205041.

Metric: ds^2 = r^2 dS^2 - 2 e^{-f} dx+ dx-,  Omega = 1/r,
ghost scalar phi (negative energy) and normal scalar psi, both massless.
First-order variables: theta± = 2 d±r, nu± = d±f, P± = r d±phi, Q± = r d±psi.

  d±theta± = -nu± theta± - 2 Omega Q±^2 + 2 Omega P±^2
  d±theta∓ = -Omega (theta+ theta-/2 + e^{-f})
  d±nu∓    = -Omega^2 (theta+ theta-/2 + e^{-f} - 2 Q+Q- + 2 P+P-)
  d±P∓     = -Omega theta∓ P±/2,   d±Q∓ = -Omega theta∓ Q±/2
  d±Omega  = -Omega^2 theta±/2,    d±f = nu±, d±phi = Omega P±, d±psi = Omega Q±

Characteristic initial data on Sigma+ (x-=0) and Sigma- (x+=0), meeting at the
throat sphere S. The grid is swept along anti-diagonals (all points on one
diagonal are independent), with a trapezoidal predictor-corrector.

Units: throat radius a = 1, G = 1. Misner-Sharp energy
  E = (1/(2 Omega)) (1 + e^f theta+ theta- / 2).
"""
import numpy as np

# variable indices
OM, TP, TM, F, NP, NM, PP, PM, QP, QM, PHI, PSI = range(12)
NV = 12
PLUS_SET = [OM, TP, TM, F, NM, PM, QM, PHI, PSI]    # obtained by integrating d+
MINUS_SET = [OM, TP, TM, F, NP, PP, QP, PHI, PSI]   # obtained by integrating d-
SHARED = [OM, TP, TM, F, PHI, PSI]
ONLY_PLUS = [NM, PM, QM]
ONLY_MINUS = [NP, PP, QP]
EXCISE = 25.0
R_EXCISE = 0.02   # excise where areal radius < this (deep inside a black hole)


def rhs_plus(u):
    Om, tp, tm, f = u[OM], u[TP], u[TM], u[F]
    ef = np.exp(-f)
    d = np.zeros_like(u)
    d[OM] = -Om**2 * tp / 2
    d[TP] = -u[NP] * tp - 2 * Om * u[QP]**2 + 2 * Om * u[PP]**2
    d[TM] = -Om * (tp * tm / 2 + ef)
    d[F] = u[NP]
    d[NM] = -Om**2 * (tp * tm / 2 + ef - 2 * u[QP] * u[QM] + 2 * u[PP] * u[PM])
    d[PM] = -Om * tm * u[PP] / 2
    d[QM] = -Om * tm * u[QP] / 2
    d[PHI] = Om * u[PP]
    d[PSI] = Om * u[QP]
    return d


def rhs_minus(u):
    Om, tp, tm, f = u[OM], u[TP], u[TM], u[F]
    ef = np.exp(-f)
    d = np.zeros_like(u)
    d[OM] = -Om**2 * tm / 2
    d[TM] = -u[NM] * tm - 2 * Om * u[QM]**2 + 2 * Om * u[PM]**2
    d[TP] = -Om * (tp * tm / 2 + ef)
    d[F] = u[NM]
    d[NP] = -Om**2 * (tp * tm / 2 + ef - 2 * u[QP] * u[QM] + 2 * u[PP] * u[PM])
    d[PP] = -Om * tp * u[PM] / 2
    d[QP] = -Om * tp * u[QM] / 2
    d[PHI] = Om * u[PM]
    d[PSI] = Om * u[QM]
    return d


def energy(u):
    return (1 + 0.5 * np.exp(u[F]) * u[TP] * u[TM]) / (2 * u[OM])


def static_state(l, a=1.0):
    r = np.sqrt(a * a + l * l)
    u = np.zeros((NV,) + np.shape(l))
    u[OM] = 1 / r
    u[TP] = np.sqrt(2) * l / r
    u[TM] = -np.sqrt(2) * l / r
    u[PP] = a / (np.sqrt(2) * r)
    u[PM] = -a / (np.sqrt(2) * r)
    u[PHI] = np.arctan(l / a)
    return u


def excise(u):
    bad = (u[OM] > 1 / R_EXCISE) | (u[OM] <= 0) | ~np.isfinite(u).all(0)
    u[:, bad] = np.nan
    return u


def integrate_ray(u0, data, rhs, given, solve, h, iters=4):
    """Integrate along a null ray: `given` variables are prescribed (data[k][i]),
    `solve` variables are integrated with the trapezoidal rule."""
    n = data[given[0]].shape[0]
    out = np.zeros((NV, n))
    out[:, 0] = u0
    for k in given:
        out[k] = data[k]
    for i in range(1, n):
        old = out[:, i - 1]
        new = out[:, i].copy()
        new[solve] = old[solve] + h * rhs(old[:, None])[solve, 0]
        for _ in range(iters):
            new[solve] = old[solve] + h * 0.5 * (rhs(old[:, None]) + rhs(new[:, None]))[solve, 0]
        out[:, i] = new
    return out


def evolve(n=2401, L=20.0, ghost=None, normal=None, iters=3, snap_every=None, a=1.0):
    """ghost/normal: lists of (amplitude, width_param c_b, centre c_c) Gaussian pulses
    added to P+ / Q+ on Sigma+ (injected from the right-hand universe).
    Returns diagnostics dict."""
    h = L / (n - 1)
    x = np.linspace(0, L, n)
    # ---- Sigma+ : x- = 0, l = x+/sqrt2 ----
    st = static_state(x / np.sqrt(2), a)
    dataP = {NP: np.zeros(n), PP: st[PP].copy(), QP: np.zeros(n)}
    lP = x / np.sqrt(2)          # proper radial coordinate along Sigma+ (paper's eq. 40 uses l)
    for (ca, cb, cc) in (ghost or []):
        dataP[PP] += ca * np.exp(-cb * (lP - cc)**2)
    for (ca, cb, cc) in (normal or []):
        dataP[QP] += ca * np.exp(-cb * (lP - cc)**2)
    S = static_state(np.array(0.0), a)
    u0 = S.copy(); u0[PP] = dataP[PP][0]; u0[QP] = dataP[QP][0]
    rayP = integrate_ray(u0, dataP, rhs_plus, [NP, PP, QP], PLUS_SET, h)
    # ---- Sigma- : x+ = 0, l = -x-/sqrt2 ----
    st = static_state(-x / np.sqrt(2), a)
    dataM = {NM: np.zeros(n), PM: st[PM].copy(), QM: np.zeros(n)}
    u0 = S.copy(); u0[PP] = dataP[PP][0]; u0[QP] = dataP[QP][0]
    rayM = integrate_ray(u0, dataM, rhs_minus, [NM, PM, QM], MINUS_SET, h)

    # diagnostics storage
    edge_right = np.full((NV, n), np.nan)   # i = n-1 (x+ = L), all j
    edge_left = np.full((NV, n), np.nan)    # j = n-1 (x- = L), all i
    centre = np.full((NV, n), np.nan)       # i = j (l = 0)
    snaps = {}
    se = snap_every or max(1, (n - 1) // 1200)
    ns = (n - 1) // se + 1
    snap_r = np.full((ns, ns), np.nan); snap_tp = np.full((ns, ns), np.nan); snap_tm = np.full((ns, ns), np.nan)

    prev = np.full((NV, n), np.nan)
    prev[:, 0] = rayP[:, 0]
    edge_right_set = []

    def record(k, cur):
        i_lo, i_hi = max(0, k - n + 1), min(k, n - 1)
        if i_hi == n - 1:
            edge_right[:, k - (n - 1)] = cur[:, n - 1]
        if k - i_lo == n - 1:  # j = n-1 at i = i_lo
            edge_left[:, i_lo] = cur[:, i_lo]
        if k % 2 == 0 and i_lo <= k // 2 <= i_hi:
            centre[:, k // 2] = cur[:, k // 2]
        ii = np.arange(i_lo, i_hi + 1)
        jj = k - ii
        m = (ii % se == 0) & (jj % se == 0)
        if m.any():
            snap_r[ii[m] // se, jj[m] // se] = 1 / cur[OM, ii[m]]
            snap_tp[ii[m] // se, jj[m] // se] = cur[TP, ii[m]]
            snap_tm[ii[m] // se, jj[m] // se] = cur[TM, ii[m]]

    record(0, prev)
    for k in range(1, 2 * n - 1):
        i_lo, i_hi = max(0, k - n + 1), min(k, n - 1)
        cur = np.full((NV, n), np.nan)
        # boundary points
        if i_lo == 0:
            cur[:, 0] = rayM[:, k]          # i = 0, j = k
        if i_hi == k:
            cur[:, k] = rayP[:, k]          # i = k, j = 0
        a_ = max(i_lo, 1); b_ = min(i_hi, k - 1)
        if b_ >= a_:
            idx = np.arange(a_, b_ + 1)
            W = prev[:, idx - 1]            # (i-1, j)
            E = prev[:, idx]                # (i, j-1)
            fW, fE = rhs_plus(W), rhs_minus(E)
            N = np.empty_like(W)
            N[ONLY_PLUS] = W[ONLY_PLUS] + h * fW[ONLY_PLUS]
            N[ONLY_MINUS] = E[ONLY_MINUS] + h * fE[ONLY_MINUS]
            N[SHARED] = 0.5 * (W[SHARED] + h * fW[SHARED] + E[SHARED] + h * fE[SHARED])
            for _ in range(iters):
                fN_p, fN_m = rhs_plus(N), rhs_minus(N)
                fromW = W + 0.5 * h * (fW + fN_p)
                fromE = E + 0.5 * h * (fE + fN_m)
                N[ONLY_PLUS] = fromW[ONLY_PLUS]
                N[ONLY_MINUS] = fromE[ONLY_MINUS]
                N[SHARED] = 0.5 * (fromW[SHARED] + fromE[SHARED])
            cur[:, idx] = excise(N)
        record(k, cur)
        prev = cur
    return dict(x=x, h=h, n=n, rayP=rayP, rayM=rayM, edge_right=edge_right, edge_left=edge_left,
                centre=centre, snap_r=snap_r, snap_tp=snap_tp, snap_tm=snap_tm, snap_x=x[::se])


def analyse(res, a=1.0):
    x, n = res['x'], res['n']
    L = x[-1]
    # E0: energy at (x+ = L, x- = 0) minus static value there
    st = static_state(np.array(L / np.sqrt(2)), a)
    E0 = energy(res['rayP'][:, -1]) - energy(st)
    er = res['edge_right']
    tp = er[TP]
    out = {'E0': float(E0)}
    # black hole: theta+ < 0 at the right edge (outgoing light no longer escapes)
    neg = np.where(tp < 0)[0]
    if len(neg):
        j = neg[0]
        # linear interpolation for the zero crossing
        if j > 0 and np.isfinite(tp[j - 1]):
            xh = x[j - 1] + (x[j] - x[j - 1]) * tp[j - 1] / (tp[j - 1] - tp[j])
        else:
            xh = x[j]
        out['xH'] = float(xh)
        out['M'] = float(np.interp(xh, x, energy(er)))
    # inflation: Hubble rate along the centre line l = 0 in proper time
    c = res['centre']
    r = 1 / c[OM]
    dr_ds = 0.5 * (c[TP] + c[TM])                 # d r / d s along x+ = x- = s
    dtau_ds = np.sqrt(2) * np.exp(-c[F] / 2)
    H = dr_ds / (dtau_ds * r)
    out['H_centre'] = H
    out['r_centre'] = r
    return out


def horizons(res, a=1.0):
    """Locate the trapping horizon seen from the right-hand universe on the
    snapshot grid. Black hole: theta+ turns negative (future-trapped).
    Inflation: theta- turns positive (past-trapped, cosmological horizon).
    Returns per-row (x+) horizon x- and areal radius, and the late-time verdict."""
    r, tp, tm, xs = res['snap_r'], res['snap_tp'], res['snap_tm'], res['snap_x']
    rows = []
    for i in range(len(xs)):
        for j in range(1, i):
            if tp[i, j - 1] > 0 and tp[i, j] <= 0:
                w = tp[i, j - 1] / (tp[i, j - 1] - tp[i, j])
                rows.append(('BH', xs[i], xs[j - 1] + w * (xs[j] - xs[j - 1]), r[i, j - 1] + w * (r[i, j] - r[i, j - 1])))
                break
            if tm[i, j - 1] < 0 and tm[i, j] >= 0:
                w = -tm[i, j - 1] / (tm[i, j] - tm[i, j - 1])
                rows.append(('INF', xs[i], xs[j - 1] + w * (xs[j] - xs[j - 1]), r[i, j - 1] + w * (r[i, j] - r[i, j - 1])))
                break
            if not (np.isfinite(tp[i, j]) and np.isfinite(tm[i, j])):
                break
    return rows


def verdict(res, a=1.0):
    """Fate seen from the right-hand universe at late x+.
    Black hole: future-trapped region (theta+ < 0) appears; xH = x- that the
    apparent horizon asymptotes to (the event-horizon generator), r_AH its radius.
    Inflation: past-trapped region (theta- > 0) reaches the right edge."""
    rows = horizons(res, a)
    out = {'E0': analyse(res, a)['E0']}
    bh = [rw for rw in rows if rw[0] == 'BH']
    inf = [rw for rw in rows if rw[0] == 'INF' and rw[1] >= 0.75 * res['x'][-1]]
    if bh and not inf:
        out['fate'] = 'black hole'
        k = int(np.argmax([rw[2] for rw in bh]))
        out['xH'] = float(bh[k][2])
        out['rAH_min'] = float(min(rw[3] for rw in bh))
    elif inf:
        out['fate'] = 'inflation'
        out['x_cosmo'] = float(inf[-1][2])
    else:
        out['fate'] = 'none'
    return out
