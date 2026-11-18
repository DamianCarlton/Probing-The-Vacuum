"""'Traveller + maintenance' experiment (Shinkai & Hayward Fig. 11, made quantitative).
A normal-matter pulse (the traveller) goes through the throat; a ghost pulse
(negative energy) follows it. Bisect the ghost amplitude to hold the throat open,
and record how long it stays open (|r - a| < 10% at the throat) vs tuning precision."""
import json, sys, time, warnings
import numpy as np
from dualnull import evolve, OM, F
warnings.filterwarnings('ignore')

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2401
TRAVELLER = (0.1, 3.0, 3.0)      # normal scalar pulse: amplitude, c_b, centre l
GHOST_AT = (3.0, 3.6)            # ghost pulse c_b, centre l (just behind the traveller)


def run(g):
    res = evolve(n=n, normal=[TRAVELLER], ghost=[(g, *GHOST_AT)])
    c = res['centre']; s = res['x']
    r = 1 / c[OM]
    tau = np.concatenate([[0], np.cumsum(np.sqrt(2) * np.exp(-np.nan_to_num(c[F][:-1]) / 2) * res['h'])])
    bad = ~np.isfinite(r) | (np.abs(r - 1) > 0.1)
    k = np.argmax(bad) if bad.any() else len(r) - 1
    fate = 'open' if not bad.any() else ('inflation' if np.isfinite(r[k]) and r[k] > 1 else 'collapse')
    return fate, float(tau[k]), float(s[k])


lo, hi = 0.0, 0.5
f_lo, f_hi = run(lo)[0], run(hi)[0]
print('no maintenance:', run(0.0), ' strong ghost:', run(hi), flush=True)
rows = []
for it in range(30):
    mid = 0.5 * (lo + hi)
    fate, tau, s = run(mid)
    rows.append({'g': mid, 'width': hi - lo, 'fate': fate, 'tau_open': tau, 's_open': s})
    print(f'g={mid:.12f} bracket={hi-lo:.1e} fate={fate} open for proper time {tau:.2f} (x={s:.2f})', flush=True)
    if fate == 'collapse':
        lo = mid
    elif fate == 'inflation':
        hi = mid
    else:
        break
json.dump({'n': n, 'traveller': TRAVELLER, 'ghost': GHOST_AT, 'rows': rows}, open(f'maintain_n{n}.json', 'w'), indent=1)
