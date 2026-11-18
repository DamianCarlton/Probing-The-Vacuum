import json, sys, time, warnings
import numpy as np
from dualnull import evolve, verdict, OM, F
warnings.filterwarnings('ignore')

n = int(sys.argv[1])
cas = [float(c) for c in sys.argv[2].split(',')]
out_file = sys.argv[3]
try:
    store = json.load(open(out_file))
except FileNotFoundError:
    store = []
for ca in cas:
    t = time.time()
    res = evolve(n=n, ghost=[(ca, 3.0, 3.0)] if ca != 0 else None)
    v = verdict(res)
    c = res['centre']
    rec = {'n': n, 'ca': ca, **{k: (float(val) if isinstance(val, (float, np.floating)) else val) for k, val in v.items()},
           'centre_s': res['x'][::max(1, n // 400)].tolist(),
           'centre_r': np.nan_to_num(1 / c[OM][::max(1, n // 400)], nan=-1).tolist(),
           'centre_f': np.nan_to_num(c[F][::max(1, n // 400)], nan=0).tolist()}
    store = [s for s in store if not (s['n'] == n and s['ca'] == ca)] + [rec]
    json.dump(store, open(out_file, 'w'))
    print(f"n={n} ca={ca:+.0e} E0={v['E0']:+.3e} fate={v['fate']} xH={v.get('xH', float('nan')):.3f} rAHmin={v.get('rAH_min', float('nan')):.3f} ({time.time()-t:.0f}s)", flush=True)
