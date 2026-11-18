import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
runs = json.load(open('runs.json'))
mt = json.load(open('maintain_n2401.json'))

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2})
fig, axs = plt.subplots(3, 1, figsize=(7.6, 14), facecolor=SURF, gridspec_kw={'hspace': 0.72})
for ax in axs:
    ax.set_facecolor(SURF)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'):
        ax.spines[s_].set_color(GRID)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def title(ax, t, sub):
    ax.set_title(t, loc='left', fontsize=13, fontweight='bold', pad=26)
    ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=9.5, color=INK2)


# --- A: the fork ---
a = axs[0]
def get(n, ca):
    return next(r for r in runs if r['n'] == n and r['ca'] == ca)
for ca, col, lab, ls in [(0.01, BLUE, 'Extra negative energy → inflates', '-'),
                         (-0.01, ORANGE, 'Less negative energy → black hole', '-'),
                         (0.0, INK2, 'No push (numerical error alone)', (0, (4, 3)))]:
    r = get(4801 if ca != 0.01 else 2401, ca)
    s = np.array(r['centre_s']); R = np.array(r['centre_r']); f = np.array(r['centre_f'])
    tau = np.concatenate([[0], np.cumsum(np.sqrt(2) * np.exp(-f[:-1] / 2) * np.diff(s))])
    m = R > 0
    a.plot(tau[m], R[m], color=col, lw=2, ls=ls, label=lab)
a.set_yscale('log'); a.set_ylim(0.03, 30); a.set_xlim(0, 14)
a.set_xlabel('Proper time at the throat (units of throat radius ÷ c)')
a.set_ylabel('Throat radius (start = 1)')
title(a, 'A wormhole cannot sit still', 'A tiny nudge either way decides its fate within a few time units')
a.legend(frameon=False, fontsize=9.5, loc='upper left')

# --- B: collapse time vs energy ---
b = axs[1]
best = {}
for r in runs:
    if r['ca'] < 0 and r.get('xH'):
        if r['ca'] not in best or r['n'] > best[r['ca']]['n']:
            best[r['ca']] = r
E = np.array([v['E0'] for v in best.values()]); X = np.array([v['xH'] for v in best.values()])
paperE = np.array([4.39e-5, 4.38e-4, 4.32e-3, 3.70e-2]); paperX = np.array([8.42, 6.98, 5.64, 4.46])
b.scatter(E, X, s=90, color=BLUE, edgecolor=SURF, lw=2, zorder=3, label='This simulation')
b.scatter(paperE, paperX, s=80, color=ORANGE, marker='D', edgecolor=SURF, lw=2, zorder=3, label='Shinkai & Hayward 2002')
ee = np.logspace(-4.6, -1.3, 50)
c0 = np.mean(X + 0.598 * np.log(E))
b.plot(ee, c0 - 0.598 * np.log(ee), color=INK2, lw=1.2, ls=(0, (4, 3)), label='Slope set by the unstable mode (0.85 timescale)')
b.set_xscale('log')
b.set_xlabel('Energy of the push (fraction of throat radius, G = c = 1)')
b.set_ylabel('Horizon forms at null time x⁻')
title(b, 'Smaller push, later collapse, but never safe', 'Every 10× smaller push delays collapse by only ~1.4 units')
b.legend(frameon=False, fontsize=9.5, loc='upper right')

# --- C: maintenance ---
c = axs[2]
rows = [r for r in mt['rows'] if r['width'] < 0.05]
w = np.array([r['width'] for r in rows]); t = np.array([r['tau_open'] for r in rows])
c.scatter(w, t, s=70, color=BLUE, edgecolor=SURF, lw=2, zorder=3)
p = np.polyfit(np.log10(w), t, 1)
ww = np.logspace(np.log10(w.min()), np.log10(w.max()), 20)
c.plot(ww, np.polyval(p, np.log10(ww)), color=INK2, lw=1.2, ls=(0, (4, 3)))
c.text(3e-6, np.polyval(p, np.log10(3e-6)) - 3.2, f'~{-p[0]:.1f} time units\nper extra digit', fontsize=9.5, color=INK2)
c.set_xscale('log'); c.invert_xaxis()
c.set_xlabel('Precision of the balancing negative-energy pulse (→ finer)')
c.set_ylabel('Time held open (throat radius ÷ c)')
title(c, 'Holding it open after a traveller passes', 'Each extra decimal digit of fine-tuning buys only ~2 time units')

fig.savefig('wormhole_fate.png', dpi=170, bbox_inches='tight', facecolor=SURF)
print('slope per digit', p[0])
