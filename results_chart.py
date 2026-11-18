import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'

sim = json.load(open('simulate_result_chaotic.json'))
ideal = list(sim['ideal'].values())
noisy = list(sim['noisy_fake_fez'].values())
conds = ['Coupling on\n(best sign)', 'Coupling on\n(opposite sign)', 'No coupling\n(control)']

# exact 7-qubit-per-side results (exact_sim.py / explore.py, best over t and |mu|)
betas = ['0 (∞ temp)', '0.5', '1', '2']
best = [0.244, 0.159, 0.125, 0.126]
opp = [0.241, 0.137, 0.077, 0.094]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'text.color': INK,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2})
fig, (a, b) = plt.subplots(2, 1, figsize=(7.6, 9.6), facecolor=SURF, gridspec_kw={'hspace': 0.62})
for ax in (a, b):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# Panel A
x = range(3)
for xi, (fi, fn) in enumerate(zip(ideal, noisy)):
    a.plot([xi, xi], [fn['F'], fi['F']], color=GRID, lw=2, zorder=1)
a.scatter([i for i in x], [d['F'] for d in ideal], s=90, color=BLUE, edgecolor=SURF, lw=2, zorder=3, label='Perfect (noiseless) machine')
a.scatter([i for i in x], [d['F'] for d in noisy], s=90, color=ORANGE, edgecolor=SURF, lw=2, zorder=3, marker='D', label='With IBM Heron noise')
for xi in x:
    hi, lo = ideal[xi]['F'], noisy[xi]['F']
    if abs(hi - lo) < 0.01:
        a.annotate(f"both {hi:.2f}", (xi + 0.09, hi + 0.012), va='center', color=INK, fontsize=10)
    else:
        a.annotate(f"{hi:.2f}", (xi + 0.09, hi), va='center', color=INK, fontsize=10)
        a.annotate(f"{lo:.2f}", (xi + 0.09, lo), va='center', color=INK, fontsize=10)
a.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
a.axhline(2 / 3, color=INK2, lw=1, ls=(0, (1, 2)))
a.text(-0.35, 0.503, 'Nothing gets through (0.50)', ha='left', va='bottom', fontsize=9, color=INK2)
a.text(-0.35, 2 / 3 + 0.003, 'Classical limit (0.67)', ha='left', va='bottom', fontsize=9, color=INK2)
a.set_xticks(list(x)); a.set_xticklabels(conds)
a.set_xlim(-0.4, 2.5); a.set_ylim(0.47, 0.70)
a.set_ylabel('Average teleportation fidelity')
a.set_title('Your 7-qubit IBM run: predicted result', loc='left', fontsize=13, fontweight='bold', pad=26)
a.text(0, 1.04, 'Message gets through only when the sides are coupled, but stays below the classical limit',
       transform=a.transAxes, fontsize=9.5, color=INK2)
a.legend(frameon=False, fontsize=9.5, loc='center right', bbox_to_anchor=(1.0, 0.55))

# Panel B
xb = range(4)
for xi in xb:
    b.plot([xi, xi], [opp[xi], best[xi]], color=GRID, lw=2, zorder=1)
b.scatter(list(xb), best, s=90, color=BLUE, edgecolor=SURF, lw=2, zorder=3, label='Coupling with the "wormhole" sign')
b.scatter(list(xb), opp, s=90, color=ORANGE, edgecolor=SURF, lw=2, zorder=3, marker='D', label='Coupling with the opposite sign')
for xi in xb:
    if abs(best[xi] - opp[xi]) < 0.01:
        b.annotate(f"both {best[xi]:.2f}", (xi + 0.1, best[xi] + 0.012), va='center', fontsize=10)
    else:
        b.annotate(f"{best[xi]:.2f}", (xi + 0.1, best[xi]), va='center', fontsize=10)
        b.annotate(f"{opp[xi]:.2f}", (xi + 0.1, opp[xi]), va='center', fontsize=10)
b.set_xticks(list(xb)); b.set_xticklabels(betas)
b.set_xlabel('Inverse temperature β of the entangled starting state')
b.set_ylabel('Information transmitted (bits, max 2)')
b.set_xlim(-0.4, 3.6); b.set_ylim(0, 0.3)
b.set_title('Bigger exact simulation (7 qubits per side)', loc='left', fontsize=13, fontweight='bold', pad=26)
b.text(0, 1.04, 'At infinite temperature the sign does not matter; once cooled, one sign wins',
       transform=b.transAxes, fontsize=9.5, color=INK2)
b.legend(frameon=False, fontsize=9.5, loc='upper right')

fig.savefig('wormhole_results.png', dpi=170, bbox_inches='tight', facecolor=SURF)
