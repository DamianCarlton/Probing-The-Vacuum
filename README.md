# Probing the Vacuum: wormhole instability and wormhole-inspired teleportation

Two small, self-contained computational studies of traversable wormholes, by Gerald Kihara Muchiri (Nairobi, Kenya).

1. **Wormhole evolution** (`wormhole_evolution/`): a compact Python reproduction of Shinkai & Hayward (2002), the dual-null evolution of the Ellis / Morris–Thorne wormhole held open by a ghost scalar field.
2. **Teleportation on quantum hardware** (`quantum_teleportation/`): a wormhole-inspired teleportation circuit, simulated exactly and run on QuTech's Tuna-17 processor.

## Main results

**Wormhole evolution**

- The static wormhole is unstable to any perturbation: positive input energy collapses it to a black hole, negative input energy inflates it, for pushes from 10⁻¹ down to 10⁻⁴ in amplitude.
- Horizon-formation times reproduce Shinkai & Hayward's table to within 0.2 null-time units; input energies agree to within 3–4%.
- Rounding error alone collapses the static solution, growing with an e-folding time of 0.86 a/c, against 0.846 from the linear analysis of González, Guzmán & Sarbach (2009).
- Shinkai & Hayward's collapse-time exponent (−0.60) equals that linear growth rate divided by √2 (0.846/√2 = 0.598); we measure −0.59.
- Holding the throat open after a traveller passes costs about one decimal digit of fine-tuning per 1.9 a/c of extra opening: for a 1 km throat, about 160,000 digits per second.
- Not reproduced: the paper's final black-hole mass (about 0.30a) and inflationary expansion rate (about 1.1/a), whose extraction is not fully specified in the paper.

**Teleportation on hardware**

- Exact simulation: average teleportation fidelity 0.594 (7-qubit circuit) and 0.559 (5-qubit circuit) with coupling, 0.500 without.
- An IBM Heron noise model keeps a clear signal (gap about 0.06 above the no-coupling control).
- On Tuna-17 (17 November 2026) both circuits came back flat: gaps of 0.009 ± 0.007 and −0.004 ± 0.003. Hardware noise erased the signal.

![Wormhole fate](wormhole_evolution/wormhole_fate.png)

## Running it

```
pip install -r requirements.txt

# Wormhole evolution (about 8 s per run at 2,401 grid points)
cd wormhole_evolution
python batch.py 2401 0.01,-0.01,0.001,-0.001 runs.json
python maintain.py 2401
python wormhole_fate_chart.py

# Teleportation circuits
cd ../quantum_teleportation
python wormhole_ibm.py scan        # exact analysis of the 7-qubit circuit
python wormhole_ibm.py simulate    # noiseless and IBM-noise-model predictions
python exact_sim.py 5              # larger exact simulation, temperature dependence
```

The `.ipynb` notebooks run the hardware experiments from Google Colab (IBM Quantum Platform or Quantum Inspire). Hardware runs need your own account; no credentials are stored in this repository.

## How this work was made

The code, simulations and analysis were developed with substantial assistance from Claude (Anthropic), an AI system, directed and checked by the author. All results can be regenerated from the code here.

## Citing

If you use this code or these results, please cite this repository (see `CITATION.cff`; a DOI is issued through Zenodo) and the original papers:

- H. Shinkai and S. A. Hayward, Phys. Rev. D 66, 044005 (2002), arXiv:gr-qc/0205041.
- J. A. González, F. S. Guzmán and O. Sarbach, Class. Quantum Grav. 26, 015010 and 015011 (2009), arXiv:0806.0608 and arXiv:0806.1370.
- D. Jafferis et al., Nature 612, 51 (2022).

## Contact and collaboration

Questions, corrections and collaboration enquiries are welcome through GitHub issues or by email (see the author's GitHub profile).

## License

Code: MIT License (see `LICENSE`).
