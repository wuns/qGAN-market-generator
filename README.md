# qGAN Market Generator

Hybrid quantum GAN for synthetic financial time series - final project for the ZHAW SoE
Continuing Education course on Quantum Computing (Spring 2026).

## Question

Can a hybrid quantum-classical GAN, where only the generator is replaced by a small
variational quantum circuit, produce synthetic joint return time series
that match the statistical properties of real returns at least as well as a comparable
classical GAN baseline?

## Approach

Three notebooks, sharing common code in `src/`:

1. `notebooks/01_classical_baseline.ipynb` — small classical MLP-GAN trained on SMI
   log-return windows (length 20). Establishes baseline metrics.
2. `notebooks/02_quantum_gan.ipynb` — same training loop and discriminator, but the
   generator is a small variational circuit (PennyLane).
3. `notebooks/03_comparison.ipynb` — loads results from both, produces the side-by-side
   plots and tables used in the final presentation.

## Project layout

```
qGAN-market-generator/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py                       # Yahoo download, log-returns, scaling, windowing
│   ├── models.py                     # ClassicalGenerator, Discriminator
│   ├── quantum_models.py             # QuantumGenerator (PennyLane TorchLayer)
│   ├── training.py                   # train_gan, build_experiment dispatcher
│   ├── evaluation.py                 # KS, Frobenius, kurtosis, ACF, plots
│   └── experiment.py                 # ExperimentConfig + run_experiment + run_many
├── notebooks/
│   ├── 01_classical_baseline.ipynb
│   ├── 02_quantum_gan.ipynb
│   └── 03_comparison.ipynb           # orchestrator: runs all configs, builds plots
├── results/                          # PNGs, CSVs, metrics.json (npy/pt gitignored)
│   ├── summary.csv
│   ├── Comparison_GAN_qGAN.png
│   ├── correlation_gan_<assets>.png
│   ├── distributions_gan_<assets>.png
│   ├── qubit_scaling.png
│   └── <run_folder>/metrics.json
└── presentation_theory/              # Beamer deck for May 13 talk
    ├── main.tex                      # title, TOC, section inputs, refs
    ├── preamble.tex                  # packages, theme, density settings, hyperref
    ├── sources.bib                   # IEEE-style references
    ├── README.md
    ├── figures/
    │   ├── comparison_classical_vs_quantum.png
    │   ├── correlation_classical_vs_quantum.png
    │   ├── distributions_classical_vs_quantum.png
    │   └── qubit_scaling.png
    ├── sections/
    │   ├── section_1_motivation.tex          # why qGAN, project scope
    │   ├── section_2_classical_gan.tex       # GAN basics, classical correlations
    │   ├── section_3_quantum_generator.tex   # hybrid pipeline, circuit details
    │   ├── section_4_entanglement.tex        # Bell example, two routes
    │   ├── section_5_evaluation.tex          # metrics, experimental design
    │   ├── section_6_results.tex             # headline comparison
    │   ├── section_7_distributions.tex       # marginal histograms
    │   ├── section_8_correlation.tex         # cross-asset heatmaps
    │   ├── section_9_scaling.tex             # qubit scaling + barren plateau
    │   └── section_10_discussion.tex         # what we found, take-aways
    └── zhawbeamer/                   # ZHAW theme files (untouched)
        ├── beamercolorthemezhaw.sty
        ├── beamerinnerthemezhaw.sty
        ├── beamerouterthemezhaw.sty
        ├── beamerthemezhaw.sty
        ├── zhaw_logo_blue.png
        └── zhaw_logo_white.png
```

## Reproducibility

- All randomness controlled via `set_seed(seed)` in `src.training`
- Library versions pinned in `requirements.txt`
- Each notebook saves its artefacts (samples, model weights, metrics) into `results/`

## Metrics

**Statistical:** mean, std, skewness, excess kurtosis, Kolmogorov–Smirnov distance,
ACF of returns and squared returns (volatility clustering).

**Computational:** parameter count, training time, inference throughput.

**Uncertainty:** each experiment is run with multiple seeds; mean ± std reported.

## Running on Google Colab

```python
!git clone https://github.com/wuns/qGAN-market-generator.git
%cd qGAN-market-generator
!pip install -q -r requirements.txt
import sys; sys.path.insert(0, '/content/qGAN-market-generator')
```
