# qGAN Market Generator

Hybrid quantum GAN for synthetic financial time series — final project for the ZHAW SoE
Continuing Education course on Quantum Computing (Spring 2026).

## Question

Can a hybrid quantum-classical GAN, where only the generator is replaced by a small
variational quantum circuit, produce synthetic Swiss Market Index (SMI) log-return windows
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
│   ├── data.py           # SMI download, log-returns, windowing, scaling
│   ├── models.py         # classical Generator and Discriminator
│   ├── training.py       # shared training loop
│   └── evaluation.py     # metrics, plots, summary tables
├── notebooks/
│   └── 01_classical_baseline.ipynb
└── results/              # .npy, .json, .pt artefacts (gitignored if large)
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

## Running locally

```bash
pip install -r requirements.txt
jupyter lab
```

## Running on Google Colab

```python
!git clone https://github.com/wuns/qGAN-market-generator.git
%cd qGAN-market-generator
!pip install -q -r requirements.txt
import sys; sys.path.insert(0, '/content/qGAN-market-generator')
```
