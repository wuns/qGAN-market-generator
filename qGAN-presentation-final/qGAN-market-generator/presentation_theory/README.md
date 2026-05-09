# Theory-only presentation

A standalone Beamer deck containing only the theoretical content of the
project: motivation, methodology, classical and quantum generator
architectures, the entanglement argument, and the evaluation methodology.

This is structured to be compiled and presented **before** simulations
finish — the results section is a single placeholder slide.

## Structure

```
presentation_theory/
├── main.tex                              Document root
├── preamble.tex                          Packages, theme setup
├── sources.bib                           Bibliography (IEEE style)
├── sections/
│   ├── section_1_motivation.tex          Why qGAN, project scope
│   ├── section_2_classical_gan.tex       GAN background, classical correlations
│   ├── section_3_quantum_generator.tex   Architecture, circuit, design rationale
│   ├── section_4_entanglement.tex        What entanglement provides
│   ├── section_5_evaluation.tex          Metrics and experimental design
│   └── section_6_placeholder.tex         "Results pending" placeholder
├── figures/                              (empty for now)
└── zhawbeamer/                           ZHAW theme files (untouched)
```

## Compiling on Overleaf

1. **Menu → Settings → Main document**: `presentation_theory/main.tex`
2. **Compiler**: pdfLaTeX
3. **TeX Live version**: 2023 or later
4. **Recompile**

Biber runs automatically.

## Estimated runtime when presented

12 frames + title + TOC + section dividers + bibliography ≈ 18 slides
total. At ~30s per content frame, this is roughly **6–7 minutes** for
the theoretical part alone — fits the 5–10 min target.

## When simulation results are ready

Two paths:

**Option A — drop-in replacement.** Edit `main.tex` and replace
`\input{sections/section_6_placeholder}` with the results-section
inputs from the main project deck (`section_3_results.tex`,
`section_4_scaling.tex`, `section_5_discussion.tex`). Copy those files
from `../presentation/sections/` if they're not already there.

**Option B — keep both decks separate.** Use `presentation_theory/`
for any contexts where only the theoretical material is needed
(e.g., a teaching seminar or a methods discussion). Use the main
`presentation/` deck for the project final.

## Where figures will go

When ready, the figures from `notebook 03` (`comparison_classical_vs_quantum.png`,
`correlation_*.png`, `qubit_scaling.png`, etc.) will live in
`figures/`. Copy them in or extend `\graphicspath{}` in `preamble.tex`
to also pull from `../results/`.
