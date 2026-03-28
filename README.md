# Biomass Conversion Optimization Framework

**DOE + Machine Learning for Agricultural Waste Valorization**

---

## Overview

I am currently developing an open-source computational framework for optimizing the conversion of U.S. agricultural and industrial waste streams into high-value biochemicals at commercially viable yields. This framework combines **Design of Experiments (DOE)** with advanced **Machine Learning** to identify optimal bioprocess conditions across a range of domestic waste feedstocks and target products.

The framework is applicable across multiple waste streams to produce target products.
---

## Current Status

**v0.1 — Optimization Engine Module (Active Development)**

| Module | Status |
|--------|--------|
| DOE Engine (Box-Behnken, CCD) | ✅ Implemented |
| Current ML Optimizer (ANN, RF, GBR, SVR) | ✅ Implemented |
| Sensitivity Analysis (Permutation/Sobol) | ✅ Implemented |
| Optimization Protocol Card Generator | ✅ Implemented |
| Hollow-Fiber Separation Module | 🔄 In progress |
| Future ML optimizer|📋 Planned|
| Process Simulation Engine | 📋 Planned|
| Scale-up Feasibility Assessment | 📋 Planned|

---

## Validated On

The optimization engine has been prototyped and validated using experimental data from the following peer-reviewed publication:

> **Okedi, O.M.** et al. (2024). *Biotechnological Conversion of Yam Peels for Enhanced Citric Acid Production.* **Industrial Crops and Products**, Impact Factor 6.2.
> - DOE: Box-Behnken Design (17 runs, 3 factors)
> - Models: ANN (R² = 0.99883), ANFIS (R² = 0.99880)
> - Result: **49.1% yield improvement** (28.90 → 43.08 g/l citric acid)
> - Key finding: Sodium fluoride is the most influential factor (67.5% variance)

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ogaga-ai/biomass-conversion-optimization.git
cd biomass-conversion-optimization

# Install dependencies
pip install -r requirements.txt

# Launch the demonstration notebook
cd notebooks
jupyter notebook CitricAcid_Optimization.ipynb
```

---

## Repository Structure

```
biomass-conversion-optimization/
├── bcof/
│   ├── optimization/
│   │   ├── doe_engine.py          # DOE matrix generation (BBD, CCD)
│   │   └── ml_optimizer.py        # ANN, RF, GBR, SVR model training + sensitivity
│   ├── separation/                # Hollow-fiber contactor module
│   ├── simulation/                # Process simulation engine (planned)
│   ├── reporting/
│   │   └── protocol_card.py       # Optimization protocol card generator
│   └── utils/
├── notebooks/
│   └── CitricAcid_Optimization.ipynb   # Full demonstration notebook
├── data/
│   └── raw/
│       └── citric_acid_doe_matrix.csv            # Published experimental data
├── outputs/
│   ├── figures/                   # Generated plots and architecture diagrams
│   └── reports/                   # Protocol cards (text output)
├── docs/
├── tests/
├── requirements.txt
└── README.md
```

---

## Technical Stack

- **Python 3.11**
- **NumPy / Pandas** — data manipulation
- **Scikit-learn** — Random Forest, SVR, ANN (MLPRegressor), permutation importance
- **SciPy** — statistical analysis, curve fitting
- **Matplotlib / Seaborn** — visualization
- **Jupyter Notebook** — interactive workflow

---

## Researcher

**Ogaga Maxwell Okedi**
M.S. Chemical Engineering — Florida A&M University (August 2023 - December 2026)
M.S. Computer Science — University of Texas at Dallas (Current)

*Google Scholar: 6 publications | 120 citations | h-index 5*



