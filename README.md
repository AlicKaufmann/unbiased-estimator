# Unbiased Estimation via Randomization — Option Pricing

Course project for *Numerical Integration of Stochastic Differential Equations* (EPFL, 2025/2026).

Implements the **coupled-sum unbiased estimator** of Rhee & Glynn (2015) and compares it with standard Monte Carlo for pricing European call options under the **GBM** and **CIR** models.

## Overview

| Component | Description |
|---|---|
| `config.py` | Model parameters (GBM and CIR) |
| `main.py` | All estimators, schemes, and study functions |
| `documentation/` | LaTeX report and compiled PDF |

### Numerical schemes

- **GBM** — Milstein scheme with Brownian refinement coupling
- **CIR** — drift-implicit Milstein scheme (scheme 25 in the project sheet)
- **CIR** — Lamperti drift-implicit Euler–Maruyama (scheme 23–24)

### Studies implemented

| Function | What it does |
|---|---|
| `coupled_sum_study()` | Q3.2 — coupled-sum estimator for GBM |
| `comparison_study(...)` | Q3.3 / Q5.2 — coupled-sum vs standard MC |
| `drift_implicit_study_strong_order()` | Q4.4 — strong order of convergence for CIR schemes |
| `plot_cir_trajectories()` | Q4.4 — sample paths for both CIR schemes |

## Installation

```bash
pip install .
```

Or in editable mode for development:

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, Matplotlib ≥ 3.7.

## Usage

Run the full study pipeline:

```bash
unbiased-estimator
```

Or from Python:

```python
from main import coupled_sum_study, comparison_study, drift_implicit_study_strong_order
from config import GBM, CIR
from main import milstein_gbm, european, ALPHA_TRUE

# Q3.2: coupled-sum estimator for GBM
coupled_sum_study()

# Q3.3 / Q5.2: compare coupled-sum vs standard MC
comparison_study(
    budgets=[20_000, 100_000, 500_000],
    x0=GBM.s0, maturity=GBM.maturity,
    scheme=milstein_gbm, test_fn=european,
    true_val=ALPHA_TRUE, title="GBM",
)

# Q4.4: strong order of convergence
drift_implicit_study_strong_order()
```

## References

1. M. B. Giles. *Multilevel Monte Carlo Path Simulation.* Operations Research, 56(3):607–617, 2008.
2. A. Neuenkirch and L. Szpruch. *First order strong approximations of scalar SDEs with values in a domain.* 2012.
3. C.-H. Rhee and P. W. Glynn. *Unbiased Estimation with Square Root Convergence for SDE Models.* Operations Research, 63(5):1026–1043, 2015.
