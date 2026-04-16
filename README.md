# Jump-SABR: Extending the SABR Option Pricing Model with Jumps

**Bachelor Thesis — University of St. Gallen (HSG)**  
Pierre-Alexandre Crouzet | Supervisor: Prof. Enrico Giovanni De Giorgi | Spring Semester 2026

---

## Overview

This thesis extends the SABR stochastic volatility model of Hagan et al. (2002) by incorporating Merton-style compound Poisson jumps, producing a seven-parameter Jump-SABR model. The model is calibrated and evaluated on Bitcoin options traded on the Deribit exchange over the period January 2024 to March 2026, and benchmarked against five alternative option pricing models.

The dataset comprises 73,819 daily implied volatility observations spanning 805 trading days and 120 distinct expiries.

---

## Key Findings

- Joint calibration of all 7 parameters simultaneously is essential — sequential calibration (SABR first, jumps second) causes jump intensities to collapse to degenerate values (identification failure).
- Jump-SABR reduces RMSE in implied volatility by **9.1%** relative to pure SABR on a common sample of 697 smiles, with the largest gains in short maturities (<30 days, −10.9%).
- Heston (1993) stochastic volatility performs *worse* than SABR on BTC options (RMSE 0.0121 vs 0.0089) — pure stochastic vol without jumps is insufficient for crypto markets.
- Bates (1996) and Jump-SABR are nearly tied overall (RMSE 0.0082 vs 0.0081), with Jump-SABR winning on 63.4% of individual smiles.
- Jump parameters respond systematically to major market events: the January 2024 spot ETF approval, the April 2024 Bitcoin halving, and the November 2024 U.S. presidential election.
- The improvement over SABR is concentrated in short maturities and high-volatility regimes.

---

## Repository Structure

```
├── notebooks/
│   └── 01_calibration.ipynb          # Full calibration pipeline and all 19 plots
├── src/
│   ├── sabr.py                       # SABR model (Hagan 2002) and Black-Scholes pricer
│   ├── jump_sabr.py                  # Jump-SABR pricing and joint calibration
│   ├── merton.py                     # Merton (1976) jump-diffusion model
│   ├── heston.py                     # Heston (1993) stochastic volatility model
│   ├── bates.py                      # Bates (1996) model: Heston + jumps
│   └── data_loader.py                # Data loading and smile slicing utilities
├── data/
│   ├── btc_daily_smile.csv           # Daily implied vol surface (73,819 obs)
│   ├── calibration_results.csv       # SABR + JS-sequential calibration (4,339 smiles)
│   ├── joint_calibration_full.csv    # JS-joint calibration (697 weekly smiles)
│   ├── benchmark_results.csv         # BSM benchmark RMSEs
│   ├── merton_weekly.csv             # Merton calibration (697 smiles)
│   ├── heston_weekly.csv             # Heston calibration (697 smiles)
│   ├── bates_weekly.csv              # Bates calibration (697 smiles)
│   ├── master_results.csv            # Merged results across all 7 models
│   └── beta_robustness.csv           # Beta (CEV exponent) sensitivity analysis
├── results/                          # All 19 generated plots (PNG)
└── setup.py
```

> **Note:** `data/btc_trades_full.csv` (337 MB raw trades) is excluded from the repository due to GitHub's file size limit.

---

## Models

| Model | Parameters | Description |
|---|---|---|
| BSM | 1 | Black-Scholes flat volatility benchmark |
| Merton (1976) | 4 | Compound Poisson jump-diffusion |
| Heston (1993) | 5 | CIR stochastic variance, mean-reverting |
| Bates (1996) | 8 | Heston + Merton jumps |
| SABR | 4 | Hagan et al. (2002) stochastic vol approximation |
| Jump-SABR (sequential) | 7 | SABR then jumps — identification failure |
| **Jump-SABR (joint)** | **7** | **All parameters jointly — main contribution** |

---

## Results Summary

| Model | Mean RMSE | vs SABR |
|---|---|---|
| BSM | 0.0319 | +258% |
| Merton | 0.0140 | +57% |
| Heston | 0.0121 | +36% |
| Bates | 0.0082 | −8% |
| SABR | 0.0089 | — |
| Jump-SABR (sequential) | 0.0089 | ≈0% *(degenerate)* |
| **Jump-SABR (joint)** | **0.0081** | **−9.1%** |

*697 weekly smiles, January 2024 – March 2026.*

---

## Plots

| # | Description |
|---|---|
| 1 | RMSE comparison — all 7 models by maturity bucket |
| 2 | Smile fit on ETF approval day |
| 3 | Smile fits on key market dates |
| 4 | Jump component analysis (λ, μ_J over time) |
| 5 | Implied vs realised volatility — variance risk premium |
| 6 | BTC price with key market events |
| 7 | Sequential vs joint calibration — identification failure |
| 8 | Realized jump validation |
| 9 | Event windows: λ dynamics around key events |
| 10 | VRP decomposition: SABR vs jump contribution |
| 11 | Jump-SABR improvement distribution (histogram + CDF) |
| 12 | Beta robustness check |
| 13 | Regime analysis |
| 14 | Maturity structure of jump risk |
| 15 | Calibration stability |
| 16 | Strike-level residuals heatmap |
| 17 | Model horse race — all 7 models on one smile |
| 18 | Bates vs Jump-SABR per-smile scatter |
| 19 | RMSE term structure — all models across maturities |

---

## References

- Hagan, P.S., Kumar, D., Lesniewski, A.S., & Woodward, D.E. (2002). Managing smile risk. *Wilmott Magazine*.
- Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
- Heston, S.L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327–343.
- Bates, D.S. (1996). Jumps and stochastic volatility. *Review of Financial Studies*, 9(1), 69–107.
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
- Lewis, A.L. (2001). A simple option formula for general jump-diffusion and other exponential Lévy processes. *Envision Financial Systems*.
