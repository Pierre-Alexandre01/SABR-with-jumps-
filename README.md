# Jump-SABR: Extending the SABR Option Pricing Model with Jumps

**Bachelor Thesis — University of St. Gallen (HSG)**  
Pierre-Alexandre Crouzet | Supervisor: Prof. Enrico Giovanni De Giorgi | Spring Semester 2026

---

## Overview

This thesis extends the SABR stochastic volatility model of Hagan et al. (2002) by incorporating Merton-style compound Poisson jumps, producing a seven-parameter Jump-SABR model. The model is calibrated and evaluated on Bitcoin options traded on the Deribit exchange over the period January 2024 to March 2026.

The dataset comprises 73,819 daily implied volatility observations spanning 805 trading days and 120 distinct expiries.

---

## Key Findings

- Joint calibration of all 7 parameters simultaneously is essential — sequential calibration (SABR first, jumps second) causes jump intensities to collapse to degenerate values.
- Jump-SABR reduces RMSE in implied volatility by **9.4%** relative to pure SABR on a common sample of 697 smiles (paired t-test, p < 0.001).
- Out-of-sample day-ahead prediction confirms structural models generalise better than surface-interpolation benchmarks (Sticky Delta).
- Jump parameters respond systematically to major market events: the January 2024 spot ETF approval, the April 2024 Bitcoin halving, and the November 2024 U.S. presidential election.
- The improvement over SABR is concentrated in high-volatility regimes and longer-dated options.

---

## Repository Structure

```
├── notebooks/
│   └── 01_calibration.ipynb       # Full calibration pipeline and all plots
├── src/
│   ├── sabr.py                    # SABR model and implied vol approximation
│   ├── jump_sabr.py               # Jump-SABR pricing (Poisson-weighted SABR)
│   ├── merton.py                  # Merton jump-diffusion model
│   └── data_loader.py             # Data loading utilities
├── data/
│   ├── btc_daily_smile.csv        # Daily implied vol surface (73,819 obs)
│   ├── calibration_results.csv    # Sequential SABR calibration results
│   ├── joint_calibration_full.csv # Joint Jump-SABR calibration (697 smiles)
│   ├── benchmark_results.csv      # BSM, Sticky Strike, Sticky Delta benchmarks
│   ├── master_results.csv         # Merged results across all models
│   ├── oos_results.csv            # Out-of-sample day-ahead predictions
│   ├── merton_weekly.csv          # Merton model calibration (weekly)
│   └── beta_robustness.csv        # Beta sensitivity analysis
├── results/                       # All generated plots (PNG)
└── setup.py
```

> **Note:** `data/btc_trades_full.csv` (337 MB raw trades) is excluded from the repository due to GitHub's file size limit.

---

## Models

| Model | Parameters | Description |
|---|---|---|
| BSM | 1 | Black-Scholes flat volatility |
| Sticky Strike | 2 | Linear IV in log-moneyness |
| Sticky Delta | 3 | Quadratic IV in log-moneyness |
| SABR | 4 | Hagan et al. (2002) stochastic vol |
| Merton | 3 | Compound Poisson jump-diffusion |
| Jump-SABR | 7 | SABR + Merton jumps (joint calibration) |

---

## References

- Hagan, P.S., Kumar, D., Lesniewski, A.S., & Woodward, D.E. (2002). Managing smile risk. *Wilmott Magazine*.
- Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
- Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, 81(3), 637–654.
