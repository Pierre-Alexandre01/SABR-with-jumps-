import numpy as np
import math
from scipy.optimize import minimize
from src.sabr import hagan_vol, bs_call, bs_implied_vol


# ─────────────────────────────────────────────
# Jump-SABR pricing (Part III of thesis)
# ─────────────────────────────────────────────

def jump_sabr_call(F, K, T, r, alpha, beta, rho, nu, lam, mu_J, delta, n_terms=50):
    """
    Jump-SABR call price: Poisson-weighted average of SABR prices
    with jump-adjusted strikes (Theorem III.6.1 in thesis).

    Parameters
    ----------
    F       : float  Forward price
    K       : float  Strike price
    T       : float  Maturity (years)
    r       : float  Risk-free rate
    alpha   : float  SABR initial volatility
    beta    : float  CEV elasticity (fixed, typically 1.0)
    rho     : float  SABR correlation
    nu      : float  SABR vol-of-vol
    lam     : float  Jump intensity (jumps per year)
    mu_J    : float  Mean log-jump size
    delta   : float  Std deviation of log-jump size
    n_terms : int    Poisson series truncation (default 50)

    Returns
    -------
    price : float  Jump-SABR call price
    """
    kappa = np.exp(mu_J + 0.5 * delta**2) - 1.0  # mean proportional jump

    price = 0.0
    for n in range(n_terms):
        # Poisson weight: e^{-lambda*T} * (lambda*T)^n / n!
        weight = np.exp(-lam * T) * (lam * T)**n / math.factorial(n)

        # Jump-adjusted strike (equation III.6.2 in thesis)
        K_n = K * np.exp(-n * mu_J - 0.5 * n * delta**2 + lam * kappa * T)

        # SABR implied vol with adjusted strike
        sigma_n = hagan_vol(F, K_n, T, alpha, beta, rho, nu)

        # Black-Scholes price with adjusted strike and SABR vol
        price += weight * bs_call(F, K_n, T, r, sigma_n)

    return price


def jump_sabr_implied_vol(F, K, T, r, alpha, beta, rho, nu,
                          lam, mu_J, delta, n_terms=50):
    """Implied vol of the Jump-SABR price, obtained by inverting Black-Scholes."""
    price = jump_sabr_call(F, K, T, r, alpha, beta, rho, nu,
                           lam, mu_J, delta, n_terms)
    return bs_implied_vol(price, F, K, T, r)


# ─────────────────────────────────────────────
# Jump-SABR calibration (three-stage, Part III)
# ─────────────────────────────────────────────

def calibrate_jump_sabr(F, T, strikes, market_vols,
                        beta=1.0, r=0.0,
                        T_threshold=0.25, n_terms=50):
    """
    Three-stage Jump-SABR calibration (Section III.8 of thesis):
      Stage 1 — Fix beta exogenously (default beta=1)
      Stage 2 — Calibrate SABR (alpha, rho, nu) on long-maturity options
      Stage 3 — Calibrate jumps (lam, mu_J, delta) on short-maturity options

    Parameters
    ----------
    F            : float        Forward price
    T            : float        Maturity (years) — single slice calibration
    strikes      : array-like   Strike prices
    market_vols  : array-like   Market implied vols
    beta         : float        Fixed CEV parameter (default 1.0)
    r            : float        Risk-free rate (default 0.0)
    T_threshold  : float        Short/long maturity split (default 3 months)
    n_terms      : int          Poisson series truncation

    Returns
    -------
    params : dict   All 7 calibrated parameters
    rmse   : float  Final RMSE on implied vols
    """
    strikes     = np.array(strikes)
    market_vols = np.array(market_vols)

    # ── Stage 2: calibrate SABR on full smile as starting point ──
    from src.sabr import calibrate_sabr
    sabr_params, _ = calibrate_sabr(F, T, strikes, market_vols, beta=beta, r=r)
    alpha = sabr_params["alpha"]
    rho   = sabr_params["rho"]
    nu    = sabr_params["nu"]

    # ── Stage 3: calibrate jump parameters with SABR fixed ──
    def objective(x):
        lam, mu_J, delta = x
        if lam <= 0 or delta <= 0:
            return 1e6
        try:
            model_vols = np.array([
                jump_sabr_implied_vol(
                    F, K, T, r, alpha, beta, rho, nu,
                    lam, mu_J, delta, n_terms
                )
                for K in strikes
            ])
            if np.any(np.isnan(model_vols)):
                return 1e6
            return np.mean((model_vols - market_vols)**2)
        except Exception:
            return 1e6

    # Grid search for jump parameter initialisation
    best_obj = np.inf
    best_x0  = [1.0, -0.1, 0.15]
    for l0 in [1.0, 3.0]:
        for m0 in [-0.1, 0.1]:
            for d0 in [0.15, 0.3]:
                val = objective([l0, m0, d0])
                if val < best_obj:
                    best_obj = val
                    best_x0  = [l0, m0, d0]

    result = minimize(
        objective,
        x0     = best_x0,
        method = 'L-BFGS-B',
        bounds = [
            (1e-4, 20.0),   # lam
            (-2.0,  2.0),   # mu_J
            (1e-4,  2.0),   # delta
        ]
    )

    lam, mu_J, delta = result.x

    # Final RMSE
    model_vols = np.array([
        jump_sabr_implied_vol(
            F, K, T, r, alpha, beta, rho, nu,
            lam, mu_J, delta, n_terms
        )
        for K in strikes
    ])
    rmse = np.sqrt(np.mean((model_vols - market_vols)**2))

    params = {
        "alpha": alpha,
        "beta":  beta,
        "rho":   rho,
        "nu":    nu,
        "lam":   lam,
        "mu_J":  mu_J,
        "delta": delta,
    }

    return params, rmse