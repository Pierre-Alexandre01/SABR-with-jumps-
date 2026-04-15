import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import minimize
from src.sabr import bs_call, bs_implied_vol


# ─────────────────────────────────────────────
# Merton (1976) jump-diffusion pricing
# ─────────────────────────────────────────────

def merton_call(F, K, T, r, sigma, lam, mu_J, delta, n_terms=50):
    """
    Merton (1976) jump-diffusion call price.
    Poisson-weighted sum of Black-Scholes prices.

    Parameters
    ----------
    F       : float  Forward price
    K       : float  Strike price
    T       : float  Maturity (years)
    r       : float  Risk-free rate
    sigma   : float  Diffusion volatility
    lam     : float  Jump intensity (jumps per year)
    mu_J    : float  Mean log-jump size
    delta   : float  Std deviation of log-jump size
    n_terms : int    Number of Poisson series terms (default 50)

    Returns
    -------
    price : float  Merton call price
    """
    # Mean proportional jump size
    kappa   = np.exp(mu_J + 0.5 * delta**2) - 1.0
    lam_adj = lam * (1.0 + kappa)          # risk-neutral intensity

    price = 0.0
    for n in range(n_terms):
        # Poisson weight
        weight = np.exp(-lam_adj * T) * (lam_adj * T)**n / math.factorial(n)

        # Adjusted volatility and rate for n jumps
        sigma_n = np.sqrt(sigma**2 + n * delta**2 / T)
        r_n     = r - lam * kappa + n * (mu_J + 0.5 * delta**2) / T

        price += weight * bs_call(F, K, T, r_n, sigma_n)

    return price


def merton_implied_vol(F, K, T, r, sigma, lam, mu_J, delta, n_terms=50):
    """Implied vol of the Merton price, obtained by inverting Black-Scholes."""
    price = merton_call(F, K, T, r, sigma, lam, mu_J, delta, n_terms)
    return bs_implied_vol(price, F, K, T, r)


# ─────────────────────────────────────────────
# Merton calibration
# ─────────────────────────────────────────────

def calibrate_merton(F, T, strikes, market_vols, r=0.0, n_terms=50):
    """
    Calibrate Merton parameters (sigma, lam, mu_J, delta) to market implied vols.

    Parameters
    ----------
    F            : float        Forward price
    T            : float        Maturity (years)
    strikes      : array-like   Strike prices
    market_vols  : array-like   Market implied vols (Black)
    r            : float        Risk-free rate (default 0.0)
    n_terms      : int          Poisson series truncation

    Returns
    -------
    params : dict   Calibrated {sigma, lam, mu_J, delta}
    rmse   : float  Root mean squared error on implied vols
    """
    strikes     = np.array(strikes)
    market_vols = np.array(market_vols)

    def objective(x):
        sigma, lam, mu_J, delta = x
        if sigma <= 0 or lam <= 0 or delta <= 0:
            return 1e6
        try:
            model_vols = np.array([
                merton_implied_vol(F, K, T, r, sigma, lam, mu_J, delta, n_terms)
                for K in strikes
            ])
            if np.any(np.isnan(model_vols)):
                return 1e6
            return np.mean((model_vols - market_vols)**2)
        except Exception:
            return 1e6

    # Grid search for initial guess
    best_obj = np.inf
    best_x0  = [0.3, 1.0, -0.1, 0.15]
    for s0 in [0.2, 0.4, 0.7]:
        for l0 in [0.5, 2.0, 5.0]:
            for m0 in [-0.2, 0.0, 0.2]:
                for d0 in [0.1, 0.25, 0.4]:
                    val = objective([s0, l0, m0, d0])
                    if val < best_obj:
                        best_obj = val
                        best_x0  = [s0, l0, m0, d0]

    result = minimize(
        objective,
        x0     = best_x0,
        method = 'L-BFGS-B',
        bounds = [
            (1e-4, 5.0),    # sigma
            (1e-4, 20.0),   # lam
            (-2.0, 2.0),    # mu_J
            (1e-4, 2.0),    # delta
        ]
    )

    sigma, lam, mu_J, delta = result.x
    model_vols = np.array([
        merton_implied_vol(F, K, T, r, sigma, lam, mu_J, delta, n_terms)
        for K in strikes
    ])
    rmse = np.sqrt(np.mean((model_vols - market_vols)**2))

    return {"sigma": sigma, "lam": lam, "mu_J": mu_J, "delta": delta}, rmse