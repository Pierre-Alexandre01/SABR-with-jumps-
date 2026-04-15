import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, brentq


# ─────────────────────────────────────────────
# Black-Scholes helper functions
# ─────────────────────────────────────────────

def bs_call(F, K, T, r, sigma):
    """Black-Scholes call price (forward form)."""
    if sigma <= 0 or T <= 0:
        return max(F - K, 0.0)
    d_plus  = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d_minus = d_plus - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm.cdf(d_plus) - K * norm.cdf(d_minus))


def bs_implied_vol(price, F, K, T, r, tol=1e-8):
    """Invert Black-Scholes to recover implied volatility via Brent's method."""
    intrinsic = max(np.exp(-r * T) * (F - K), 0.0)
    if price <= intrinsic:
        return np.nan
    try:
        vol = brentq(
            lambda s: bs_call(F, K, T, r, s) - price,
            1e-6, 10.0, xtol=tol
        )
        return vol
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────
# Hagan et al. (2002) SABR implied volatility
# ─────────────────────────────────────────────

def hagan_vol(F, K, T, alpha, beta, rho, nu):
    """
    SABR implied volatility approximation from Hagan et al. (2002).

    Parameters
    ----------
    F     : float  Current forward price
    K     : float  Strike price
    T     : float  Time to maturity (in years)
    alpha : float  Initial volatility (alpha > 0)
    beta  : float  CEV elasticity (beta in [0,1])
    rho   : float  Correlation (rho in (-1,1))
    nu    : float  Volatility of volatility (nu > 0)

    Returns
    -------
    sigma_BS : float  SABR implied Black volatility
    """
    # ATM case
    if abs(F - K) < 1e-10:
        F_mid = F
        term1 = alpha / (F_mid ** (1 - beta))
        term2 = 1.0 + (
            ((1 - beta)**2 / 24) * (alpha**2 / F_mid**(2 * (1 - beta)))
            + (rho * beta * nu * alpha) / (4 * F_mid**(1 - beta))
            + (2 - 3 * rho**2) / 24 * nu**2
        ) * T
        return term1 * term2

    # Non-ATM case
    F_mid = np.sqrt(F * K)
    log_FK = np.log(F / K)

    # CEV backbone term
    backbone = (
        alpha / (
            F_mid**(1 - beta) * (
                1
                + ((1 - beta)**2 / 24) * log_FK**2
                + ((1 - beta)**4 / 1920) * log_FK**4
            )
        )
    )

    # z and chi(z) — correlation correction
    z = (nu / alpha) * F_mid**(1 - beta) * log_FK
    if abs(z) < 1e-10:
        z_over_chi = 1.0
    else:
        chi = np.log(
            (np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho)
        )
        z_over_chi = z / chi

    # Time correction term
    time_correction = 1.0 + (
        ((1 - beta)**2 / 24) * (alpha**2 / F_mid**(2 * (1 - beta)))
        + (rho * beta * nu * alpha) / (4 * F_mid**(1 - beta))
        + (2 - 3 * rho**2) / 24 * nu**2
    ) * T

    return backbone * z_over_chi * time_correction


# ─────────────────────────────────────────────
# SABR call price
# ─────────────────────────────────────────────

def sabr_call(F, K, T, r, alpha, beta, rho, nu):
    """SABR call price via Hagan vol composed with Black-Scholes."""
    sigma = hagan_vol(F, K, T, alpha, beta, rho, nu)
    return bs_call(F, K, T, r, sigma)


# ─────────────────────────────────────────────
# SABR calibration
# ─────────────────────────────────────────────

def calibrate_sabr(F, T, strikes, market_vols, beta=1.0, r=0.0):
    """
    Calibrate SABR parameters (alpha, rho, nu) to a market implied vol smile.
    Beta is fixed exogenously (default = 1).

    Parameters
    ----------
    F            : float        Forward price
    T            : float        Maturity (years)
    strikes      : array-like   Strike prices
    market_vols  : array-like   Market implied vols (Black)
    beta         : float        Fixed CEV parameter (default 1.0)
    r            : float        Risk-free rate (default 0.0)

    Returns
    -------
    params : dict   Calibrated {alpha, beta, rho, nu}
    rmse   : float  Root mean squared error on implied vols
    """
    strikes     = np.array(strikes)
    market_vols = np.array(market_vols)

    def objective(x):
        alpha, rho, nu = x
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e6
        model_vols = np.array([
            hagan_vol(F, K, T, alpha, beta, rho, nu) for K in strikes
        ])
        return np.mean((model_vols - market_vols)**2)

    # Grid search for initial guess
    best_obj = np.inf
    best_x0  = [0.3, -0.3, 0.3]
    for a0 in [0.2, 0.5]:
        for r0 in [-0.3, 0.3]:
            for n0 in [0.3, 0.7]:
                val = objective([a0, r0, n0])
                if val < best_obj:
                    best_obj = val
                    best_x0  = [a0, r0, n0]

    result = minimize(
        objective,
        x0     = best_x0,
        method = 'L-BFGS-B',
        bounds = [(1e-4, 5.0), (-0.999, 0.999), (1e-4, 5.0)]
    )

    alpha, rho, nu = result.x
    model_vols = np.array([
        hagan_vol(F, K, T, alpha, beta, rho, nu) for K in strikes
    ])
    rmse = np.sqrt(np.mean((model_vols - market_vols)**2))

    return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu}, rmse