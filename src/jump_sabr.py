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
# Sequential calibration (SABR first, jumps second)
# NOTE: this produces degenerate jump parameters — λ collapses to ~1.0.
# Use calibrate_jump_sabr_joint for thesis results.
# ─────────────────────────────────────────────

def calibrate_jump_sabr(F, T, strikes, market_vols,
                        beta=1.0, r=0.0,
                        T_threshold=0.25, n_terms=50):
    """
    Sequential Jump-SABR calibration (Stage 1: SABR, Stage 2: jumps).

    WARNING: Sequential calibration causes an identification failure —
    SABR absorbs all smile variation in Stage 1, leaving the jump
    parameters unidentified. Jump intensity λ collapses to degenerate
    values. This is documented in Section 3 of the thesis.
    Use calibrate_jump_sabr_joint for economically meaningful results.

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

    # ── Stage 1: calibrate SABR on full smile as starting point ──
    from src.sabr import calibrate_sabr
    sabr_params, _ = calibrate_sabr(F, T, strikes, market_vols, beta=beta, r=r)
    alpha = sabr_params["alpha"]
    rho   = sabr_params["rho"]
    nu    = sabr_params["nu"]

    # ── Stage 2: calibrate jump parameters with SABR fixed ──
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


# ─────────────────────────────────────────────
# Joint calibration — all 7 parameters simultaneously
# This is the method used for thesis results.
# ─────────────────────────────────────────────

def calibrate_jump_sabr_joint(F, T, strikes, market_vols,
                              beta=1.0, r=0.0, n_terms=15):
    """
    Joint calibration of all 7 Jump-SABR parameters simultaneously.

    Resolves the identification failure present in sequential calibration.
    All parameters (alpha, rho, nu, lam, mu_J, delta) are estimated jointly
    via grid search initialisation followed by L-BFGS-B optimisation.

    Parameters
    ----------
    F            : float        Forward price
    T            : float        Maturity (years)
    strikes      : array-like   Strike prices
    market_vols  : array-like   Market implied vols (Black)
    beta         : float        Fixed CEV exponent (default 1.0)
    r            : float        Risk-free rate (default 0.0)
    n_terms      : int          Poisson series truncation (default 15)

    Returns
    -------
    params : dict   All 7 calibrated parameters {alpha, beta, rho, nu, lam, mu_J, delta}
    rmse   : float  Root mean squared error on implied vols
    """
    strikes     = np.array(strikes)
    market_vols = np.array(market_vols)

    def objective(x):
        alpha, rho, nu, lam, mu_J, delta = x
        if (alpha <= 0 or nu <= 0 or delta <= 0 or
                lam <= 0 or abs(rho) >= 1):
            return 1e6
        try:
            model_vols = np.array([
                jump_sabr_implied_vol(F, K, T, r, alpha, beta, rho, nu,
                                     lam, mu_J, delta, n_terms)
                for K in strikes
            ])
            if np.any(np.isnan(model_vols)):
                return 1e6
            return np.mean((model_vols - market_vols)**2)
        except Exception:
            return 1e6

    # Grid search over all parameters
    best_obj = np.inf
    best_x0  = [0.3, -0.3, 0.4, 1.0, -0.1, 0.15]

    for a0 in [0.2, 0.5]:
        for r0 in [-0.3, 0.0, 0.3]:
            for n0 in [0.3, 0.7]:
                for l0 in [0.5, 2.0]:
                    for m0 in [-0.1, 0.1]:
                        for d0 in [0.15, 0.3]:
                            val = objective([a0, r0, n0, l0, m0, d0])
                            if val < best_obj:
                                best_obj = val
                                best_x0  = [a0, r0, n0, l0, m0, d0]

    result = minimize(
        objective,
        x0     = best_x0,
        method = 'L-BFGS-B',
        bounds = [
            (1e-4,  5.0),    # alpha
            (-0.999, 0.999), # rho
            (1e-4,  5.0),    # nu
            (1e-4, 20.0),    # lam
            (-2.0,  2.0),    # mu_J
            (1e-4,  2.0),    # delta
        ]
    )

    alpha, rho, nu, lam, mu_J, delta = result.x
    model_vols = np.array([
        jump_sabr_implied_vol(F, K, T, r, alpha, beta, rho, nu,
                              lam, mu_J, delta, n_terms)
        for K in strikes
    ])
    rmse = np.sqrt(np.mean((model_vols - market_vols)**2))

    return {
        "alpha": alpha, "beta": beta, "rho": rho, "nu": nu,
        "lam": lam, "mu_J": mu_J, "delta": delta,
    }, rmse
