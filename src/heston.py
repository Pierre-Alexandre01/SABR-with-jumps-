import numpy as np
from scipy.optimize import minimize
from src.sabr import bs_implied_vol


# ─────────────────────────────────────────────
# Heston (1993) stochastic volatility model
# Priced via Lewis (2001) single-integral formula
# ─────────────────────────────────────────────

def _heston_cf(u, T, v0, kappa, theta, sigma_v, rho):
    """
    Vectorized Heston characteristic function of log(F_T / F).
    Uses the numerically stable form (Re(d) >= 0 enforced) to avoid
    branch-cut crossings for large T or extreme parameters.

    Parameters
    ----------
    u       : array-like (complex)   Evaluation points; shifted to u-0.5j externally
    T       : float                  Maturity (years)
    v0      : float                  Initial variance (v0 > 0)
    kappa   : float                  Mean-reversion speed
    theta   : float                  Long-run variance
    sigma_v : float                  Volatility of variance
    rho     : float                  Spot-vol correlation

    Returns
    -------
    cf : np.ndarray (complex)   CF values at each u
    """
    u = np.asarray(u, dtype=complex)

    xi = kappa - rho * sigma_v * 1j * u
    d  = np.sqrt(xi**2 + sigma_v**2 * (u**2 + 1j * u))

    # Enforce Re(d) >= 0 for numerical stability (avoids branch-cut crossings)
    d = np.where(np.real(d) >= 0, d, -d)

    g = (xi - d) / (xi + d)

    exp_neg_dT = np.exp(-d * T)
    log_num    = 1.0 - g * exp_neg_dT
    log_den    = 1.0 - g

    # Guard against zero log arguments (pathological params)
    log_num = np.where(np.abs(log_num) < 1e-12, 1e-12 + 0j, log_num)
    log_den = np.where(np.abs(log_den) < 1e-12, 1e-12 + 0j, log_den)

    C = (kappa * theta / sigma_v**2) * (
        (xi - d) * T - 2.0 * (np.log(log_num) - np.log(log_den))
    )
    D = ((xi - d) / sigma_v**2) * (1.0 - exp_neg_dT) / log_num

    return np.exp(C + D * v0)


def heston_call_vec(F, K_arr, T, r, v0, kappa, theta, sigma_v, rho):
    """
    Vectorized Heston call prices via the P1/P2 formula (Heston 1993).
    All strikes priced with two CF evaluations — correct and fast.

    C = e^{-rT} * (F * f1 - K * f2)

    where:
      f2 = 1/2 + 1/pi * int_0^inf Im[exp(-iu*k) * phi_fwd(u)] / u du
      f1 = 1/2 + 1/pi * int_0^inf Im[exp(-iu*k) * phi_fwd(u-i)] / u du

    and phi_fwd(u) = CF of log(F_T/F) under the risk-neutral measure.
    phi_fwd(-i) = 1 for the Heston model (martingale condition for forwards).

    Parameters
    ----------
    F       : float       Forward price
    K_arr   : array-like  Strike prices
    T, r    : float       Maturity, risk-free rate
    v0, kappa, theta, sigma_v, rho : Heston parameters

    Returns
    -------
    prices : np.ndarray  Call prices, same length as K_arr
    """
    K_arr = np.asarray(K_arr, dtype=float)
    if T <= 0:
        return np.maximum(F - K_arr, 0.0)

    # Quadrature grid — integrand Im[CF]/u decays Gaussianly,
    # negligible beyond u ~ 50 for typical BTC options.
    u = np.linspace(1e-5, 100.0, 2000)   # du = 0.05, smooth near u=0

    # CF at real u (for f2) and u-i (for f1), evaluated once for all strikes
    cf2 = _heston_cf(u + 0j, T, v0, kappa, theta, sigma_v, rho)   # (N,) complex
    cf1 = _heston_cf(u - 1j, T, v0, kappa, theta, sigma_v, rho)   # (N,) complex
    # phi_fwd(-i) = 1 for Heston (verified), so no normalisation needed

    # Log-moneyness for all strikes: shape (M,)
    log_KF = np.log(K_arr / F)

    # Phase matrix: shape (M, N)
    phase = np.exp(-1j * np.outer(log_KF, u))   # exp(-iu*k_tilde)

    # f2 integrand: Im[phase * cf2] / u  — shape (M, N)
    integ2 = np.imag(phase * cf2[np.newaxis, :]) / u[np.newaxis, :]
    # f1 integrand: Im[phase * cf1] / u  — shape (M, N)
    integ1 = np.imag(phase * cf1[np.newaxis, :]) / u[np.newaxis, :]

    # Integrate (trapezoid): shape (M,)
    f2 = 0.5 + np.trapz(integ2, u, axis=1) / np.pi
    f1 = 0.5 + np.trapz(integ1, u, axis=1) / np.pi

    # Call prices with intrinsic-value floor
    prices    = np.exp(-r * T) * (F * f1 - K_arr * f2)
    intrinsic = np.exp(-r * T) * np.maximum(F - K_arr, 0.0)
    return np.maximum(prices, intrinsic)


def heston_call(F, K, T, r, v0, kappa, theta, sigma_v, rho):
    """Single Heston call price."""
    return float(heston_call_vec(F, np.array([K]), T, r, v0, kappa, theta, sigma_v, rho)[0])


def heston_implied_vol_vec(F, K_arr, T, r, v0, kappa, theta, sigma_v, rho):
    """Implied vols for multiple strikes (single CF pass — fast for calibration)."""
    K_arr  = np.asarray(K_arr, dtype=float)
    prices = heston_call_vec(F, K_arr, T, r, v0, kappa, theta, sigma_v, rho)
    return np.array([bs_implied_vol(p, F, K, T, r) for p, K in zip(prices, K_arr)])


def heston_implied_vol(F, K, T, r, v0, kappa, theta, sigma_v, rho):
    """Implied vol of a single Heston call price."""
    price = heston_call(F, K, T, r, v0, kappa, theta, sigma_v, rho)
    return bs_implied_vol(price, F, K, T, r)


# ─────────────────────────────────────────────
# Heston calibration
# ─────────────────────────────────────────────

def calibrate_heston(F, T, strikes, market_vols, r=0.0):
    """
    Calibrate Heston parameters (v0, kappa, theta, sigma_v, rho) to market
    implied vols via grid-search initialisation followed by L-BFGS-B.

    Parameters
    ----------
    F            : float        Forward price
    T            : float        Maturity (years)
    strikes      : array-like   Strike prices
    market_vols  : array-like   Market implied vols (Black)
    r            : float        Risk-free rate (default 0.0)

    Returns
    -------
    params : dict   Calibrated {v0, kappa, theta, sigma_v, rho}
    rmse   : float  Root mean squared error on implied vols
    """
    strikes     = np.array(strikes, dtype=float)
    market_vols = np.array(market_vols, dtype=float)

    def objective(x):
        v0, kappa, theta, sigma_v, rho = x
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0 or abs(rho) >= 1:
            return 1e6
        try:
            model_vols = heston_implied_vol_vec(F, strikes, T, r,
                                                v0, kappa, theta, sigma_v, rho)
            if np.any(np.isnan(model_vols)):
                return 1e6
            return float(np.mean((model_vols - market_vols)**2))
        except Exception:
            return 1e6

    # Grid search: 3×3×2×2×3 = 108 points
    best_obj = np.inf
    best_x0  = [0.2, 2.0, 0.2, 0.5, -0.3]

    for v0_0 in [0.1, 0.4, 0.8]:
        for kap_0 in [1.0, 3.0, 8.0]:
            for th_0 in [0.1, 0.5]:
                for sv_0 in [0.3, 1.0]:
                    for rho_0 in [-0.5, 0.0, 0.3]:
                        val = objective([v0_0, kap_0, th_0, sv_0, rho_0])
                        if val < best_obj:
                            best_obj = val
                            best_x0  = [v0_0, kap_0, th_0, sv_0, rho_0]

    result = minimize(
        objective,
        x0     = best_x0,
        method = 'L-BFGS-B',
        bounds = [
            (1e-4, 4.0),    # v0
            (0.1,  20.0),   # kappa
            (1e-4, 4.0),    # theta
            (1e-4, 5.0),    # sigma_v
            (-0.999, 0.999),# rho
        ]
    )

    v0, kappa, theta, sigma_v, rho = result.x
    model_vols = heston_implied_vol_vec(F, strikes, T, r, v0, kappa, theta, sigma_v, rho)
    rmse = float(np.sqrt(np.mean((model_vols - market_vols)**2)))

    return {
        "v0": v0, "kappa": kappa, "theta": theta,
        "sigma_v": sigma_v, "rho": rho,
    }, rmse
