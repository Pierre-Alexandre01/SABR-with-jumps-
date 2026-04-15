import numpy as np
from scipy.optimize import minimize
from src.sabr import bs_implied_vol
from src.heston import _heston_cf, calibrate_heston


# ─────────────────────────────────────────────
# Bates (1996) model: Heston + Merton jumps
# Priced via Lewis (2001) single-integral formula
# ─────────────────────────────────────────────

def _bates_cf(u, T, v0, kappa, theta, sigma_v, rho, lam, mu_J, delta):
    """
    Vectorized Bates (1996) characteristic function of log(F_T / F).
    = Heston CF × Merton-jump correction.

    The jump correction enforces the martingale condition (E[F_T] = F) via
    the compensator term -i*u*kappa_J, where kappa_J = E[J-1] = exp(mu_J + 0.5*delta²) - 1.

    Parameters
    ----------
    u        : array-like (complex)
    T        : float   Maturity
    v0       : float   Initial variance
    kappa    : float   Mean-reversion speed
    theta    : float   Long-run variance
    sigma_v  : float   Vol of variance
    rho      : float   Spot-vol correlation
    lam      : float   Jump intensity (jumps per year)
    mu_J     : float   Mean log-jump size
    delta    : float   Std dev of log-jump size

    Returns
    -------
    cf : np.ndarray (complex)
    """
    u = np.asarray(u, dtype=complex)

    cf_heston = _heston_cf(u, T, v0, kappa, theta, sigma_v, rho)

    kappa_J   = np.exp(mu_J + 0.5 * delta**2) - 1.0
    jump_comp = lam * T * (
        np.exp(1j * u * mu_J - 0.5 * delta**2 * u**2)
        - 1.0
        - 1j * u * kappa_J
    )

    return cf_heston * np.exp(jump_comp)


def bates_call_vec(F, K_arr, T, r, v0, kappa, theta, sigma_v, rho,
                   lam, mu_J, delta):
    """
    Vectorized Bates call prices via the P1/P2 formula (Heston 1993).
    Bates CF = Heston CF × jump correction; same integration structure.

    C = e^{-rT} * (F * f1 - K * f2)

    Parameters
    ----------
    F       : float       Forward price
    K_arr   : array-like  Strike prices
    T, r    : float       Maturity, risk-free rate
    v0, kappa, theta, sigma_v, rho : Heston parameters
    lam, mu_J, delta                : Jump parameters

    Returns
    -------
    prices : np.ndarray  Call prices
    """
    K_arr = np.asarray(K_arr, dtype=float)
    if T <= 0:
        return np.maximum(F - K_arr, 0.0)

    u = np.linspace(1e-5, 100.0, 2000)

    cf2 = _bates_cf(u + 0j, T, v0, kappa, theta, sigma_v, rho,
                    lam, mu_J, delta)             # (N,) real u
    cf1 = _bates_cf(u - 1j, T, v0, kappa, theta, sigma_v, rho,
                    lam, mu_J, delta)             # (N,) u-i shift

    log_KF = np.log(K_arr / F)                   # (M,)
    phase  = np.exp(-1j * np.outer(log_KF, u))   # (M, N)

    integ2 = np.imag(phase * cf2[np.newaxis, :]) / u[np.newaxis, :]
    integ1 = np.imag(phase * cf1[np.newaxis, :]) / u[np.newaxis, :]

    f2 = 0.5 + np.trapz(integ2, u, axis=1) / np.pi
    f1 = 0.5 + np.trapz(integ1, u, axis=1) / np.pi

    prices    = np.exp(-r * T) * (F * f1 - K_arr * f2)
    intrinsic = np.exp(-r * T) * np.maximum(F - K_arr, 0.0)
    return np.maximum(prices, intrinsic)


def bates_call(F, K, T, r, v0, kappa, theta, sigma_v, rho, lam, mu_J, delta):
    """Single Bates call price."""
    return float(bates_call_vec(
        F, np.array([K]), T, r,
        v0, kappa, theta, sigma_v, rho, lam, mu_J, delta
    )[0])


def bates_implied_vol_vec(F, K_arr, T, r, v0, kappa, theta, sigma_v, rho,
                           lam, mu_J, delta):
    """Implied vols for multiple strikes (single CF pass)."""
    K_arr  = np.asarray(K_arr, dtype=float)
    prices = bates_call_vec(F, K_arr, T, r,
                             v0, kappa, theta, sigma_v, rho,
                             lam, mu_J, delta)
    return np.array([bs_implied_vol(p, F, K, T, r) for p, K in zip(prices, K_arr)])


def bates_implied_vol(F, K, T, r, v0, kappa, theta, sigma_v, rho,
                       lam, mu_J, delta):
    """Implied vol of a single Bates call price."""
    price = bates_call(F, K, T, r, v0, kappa, theta, sigma_v, rho, lam, mu_J, delta)
    return bs_implied_vol(price, F, K, T, r)


# ─────────────────────────────────────────────
# Bates calibration
# ─────────────────────────────────────────────

def calibrate_bates(F, T, strikes, market_vols, r=0.0, heston_params=None):
    """
    Calibrate Bates parameters to market implied vols.

    Strategy: if heston_params is provided (warm start from a prior Heston
    calibration), fix the Heston component and grid-search only over jump
    parameters — much faster and avoids identifiability issues.
    Otherwise runs a full joint grid search.

    Parameters
    ----------
    F            : float        Forward price
    T            : float        Maturity (years)
    strikes      : array-like   Strike prices
    market_vols  : array-like   Market implied vols (Black)
    r            : float        Risk-free rate (default 0.0)
    heston_params: dict or None Warm-start Heston params {v0,kappa,theta,sigma_v,rho}

    Returns
    -------
    params : dict   {v0, kappa, theta, sigma_v, rho, lam, mu_J, delta}
    rmse   : float
    """
    strikes     = np.array(strikes, dtype=float)
    market_vols = np.array(market_vols, dtype=float)

    def objective(x):
        v0, kappa, theta, sigma_v, rho, lam, mu_J, delta = x
        if (v0 <= 0 or kappa <= 0 or theta <= 0 or sigma_v <= 0
                or abs(rho) >= 1 or lam <= 0 or delta <= 0):
            return 1e6
        try:
            model_vols = bates_implied_vol_vec(
                F, strikes, T, r,
                v0, kappa, theta, sigma_v, rho, lam, mu_J, delta
            )
            if np.any(np.isnan(model_vols)):
                return 1e6
            return float(np.mean((model_vols - market_vols)**2))
        except Exception:
            return 1e6

    # ── Initialise ──
    if heston_params is not None:
        # Warm start: fix Heston, grid-search only jump params (18 points)
        h = heston_params
        best_obj = np.inf
        best_x0  = [h['v0'], h['kappa'], h['theta'], h['sigma_v'], h['rho'],
                    0.5, -0.05, 0.15]

        for l0 in [0.5, 2.0, 5.0]:
            for m0 in [-0.2, 0.0, 0.2]:
                for d0 in [0.1, 0.3]:
                    x_try = [h['v0'], h['kappa'], h['theta'],
                              h['sigma_v'], h['rho'], l0, m0, d0]
                    val = objective(x_try)
                    if val < best_obj:
                        best_obj = val
                        best_x0  = x_try
    else:
        # Full joint grid search: 3×3×2×2×3 × 3×3×2 = 108 × 18 = ~972 points
        # Reduced to 2×2×2×2×2 × 3×3×2 = 32 × 18 = 576 for speed
        best_obj = np.inf
        best_x0  = [0.2, 2.0, 0.2, 0.5, -0.3, 1.0, -0.1, 0.15]

        for v0_0 in [0.1, 0.5]:
            for kap_0 in [1.0, 5.0]:
                for th_0 in [0.1, 0.4]:
                    for sv_0 in [0.3, 1.0]:
                        for rho_0 in [-0.5, 0.0, 0.3]:
                            for l0 in [0.5, 2.0, 5.0]:
                                for m0 in [-0.2, 0.0, 0.2]:
                                    for d0 in [0.1, 0.3]:
                                        x_try = [v0_0, kap_0, th_0, sv_0,
                                                 rho_0, l0, m0, d0]
                                        val = objective(x_try)
                                        if val < best_obj:
                                            best_obj = val
                                            best_x0  = x_try

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
            (1e-4, 20.0),   # lam
            (-2.0,  2.0),   # mu_J
            (1e-4,  2.0),   # delta
        ]
    )

    v0, kappa, theta, sigma_v, rho, lam, mu_J, delta = result.x
    model_vols = bates_implied_vol_vec(
        F, strikes, T, r, v0, kappa, theta, sigma_v, rho, lam, mu_J, delta
    )
    rmse = float(np.sqrt(np.mean((model_vols - market_vols)**2)))

    return {
        "v0": v0, "kappa": kappa, "theta": theta,
        "sigma_v": sigma_v, "rho": rho,
        "lam": lam, "mu_J": mu_J, "delta": delta,
    }, rmse
