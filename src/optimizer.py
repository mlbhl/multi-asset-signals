"""
optimizer.py — Risk model and mean-variance portfolio optimisation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import cvxpy as cp


# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------
def risk_model(
    ret: pd.DataFrame,
    shrinkage: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Estimate monthly volatilities, (shrunk) correlations, and covariances.

    Parameters
    ----------
    ret       : monthly return DataFrame
    shrinkage : correlation shrinkage factor toward identity matrix (0 = no shrinkage)

    Returns
    -------
    vols       : annualised volatility DataFrame  (monthly index)
    corr_shrunk: shrunk correlation DataFrame     (MultiIndex: date × asset)
    cov_shrunk : shrunk covariance DataFrame      (MultiIndex: date × asset)
    """
    vols = (ret.rolling(24).std() * np.sqrt(12)).dropna()
    corr = ret.rolling(60).corr().dropna()
    # Align vol start date to correlation start
    vols = vols[corr.index.get_level_values(0).min():]

    identity = np.eye(len(ret.columns))
    dates    = vols.index

    corr_shrunk = pd.DataFrame(0.0, index=corr.index, columns=corr.columns)
    cov_shrunk  = pd.DataFrame(0.0, index=corr.index, columns=corr.columns)

    for t in dates:
        c   = corr.loc[t].values
        cs  = (1 - shrinkage) * c + shrinkage * identity
        corr_shrunk.loc[t] = cs
        v   = vols.loc[t].values
        cov_shrunk.loc[t]  = cs * np.outer(v, v)

    return vols, corr_shrunk, cov_shrunk


# ---------------------------------------------------------------------------
# Optimisation strategies
# ---------------------------------------------------------------------------
def _bm_weights(exp_ret: pd.Series, bm: Optional[np.ndarray]) -> np.ndarray:
    """Return benchmark weights (equal weight if bm is None)."""
    if bm is None:
        n = len(exp_ret)
        return np.full(n, 1.0 / n)
    return np.asarray(bm)


def _mvo(
    exp_ret: pd.Series,
    cov: np.ndarray,
    prev: Optional[np.ndarray] = None,
    bm: Optional[np.ndarray] = None,
    min_cons: Optional[np.ndarray] = None,
    max_cons: Optional[np.ndarray] = None,
    min_eq_cons: Optional[float] = None,
    max_eq_cons: Optional[float] = None,
    n_eq: Optional[int] = None,
    ar_cons: Optional[float] = None,
    vol_cons: Optional[float] = None,
    to_cons: Optional[float] = None,
    te_cons: Optional[float] = None,
    gamma: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Mean-variance optimisation (long-only, fully invested).

    Parameters
    ----------
    exp_ret      : expected excess returns (annual %)
    cov          : covariance matrix (annualised)
    prev         : previous portfolio weights (for turnover constraint)
    bm           : benchmark weights (for active-share / TE constraints)
    min_cons     : per-asset lower bounds (array)
    max_cons     : per-asset upper bounds (array)
    min_eq_cons  : minimum total equity allocation
    max_eq_cons  : maximum total equity allocation
    n_eq         : number of equity assets (first n_eq in the weight vector)
    ar_cons      : maximum active share (0-1 scale)
    vol_cons     : maximum annualised volatility (%, e.g. 10 → 10%)
    to_cons      : maximum one-way monthly turnover (%, e.g. 10 → 10%)
    te_cons      : maximum tracking error (%, e.g. 5 → 5%)
    gamma        : risk-aversion coefficient

    Returns
    -------
    Optimal weight array, or None if all solvers fail.
    """
    n = len(exp_ret)
    x = cp.Variable(n)

    pret = np.array(exp_ret) @ x
    risk = cp.quad_form(x, cov)

    cons = [cp.sum(x) == 1]
    cons.append(x >= (0.01 if min_cons is None else min_cons))
    cons.append(x <= (0.50 if max_cons is None else max_cons))

    if min_eq_cons is not None and n_eq is not None:
        cons.append(cp.sum(x[:n_eq]) >= min_eq_cons)
    if max_eq_cons is not None and n_eq is not None:
        cons.append(cp.sum(x[:n_eq]) <= max_eq_cons)

    if ar_cons is not None and bm is not None:
        cons.append(cp.sum(cp.abs(x - bm)) <= 2 * ar_cons)

    if vol_cons is not None:
        risk_target = (vol_cons / (np.sqrt(12) * 100)) ** 2
        cons.append(risk <= risk_target)

    if to_cons is not None and prev is not None:
        cons.append(cp.sum(cp.abs(x - prev)) / 2 <= to_cons / 100)

    if te_cons is not None and bm is not None:
        te_target = (te_cons / (np.sqrt(12) * 100)) ** 2
        cons.append(cp.quad_form(x - bm, cov) <= te_target)

    prob = cp.Problem(cp.Maximize(pret - gamma * risk), cons)

    for solver in ['SCS', 'OSQP', 'ECOS']:
        try:
            prob.solve(solver=solver)
            if x.value is not None:
                return x.value
        except cp.error.SolverError:
            continue

    return None  # all solvers failed


# ---------------------------------------------------------------------------
# Rolling optimised weights
# ---------------------------------------------------------------------------
def generate_opt_weight(
    exp_rets: pd.DataFrame,
    vols: pd.DataFrame,
    corrs: pd.DataFrame,
    covs: pd.DataFrame,
    strategy: str,
    h_period: int = 1,
    bm: Optional[np.ndarray] = None,
    min_cons: Optional[np.ndarray] = None,
    max_cons: Optional[np.ndarray] = None,
    min_eq_cons: Optional[float] = None,
    max_eq_cons: Optional[float] = None,
    n_eq: Optional[int] = None,
    ar_cons: Optional[float] = None,
    vol_cons: Optional[float] = None,
    to_cons: Optional[float] = None,
    te_cons: Optional[float] = None,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate rolling optimal portfolio weights.

    Parameters
    ----------
    exp_rets  : expected return signals (from combined strategy weights)
    vols      : annualised volatility estimates from risk_model()
    corrs     : shrunk correlation estimates from risk_model()
    covs      : shrunk covariance estimates from risk_model()
    strategy  : 'BM' (passive benchmark) or 'SIGNAL' (MVO using signals)
    h_period  : weight-smoothing holding period (months)
    start_date: optional start date filter for output weights

    Returns
    -------
    weights : DataFrame of portfolio weights indexed by rebalancing date
    """
    # Determine rebalancing dates (intersection of signal and risk data)
    reb_date = exp_rets.index[exp_rets.index >= vols.index.min()]

    # Make mutable copies
    vols_  = vols.copy()
    corrs_ = corrs.copy()
    covs_  = covs.copy()

    # Extend risk estimates to the latest signal date if needed
    last_vol  = vols_.index[-1]
    last_cov  = covs_.index[-1][0]
    last_reb  = reb_date[-1]

    if last_vol < last_reb:
        vols_.rename(index={last_vol: last_reb}, inplace=True)
    if last_cov < last_reb:
        corrs_.rename(index={last_cov: last_reb}, level=0, inplace=True)
        covs_.rename(index={last_cov: last_reb},  level=0, inplace=True)

    rows = []
    prev_wgt: Optional[np.ndarray] = None

    for t in reb_date:
        er  = exp_rets.loc[t]
        cov = covs_.loc[t].values if t in covs_.index.get_level_values(0) else None

        if strategy == 'BM':
            wt = _bm_weights(er, bm)

        elif strategy == 'SIGNAL':
            if cov is None:
                wt = _bm_weights(er, bm)
            else:
                solved = _mvo(
                    er, cov,
                    prev=prev_wgt, bm=bm,
                    min_cons=min_cons, max_cons=max_cons,
                    min_eq_cons=min_eq_cons, max_eq_cons=max_eq_cons,
                    n_eq=n_eq,
                    ar_cons=ar_cons, vol_cons=vol_cons,
                    to_cons=to_cons, te_cons=te_cons,
                )
                wt = solved if solved is not None else _bm_weights(er, bm)
            prev_wgt = wt

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        rows.append(pd.DataFrame([wt], index=[t], columns=exp_rets.columns))

    weights = pd.concat(rows)
    weights = weights.rolling(h_period, min_periods=h_period).mean().dropna()

    if start_date is not None:
        weights = weights[start_date:]

    return weights
