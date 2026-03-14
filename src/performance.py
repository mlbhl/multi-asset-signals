"""
performance.py — Performance measurement utilities.

Convention
----------
price : cumulative price index (excess returns rebased to 1000)
ret   : period excess returns
x     : either price or ret (context determines which is appropriate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = [
    'rebase',
    'annual_factor',
    'calc_nyears',
    'calc_cagr',
    'calc_mean',
    'calc_robust_t',
    'calc_vol',
    'calc_skew',
    'calc_kurt',
    'calc_max',
    'calc_min',
    'calc_sharpe',
    'calc_mdd',
    'calc_te',
    'calc_ir',
    'roll_cagr',
    'roll_mean',
    'roll_vol',
    'roll_sharpe',
    'roll_te',
    'roll_ir',
    'factor_reg',
    'summary_stats',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def rebase(price: pd.DataFrame, start_date: str | None = None) -> pd.DataFrame:
    """Rebase price index to 1000 from start_date (or from first available date)."""
    if start_date is not None:
        price = price[start_date:]
    ret = price.pct_change().fillna(0)
    return np.cumprod(1 + ret) * 1000


def annual_factor(x: pd.DataFrame | pd.Series) -> int:
    """Infer annualisation factor from the frequency of x's date index."""
    n_year = (x.index[-1] - x.index[0]).days / 365
    if n_year == 0:
        return 1
    possible = [260, 52, 26, 13, 12, 6, 4, 2, 1]
    L = np.abs(np.array(possible) - len(x) / n_year)
    return possible[int(np.argmin(L))]


# ---------------------------------------------------------------------------
# Point-in-time statistics
# ---------------------------------------------------------------------------
def calc_nyears(x: pd.DataFrame | pd.Series) -> float:
    return (x.index[-1] - x.index[0]).days / 365


def calc_cagr(price: pd.DataFrame | pd.Series) -> pd.Series | float:
    af = annual_factor(price)
    n  = len(price) - 1
    if n == 0:
        return np.nan
    return (price.iloc[-1] / price.iloc[0]) ** (af / n) - 1


def calc_mean(ret: pd.DataFrame | pd.Series) -> pd.Series | float:
    return ret.mean() * annual_factor(ret)


def calc_robust_t(ret: pd.DataFrame) -> pd.Series:
    """Newey-West (1987) robust t-statistic for mean."""
    n        = len(ret)
    x        = np.ones((n, 1))
    af       = annual_factor(ret)
    lags     = 12 if af == 12 else (10 if af == 260 else 1)
    result   = pd.Series(index=ret.columns, dtype=float)
    for col in ret.columns:
        y   = ret[col].values
        res = sm.OLS(y, x).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        result[col] = res.tvalues[0]
    return result


def calc_vol(ret: pd.DataFrame | pd.Series) -> pd.Series | float:
    return ret.std() * np.sqrt(annual_factor(ret))


def calc_skew(ret: pd.DataFrame | pd.Series) -> pd.Series | float:
    return ret.skew()


def calc_kurt(ret: pd.DataFrame | pd.Series) -> pd.Series | float:
    return ret.kurtosis()


def calc_max(ret: pd.DataFrame | pd.Series) -> pd.Series | float:
    return ret.max()


def calc_min(ret: pd.DataFrame | pd.Series) -> pd.Series | float:
    return ret.min()


def calc_sharpe(ret: pd.DataFrame | pd.Series) -> pd.Series | float:
    return calc_mean(ret) / calc_vol(ret)


def calc_mdd(price: pd.DataFrame | pd.Series) -> pd.Series | float:
    return (price / price.cummax() - 1).min()


def calc_te(ret: pd.DataFrame, bm_ret: pd.Series) -> pd.Series | float:
    """Tracking error (annualised). bm_ret must be a Series."""
    return calc_vol(ret.sub(bm_ret, axis=0))


def calc_ir(ret: pd.DataFrame, bm_ret: pd.Series) -> pd.Series | float:
    """Information ratio. bm_ret must be a Series."""
    excess = ret.sub(bm_ret, axis=0)
    return calc_mean(excess) / calc_vol(excess)


# ---------------------------------------------------------------------------
# Rolling statistics (window in years)
# ---------------------------------------------------------------------------
def roll_cagr(price: pd.DataFrame, window: int) -> pd.DataFrame:
    af = annual_factor(price)
    w  = window * af
    return np.exp(
        np.log1p(price.pct_change())
        .rolling(w, min_periods=w).mean() * af
    ) - 1


def roll_mean(price: pd.DataFrame, window: int) -> pd.DataFrame:
    af = annual_factor(price)
    w  = window * af
    return price.pct_change().rolling(w, min_periods=w).mean() * af


def roll_vol(price: pd.DataFrame, window: int) -> pd.DataFrame:
    af = annual_factor(price)
    w  = window * af
    return price.pct_change().rolling(w, min_periods=w).std() * np.sqrt(af)


def roll_sharpe(price: pd.DataFrame, window: int) -> pd.DataFrame:
    return roll_mean(price, window) / roll_vol(price, window)


def roll_te(price: pd.DataFrame, bm_price: pd.Series, window: int) -> pd.DataFrame:
    """bm_price must be a Series."""
    af     = annual_factor(price)
    w      = window * af
    exret  = price.pct_change().sub(bm_price.pct_change(), axis=0)
    return exret.rolling(w, min_periods=w).std() * np.sqrt(af)


def roll_ir(price: pd.DataFrame, bm_price: pd.Series, window: int) -> pd.DataFrame:
    """bm_price must be a Series."""
    af     = annual_factor(price)
    w      = window * af
    exret  = price.pct_change().sub(bm_price.pct_change(), axis=0)
    rmean  = exret.rolling(w, min_periods=w).mean() * af
    rte    = exret.rolling(w, min_periods=w).std() * np.sqrt(af)
    return rmean / rte


# ---------------------------------------------------------------------------
# Factor regression
# ---------------------------------------------------------------------------
def factor_reg(
    X: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    const: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OLS regression with Newey-West standard errors.

    Returns
    -------
    coef, tval, pval  (all np.ndarray, intercept first if const=True)
    """
    n  = len(X.columns)
    af = annual_factor(X)
    lags = 12 if af == 12 else (10 if af == 260 else 1)

    X_arr = X.to_numpy().reshape(-1, n)
    y_arr = y.to_numpy().reshape(-1, 1)
    if const:
        X_arr = sm.add_constant(X_arr)

    res  = sm.OLS(y_arr, X_arr).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return res.params, res.tvalues, res.pvalues


# ---------------------------------------------------------------------------
# Summary statistics table
# ---------------------------------------------------------------------------
def summary_stats(
    price: pd.DataFrame,
    factor_ret: pd.DataFrame | None = None,
    bm_price: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute a table of performance statistics.

    Parameters
    ----------
    price      : cumulative price index DataFrame (one column per portfolio)
    factor_ret : factor return DataFrame for alpha estimation (optional)
    bm_price   : benchmark price Series for TE / IR (optional)

    Returns
    -------
    DataFrame with statistics as rows and portfolios as columns.
    """
    stat_names = [
        'nyears', 'nsamples', 'cagr', 'mean', 'mean-t-stat',
        'vol', 'skew', 'kurt', 'max', 'min', 'sharpe', 'mdd',
    ]
    if factor_ret is not None:
        stat_names += ['alpha', 'alpha-t-stat']
    if bm_price is not None:
        stat_names += ['te', 'ir']

    table = pd.DataFrame(index=stat_names)

    for col in price.columns:
        p = price[[col]].dropna()
        if p.isnull().all().all():
            table[col] = np.nan
            continue

        r = p.pct_change().dropna()
        af = annual_factor(r)

        def _s(v) -> float:
            """Extract scalar from a Series or return as float."""
            return float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)

        row: dict = {
            'nyears':      calc_nyears(p),
            'nsamples':    len(r),
            'cagr':        _s(calc_cagr(p)),
            'mean':        _s(calc_mean(r)),
            'mean-t-stat': _s(calc_robust_t(r)),
            'vol':         _s(calc_vol(r)),
            'skew':        _s(calc_skew(r)),
            'kurt':        _s(calc_kurt(r)),
            'max':         _s(calc_max(r)),
            'min':         _s(calc_min(r)),
            'sharpe':      _s(calc_sharpe(r)),
            'mdd':         _s(calc_mdd(p)),
        }

        if factor_ret is not None:
            fr = factor_ret.reindex(r.index).dropna()
            r_ = r.loc[fr.index]
            coef, tval, _ = factor_reg(fr, r_)
            row['alpha']        = float(coef[0]) * af
            row['alpha-t-stat'] = float(tval[0])

        if bm_price is not None:
            bm_r = bm_price.pct_change().loc[r.index].dropna()
            r_   = r.loc[bm_r.index]
            te_v = calc_te(r_, bm_r)
            ir_v = calc_ir(r_, bm_r)
            row['te'] = float(te_v.iloc[0]) if isinstance(te_v, pd.Series) else float(te_v)
            row['ir'] = float(ir_v.iloc[0]) if isinstance(ir_v, pd.Series) else float(ir_v)

        table[col] = pd.Series(row)

    return table
