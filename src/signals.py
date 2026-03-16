"""
signals.py — Return-based signal generation and portfolio weight construction.

Cross-sectional (CS) signals
------------------------------
  mom   : momentum          — Jegadeesh & Titman (1993)
  rev   : long-term reversal — De Bondt & Thaler (1985)
  seas  : seasonality        — Heston & Sadka (2008)

Time-series (TS) signals (sign signals)
-----------------------------------------
  ftrd  : fast trend / TS momentum  — Moskowitz, Ooi & Pedersen (2012)
  strd  : slow trend / TS momentum
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
def generate_signal(
    ret: pd.DataFrame,
    momp:  int = 6,
    momk:  int = 0,
    revp:  int = 60,
    revk:  int = 0,
    seasp: int = 120,
    ftrdp: int = 1,
    strdp: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate five return-based signals from excess returns.

    Parameters
    ----------
    ret   : monthly excess return DataFrame
    momp  : CS momentum lookback (months)
    momk  : CS momentum skip period (most recent months excluded)
    revp  : CS reversal lookback (months)
    revk  : CS reversal skip period
    seasp : CS seasonality lookback (months); must be a multiple of 12
    ftrdp : TS fast-trend lookback (months)
    strdp : TS slow-trend lookback (months)

    Returns
    -------
    mom, rev, seas, ftrd, strd
    """
    # Cross-sectional signals
    mom  = ret.shift(momk).rolling(momp - momk).mean()[momp - 1:]
    rev  = ret.shift(revk).rolling(revp - revk).mean().mul(-1)[revp - 1:]
    seas = (
        sum(ret.shift(t) for t in range(11, seasp, 12)) / (seasp / 12)
    )[seasp - 1:]

    # Time-series signals (sign: +1 / -1 / NaN)
    ftrd = ret.rolling(ftrdp).mean()[ftrdp - 1:].map(
        lambda x: np.nan if np.isnan(x) else (1.0 if x >= 0 else -1.0)
    )
    strd = ret.rolling(strdp).mean()[strdp - 1:].map(
        lambda x: np.nan if np.isnan(x) else (1.0 if x >= 0 else -1.0)
    )

    return mom, rev, seas, ftrd, strd


# ---------------------------------------------------------------------------
# Rank normalisation
# ---------------------------------------------------------------------------
def calc_ranknorm(signal: pd.DataFrame) -> pd.DataFrame:
    """
    Quantile (rank) normalisation to [0, 1] applied row-wise.

    Uses sklearn's QuantileTransformer with uniform output distribution.
    """
    scaled = signal.copy()

    def _norm_row(row: pd.Series) -> pd.Series:
        mask = row.notna()
        n    = mask.sum()
        if n < 2:
            return row
        qt = QuantileTransformer(
            n_quantiles=n, output_distribution='uniform', random_state=0
        )
        row.loc[mask] = qt.fit_transform(
            row[mask].values.reshape(-1, 1)
        ).ravel()
        return row

    return scaled.apply(_norm_row, axis=1)


# ---------------------------------------------------------------------------
# Cross-sectional (CS) weights
# ---------------------------------------------------------------------------
def generate_cs_weight(
    signal: pd.DataFrame,
    method: str = 'quantile',
    quantile: str = 'tercile',
    holding: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct long-short and long-only portfolio weights from CS signals.

    Parameters
    ----------
    signal   : monthly CS signal DataFrame (raw, not yet normalised)
    method   : 'quantile' | 'proportion' | 'rank'
    quantile : 'median' | 'tercile' | 'quartile' | 'quintile' | 'decile'
               (used only when method == 'quantile')
    holding  : holding period in months (Jegadeesh & Titman 1993)

    Returns
    -------
    ls_wgt : long-short weights (long leg sums to +1, short leg sums to -1)
    lo_wgt : long-only weights  (long leg sums to +1)
    """
    _QUANTILE_MAP = {
        'median': 2, 'tercile': 3, 'quartile': 4, 'quintile': 5, 'decile': 10,
    }

    if method == 'quantile':
        quant = _QUANTILE_MAP.get(quantile)
        if quant is None:
            raise ValueError(f"Unknown quantile: {quantile!r}")
        if len(signal.columns) < quant:
            raise ValueError(
                f"Not enough assets ({len(signal.columns)}) for quantile={quantile!r}"
            )

        scaled = calc_ranknorm(signal)
        eps    = 1e-6  # guard against floating-point boundary issues

        # Need at least `quant` valid assets to form quantile groups;
        # rows with fewer valid assets get NaN so weights become zero.
        valid_count = signal.notna().sum(axis=1)
        scaled.loc[valid_count < quant] = np.nan

        ls_wgt = scaled.copy()
        ls_wgt[ls_wgt > 1 - 1 / quant + eps] =  1
        ls_wgt[ls_wgt < 1 / quant - eps]      = -1
        ls_wgt[~((ls_wgt == 1) | (ls_wgt == -1))] = 0

        lo_wgt = scaled.copy()
        lo_wgt[lo_wgt > 1 - 1 / quant + eps] = 1
        lo_wgt[lo_wgt != 1] = 0
        # Normalise: each selected asset receives equal weight summing to 1
        lo_wgt[lo_wgt == 1] = lo_wgt[lo_wgt == 1].div(
            lo_wgt[lo_wgt == 1].count(axis=1), axis=0
        )

    elif method == 'proportion':
        demeaned = signal.sub(signal.mean(axis=1), axis=0)

        ls_wgt = demeaned.copy()
        ls_wgt[ls_wgt > 0] = ls_wgt[ls_wgt > 0].div(
            ls_wgt[ls_wgt > 0].sum(axis=1), axis=0)
        ls_wgt[ls_wgt < 0] = ls_wgt[ls_wgt < 0].div(
            ls_wgt[ls_wgt < 0].sum(axis=1), axis=0).mul(-1)
        ls_wgt.fillna(0, inplace=True)

        lo_wgt = demeaned.copy()
        lo_wgt[lo_wgt <= 0] = 0
        lo_wgt = lo_wgt.div(lo_wgt.sum(axis=1), axis=0)

    elif method == 'rank':
        rank_sig = signal.rank(axis=1, ascending=True).pow(2)

        ls_wgt = rank_sig.sub(rank_sig.mean(axis=1), axis=0)
        ls_wgt[ls_wgt > 0] = ls_wgt[ls_wgt > 0].div(
            ls_wgt[ls_wgt > 0].sum(axis=1), axis=0)
        ls_wgt[ls_wgt < 0] = ls_wgt[ls_wgt < 0].div(
            ls_wgt[ls_wgt < 0].sum(axis=1), axis=0).mul(-1)
        ls_wgt.fillna(0, inplace=True)

        lo_wgt = rank_sig.div(rank_sig.sum(axis=1), axis=0).fillna(0)

    else:
        raise ValueError(f"Unknown method: {method!r}")

    # --- Scale long and short legs separately, then apply holding period ---
    def _rescale(w: pd.DataFrame) -> pd.DataFrame:
        pos = w > 0
        neg = w < 0
        w[pos] = w[pos].div(w[pos].sum(axis=1), axis=0)
        w[neg] = w[neg].div(w[neg].sum(axis=1), axis=0).mul(-1)
        return w

    ls_wgt = _rescale(
        ls_wgt.rolling(holding, min_periods=holding).mean().dropna()
    )
    lo_wgt = lo_wgt.rolling(holding, min_periods=holding).mean().dropna()

    # Drop leading all-zero rows (e.g. when early data has too few assets)
    def _drop_leading_zeros(w: pd.DataFrame) -> pd.DataFrame:
        has_nonzero = (w != 0).any(axis=1)
        if has_nonzero.any():
            return w.loc[has_nonzero.idxmax():]
        return w

    ls_wgt = _drop_leading_zeros(ls_wgt)
    lo_wgt = _drop_leading_zeros(lo_wgt)

    return ls_wgt, lo_wgt


# ---------------------------------------------------------------------------
# Time-series (TS) weights
# ---------------------------------------------------------------------------
def generate_ts_weight(
    signal: pd.DataFrame,
    holding: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct long-short and long-only portfolio weights from TS sign signals.

    TS portfolios have time-varying net long investment (may not be fully invested).
    The gross notional (long + |short|) is scaled to 2 to match CS strategies.

    Parameters
    ----------
    signal  : monthly TS signal DataFrame (values in {+1, -1, NaN})
    holding : holding period in months (Jegadeesh & Titman 1993)

    Returns
    -------
    ls_wgt, lo_wgt
    """
    n_valid = signal.count(axis=1)

    ls_wgt = signal.copy()
    ls_wgt[~((ls_wgt == 1) | (ls_wgt == -1))] = 0
    ls_wgt[ ls_wgt ==  1] = ls_wgt[ls_wgt ==  1].mul(2 / n_valid, axis=0)
    ls_wgt[ ls_wgt == -1] = ls_wgt[ls_wgt == -1].mul(2 / n_valid, axis=0)
    ls_wgt = ls_wgt.rolling(holding, min_periods=holding).mean().dropna()

    lo_wgt = signal.copy()
    lo_wgt[lo_wgt != 1] = 0
    lo_wgt[lo_wgt == 1] = lo_wgt[lo_wgt == 1].mul(2 / n_valid, axis=0)
    lo_wgt = lo_wgt.rolling(holding, min_periods=holding).mean().dropna()

    return ls_wgt, lo_wgt


# ---------------------------------------------------------------------------
# Weight combination
# ---------------------------------------------------------------------------
def generate_average_weight(
    weights: list[pd.DataFrame],
    holding: int = 1,
) -> pd.DataFrame:
    """
    Combine multiple strategy weights by simple averaging (Timmermann 2006).

    Parameters
    ----------
    weights : list of weight DataFrames
    holding : additional holding-period smoothing (months)

    Returns
    -------
    avg_wgt : averaged and smoothed weights
    """
    stacked  = pd.concat(weights)
    avg_wgt  = stacked.groupby(stacked.index).mean()
    avg_wgt  = avg_wgt.rolling(holding, min_periods=holding).mean().dropna()

    # Rescale long-short only: gross notional = 2
    has_short = (avg_wgt < 0).any(axis=1).any()
    if has_short:
        gross = avg_wgt.abs().sum(axis=1)
        gross = gross.replace(0, np.nan)
        avg_wgt = avg_wgt.mul(2 / gross, axis=0)
        avg_wgt = avg_wgt.fillna(0)

    return avg_wgt
