"""
portfolio.py — Portfolio construction and main results generator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelConfig, DEFAULT_CONFIG
from .signals import (
    generate_signal,
    generate_cs_weight,
    generate_ts_weight,
    generate_average_weight,
)


# ---------------------------------------------------------------------------
# Single portfolio
# ---------------------------------------------------------------------------
def generate_portfolio(
    ret: pd.DataFrame,
    weight: pd.DataFrame,
    tcost: float = 0.000,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Construct a portfolio price index with and without transaction costs.

    Parameters
    ----------
    ret    : monthly excess return DataFrame
    weight : monthly portfolio weights for the *next* month
             (shift() is applied internally to align t→t+1)
    tcost  : one-way transaction cost rate

    Returns
    -------
    portfolio      : cumulative return index (gross, base 1000)
    portfolio_cost : cumulative return index (net of tcost, base 1000)
    turnover       : one-way monthly turnover
    """
    ret_    = ret.loc[weight.index]

    # Gross portfolio return (NaN weights treated as zero via sum)
    port_ret = (ret_ * weight.shift()).iloc[0:].sum(axis=1)
    portfolio = np.cumprod(1 + port_ret) * 1000

    # Turnover: |Δweight| / 2  (one-way)
    drift    = weight.shift() * (1 + ret_)
    port_val = (1 + port_ret)
    turnover = (weight - drift.div(port_val, axis=0)).abs().sum(axis=1).div(2)

    # Net-of-cost return (costs paid in the *prior* period)
    port_ret_net = port_ret - turnover.shift().fillna(0) * tcost * 2
    portfolio_cost = np.cumprod(1 + port_ret_net) * 1000

    return portfolio, portfolio_cost, turnover


# ---------------------------------------------------------------------------
# Main results generator
# ---------------------------------------------------------------------------
def generate_results(
    ret: pd.DataFrame,
    config: ModelConfig | None = None,
    output: str = 'both',
) -> tuple:
    """
    Generate signals, weights, and portfolios for all five strategies plus
    their combination and an equal-weight benchmark.

    Parameters
    ----------
    ret    : monthly excess return DataFrame
    config : ModelConfig (defaults to DEFAULT_CONFIG)
    output : 'both' | 'ls' | 'lo'

    Returns
    -------
    output == 'both' →
        weights_ls, turnovers_ls, portfolios_ls, portfolios_cost_ls,
        weights_lo, turnovers_lo, portfolios_lo, portfolios_cost_lo

    output == 'ls'   → weights_ls, turnovers_ls, portfolios_ls, portfolios_cost_ls
    output == 'lo'   → weights_lo, turnovers_lo, portfolios_lo, portfolios_cost_lo
    """
    if config is None:
        config = DEFAULT_CONFIG

    # --- 1. Signals ---
    mom, rev, seas, ftrd, strd = generate_signal(
        ret,
        momp=config.momp,   momk=config.momk,
        revp=config.revp,   revk=config.revk,
        seasp=config.seasp,
        ftrdp=config.ftrdp,
        strdp=config.strdp,
    )

    # --- 2. Individual weights ---
    _signals  = {'mom': mom, 'rev': rev, 'seas': seas, 'ftrd': ftrd, 'strd': strd}
    _types    = {'mom': 'cs', 'rev': 'cs', 'seas': 'cs', 'ftrd': 'ts', 'strd': 'ts'}
    _holdings = {
        'mom':  config.holding_mom,
        'rev':  config.holding_rev,
        'seas': config.holding_seas,
        'ftrd': config.holding_ftrd,
        'strd': config.holding_strd,
    }

    weights_ls: dict[str, pd.DataFrame] = {}
    weights_lo: dict[str, pd.DataFrame] = {}

    for name, sig in _signals.items():
        if _types[name] == 'cs':
            wls, wlo = generate_cs_weight(
                sig,
                method=config.method,
                quantile=config.quantile,
                holding=_holdings[name],
            )
        else:
            wls, wlo = generate_ts_weight(sig, holding=_holdings[name])
        weights_ls[name] = wls
        weights_lo[name] = wlo

    # --- 3. Combined weights ---
    comb_ls = generate_average_weight(
        list(weights_ls.values()), holding=config.holding_comb
    )
    comb_lo = generate_average_weight(
        list(weights_lo.values()), holding=config.holding_comb
    )
    weights_ls['comb'] = comb_ls
    weights_lo['comb'] = comb_lo

    # --- 4. Equal-weight benchmark (aligned to ftrd start) ---
    bm_lo = ret.loc[weights_lo['ftrd'].index].apply(
        lambda row: row.where(row.isna(), 1 / row.notna().sum()).fillna(0), axis=1
    )
    weights_lo['bm'] = bm_lo

    # --- 5. Portfolios ---
    need_ls = output in ('both', 'ls')
    need_lo = output in ('both', 'lo')

    turnovers_ls:       dict[str, pd.Series] = {}
    portfolios_ls:      dict[str, pd.Series] = {}
    portfolios_cost_ls: dict[str, pd.Series] = {}

    turnovers_lo:       dict[str, pd.Series] = {}
    portfolios_lo:      dict[str, pd.Series] = {}
    portfolios_cost_lo: dict[str, pd.Series] = {}

    if need_ls:
        for name, wgt in weights_ls.items():
            pf, pfc, turn = generate_portfolio(ret, wgt, tcost=config.tcost)
            portfolios_ls[name]      = pf
            portfolios_cost_ls[name] = pfc
            turnovers_ls[name]       = turn

    if need_lo:
        for name, wgt in weights_lo.items():
            pf, pfc, turn = generate_portfolio(ret, wgt, tcost=config.tcost)
            portfolios_lo[name]      = pf
            portfolios_cost_lo[name] = pfc
            turnovers_lo[name]       = turn

    if output == 'both':
        return (
            weights_ls, turnovers_ls, portfolios_ls, portfolios_cost_ls,
            weights_lo, turnovers_lo, portfolios_lo, portfolios_cost_lo,
        )
    elif output == 'ls':
        return weights_ls, turnovers_ls, portfolios_ls, portfolios_cost_ls
    else:  # 'lo'
        return weights_lo, turnovers_lo, portfolios_lo, portfolios_cost_lo


# ---------------------------------------------------------------------------
# Signal efficacy test
# ---------------------------------------------------------------------------
def test_signal(
    ret: pd.DataFrame,
    weight: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute average returns conditional on long (+) and short (-) weight signs.

    Parameters
    ----------
    ret    : monthly excess return DataFrame
    weight : monthly long-short portfolio weight DataFrame

    Returns
    -------
    DataFrame with rows ['short', 'long'] and columns = asset tickers
    """
    sign_df = weight.map(
        lambda x: 1 if x > 0 else (-1 if x < 0 else np.nan)
    )
    sign_df.columns = [f'{c}_sign' for c in sign_df.columns]
    combined = sign_df.shift()[1:].join(ret)

    result = pd.DataFrame(index=['short', 'long'])
    for asset in ret.columns:
        grouped = combined.groupby(f'{asset}_sign')[asset].mean()
        result[asset] = [grouped.get(-1, np.nan), grouped.get(1, np.nan)]

    return result
