"""
run_robustness.py — Section 4: Robustness Checks (Long-only portfolios)

Tests:
  4.1  Model specifications  (ranking / holding / weighting)
  4.2  Asset universes       (random sub-samples of assets)
  4.3  Subsamples            (decade-level sub-periods)
  4.4  Transaction costs     (varying one-way cost)

Output saved to: results/robustness_YYYYMMDD.xlsx
"""
from __future__ import annotations

import argparse
import os
import random
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from datetime import date
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, ASSETS, RESULTS_DIR
from src.data_loader import load_all
from src.performance import summary_stats
from src.portfolio import generate_results


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run robustness checks')
    p.add_argument('--proxy',  type=str,  default=None,
                   help='Proxy URL, e.g. http://host:port')
    p.add_argument('--source', type=str,  default='excel',
                   choices=['excel', 'yfinance', 'alphavantage', 'eodhd'],
                   help='Data source: excel (default), yfinance, alphavantage, or eodhd')
    p.add_argument('--av-api-key', type=str,
                   default=os.environ.get('ALPHA_VANTAGE_API_KEY'),
                   help='Alpha Vantage API key (default: from .env)')
    p.add_argument('--eodhd-api-key', type=str,
                   default=os.environ.get('EODHD_API_KEY'),
                   help='EODHD API key (default: from .env)')
    p.add_argument('--seed',   type=int,  default=42,
                   help='Random seed for universe sampling')
    p.add_argument('--n-universe-samples', type=int, default=20,
                   help='Number of random universe sub-samples')
    p.add_argument('--n-sampled-assets',   type=int, default=10,
                   help='Assets per random sub-sample')
    return p.parse_args()


# ============================================================
# Section 4.1a — Individual signal robustness
# ============================================================
def _test_spec_individual(
    ret: pd.DataFrame,
    sevenfactor: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Vary ranking period and holding period for each signal individually.
    All other parameters are held at DEFAULT_CONFIG values.
    """
    STAT_ROWS = ['nsamples', 'ir', 'alpha', 'alpha-t-stat']
    cfg = DEFAULT_CONFIG.replace(tcost=0.0)

    grid: dict[str, tuple[list, list]] = {
        'mom':  ([3, 6, 9],            [1, 6, 12, 24, 60]),
        'rev':  ([36, 60, 120],        [1, 6, 12, 24, 60]),
        'seas': ([120, 180, 240],      [1, 6, 12, 24, 60]),
        'ftrd': ([1, 2, 3],            [1, 6, 12, 24, 60]),
        'strd': ([10, 11, 12],         [1, 6, 12, 24, 60]),
    }
    # Map signal name → index in generate_signal return tuple
    sig_idx = {'mom': 0, 'rev': 1, 'seas': 2, 'ftrd': 3, 'strd': 4}

    from src.signals import (
        generate_signal, generate_cs_weight, generate_ts_weight
    )
    from src.portfolio import generate_portfolio

    results: dict[str, pd.DataFrame] = {}

    for sig_name, (ranks, holds) in grid.items():
        idx  = sig_idx[sig_name]
        stype = 'cs' if sig_name in ('mom', 'rev', 'seas') else 'ts'

        table = pd.DataFrame(
            index=pd.MultiIndex.from_product([ranks, STAT_ROWS]),
            columns=holds,
            dtype=float,
        )

        for r, h in product(ranks, holds):
            # Generate signal with one parameter varied, others at default
            kw = {f'{sig_name}p': r}
            sig = generate_signal(ret, **kw)[idx]

            if stype == 'cs':
                _, lo_wgt = generate_cs_weight(
                    sig, method=cfg.method, quantile=cfg.quantile, holding=h
                )
            else:
                _, lo_wgt = generate_ts_weight(sig, holding=h)

            pf, _, _ = generate_portfolio(ret, lo_wgt, tcost=0.0)

            # Equal-weight benchmark
            bm_wgt = ret.loc[lo_wgt.index].apply(
                lambda row: row.where(row.isna(), 1 / row.notna().sum()).fillna(0),
                axis=1,
            )
            bm_pf, _, _ = generate_portfolio(ret, bm_wgt, tcost=0.0)

            stat = summary_stats(
                pd.DataFrame(pf).iloc[:-2],
                factor_ret=sevenfactor,
                bm_price=bm_pf,
            ).loc[STAT_ROWS]

            for row_name in STAT_ROWS:
                table.loc[(r, row_name), h] = float(stat.loc[row_name].iloc[0])

        results[sig_name] = table

    return results


# ============================================================
# Section 4.1b — Combined signal: holding period robustness
# ============================================================
def _test_spec_comb(
    ret: pd.DataFrame,
    sevenfactor: pd.DataFrame,
) -> pd.DataFrame:
    STAT_ROWS    = ['nsamples', 'ir', 'alpha', 'alpha-t-stat']
    holding_combs = [1, 2, 3, 6, 12, 24, 60]

    table = pd.DataFrame(index=STAT_ROWS, columns=holding_combs, dtype=float)

    for h in holding_combs:
        cfg = DEFAULT_CONFIG.replace(holding_comb=h, tcost=0.0)
        weights_lo, _, portfolios_lo, _ = generate_results(ret, cfg, 'lo')

        stat = summary_stats(
            pd.DataFrame(portfolios_lo['comb']).iloc[:-2],
            factor_ret=sevenfactor,
            bm_price=portfolios_lo['bm'],
        ).loc[STAT_ROWS]

        table[h] = stat.values.ravel()

    return table


# ============================================================
# Section 4.1c — Weighting / grouping methods
# ============================================================
def _test_method(
    ret: pd.DataFrame,
    sevenfactor: pd.DataFrame,
) -> pd.DataFrame:
    STAT_ROWS = ['nsamples', 'ir', 'alpha', 'alpha-t-stat']
    methods   = ['quantile', 'proportion', 'rank']
    quantiles = ['median', 'tercile', 'quartile']
    rows      = []

    for method, quant in product(methods, quantiles):
        # proportion / rank don't use quantile grouping → test only once
        if method in ('proportion', 'rank') and quant in ('tercile', 'quartile'):
            continue

        cfg = DEFAULT_CONFIG.replace(tcost=0.0, method=method, quantile=quant)
        weights_lo, _, portfolios_lo, _ = generate_results(ret, cfg, 'lo')

        label = method if method in ('proportion', 'rank') else f'{method}-{quant}'
        stat  = summary_stats(
            pd.DataFrame(portfolios_lo['comb']).iloc[:-2],
            factor_ret=sevenfactor,
            bm_price=portfolios_lo['bm'],
        ).loc[STAT_ROWS]

        rows.append(pd.DataFrame(stat.values.T, index=[label], columns=STAT_ROWS))

    return pd.concat(rows)


# ============================================================
# Section 4.2 — Asset universe robustness
# ============================================================
def _test_universe(
    ret: pd.DataFrame,
    sevenfactor: pd.DataFrame,
    n_samples: int,
    n_assets: int,
    seed: int,
) -> pd.DataFrame:
    random.seed(seed)
    STAT_ROWS = ['ir', 'alpha', 'alpha-t-stat']
    cfg       = DEFAULT_CONFIG.replace(tcost=0.0)

    results: dict[int, pd.DataFrame] = {}
    for i in range(n_samples):
        sampled     = random.sample(ASSETS, n_assets)
        sampled_ret = ret[sampled]
        _, _, portfolios_lo, _ = generate_results(sampled_ret, cfg, 'lo')

        stat = summary_stats(
            pd.DataFrame(portfolios_lo).iloc[:-2].drop(columns=['bm']),
            factor_ret=sevenfactor,
            bm_price=portfolios_lo['bm'],
        ).loc[STAT_ROWS]
        results[i] = stat

    return pd.concat(results)


# ============================================================
# Section 4.3 — Subsamples (decade-level)
# ============================================================
def _test_subsample(
    ret: pd.DataFrame,
    sevenfactor: pd.DataFrame,
) -> pd.DataFrame:
    cfg        = DEFAULT_CONFIG.replace(tcost=0.0)
    _, _, portfolios_lo, _ = generate_results(ret, cfg, 'lo')

    STAT_ROWS   = ['ir', 'alpha', 'alpha-t-stat']
    start_years = [1970, 1980, 1990, 2000, 2010, 2020]
    interval    = 10
    results: dict[str, pd.DataFrame] = {}

    pf_df = pd.DataFrame(portfolios_lo)

    for yr in start_years:
        sub = pf_df.loc[f'{yr - 1}-12-01': f'{yr + interval - 1}-12-31']
        stat = summary_stats(
            sub.iloc[:-2].drop(columns=['bm']),
            factor_ret=sevenfactor,
            bm_price=sub['bm'],
        ).loc[STAT_ROWS]
        results[str(yr)] = stat

    return pd.concat(results)


# ============================================================
# Section 4.4 — Transaction costs
# ============================================================
def _test_tcost(
    ret: pd.DataFrame,
    sevenfactor: pd.DataFrame,
) -> pd.DataFrame:
    STAT_ROWS = ['ir', 'alpha', 'alpha-t-stat']
    tcosts    = [0.000, 0.001, 0.002, 0.003, 0.004, 0.005]
    results   = []

    for tc in tcosts:
        cfg = DEFAULT_CONFIG.replace(tcost=tc)
        weights_lo, turnovers_lo, _, portfolios_cost_lo = generate_results(ret, cfg, 'lo')

        pf_cost = pd.DataFrame(portfolios_cost_lo)
        turns   = pd.DataFrame(turnovers_lo)

        stat = summary_stats(
            pf_cost.iloc[:-2].drop(columns=['bm']),
            factor_ret=sevenfactor,
            bm_price=pf_cost['bm'],
        ).loc[STAT_ROWS]
        stat.loc['avg_turnover'] = turns.iloc[:-2].drop(columns=['bm']).mean()
        stat.loc['nsamples']     = (
            summary_stats(pf_cost.iloc[:-2].drop(columns=['bm'])).loc['nsamples']
        )
        results.append((f'{tc * 100:.1f}%', stat))

    # Stack all tcost frames; append nsamples and avg_turnover at bottom
    combined = pd.concat(
        {label: df for label, df in results}
    )
    return combined


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()

    print("Loading data …")
    data        = load_all(proxy=args.proxy, source=args.source,
                           av_api_key=getattr(args, 'av_api_key', None),
                           eodhd_api_key=getattr(args, 'eodhd_api_key', None))
    ret         = data['ret']
    sevenfactor = data['sevenfactor']

    print("4.1 Model specification tests …")
    spec_individual = _test_spec_individual(ret, sevenfactor)
    spec_comb       = _test_spec_comb(ret, sevenfactor)
    test_method_df  = _test_method(ret, sevenfactor)

    print("4.2 Asset universe tests …")
    universe_df = _test_universe(
        ret, sevenfactor,
        n_samples=args.n_universe_samples,
        n_assets=args.n_sampled_assets,
        seed=args.seed,
    )

    print("4.3 Subsample tests …")
    subsample_df = _test_subsample(ret, sevenfactor)

    print("4.4 Transaction cost tests …")
    tcost_df = _test_tcost(ret, sevenfactor)

    # ------------------------------------------------------------------
    # Excel export
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"robustness_{date.today():%Y%m%d}.xlsx"
    print(f"Writing {out_path} …")

    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        for sig_name, df in spec_individual.items():
            df.to_excel(writer, sheet_name=f'spec_{sig_name}')

        spec_comb.to_excel(writer,      sheet_name='spec_comb')
        test_method_df.to_excel(writer, sheet_name='spec_method')
        universe_df.to_excel(writer,    sheet_name='universe')
        subsample_df.to_excel(writer,   sheet_name='subsample')
        tcost_df.to_excel(writer,       sheet_name='tcost')

    print("Done.")


if __name__ == '__main__':
    main()
