"""
run_main_test.py — Section 3: Main Results

Generates signal-based portfolios and performance statistics.
Output saved to: results/main_test_YYYYMMDD.xlsx
"""
from __future__ import annotations

import argparse
import os
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
from scipy.linalg import sqrtm
import cvxpy as cp

from src.config import DEFAULT_CONFIG, RESULTS_DIR, ASSET_TITLES
from src.data_loader import load_all
from src.performance import summary_stats, factor_reg, rebase
from src.portfolio import generate_results, test_signal


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run main signal-portfolio test')
    p.add_argument('--proxy',   type=str,  default=None,
                   help='Proxy URL, e.g. http://host:port')
    p.add_argument('--source',  type=str,  default='excel',
                   choices=['excel', 'yfinance', 'alphavantage', 'eodhd'],
                   help='Data source: excel (default), yfinance, alphavantage, or eodhd')
    p.add_argument('--av-api-key', type=str,
                   default=os.environ.get('ALPHA_VANTAGE_API_KEY'),
                   help='Alpha Vantage API key (default: from .env)')
    p.add_argument('--eodhd-api-key', type=str,
                   default=os.environ.get('EODHD_API_KEY'),
                   help='EODHD API key (default: from .env)')
    p.add_argument('--no-chart', action='store_true',
                   help='Skip chart rendering')
    return p.parse_args()


# ============================================================
# Excel export helper
# ============================================================
def _fmt_float(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame) -> None:
    """Apply 4dp float format to a sheet."""
    ws = writer.sheets[sheet]
    fmt = writer.book.add_format({'num_format': '0.0000'})
    for col_num in range(1, df.shape[1] + 1):
        ws.set_column(col_num, col_num, 14, fmt)


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("Loading data …")
    data = load_all(proxy=args.proxy, source=args.source,
                    av_api_key=getattr(args, 'av_api_key', None),
                    eodhd_api_key=getattr(args, 'eodhd_api_key', None))

    ret        = data['ret']
    price      = data['price']
    sevenfactor= data['sevenfactor']

    # ------------------------------------------------------------------
    # 2. Asset statistics
    # ------------------------------------------------------------------
    print("Computing asset statistics …")
    asset_stats = summary_stats(price)

    # ------------------------------------------------------------------
    # 3. Generate main results
    # ------------------------------------------------------------------
    print("Generating portfolios …")
    config = DEFAULT_CONFIG

    (weights_ls, turnovers_ls, portfolios_ls, portfolios_cost_ls,
     weights_lo, turnovers_lo, portfolios_lo, portfolios_cost_lo) = generate_results(ret, config)

    picks_ls = pd.DataFrame({k: portfolios_cost_ls[k]
                             for k in ['mom', 'rev', 'seas', 'ftrd', 'strd', 'comb']})
    picks_lo = pd.DataFrame({k: portfolios_cost_lo[k]
                             for k in ['mom', 'rev', 'seas', 'ftrd', 'strd', 'comb', 'bm']})

    # Exclude last 5 rows (data delay / incomplete factor data)
    TRIM = 5

    # ------------------------------------------------------------------
    # 4. Signal efficacy
    # ------------------------------------------------------------------
    print("Computing signal efficacy …")
    signal_efficacy = pd.DataFrame()
    for name, wgt in weights_ls.items():
        tester = test_signal(ret, wgt).mean(axis=1)
        tester = pd.concat([tester, pd.Series({'nsamples': len(wgt) - 1})])
        tester.name = name
        signal_efficacy = pd.concat([signal_efficacy, tester], axis=1)

    # ------------------------------------------------------------------
    # 5. Signal correlations
    # ------------------------------------------------------------------
    signal_corr = pd.DataFrame(portfolios_ls).pct_change().corr()

    # ------------------------------------------------------------------
    # 6. Seasonality
    # ------------------------------------------------------------------
    signal_season = pd.DataFrame()
    for name, pf in portfolios_ls.items():
        s = pf.pct_change().dropna().groupby(pf.pct_change().dropna().index.month).mean()
        s.name = name
        signal_season = pd.concat([signal_season, s], axis=1)
    signal_season = pd.concat(
        [signal_season, pd.DataFrame(signal_season.std(), columns=['stdev']).T]
    )
    signal_season.index.name = 'month'

    # ------------------------------------------------------------------
    # 7. Performance statistics
    # ------------------------------------------------------------------
    print("Computing performance statistics …")
    pf_cost_ls = pd.DataFrame(portfolios_cost_ls)
    pf_cost_lo = pd.DataFrame(portfolios_cost_lo)
    turn_ls    = pd.DataFrame(turnovers_ls)
    turn_lo    = pd.DataFrame(turnovers_lo)
    bm_series  = pf_cost_lo['bm']

    def _stats_ls(df: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([
            summary_stats(df.iloc[:-TRIM], factor_ret=sevenfactor),
            pd.DataFrame(turns.iloc[:-TRIM].mean(), columns=['avg_turnover']).T,
        ])

    def _stats_lo(df: pd.DataFrame, turns: pd.DataFrame, bm: pd.Series) -> pd.DataFrame:
        return pd.concat([
            summary_stats(df.iloc[:-TRIM], factor_ret=sevenfactor, bm_price=bm),
            pd.DataFrame(turns.iloc[:-TRIM].mean(), columns=['avg_turnover']).T,
        ])

    stats_ls_full    = _stats_ls(pf_cost_ls,         turn_ls)
    stats_ls_same    = _stats_ls(pf_cost_ls.dropna(), turn_ls.dropna())
    stats_ls_postgfc = _stats_ls(pf_cost_ls.dropna()['2011':], turn_ls.dropna()['2011':])

    stats_lo_full    = _stats_lo(pf_cost_lo,          turn_lo, bm_series)
    stats_lo_same    = _stats_lo(pf_cost_lo.dropna(),  turn_lo.dropna(), bm_series)
    stats_lo_postgfc = _stats_lo(pf_cost_lo.dropna()['2011':], turn_lo.dropna()['2011':], bm_series)

    # ------------------------------------------------------------------
    # 8. Factor regression (7-factor model)
    # ------------------------------------------------------------------
    print("Running factor regressions …")
    FF7 = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'TERM', 'DEF']
    factor_cols = ['alpha'] + FF7

    def _ff7_table(pf_df: pd.DataFrame) -> pd.DataFrame:
        results = {}
        joined  = pf_df.pct_change().join(sevenfactor)
        for col in pf_df.columns:
            y = joined[col].dropna()
            X = joined[FF7].loc[y.index].dropna()
            y = y.loc[X.index]
            coef, tval, pval = factor_reg(X, y)
            results[col] = pd.DataFrame(
                [coef, tval, pval],
                index=['coef', 'tval', 'pval'],
                columns=factor_cols,
            )
        frames = [df.unstack() for df in results.values()]
        return pd.concat(frames, axis=1, keys=pf_df.columns)

    ff7_ls = _ff7_table(pf_cost_ls)
    ff7_lo = _ff7_table(pf_cost_lo)

    # ------------------------------------------------------------------
    # 9. Charts (optional)
    # ------------------------------------------------------------------
    if not args.no_chart:
        print("Rendering charts …")
        _plot_main(picks_ls, picks_lo, RESULTS_DIR)

    # ------------------------------------------------------------------
    # 10. Excel export
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"main_test_{date.today():%Y%m%d}.xlsx"
    print(f"Writing {out_path} …")

    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        def _write(df: pd.DataFrame, sheet: str) -> None:
            df.to_excel(writer, sheet_name=sheet)

        _write(asset_stats,       'asset_stats')
        _write(signal_efficacy,   'signal_efficacy')
        _write(signal_corr,       'signal_corr')
        _write(signal_season,     'signal_season')
        _write(stats_ls_full,     'stats_ls_full')
        _write(stats_ls_same,     'stats_ls_same')
        _write(stats_ls_postgfc,  'stats_ls_postgfc')
        _write(stats_lo_full,     'stats_lo_full')
        _write(stats_lo_same,     'stats_lo_same')
        _write(stats_lo_postgfc,  'stats_lo_postgfc')
        _write(ff7_ls,            'ff7_ls')
        _write(ff7_lo,            'ff7_lo')
        def _latest_weights(wgt_dict):
            return pd.concat(
                [v.iloc[-1:].rename(index={v.index[-1]: k}) for k, v in wgt_dict.items()]
            )
        _write(_latest_weights(weights_ls), 'weights_ls_latest')
        _write(_latest_weights(weights_lo), 'weights_lo_latest')
        _write(picks_ls,          'picks_ls')
        _write(picks_lo,          'picks_lo')
        _write(weights_ls['comb'], 'weights_ls_comb_hist')
        _write(weights_lo['comb'], 'weights_lo_comb_hist')

    print("Done.")


# ============================================================
# Chart helpers
# ============================================================
def _plot_main(picks_ls: pd.DataFrame, picks_lo: pd.DataFrame, out_dir: Path) -> None:
    custom_start = str(picks_ls.index[-1].year - 3)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7), sharex='col')

    ax1.plot(rebase(picks_lo), linewidth=1)
    ax1.set_title('Long-only (since inception)', fontsize=10)
    ax1.legend(picks_lo.columns, loc='upper left', fontsize=8)

    ax2.plot(rebase(picks_lo, custom_start), linewidth=1)
    ax2.set_title(f'Long-only (since {custom_start})', fontsize=10)
    ax2.legend(picks_lo.columns, loc='lower left', fontsize=8)

    ax3.plot(rebase(picks_ls), linewidth=1)
    ax3.set_title('Long-short (since inception)', fontsize=10)
    ax3.legend(picks_ls.columns, loc='upper left', fontsize=8)
    ax3.xaxis.set_major_locator(mdates.YearLocator(base=10))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax4.plot(rebase(picks_ls, custom_start), linewidth=1)
    ax4.set_title(f'Long-short (since {custom_start})', fontsize=10)
    ax4.legend(picks_ls.columns, loc='upper left', fontsize=8)
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    fig.savefig(out_dir / f"chart_cumulative_{date.today():%Y%m%d}.png", dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
