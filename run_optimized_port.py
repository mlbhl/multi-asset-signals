"""
run_optimized_port.py — Section 6: Optimised Portfolios (MVO)

Uses combined signal weights as expected returns and a shrinkage covariance
matrix to run constrained mean-variance optimisation for target-date
fund benchmarks (T-2020 through T-2055).

Output saved to: results/optimized_port_YYYYMMDD.xlsx
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings('ignore')

from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.config import (
    DEFAULT_CONFIG, RESULTS_DIR, ASSETS,
    BMS, PORTFOLIO_NAMES,
    UP_SCALE, DOWN_SCALE, SPEC_UP_SCALE, SPEC_DOWN_SCALE,
    N_EQUITY,
)
from src.data_loader import load_all
from src.performance import summary_stats, rebase
from src.portfolio import generate_results
from src.optimizer import risk_model, generate_opt_weight


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run optimised portfolio construction')
    p.add_argument('--proxy',       type=str,   default=None,
                   help='Proxy URL, e.g. http://host:port')
    p.add_argument('--h-period',    type=int,   default=1,
                   help='Weight-smoothing holding period (months)')
    p.add_argument('--shrinkage',   type=float, default=0.5,
                   help='Correlation shrinkage factor (0=none, 1=full)')
    p.add_argument('--to-cons',     type=float, default=10.0,
                   help='Max one-way monthly turnover (%)')
    p.add_argument('--start-date',  type=str,   default='2016',
                   help='Chart / stats start date for comparison (YYYY)')
    p.add_argument('--no-chart',    action='store_true',
                   help='Skip chart rendering')
    return p.parse_args()


# ============================================================
# Build per-portfolio constraint spec
# ============================================================
def _build_pf_specs(
    n_assets: int = len(ASSETS),
    to_cons: float = 10.0,
    ar_cons: float | None = None,
    vol_cons: float | None = None,
    te_cons: float | None = None,
) -> dict[str, dict]:
    """
    Build constraint dictionaries for each target-date portfolio.

    Up/down limit arrays are derived by multiplying BM weights by scale factors.
    """
    L         = n_assets
    up_mult   = np.array([SPEC_UP_SCALE.get(i,   UP_SCALE)   for i in range(L)])
    down_mult = np.array([SPEC_DOWN_SCALE.get(i, DOWN_SCALE) for i in range(L)])

    pf_specs: dict[str, dict] = {}
    for pf in PORTFOLIO_NAMES:
        bm_key = 'b' + pf[1:]
        b      = BMS[bm_key]
        u      = b * up_mult
        d      = b * down_mult

        pf_specs[pf] = dict(
            bm       = b,
            min_cons = d,
            max_cons = u,
            mineq    = b[:N_EQUITY].sum() * 0.95,
            maxeq    = min(b[:N_EQUITY].sum() * 1.10, 0.79),
            to_cons  = to_cons,
            ar_cons  = ar_cons,
            vol_cons = vol_cons,
            te_cons  = te_cons,
        )
    return pf_specs


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("Loading data …")
    data = load_all(proxy=args.proxy)
    ret  = data['ret']

    # ------------------------------------------------------------------
    # 2. Signal-based expected returns
    # ------------------------------------------------------------------
    print("Generating signal weights …")
    cfg = DEFAULT_CONFIG.replace(tcost=args.to_cons / 1000)  # approximate
    weights_lo, _, _, _ = generate_results(ret, cfg, 'lo')
    exp_ret = weights_lo['comb'].copy()

    # ------------------------------------------------------------------
    # 3. Risk model
    # ------------------------------------------------------------------
    print("Estimating risk model …")
    vols, corrs, covs = risk_model(ret, shrinkage=args.shrinkage)

    # ------------------------------------------------------------------
    # 4. Per-portfolio optimisation
    # ------------------------------------------------------------------
    pf_specs = _build_pf_specs(
        n_assets=len(ASSETS),
        to_cons=args.to_cons,
    )

    result: dict[str, dict] = {}
    for pf in PORTFOLIO_NAMES:
        print(f"  Optimising {pf} …")
        spec = pf_specs[pf]
        b    = spec['bm']

        bm_wgt = generate_opt_weight(
            exp_ret, vols, corrs, covs,
            strategy='BM',
            h_period=args.h_period,
            bm=b,
        )

        qt_wgt = generate_opt_weight(
            exp_ret, vols, corrs, covs,
            strategy='SIGNAL',
            h_period=args.h_period,
            bm=b,
            min_cons=spec['min_cons'],
            max_cons=spec['max_cons'],
            min_eq_cons=spec['mineq'],
            max_eq_cons=spec['maxeq'],
            n_eq=N_EQUITY,
            ar_cons=spec['ar_cons'],
            vol_cons=spec['vol_cons'],
            to_cons=spec['to_cons'],
            te_cons=spec['te_cons'],
        )
        qt_wgt = qt_wgt.clip(lower=0)   # remove numerically tiny negatives

        from src.portfolio import generate_portfolio
        bm_idx,    _,          _          = generate_portfolio(ret, bm_wgt, tcost=0.0)
        qt_idx,    qt_idx_cost, qt_turn   = generate_portfolio(ret, qt_wgt, tcost=args.to_cons / 10000)

        result[pf] = dict(
            qt_wgt      = qt_wgt,
            bm_wgt      = bm_wgt,
            qt_idx      = qt_idx,
            qt_idx_cost = qt_idx_cost,
            bm_idx      = bm_idx,
            qt_turnover = qt_turn,
        )

    # ------------------------------------------------------------------
    # 5. Performance statistics
    # ------------------------------------------------------------------
    print("Computing performance statistics …")
    key_stats   = ['cagr', 'sharpe', 'mdd']
    stat_rows   = pd.MultiIndex.from_product([PORTFOLIO_NAMES, ['qt', 'bm']])
    stat_table  = pd.DataFrame(index=stat_rows, columns=key_stats + ['avg_turnover'])

    START = args.start_date

    for pf in PORTFOLIO_NAMES:
        port = rebase(
            pd.concat([result[pf]['qt_idx_cost'],
                       result[pf]['bm_idx']], axis=1).rename(columns={0: 'qt', 1: 'bm'}),
            START,
        )
        perf = summary_stats(port, factor_ret=None, bm_price=port['bm'])
        for leg in ('qt', 'bm'):
            stat_table.loc[(pf, leg), key_stats] = perf[leg][key_stats].values
        stat_table.loc[(pf, 'qt'), 'avg_turnover'] = result[pf]['qt_turnover'].mean()

    # ------------------------------------------------------------------
    # 6. Final weights and active tilts
    # ------------------------------------------------------------------
    final_wgts = pd.concat(
        [result[pf]['qt_wgt'].tail(1).rename(index={result[pf]['qt_wgt'].index[-1]: pf})
         for pf in PORTFOLIO_NAMES]
    )
    final_wgts.columns = ASSETS

    bms_df = pd.DataFrame(
        {pf: BMS['b' + pf[1:]] for pf in PORTFOLIO_NAMES},
        index=ASSETS,
    ).T

    active_wgts = final_wgts - bms_df

    # ------------------------------------------------------------------
    # 7. Charts
    # ------------------------------------------------------------------
    if not args.no_chart:
        _plot_portfolios(result, PORTFOLIO_NAMES, START, RESULTS_DIR)

    # ------------------------------------------------------------------
    # 8. Excel export
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"optimized_port_{date.today():%Y%m%d}.xlsx"
    print(f"Writing {out_path} …")

    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        stat_table.to_excel(writer,  sheet_name='perf_stats')
        final_wgts.to_excel(writer,  sheet_name='final_weights')
        active_wgts.to_excel(writer, sheet_name='active_weights')
        bms_df.to_excel(writer,      sheet_name='bm_weights')

        # One sheet per portfolio: full weight history
        for pf in PORTFOLIO_NAMES:
            result[pf]['qt_wgt'].to_excel(writer, sheet_name=f'wgt_{pf}')

        # Latest active tilt vs last month return
        latest_tilt = pd.concat(
            [active_wgts.loc[PORTFOLIO_NAMES[1]].rename('active_wgt'),
             ret.iloc[-1].rename('last_ret')],
            axis=1,
        )
        latest_tilt.to_excel(writer, sheet_name='latest_tilt')

    print("Done.")


# ============================================================
# Chart helper
# ============================================================
def _plot_portfolios(
    result: dict,
    pf_names: list[str],
    start: str,
    out_dir,
) -> None:
    from src.performance import rebase

    n = len(pf_names)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows), sharex=True)
    axs = axs.flatten()

    for i, pf in enumerate(pf_names):
        port = rebase(
            pd.concat([result[pf]['qt_idx_cost'],
                       result[pf]['bm_idx']], axis=1).rename(columns={0: 'qt', 1: 'bm'}),
            start,
        )
        port.plot(ax=axs[i], title=pf, linewidth=1)
        axs[i].legend(fontsize=8)

    for j in range(len(pf_names), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    fig.savefig(out_dir / f"chart_opt_port_{date.today():%Y%m%d}.png", dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
