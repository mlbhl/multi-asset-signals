"""
run_driving_forces.py — Section 5: Driving Forces (Long-short portfolios)

Tests:
  5.1  Market states        (Cooper, Gutierrez & Hameed 2004)
  5.2a Volatility states    (Wang & Xu 2015)
  5.2b Business cycle       (Petkova & Zhang 2005)
       — 4-state regression (peak / expansion / recession / trough)
       — trough vs. non-trough

Output saved to: results/driving_forces_YYYYMMDD.xlsx
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings('ignore')

from datetime import date

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.config import DEFAULT_CONFIG, RESULTS_DIR
from src.data_loader import load_all, build_emrp
from src.portfolio import generate_results


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Run driving forces analysis')
    p.add_argument('--proxy', type=str, default=None,
                   help='Proxy URL, e.g. http://host:port')
    return p.parse_args()


# ============================================================
# Section 5.1 — Market states
# ============================================================
def _test_market_states(
    portfolios: dict[str, pd.Series],
    mkt: pd.DataFrame,
    signals: list[str],
) -> pd.DataFrame:
    """
    OLS of signal returns on market-up and market-down state dummies
    (no intercept, so coefficients are conditional means).

    Cooper, Gutierrez & Hameed (2004).
    """
    pf_df  = pd.DataFrame(portfolios)
    joined = pf_df.pct_change().join(mkt)
    results: dict[str, pd.DataFrame] = {}

    for sig in signals:
        sample = joined[[sig, 'up_state', 'down_state']].dropna()
        y = sample[sig]
        X = sample[['up_state', 'down_state']]
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        m0, m1 = res.params
        se0, se1 = res.bse
        t0, t1 = res.tvalues
        p0, p1 = res.pvalues

        m2 = m0 - m1
        t2 = m2 / np.sqrt(se0 ** 2 + se1 ** 2)
        p2 = 2 * (1 - stats.t.cdf(abs(t2), df=res.df_resid - 2))

        results[sig] = pd.DataFrame(
            [[m0 * 12, m1 * 12, m2 * 12], [t0, t1, t2], [p0, p1, p2]],
            index=['coef', 'tval', 'pval'],
            columns=['up_state', 'down_state', 'diff'],
        )

    return pd.concat(results)


# ============================================================
# Section 5.2a — Volatility states
# ============================================================
def _test_vol_states(
    portfolios: dict[str, pd.Series],
    vol: pd.DataFrame,
    signals: list[str],
) -> pd.DataFrame:
    """
    OLS of signal returns on high-vol and low-vol state dummies.

    Wang & Xu (2015).
    """
    pf_df  = pd.DataFrame(portfolios)
    joined = pf_df.pct_change().join(vol)
    results: dict[str, pd.DataFrame] = {}

    for sig in signals:
        sample = joined[[sig, 'hivol_state', 'lovol_state']].dropna()
        y = sample[sig]
        X = sample[['hivol_state', 'lovol_state']]
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        m0, m1 = res.params
        se0, se1 = res.bse
        t0, t1 = res.tvalues
        p0, p1 = res.pvalues

        m2 = m0 - m1
        t2 = m2 / np.sqrt(se0 ** 2 + se1 ** 2)
        p2 = 2 * (1 - stats.t.cdf(abs(t2), df=res.df_resid - 2))

        results[sig] = pd.DataFrame(
            [[m0 * 12, m1 * 12, m2 * 12], [t0, t1, t2], [p0, p1, p2]],
            index=['coef', 'tval', 'pval'],
            columns=['hivol_state', 'lovol_state', 'diff'],
        )

    return pd.concat(results)


# ============================================================
# Section 5.2b — Business cycle: 4-state regression
# ============================================================
def _test_macro_4state(
    portfolios: dict[str, pd.Series],
    emrp: pd.DataFrame,
    signals: list[str],
) -> pd.DataFrame:
    """
    OLS of signal returns on 4 business-cycle state dummies
    (peak / expansion / recession / trough).

    Petkova & Zhang (2005).
    Returns full statsmodels summary text concatenated into one DataFrame.
    """
    pf_df  = pd.DataFrame(portfolios)
    states = ['peak', 'expansion', 'recession', 'trough']
    joined = pf_df.pct_change().join(emrp[states]).dropna(how='all')

    results: dict[str, pd.DataFrame] = {}
    for sig in signals:
        sample = joined[[sig] + states].dropna()
        y = sample[sig]
        X = sample[states]
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        results[sig] = pd.DataFrame(
            [res.params * 12, res.tvalues, res.pvalues],
            index=['coef (ann.)', 'tval', 'pval'],
            columns=states,
        )

    return pd.concat(results)


# ============================================================
# Section 5.2b — Business cycle: trough vs. non-trough
# ============================================================
def _test_macro_trough(
    portfolios: dict[str, pd.Series],
    emrp: pd.DataFrame,
    signals: list[str],
) -> pd.DataFrame:
    """
    OLS of signal returns on trough and non-trough state dummies.
    Reports coefficient difference and its t-statistic.
    """
    pf_df  = pd.DataFrame(portfolios)
    states = ['trough', 'non-trough']
    joined = pf_df.pct_change().join(emrp[states]).dropna(how='all')

    results: dict[str, pd.DataFrame] = {}
    for sig in signals:
        sample = joined[[sig] + states].dropna()
        y = sample[sig]
        X = sample[states]
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

        m0, m1 = res.params
        se0, se1 = res.bse
        t0, t1 = res.tvalues
        p0, p1 = res.pvalues

        m2 = m0 - m1
        t2 = m2 / np.sqrt(se0 ** 2 + se1 ** 2)
        p2 = 2 * (1 - stats.t.cdf(abs(t2), df=res.df_resid - 2))

        results[sig] = pd.DataFrame(
            [[m0 * 12, m1 * 12, m2 * 12], [t0, t1, t2], [p0, p1, p2]],
            index=['coef (ann.)', 'tval', 'pval'],
            columns=['trough', 'non-trough', 'diff'],
        )

    return pd.concat(results)


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()

    print("Loading data …")
    data = load_all(proxy=args.proxy)

    ret         = data['ret']
    mkt         = data['mkt']
    vol         = data['vol']
    macrofactor = data['macrofactor']
    fivefactor  = data['fivefactor']

    print("Generating portfolios …")
    cfg = DEFAULT_CONFIG.replace(tcost=0.0)
    _, _, portfolios_ls, _ = generate_results(ret, cfg, 'ls')

    signals = ['mom', 'rev', 'seas', 'ftrd', 'strd', 'comb']

    # EMRP for business cycle analysis
    print("Estimating EMRP …")
    emrp, emrp_model = build_emrp(macrofactor, fivefactor)

    # Business cycle state distribution
    cycles = ['peak', 'expansion', 'recession', 'trough']
    emrp_cycle = pd.DataFrame(index=['mean (ann.)', 'count'], columns=cycles)
    for cyc in cycles:
        grp = emrp.groupby(cyc)['EMRP']
        emrp_cycle[cyc] = [grp.mean()[1] * 12, grp.count()[1]]

    print("5.1 Market states …")
    mkt_state_df = _test_market_states(portfolios_ls, mkt, signals)

    print("5.2a Volatility states …")
    vol_state_df = _test_vol_states(portfolios_ls, vol, signals)

    print("5.2b Business cycle …")
    macro_4state_df = _test_macro_4state(portfolios_ls, emrp, signals)
    macro_trough_df = _test_macro_trough(portfolios_ls, emrp, signals)

    # EMRP model summary as a text-based DataFrame
    emrp_summary_text = str(emrp_model.summary())

    # ------------------------------------------------------------------
    # Excel export
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"driving_forces_{date.today():%Y%m%d}.xlsx"
    print(f"Writing {out_path} …")

    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        mkt_state_df.to_excel(writer,   sheet_name='mkt_state')
        vol_state_df.to_excel(writer,   sheet_name='vol_state')
        macro_4state_df.to_excel(writer,sheet_name='macro_4state')
        macro_trough_df.to_excel(writer,sheet_name='macro_trough')
        emrp_cycle.to_excel(writer,     sheet_name='emrp_cycle')

        # EMRP regression model summary
        # write_string() prevents lines starting with '=' being parsed as formulas
        ws = writer.book.add_worksheet('emrp_model')
        writer.sheets['emrp_model'] = ws
        for i, line in enumerate(emrp_summary_text.split('\n')):
            ws.write_string(i, 0, line)

    print("Done.")


if __name__ == '__main__':
    main()
