"""
data_loader.py — Load and preprocess all datasets.

Proxy usage
-----------
Pass a proxy URL string to any load function, e.g.
    proxy = 'http://46.2.90.210:8080'
Pass None (default) to disable proxy.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests
import statsmodels.api as sm
import yfinance as yf

from .config import DATA_DIR, ASSETS


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _make_session(proxy: Optional[str]) -> Optional[requests.Session]:
    """Return a requests.Session with proxy settings, or None."""
    if proxy is None:
        return None
    session = requests.Session()
    session.proxies.update({'http': proxy, 'https': proxy})
    return session


# ---------------------------------------------------------------------------
# Fama-French data
# ---------------------------------------------------------------------------
def load_ff_factors(
    start: str = '1970-01-01',
    proxy: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load Fama-French 5-factor (monthly) and market factor (daily) data.

    Note: Ken French's website has up to 2-month data delay.

    Returns
    -------
    fivefactor : monthly DataFrame  (MKT, SMB, HML, RMW, CMA)
    riskfree   : monthly DataFrame  (RF)
    mkt_daily  : daily DataFrame    (MKT)
    """
    kwargs: dict = {}
    session = _make_session(proxy)
    if session is not None:
        kwargs['session'] = session

    ff5_dict = web.DataReader(
        'F-F_Research_Data_5_Factors_2x3', 'famafrench', start=start, **kwargs
    )
    fivefactor = (
        ff5_dict[0]
        .div(100)
        .rename(columns={'Mkt-RF': 'MKT'})
        .drop(columns=['RF'])
    )
    fivefactor.index = fivefactor.index.astype('datetime64[ns]')
    fivefactor = fivefactor.resample('BM').last()

    riskfree = ff5_dict[0].div(100)[['RF']]
    riskfree.index = riskfree.index.astype('datetime64[ns]')
    riskfree = riskfree.resample('BM').last()

    ff_daily_dict = web.DataReader(
        'F-F_Research_Data_Factors_daily', 'famafrench', start=start, **kwargs
    )
    mkt_daily = ff_daily_dict[0].div(100).rename(columns={'Mkt-RF': 'MKT'})[['MKT']]

    return fivefactor, riskfree, mkt_daily


# ---------------------------------------------------------------------------
# Local Excel data
# ---------------------------------------------------------------------------
def load_asset_data(
    filepath: Optional[Path] = None,
    assets: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load asset price data and macro predictors from the Excel file.

    Returns
    -------
    df       : monthly asset prices (BM frequency)
    macro_df : monthly macro predictors
    raw      : full raw DataFrame (all columns, original frequency)
    """
    if filepath is None:
        filepath = DATA_DIR / 'mas_dataset.xlsx'
    if assets is None:
        assets = ASSETS

    raw = pd.read_excel(filepath, sheet_name='data', index_col='Date')

    df = raw[assets].resample('BM').last()

    macros = ['dp', 'tbl', 'lty', 'aaa', 'baa']
    macro_df = raw[macros].resample('BM').last()

    return df, macro_df, raw


# ---------------------------------------------------------------------------
# yfinance data
# ---------------------------------------------------------------------------
_FRED_MACRO_CODES = {
    'tbl': 'TB3MS',     # 3-Month Treasury Bill
    'lty': 'GS10',      # 10-Year Treasury Constant Maturity
    'aaa': 'AAA',       # Moody's Aaa Corporate Bond Yield
    'baa': 'BAA',       # Moody's Baa Corporate Bond Yield
}


def load_asset_data_yfinance(
    assets: Optional[list[str]] = None,
    start: str = '2007-01-01',
    end: Optional[str] = None,
    proxy: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load asset price data from yfinance and macro predictors from FRED.

    Parameters
    ----------
    assets : list of ETF ticker symbols (default: ASSETS from config)
    start  : download start date
    end    : download end date (default: today)
    proxy  : optional proxy URL

    Returns
    -------
    df       : monthly asset prices (BM frequency)
    macro_df : monthly macro predictors (tbl, lty, aaa, baa, dp)
    raw      : daily DataFrame with all asset prices
    """
    if assets is None:
        assets = ASSETS

    # --- Asset prices from yfinance ---
    # Include VCLT (needed by build_seven_factor) even if not in assets
    extra_tickers = [t for t in ['VCLT'] if t not in assets]
    download_tickers = assets + extra_tickers

    raw = yf.download(
        download_tickers,
        start=start,
        end=end,
        proxy=proxy,
        progress=False,
    )['Close']

    # yf.download returns MultiIndex columns for multiple tickers;
    # ensure column order matches the requested tickers list
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(0, axis=1)
    raw = raw[download_tickers]
    raw.index = pd.to_datetime(raw.index)
    raw.index.name = 'Date'

    df = raw[assets].resample('BM').last()

    # --- Macro predictors from FRED ---
    kwargs: dict = {}
    session = _make_session(proxy)
    if session is not None:
        kwargs['session'] = session

    macro_parts = {}
    for name, fred_code in _FRED_MACRO_CODES.items():
        series = web.DataReader(fred_code, 'fred', start=start, **kwargs)
        macro_parts[name] = series.iloc[:, 0]  # keep as % (consistent with Excel source)

    macro_raw = pd.DataFrame(macro_parts)

    # Dividend-price ratio (dp): approximate from S&P 500 ETF (VOO / SPY)
    # Trailing 12-month dividend yield in %, consistent with Excel source
    sp_ticker = 'VOO' if 'VOO' in assets else 'SPY'
    sp = yf.Ticker(sp_ticker)
    divs = sp.dividends
    if divs is not None and len(divs) > 0:
        divs.index = divs.index.tz_localize(None)
        annual_div = divs.resample('BM').sum().rolling(12).sum()
        sp_price = raw[sp_ticker].resample('BM').last()
        dp = (annual_div / sp_price * 100).dropna()
        dp.name = 'dp'
        macro_raw = macro_raw.join(dp)
    else:
        macro_raw['dp'] = np.nan

    macro_df = macro_raw.resample('BM').last()

    return df, macro_df, raw


# ---------------------------------------------------------------------------
# Derived series
# ---------------------------------------------------------------------------
def build_returns(
    df: pd.DataFrame,
    riskfree: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute monthly excess returns and cumulative excess-return price index.

    Returns
    -------
    ret   : monthly excess returns (excess over RF)
    price : cumulative excess-return index (base 1000)
    """
    rf_ = df.join(riskfree)['RF'].ffill()
    ret = df.pct_change().sub(rf_, axis=0)[1:]
    price = np.cumprod(1 + ret) * 1000
    return ret, price


def build_market_states(
    fivefactor: pd.DataFrame,
    mkt_daily: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct market-state and volatility-state indicator variables.

    Market states — Goyal et al. (2024), Eq. (9):
        up_state   = 1 if 3-year cumulative log return >= 0
        down_state = 1 otherwise

    Volatility states — Goyal et al. (2024), Eq. (10):
        hivol_state = 1 if 12-month vol >= 36-month vol
        lovol_state = 1 otherwise

    All state variables are lagged one period to avoid look-ahead bias.
    """
    # Market state
    mkt = (
        np.log1p(fivefactor[['MKT']])
        .rolling(36).sum()
        .dropna()
        .ge(0)
        .astype(int)
        .rename(columns={'MKT': 'up_state'})
    )
    mkt['down_state'] = (mkt['up_state'] == 0).astype(int)
    mkt = mkt.shift()[1:]

    # Volatility state
    vol_12m = (mkt_daily.rolling(21 * 12).std() * np.sqrt(21)).resample('BM').last()[35:]
    vol_36m = (mkt_daily.rolling(21 * 36).std() * np.sqrt(21)).resample('BM').last()[35:]
    vol = (vol_12m >= vol_36m).astype(int).rename(columns={'MKT': 'hivol_state'})
    vol['lovol_state'] = (vol['hivol_state'] == 0).astype(int)
    vol = vol.shift()[1:]

    return mkt, vol


def build_macro_factors(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct macroeconomic predictor variables and lag one period.

    Chordia and Shivakumar (2002), Eq. (1).
    """
    df = macro_df.copy()
    df['term_factor'] = df['lty'] - df['tbl']
    df['def_factor']  = df['baa'] - df['aaa']
    macrofactor = df[['dp', 'tbl', 'term_factor', 'def_factor']].rename(
        columns={'dp': 'div_factor', 'tbl': 'yld_factor'}
    )
    return macrofactor.shift()[1:]


def build_seven_factor(
    fivefactor: pd.DataFrame,
    raw: pd.DataFrame,
    riskfree: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct 7-factor model:
        5 Fama-French factors + TERM (term premium) + DEF (default premium).

    TERM = VGLT excess return over RF
    DEF  = VCLT excess return over VGLT
    """
    bond_ret = raw[['VGLT', 'VCLT']].resample('BM').last().pct_change()
    sevenfactor = fivefactor.join(bond_ret).join(riskfree)
    sevenfactor['TERM'] = sevenfactor['VGLT'] - sevenfactor['RF']
    sevenfactor['DEF']  = sevenfactor['VCLT'] - sevenfactor['VGLT']
    sevenfactor.drop(columns=['VGLT', 'VCLT', 'RF'], inplace=True)
    sevenfactor.dropna(inplace=True)
    return sevenfactor


def build_emrp(
    macrofactor: pd.DataFrame,
    fivefactor: pd.DataFrame,
) -> tuple[pd.DataFrame, object]:
    """
    Estimate Expected Market Risk Premium (EMRP) via OLS of MKT on macro factors.

    Petkova and Zhang (2005).

    Returns
    -------
    emrp : DataFrame with EMRP fitted values and business-cycle state dummies
    res  : statsmodels OLS result object
    """
    data = macrofactor.join(fivefactor[['MKT']]).dropna()
    y = data['MKT']
    X = sm.add_constant(data.drop(columns=['MKT']))
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

    emrp = pd.DataFrame(res.fittedvalues, columns=['EMRP'])
    upper_10 = np.percentile(emrp['EMRP'], 90)
    avg      = emrp['EMRP'].mean()
    lower_10 = np.percentile(emrp['EMRP'], 10)

    emrp['peak']      = (emrp['EMRP'] <= lower_10).astype(int)
    emrp['expansion'] = ((emrp['EMRP'] > lower_10) & (emrp['EMRP'] < avg)).astype(int)
    emrp['recession'] = ((emrp['EMRP'] > avg) & (emrp['EMRP'] < upper_10)).astype(int)
    emrp['trough']    = (emrp['EMRP'] >= upper_10).astype(int)
    emrp['non-trough']= (emrp['EMRP'] < upper_10).astype(int)

    return emrp, res


# ---------------------------------------------------------------------------
# Convenience: load everything at once
# ---------------------------------------------------------------------------
def load_all(
    data_file: Optional[Path] = None,
    ff_start: str = '1970-01-01',
    proxy: Optional[str] = None,
    assets: Optional[list[str]] = None,
    source: str = 'excel',
) -> dict:
    """
    Load and assemble all datasets.

    Parameters
    ----------
    data_file : path to the Excel data file (default: DATA_DIR / 'mas_dataset.xlsx')
    ff_start  : start date for Fama-French download
    proxy     : optional proxy URL, e.g. 'http://host:port'
    assets    : asset ticker list (default: ASSETS from config)
    source    : 'excel' (default) or 'yfinance'

    Returns
    -------
    dict with keys:
        fivefactor, riskfree, mkt_daily,
        df, macro_df, raw,
        ret, price,
        mkt, vol,
        macrofactor, sevenfactor
    """
    fivefactor, riskfree, mkt_daily = load_ff_factors(start=ff_start, proxy=proxy)

    if source == 'yfinance':
        df, macro_df, raw = load_asset_data_yfinance(
            assets=assets, start=ff_start, proxy=proxy,
        )
    else:
        df, macro_df, raw = load_asset_data(filepath=data_file, assets=assets)

    ret, price                      = build_returns(df, riskfree)
    mkt, vol                        = build_market_states(fivefactor, mkt_daily)
    macrofactor                     = build_macro_factors(macro_df)
    sevenfactor                     = build_seven_factor(fivefactor, raw, riskfree)

    return dict(
        fivefactor=fivefactor, riskfree=riskfree, mkt_daily=mkt_daily,
        df=df, macro_df=macro_df, raw=raw,
        ret=ret, price=price,
        mkt=mkt, vol=vol,
        macrofactor=macrofactor, sevenfactor=sevenfactor,
    )
