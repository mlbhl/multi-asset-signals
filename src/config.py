"""
config.py — Project-level constants and model configuration.
"""
from __future__ import annotations
from dataclasses import dataclass, replace as dc_replace
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).parent.parent
DATA_DIR    = ROOT_DIR / 'data'
RESULTS_DIR = ROOT_DIR / 'results'

# ---------------------------------------------------------------------------
# Asset universe
# ---------------------------------------------------------------------------
ASSETS = [
    'VOO', 'QQQ', 'IWM', 'IDEV', 'IEMG',    # Global Equity (5)
    'VGIT', 'VGLT', 'LQD', 'HYG', 'BNDX',   # Global Bonds  (5)
    'GSG', 'GLD', 'IYR',                      # Alternatives  (3)
]
N_EQUITY = 5   # first N_EQUITY assets are equity (used in optimiser constraints)

ASSET_TITLES = [
    'S&P 500', 'NASDAQ 100', 'Russell 2000', 'MSCI World ex USA', 'MSCI EM',
    'US Treasury-Mid', 'US Treasury-Long', 'US IG', 'US HY', 'Global ex-US Bond',
    'Commodities', 'Gold', 'REITs',
]

# ---------------------------------------------------------------------------
# Default model parameters
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # --- Signal lookback periods ---
    momp:  int = 6    # Cross-sectional momentum (Jegadeesh & Titman 1993)
    momk:  int = 0    # Momentum skip period
    revp:  int = 60   # Long-term reversal (De Bondt & Thaler 1985)
    revk:  int = 12   # Reversal skip period
    seasp: int = 120  # Seasonality (Heston & Sadka 2008)
    ftrdp: int = 1    # Fast trend / TS momentum (Moskowitz, Ooi & Pedersen 2012)
    strdp: int = 12   # Slow trend / TS momentum

    # --- Holding periods (Jegadeesh & Titman 1993 method) ---
    holding_mom:  int = 1
    holding_rev:  int = 1
    holding_seas: int = 1
    holding_ftrd: int = 1
    holding_strd: int = 1
    holding_comb: int = 1

    # --- Weighting method ---
    method:   str = 'quantile'  # 'quantile' | 'proportion' | 'rank'
    quantile: str = 'tercile'   # 'median' | 'tercile' | 'quartile' | 'quintile' | 'decile'

    # --- Transaction costs (one-way) ---
    tcost: float = 0.001

    def replace(self, **kwargs) -> 'ModelConfig':
        """Return a new ModelConfig with selected fields overridden."""
        return dc_replace(self, **kwargs)


DEFAULT_CONFIG = ModelConfig()

# ---------------------------------------------------------------------------
# Target-date fund benchmark weights
# ---------------------------------------------------------------------------
# Original spec had 15-element arrays; the last 2 (0.02 each, non-investable
# components) have been removed and remaining weights renormalized to sum = 1.
_BMS_RAW: dict[str, np.ndarray] = {
    'b2055': np.array([0.077, 0.3962, 0.1698, 0.0104, 0.0508,
                       0.0177, 0.0088, 0.0394, 0.0510, 0.0255,
                       0.0085, 0.0840, 0.0210]),
    'b2050': np.array([0.072, 0.3704, 0.1588, 0.0097, 0.0475,
                       0.0165, 0.0082, 0.0368, 0.0660, 0.0330,
                       0.0110, 0.1040, 0.0260]),
    'b2045': np.array([0.068, 0.3499, 0.1499, 0.0092, 0.0449,
                       0.0156, 0.0077, 0.0348, 0.0780, 0.0390,
                       0.0130, 0.1200, 0.0300]),
    'b2040': np.array([0.0635, 0.3267, 0.1400, 0.0086, 0.0419,
                       0.0146, 0.0072, 0.0325, 0.0960, 0.0480,
                       0.0160, 0.1400, 0.0350]),
    'b2035': np.array([0.0595, 0.3061, 0.1312, 0.0081, 0.0393,
                       0.0136, 0.0068, 0.0304, 0.1080, 0.0540,
                       0.0180, 0.1560, 0.0390]),
    'b2030': np.array([0.0535, 0.2753, 0.1180, 0.0072, 0.0353,
                       0.0123, 0.0061, 0.0274, 0.1260, 0.0630,
                       0.0210, 0.1800, 0.0450]),
    'b2020': np.array([0.034,  0.1749, 0.0750, 0.0046, 0.0224,
                       0.0078, 0.0039, 0.0174, 0.1860, 0.0930,
                       0.0310, 0.2560, 0.0640]),
}
BMS: dict[str, np.ndarray] = {k: v / v.sum() for k, v in _BMS_RAW.items()}

PORTFOLIO_NAMES = ['t2055', 't2050', 't2045', 't2040', 't2035', 't2030', 't2020']

# Allocation constraint multipliers (applied to BM weights)
UP_SCALE         = 1.2
DOWN_SCALE       = 0.8
SPEC_UP_SCALE    = {1: 1.1}   # index 1 (QQQ): tighter upper bound
SPEC_DOWN_SCALE  = {1: 0.9}   # index 1 (QQQ): tighter lower bound
