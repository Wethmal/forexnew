"""
╔==============================================================================╗
║           HALDANE TRADING BOT v3.0 — FULL PRODUCTION SYSTEM                ║
║                                                                              ║
║  Strategy  : Multi-timeframe Trend + Momentum + Volatility Breakout         ║
║  Models    : EMA Trend Filter + RSI + MACD + Bollinger + Stochastic + ATR   ║
║  Execution : MetaTrader5 (live) / Yahoo Finance (paper)                     ║
║  Risk Mgmt : Fixed fractional sizing, ATR-based SL/TP, daily loss limits    ║
║  Dashboard : JSON export for live monitoring                                 ║
║  Backtest  : Walk-forward simulation with full trade log                    ║
║                                                                              ║
║  Author    : Haldane System                                                  ║
║  Version   : 3.0.0                                                           ║
╚==============================================================================╝

SENIOR TRADER NOTES:
  - This bot uses CONFLUENCE: at least 3/5 indicators must agree before entry
  - Never trades against the 1H trend (EMA50 filter is non-negotiable)
  - Every trade has a pre-set SL and TP before entry — no exceptions
  - Daily drawdown limit: if you lose 3% of equity in a day, bot stops trading
  - Position sizing is automatic based on your account equity and SL distance
  - Min 2:1 reward-to-risk on every trade — if you can't get 2R, skip the trade

HOW TO RUN:
  Paper mode  : python haldane_bot_full.py
  Backtest    : python haldane_bot_full.py --backtest
  Signal check: python haldane_bot_full.py --signal
  Live mode   : Set live_trading=True + MT5 env vars

ENV VARS FOR LIVE MT5:
  MT5_LOGIN     = your account number
  MT5_PASSWORD  = your password
  MT5_SERVER    = your broker server name
"""

# -----------------------------------------------------------------------------
#  IMPORTS
# -----------------------------------------------------------------------------

import os
import sys
import json
import time
import asyncio
import threading
import logging
import warnings
import traceback
from copy import deepcopy
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple
from enum import Enum

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# -- Optional MT5 --------------------------------------------------------------
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False

# -- Optional TensorFlow / LSTM ------------------------------------------------
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False

# -- Optional scikit-learn -----------------------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    HAS_SKL = True
except ImportError:
    HAS_SKL = False


# -----------------------------------------------------------------------------
#  LOGGING SETUP
# -----------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("HaldaneBot")

class ColorFormatter(logging.Formatter):
    """Colored terminal output for readability."""
    COLORS = {
        logging.DEBUG:    "\033[90m",   # gray
        logging.INFO:     "\033[0m",    # normal
        logging.WARNING:  "\033[93m",   # yellow
        logging.ERROR:    "\033[91m",   # red
        logging.CRITICAL: "\033[95m",   # magenta
    }
    RESET = "\033[0m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        # Highlight BUY / SELL signals
        msg = super().format(record)
        if "BUY" in msg:
            msg = msg.replace("BUY", f"{self.GREEN}BUY{self.RESET}")
        if "SELL" in msg:
            msg = msg.replace("SELL", f"\033[91mSELL{self.RESET}")
        return f"{color}{msg}{self.RESET}"

# Apply colored formatter to console handler
for handler in logging.root.handlers:
    handler.setFormatter(ColorFormatter(LOG_FORMAT))


# -----------------------------------------------------------------------------
#  CONFIGURATION
# -----------------------------------------------------------------------------

@dataclass
class BotConfig:
    # -- Symbols ---------------------------------------------------------------
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"
    ])
    # MT5 symbol suffix if your broker uses one (e.g. "m" → "EURUSDm")
    mt5_symbol_suffix: str = ""

    # -- Timeframes ------------------------------------------------------------
    trend_tf: str    = "1h"    # Higher timeframe for EMA trend filter
    entry_tf: str    = "15m"   # Lower timeframe for entry signals
    confirm_tf: str  = "4h"    # Confirmation / big-picture bias

    # -- Indicator settings ----------------------------------------------------
    ema_fast: int    = 9
    ema_slow: int    = 21
    ema_trend: int   = 50
    ema_macro: int   = 200

    rsi_period: int  = 14
    rsi_buy_min: float  = 45.0   # RSI must be above this to buy
    rsi_buy_max: float  = 70.0   # RSI must be below this to buy (not overbought)
    rsi_sell_min: float = 30.0   # RSI must be above this to sell (not oversold)
    rsi_sell_max: float = 55.0   # RSI must be below this to sell

    macd_fast: int   = 12
    macd_slow: int   = 26
    macd_signal: int = 9

    bb_period: int   = 20
    bb_std: float    = 2.0

    stoch_k: int     = 14
    stoch_d: int     = 3
    stoch_smooth: int = 3

    atr_period: int  = 14
    adx_period: int  = 14

    volume_ma: int   = 20

    # -- Signal confluence -----------------------------------------------------
    # Minimum number of confirming indicators required (out of 5) to take a trade
    min_confluence: int = 3

    # -- Risk management -------------------------------------------------------
    risk_per_trade_pct: float  = 0.01   # 1% of equity per trade
    sl_atr_multiplier: float   = 1.5    # SL = 1.5 × ATR from entry
    tp_rr_ratio: float         = 2.0    # TP = 2× the SL distance
    min_rr_ratio: float        = 1.8    # Skip trade if we can't get at least 1.8R
    max_positions: int         = 3      # Max simultaneous open trades
    max_daily_loss_pct: float  = 0.03   # Stop trading if daily loss > 3%
    max_position_pct: float    = 0.10   # No single position > 10% of equity
    trailing_sl_atr: float     = 1.0    # Trail SL by 1× ATR when in profit

    # -- Execution ------------------------------------------------------------
    live_trading: bool       = False
    mt5_magic: int           = 20250415
    mt5_comment: str         = "Haldane v3"
    lot_min: float           = 0.01
    lot_max: float           = 0.50
    lot_step: float          = 0.01
    spread_max_pips: float   = 1.5     # Default max spread allowed in pips
    symbol_spread_limits: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 1.2,
        "GBPUSD": 1.5,
        "USDJPY": 1.5,
        "AUDUSD": 1.5,
        "USDCAD": 1.5,
        "GBPJPY": 2.5,
        "EURJPY": 2.0,
    })
    mt5_deviation: int       = 10      # Max slippage in points

    # -- Loop timing ----------------------------------------------------------
    interval_seconds: int    = 60
    candle_close_wait: int   = 5       # Wait N seconds after candle close

    # -- ML models ------------------------------------------------------------
    use_ml: bool             = False   # Enable ML probability filter
    ml_min_probability: float = 0.60  # Min ML prob to take trade
    lstm_lookback: int       = 60      # Candles fed to LSTM
    lstm_epochs: int         = 50
    lstm_batch: int          = 32
    rf_n_estimators: int     = 200

    # -- Session filters -------------------------------------------------------
    # Only trade during these UTC hours (most liquid sessions)
    # London: 07:00-16:00, New York: 13:00-22:00, overlap 13:00-16:00
    trade_sessions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (7, 16),   # London session
        (13, 22),  # New York session
    ])

    # -- File paths -----------------------------------------------------------
    dashboard_file: str  = "dashboard_data.json"
    trade_log_file: str  = "trade_log.json"
    model_cache_dir: str = "models"


# -----------------------------------------------------------------------------
#  DATA CLASSES
# -----------------------------------------------------------------------------

class SignalType(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeSignal:
    symbol: str
    signal: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    sl_distance: float
    tp_distance: float
    rr_ratio: float
    lots: float
    confidence: float           # 0.0 to 1.0
    confluence_score: int       # How many indicators agree (0-5)
    trend: str                  # "UP" or "DOWN"
    rsi: float
    macd_hist: float
    atr: float
    spread_pips: float
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    ml_probability: float = 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = self.signal.value
        return d


@dataclass
class TradeResult:
    symbol: str
    signal: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lots: float
    pnl_pips: float
    pnl_usd: float
    outcome: str           # "WIN", "LOSS", "BE" (breakeven)
    entry_time: str
    exit_time: str
    duration_minutes: float
    reason_for_exit: str   # "TP_HIT", "SL_HIT", "MANUAL", "TIMEOUT"

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
#  TECHNICAL INDICATORS
# -----------------------------------------------------------------------------

class Indicators:
    """All technical indicator calculations in one place."""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26,
             signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        exp_fast   = series.ewm(span=fast, adjust=False).mean()
        exp_slow   = series.ewm(span=slow, adjust=False).mean()
        macd_line  = exp_fast - exp_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram  = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20,
                        std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        middle = series.rolling(window=period).mean()
        std    = series.rolling(window=period).std()
        upper  = middle + (std * std_dev)
        lower  = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def stochastic(df: pd.DataFrame, k: int = 14, d: int = 3,
                   smooth: int = 3) -> Tuple[pd.Series, pd.Series]:
        low_min  = df["Low"].rolling(k).min()
        high_max = df["High"].rolling(k).max()
        denom    = (high_max - low_min).replace(0, np.nan)
        stoch_k  = 100 * ((df["Close"] - low_min) / denom)
        stoch_k  = stoch_k.rolling(smooth).mean()   # Smooth %K
        stoch_d  = stoch_k.rolling(d).mean()
        return stoch_k, stoch_d

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl  = df["High"] - df["Low"]
        hc  = (df["High"] - df["Close"].shift()).abs()
        lc  = (df["Low"]  - df["Close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index — measures trend strength."""
        high = df["High"]
        low  = df["Low"]
        close= df["Close"]

        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm  < 0] = 0
        minus_dm[minus_dm < 0] = 0
        # Keep only the larger move
        plus_dm[plus_dm < minus_dm]  = 0
        minus_dm[minus_dm < plus_dm] = 0

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr_val   = tr.rolling(period).mean()
        plus_di   = 100 * (plus_dm.rolling(period).mean()  / atr_val)
        minus_di  = 100 * (minus_dm.rolling(period).mean() / atr_val)
        dx        = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
        adx_val   = dx.rolling(period).mean()
        return adx_val, plus_di, minus_di

    @staticmethod
    def volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
        vol_sma = df["Volume"].rolling(period).mean()
        return df["Volume"] / vol_sma.replace(0, np.nan)

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_max = df["High"].rolling(period).max()
        low_min  = df["Low"].rolling(period).min()
        denom    = (high_max - low_min).replace(0, np.nan)
        return -100 * ((high_max - df["Close"]) / denom)

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp  = (df["High"] + df["Low"] + df["Close"]) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (tp - sma) / (0.015 * mad)

    @classmethod
    def add_all(cls, df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
        """Add all indicators to a DataFrame and return it."""
        df = df.copy()
        c = df["Close"]

        # EMAs
        df["ema_fast"]  = cls.ema(c, cfg.ema_fast)
        df["ema_slow"]  = cls.ema(c, cfg.ema_slow)
        df["ema_trend"] = cls.ema(c, cfg.ema_trend)
        df["ema_macro"] = cls.ema(c, cfg.ema_macro)

        # RSI
        df["rsi"] = cls.rsi(c, cfg.rsi_period)

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = cls.macd(
            c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )

        # Bollinger Bands
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = cls.bollinger_bands(
            c, cfg.bb_period, cfg.bb_std
        )
        df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_position"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        # Stochastic
        df["stoch_k"], df["stoch_d"] = cls.stochastic(
            df, cfg.stoch_k, cfg.stoch_d, cfg.stoch_smooth
        )

        # ATR
        df["atr"] = cls.atr(df, cfg.atr_period)

        # ADX
        df["adx"], df["plus_di"], df["minus_di"] = cls.adx(df, cfg.adx_period)

        # Volume
        df["volume_ratio"] = cls.volume_ratio(df, cfg.volume_ma)

        # Williams %R
        df["williams_r"] = cls.williams_r(df, 14)

        # CCI
        df["cci"] = cls.cci(df, 20)

        # Price action features
        df["price_change"]    = c.pct_change()
        df["price_change_5"]  = c.pct_change(5)
        df["price_change_10"] = c.pct_change(10)
        df["volatility"]      = df["price_change"].rolling(20).std()
        df["candle_body"]     = (df["Close"] - df["Open"]).abs()
        df["candle_range"]    = df["High"] - df["Low"]
        df["body_ratio"]      = df["candle_body"] / df["candle_range"].replace(0, np.nan)

        # Higher highs / lower lows (swing structure)
        df["swing_high"] = df["High"].rolling(5).max()
        df["swing_low"]  = df["Low"].rolling(5).min()

        df.dropna(inplace=True)
        return df


# -----------------------------------------------------------------------------
#  DATA FETCHER
# -----------------------------------------------------------------------------

class DataFetcher:
    """Fetches OHLCV data from Yahoo Finance or MetaTrader 5."""

    # Map short TF strings to Yahoo Finance interval strings
    YF_INTERVAL_MAP = {
        "1m": "1m",  "5m": "5m",  "15m": "15m", "30m": "30m",
        "1h": "1h",  "4h": "1h",  "1d": "1d",   "1wk": "1wk",
    }
    YF_PERIOD_MAP = {
        "1m": "7d",  "5m": "30d", "15m": "60d", "30m": "60d",
        "1h": "90d", "4h": "180d","1d": "2y",   "1wk": "5y",
    }

    def __init__(self, cfg: BotConfig):
        self.cfg = cfg

    def _yf_symbol(self, symbol: str) -> str:
        """Convert plain forex symbol to Yahoo Finance format."""
        if len(symbol) == 6 and symbol.isalpha():
            return f"{symbol[:3]}{symbol[3:]}=X"
        return symbol

    def fetch_yf(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        yf_sym    = self._yf_symbol(symbol)
        interval  = self.YF_INTERVAL_MAP.get(timeframe, "1h")
        period    = self.YF_PERIOD_MAP.get(timeframe, "90d")

        try:
            df = yf.download(yf_sym, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                log.warning(f"YF: No data for {yf_sym} ({interval})")
                return pd.DataFrame()

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure required columns exist
            required = ["Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in df.columns:
                    log.error(f"YF: Missing column {col} for {yf_sym}")
                    return pd.DataFrame()

            df = df[required].copy()
            df.dropna(inplace=True)
            log.debug(f"YF: fetched {len(df)} bars for {yf_sym} ({interval})")
            return df

        except Exception as e:
            log.error(f"YF fetch error for {yf_sym}: {e}")
            return pd.DataFrame()

    def fetch_mt5(self, symbol: str, timeframe: str,
                  count: int = 500) -> pd.DataFrame:
        """Fetch data from MetaTrader 5."""
        if not HAS_MT5:
            raise RuntimeError("MetaTrader5 not installed")

        TF_MAP = {
            "1m":  mt5.TIMEFRAME_M1,  "5m":  mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
            "1h":  mt5.TIMEFRAME_H1,  "4h":  mt5.TIMEFRAME_H4,
            "1d":  mt5.TIMEFRAME_D1,  "1wk": mt5.TIMEFRAME_W1,
        }
        tf = TF_MAP.get(timeframe, mt5.TIMEFRAME_H1)
        mt5_sym = symbol
        if self.cfg.mt5_symbol_suffix and not symbol.endswith(self.cfg.mt5_symbol_suffix):
            mt5_sym += self.cfg.mt5_symbol_suffix

        rates = mt5.copy_rates_from_pos(mt5_sym, tf, 0, count)
        if rates is None or len(rates) == 0:
            log.error(f"MT5: No data for {mt5_sym}. Error: {mt5.last_error()}")
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume",
        }, inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        log.debug(f"MT5: fetched {len(df)} bars for {mt5_sym} ({timeframe})")
        return df

    def fetch(self, symbol: str, timeframe: str,
              use_mt5: bool = False) -> pd.DataFrame:
        """Unified fetch method."""
        if use_mt5 and HAS_MT5:
            return self.fetch_mt5(symbol, timeframe)
        return self.fetch_yf(symbol, timeframe)

    def get_current_spread_pips(self, symbol: str) -> float:
        """Get current spread in pips from MT5 tick data."""
        if not HAS_MT5:
            return 1.0   # Assume 1 pip for paper trading
        mt5_sym = symbol + self.cfg.mt5_symbol_suffix
        tick = mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            return 99.0
        symbol_info = mt5.symbol_info(mt5_sym)
        if symbol_info is None:
            return 99.0
        spread_points = tick.ask - tick.bid
        pip_size = 0.0001 if symbol_info.digits == 5 else 0.01
        return spread_points / pip_size


# -----------------------------------------------------------------------------
#  SIGNAL ENGINE
# -----------------------------------------------------------------------------

class SignalEngine:
    """
    Multi-timeframe signal generation with confluence scoring.

    Flow:
      1. Fetch 4H data → macro bias (overall market direction)
      2. Fetch 1H data → trend filter (EMA50, EMA200)
      3. Fetch 15m data → entry trigger (RSI, MACD, Stoch, BB)
      4. Count how many indicators confirm the trend direction
      5. Only signal if confluence >= min_confluence
    """

    def __init__(self, cfg: BotConfig, fetcher: DataFetcher):
        self.cfg     = cfg
        self.fetcher = fetcher

    # -- Session check ---------------------------------------------------------
    def is_trading_session(self) -> bool:
        """Return True if current UTC time is within an active trading session."""
        hour = datetime.utcnow().hour
        for start, end in self.cfg.trade_sessions:
            if start <= hour < end:
                return True
        return False

    # -- Macro bias ------------------------------------------------------------
    def get_macro_bias(self, symbol: str, use_mt5: bool) -> Optional[str]:
        """
        4H chart: price vs EMA200.
        Returns "UP", "DOWN", or None if unclear.
        """
        df = self.fetcher.fetch(symbol, self.cfg.confirm_tf, use_mt5)
        if df.empty or len(df) < 210:
            return None

        df = Indicators.add_all(df, self.cfg)
        last = df.iloc[-1]

        if last["Close"] > last["ema_macro"]:
            return "UP"
        elif last["Close"] < last["ema_macro"]:
            return "DOWN"
        return None

    # -- Trend filter ---------------------------------------------------------
    def get_trend(self, symbol: str, use_mt5: bool) -> Tuple[Optional[str], pd.Series]:
        """
        1H chart: EMA50 trend filter.
        Returns ("UP"/"DOWN"/None, last_row).
        """
        df = self.fetcher.fetch(symbol, self.cfg.trend_tf, use_mt5)
        if df.empty or len(df) < 60:
            log.warning(f"{symbol}: insufficient 1H data")
            return None, pd.Series()

        df = Indicators.add_all(df, self.cfg)
        last = df.iloc[-1]

        if last["Close"] > last["ema_trend"] and last["ema_fast"] > last["ema_slow"]:
            trend = "UP"
        elif last["Close"] < last["ema_trend"] and last["ema_fast"] < last["ema_slow"]:
            trend = "DOWN"
        else:
            trend = None  # No clear trend — skip

        return trend, last

    # -- Entry confluence ------------------------------------------------------
    def score_entry(self, df15: pd.DataFrame, trend: str) -> Tuple[int, List[str]]:
        """
        Score how many of 5 entry indicators confirm the trade direction.

        Indicators scored (1 point each):
          1. RSI in valid range
          2. MACD histogram in trend direction
          3. Stochastic not overbought/oversold
          4. Bollinger Band position
          5. ADX trend strength

        Returns (score, list of reasons).
        """
        if len(df15) < 2:
            return 0, []

        cur    = df15.iloc[-1]
        prev   = df15.iloc[-2]
        score  = 0
        reasons = []

        rsi = cur["rsi"]
        macd_hist = cur["macd_hist"]
        stoch_k   = cur["stoch_k"]
        stoch_d   = cur["stoch_d"]
        bb_pos    = cur["bb_position"]   # 0=at lower band, 1=at upper band
        adx       = cur["adx"]
        plus_di   = cur["plus_di"]
        minus_di  = cur["minus_di"]

        if trend == "UP":
            # 1. RSI between 45 and 70 (momentum but not overbought)
            if self.cfg.rsi_buy_min <= rsi <= self.cfg.rsi_buy_max:
                score += 1
                reasons.append(f"RSI={rsi:.1f} ✓ (buy zone)")

            # 2. MACD histogram positive or crossing up
            if macd_hist > 0 or (prev["macd_hist"] < 0 < macd_hist):
                score += 1
                reasons.append(f"MACD hist={macd_hist:.5f} ✓ (bullish)")

            # 3. Stochastic not overbought, ideally rising
            if stoch_k < 80 and stoch_k > stoch_d:
                score += 1
                reasons.append(f"Stoch K={stoch_k:.1f} ✓ (not overbought, K>D)")

            # 4. Price in lower half of Bollinger Band (room to run)
            if bb_pos < 0.6:
                score += 1
                reasons.append(f"BB pos={bb_pos:.2f} ✓ (room to upper band)")

            # 5. ADX > 20 (trending) with +DI > -DI
            if adx > 20 and plus_di > minus_di:
                score += 1
                reasons.append(f"ADX={adx:.1f} ✓ (+DI>{minus_di:.1f})")

        elif trend == "DOWN":
            # 1. RSI between 30 and 55
            if self.cfg.rsi_sell_min <= rsi <= self.cfg.rsi_sell_max:
                score += 1
                reasons.append(f"RSI={rsi:.1f} ✓ (sell zone)")

            # 2. MACD histogram negative or crossing down
            if macd_hist < 0 or (prev["macd_hist"] > 0 > macd_hist):
                score += 1
                reasons.append(f"MACD hist={macd_hist:.5f} ✓ (bearish)")

            # 3. Stochastic not oversold, ideally falling
            if stoch_k > 20 and stoch_k < stoch_d:
                score += 1
                reasons.append(f"Stoch K={stoch_k:.1f} ✓ (not oversold, K<D)")

            # 4. Price in upper half of BB (room to fall)
            if bb_pos > 0.4:
                score += 1
                reasons.append(f"BB pos={bb_pos:.2f} ✓ (room to lower band)")

            # 5. ADX > 20 with -DI > +DI
            if adx > 20 and minus_di > plus_di:
                score += 1
                reasons.append(f"ADX={adx:.1f} ✓ (-DI>{plus_di:.1f})")

        return score, reasons

    # -- Compute SL / TP -------------------------------------------------------
    def compute_sl_tp(self, signal_type: SignalType, entry: float,
                      atr: float, df15: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate SL behind last swing high/low, then TP at RR ratio.
        """
        sl_dist = atr * self.cfg.sl_atr_multiplier

        # Refine SL using swing structure
        if signal_type == SignalType.BUY:
            # SL below recent swing low
            recent_low = df15["Low"].rolling(10).min().iloc[-1]
            sl_from_swing = entry - recent_low
            sl_dist = max(sl_dist, sl_from_swing * 1.1)   # at least ATR-based, or behind swing
            stop_loss   = entry - sl_dist
            take_profit = entry + (sl_dist * self.cfg.tp_rr_ratio)
        else:
            # SL above recent swing high
            recent_high = df15["High"].rolling(10).max().iloc[-1]
            sl_from_swing = recent_high - entry
            sl_dist = max(sl_dist, sl_from_swing * 1.1)
            stop_loss   = entry + sl_dist
            take_profit = entry - (sl_dist * self.cfg.tp_rr_ratio)

        return stop_loss, take_profit, sl_dist

    # -- Main generate function ------------------------------------------------
    def generate(self, symbol: str, equity: float,
                 use_mt5: bool = False) -> Optional[TradeSignal]:
        """
        Full signal generation pipeline for one symbol.
        Returns a TradeSignal or None.
        """
        # -- Session filter ----------------------------------------------------
        if not self.is_trading_session():
            log.debug(f"{symbol}: outside trading session")
            return None

        # -- Spread check -----------------------------------------------------
        spread_pips = self.fetcher.get_current_spread_pips(symbol)
        max_allowed = self.cfg.symbol_spread_limits.get(symbol, self.cfg.spread_max_pips)
        
        if spread_pips > max_allowed:
            log.warning(f"{symbol}: spread too wide ({spread_pips:.1f} pips > "
                        f"{max_allowed}). Skipping.")
            return None

        # -- Macro bias (4H) --------------------------------------------------
        macro = self.get_macro_bias(symbol, use_mt5)
        if macro is None:
            log.debug(f"{symbol}: macro bias unclear")
            # Don't hard-block — macro is a soft filter

        # -- Trend filter (1H) ------------------------------------------------
        trend, trend_row = self.get_trend(symbol, use_mt5)
        if trend is None:
            log.info(f"{symbol}: no clear 1H trend — skipping")
            return None

        # Macro must agree with trend (or be unknown)
        if macro is not None and macro != trend:
            log.info(f"{symbol}: macro ({macro}) disagrees with 1H trend ({trend}) — skipping")
            return None

        # -- Entry signals (15m) ----------------------------------------------
        df15 = self.fetcher.fetch(symbol, self.cfg.entry_tf, use_mt5)
        if df15.empty or len(df15) < 80:
            log.warning(f"{symbol}: insufficient 15m data")
            return None

        df15 = Indicators.add_all(df15, self.cfg)
        confluence_score, reasons = self.score_entry(df15, trend)

        if confluence_score < self.cfg.min_confluence:
            log.info(f"{symbol}: confluence too low ({confluence_score}/"
                     f"{self.cfg.min_confluence}) — skipping")
            return None

        cur    = df15.iloc[-1]
        entry  = cur["Close"]
        atr    = cur["atr"]
        rsi    = cur["rsi"]
        macd_h = cur["macd_hist"]

        signal_type = SignalType.BUY if trend == "UP" else SignalType.SELL

        # -- SL / TP calculation -----------------------------------------------
        stop_loss, take_profit, sl_dist = self.compute_sl_tp(
            signal_type, entry, atr, df15
        )

        # Check minimum RR ratio
        tp_dist = abs(take_profit - entry)
        actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if actual_rr < self.cfg.min_rr_ratio:
            log.info(f"{symbol}: RR={actual_rr:.2f} below minimum {self.cfg.min_rr_ratio} — skipping")
            return None

        # -- Position sizing ---------------------------------------------------
        lots = self._calc_lots(equity, sl_dist, symbol)
        if lots <= 0:
            log.warning(f"{symbol}: position size is 0, skipping")
            return None

        # -- Confidence score -------------------------------------------------
        confidence = confluence_score / 5.0

        reason_str = " | ".join(reasons)
        full_reason = (
            f"Trend={trend} Macro={macro} | Score={confluence_score}/5 | "
            f"{reason_str}"
        )

        signal = TradeSignal(
            symbol=symbol,
            signal=signal_type,
            entry_price=round(float(entry), 5),
            stop_loss=round(float(stop_loss), 5),
            take_profit=round(float(take_profit), 5),
            sl_distance=round(float(sl_dist), 5),
            tp_distance=round(float(tp_dist), 5),
            rr_ratio=round(actual_rr, 2),
            lots=lots,
            confidence=round(confidence, 3),
            confluence_score=confluence_score,
            trend=trend,
            rsi=round(float(rsi), 2),
            macd_hist=round(float(macd_h), 6),
            atr=round(float(atr), 5),
            spread_pips=round(spread_pips, 1),
            reason=full_reason,
        )

        log.info(
            f"SIGNAL ► {signal.signal.value} {symbol} | "
            f"Entry={signal.entry_price} SL={signal.stop_loss} TP={signal.take_profit} | "
            f"RR={signal.rr_ratio} Lots={signal.lots} | Score={confluence_score}/5"
        )
        return signal

    def _calc_lots(self, equity: float, sl_distance: float,
                   symbol: str) -> float:
        """
        Dynamic position sizing using MT5 tick value/size.
        Falls back to approximate pip-value logic for paper trading.
        """
        cfg = self.cfg
        risk_amount = equity * cfg.risk_per_trade_pct

        if sl_distance <= 0:
            return 0.0

        per_lot_risk = 0.0
        
        # 1. Try to get exact risk from MT5 (Live Mode)
        if HAS_MT5 and mt5.terminal_info():
            mt5_sym = symbol + cfg.mt5_symbol_suffix
            info = mt5.symbol_info(mt5_sym)
            if info:
                # trade_tick_value is the value of 1 tick (min price change) for 1 lot in Deposit Currency
                tv = info.trade_tick_value
                ts = info.trade_tick_size
                if tv > 0 and ts > 0:
                    per_lot_risk = (sl_distance / ts) * tv
                    log.debug(f"{symbol}: MT5 dynamic per-lot risk calculation used.")

        # 2. Fallback to approximate logic (Paper Mode / MT5 Error)
        if per_lot_risk <= 0:
            pip_size = 0.01 if "JPY" in symbol else 0.0001
            # Standard majors are $10/pip/lot. JPY pairs approx $9/pip/lot.
            pip_val = 9.0 if "JPY" in symbol else 10.0
            sl_pips = sl_distance / pip_size
            per_lot_risk = sl_pips * pip_val
            log.debug(f"{symbol}: Fallback approximate per-lot risk used.")

        # Calculate lots = Risk / (Risk Per Lot)
        lots = risk_amount / per_lot_risk if per_lot_risk > 0 else 0.0

        # Snap to lot step (usually 0.01) and apply min/max boundaries
        lots = round(lots / cfg.lot_step) * cfg.lot_step
        lots = max(cfg.lot_min, min(lots, cfg.lot_max))

        # Secondary safety cap: ensure the lot size doesn't exceed max_position_pct of equity
        max_risk_allowed = equity * cfg.max_position_pct
        max_lots_by_equity = max_risk_allowed / per_lot_risk if per_lot_risk > 0 else 0
        lots = min(lots, max_lots_by_equity)

        return round(lots, 2)



# -----------------------------------------------------------------------------
#  ML PROBABILITY FILTER (optional)
# -----------------------------------------------------------------------------

class MLFilter:
    """
    Uses a gradient-boosted ensemble trained on historical entry-timeframe
    bars. Target is whether price hits a 2R Take Profit before a 1.5R Stop Loss.
    Only activates if HAS_SKL is True.
    """

    FEATURE_COLS = [
        "ema_fast", "ema_slow", "ema_trend",
        "rsi", "macd", "macd_hist",
        "bb_position", "bb_width",
        "stoch_k", "stoch_d",
        "atr", "adx", "plus_di", "minus_di",
        "volume_ratio", "price_change", "price_change_5",
        "price_change_10", "volatility", "body_ratio",
        "williams_r", "cci",
    ]

    def __init__(self, cfg: BotConfig):
        self.cfg      = cfg
        self.models: Dict[str, object] = {}
        self.scalers: Dict[str, object] = {}
        self.trained  = False

        if not HAS_SKL:
            log.warning("scikit-learn not installed. ML filter disabled.")

    def _build_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X and target y using 2R TP vs 1.5R SL logic."""
        df = df.copy()
        
        if "atr" not in df.columns:
            return np.array([]), np.array([])

        closes = df["Close"].values
        highs  = df["High"].values
        lows   = df["Low"].values
        atrs   = df["atr"].values
        
        n = len(df)
        targets = np.zeros(n)
        valid_mask = np.zeros(n, dtype=bool)
        
        # Look forward to see if TP or SL is hit first
        # 100 bars on 15m is ~25 hours
        look_ahead = 100 
        
        sl_mult = self.cfg.sl_atr_multiplier  # default 1.5
        tp_mult = sl_mult * self.cfg.tp_rr_ratio # 1.5 * 2.0 = 3.0
        
        for i in range(n - look_ahead):
            entry = closes[i]
            sl_price = entry - (atrs[i] * sl_mult)
            tp_price = entry + (atrs[i] * tp_mult)
            
            for j in range(i + 1, i + look_ahead):
                if lows[j] <= sl_price:
                    targets[i] = 0
                    valid_mask[i] = True
                    break
                if highs[j] >= tp_price:
                    targets[i] = 1
                    valid_mask[i] = True
                    break
        
        df_final = df.iloc[valid_mask].copy()
        if df_final.empty:
            return np.array([]), np.array([])
            
        available = [c for c in self.FEATURE_COLS if c in df_final.columns]
        X = df_final[available].values
        y = targets[valid_mask]
        
        return X, y

    def train(self, symbol: str, df: pd.DataFrame) -> float:
        """Train ensemble on historical data. Returns test accuracy."""
        if not HAS_SKL:
            return 0.0

        X, y = self._build_features(df)
        if len(X) < 200:
            log.warning(f"ML: too little data to train for {symbol}")
            return 0.0

        # Time-series split — no data leakage
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)

        self.models[symbol]  = model
        self.scalers[symbol] = scaler

        log.info(f"ML {symbol}: accuracy={acc:.3f} f1={f1:.3f} (test set)")
        return acc

    def predict_probability(self, symbol: str, df: pd.DataFrame,
                             direction: str) -> float:
        """
        Return probability of a successful trade in the given direction.
        Returns 0.5 if model not trained or unavailable.
        """
        if not HAS_SKL or symbol not in self.models:
            return 0.5

        available = [c for c in self.FEATURE_COLS if c in df.columns]
        last_row  = df[available].iloc[-1:].values

        try:
            scaler = self.scalers[symbol]
            model  = self.models[symbol]
            X_sc   = scaler.transform(last_row)
            prob   = model.predict_proba(X_sc)[0]
            # prob[1] = probability of price going up
            return float(prob[1]) if direction == "UP" else float(prob[0])
        except Exception as e:
            log.error(f"ML predict error for {symbol}: {e}")
            return 0.5


# -----------------------------------------------------------------------------
#  TRADE EXECUTOR
# -----------------------------------------------------------------------------

class TradeExecutor:
    """Handles order execution, position management, and SL/TP trailing."""

    def __init__(self, cfg: BotConfig, fetcher: DataFetcher):
        self.cfg     = cfg
        self.fetcher = fetcher

    # -- MT5 helpers -----------------------------------------------------------
    def _mt5_symbol(self, symbol: str) -> str:
        if self.cfg.mt5_symbol_suffix and not symbol.endswith(self.cfg.mt5_symbol_suffix):
            return symbol + self.cfg.mt5_symbol_suffix
        return symbol

    def get_open_positions(self) -> List[dict]:
        """Return list of open positions as dicts."""
        if not HAS_MT5:
            return []
        positions = mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            result.append({
                "ticket":      p.ticket,
                "symbol":      p.symbol,
                "type":        "BUY" if p.type == 0 else "SELL",
                "volume":      p.volume,
                "price_open":  p.price_open,
                "price_cur":   p.price_current,
                "sl":          p.sl,
                "tp":          p.tp,
                "profit":      p.profit,
                "magic":       p.magic,
            })
        return result

    def has_position_for(self, symbol: str) -> bool:
        """Check if we already have an open position on this symbol."""
        sym = self._mt5_symbol(symbol)
        for p in self.get_open_positions():
            if p["symbol"] == sym and p["magic"] == self.cfg.mt5_magic:
                return True
        return False

    def execute_mt5(self, signal: TradeSignal) -> bool:
        """Send a market order to MT5."""
        if not HAS_MT5:
            return False

        mt5_sym = self._mt5_symbol(signal.symbol)

        # Check symbol is available
        if not mt5.symbol_select(mt5_sym, True):
            log.error(f"MT5: Cannot select symbol {mt5_sym}")
            return False

        tick = mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            log.error(f"MT5: No tick data for {mt5_sym}")
            return False

        if signal.signal == SignalType.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        # Dynamically determine filling mode
        symbol_info = mt5.symbol_info(mt5_sym)
        if symbol_info is None:
            log.error(f"MT5: Cannot get symbol info for {mt5_sym}")
            return False

        # Determine filling mode based on what the symbol/broker supports
        # SYMBOL_FILLING_FOK = 1, SYMBOL_FILLING_IOC = 2, SYMBOL_FILLING_RETURN = 4
        filling_mode = mt5.ORDER_FILLING_IOC # Default safest
        
        mode_flags = symbol_info.filling_mode
        if mode_flags & 1: # SYMBOL_FILLING_FOK
            filling_mode = mt5.ORDER_FILLING_FOK
        elif mode_flags & 2: # SYMBOL_FILLING_IOC
            filling_mode = mt5.ORDER_FILLING_IOC
        elif mode_flags & 4: # SYMBOL_FILLING_RETURN
            filling_mode = mt5.ORDER_FILLING_RETURN

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       mt5_sym,
            "volume":       float(signal.lots),
            "type":         order_type,
            "price":        price,
            "sl":           signal.stop_loss,
            "tp":           signal.take_profit,
            "magic":        self.cfg.mt5_magic,
            "comment":      self.cfg.mt5_comment,
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": filling_mode,
            "deviation":    self.cfg.mt5_deviation,
        }

        # CHECK MARGIN BEFORE SENDING
        check_result = mt5.order_check(request)
        if check_result is None:
            log.error(f"MT5: order_check returned None for {mt5_sym}")
            return False

        # Note: some brokers/versions return 0 for successful order_check, 
        # while others return TRADE_RETCODE_DONE (10009).
        if check_result.retcode not in [0, mt5.TRADE_RETCODE_DONE]:
            log.warning(
                f"MT5: Insufficient margin/invalid order for {mt5_sym}. "
                f"Required: {check_result.margin}, Available: {check_result.margin_free}. "
                f"Retcode: {check_result.retcode}"
            )
            return False

        result = mt5.order_send(request)
        if result is None:
            log.error(f"MT5: order_send returned None for {mt5_sym}")
            return False

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(
                f"MT5: order failed retcode={result.retcode} "
                f"({mt5.last_error()}) for {mt5_sym}"
            )
            return False

        log.info(
            f"MT5 ORDER PLACED ► {signal.signal.value} {mt5_sym} "
            f"{signal.lots} lots @ {price:.5f} "
            f"SL={signal.stop_loss:.5f} TP={signal.take_profit:.5f}"
        )
        return True

    def execute_paper(self, signal: TradeSignal) -> bool:
        """Log a paper trade."""
        log.info(
            f"[PAPER] ► {signal.signal.value} {signal.symbol} "
            f"{signal.lots} lots @ {signal.entry_price:.5f} | "
            f"SL={signal.stop_loss:.5f} TP={signal.take_profit:.5f} | "
            f"RR={signal.rr_ratio} Confidence={signal.confidence:.0%}"
        )
        return True

    def execute(self, signal: TradeSignal, live: bool = False) -> bool:
        """Route to live or paper execution."""
        if live and HAS_MT5:
            return self.execute_mt5(signal)
        return self.execute_paper(signal)

    def update_trailing_sl(self):
        """
        Move stop losses to lock in profit as price moves in our favour.
        Called every loop iteration.
        """
        if not HAS_MT5:
            return

        for pos in self.get_open_positions():
            if pos["magic"] != self.cfg.mt5_magic:
                continue

            symbol = pos["symbol"]
            df = self.fetcher.fetch_mt5(symbol, "15m")
            if df.empty:
                continue
            df = Indicators.add_all(df, self.cfg)
            atr = float(df["atr"].iloc[-1])
            trail_dist = atr * self.cfg.trailing_sl_atr

            if pos["type"] == "BUY":
                new_sl = pos["price_cur"] - trail_dist
                if new_sl > pos["sl"] and new_sl < pos["price_cur"]:
                    self._modify_position_sl(pos["ticket"], symbol, new_sl, pos["tp"])

            elif pos["type"] == "SELL":
                new_sl = pos["price_cur"] + trail_dist
                if new_sl < pos["sl"] and new_sl > pos["price_cur"]:
                    self._modify_position_sl(pos["ticket"], symbol, new_sl, pos["tp"])

    def _modify_position_sl(self, ticket: int, symbol: str,
                             new_sl: float, tp: float):
        """Send a position modify request to MT5."""
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   symbol,
            "sl":       new_sl,
            "tp":       tp,
            "position": ticket,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"Trailing SL updated: ticket={ticket} new_sl={new_sl:.5f}")
        else:
            log.warning(f"Failed to update trailing SL for ticket={ticket}")


# -----------------------------------------------------------------------------
#  RISK MANAGER
# -----------------------------------------------------------------------------

class RiskManager:
    """
    Enforces daily loss limits and overall risk controls.
    Tracks daily PnL and pauses trading if limits are breached.
    """

    def __init__(self, cfg: BotConfig):
        self.cfg            = cfg
        self.day_start_equity: float = 0.0
        self.current_date   = datetime.utcnow().date()
        self.trading_paused = False
        self.pause_reason   = ""

    def update_equity(self, equity: float):
        """Call at start of each day (or first run) to record starting equity."""
        today = datetime.utcnow().date()
        if today != self.current_date:
            log.info(f"New trading day. Resetting daily PnL tracking. "
                     f"Previous equity: {self.day_start_equity:.2f}")
            self.day_start_equity = equity
            self.current_date     = today
            self.trading_paused   = False
            self.pause_reason     = ""
        elif self.day_start_equity == 0.0:
            self.day_start_equity = equity

    def check_daily_loss(self, equity: float) -> bool:
        """
        Returns True if trading is allowed, False if daily loss limit hit.
        """
        if self.day_start_equity <= 0:
            return True

        daily_pnl_pct = (equity - self.day_start_equity) / self.day_start_equity

        if daily_pnl_pct <= -self.cfg.max_daily_loss_pct:
            if not self.trading_paused:
                self.trading_paused = True
                self.pause_reason = (
                    f"Daily loss limit reached: {daily_pnl_pct:.1%} "
                    f"(limit: -{self.cfg.max_daily_loss_pct:.1%})"
                )
                log.warning(f"⛔ TRADING PAUSED: {self.pause_reason}")
            return False

        if self.trading_paused:
            log.info("Trading resumed.")
            self.trading_paused = False
            self.pause_reason   = ""

        return True

    def can_open_trade(self, equity: float, open_position_count: int) -> Tuple[bool, str]:
        """Final pre-trade checks."""
        if self.trading_paused:
            return False, self.pause_reason

        if not self.check_daily_loss(equity):
            return False, "Daily loss limit"

        if open_position_count >= self.cfg.max_positions:
            return False, f"Max positions ({self.cfg.max_positions}) reached"

        return True, ""


# -----------------------------------------------------------------------------
#  TRADE LOGGER & PERFORMANCE TRACKER
# -----------------------------------------------------------------------------

class TradeLogger:
    """Persists trade history and computes performance stats."""

    def __init__(self, cfg: BotConfig):
        self.cfg   = cfg
        self.trades: List[dict] = []
        self.lock = threading.RLock()
        self._load()

    def _load(self):
        with self.lock:
            if os.path.exists(self.cfg.trade_log_file):
                try:
                    with open(self.cfg.trade_log_file) as f:
                        self.trades = json.load(f)
                    log.info(f"Loaded {len(self.trades)} trades from log")
                except Exception as e:
                    log.error(f"Could not load trade log: {e}")

    def save(self):
        with self.lock:
            for attempt in range(5):
                try:
                    tmp = self.cfg.trade_log_file + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(self.trades, f, indent=2)
                    os.replace(tmp, self.cfg.trade_log_file)
                    return # Success
                except (IOError, OSError) as e:
                    if attempt < 4:
                        time.sleep(0.1)
                        continue
                    log.error(f"Could not save trade log after 5 attempts: {e}")

    def log_signal(self, signal: TradeSignal):
        with self.lock:
            entry = signal.to_dict()
            entry["type"] = "signal"
            self.trades.append(entry)
            self.save()

    def log_result(self, result: TradeResult):
        with self.lock:
            entry = result.to_dict()
            entry["type"] = "result"
            self.trades.append(entry)
            self.save()

    def get_stats(self) -> dict:
        """Compute performance statistics from logged results."""
        with self.lock:
            results = [t for t in self.trades if t.get("type") == "result"]
        
        if not results:
            return {"total_trades": 0}

        pnls    = [t["pnl_usd"] for t in results]
        wins    = [p for p in pnls if p > 0]
        losses  = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate  = len(wins) / len(pnls) if pnls else 0

        # Max drawdown
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        drawdown = peak - cumulative
        max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0

        # Profit factor
        gross_win  = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = gross_win / gross_loss

        # Expectancy per trade
        avg_win  = np.mean(wins)  if wins   else 0
        avg_loss = np.mean(losses) if losses else 0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            "total_trades":   len(results),
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate":       round(win_rate * 100, 1),
            "total_pnl_usd":  round(total_pnl, 2),
            "avg_win_usd":    round(float(avg_win), 2),
            "avg_loss_usd":   round(float(avg_loss), 2),
            "profit_factor":  round(profit_factor, 2),
            "expectancy_usd": round(float(expectancy), 2),
            "max_drawdown_usd": round(max_dd, 2),
            "best_trade_usd": round(max(pnls), 2) if pnls else 0,
            "worst_trade_usd": round(min(pnls), 2) if pnls else 0,
            "closed_trades":  results[-20:]  # Include last 20 trades
        }


# -----------------------------------------------------------------------------
#  DASHBOARD EXPORTER
# -----------------------------------------------------------------------------

class Dashboard:
    """Exports bot state to JSON for external dashboards / monitoring."""

    def __init__(self, cfg: BotConfig, logger: TradeLogger,
                 executor: TradeExecutor):
        self.cfg      = cfg
        self.logger   = logger
        self.executor = executor
        self.lock     = threading.Lock()

    def update(self, latest_signals: List[dict], equity: float = 0.0,
               risk_status: str = "OK"):
        """Write current state to dashboard JSON file."""
        with self.lock:
            positions = []
            try:
                positions = self.executor.get_open_positions()
            except Exception:
                pass

        stats = self.logger.get_stats()

        data = {
            "last_update":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "equity":         round(equity, 2),
            "risk_status":    risk_status,
            "open_positions": positions,
            "latest_signals": latest_signals[-5:],
            "performance":    stats,
            "closed_trades":  stats.get("closed_trades", []),
            "total_signals":  len([t for t in self.logger.trades
                                   if t.get("type") == "signal"]),
        }

        # MT5 account details
        if HAS_MT5 and mt5.terminal_info():
            acc = mt5.account_info()
            if acc:
                data["account"] = {
                    "login":   acc.login,
                    "balance": acc.balance,
                    "equity":  acc.equity,
                    "margin":  acc.margin,
                    "profit":  acc.profit,
                    "server":  acc.server,
                }

        for attempt in range(5):
            try:
                tmp = self.cfg.dashboard_file + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp, self.cfg.dashboard_file)
                break
            except (IOError, OSError) as e:
                if attempt < 4:
                    time.sleep(0.1)
                    continue
                log.error(f"Dashboard write error after 5 attempts: {e}")


# -----------------------------------------------------------------------------
#  BACKTESTER
# -----------------------------------------------------------------------------

class Backtester:
    """
    MTF Backtester (15m entry, 1H trend, 4H macro).
    Aligns timeframes using pandas merge_asof to ensure no lookahead bias.
    Iterates over 15m bars and evaluates SL/TP sequentially.
    """

    def __init__(self, cfg: BotConfig):
        self.cfg     = cfg
        self.fetcher = DataFetcher(cfg)

    def run(self, symbol: str, initial_capital: float = 10_000.0) -> dict:
        """Run multi-timeframe backtest on 15m data."""
        log.info(f"Backtesting {symbol} (MTF) | Capital=${initial_capital:,.0f}")

        # 1. Fetch data
        # Note: Yahoo Finance allows 60 days of 15m data
        df15 = self.fetcher.fetch_yf(symbol, self.cfg.entry_tf) # 15m
        df1h = self.fetcher.fetch_yf(symbol, self.cfg.trend_tf) # 1h
        
        if df15.empty or df1h.empty:
            log.error(f"Backtest: insufficient data for {symbol}")
            return {}

        # 2. Add indicators to each timeframe separately
        df1h = Indicators.add_all(df1h, self.cfg)
        
        # Create 4H macro timeframe from 1H data
        df4h = df1h.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()
        df4h = Indicators.add_all(df4h, self.cfg)

        # 3. Prefix columns to avoid collisions during merge
        df1h = df1h.add_prefix("h1_")
        df4h = df4h.add_prefix("h4_")

        # 4. Align timeframes without lookahead bias
        # We must shift higher timeframe data so it's only available AFTER the candle closes.
        # YF Datetime is the start of the candle.
        df1h_available = df1h.copy()
        df1h_available.index = df1h_available.index + pd.Timedelta(hours=1)
        
        df4h_available = df4h.copy()
        df4h_available.index = df4h_available.index + pd.Timedelta(hours=4)

        # Merge using merge_asof (ensures we pick the latest available H1/H4 candle)
        # Normalize index types to avoid MergeError (s vs us vs ns)
        df15.index = pd.to_datetime(df15.index, utc=True).astype('datetime64[ns, UTC]')
        df1h_available.index = pd.to_datetime(df1h_available.index, utc=True).astype('datetime64[ns, UTC]')
        df4h_available.index = pd.to_datetime(df4h_available.index, utc=True).astype('datetime64[ns, UTC]')

        df15.sort_index(inplace=True)
        df1h_available.sort_index(inplace=True)
        df4h_available.sort_index(inplace=True)

        df = pd.merge_asof(
            df15, df1h_available,
            left_index=True, right_index=True,
            direction="backward"
        )
        df = pd.merge_asof(
            df, df4h_available,
            left_index=True, right_index=True,
            direction="backward"
        )

        # 5. Add 15m indicators to the merged master frame
        df = Indicators.add_all(df, self.cfg)
        
        # Remove bars where we don't have enough MTF context yet
        df.dropna(subset=["h1_Close", "h4_Close"], inplace=True)
        
        log.info(f"Backtest: {len(df)} 15m bars aligned for simulation.")

        # 6. Simulation Loop
        capital      = initial_capital
        trades       = []
        position     = None
        equity_curve = [capital]

        # Iterate through 15m bars
        for i in range(1, len(df)):
            bar  = df.iloc[i]
            prev = df.iloc[i-1]
            time = df.index[i]

            # -- Manage open position (Check SL/TP sequentially) ------------
            if position is not None:
                high = bar["High"]
                low  = bar["Low"]
                closed = False
                exit_p = 0.0
                outcome = ""

                if position["side"] == "BUY":
                    # Conservative: Check SL first if both hit in same candle
                    if low <= position["sl"]:
                        exit_p = position["sl"]
                        outcome = "LOSS"
                        closed = True
                    elif high >= position["tp"]:
                        exit_p = position["tp"]
                        outcome = "WIN"
                        closed = True
                else: # SELL
                    if high >= position["sl"]:
                        exit_p = position["sl"]
                        outcome = "LOSS"
                        closed = True
                    elif low <= position["tp"]:
                        exit_p = position["tp"]
                        outcome = "WIN"
                        closed = True

                if closed:
                    pnl_pts = (exit_p - position["entry"]) if position["side"] == "BUY" \
                              else (position["entry"] - exit_p)
                    
                    # Pip math
                    pip_size = 0.01 if "JPY" in symbol else 0.0001
                    pip_val  = 9.0 if "JPY" in symbol else 10.0
                    pnl_pips = pnl_pts / pip_size
                    pnl_usd  = pnl_pips * pip_val * position["lots"]
                    
                    capital += pnl_usd
                    trades.append({
                        "symbol":    symbol,
                        "entry_time": position["time"],
                        "exit_time":  time.isoformat(),
                        "side":       position["side"],
                        "entry":      position["entry"],
                        "exit":       exit_p,
                        "lots":       position["lots"],
                        "pnl_pips":   round(pnl_pips, 1),
                        "pnl_usd":    round(pnl_usd, 2),
                        "outcome":    outcome,
                    })
                    position = None

            # -- Check for new entry signal --------------------------------
            if position is None:
                # Signal engine needs current + previous to check crossovers
                window = df.iloc[i-1:i+1] 
                sig_type, score = self._backtest_signal(window)

                if sig_type and score >= self.cfg.min_confluence:
                    entry = float(bar["Close"])
                    atr   = float(bar["atr"])
                    
                    # SL/TP calculation (similar to SignalEngine)
                    sl_dist = atr * self.cfg.sl_atr_multiplier
                    tp_dist = sl_dist * self.cfg.tp_rr_ratio
                    
                    if sig_type == "BUY":
                        sl = entry - sl_dist
                        tp = entry + tp_dist
                    else:
                        sl = entry + sl_dist
                        tp = entry - tp_dist

                    # Risk-based lot sizing
                    pip_size = 0.01 if "JPY" in symbol else 0.0001
                    pip_val  = 9.0 if "JPY" in symbol else 10.0
                    risk_usd = capital * self.cfg.risk_per_trade_pct
                    sl_pips  = sl_dist / pip_size
                    lots     = risk_usd / (sl_pips * pip_val) if sl_pips > 0 else self.cfg.lot_min
                    
                    lots = round(lots / self.cfg.lot_step) * self.cfg.lot_step
                    lots = max(self.cfg.lot_min, min(lots, self.cfg.lot_max))

                    position = {
                        "side":  sig_type,
                        "entry": entry,
                        "sl":    sl,
                        "tp":    tp,
                        "lots":  lots,
                        "time":  time.isoformat(),
                    }

            equity_curve.append(capital)

        return self._stats(trades, initial_capital, capital, equity_curve)

    def _backtest_signal(self, window: pd.DataFrame) -> Tuple[Optional[str], int]:
        """
        MTF Signal Logic for Backtesting.
        Matches live SignalEngine confluence scoring.
        """
        if len(window) < 2:
            return None, 0

        cur  = window.iloc[-1]
        prev = window.iloc[-2]

        # 1. Macro Bias (4H merged)
        macro = None
        if cur["h4_Close"] > cur["h4_ema_macro"]: macro = "UP"
        elif cur["h4_Close"] < cur["h4_ema_macro"]: macro = "DOWN"

        # 2. Trend Filter (1H merged)
        trend = None
        if cur["h1_Close"] > cur["h1_ema_trend"] and cur["h1_ema_fast"] > cur["h1_ema_slow"]:
            trend = "UP"
        elif cur["h1_Close"] < cur["h1_ema_trend"] and cur["h1_ema_fast"] < cur["h1_ema_slow"]:
            trend = "DOWN"

        if trend is None:
            return None, 0
            
        # Macro must agree with trend
        if macro is not None and macro != trend:
            return None, 0

        # 3. Confluence Scoring (15m entry)
        score = 0
        rsi   = cur["rsi"]
        mh    = cur["macd_hist"]
        sk    = cur["stoch_k"]
        sd    = cur["stoch_d"]
        bb    = cur["bb_position"]
        adx   = cur["adx"]

        if trend == "UP":
            if 45 <= rsi <= 70: score += 1
            if mh > 0 or (prev["macd_hist"] < 0 < mh): score += 1
            if sk < 80 and sk > sd: score += 1
            if bb < 0.6: score += 1
            if adx > 20 and cur["plus_di"] > cur["minus_di"]: score += 1
        else: # DOWN
            if 30 <= rsi <= 55: score += 1
            if mh < 0 or (prev["macd_hist"] > 0 > mh): score += 1
            if sk > 20 and sk < sd: score += 1
            if bb > 0.4: score += 1
            if adx > 20 and cur["minus_di"] > cur["plus_di"]: score += 1

        return (trend if score >= self.cfg.min_confluence else None), score

    def _stats(self, trades: list, initial: float, final: float,
               equity_curve: list) -> dict:
        if not trades:
            log.warning("Backtest: no trades generated")
            return {"total_trades": 0, "note": "No trades — check confluence settings"}

        pnls   = [t["pnl_usd"] for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd   = (peak - eq) / peak
        max_dd_pct = float(np.max(dd)) * 100

        gross_win  = sum(wins)   if wins   else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        pf = gross_win / gross_loss

        stats = {
            "symbol":            "N/A",
            "initial_capital":   round(initial, 2),
            "final_capital":     round(final, 2),
            "total_return_pct":  round((final - initial) / initial * 100, 2),
            "total_trades":      len(trades),
            "wins":              len(wins),
            "losses":            len(losses),
            "win_rate_pct":      round(len(wins) / len(trades) * 100, 1),
            "profit_factor":     round(pf, 2),
            "avg_win_usd":       round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss_usd":      round(float(np.mean(losses)), 2) if losses else 0,
            "best_trade_usd":    round(max(pnls), 2),
            "worst_trade_usd":   round(min(pnls), 2),
            "max_drawdown_pct":  round(max_dd_pct, 2),
            "expectancy_usd":    round(float(np.mean(pnls)), 2),
        }

        log.info("=" * 55)
        log.info("  BACKTEST RESULTS")
        log.info("=" * 55)
        for k, v in stats.items():
            log.info(f"  {k:<25}: {v}")
        log.info("=" * 55)
        return stats


# -----------------------------------------------------------------------------
#  MT5 CONNECTION MANAGER
# -----------------------------------------------------------------------------

class MT5Manager:
    """Handles MT5 initialization and reconnection."""

    def __init__(self, cfg: BotConfig):
        self.cfg       = cfg
        self.connected = False

    def _load_config_file(self) -> Dict[str, str]:
        """Lazy load credentials from the 'config' file if present."""
        creds = {}
        try:
            if os.path.exists("config"):
                with open("config", "r") as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line:
                            key, val = line.split("=", 1)
                            creds[key.strip().upper()] = val.strip().strip('"')
        except Exception as e:
            log.warning(f"Failed to read 'config' file: {e}")
        return creds

    def connect(self) -> bool:
        if not HAS_MT5:
            log.error("MetaTrader5 package not installed")
            return False

        # Try environment variables first
        login_str = os.environ.get("MT5_LOGIN")
        password  = os.environ.get("MT5_PASSWORD")
        server    = os.environ.get("MT5_SERVER")

        # Fallback to config file
        if not login_str or not password or not server:
            creds = self._load_config_file()
            login_str = login_str or creds.get("LOGIN")
            password  = password  or creds.get("PASSWORD")
            server    = server    or creds.get("SERVER")

        if not login_str:
            log.error("MT5_LOGIN not set (env var or config file)")
            return False

        try:
            login = int(login_str)
        except ValueError:
            log.error(f"MT5_LOGIN must be an integer, got '{login_str}'")
            return False

        if not mt5.initialize():
            log.error(f"mt5.initialize() failed: {mt5.last_error()}")
            return False

        if not mt5.login(login=login, password=password, server=server):
            log.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

        acc = mt5.account_info()
        log.info(
            f"MT5 connected ► Login={acc.login} Balance={acc.balance:.2f} "
            f"Equity={acc.equity:.2f} Server={acc.server}"
        )
        self.connected = True
        return True

    def ensure_connected(self) -> bool:
        """Reconnect if terminal disconnected."""
        if not HAS_MT5:
            return False
        if mt5.terminal_info() is None:
            log.warning("MT5 disconnected. Reconnecting...")
            self.connected = False
            return self.connect()
        return True

    def get_equity(self) -> float:
        if not HAS_MT5 or not self.ensure_connected():
            return 10.0  # Paper default
        acc = mt5.account_info()
        return float(acc.equity) if acc else 10.0

    def disconnect(self):
        if HAS_MT5:
            mt5.shutdown()
            self.connected = False
            log.info("MT5 disconnected")


# -----------------------------------------------------------------------------
#  MAIN TRADING BOT
# -----------------------------------------------------------------------------

class HaldaneTradingBot:
    """
    The main orchestrator. Ties all components together.

    Components:
      DataFetcher    — OHLCV data from YF or MT5
      SignalEngine   — Multi-timeframe confluence signals
      MLFilter       — Optional probability filter
      TradeExecutor  — Order placement and trailing SL
      RiskManager    — Daily loss limits, position sizing checks
      TradeLogger    — Persistent trade history + stats
      Dashboard      — JSON status export
      MT5Manager     — Connection management
      Backtester     — Historical simulation
    """

    def __init__(self, cfg: Optional[BotConfig] = None):
        self.cfg      = cfg or BotConfig()
        self.fetcher  = DataFetcher(self.cfg)
        self.engine   = SignalEngine(self.cfg, self.fetcher)
        self.ml       = MLFilter(self.cfg)
        self.executor = TradeExecutor(self.cfg, self.fetcher)
        self.risk_mgr = RiskManager(self.cfg)
        self.logger   = TradeLogger(self.cfg)
        self.mt5_mgr  = MT5Manager(self.cfg)
        self.dashboard = Dashboard(self.cfg, self.logger, self.executor)

        self.recent_signals: List[dict] = []
        self.running = False

        log.info("HaldaneBot v3.0 initialized")
        log.info(f"  Symbols: {self.cfg.symbols}")
        log.info(f"  Live trading: {self.cfg.live_trading}")
        log.info(f"  ML filter: {self.cfg.use_ml}")
        log.info(f"  Risk per trade: {self.cfg.risk_per_trade_pct:.0%}")
        log.info(f"  Max positions: {self.cfg.max_positions}")
        log.info(f"  Min confluence: {self.cfg.min_confluence}/5")

    # -- Setup -----------------------------------------------------------------
    def setup(self):
        """Initialize connections and optionally train ML models."""
        if self.cfg.live_trading:
            if not self.mt5_mgr.connect():
                log.error("Could not connect to MT5. Falling back to paper mode.")
                self.cfg.live_trading = False

        if self.cfg.use_ml and HAS_SKL:
            log.info(f"Training ML models on {self.cfg.entry_tf} data...")
            for symbol in self.cfg.symbols:
                try:
                    # ML must be trained on the same timeframe used for entries
                    df = self.fetcher.fetch_yf(symbol, self.cfg.entry_tf)
                    if not df.empty:
                        df = Indicators.add_all(df, self.cfg)
                        self.ml.train(symbol, df)
                except Exception as e:
                    log.error(f"ML training failed for {symbol}: {e}")
        
        # Initial history sync
        if self.cfg.live_trading:
            self.sync_historical_trades()

    def sync_historical_trades(self):
        """Fetch recently closed trades from MT5 history and update logger."""
        if not HAS_MT5 or not self.cfg.live_trading:
            return

        try:
            # Sync last 7 days of history
            from_date = datetime.now() - timedelta(days=7)
            to_date = datetime.now()
            
            deals = mt5.history_deals_get(from_date, to_date)
            if deals is None:
                return

            # Get tickets we already have in our log to avoid duplicates
            with self.logger.lock:
                existing_tickets = {t.get("ticket") for t in self.logger.trades if t.get("type") == "result"}
                
                new_trades_count = 0
                for d in deals:
                    # We only care about deals that close a position (Entry OUT = 1 or IN/OUT = 2)
                    # and have a non-zero profit (to exclude deposits/withdrawals)
                    if d.entry in [1, 2] and d.ticket not in existing_tickets and d.profit != 0:
                        # Create a TradeResult-like dict
                        res = {
                            "type": "result",
                            "ticket": d.ticket,
                            "symbol": d.symbol,
                            "signal": "BUY" if d.type == 1 else "SELL", 
                            "entry_price": d.price, 
                            "exit_price": d.price,
                            "pnl_usd": d.profit,
                            "outcome": "WIN" if d.profit > 0 else "LOSS",
                            "exit_time": datetime.fromtimestamp(d.time).isoformat(),
                            "magic": d.magic,
                            "comment": d.comment
                        }
                        self.logger.trades.append(res)
                        existing_tickets.add(d.ticket)
                        new_trades_count += 1
                
                if new_trades_count > 0:
                    self.logger.save()
                    log.info(f"Sync: Added {new_trades_count} closed trades from MT5 history")

        except Exception as e:
            log.error(f"Error syncing history: {e}")

    # -- Single symbol processing ----------------------------------------------
    # -- Signal processing and execution -------------------------------------
    def process_trade(self, signal: TradeSignal, equity: float,
                      open_count: int, max_pos_limit: int) -> bool:
        """
        Final verification and execution of a trade signal.
        """
        # Risk check with dynamic limit
        if open_count >= max_pos_limit:
            log.debug(f"{signal.symbol}: trade limit reached ({open_count}/{max_pos_limit}). Skipping.")
            return False

        use_mt5 = self.cfg.live_trading and HAS_MT5

        # Already have position on this symbol?
        if use_mt5 and self.executor.has_position_for(signal.symbol):
            log.debug(f"{signal.symbol}: position already open")
            return False

        # ML probability filter
        if self.cfg.use_ml and signal.symbol in self.ml.models:
            df15 = self.fetcher.fetch(signal.symbol, self.cfg.entry_tf, use_mt5)
            if not df15.empty:
                df15 = Indicators.add_all(df15, self.cfg)
                ml_prob = self.ml.predict_probability(signal.symbol, df15, signal.trend)
                signal.ml_probability = ml_prob
                if ml_prob < self.cfg.ml_min_probability:
                    log.info(
                        f"{signal.symbol}: ML prob {ml_prob:.2f} < "
                        f"{self.cfg.ml_min_probability}. Skipping."
                    )
                    return False

        # Log and execute
        self.logger.log_signal(signal)
        self.recent_signals.append(signal.to_dict())

        executed = self.executor.execute(signal, live=self.cfg.live_trading)
        return executed

    def process_symbol(self, symbol: str, equity: float,
                       open_count: int) -> Optional[TradeSignal]:
        """
        Legacy method kept for compatibility.
        """
        use_mt5 = self.cfg.live_trading and HAS_MT5
        sig = self.engine.generate(symbol, equity, use_mt5)
        if sig:
            if self.process_trade(sig, equity, open_count, self.cfg.max_positions):
                return sig
        return None

    # -- Main loops -------------------------------------------------------------
    async def run(self):
        """Asynchronous live / paper trading loop."""
        self.running = True
        self.setup()

        log.info("=" * 55)
        log.info("  HALDANE BOT v3 — ASYNC TRADING SYSTEM STARTED")
        log.info("=" * 55)

        # Create concurrent tasks
        tasks = [
            asyncio.create_task(self._main_strategy_loop()),
            asyncio.create_task(self._trailing_stop_loop()),
            asyncio.create_task(self._dashboard_loop()),
            asyncio.create_task(self._history_sync_loop()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            log.info("Tasks cancelled. Shutting down...")
        except Exception as e:
            log.error(f"Critical error in main task aggregator: {e}")
            log.debug(traceback.format_exc())
        finally:
            self.shutdown()

    async def _main_strategy_loop(self):
        """Loop for processing symbols and generating signals."""
        loop_count = 0
        while self.running:
            try:
                loop_count += 1
                log.info(f"\n--- Strategy Cycle #{loop_count} | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ---")

                if self.cfg.live_trading:
                    await asyncio.to_thread(self.mt5_mgr.ensure_connected)

                equity = await asyncio.to_thread(self.mt5_mgr.get_equity)
                self.risk_mgr.update_equity(equity)

                if not self.risk_mgr.check_daily_loss(equity):
                    log.warning("⛔ Daily loss limit active. Skipping strategy cycle.")
                    await asyncio.sleep(self.cfg.interval_seconds)
                    continue

                positions = await asyncio.to_thread(self.executor.get_open_positions)
                open_count = len([p for p in positions if p.get("magic") == self.cfg.mt5_magic])
                log.info(f"Open positions: {open_count}/{self.cfg.max_positions} | Equity: ${equity:.2f}")

                # Process symbols concurrently
                symbol_tasks = [
                    self._process_symbol_async(symbol, equity)
                    for symbol in self.cfg.symbols
                ]
                await asyncio.gather(*symbol_tasks)

                log.info(f"Strategy cycle complete. Waiting {self.cfg.interval_seconds}s...")
                await asyncio.sleep(self.cfg.interval_seconds)

            except Exception as e:
                log.error(f"Error in strategy loop: {e}")
                log.debug(traceback.format_exc())
                await asyncio.sleep(30)

    async def _process_symbol_async(self, symbol: str, equity: float):
        """Asynchronously process a single symbol."""
        try:
            use_mt5 = self.cfg.live_trading and HAS_MT5
            
            # Offload blocking data fetching and signal generation to a thread
            sig = await asyncio.to_thread(self.engine.generate, symbol, equity, use_mt5)
            
            if sig:
                # Re-check open count before execution
                positions = await asyncio.to_thread(self.executor.get_open_positions)
                open_count = len([p for p in positions if p.get("magic") == self.cfg.mt5_magic])
                
                # Dynamic max positions: if a signal has very high confidence, we temporarily allow more
                current_max = self.cfg.max_positions
                if sig.confidence >= 0.4:
                    log.info(f"🔥 High confidence detected for {symbol} ({sig.confidence}). Boosting max trades to {self.cfg.max_positions + 1}")
                    current_max = self.cfg.max_positions + 1
                
                if open_count < current_max:
                    await asyncio.to_thread(self.process_trade, sig, equity, open_count, current_max)
                else:
                    log.debug(f"{symbol}: Trade limit reached, signal ignored.")
        except Exception as e:
            log.error(f"Error processing {symbol} (async): {e}")

    async def _trailing_stop_loop(self):
        """Concurrent task for updating trailing stops."""
        if not self.cfg.live_trading:
            return

        while self.running:
            try:
                await asyncio.to_thread(self.executor.update_trailing_sl)
                # Run more frequently than the strategy loop for tighter trailing
                await asyncio.sleep(20) 
            except Exception as e:
                log.error(f"Error in trailing stop loop: {e}")
                await asyncio.sleep(10)

    async def _dashboard_loop(self):
        """Concurrent task for updating the dashboard."""
        while self.running:
            try:
                equity = await asyncio.to_thread(self.mt5_mgr.get_equity)
                risk_status = "PAUSED - Daily Loss" if self.risk_mgr.trading_paused else "OK"
                
                await asyncio.to_thread(
                    self.dashboard.update,
                    self.recent_signals[-10:],
                    equity=equity,
                    risk_status=risk_status
                )
                
                stats = self.logger.get_stats()
                if stats.get("total_trades", 0) > 0:
                    log.info(
                        f"Status | Trades={stats['total_trades']} "
                        f"Win={stats.get('win_rate', 0)}% "
                        f"PnL=${stats.get('total_pnl_usd', 0):.2f}"
                    )
                
                await asyncio.sleep(max(30, self.cfg.interval_seconds // 2))
            except Exception as e:
                log.error(f"Error in dashboard loop: {e}")
                await asyncio.sleep(10)

    async def _history_sync_loop(self):
        """Concurrent task for syncing trade history."""
        if not self.cfg.live_trading:
            return

        while self.running:
            try:
                await asyncio.to_thread(self.sync_historical_trades)
                await asyncio.sleep(300) # Sync every 5 minutes
            except Exception as e:
                log.error(f"Error in history sync loop: {e}")
                await asyncio.sleep(60)

    def shutdown(self):
        """Clean shutdown."""
        log.info("Shutting down HaldaneBot…")
        self.logger.save()
        if self.cfg.live_trading:
            self.mt5_mgr.disconnect()
        log.info("Shutdown complete.")

    # -- Convenience methods ---------------------------------------------------
    def get_signals_now(self) -> List[dict]:
        """Get current signals for all symbols without executing trades."""
        signals = []
        use_mt5 = self.cfg.live_trading and HAS_MT5
        equity  = self.mt5_mgr.get_equity()

        for symbol in self.cfg.symbols:
            sig = self.engine.generate(symbol, equity, use_mt5)
            if sig:
                signals.append(sig.to_dict())
            else:
                signals.append({"symbol": symbol, "signal": "HOLD",
                                 "reason": "No signal"})
        return signals

    def run_backtest(self, symbol: Optional[str] = None,
                     initial_capital: float = 10_000.0) -> dict:
        """Run backtest on one symbol or all configured symbols."""
        bt = Backtester(self.cfg)
        symbols = [symbol] if symbol else self.cfg.symbols
        all_results = {}
        for sym in symbols:
            result = bt.run(sym, initial_capital=initial_capital)
            result["symbol"] = sym
            all_results[sym] = result
        return all_results


# -----------------------------------------------------------------------------
#  ENTRY POINT
# -----------------------------------------------------------------------------

def main():
    """Main entry point. Parses arguments and runs the appropriate mode."""

    # -- Configuration ---------------------------------------------------------
    cfg = BotConfig(
        symbols=[
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        ],
        mt5_symbol_suffix="m",           # If your broker uses EURUSDm, etc.
        trend_tf="1h",
        entry_tf="15m",
        confirm_tf="4h",

        # Indicators
        ema_trend=50,
        ema_macro=200,
        rsi_period=14,
        atr_period=14,

        # Signal quality
        min_confluence=3,                # Need 3/5 indicators to agree

        # Risk management
        risk_per_trade_pct=0.01,         # 1% per trade — NEVER go higher
        sl_atr_multiplier=1.5,
        tp_rr_ratio=2.0,
        min_rr_ratio=1.8,
        max_positions=3,
        max_daily_loss_pct=0.03,         # Stop trading after 3% daily loss

        # Execution
        live_trading=True,               # ← Set True to trade on MT5
        lot_min=0.01,
        lot_max=0.10,                    # Keep small on a $10 account
        spread_max_pips=1.5,
        symbol_spread_limits={
            "EURUSD": 1.0,
            "GBPUSD": 1.2,
            "USDJPY": 1.2,
            "GBPJPY": 2.5,
        },

        # Sessions
        trade_sessions=[(7, 16), (13, 22)],

        # ML (optional, needs scikit-learn)
        use_ml=False,
        ml_min_probability=0.60,

        interval_seconds=60,
    )

    # -- Argument parsing ------------------------------------------------------
    args = sys.argv[1:]

    if "--backtest" in args:
        # -- Backtest mode ----------------------------------------------------
        print("\n" + "=" * 60)
        print("  HALDANE BOT v3 — BACKTEST MODE")
        print("=" * 60)

        bot = HaldaneTradingBot(cfg)
        capital = 10_000.0   # Use $10k for backtest to get meaningful numbers
        results = bot.run_backtest(initial_capital=capital)

        for symbol, res in results.items():
            print(f"\n-- {symbol} ------------------------------")
            for k, v in res.items():
                print(f"  {k:<28}: {v}")

    elif "--signal" in args:
        # -- Signal check mode ------------------------------------------------
        print("\n" + "=" * 60)
        print("  HALDANE BOT v3 — CURRENT SIGNALS")
        print("=" * 60)

        bot = HaldaneTradingBot(cfg)
        signals = bot.get_signals_now()

        for sig in signals:
            symbol = sig.get("symbol", "?")
            s      = sig.get("signal", "HOLD")
            if hasattr(s, "value"):
                s = s.value
            entry  = sig.get("entry_price", "—")
            sl     = sig.get("stop_loss", "—")
            tp     = sig.get("take_profit", "—")
            score  = sig.get("confluence_score", "—")
            reason = sig.get("reason", sig.get("reason", "No signal"))
            rr     = sig.get("rr_ratio", "—")

            marker = "►" if s in ("BUY", "SELL") else "·"
            print(f"\n  {marker} {symbol}: {s}")
            if s in ("BUY", "SELL"):
                print(f"    Entry={entry}  SL={sl}  TP={tp}  RR={rr}")
                print(f"    Score={score}/5")
                print(f"    {reason}")

    elif "--stats" in args:
        # -- Performance stats mode -------------------------------------------
        bot = HaldaneTradingBot(cfg)
        stats = bot.logger.get_stats()
        print("\n" + "=" * 60)
        print("  PERFORMANCE STATISTICS")
        print("=" * 60)
        for k, v in stats.items():
            print(f"  {k:<28}: {v}")

    else:
        # -- Live / paper trading mode ----------------------------------------
        print("\n" + "=" * 60)
        print("  HALDANE BOT v3 — LIVE TRADING")
        print(f"  Mode: {'LIVE (MT5)' if cfg.live_trading else 'PAPER'}")
        print("=" * 60)
        print("\n  MT5 env vars needed for live mode:")
        print("    MT5_LOGIN=<account>")
        print("    MT5_PASSWORD=<password>")
        print("    MT5_SERVER=<broker server>")
        print("\n  Set cfg.live_trading=True in main() to go live.\n")

        bot = HaldaneTradingBot(cfg)
        try:
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            log.info("Interrupted by user, shutting down...")


if __name__ == "__main__":
    main()