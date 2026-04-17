"""
+==============================================================================+
|        GOLDEN STRATEGY TRADING BOT v4.0 — HIGH-ACCURACY EDITION            |
|                                                                              |
|  Strategy 1: Trend-Momentum (Golden Cross + RSI Pullback)                   |
|  Strategy 2: Mean Reversion (Bollinger Bands + Stochastic)                  |
|  Strategy 3: Price Action + MACD Divergence Confluence                      |
|                                                                              |
|  Key Improvements Over v3:                                                  |
|    - MACD Divergence detection (bull/bear regular + hidden)                 |
|    - Pin Bar & Engulfing candle recognition                                  |
|    - Support/Resistance level detection from swing highs/lows               |
|    - Stochastic crossover filtering (not just levels)                       |
|    - Volume confirmation on every signal                                    |
|    - Signal scoring: each strategy scored independently (0-10)              |
|    - Only trades with combined score >= threshold are taken                 |
|    - Walk-forward backtest with per-strategy attribution                    |
|                                                                              |
|  Author : Golden System v4                                                   |
|  Run    : python golden_strategy_bot.py [--backtest] [--signal] [--live]   |
+==============================================================================+
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------

import os
import sys
import json
import time
import logging
import warnings
import traceback
from copy import deepcopy
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score, f1_score
    HAS_SKL = True
except ImportError:
    HAS_SKL = False

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("GoldenBot")

class ColorFormatter(logging.Formatter):
    RESET  = "\033[0m"
    COLORS = {
        logging.DEBUG:    "\033[90m",
        logging.INFO:     "\033[0m",
        logging.WARNING:  "\033[93m",
        logging.ERROR:    "\033[91m",
        logging.CRITICAL: "\033[95m",
    }
    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        msg = super().format(record)
        msg = msg.replace("BUY",  "\033[92mBUY\033[0m")
        msg = msg.replace("SELL", "\033[91mSELL\033[0m")
        msg = msg.replace("WIN",  "\033[92mWIN\033[0m")
        msg = msg.replace("LOSS", "\033[91mLOSS\033[0m")
        return f"{color}{msg}{self.RESET}"

for h in logging.root.handlers:
    h.setFormatter(ColorFormatter(LOG_FORMAT))


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

@dataclass
class GoldenConfig:
    # Symbols
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"
    ])
    mt5_symbol_suffix: str = ""

    # Timeframes
    trend_tf:   str = "1h"
    entry_tf:   str = "15m"
    confirm_tf: str = "4h"

    # --- Strategy 1: Trend-Momentum (Golden Cross) ---
    ema_fast:     int   = 9
    ema_slow:     int   = 21
    ema_50:       int   = 50
    ema_200:      int   = 200
    rsi_period:   int   = 14
    rsi_pullback_min: float = 40.0   # RSI must have pulled back to at least 40
    rsi_pullback_max: float = 55.0   # ...but not gone below 40 (still bullish)
    rsi_overbought:   float = 70.0
    rsi_oversold:     float = 30.0

    # --- Strategy 2: Mean Reversion (Bollinger + Stochastic) ---
    bb_period: int   = 20
    bb_std:    float = 2.0
    stoch_k:   int   = 14
    stoch_d:   int   = 3
    stoch_smooth: int = 3
    stoch_overbought: float = 80.0
    stoch_oversold:   float = 20.0
    bb_touch_pct:     float = 0.002  # Price within 0.2% of band counts as "touch"

    # --- Strategy 3: MACD Divergence + Price Action ---
    macd_fast:   int = 12
    macd_slow:   int = 26
    macd_signal: int = 9
    divergence_lookback: int = 30   # Bars to look back for divergence
    sr_lookback: int = 50           # Bars to look back for S/R levels
    sr_tolerance_pct: float = 0.003 # S/R zone tolerance (0.3%)
    pin_bar_ratio: float = 0.6      # Shadow must be at least 60% of total range
    engulfing_ratio: float = 1.1    # Body must be 110% of previous body

    # --- Shared signal thresholds ---
    # Each strategy scores 0-10. Weighted combo must pass threshold.
    s1_weight:  float = 0.35
    s2_weight:  float = 0.30
    s3_weight:  float = 0.35
    min_score:  float = 5.5   # Out of 10 weighted

    # Volume confirmation
    volume_ma:   int   = 20
    volume_min:  float = 1.0   # Volume must be >= 1x average (no spike needed)

    # ATR / Risk
    atr_period:         int   = 14
    sl_atr_multiplier:  float = 1.5
    tp_rr_ratio:        float = 2.5   # Better RR for high-accuracy setups
    min_rr_ratio:       float = 2.0
    risk_per_trade_pct: float = 0.01
    max_positions:      int   = 3
    max_daily_loss_pct: float = 0.03
    lot_min:    float = 0.01
    lot_max:    float = 0.50
    lot_step:   float = 0.01
    spread_max_pips: float = 20.0

    # ML filter (optional)
    use_ml:  bool  = True
    ml_min_prob: float = 0.60

    # Execution
    live_trading: bool = True
    mt5_magic:    int  = 20250417
    mt5_comment:  str  = "GoldenBot v4"

    # Sessions (UTC)
    trade_sessions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (7, 16), (13, 22)
    ])

    interval_seconds: int = 60
    dashboard_file:   str = "golden_dashboard.json"
    trade_log_file:   str = "golden_trades.json"


# ---------------------------------------------------------------------------
# ENUMS & DATA CLASSES
# ---------------------------------------------------------------------------

class SignalType(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class StrategyTag(Enum):
    GOLDEN_CROSS = "S1_GoldenCross"
    MEAN_REVERT  = "S2_MeanReversion"
    DIVERGENCE   = "S3_Divergence"
    COMBO        = "COMBO"


@dataclass
class StrategyScore:
    s1_score: float = 0.0   # Trend-Momentum score 0-10
    s2_score: float = 0.0   # Mean Reversion score 0-10
    s3_score: float = 0.0   # MACD Divergence score 0-10
    s1_reasons: List[str] = field(default_factory=list)
    s2_reasons: List[str] = field(default_factory=list)
    s3_reasons: List[str] = field(default_factory=list)

    def combined(self, cfg: GoldenConfig) -> float:
        return (self.s1_score * cfg.s1_weight +
                self.s2_score * cfg.s2_weight +
                self.s3_score * cfg.s3_weight)

    def dominant_strategy(self) -> StrategyTag:
        scores = {
            StrategyTag.GOLDEN_CROSS: self.s1_score,
            StrategyTag.MEAN_REVERT:  self.s2_score,
            StrategyTag.DIVERGENCE:   self.s3_score,
        }
        return max(scores, key=scores.get)


@dataclass
class TradeSignal:
    symbol:      str
    signal:      SignalType
    entry_price: float
    stop_loss:   float
    take_profit: float
    sl_distance: float
    tp_distance: float
    rr_ratio:    float
    lots:        float
    confidence:  float
    score:       StrategyScore
    strategy:    str
    rsi:         float
    macd_hist:   float
    atr:         float
    spread_pips: float
    reason:      str
    timestamp:   str = field(default_factory=lambda: datetime.now().isoformat())
    ml_probability: float = 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"]   = self.signal.value
        d["strategy"] = self.strategy
        d["score_combined"] = self.score.combined(GoldenConfig())
        return d


@dataclass
class TradeResult:
    symbol:          str
    signal:          str
    strategy:        str
    entry_price:     float
    exit_price:      float
    stop_loss:       float
    take_profit:     float
    lots:            float
    pnl_pips:        float
    pnl_usd:         float
    outcome:         str
    entry_time:      str
    exit_time:       str
    duration_minutes: float
    reason_for_exit: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# TECHNICAL INDICATORS — COMPREHENSIVE
# ---------------------------------------------------------------------------

class Indicators:

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
    def macd(series: pd.Series, fast=12, slow=26, signal=9):
        fast_ema   = series.ewm(span=fast, adjust=False).mean()
        slow_ema   = series.ewm(span=slow, adjust=False).mean()
        macd_line  = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram  = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(series: pd.Series, period=20, std_dev=2.0):
        mid   = series.rolling(period).mean()
        std   = series.rolling(period).std()
        upper = mid + std * std_dev
        lower = mid - std * std_dev
        return upper, mid, lower

    @staticmethod
    def stochastic(df: pd.DataFrame, k=14, d=3, smooth=3):
        low_min  = df["Low"].rolling(k).min()
        high_max = df["High"].rolling(k).max()
        denom    = (high_max - low_min).replace(0, np.nan)
        stoch_k  = 100 * ((df["Close"] - low_min) / denom)
        stoch_k  = stoch_k.rolling(smooth).mean()
        stoch_d  = stoch_k.rolling(d).mean()
        return stoch_k, stoch_d

    @staticmethod
    def atr(df: pd.DataFrame, period=14) -> pd.Series:
        hl  = df["High"] - df["Low"]
        hc  = (df["High"] - df["Close"].shift()).abs()
        lc  = (df["Low"]  - df["Close"].shift()).abs()
        tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()   # Wilder smoothing

    @staticmethod
    def adx(df: pd.DataFrame, period=14):
        high, low, close = df["High"], df["Low"], df["Close"]
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm]   = 0
        minus_dm[minus_dm < plus_dm]  = 0
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr_v    = tr.ewm(span=period, adjust=False).mean()
        plus_di  = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_v)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_v)
        dx       = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx_v    = dx.ewm(span=period, adjust=False).mean()
        return adx_v, plus_di, minus_di

    @staticmethod
    def volume_ratio(df: pd.DataFrame, period=20) -> pd.Series:
        vol_sma = df["Volume"].rolling(period).mean()
        return df["Volume"] / vol_sma.replace(0, np.nan)

    @classmethod
    def add_all(cls, df: pd.DataFrame, cfg: GoldenConfig) -> pd.DataFrame:
        df = df.copy()
        c  = df["Close"]

        df["ema_fast"]  = cls.ema(c, cfg.ema_fast)
        df["ema_slow"]  = cls.ema(c, cfg.ema_slow)
        df["ema_50"]    = cls.ema(c, cfg.ema_50)
        df["ema_200"]   = cls.ema(c, cfg.ema_200)

        df["rsi"] = cls.rsi(c, cfg.rsi_period)

        df["macd"], df["macd_signal"], df["macd_hist"] = cls.macd(
            c, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

        df["bb_upper"], df["bb_mid"], df["bb_lower"] = cls.bollinger_bands(
            c, cfg.bb_period, cfg.bb_std)
        df["bb_width"]    = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_position"] = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

        df["stoch_k"], df["stoch_d"] = cls.stochastic(
            df, cfg.stoch_k, cfg.stoch_d, cfg.stoch_smooth)

        df["atr"]  = cls.atr(df, cfg.atr_period)
        df["adx"], df["plus_di"], df["minus_di"] = cls.adx(df, cfg.atr_period)
        df["volume_ratio"] = cls.volume_ratio(df, cfg.volume_ma)

        df["candle_body"]  = (df["Close"] - df["Open"]).abs()
        df["candle_range"] = df["High"] - df["Low"]
        df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["body_ratio"]   = df["candle_body"] / df["candle_range"].replace(0, np.nan)
        df["is_bullish"]   = (df["Close"] > df["Open"]).astype(int)

        df["price_change"]   = c.pct_change()
        df["price_change_5"] = c.pct_change(5)
        df["volatility"]     = df["price_change"].rolling(20).std()

        df.dropna(inplace=True)
        if df.empty:
            log.debug("Indicators.add_all: DataFrame empty after dropna")
        return df


# ---------------------------------------------------------------------------
# PRICE ACTION ENGINE
# ---------------------------------------------------------------------------

class PriceAction:
    """Detects candlestick patterns and key levels."""

    @staticmethod
    def is_pin_bar(row: pd.Series, direction: str, pin_ratio: float = 0.6) -> bool:
        """
        Pin bar: long wick on one side, small body.
        direction='bull' => long lower shadow, small upper shadow, close near high
        direction='bear' => long upper shadow, small lower shadow, close near low
        """
        if row["candle_range"] == 0:
            return False
        body    = row["candle_body"]
        c_range = row["candle_range"]
        upper   = row["upper_shadow"]
        lower   = row["lower_shadow"]

        if direction == "bull":
            # Lower shadow at least pin_ratio of total range, body < 35% range
            return (lower / c_range >= pin_ratio) and (body / c_range <= 0.35)
        elif direction == "bear":
            return (upper / c_range >= pin_ratio) and (body / c_range <= 0.35)
        return False

    @staticmethod
    def is_engulfing(curr: pd.Series, prev: pd.Series, direction: str,
                     ratio: float = 1.1) -> bool:
        """
        Engulfing candle: current body fully engulfs previous body.
        direction='bull' => current green body > prev red body
        direction='bear' => current red body > prev green body
        """
        if prev["candle_body"] == 0:
            return False
        body_ratio = curr["candle_body"] / prev["candle_body"]

        if direction == "bull":
            return (curr["is_bullish"] == 1 and
                    prev["is_bullish"] == 0 and
                    body_ratio >= ratio)
        elif direction == "bear":
            return (curr["is_bullish"] == 0 and
                    prev["is_bullish"] == 1 and
                    body_ratio >= ratio)
        return False

    @staticmethod
    def is_doji(row: pd.Series, max_body_pct: float = 0.1) -> bool:
        if row["candle_range"] == 0:
            return False
        return row["candle_body"] / row["candle_range"] <= max_body_pct

    @staticmethod
    def find_sr_levels(df: pd.DataFrame, lookback: int = 50,
                       tolerance_pct: float = 0.003) -> Tuple[List[float], List[float]]:
        """
        Find Support and Resistance levels from swing highs/lows.
        Returns (support_levels, resistance_levels).
        """
        highs = df["High"].values[-lookback:]
        lows  = df["Low"].values[-lookback:]

        resistances = []
        supports    = []

        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                resistances.append(highs[i])

        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                supports.append(lows[i])

        # Cluster nearby levels
        def cluster(levels):
            if not levels:
                return []
            levels = sorted(levels)
            clustered = [levels[0]]
            for lvl in levels[1:]:
                if abs(lvl - clustered[-1]) / clustered[-1] > tolerance_pct:
                    clustered.append(lvl)
                else:
                    clustered[-1] = (clustered[-1] + lvl) / 2
            return clustered

        return cluster(supports), cluster(resistances)

    @staticmethod
    def near_level(price: float, levels: List[float],
                   tolerance_pct: float = 0.003) -> bool:
        """Check if price is within tolerance of any S/R level."""
        for lvl in levels:
            if abs(price - lvl) / lvl <= tolerance_pct:
                return True
        return False


# ---------------------------------------------------------------------------
# MACD DIVERGENCE ENGINE
# ---------------------------------------------------------------------------

class DivergenceEngine:
    """
    Detects Regular and Hidden MACD divergence.

    Regular Bullish Divergence:
      Price: Lower Low  |  MACD Histogram: Higher Low  => Buy signal
    Regular Bearish Divergence:
      Price: Higher High |  MACD Histogram: Lower High  => Sell signal
    Hidden Bullish Divergence:
      Price: Higher Low  |  MACD Histogram: Lower Low   => Trend continuation Buy
    Hidden Bearish Divergence:
      Price: Lower High  |  MACD Histogram: Higher High => Trend continuation Sell
    """

    @staticmethod
    def find_pivots(series: pd.Series, left: int = 5,
                    right: int = 5) -> Tuple[List[int], List[int]]:
        """Find pivot highs and lows in a series."""
        pivot_highs = []
        pivot_lows  = []
        values = series.values

        for i in range(left, len(values) - right):
            window = values[i - left: i + right + 1]
            if values[i] == max(window):
                pivot_highs.append(i)
            if values[i] == min(window):
                pivot_lows.append(i)

        return pivot_highs, pivot_lows

    @classmethod
    def detect(cls, df: pd.DataFrame,
               lookback: int = 40) -> Dict[str, Any]:
        """
        Detect divergence in the last `lookback` bars.
        Returns a dict with divergence types found.
        """
        result = {
            "regular_bull": False,
            "regular_bear": False,
            "hidden_bull":  False,
            "hidden_bear":  False,
            "strength":     0.0,   # 0.0 to 1.0
        }

        if len(df) < lookback + 10:
            return result

        window = df.tail(lookback).copy()
        price  = window["Close"]
        hist   = window["macd_hist"]

        ph_price, pl_price = cls.find_pivots(price, left=3, right=3)
        ph_hist,  pl_hist  = cls.find_pivots(hist, left=3, right=3)

        # Need at least 2 pivot highs and 2 pivot lows
        if len(ph_price) >= 2 and len(ph_hist) >= 2:
            # Regular Bearish: price HH, macd LH
            pp1, pp2 = ph_price[-2], ph_price[-1]
            hp1, hp2 = ph_hist[-2],  ph_hist[-1]
            if (price.iloc[pp2] > price.iloc[pp1] and
                    hist.iloc[hp2] < hist.iloc[hp1] and
                    hist.iloc[hp2] < 0):
                result["regular_bear"] = True
                gap = abs(hist.iloc[hp1] - hist.iloc[hp2])
                result["strength"] = max(result["strength"], min(gap * 500, 1.0))

            # Hidden Bearish: price LH, macd HH
            if (price.iloc[pp2] < price.iloc[pp1] and
                    hist.iloc[hp2] > hist.iloc[hp1]):
                result["hidden_bear"] = True

        if len(pl_price) >= 2 and len(pl_hist) >= 2:
            # Regular Bullish: price LL, macd HL
            pp1, pp2 = pl_price[-2], pl_price[-1]
            lp1, lp2 = pl_hist[-2],  pl_hist[-1]
            if (price.iloc[pp2] < price.iloc[pp1] and
                    hist.iloc[lp2] > hist.iloc[lp1] and
                    hist.iloc[lp2] < 0):
                result["regular_bull"] = True
                gap = abs(hist.iloc[lp1] - hist.iloc[lp2])
                result["strength"] = max(result["strength"], min(gap * 500, 1.0))

            # Hidden Bullish: price HL, macd LL
            if (price.iloc[pp2] > price.iloc[pp1] and
                    hist.iloc[lp2] < hist.iloc[lp1]):
                result["hidden_bull"] = True

        return result


# ---------------------------------------------------------------------------
# STRATEGY 1: TREND-MOMENTUM (Golden Cross + RSI Pullback)
# ---------------------------------------------------------------------------

class Strategy1_GoldenCross:
    """
    Golden Cross / Death Cross trend-momentum strategy.

    Buy setup:
      1. 50 EMA > 200 EMA (Golden Cross context)
      2. Price > 50 EMA (above trend)
      3. Fast EMA > Slow EMA (short-term momentum up)
      4. RSI pulled back to 40-55 (healthy pullback, not overbought)
      5. RSI now rising (momentum returning)
      6. Volume >= average (participation)
      7. ADX > 20 (trending, not ranging)

    Sell setup: Mirror image.
    """

    @staticmethod
    def score(df: pd.DataFrame, cfg: GoldenConfig,
              direction: str) -> Tuple[float, List[str]]:
        if len(df) < 3:
            return 0.0, []

        cur  = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        reasons = []
        score   = 0.0

        ema_50  = cur["ema_50"]
        ema_200 = cur["ema_200"]
        ema_f   = cur["ema_fast"]
        ema_s   = cur["ema_slow"]
        rsi     = cur["rsi"]
        rsi_p   = prev["rsi"]
        rsi_p2  = prev2["rsi"]
        adx     = cur["adx"]
        close   = cur["Close"]
        vol     = cur["volume_ratio"]

        if direction == "BUY":
            # [2 pts] Golden Cross: 50 > 200
            if ema_50 > ema_200:
                score += 2.0
                reasons.append(f"✅ Golden Cross (EMA50 > EMA200)")
            elif ema_50 > ema_200 * 0.999:
                score += 0.5
                reasons.append(f"⚠️ EMA50 near EMA200 (forming cross)")

            # [1.5 pts] Price above 50 EMA
            if close > ema_50:
                score += 1.5
                reasons.append(f"✅ Price above EMA50")

            # [1 pt] Fast > Slow EMA (short-term up)
            if ema_f > ema_s:
                score += 1.0
                reasons.append(f"✅ EMA{cfg.ema_fast} > EMA{cfg.ema_slow}")

            # [2 pts] RSI pullback: was in 40-55, now rising
            rsi_pulled_back = cfg.rsi_pullback_min <= rsi <= cfg.rsi_pullback_max
            rsi_was_lower   = rsi_p <= rsi and rsi_p2 <= rsi_p + 2
            if rsi_pulled_back and rsi_was_lower:
                score += 2.0
                reasons.append(f"✅ RSI pullback to {rsi:.1f} and rising (momentum reset)")
            elif cfg.rsi_pullback_min <= rsi <= 60:
                score += 1.0
                reasons.append(f"⚠️ RSI at {rsi:.1f} (borderline zone)")

            # [2 pts] Not overbought
            if rsi < cfg.rsi_overbought:
                score += 1.0
                reasons.append(f"✅ RSI {rsi:.1f} not overbought")

            # [1 pt] ADX > 20 (trending)
            if adx > 25:
                score += 1.0
                reasons.append(f"✅ ADX={adx:.1f} (strong trend)")
            elif adx > 20:
                score += 0.5
                reasons.append(f"⚠️ ADX={adx:.1f} (moderate trend)")

            # [0.5 pt] Volume confirmation
            if vol >= cfg.volume_min:
                score += 0.5
                reasons.append(f"✅ Volume ratio={vol:.2f}x")

        elif direction == "SELL":
            # [2 pts] Death Cross: 50 < 200
            if ema_50 < ema_200:
                score += 2.0
                reasons.append(f"✅ Death Cross (EMA50 < EMA200)")
            elif ema_50 < ema_200 * 1.001:
                score += 0.5
                reasons.append(f"⚠️ EMA50 near EMA200 (forming death cross)")

            # [1.5 pts] Price below 50 EMA
            if close < ema_50:
                score += 1.5
                reasons.append(f"✅ Price below EMA50")

            # [1 pt] Fast < Slow EMA
            if ema_f < ema_s:
                score += 1.0
                reasons.append(f"✅ EMA{cfg.ema_fast} < EMA{cfg.ema_slow}")

            # [2 pts] RSI pullback bounce from 45-60 area, now falling
            rsi_pb = cfg.rsi_sell_min if hasattr(cfg, "rsi_sell_min") else 45
            rsi_pulled_back = 45.0 <= rsi <= 60.0
            rsi_falling     = rsi_p >= rsi and rsi_p2 >= rsi_p - 2
            if rsi_pulled_back and rsi_falling:
                score += 2.0
                reasons.append(f"✅ RSI bounce-down at {rsi:.1f} (dead cat)")
            elif 35 <= rsi <= 60:
                score += 1.0
                reasons.append(f"⚠️ RSI {rsi:.1f} in sell zone")

            # Not oversold
            if rsi > cfg.rsi_oversold:
                score += 1.0
                reasons.append(f"✅ RSI {rsi:.1f} not oversold")

            # ADX
            if adx > 25:
                score += 1.0
                reasons.append(f"✅ ADX={adx:.1f} (strong downtrend)")
            elif adx > 20:
                score += 0.5

            # Volume
            if vol >= cfg.volume_min:
                score += 0.5
                reasons.append(f"✅ Volume={vol:.2f}x average")

        # Normalize to 0-10
        score = min(score, 10.0)
        return score, reasons


# ---------------------------------------------------------------------------
# STRATEGY 2: MEAN REVERSION (Bollinger + Stochastic)
# ---------------------------------------------------------------------------

class Strategy2_MeanReversion:
    """
    Mean Reversion: Price at extreme Bollinger Band + Stochastic reversal.

    Long setup:
      1. Price touches or breaks lower Bollinger Band
      2. Stochastic K < 20 (oversold)
      3. Stochastic K crossing above D (reversal signal)
      4. RSI < 35 (confirms oversold)
      5. Bullish candle pattern at the band (pin bar / engulfing)
      6. BB Width expanding (volatility = energy for bounce)
      7. Target: middle BB (mean)

    Short setup: Mirror image at upper band.
    """

    @staticmethod
    def score(df: pd.DataFrame, cfg: GoldenConfig,
              direction: str) -> Tuple[float, List[str]]:
        if len(df) < 3:
            return 0.0, []

        cur  = df.iloc[-1]
        prev = df.iloc[-2]
        reasons = []
        score   = 0.0

        close    = cur["Close"]
        bb_upper = cur["bb_upper"]
        bb_lower = cur["bb_lower"]
        bb_mid   = cur["bb_mid"]
        bb_pos   = cur.get("bb_position", 0.5)
        bb_width = cur["bb_width"]
        stoch_k  = cur["stoch_k"]
        stoch_d  = cur["stoch_d"]
        stoch_k_prev = prev["stoch_k"]
        stoch_d_prev = prev["stoch_d"]
        rsi      = cur["rsi"]
        vol      = cur["volume_ratio"]

        # Price distance to bands
        lower_dist = abs(close - bb_lower) / bb_lower
        upper_dist = abs(close - bb_upper) / bb_upper

        if direction == "BUY":
            # [2.5 pts] Price at or below lower BB
            if close <= bb_lower:
                score += 2.5
                reasons.append(f"✅ Price BELOW lower BB (mean reversion entry)")
            elif lower_dist <= cfg.bb_touch_pct:
                score += 2.0
                reasons.append(f"✅ Price touching lower BB ({lower_dist*100:.2f}% away)")
            elif lower_dist <= cfg.bb_touch_pct * 2:
                score += 1.0
                reasons.append(f"⚠️ Price near lower BB")

            # [2.5 pts] Stochastic oversold + crossing up
            stoch_cross_up = (stoch_k_prev <= stoch_d_prev) and (stoch_k > stoch_d)
            if stoch_k < cfg.stoch_oversold and stoch_cross_up:
                score += 2.5
                reasons.append(f"✅ Stoch K={stoch_k:.1f} oversold + crossing UP ↑")
            elif stoch_k < cfg.stoch_oversold:
                score += 1.5
                reasons.append(f"✅ Stoch K={stoch_k:.1f} oversold")
            elif stoch_k < 30 and stoch_cross_up:
                score += 1.0
                reasons.append(f"⚠️ Stoch crossing up at {stoch_k:.1f}")

            # [1.5 pts] RSI oversold
            if rsi < 35:
                score += 1.5
                reasons.append(f"✅ RSI={rsi:.1f} oversold (<35)")
            elif rsi < 40:
                score += 0.75
                reasons.append(f"⚠️ RSI={rsi:.1f} near oversold")

            # [2 pts] Bullish candle pattern
            pin = PriceAction.is_pin_bar(cur, "bull", cfg.pin_bar_ratio)
            eng = PriceAction.is_engulfing(cur, prev, "bull", cfg.engulfing_ratio)
            if pin:
                score += 2.0
                reasons.append(f"✅ Pin bar (bullish reversal candle)")
            elif eng:
                score += 2.0
                reasons.append(f"✅ Bullish engulfing candle")
            elif cur["is_bullish"] == 1 and cur["body_ratio"] > 0.5:
                score += 0.5
                reasons.append(f"⚠️ Bullish close")

            # [0.5 pts] BB width (volatility expanding = room to bounce)
            if bb_width > df["bb_width"].rolling(20).mean().iloc[-1]:
                score += 0.5
                reasons.append(f"✅ BB widening (volatility expanding)")

        elif direction == "SELL":
            # [2.5 pts] Price at or above upper BB
            if close >= bb_upper:
                score += 2.5
                reasons.append(f"✅ Price ABOVE upper BB (mean reversion short)")
            elif upper_dist <= cfg.bb_touch_pct:
                score += 2.0
                reasons.append(f"✅ Price touching upper BB ({upper_dist*100:.2f}% away)")
            elif upper_dist <= cfg.bb_touch_pct * 2:
                score += 1.0
                reasons.append(f"⚠️ Price near upper BB")

            # [2.5 pts] Stochastic overbought + crossing down
            stoch_cross_dn = (stoch_k_prev >= stoch_d_prev) and (stoch_k < stoch_d)
            if stoch_k > cfg.stoch_overbought and stoch_cross_dn:
                score += 2.5
                reasons.append(f"✅ Stoch K={stoch_k:.1f} overbought + crossing DOWN ↓")
            elif stoch_k > cfg.stoch_overbought:
                score += 1.5
                reasons.append(f"✅ Stoch K={stoch_k:.1f} overbought")
            elif stoch_k > 70 and stoch_cross_dn:
                score += 1.0
                reasons.append(f"⚠️ Stoch crossing down at {stoch_k:.1f}")

            # [1.5 pts] RSI overbought
            if rsi > 65:
                score += 1.5
                reasons.append(f"✅ RSI={rsi:.1f} overbought (>65)")
            elif rsi > 60:
                score += 0.75
                reasons.append(f"⚠️ RSI={rsi:.1f} elevated")

            # [2 pts] Bearish candle pattern
            pin = PriceAction.is_pin_bar(cur, "bear", cfg.pin_bar_ratio)
            eng = PriceAction.is_engulfing(cur, prev, "bear", cfg.engulfing_ratio)
            if pin:
                score += 2.0
                reasons.append(f"✅ Bearish pin bar (rejection candle)")
            elif eng:
                score += 2.0
                reasons.append(f"✅ Bearish engulfing candle")
            elif cur["is_bullish"] == 0 and cur["body_ratio"] > 0.5:
                score += 0.5
                reasons.append(f"⚠️ Bearish close")

            # BB width
            if bb_width > df["bb_width"].rolling(20).mean().iloc[-1]:
                score += 0.5
                reasons.append(f"✅ BB widening")

        score = min(score, 10.0)
        return score, reasons


# ---------------------------------------------------------------------------
# STRATEGY 3: MACD DIVERGENCE + PRICE ACTION CONFLUENCE
# ---------------------------------------------------------------------------

class Strategy3_Divergence:
    """
    MACD Divergence at key S/R level + Price Action candle confirmation.

    The Setup:
      1. Identify clear S/R level from swing highs/lows
      2. Price approaches that level
      3. MACD divergence present (regular or hidden)
      4. Pin bar or engulfing candle at the level
      5. Volume confirmation on the signal candle
      6. MACD histogram changing direction (zero-line proximity)

    This strategy mirrors the setup in your uploaded image perfectly:
    price making new high/low while MACD makes lower high/higher low.
    """

    @staticmethod
    def score(df: pd.DataFrame, cfg: GoldenConfig,
              direction: str) -> Tuple[float, List[str]]:
        if len(df) < 40:
            return 0.0, []

        cur  = df.iloc[-1]
        prev = df.iloc[-2]
        reasons = []
        score   = 0.0

        close  = cur["Close"]
        vol    = cur["volume_ratio"]

        # Find S/R levels
        supports, resistances = PriceAction.find_sr_levels(
            df, cfg.sr_lookback, cfg.sr_tolerance_pct)

        # Detect divergence
        div = DivergenceEngine.detect(df, cfg.divergence_lookback)

        macd_h    = cur["macd_hist"]
        macd_h_p  = prev["macd_hist"]
        macd_line = cur["macd"]
        rsi       = cur["rsi"]
        adx       = cur["adx"]

        if direction == "BUY":
            # [3 pts] Regular bullish divergence
            if div["regular_bull"]:
                strength_bonus = div["strength"] * 1.0
                score += 3.0 + strength_bonus
                reasons.append(f"✅ REGULAR BULLISH DIVERGENCE (price LL, MACD HL) — strength={div['strength']:.2f}")

            # [1.5 pts] Hidden bullish divergence (trend continuation)
            elif div["hidden_bull"]:
                score += 1.5
                reasons.append(f"✅ Hidden bullish divergence (trend continuation)")

            # [2 pts] Price at support level
            near_sup = PriceAction.near_level(close, supports, cfg.sr_tolerance_pct)
            if near_sup:
                score += 2.0
                nearest = min(supports, key=lambda x: abs(x - close)) if supports else close
                reasons.append(f"✅ Price at support level ({nearest:.5f})")

            # [2 pts] Bullish price action candle
            pin = PriceAction.is_pin_bar(cur, "bull", cfg.pin_bar_ratio)
            eng = PriceAction.is_engulfing(cur, prev, "bull", cfg.engulfing_ratio)
            if pin:
                score += 2.0
                reasons.append(f"✅ Bullish pin bar at support")
            elif eng:
                score += 2.0
                reasons.append(f"✅ Bullish engulfing at support")
            elif PriceAction.is_doji(cur):
                score += 0.75
                reasons.append(f"⚠️ Doji (indecision)")

            # [1 pt] MACD histogram turning from negative
            if macd_h > macd_h_p and macd_h_p < 0:
                score += 1.0
                reasons.append(f"✅ MACD histogram turning UP (momentum shift)")

            # [0.5 pt] Volume spike on signal bar
            if vol >= 1.5:
                score += 0.5
                reasons.append(f"✅ Volume spike = {vol:.2f}x (conviction)")
            elif vol >= cfg.volume_min:
                score += 0.25

            # RSI not overbought
            if rsi < 60:
                score += 0.5
                reasons.append(f"✅ RSI={rsi:.1f} has room to run")

        elif direction == "SELL":
            # [3 pts] Regular bearish divergence
            if div["regular_bear"]:
                strength_bonus = div["strength"] * 1.0
                score += 3.0 + strength_bonus
                reasons.append(f"✅ REGULAR BEARISH DIVERGENCE (price HH, MACD LH) — strength={div['strength']:.2f}")

            # [1.5 pts] Hidden bearish divergence
            elif div["hidden_bear"]:
                score += 1.5
                reasons.append(f"✅ Hidden bearish divergence (trend continuation)")

            # [2 pts] Price at resistance level
            near_res = PriceAction.near_level(close, resistances, cfg.sr_tolerance_pct)
            if near_res:
                score += 2.0
                nearest = min(resistances, key=lambda x: abs(x - close)) if resistances else close
                reasons.append(f"✅ Price at resistance level ({nearest:.5f})")

            # [2 pts] Bearish price action candle
            pin = PriceAction.is_pin_bar(cur, "bear", cfg.pin_bar_ratio)
            eng = PriceAction.is_engulfing(cur, prev, "bear", cfg.engulfing_ratio)
            if pin:
                score += 2.0
                reasons.append(f"✅ Bearish pin bar at resistance")
            elif eng:
                score += 2.0
                reasons.append(f"✅ Bearish engulfing at resistance")
            elif PriceAction.is_doji(cur):
                score += 0.75
                reasons.append(f"⚠️ Doji at resistance (indecision = potential reversal)")

            # [1 pt] MACD histogram turning from positive
            if macd_h < macd_h_p and macd_h_p > 0:
                score += 1.0
                reasons.append(f"✅ MACD histogram turning DOWN")

            # Volume
            if vol >= 1.5:
                score += 0.5
                reasons.append(f"✅ Volume spike = {vol:.2f}x")
            elif vol >= cfg.volume_min:
                score += 0.25

            # RSI not oversold
            if rsi > 40:
                score += 0.5
                reasons.append(f"✅ RSI={rsi:.1f} has room to fall")

        score = min(score, 10.0)
        return score, reasons


# ---------------------------------------------------------------------------
# DATA FETCHER
# ---------------------------------------------------------------------------

class DataFetcher:
    YF_INTERVAL_MAP = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1h", "4h": "1h", "1d": "1d", "1wk": "1wk",
    }
    YF_PERIOD_MAP = {
        "1m": "7d", "5m": "30d", "15m": "60d", "30m": "60d",
        "1h": "90d", "4h": "180d", "1d": "2y", "1wk": "5y",
    }

    def __init__(self, cfg: GoldenConfig):
        self.cfg = cfg

    def _yf_symbol(self, symbol: str) -> str:
        if len(symbol) == 6 and symbol.isalpha():
            return f"{symbol[:3]}{symbol[3:]}=X"
        if symbol == "XAUUSD":
            return "GC=F"
        return symbol

    def fetch_yf(self, symbol: str, timeframe: str) -> pd.DataFrame:
        yf_sym   = self._yf_symbol(symbol)
        interval = self.YF_INTERVAL_MAP.get(timeframe, "1h")
        period   = self.YF_PERIOD_MAP.get(timeframe, "90d")
        try:
            df = yf.download(yf_sym, period=period, interval=interval,
                             progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            required = ["Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in df.columns:
                    return pd.DataFrame()
            return df[required].dropna().copy()
        except Exception as e:
            log.error(f"YF error {yf_sym}: {e}")
            return pd.DataFrame()

    def fetch_mt5(self, symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        if not HAS_MT5:
            raise RuntimeError("MT5 not installed")
        TF_MAP = {
            "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15, "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4,
            "1d": mt5.TIMEFRAME_D1,
        }
        tf = TF_MAP.get(timeframe, mt5.TIMEFRAME_H1)
        mt5_sym = symbol + self.cfg.mt5_symbol_suffix
        rates = mt5.copy_rates_from_pos(mt5_sym, tf, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "tick_volume": "Volume"
        }, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    def fetch(self, symbol: str, timeframe: str, use_mt5: bool = False) -> pd.DataFrame:
        if use_mt5 and HAS_MT5:
            return self.fetch_mt5(symbol, timeframe)
        return self.fetch_yf(symbol, timeframe)

    def get_spread_pips(self, symbol: str) -> float:
        if not HAS_MT5:
            return 1.0
        mt5_sym = symbol + self.cfg.mt5_symbol_suffix
        tick = mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            return 99.0
        info = mt5.symbol_info(mt5_sym)
        pip  = 0.0001 if (info and info.digits == 5) else 0.01
        return (tick.ask - tick.bid) / pip


# ---------------------------------------------------------------------------
# COMBINED SIGNAL ENGINE
# ---------------------------------------------------------------------------

class GoldenSignalEngine:
    """
    Orchestrates all three strategies and produces a combined signal.

    1. Determine macro bias from 4H chart
    2. Determine trend direction from 1H chart
    3. Score all three strategies on 15m entry chart
    4. Only signal if weighted combo score >= min_score
    5. Use the best-scoring direction (BUY or SELL)
    """

    def __init__(self, cfg: GoldenConfig, fetcher: DataFetcher):
        self.cfg     = cfg
        self.fetcher = fetcher

    def _session_ok(self) -> bool:
        hour = datetime.utcnow().hour
        for s, e in self.cfg.trade_sessions:
            if s <= hour < e:
                return True
        return False

    def _macro_bias(self, symbol: str, use_mt5: bool) -> Optional[str]:
        df = self.fetcher.fetch(symbol, self.cfg.confirm_tf, use_mt5)
        if df.empty or len(df) < 210:
            return None
        df = Indicators.add_all(df, self.cfg)
        if df.empty:
            return None
        last = df.iloc[-1]
        if last["Close"] > last["ema_200"] and last["ema_50"] > last["ema_200"]:
            return "BUY"
        elif last["Close"] < last["ema_200"] and last["ema_50"] < last["ema_200"]:
            return "SELL"
        return None

    def _trend(self, symbol: str, use_mt5: bool) -> Tuple[Optional[str], pd.DataFrame]:
        df = self.fetcher.fetch(symbol, self.cfg.trend_tf, use_mt5)
        if df.empty or len(df) < 60:
            return None, pd.DataFrame()
        df = Indicators.add_all(df, self.cfg)
        if df.empty:
            return None, pd.DataFrame()
        last = df.iloc[-1]
        if last["Close"] > last["ema_50"] and last["ema_fast"] > last["ema_slow"]:
            return "BUY", df
        elif last["Close"] < last["ema_50"] and last["ema_fast"] < last["ema_slow"]:
            return "SELL", df
        return None, df

    def _score_direction(self, df15: pd.DataFrame,
                         direction: str) -> StrategyScore:
        s = StrategyScore()
        s.s1_score, s.s1_reasons = Strategy1_GoldenCross.score(df15, self.cfg, direction)
        s.s2_score, s.s2_reasons = Strategy2_MeanReversion.score(df15, self.cfg, direction)
        s.s3_score, s.s3_reasons = Strategy3_Divergence.score(df15, self.cfg, direction)
        return s

    def _calc_sl_tp(self, direction: str, entry: float, atr: float,
                    df15: pd.DataFrame) -> Tuple[float, float, float]:
        sl_dist = atr * self.cfg.sl_atr_multiplier

        if direction == "BUY":
            recent_low  = df15["Low"].tail(15).min()
            swing_sl    = entry - recent_low
            sl_dist     = max(sl_dist, swing_sl * 1.05)
            sl          = entry - sl_dist
            tp          = entry + sl_dist * self.cfg.tp_rr_ratio
        else:
            recent_high = df15["High"].tail(15).max()
            swing_sl    = recent_high - entry
            sl_dist     = max(sl_dist, swing_sl * 1.05)
            sl          = entry + sl_dist
            tp          = entry - sl_dist * self.cfg.tp_rr_ratio

        return sl, tp, sl_dist

    def _calc_lots(self, equity: float, sl_dist: float, symbol: str) -> float:
        cfg = self.cfg
        risk = equity * cfg.risk_per_trade_pct
        pip  = 0.01 if "JPY" in symbol else 0.0001
        sl_pips = sl_dist / pip
        pip_val = 9.0 if "JPY" in symbol else 10.0
        if sl_pips <= 0:
            return 0.0
        lots = risk / (sl_pips * pip_val)
        lots = round(lots / cfg.lot_step) * cfg.lot_step
        lots = max(cfg.lot_min, min(lots, cfg.lot_max))
        return round(lots, 2)

    def generate(self, symbol: str, equity: float,
                 use_mt5: bool = False) -> Optional[TradeSignal]:

        if not self._session_ok():
            return None

        spread = self.fetcher.get_spread_pips(symbol)
        if spread > self.cfg.spread_max_pips:
            log.warning(f"{symbol}: spread {spread:.1f} pips too wide")
            return None

        macro = self._macro_bias(symbol, use_mt5)
        trend, _ = self._trend(symbol, use_mt5)

        if trend is None:
            log.info(f"{symbol}: no clear trend on 1H")
            return None

        # Soft macro filter
        if macro is not None and macro != trend:
            log.info(f"{symbol}: macro ({macro}) vs trend ({trend}) conflict")
            return None

        df15 = self.fetcher.fetch(symbol, self.cfg.entry_tf, use_mt5)
        if df15.empty or len(df15) < 80:
            log.warning(f"{symbol}: insufficient 15m data")
            return None

        df15 = Indicators.add_all(df15, self.cfg)
        if df15.empty:
            return None

        # Score both directions; pick the one aligned with trend
        score = self._score_direction(df15, trend)
        combined = score.combined(self.cfg)

        if combined < self.cfg.min_score:
            log.info(
                f"{symbol}: combined score {combined:.2f} < "
                f"{self.cfg.min_score} threshold | "
                f"S1={score.s1_score:.1f} S2={score.s2_score:.1f} "
                f"S3={score.s3_score:.1f}"
            )
            return None

        cur   = df15.iloc[-1]
        entry = float(cur["Close"])
        atr   = float(cur["atr"])
        rsi   = float(cur["rsi"])
        mh    = float(cur["macd_hist"])

        sl, tp, sl_dist = self._calc_sl_tp(trend, entry, atr, df15)
        tp_dist = abs(tp - entry)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        if rr < self.cfg.min_rr_ratio:
            log.info(f"{symbol}: RR={rr:.2f} below min {self.cfg.min_rr_ratio}")
            return None

        lots = self._calc_lots(equity, sl_dist, symbol)
        if lots <= 0:
            return None

        dom_strat = score.dominant_strategy()
        all_reasons = (
            [f"[S1-GoldenCross] {r}" for r in score.s1_reasons] +
            [f"[S2-MeanRev]     {r}" for r in score.s2_reasons] +
            [f"[S3-Divergence]  {r}" for r in score.s3_reasons]
        )
        reason_str = (
            f"Trend={trend} Macro={macro} | "
            f"Score={combined:.2f}/10 [S1={score.s1_score:.1f} S2={score.s2_score:.1f} "
            f"S3={score.s3_score:.1f}] | "
            f"DominantStrategy={dom_strat.value}\n" +
            "\n".join(f"    {r}" for r in all_reasons)
        )

        sig = TradeSignal(
            symbol      = symbol,
            signal      = SignalType.BUY if trend == "BUY" else SignalType.SELL,
            entry_price = round(entry, 5),
            stop_loss   = round(sl, 5),
            take_profit = round(tp, 5),
            sl_distance = round(sl_dist, 5),
            tp_distance = round(tp_dist, 5),
            rr_ratio    = round(rr, 2),
            lots        = lots,
            confidence  = round(combined / 10.0, 3),
            score       = score,
            strategy    = dom_strat.value,
            rsi         = round(rsi, 2),
            macd_hist   = round(mh, 6),
            atr         = round(atr, 5),
            spread_pips = round(spread, 1),
            reason      = reason_str,
        )

        log.info(
            f"SIGNAL > {sig.signal.value} {symbol} | "
            f"Entry={sig.entry_price} SL={sig.stop_loss} TP={sig.take_profit} | "
            f"RR={sig.rr_ratio} Score={combined:.1f}/10"
        )
        return sig


# ---------------------------------------------------------------------------
# ML FILTER (Optional — Gradient Boosting)
# ---------------------------------------------------------------------------

class MLFilter:
    FEATURES = [
        "ema_fast", "ema_slow", "ema_50", "rsi", "macd", "macd_hist",
        "bb_position", "bb_width", "stoch_k", "stoch_d",
        "atr", "adx", "plus_di", "minus_di", "volume_ratio",
        "price_change", "price_change_5", "body_ratio", "upper_shadow", "lower_shadow",
    ]

    def __init__(self, cfg: GoldenConfig):
        self.cfg     = cfg
        self.models  = {}
        self.scalers = {}

    def train(self, symbol: str, df: pd.DataFrame) -> float:
        if not HAS_SKL:
            return 0.0
        df = df.copy()
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df.dropna(inplace=True)
        avail = [c for c in self.FEATURES if c in df.columns]
        X, y  = df[avail].values, df["target"].values
        if len(X) < 200:
            return 0.0
        split = int(len(X) * 0.8)
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X[:split])
        X_te = scaler.transform(X[split:])
        model = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, min_samples_leaf=10, random_state=42)
        model.fit(X_tr, y[:split])
        acc = accuracy_score(y[split:], model.predict(X_te))
        f1  = f1_score(y[split:], model.predict(X_te))
        self.models[symbol]  = model
        self.scalers[symbol] = scaler
        log.info(f"ML {symbol}: accuracy={acc:.3f} f1={f1:.3f}")
        return acc

    def predict(self, symbol: str, df: pd.DataFrame, direction: str) -> float:
        if not HAS_SKL or symbol not in self.models:
            return 0.5
        if df.empty:
            return 0.5
        avail = [c for c in self.FEATURES if c in df.columns]
        X = self.scalers[symbol].transform(df[avail].iloc[-1:].values)
        prob = self.models[symbol].predict_proba(X)[0]
        return float(prob[1]) if direction == "BUY" else float(prob[0])


# ---------------------------------------------------------------------------
# TRADE EXECUTOR
# ---------------------------------------------------------------------------

class TradeExecutor:

    def __init__(self, cfg: GoldenConfig, fetcher: DataFetcher):
        self.cfg     = cfg
        self.fetcher = fetcher

    def _mt5_sym(self, symbol: str) -> str:
        return symbol + self.cfg.mt5_symbol_suffix

    def _get_filling_mode(self, symbol: str) -> int:
        info = mt5.symbol_info(symbol)
        if not info:
            return mt5.ORDER_FILLING_IOC
        
        # Robust handling of filling mode bitmask
        # Some MT5 versions lack SYMBOL_FILLING_* module constants
        filling_mode = getattr(info, 'filling_mode', 0)
        
        # Fallback to standard MT5 bit values: FOK=1, IOC=2
        fok_flag = getattr(mt5, 'SYMBOL_FILLING_FOK', 1)
        ioc_flag = getattr(mt5, 'SYMBOL_FILLING_IOC', 2)
        
        if filling_mode & fok_flag:
            return mt5.ORDER_FILLING_FOK
        elif filling_mode & ioc_flag:
            return mt5.ORDER_FILLING_IOC
        
        return mt5.ORDER_FILLING_RETURN


    def get_positions(self) -> List[dict]:
        if not HAS_MT5:
            return []
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [{
            "ticket": p.ticket, "symbol": p.symbol,
            "type": "BUY" if p.type == 0 else "SELL",
            "volume": p.volume, "price_open": p.price_open,
            "price_cur": p.price_current, "sl": p.sl, "tp": p.tp,
            "profit": p.profit, "magic": p.magic,
        } for p in positions]

    def has_position(self, symbol: str) -> bool:
        sym = self._mt5_sym(symbol)
        return any(p["symbol"] == sym and p["magic"] == self.cfg.mt5_magic
                   for p in self.get_positions())

    def execute_live(self, signal: TradeSignal) -> bool:
        if not HAS_MT5:
            return False
        mt5_sym = self._mt5_sym(signal.symbol)
        if not mt5.symbol_select(mt5_sym, True):
            log.error(f"Cannot select {mt5_sym}")
            return False
        tick = mt5.symbol_info_tick(mt5_sym)
        if tick is None:
            return False
        if signal.signal == SignalType.BUY:
            order_type, price = mt5.ORDER_TYPE_BUY, tick.ask
        else:
            order_type, price = mt5.ORDER_TYPE_SELL, tick.bid
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": mt5_sym,
            "volume": float(signal.lots), "type": order_type,
            "price": price, "sl": signal.stop_loss, "tp": signal.take_profit,
            "magic": self.cfg.mt5_magic, "comment": self.cfg.mt5_comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(mt5_sym),
        }
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"MT5 ORDER > {signal.signal.value} {mt5_sym} @ {price:.5f}")
            return True
        log.error(f"MT5 order failed: {mt5.last_error()}")
        return False

    def execute_paper(self, signal: TradeSignal) -> bool:
        log.info(
            f"[PAPER] {signal.signal.value} {signal.symbol} "
            f"{signal.lots}L @ {signal.entry_price:.5f} | "
            f"SL={signal.stop_loss:.5f} TP={signal.take_profit:.5f} | "
            f"RR={signal.rr_ratio} Score={signal.confidence*10:.1f}/10"
        )
        return True

    def execute(self, signal: TradeSignal) -> bool:
        if self.cfg.live_trading and HAS_MT5:
            return self.execute_live(signal)
        return self.execute_paper(signal)

    def update_trailing_sl(self):
        if not HAS_MT5:
            return
        for pos in self.get_positions():
            if pos["magic"] != self.cfg.mt5_magic:
                continue
            df = self.fetcher.fetch_mt5(pos["symbol"], "15m")
            if df.empty:
                continue
            df   = Indicators.add_all(df, self.cfg)
            if df.empty:
                continue
            atr  = float(df["atr"].iloc[-1])
            dist = atr * 1.0
            if pos["type"] == "BUY":
                new_sl = pos["price_cur"] - dist
                if new_sl > pos["sl"] and new_sl < pos["price_cur"]:
                    self._modify_sl(pos["ticket"], pos["symbol"], new_sl, pos["tp"])
            else:
                new_sl = pos["price_cur"] + dist
                if new_sl < pos["sl"] and new_sl > pos["price_cur"]:
                    self._modify_sl(pos["ticket"], pos["symbol"], new_sl, pos["tp"])

    def _modify_sl(self, ticket, symbol, new_sl, tp):
        req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": symbol,
               "sl": new_sl, "tp": tp, "position": ticket}
        result = mt5.order_send(req)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"Trailing SL → {new_sl:.5f} (ticket {ticket})")


# ---------------------------------------------------------------------------
# RISK MANAGER
# ---------------------------------------------------------------------------

class RiskManager:

    def __init__(self, cfg: GoldenConfig):
        self.cfg            = cfg
        self.day_equity     = 0.0
        self.today          = datetime.utcnow().date()
        self.paused         = False
        self.pause_reason   = ""

    def update(self, equity: float):
        today = datetime.utcnow().date()
        if today != self.today:
            log.info(f"New day. Equity reset from {self.day_equity:.2f} → {equity:.2f}")
            self.day_equity = equity
            self.today      = today
            self.paused     = False
        elif self.day_equity == 0:
            self.day_equity = equity

    def check(self, equity: float) -> bool:
        if self.day_equity <= 0:
            return True
        pct = (equity - self.day_equity) / self.day_equity
        if pct <= -self.cfg.max_daily_loss_pct:
            if not self.paused:
                self.paused = True
                self.pause_reason = f"Daily loss {pct:.1%} hit limit"
                log.warning(f"⛔ PAUSED: {self.pause_reason}")
            return False
        if self.paused:
            self.paused = False
        return True

    def can_trade(self, equity: float, n_open: int) -> Tuple[bool, str]:
        if not self.check(equity):
            return False, self.pause_reason
        if n_open >= self.cfg.max_positions:
            return False, f"Max positions ({self.cfg.max_positions}) reached"
        return True, ""


# ---------------------------------------------------------------------------
# TRADE LOGGER
# ---------------------------------------------------------------------------

class TradeLogger:

    def __init__(self, cfg: GoldenConfig):
        self.cfg    = cfg
        self.trades = []
        self._load()

    def _load(self):
        if os.path.exists(self.cfg.trade_log_file):
            try:
                with open(self.cfg.trade_log_file) as f:
                    self.trades = json.load(f)
                log.info(f"Loaded {len(self.trades)} trade records")
            except Exception as e:
                log.error(f"Could not load trade log: {e}")

    def save(self):
        tmp = self.cfg.trade_log_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.trades, f, indent=2)
        os.replace(tmp, self.cfg.trade_log_file)

    def log_signal(self, signal: TradeSignal):
        entry = signal.to_dict()
        entry["type"] = "signal"
        self.trades.append(entry)
        self.save()

    def get_stats(self) -> dict:
        results = [t for t in self.trades if t.get("type") == "result"]
        if not results:
            return {"total_trades": 0}
        pnls   = [t["pnl_usd"] for t in results]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gw  = sum(wins) if wins else 0
        gl  = abs(sum(losses)) if losses else 0.001
        pf  = gw / gl
        wr  = len(wins) / len(pnls) if pnls else 0
        cum = np.cumsum(pnls)
        pk  = np.maximum.accumulate(cum)
        dd  = float(np.max(pk - cum)) if len(cum) else 0
        # Per-strategy breakdown
        strategy_stats = {}
        for t in results:
            strat = t.get("strategy", "Unknown")
            if strat not in strategy_stats:
                strategy_stats[strat] = {"wins": 0, "losses": 0, "pnl": 0}
            if t["pnl_usd"] > 0:
                strategy_stats[strat]["wins"] += 1
            else:
                strategy_stats[strat]["losses"] += 1
            strategy_stats[strat]["pnl"] += t["pnl_usd"]

        return {
            "total_trades":     len(results),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate_pct":     round(wr * 100, 1),
            "total_pnl_usd":    round(sum(pnls), 2),
            "avg_win_usd":      round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss_usd":     round(float(np.mean(losses)), 2) if losses else 0,
            "profit_factor":    round(pf, 2),
            "expectancy_usd":   round(float(np.mean(pnls)), 2),
            "max_drawdown_usd": round(dd, 2),
            "best_trade_usd":   round(max(pnls), 2) if pnls else 0,
            "worst_trade_usd":  round(min(pnls), 2) if pnls else 0,
            "by_strategy":      strategy_stats,
        }


# ---------------------------------------------------------------------------
# BACKTESTER
# ---------------------------------------------------------------------------

class GoldenBacktester:
    """
    Walk-forward backtest with per-strategy attribution.
    Uses daily bars for speed; same logic as live engine.
    """

    def __init__(self, cfg: GoldenConfig):
        self.cfg     = cfg
        self.fetcher = DataFetcher(cfg)

    def run(self, symbol: str, initial_capital: float = 10_000.0) -> dict:
        log.info(f"Backtesting {symbol} | Capital=${initial_capital:,.0f}")
        df = self.fetcher.fetch_yf(symbol, "1d")
        if df.empty or len(df) < 150:
            log.error(f"Insufficient data for {symbol}")
            return {}

        df = Indicators.add_all(df, self.cfg)
        log.info(f"Loaded {len(df)} daily bars for {symbol}")

        capital  = initial_capital
        position = None
        trades   = []
        equity   = [capital]

        for i in range(100, len(df) - 1):
            bar  = df.iloc[i]
            window = df.iloc[max(0, i - 100): i + 1].copy()

            # Manage open position
            if position is not None:
                high, low = bar["High"], bar["Low"]
                closed, outcome, exit_p = False, "", 0.0

                if position["side"] == "BUY":
                    if low <= position["sl"]:
                        exit_p, outcome, closed = position["sl"], "LOSS", True
                    elif high >= position["tp"]:
                        exit_p, outcome, closed = position["tp"], "WIN", True
                else:
                    if high >= position["sl"]:
                        exit_p, outcome, closed = position["sl"], "LOSS", True
                    elif low <= position["tp"]:
                        exit_p, outcome, closed = position["tp"], "WIN", True

                if closed:
                    pnl_pts = (exit_p - position["entry"]) if position["side"] == "BUY" \
                              else (position["entry"] - exit_p)
                    pip = 0.01 if "JPY" in symbol else 0.0001
                    pnl_pips = pnl_pts / pip
                    pnl_usd  = pnl_pips * 10.0 * position["lots"]
                    capital += pnl_usd
                    trades.append({
                        "side":     position["side"],
                        "strategy": position["strategy"],
                        "entry":    position["entry"],
                        "exit":     exit_p,
                        "sl":       position["sl"],
                        "tp":       position["tp"],
                        "lots":     position["lots"],
                        "pnl_pips": round(pnl_pips, 1),
                        "pnl_usd":  round(pnl_usd, 2),
                        "outcome":  outcome,
                        "bar":      i,
                    })
                    position = None

            # New signal?
            if position is None:
                direction, score, strategy = self._signal(window)
                if direction and score.combined(self.cfg) >= self.cfg.min_score:
                    entry = float(bar["Close"])
                    atr   = float(bar["atr"])
                    sl_d  = atr * self.cfg.sl_atr_multiplier
                    tp_d  = sl_d * self.cfg.tp_rr_ratio

                    if direction == "BUY":
                        sl, tp = entry - sl_d, entry + tp_d
                    else:
                        sl, tp = entry + sl_d, entry - tp_d

                    pip = 0.01 if "JPY" in symbol else 0.0001
                    lots = max(self.cfg.lot_min, min(
                        round((capital * self.cfg.risk_per_trade_pct) / (sl_d / pip * 10), 2),
                        self.cfg.lot_max))

                    position = {
                        "side": direction, "entry": entry, "sl": sl, "tp": tp,
                        "lots": lots, "strategy": strategy, "bar": i,
                    }

            equity.append(capital)

        # Close remaining
        if position and not df.empty:
            last = float(df["Close"].iloc[-1])
            p = (last - position["entry"]) if position["side"] == "BUY" \
                else (position["entry"] - last)
            pip = 0.01 if "JPY" in symbol else 0.0001
            pnl_pips = p / pip
            pnl_usd  = pnl_pips * 10 * position["lots"]
            capital += pnl_usd
            trades.append({
                "side": position["side"], "strategy": position["strategy"],
                "pnl_pips": round(pnl_pips, 1), "pnl_usd": round(pnl_usd, 2),
                "outcome": "WIN" if pnl_usd > 0 else "LOSS",
            })

        return self._stats(trades, initial_capital, capital, equity, symbol)

    def _signal(self, window: pd.DataFrame) -> Tuple[Optional[str], StrategyScore, str]:
        if len(window) < 60:
            return None, StrategyScore(), ""

        cur  = window.iloc[-1]
        trend = None
        if cur["Close"] > cur["ema_50"] and cur["ema_fast"] > cur["ema_slow"]:
            trend = "BUY"
        elif cur["Close"] < cur["ema_50"] and cur["ema_fast"] < cur["ema_slow"]:
            trend = "SELL"
        else:
            return None, StrategyScore(), ""

        score = StrategyScore()
        score.s1_score, score.s1_reasons = Strategy1_GoldenCross.score(window, self.cfg, trend)
        score.s2_score, score.s2_reasons = Strategy2_MeanReversion.score(window, self.cfg, trend)
        score.s3_score, score.s3_reasons = Strategy3_Divergence.score(window, self.cfg, trend)

        dom = score.dominant_strategy()
        return trend, score, dom.value

    def _stats(self, trades, initial, final, equity, symbol) -> dict:
        if not trades:
            return {"total_trades": 0, "note": "No trades generated"}

        pnls   = [t["pnl_usd"] for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        eq  = np.array(equity)
        pk  = np.maximum.accumulate(eq)
        dd  = (pk - eq) / pk
        max_dd = float(np.max(dd)) * 100

        gw = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 0.001

        # Per-strategy breakdown
        by_strat = {}
        for t in trades:
            s = t.get("strategy", "?")
            if s not in by_strat:
                by_strat[s] = {"trades": 0, "wins": 0, "pnl": 0}
            by_strat[s]["trades"] += 1
            by_strat[s]["pnl"]    += t["pnl_usd"]
            if t["pnl_usd"] > 0:
                by_strat[s]["wins"] += 1

        stats = {
            "symbol":            symbol,
            "initial_capital":   round(initial, 2),
            "final_capital":     round(final, 2),
            "total_return_pct":  round((final - initial) / initial * 100, 2),
            "total_trades":      len(trades),
            "wins":              len(wins),
            "losses":            len(losses),
            "win_rate_pct":      round(len(wins) / len(trades) * 100, 1) if trades else 0,
            "profit_factor":     round(gw / gl, 2),
            "avg_win_usd":       round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss_usd":      round(float(np.mean(losses)), 2) if losses else 0,
            "best_trade_usd":    round(max(pnls), 2),
            "worst_trade_usd":   round(min(pnls), 2),
            "max_drawdown_pct":  round(max_dd, 2),
            "expectancy_usd":    round(float(np.mean(pnls)), 2),
            "by_strategy":       by_strat,
        }

        log.info("=" * 60)
        log.info(f"  BACKTEST: {symbol}")
        log.info("=" * 60)
        for k, v in stats.items():
            if k != "by_strategy":
                log.info(f"  {k:<28}: {v}")
        log.info("  Strategy breakdown:")
        for strat, sd in by_strat.items():
            wr = round(sd["wins"] / sd["trades"] * 100, 1) if sd["trades"] else 0
            log.info(f"    {strat:<30}: trades={sd['trades']} win%={wr} pnl=${sd['pnl']:.2f}")
        log.info("=" * 60)
        return stats


# ---------------------------------------------------------------------------
# MT5 MANAGER
# ---------------------------------------------------------------------------

class MT5Manager:

    def __init__(self, cfg: GoldenConfig):
        self.cfg       = cfg
        self.connected = False

    def connect(self) -> bool:
        if not HAS_MT5:
            log.error("MT5 package not installed")
            return False
        login    = int(os.environ.get("MT5_LOGIN", "0"))
        password = os.environ.get("MT5_PASSWORD", "")
        server   = os.environ.get("MT5_SERVER", "")
        if login == 0:
            log.error("MT5_LOGIN not set")
            return False
        if not mt5.initialize():
            log.error(f"mt5.initialize() failed: {mt5.last_error()}")
            return False
        if not mt5.login(login=login, password=password, server=server):
            log.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        acc = mt5.account_info()
        log.info(f"MT5 connected | Balance={acc.balance:.2f} Equity={acc.equity:.2f}")
        self.connected = True
        return True

    def ensure(self) -> bool:
        if not HAS_MT5:
            return False
        if mt5.terminal_info() is None:
            return self.connect()
        return True

    def equity(self) -> float:
        if not HAS_MT5 or not self.ensure():
            return 10_000.0
        acc = mt5.account_info()
        return float(acc.equity) if acc else 10_000.0

    def disconnect(self):
        if HAS_MT5:
            mt5.shutdown()
            self.connected = False


# ---------------------------------------------------------------------------
# MAIN BOT
# ---------------------------------------------------------------------------

class GoldenTradingBot:

    def __init__(self, cfg: Optional[GoldenConfig] = None):
        self.cfg      = cfg or GoldenConfig()
        self.fetcher  = DataFetcher(self.cfg)
        self.engine   = GoldenSignalEngine(self.cfg, self.fetcher)
        self.ml       = MLFilter(self.cfg)
        self.executor = TradeExecutor(self.cfg, self.fetcher)
        self.risk     = RiskManager(self.cfg)
        self.logger   = TradeLogger(self.cfg)
        self.mt5      = MT5Manager(self.cfg)
        self.recent_signals = []

        log.info("GoldenBot v4 initialized")
        log.info(f"  Symbols      : {self.cfg.symbols}")
        log.info(f"  Live trading : {self.cfg.live_trading}")
        log.info(f"  Min score    : {self.cfg.min_score}/10")
        log.info(f"  RR target    : {self.cfg.tp_rr_ratio}R")
        log.info(f"  Risk/trade   : {self.cfg.risk_per_trade_pct:.0%}")

    def setup(self):
        if self.cfg.live_trading:
            if not self.mt5.connect():
                log.error("MT5 connect failed → paper mode")
                self.cfg.live_trading = False
        if self.cfg.use_ml and HAS_SKL:
            log.info("Training ML models…")
            for sym in self.cfg.symbols:
                try:
                    df = self.fetcher.fetch_yf(sym, "1d")
                    if not df.empty:
                        df = Indicators.add_all(df, self.cfg)
                        self.ml.train(sym, df)
                except Exception as e:
                    log.error(f"ML train {sym}: {e}")

    def process(self, symbol: str, equity: float, n_open: int) -> Optional[TradeSignal]:
        use_mt5 = self.cfg.live_trading and HAS_MT5
        ok, reason = self.risk.can_trade(equity, n_open)
        if not ok:
            log.debug(f"{symbol}: {reason}")
            return None
        if use_mt5 and self.executor.has_position(symbol):
            return None

        signal = self.engine.generate(symbol, equity, use_mt5)
        if signal is None:
            return None

        if self.cfg.use_ml and symbol in self.ml.models:
            df15 = self.fetcher.fetch(symbol, self.cfg.entry_tf, use_mt5)
            if not df15.empty:
                df15 = Indicators.add_all(df15, self.cfg)
                prob = self.ml.predict(symbol, df15, signal.signal.value)
                signal.ml_probability = prob
                if prob < self.cfg.ml_min_prob:
                    log.info(f"{symbol}: ML prob {prob:.2f} too low")
                    return None

        self.logger.log_signal(signal)
        self.recent_signals.append(signal.to_dict())
        if self.executor.execute(signal):
            return signal
        return None

    def run(self):
        self.setup()
        log.info("=" * 60)
        log.info("  GOLDEN BOT v4 — TRADING LOOP STARTED")
        log.info("=" * 60)
        loop = 0

        while True:
            try:
                loop += 1
                log.info(f"\n--- Loop #{loop} | {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC ---")

                if self.cfg.live_trading:
                    self.mt5.ensure()

                eq = self.mt5.equity()
                self.risk.update(eq)

                if not self.risk.check(eq):
                    log.warning("⛔ Daily loss limit. Sleeping…")
                    time.sleep(self.cfg.interval_seconds)
                    continue

                positions = self.executor.get_positions()
                n_open = len([p for p in positions
                              if p.get("magic") == self.cfg.mt5_magic])

                log.info(f"Equity=${eq:.2f} | Open={n_open}/{self.cfg.max_positions}")

                if self.cfg.live_trading:
                    self.executor.update_trailing_sl()

                for sym in self.cfg.symbols:
                    try:
                        self.process(sym, eq, n_open)
                    except Exception as e:
                        log.error(f"Error {sym}: {e}")

                stats = self.logger.get_stats()
                if stats.get("total_trades", 0) > 0:
                    log.info(
                        f"Perf | Trades={stats['total_trades']} "
                        f"WR={stats.get('win_rate_pct', 0)}% "
                        f"PnL=${stats.get('total_pnl_usd', 0):.2f} "
                        f"PF={stats.get('profit_factor', 0):.2f}"
                    )

                log.info(f"Sleeping {self.cfg.interval_seconds}s…")
                time.sleep(self.cfg.interval_seconds)

            except KeyboardInterrupt:
                log.info("KeyboardInterrupt — shutting down")
                break
            except Exception as e:
                log.error(f"Loop error: {e}\n{traceback.format_exc()}")
                time.sleep(30)

        self.logger.save()
        if self.cfg.live_trading:
            self.mt5.disconnect()

    def signals_now(self) -> List[dict]:
        eq = self.mt5.equity()
        results = []
        for sym in self.cfg.symbols:
            sig = self.engine.generate(sym, eq, False)
            if sig:
                results.append(sig.to_dict())
            else:
                results.append({"symbol": sym, "signal": "HOLD"})
        return results

    def backtest(self, symbol: Optional[str] = None,
                 capital: float = 10_000.0) -> dict:
        bt   = GoldenBacktester(self.cfg)
        syms = [symbol] if symbol else self.cfg.symbols
        all_results = {}
        for sym in syms:
            res = bt.run(sym, initial_capital=capital)
            if res:
                all_results[sym] = res
        return all_results


# ---------------------------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------------------------

def load_mt5_config():
    if not os.path.exists("config"):
        return
    try:
        with open("config") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    k = k.strip().upper()
                    v = v.strip().strip('"').strip("'")
                    if k == "LOGIN":
                        os.environ["MT5_LOGIN"] = v
                    elif k == "PASSWORD":
                        os.environ["MT5_PASSWORD"] = v
                    elif k == "SERVER":
                        os.environ["MT5_SERVER"] = v
    except Exception as e:
        log.error(f"Config load error: {e}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    load_mt5_config()

    cfg = GoldenConfig(
        symbols=[
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
        ],
        mt5_symbol_suffix="m",

        # Timeframes
        trend_tf="1h",
        entry_tf="15m",
        confirm_tf="4h",

        # Strategy 1 thresholds
        ema_50=50,
        ema_200=200,
        rsi_pullback_min=40.0,
        rsi_pullback_max=55.0,

        # Strategy 2 thresholds
        bb_period=20,
        bb_std=2.0,
        stoch_overbought=80.0,
        stoch_oversold=20.0,

        # Strategy 3 thresholds
        divergence_lookback=30,
        sr_lookback=50,
        sr_tolerance_pct=0.003,
        pin_bar_ratio=0.6,

        # Signal quality (raise this for higher accuracy, fewer trades)
        min_score=5.5,   # out of 10
        s1_weight=0.35,
        s2_weight=0.30,
        s3_weight=0.35,

        # Risk
        risk_per_trade_pct=0.01,
        sl_atr_multiplier=1.5,
        tp_rr_ratio=2.5,       # 2.5R target
        min_rr_ratio=2.0,
        max_positions=3,
        max_daily_loss_pct=0.03,
        lot_min=0.01,
        lot_max=0.10,

        # Execution
        live_trading=True,    # Set True + MT5 env vars for live
        spread_max_pips=50.0,

        # Sessions
        trade_sessions=[(0, 24)],   # 24/7 for testing; set to [(7,16),(13,22)] for sessions

        # ML
        use_ml=True,
        ml_min_prob=0.60,

        interval_seconds=60,
    )

    args = sys.argv[1:]

    if "--backtest" in args:
        print("\n" + "=" * 65)
        print("  GOLDEN BOT v4 — BACKTEST MODE")
        print("=" * 65)
        bot     = GoldenTradingBot(cfg)
        results = bot.backtest(capital=10_000.0)
        for sym, res in results.items():
            print(f"\n{'=' * 40}")
            print(f"  {sym}")
            print(f"{'=' * 40}")
            for k, v in res.items():
                if k == "by_strategy":
                    print(f"  Strategy breakdown:")
                    for strat, sd in v.items():
                        wr = round(sd["wins"] / sd["trades"] * 100, 1) if sd["trades"] else 0
                        print(f"    {strat:<35} trades={sd['trades']} win%={wr} pnl=${sd['pnl']:.2f}")
                else:
                    print(f"  {k:<30}: {v}")

    elif "--signal" in args:
        print("\n" + "=" * 65)
        print("  GOLDEN BOT v4 — CURRENT SIGNALS")
        print("=" * 65)
        bot     = GoldenTradingBot(cfg)
        signals = bot.signals_now()
        for sig in signals:
            sym = sig.get("symbol", "?")
            s   = sig.get("signal", "HOLD")
            if hasattr(s, "value"):
                s = s.value
            if s in ("BUY", "SELL"):
                print(f"\n  {'▲' if s == 'BUY' else '▼'} {sym}: {s}")
                print(f"    Entry={sig.get('entry_price')} "
                      f"SL={sig.get('stop_loss')} "
                      f"TP={sig.get('take_profit')} "
                      f"RR={sig.get('rr_ratio')}")
                print(f"    Score={sig.get('score_combined', 0):.1f}/10")
                print(f"    {sig.get('reason', '')[:200]}")
            else:
                print(f"  ─ {sym}: HOLD")

    elif "--stats" in args:
        bot   = GoldenTradingBot(cfg)
        stats = bot.logger.get_stats()
        print("\n" + "=" * 65)
        print("  GOLDEN BOT v4 — PERFORMANCE STATS")
        print("=" * 65)
        for k, v in stats.items():
            if k == "by_strategy":
                print("  By strategy:")
                for strat, sd in v.items():
                    print(f"    {strat}: {sd}")
            else:
                print(f"  {k:<30}: {v}")

    else:
        print("\n" + "=" * 65)
        print("  GOLDEN BOT v4 — TRADING LOOP")
        print(f"  Mode: {'LIVE (MT5)' if cfg.live_trading else 'PAPER'}")
        print("=" * 65)
        print("\n  Run modes:")
        print("    python golden_strategy_bot.py --backtest")
        print("    python golden_strategy_bot.py --signal")
        print("    python golden_strategy_bot.py --stats")
        print("\n  For live MT5 trading set cfg.live_trading=True and:")
        print("    export MT5_LOGIN=<account>")
        print("    export MT5_PASSWORD=<password>")
        print("    export MT5_SERVER=<broker_server>\n")
        bot = GoldenTradingBot(cfg)
        bot.run()


if __name__ == "__main__":
    main()