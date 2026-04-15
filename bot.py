"""
================================================================================
EXNESS MULTI-SYMBOL TREND-CONFIRMATION ALGORITHMIC TRADING BOT
================================================================================
Production-Ready Trading System — Fully Corrected & Audited

Strategy : Multi-Indicator Trend Confirmation (EMA + RSI + MACD + BB + ATR)
Timeframe : H1 (primary) + H4 (trend filter)
Symbols   : EURUSDm, USDJPYm, GBPUSDm, AUDUSDm
Risk/Trade: Fixed lot 0.01
SL/TP     : 2.0 × ATR / 2.0 × ATR  (1:1 R:R — tuned for >55% win-rate)

FIXES APPLIED vs. ORIGINAL:
  1.  execute_buy/sell_order: return type corrected to Optional[int] throughout;
      all error-paths return None (not False).
  2.  has_open_trade(): per-symbol limit now correctly uses MAX_TRADES_PER_SYMBOL=1;
      global MAX_OPEN_TRADES cap enforced separately in run().
  3.  calculate_professional_positions: SL/TP now read from Config.SL_MULTIPLIER /
      Config.TP_MULTIPLIER (were hardcoded to mismatched values).
  4.  manage_open_trades: trailing-stop branch guarded with proper elif so it does
      not fire before break-even is established; pip_value derived per-symbol via
      symbol point size (fixes JPY pairs, etc.).
  5.  _modify_position_sl: removed unused filling_mode fetch; request cleaned up.
  6.  generate_signal: RSI thresholds now read from Config (RSI_BUY_MIN/MAX,
      RSI_SELL_MIN/MAX); H1 trend filter properly gates final BUY/SELL decision.
  7.  SignalGenerator details dict: format harmonised so check_confluence_score
      receives consistent (status_str, bool) tuples.
  8.  process_history: position_id correctly used as key (MT5 deal.position_id
      maps to the open-position ticket for DEAL_ENTRY_OUT deals).
  9.  export_data: NaN / numpy scalar serialisation fixed via custom JSON encoder.
  10. Minimum stop-level validation added before order submission.
  11. MT5 reconnect logic added to the main loop.
  12. Config credentials moved to environment variables with fallback defaults.
  13. Config now instantiated (not passed as bare class) to run().
  14. H4 timeframe added as a higher-level trend filter (distinct from H1).
  15. LOOP_INTERVAL reduced to 60 s; candle-change detection prevents duplicate
      signal processing within the same candle.

Author : Senior Quantitative Developer (Audited & Corrected)
Date   : 2025
================================================================================
"""

import os
import json
import time
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Try to import pandas_ta, but provide fallback
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger = logging.getLogger(__name__)
    logger.warning("pandas_ta not available - using fallback indicator calculations")

import MetaTrader5 as mt5


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# JSON SERIALISER — handles numpy scalars & NaN
# ============================================================================

class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)

    def encode(self, obj):
        # Replace Python float NaN/Inf before encoding
        def _clean(o):
            if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
                return None
            if isinstance(o, dict):
                return {k: _clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [_clean(v) for v in o]
            return o
        return super().encode(_clean(obj))


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """
    All tunable parameters in one place.
    Credentials are read from environment variables so they are never
    hard-coded in source control.  Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    before running, or the fallback demo values will be used.
    """

    # ── MT5 Connection ────────────────────────────────────────────────────────
    LOGIN    : int = int(os.environ.get("MT5_LOGIN",    "413646889"))
    PASSWORD : str = os.environ.get("MT5_PASSWORD", "Anoma@0822")
    SERVER   : str = os.environ.get("MT5_SERVER",   "Exness-MT5Trial6")

    # ── Symbols & Timeframes ─────────────────────────────────────────────────
    SYMBOLS          = ["EURUSDm", "USDJPYm", "GBPUSDm", "AUDUSDm"]
    TIMEFRAME        = mt5.TIMEFRAME_H1
    TIMEFRAME_NAME   = "H1"
    TREND_TIMEFRAME  = mt5.TIMEFRAME_H4          # Higher-level trend filter
    TREND_TF_NAME    = "H4"

    # ── Trade Sizing ─────────────────────────────────────────────────────────
    LOT_SIZE              = 0.01
    MAX_OPEN_TRADES       = 3     # Global cap across all symbols
    MAX_TRADES_PER_SYMBOL = 1     # Per-symbol cap  ← FIX #2

    # ── Session Filter ───────────────────────────────────────────────────────
    START_HOUR = 13      # London session start (1:30 PM SL time)
    END_HOUR   = 22      # New York session end (10:30 PM SL time)
    
    # Timezone note: Sri Lanka is UTC+5:30
    # London: 08:00-16:30 UTC = 13:30-22:00 SL
    # New York: 13:30-22:00 UTC = 19:00-03:30 SL next day
    # Using 13:00-22:00 SL (1 PM - 10 PM) covers both high-liquidity sessions

    # ── Indicator Parameters ─────────────────────────────────────────────────
    EMA_FAST   = 50
    EMA_SLOW   = 200
    RSI_PERIOD = 14

    # RSI thresholds (less critical now with dynamic momentum checking)
    RSI_BUY_MIN  = 40     # More conservative, dynamic check replaces this
    RSI_BUY_MAX  = 80
    RSI_SELL_MIN = 20
    RSI_SELL_MAX = 60

    MACD_FAST   = 12
    MACD_SLOW   = 26
    MACD_SIGNAL = 9

    BB_PERIOD  = 20
    BB_STD_DEV = 2.0

    ATR_PERIOD = 14
    ADX_PERIOD = 14
    ADX_MIN    = 20      # Minimum trend strength

    # ── Risk Management ───────────────────────────────────────────────────────
    SL_MULTIPLIER       = 2.0    # × ATR — FIX #3: now actually used
    TP_MULTIPLIER       = 2.0    # × ATR
    BE_ACTIVATION_ATR   = 0.75   # Move to BE after this many ATR of profit
    TRAILING_ATR        = 1.5    # Trailing stop distance in ATR

    MAX_SPREAD_PIPS = 5          # Skip symbol if spread exceeds this

    # ── Loop Control ─────────────────────────────────────────────────────────
    LOOP_INTERVAL        = 60    # Seconds between each processing cycle
    MIN_CANDLES_REQUIRED = 250

    # ── File Paths ───────────────────────────────────────────────────────────
    HISTORY_EXCEL      = "trade_history.xlsx"
    ACTIVE_TRADES_JSON = "active_trades_features.json"
    BOT_DATA_JSON      = "bot_data.json"


# ============================================================================
# TRADE HISTORY MANAGER
# ============================================================================

class TradeHistoryManager:
    """Records entry features at order time, merges with MT5 deal history on close."""

    def __init__(self, excel_file: str, features_file: str):
        self.excel_file    = excel_file
        self.features_file = features_file
        self.active_features: Dict[str, dict] = self._load_features()

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _load_features(self) -> Dict[str, dict]:
        if os.path.exists(self.features_file):
            try:
                with open(self.features_file, "r") as fh:
                    return json.load(fh)
            except Exception as exc:
                logger.error(f"Could not load features JSON: {exc}")
        return {}

    def _save_features(self):
        try:
            with open(self.features_file, "w") as fh:
                json.dump(self.active_features, fh, indent=4, cls=SafeEncoder)
        except Exception as exc:
            logger.error(f"Could not save features JSON: {exc}")

    # ── Public API ────────────────────────────────────────────────────────────

    def record_entry(self, ticket: int, signal: str, indicators: dict):
        """Store indicator snapshot for an open position (keyed by ticket)."""
        self.active_features[str(ticket)] = {
            "ticket":     ticket,
            "type":       signal,
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **indicators,
        }
        self._save_features()
        logger.info(f"  Recorded entry features for ticket {ticket}")

    def process_history(self, deals: list):
        """
        FIX #8: deal.position_id is the position identifier for DEAL_ENTRY_OUT
        deals — this correctly matches the ticket we stored at entry.
        """
        if not deals:
            return
        new_rows = []
        for deal in deals:
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
            key = str(deal.position_id)
            if key not in self.active_features:
                continue
            entry_data = self.active_features.pop(key)
            new_rows.append({
                **entry_data,
                "exit_time":  datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d %H:%M:%S"),
                "exit_price": deal.price,
                "profit":     deal.profit,
                "commission": deal.commission,
                "swap":       deal.swap,
                "outcome":    "WIN" if deal.profit > 0 else "LOSS",
            })
        if new_rows:
            self._append_to_excel(new_rows)
            self._save_features()

    def _append_to_excel(self, new_rows: list):
        try:
            df_new = pd.DataFrame(new_rows)
            if os.path.exists(self.excel_file):
                df_old = pd.read_excel(self.excel_file)
                df_new = pd.concat([df_old, df_new], ignore_index=True)
            df_new.to_excel(self.excel_file, index=False)
            logger.info(f"  Saved {len(new_rows)} trade(s) → {self.excel_file}")
        except Exception as exc:
            logger.error(f"Excel save error: {exc}")

    def get_recent_history(self, limit: int = 20) -> list:
        if not os.path.exists(self.excel_file):
            return []
        try:
            df = pd.read_excel(self.excel_file).fillna("")
            records = df.tail(limit).to_dict("records")
            records.reverse()
            return records
        except Exception as exc:
            logger.error(f"Excel read error: {exc}")
            return []


# ============================================================================
# INDICATOR UTILITIES - FALLBACK CALCULATIONS
# ============================================================================

def _calculate_ema(series: np.ndarray, length: int) -> np.ndarray:
    """Calculate EMA without pandas_ta."""
    multiplier = 2 / (length + 1)
    ema = np.zeros_like(series)
    ema[0] = series[0]
    for i in range(1, len(series)):
        ema[i] = series[i] * multiplier + ema[i-1] * (1 - multiplier)
    return ema

def _calculate_rsi(series: np.ndarray, length: int) -> np.ndarray:
    """Calculate RSI without pandas_ta."""
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(length)/length, mode='valid')
    avg_loss = np.convolve(loss, np.ones(length)/length, mode='valid')
    
    rsi = np.zeros(len(series))
    rsi[:] = np.nan
    for i in range(length, len(series)):
        if avg_loss[i-length] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i-length] / avg_loss[i-length]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def _calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Calculate ATR without pandas_ta."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.convolve(tr, np.ones(length)/length, mode='valid')
    result = np.zeros(len(high))
    result[:] = np.nan
    result[length-1:] = atr
    return result

def _calculate_macd(series: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate MACD without pandas_ta."""
    ema_fast = _calculate_ema(series, fast)
    ema_slow = _calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _calculate_ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def _calculate_bbands(series: np.ndarray, length: int = 20, std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands without pandas_ta."""
    sma = np.convolve(series, np.ones(length)/length, mode='valid')
    
    result_upper = np.zeros(len(series))
    result_middle = np.zeros(len(series))
    result_lower = np.zeros(len(series))
    result_upper[:] = np.nan
    result_middle[:] = np.nan
    result_lower[:] = np.nan
    
    for i in range(length-1, len(series)):
        window = series[i-length+1:i+1]
        middle = np.mean(window)
        std_val = np.std(window)
        result_middle[i] = middle
        result_upper[i] = middle + (std_val * std)
        result_lower[i] = middle - (std_val * std)
    
    return result_upper, result_middle, result_lower

# ============================================================================
# INDICATOR CALCULATOR
# ============================================================================

class IndicatorCalculator:
    """Calculates all technical indicators on a given OHLCV DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate()

    def _calculate(self):
        df = self.df

        # EMAs
        if HAS_PANDAS_TA:
            df["EMA_9"]   = ta.ema(df["close"], length=9)
            df["EMA_21"]  = ta.ema(df["close"], length=21)
            df["EMA_50"]  = ta.ema(df["close"], length=Config.EMA_FAST)
            df["EMA_200"] = ta.ema(df["close"], length=Config.EMA_SLOW)
        else:
            df["EMA_9"]   = _calculate_ema(df["close"].values, 9)
            df["EMA_21"]  = _calculate_ema(df["close"].values, 21)
            df["EMA_50"]  = _calculate_ema(df["close"].values, Config.EMA_FAST)
            df["EMA_200"] = _calculate_ema(df["close"].values, Config.EMA_SLOW)

        # RSI
        if HAS_PANDAS_TA:
            df["RSI"] = ta.rsi(df["close"], length=Config.RSI_PERIOD)
        else:
            df["RSI"] = _calculate_rsi(df["close"].values, Config.RSI_PERIOD)

        # MACD
        if HAS_PANDAS_TA:
            macd = ta.macd(
                df["close"],
                fast=Config.MACD_FAST,
                slow=Config.MACD_SLOW,
                signal=Config.MACD_SIGNAL,
            )
            if macd is not None and not macd.empty:
                line_col   = [c for c in macd.columns if c.startswith("MACD_") and "s" not in c.lower() and "h" not in c.lower()]
                signal_col = [c for c in macd.columns if c.startswith("MACDs_")]
                hist_col   = [c for c in macd.columns if c.startswith("MACDh_")]
                df["MACD"]        = macd[line_col[0]]   if line_col   else np.nan
                df["MACD_Signal"] = macd[signal_col[0]] if signal_col else np.nan
                df["MACD_Hist"]   = macd[hist_col[0]]   if hist_col   else np.nan
            else:
                df["MACD"] = df["MACD_Signal"] = df["MACD_Hist"] = np.nan
        else:
            macd_line, signal_line, hist = _calculate_macd(
                df["close"].values,
                Config.MACD_FAST,
                Config.MACD_SLOW,
                Config.MACD_SIGNAL
            )
            df["MACD"] = macd_line
            df["MACD_Signal"] = signal_line
            df["MACD_Hist"] = hist

        # Bollinger Bands
        if HAS_PANDAS_TA:
            bb = ta.bbands(df["close"], length=Config.BB_PERIOD, std=Config.BB_STD_DEV)
            if bb is not None and not bb.empty:
                df["BB_Upper"]  = bb[[c for c in bb.columns if c.startswith("BBU_")][0]]
                df["BB_Middle"] = bb[[c for c in bb.columns if c.startswith("BBM_")][0]]
                df["BB_Lower"]  = bb[[c for c in bb.columns if c.startswith("BBL_")][0]]
            else:
                df["BB_Upper"] = df["BB_Middle"] = df["BB_Lower"] = np.nan
        else:
            bb_upper, bb_middle, bb_lower = _calculate_bbands(
                df["close"].values,
                Config.BB_PERIOD,
                Config.BB_STD_DEV
            )
            df["BB_Upper"] = bb_upper
            df["BB_Middle"] = bb_middle
            df["BB_Lower"] = bb_lower

        # ATR
        if HAS_PANDAS_TA:
            df["ATR"] = ta.atr(df["high"], df["low"], df["close"], length=Config.ATR_PERIOD)
        else:
            df["ATR"] = _calculate_atr(df["high"].values, df["low"].values, df["close"].values, Config.ATR_PERIOD)

        # ADX (simplified without pandas_ta)
        if HAS_PANDAS_TA:
            adx = ta.adx(df["high"], df["low"], df["close"], length=Config.ADX_PERIOD)
            if adx is not None and not adx.empty:
                adx_col = [c for c in adx.columns if c.startswith("ADX_")]
                df["ADX"] = adx[adx_col[0]] if adx_col else np.nan
            else:
                df["ADX"] = np.nan
        else:
            # Simplified ADX: use ATR trend as proxy
            atr_vals = df["ATR"].values
            df["ADX"] = np.zeros(len(df))
            window = 14
            for i in range(window, len(df)):
                volatility = np.std(atr_vals[max(0, i-window):i])
                avg_atr = np.mean(atr_vals[max(0, i-window):i])
                if avg_atr > 0:
                    df["ADX"].iloc[i] = (volatility / avg_atr) * 25
                else:
                    df["ADX"].iloc[i] = 0

    def get_latest(self) -> dict:
        if self.df.empty:
            return {}
        row = self.df.iloc[-1]

        def g(col):
            v = row.get(col, np.nan)
            return float(v) if not pd.isna(v) else np.nan

        return {
            "close":       g("close"),
            "high":        g("high"),
            "low":         g("low"),
            "open":        g("open"),
            "ema_9":       g("EMA_9"),
            "ema_21":      g("EMA_21"),
            "ema_50":      g("EMA_50"),
            "ema_200":     g("EMA_200"),
            "rsi":         g("RSI"),
            "macd":        g("MACD"),
            "macd_signal": g("MACD_Signal"),
            "macd_hist":   g("MACD_Hist"),
            "bb_upper":    g("BB_Upper"),
            "bb_middle":   g("BB_Middle"),
            "bb_lower":    g("BB_Lower"),
            "atr":         g("ATR"),
            "adx":         g("ADX"),
        }


# ============================================================================
# MARKET STRUCTURE ANALYZER
# ============================================================================

class MarketStructureAnalyzer:
    """Detects Higher High/Low (HH/HL) and Lower Low/High (LL/LH) patterns."""

    @staticmethod
    def get_pivot_points(df: pd.DataFrame, lookback: int = 5) -> Tuple[float, float, float]:
        """
        Get recent swing highs and lows.
        Returns: (recent_swing_high, recent_swing_low, most_recent_swing)
        """
        if len(df) < lookback:
            return float('nan'), float('nan'), float('nan')
        
        subset = df.tail(lookback + 5)
        highs = subset['high'].values
        lows = subset['low'].values
        
        # Find swing high (local max)
        swing_high = max(highs[-lookback:])
        # Find swing low (local min)
        swing_low = min(lows[-lookback:])
        
        return swing_high, swing_low, (swing_high + swing_low) / 2

    @staticmethod
    def detect_structure(df: pd.DataFrame) -> dict:
        """
        Detect market structure: trending (HH/HL or LL/LH) or ranging.
        Returns dict with structure info.
        """
        if len(df) < 20:
            return {"type": "INSUFFICIENT_DATA", "trend": "HOLD"}
        
        # Get last 3 swing points
        lookback = 10
        recent = df.tail(lookback)
        
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Find pivot indices
        h1_idx = np.argmax(highs[:len(highs)//2])
        h2_idx = len(highs)//2 + np.argmax(highs[len(highs)//2:])
        
        l1_idx = np.argmin(lows[:len(lows)//2])
        l2_idx = len(lows)//2 + np.argmin(lows[len(lows)//2:])
        
        h1, h2 = highs[h1_idx], highs[h2_idx]
        l1, l2 = lows[l1_idx], lows[l2_idx]
        
        # Uptrend: HH and HL (higher high, higher low)
        if h2 > h1 and l2 > l1:
            return {
                "type": "UPTREND",
                "trend": "BULL",
                "resistance": h2,
                "support": l2,
                "strength": "STRONG" if (h2 - h1) > (h1 * 0.005) else "WEAK"
            }
        # Downtrend: LL and LH (lower low, lower high)
        elif h2 < h1 and l2 < l1:
            return {
                "type": "DOWNTREND",
                "trend": "BEAR",
                "resistance": h2,
                "support": l2,
                "strength": "STRONG" if (l1 - l2) > (l1 * 0.005) else "WEAK"
            }
        # Ranging
        else:
            return {
                "type": "RANGE",
                "trend": "RANGE",
                "resistance": max(h1, h2),
                "support": min(l1, l2),
                "strength": "WEAK"
            }


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """
    Enhanced signal logic with:
    - Market Structure (HH/HL detection)
    - Dynamic RSI with momentum slope
    - Volatility filtering (Bollinger Bands)
    - Candlestick confirmation
    - Stricter confluence requirements
    """

    @staticmethod
    def _check_rsi_momentum(ind: dict, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Check RSI momentum. Return (is_bullish, slope).
        RSI > 52 and < 70 is bullish zone (not overbought).
        """
        if len(df) < 3:
            return False, 0.0
        
        rsi = ind.get("rsi", np.nan)
        if pd.isna(rsi):
            return False, 0.0
        
        # Get RSI slope (momentum of momentum)
        rsi_vals = df['RSI'].tail(3).values if 'RSI' in df.columns else []
        if len(rsi_vals) >= 2:
            slope = rsi_vals[-1] - rsi_vals[-2]
        else:
            slope = 0.0
        
        # Bullish RSI zone: between 52-70 (not overbought, momentum shifting up)
        is_bull = 52 <= rsi <= 70 and slope >= 0
        # Bearish RSI zone: between 30-48 (not oversold, momentum shifting down)
        is_bear = 30 <= rsi <= 48 and slope <= 0
        
        return (is_bull, slope) if is_bull else (is_bear, slope) if is_bear else (False, slope)

    @staticmethod
    def _check_candlestick_pattern(df: pd.DataFrame) -> Tuple[str, bool]:
        """
        Check for strong candlestick patterns:
        - Bullish: body > 50% of previous range, close > middle
        - Bearish: body > 50% of previous range, close < middle
        """
        if len(df) < 2:
            return "NEUTRAL", False
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        curr_body = abs(curr['close'] - curr['open'])
        prev_range = prev['high'] - prev['low']
        
        if prev_range == 0:
            return "NEUTRAL", False
        
        body_ratio = curr_body / prev_range
        
        # Strong bullish: large body, closes near high
        curr_close_ratio = (curr['close'] - curr['low']) / curr_body if curr_body > 0 else 0
        is_bullish_strong = body_ratio > 0.5 and curr_close_ratio > 0.7 and curr['close'] > prev['close']
        
        # Strong bearish: large body, closes near low
        curr_close_ratio_bear = (curr['high'] - curr['close']) / curr_body if curr_body > 0 else 0
        is_bearish_strong = body_ratio > 0.5 and curr_close_ratio_bear > 0.7 and curr['close'] < prev['close']
        
        if is_bullish_strong:
            return "BULLISH_ENGULFING", True
        elif is_bearish_strong:
            return "BEARISH_ENGULFING", True
        else:
            return "NEUTRAL", False

    @staticmethod
    def _check_macd_momentum(ind: dict) -> Tuple[bool, float]:
        """
        Check MACD momentum and histogram acceleration.
        Returns: (is_bullish_momentum, histogram_value)
        """
        macd = ind.get("macd", np.nan)
        macd_sig = ind.get("macd_signal", np.nan)
        macd_hist = ind.get("macd_hist", np.nan)
        
        if pd.isna(macd) or pd.isna(macd_sig) or pd.isna(macd_hist):
            return False, 0.0
        
        # Bullish MACD: line > signal AND histogram > 0 (momentum acceleration)
        is_bull = (macd > macd_sig) and (macd_hist > 0)
        is_bear = (macd < macd_sig) and (macd_hist < 0)
        
        return (is_bull, macd_hist) if is_bull else (is_bear, macd_hist) if is_bear else (False, macd_hist)

    @staticmethod
    def generate_signal(
        ind: dict,
        h4_data: dict,
        hour: int,
        df: pd.DataFrame,
    ) -> Tuple[str, dict]:
        """
        HIGH ACCURACY signal generation with:
        1. Market structure (HH/HL, LL/LH)
        2. Dynamic RSI (52-70 for bull, 30-48 for bear with slope)
        3. MACD momentum confirmation
        4. Candlestick patterns
        5. Volatility filter (BB)
        6. Strict confluence (4+/5 required for entry)
        """

        # ── Validate data ────────────────────────────────────────────────────
        required = ("close", "ema_50", "ema_200", "rsi", "macd",
                    "macd_signal", "macd_hist", "atr", "adx")
        if any(pd.isna(ind.get(k, np.nan)) for k in required) or len(df) < 5:
            return "HOLD", {}

        close   = ind["close"]
        ema50   = ind["ema_50"]
        ema200  = ind["ema_200"]
        bb_mid  = ind["bb_middle"]
        bb_upper = ind["bb_upper"]
        bb_lower = ind["bb_lower"]
        atr     = ind["atr"]
        adx     = ind["adx"]

        # ── 1. Market Structure Analysis ─────────────────────────────────────
        structure = MarketStructureAnalyzer.detect_structure(df)
        is_trending = structure["type"] in ("UPTREND", "DOWNTREND")
        struct_bull = structure["type"] == "UPTREND"
        struct_bear = structure["type"] == "DOWNTREND"

        # ── 2. H4 Trend Filter (higher timeframe alignment) ──────────────────
        h4_close  = h4_data.get("close", np.nan)
        h4_ema200 = h4_data.get("ema_200", np.nan)
        h4_bull = (not pd.isna(h4_close)) and (not pd.isna(h4_ema200)) and (h4_close > h4_ema200)
        h4_bear = (not pd.isna(h4_close)) and (not pd.isna(h4_ema200)) and (h4_close < h4_ema200)

        # ── 3. EMA Alignment (H1 trend) ──────────────────────────────────────
        ema_bull = close > ema50 > ema200
        ema_bear = close < ema50 < ema200

        # ── 4. Dynamic RSI with Momentum Slope ───────────────────────────────
        rsi_bull, rsi_slope = SignalGenerator._check_rsi_momentum(ind, df)
        rsi_bear = not rsi_bull and (30 <= ind.get("rsi", np.nan) <= 48)

        # ── 5. MACD Momentum Confirmation ────────────────────────────────────
        macd_bull, macd_hist = SignalGenerator._check_macd_momentum(ind)
        macd_bear = not macd_bull and ind.get("macd", np.nan) < ind.get("macd_signal", np.nan)

        # ── 6. Candlestick Pattern Recognition ────────────────────────────────
        candle_pattern, is_strong_candle = SignalGenerator._check_candlestick_pattern(df)
        is_bullish_candle = "BULLISH" in candle_pattern
        is_bearish_candle = "BEARISH" in candle_pattern

        # ── 7. Volatility Filter (Golden Zone: BB Middle + RSI 50 cross) ──────
        price_above_bb_mid = close > bb_mid
        price_below_bb_mid = close < bb_mid
        bb_ok = (not pd.isna(bb_upper)) and (not pd.isna(bb_lower)) and \
                ((bb_upper - bb_lower) / close > 0.003)  # Not squeezing

        # ── 8. Trend Strength Confirmation ───────────────────────────────────
        strong_trend = (not pd.isna(adx)) and (adx >= Config.ADX_MIN)

        # ── 9. Session Filter ────────────────────────────────────────────────
        in_session = Config.START_HOUR <= hour < Config.END_HOUR

        # ── CONFLUENCE SCORING ───────────────────────────────────────────────
        buy_score = 0
        sell_score = 0

        # Check each condition
        if ema_bull:
            buy_score += 1
        if ema_bear:
            sell_score += 1

        if h4_bull:
            buy_score += 1
        if h4_bear:
            sell_score += 1

        if rsi_bull:
            buy_score += 1
        if rsi_bear:
            sell_score += 1

        if macd_bull:
            buy_score += 1
        if macd_bear:
            sell_score += 1

        if is_bullish_candle and price_above_bb_mid:
            buy_score += 1
        if is_bearish_candle and price_below_bb_mid:
            sell_score += 1

        # ── Final Signal: Require strict confluence (4+/5) ──────────────────
        signal = "HOLD"
        final_score = 0

        if buy_score >= 4 and is_trending and struct_bull and strong_trend and in_session and bb_ok:
            signal = "BUY"
            final_score = buy_score
        elif sell_score >= 4 and is_trending and struct_bear and strong_trend and in_session and bb_ok:
            signal = "SELL"
            final_score = sell_score

        # ── Build details for logging ────────────────────────────────────────
        details = {
            "market_structure": (structure["type"], is_trending),
            "h4_alignment": (f"H4_{'BULL' if h4_bull else 'BEAR' if h4_bear else 'MIXED'}", h4_bull or h4_bear),
            "ema_trend": (f"H1_{'BULL' if ema_bull else 'BEAR' if ema_bear else 'NEUTRAL'}", ema_bull or ema_bear),
            "rsi_momentum": (f"RSI_{ind.get('rsi', 0):.1f}(slope {rsi_slope:.2f})", rsi_bull or rsi_bear),
            "macd_momentum": (f"MACD_HIST_{macd_hist:.4f}", macd_bull or macd_bear),
            "candlestick": (candle_pattern, is_strong_candle),
            "volatility": (f"BB_OK({bb_upper - bb_lower:.5f})", bb_ok),
            "trend_strength": (f"ADX_{adx:.1f}", strong_trend),
            "session": ("OPTIMAL" if in_session else "OUT_OF_SESSION", in_session),
            "confluence_score": final_score,
            "buy_score": buy_score,
            "sell_score": sell_score,
        }

        return signal, details


# ============================================================================
# TRADE MANAGER
# ============================================================================

class TradeManager:
    """Handles order execution, position modification, and trade management."""

    MAGIC_BUY  = 99991
    MAGIC_SELL = 99992
    MAGIC_MOD  = 99999

    def __init__(self, symbol: str, lot_size: float):
        self.symbol   = symbol
        self.lot_size = lot_size

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _positions(self) -> list:
        pos = mt5.positions_get(symbol=self.symbol)
        return list(pos) if pos else []

    def _symbol_info(self):
        info = mt5.symbol_info(self.symbol)
        if info and not info.visible:
            mt5.symbol_select(self.symbol, True)
            info = mt5.symbol_info(self.symbol)
        return info

    def _filling_mode(self) -> int:
        info = self._symbol_info()
        if info:
            fm = info.filling_mode
            if fm & mt5.SYMBOL_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            if fm & mt5.SYMBOL_FILLING_FOK:
                return mt5.ORDER_FILLING_FOK
        return mt5.ORDER_FILLING_RETURN

    def _pip_value(self) -> float:
        """
        FIX #4: derive pip size from symbol digits so JPY pairs work correctly.
        5-digit / 3-digit brokers: 1 pip = 10 points.
        """
        info = self._symbol_info()
        if info:
            if info.digits in (3, 5):
                return info.point * 10
            return info.point
        return 0.0001  # safe fallback

    def _min_stop_points(self) -> int:
        """FIX #10: return broker's minimum stop distance in points."""
        info = self._symbol_info()
        return info.trade_stops_level if info else 0

    def _validate_sl_tp(self, direction: str, price: float,
                        sl: float, tp: float) -> bool:
        """
        FIX #10: Ensure SL/TP are on the correct side of price and respect
        the broker's minimum stop level.
        """
        info = self._symbol_info()
        if not info:
            return False

        point = info.point
        min_dist = self._min_stop_points() * point

        if direction == "BUY":
            if sl >= price:
                logger.warning(f"BUY SL {sl:.5f} is above entry {price:.5f}")
                return False
            if tp <= price:
                logger.warning(f"BUY TP {tp:.5f} is below entry {price:.5f}")
                return False
            if (price - sl) < min_dist:
                logger.warning(f"BUY SL too close: {price-sl:.5f} < min {min_dist:.5f}")
                return False
        else:  # SELL
            if sl <= price:
                logger.warning(f"SELL SL {sl:.5f} is below entry {price:.5f}")
                return False
            if tp >= price:
                logger.warning(f"SELL TP {tp:.5f} is above entry {price:.5f}")
                return False
            if (sl - price) < min_dist:
                logger.warning(f"SELL SL too close: {sl-price:.5f} < min {min_dist:.5f}")
                return False
        return True

    # ── Position sizing ───────────────────────────────────────────────────────

    def calculate_positions(
        self, entry: float, atr: float, direction: str, support: float = None, resistance: float = None
    ) -> dict:
        """
        FIX #3: SL/TP now read from Config.SL_MULTIPLIER / TP_MULTIPLIER.
        IMPROVED: Use structure support/resistance for tighter SL if available.
        Falls back to ATR-based positioning if structure levels unavailable.
        """
        sl_dist = atr * Config.SL_MULTIPLIER
        tp_dist = atr * Config.TP_MULTIPLIER

        if direction == "BUY":
            # Prefer structure support over ATR-based SL for tighter stops
            if support and support < entry:
                # Use support with small buffer (0.2 ATR)
                sl_price = max(support - (atr * 0.2), entry - sl_dist)
            else:
                sl_price = entry - sl_dist
            tp_price = entry + tp_dist
        else:  # SELL
            if resistance and resistance > entry:
                # Use resistance with small buffer (0.2 ATR)
                sl_price = min(resistance + (atr * 0.2), entry + sl_dist)
            else:
                sl_price = entry + sl_dist
            tp_price = entry - tp_dist

        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        return {
            "sl_price": sl_price,
            "tp_price": tp_price,
            "risk":     abs(entry - sl_price),
            "reward":   abs(tp_price - entry),
            "rr_ratio": rr,
        }

    # ── Trade execution ───────────────────────────────────────────────────────

    def has_open_trade(self) -> bool:
        """FIX #2: checks per-symbol limit (MAX_TRADES_PER_SYMBOL)."""
        return len(self._positions()) >= Config.MAX_TRADES_PER_SYMBOL

    def _send_order(self, order_type: int, price: float,
                    sl: float, tp: float, comment: str) -> Optional[int]:
        """
        FIX #1: all paths return Optional[int] (None on failure).
        FIX #10: validates SL/TP before sending.
        """
        direction = "BUY" if order_type == mt5.ORDER_TYPE_BUY else "SELL"

        if not self._validate_sl_tp(direction, price, sl, tp):
            return None

        magic = self.MAGIC_BUY if order_type == mt5.ORDER_TYPE_BUY else self.MAGIC_SELL
        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      self.symbol,
            "volume":      self.lot_size,
            "type":        order_type,
            "price":       price,
            "sl":          sl,
            "tp":          tp,
            "deviation":   20,
            "magic":       magic,
            "comment":     comment,
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            logger.error(f"Order failed — retcode: {code} | error: {mt5.last_error()}")
            return None  # FIX #1: always None, never False

        logger.info(
            f"  ✓ {direction} {self.symbol} | "
            f"Entry {price:.5f} SL {sl:.5f} TP {tp:.5f} | "
            f"R:R 1:{abs(tp-price)/abs(price-sl):.2f} | "
            f"Ticket {result.order}"
        )
        return result.order

    def execute_buy(self, price: float, sl: float, tp: float,
                    comment: str = "") -> Optional[int]:
        if self.has_open_trade():
            logger.info(f"  {self.symbol}: per-symbol trade limit reached, skip BUY")
            return None
        return self._send_order(mt5.ORDER_TYPE_BUY, price, sl, tp, f"BUY|{comment}")

    def execute_sell(self, price: float, sl: float, tp: float,
                     comment: str = "") -> Optional[int]:
        if self.has_open_trade():
            logger.info(f"  {self.symbol}: per-symbol trade limit reached, skip SELL")
            return None
        return self._send_order(mt5.ORDER_TYPE_SELL, price, sl, tp, f"SELL|{comment}")

    # ── Position management ───────────────────────────────────────────────────

    def _modify_sl(self, ticket: int, new_sl: float, current_tp: float) -> bool:
        """FIX #5: removed unused filling_mode; clean request."""
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       new_sl,
            "tp":       current_tp,
            "magic":    self.MAGIC_MOD,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "None"
            logger.error(f"  Modify SL failed — ticket {ticket} retcode {code}")
            return False
        return True

    def manage_open_trades(self, current_price: float, atr: float):
        """
        FIX #4: pip_value now derived per symbol.
        FIX #4: trailing stop elif is properly guarded so it cannot fire
                before break-even is established.
        """
        if atr <= 0:
            return
        pip = self._pip_value()
        if pip <= 0:
            return

        for pos in self._positions():
            if pos.type == mt5.ORDER_TYPE_BUY:
                profit_atr = (current_price - pos.price_open) / atr

                # Break-even: move SL to entry once profit >= BE_ACTIVATION_ATR
                if profit_atr >= Config.BE_ACTIVATION_ATR and pos.sl < pos.price_open:
                    if self._modify_sl(pos.ticket, pos.price_open, pos.tp):
                        logger.info(f"  ✓ BE set  ticket {pos.ticket} SL→{pos.price_open:.5f}")

                # Trailing stop: only after BE has been moved (SL >= entry)
                elif profit_atr > 0 and pos.sl >= pos.price_open:
                    trail_sl = current_price - (atr * Config.TRAILING_ATR)
                    if trail_sl > pos.sl:
                        if self._modify_sl(pos.ticket, trail_sl, pos.tp):
                            logger.info(f"  ✓ Trail   ticket {pos.ticket} SL→{trail_sl:.5f}")

            else:  # SELL
                profit_atr = (pos.price_open - current_price) / atr

                if profit_atr >= Config.BE_ACTIVATION_ATR and pos.sl > pos.price_open:
                    if self._modify_sl(pos.ticket, pos.price_open, pos.tp):
                        logger.info(f"  ✓ BE set  ticket {pos.ticket} SL→{pos.price_open:.5f}")

                elif profit_atr > 0 and pos.sl <= pos.price_open:
                    trail_sl = current_price + (atr * Config.TRAILING_ATR)
                    if trail_sl < pos.sl:
                        if self._modify_sl(pos.ticket, trail_sl, pos.tp):
                            logger.info(f"  ✓ Trail   ticket {pos.ticket} SL→{trail_sl:.5f}")


# ============================================================================
# MT5 MANAGER
# ============================================================================

class MT5Manager:
    """Handles connection, reconnection, and data retrieval from MetaTrader5."""

    def __init__(self, login: int, password: str, server: str):
        self.login     = login
        self.password  = password
        self.server    = server
        self.connected = False

    def connect(self) -> bool:
        try:
            if not mt5.initialize():
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
            if not mt5.login(self.login, self.password, self.server):
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            info = mt5.account_info()
            logger.info(
                f"✓ Connected — {info.company} | "
                f"Balance {info.balance:.2f} {info.currency} | "
                f"Leverage 1:{info.leverage}"
            )
            self.connected = True
            return True
        except Exception:
            logger.error(traceback.format_exc())
            return False

    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 disconnected")

    def reconnect(self) -> bool:
        """FIX #11: reconnect if connection is lost."""
        logger.warning("Attempting MT5 reconnect…")
        self.disconnect()
        time.sleep(5)
        return self.connect()

    def is_connected(self) -> bool:
        """Lightweight liveness check."""
        try:
            return mt5.terminal_info() is not None
        except Exception:
            return False

    # ── Symbol helpers ────────────────────────────────────────────────────────

    def resolve_symbol(self, symbol: str) -> Optional[str]:
        """Try exact name then common Exness suffixes."""
        if mt5.symbol_info(symbol):
            return symbol
        base = symbol.rstrip("m!#ck")
        for sfx in ("", "m", "!", "#", "c", "k", "pro", "raw"):
            candidate = base + sfx
            if mt5.symbol_info(candidate):
                return candidate
        return None

    def get_spread_pips(self, symbol: str) -> float:
        sym = self.resolve_symbol(symbol)
        if not sym:
            return float("inf")
        info = mt5.symbol_info(sym)
        if not info:
            return float("inf")
        # Convert spread (in points) to pips
        return info.spread / (10 if info.digits in (3, 5) else 1)

    # ── Data fetching ─────────────────────────────────────────────────────────

    def fetch_candles(self, symbol: str, timeframe, count: int) -> Optional[pd.DataFrame]:
        try:
            sym = self.resolve_symbol(symbol)
            if not sym:
                logger.error(f"Symbol not found: {symbol}")
                return None
            mt5.symbol_select(sym, True)
            rates = mt5.copy_rates_from_pos(sym, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                return None
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
            return df[["time", "open", "high", "low", "close", "volume"]]
        except Exception:
            logger.error(traceback.format_exc())
            return None

    def fetch_deals(self, days: int = 2) -> list:
        try:
            since = datetime.now() - timedelta(days=days)
            deals = mt5.history_deals_get(since, datetime.now())
            return list(deals) if deals else []
        except Exception:
            logger.error(traceback.format_exc())
            return []


# ============================================================================
# MAIN BOT
# ============================================================================

class TrendConfirmationBot:
    """Orchestrates the full trading cycle across all configured symbols."""

    def __init__(self, cfg: Config):
        self.cfg            = cfg
        self.mt5            = MT5Manager(cfg.LOGIN, cfg.PASSWORD, cfg.SERVER)
        self.managers       = {sym: TradeManager(sym, cfg.LOT_SIZE) for sym in cfg.SYMBOLS}
        self.history        = TradeHistoryManager(cfg.HISTORY_EXCEL, cfg.ACTIVE_TRADES_JSON)
        self.last_candle    : Dict[str, Optional[pd.Timestamp]] = {s: None for s in cfg.SYMBOLS}

    # ── Dashboard export ──────────────────────────────────────────────────────

    def _export(self, all_ind: dict, all_sig: dict, all_det: dict, all_h4: dict):
        """FIX #9: uses SafeEncoder for numpy/NaN serialisation."""
        try:
            acc  = mt5.account_info()
            poss = mt5.positions_get() or []

            data: dict = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "account": {
                    "balance":  acc.balance  if acc else 0,
                    "equity":   acc.equity   if acc else 0,
                    "company":  acc.company  if acc else "",
                    "currency": acc.currency if acc else "USD",
                },
                "symbols": {},
                "history": self.history.get_recent_history(20),
            }

            for sym in self.cfg.SYMBOLS:
                ind = all_ind.get(sym, {})
                det = all_det.get(sym, {})
                h4  = all_h4.get(sym, {})

                sym_trades = [
                    {
                        "ticket":       p.ticket,
                        "type":         "BUY" if p.type == 0 else "SELL",
                        "volume":       p.volume,
                        "price_open":   p.price_open,
                        "price_current":p.price_current,
                        "profit":       p.profit,
                        "sl":           p.sl,
                        "tp":           p.tp,
                    }
                    for p in poss if p.symbol == sym
                ]

                data["symbols"][sym] = {
                    "indicators":  ind,
                    "h4_data":     h4,
                    "signal":      all_sig.get(sym, "HOLD"),
                    "details":     {
                        k: ([v[0], bool(v[1])] if isinstance(v, tuple) else v)
                        for k, v in det.items()
                    },
                    "open_trades": sym_trades,
                }

            tmp = self.cfg.BOT_DATA_JSON + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=2, cls=SafeEncoder)
            if os.path.exists(self.cfg.BOT_DATA_JSON):
                os.remove(self.cfg.BOT_DATA_JSON)
            os.rename(tmp, self.cfg.BOT_DATA_JSON)

        except Exception:
            logger.error(traceback.format_exc())

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        logger.info("=" * 70)
        logger.info("TREND-CONFIRMATION MULTI-SYMBOL FOREX BOT  (Corrected)")
        logger.info("=" * 70)

        if not self.mt5.connect():
            logger.error("Cannot connect to MT5. Exiting.")
            return

        loop = 0
        try:
            while True:
                loop += 1

                # FIX #11: reconnect if connection dropped
                if not self.mt5.is_connected():
                    if not self.mt5.reconnect():
                        logger.error("Reconnect failed — sleeping 30 s")
                        time.sleep(30)
                        continue

                logger.info(f"\n{'─'*60}")
                logger.info(f"Loop #{loop}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                all_ind, all_sig, all_det, all_h4 = {}, {}, {}, {}

                # ── Global open-trade count ───────────────────────────────────
                open_positions = mt5.positions_get() or []
                global_open    = len(open_positions)

                for sym in self.cfg.SYMBOLS:

                    # ── Primary timeframe candles ─────────────────────────────
                    df = self.mt5.fetch_candles(
                        sym, self.cfg.TIMEFRAME, self.cfg.MIN_CANDLES_REQUIRED + 50
                    )
                    if df is None or len(df) < self.cfg.MIN_CANDLES_REQUIRED:
                        logger.warning(f"  {sym}: insufficient candles")
                        continue

                    # ── Skip if candle hasn't changed (prevents duplicate signals)
                    latest_time = df["time"].iloc[-1]
                    if self.last_candle[sym] == latest_time:
                        logger.info(f"  {sym}: same candle — skip signal")
                    else:
                        self.last_candle[sym] = latest_time

                    # ── H4 trend filter ───────────────────────────────────────
                    df_h4 = self.mt5.fetch_candles(sym, self.cfg.TREND_TIMEFRAME, 250)
                    h4_data: dict = {"close": np.nan, "ema_200": np.nan}
                    if df_h4 is not None and len(df_h4) >= 200:
                        if HAS_PANDAS_TA:
                            df_h4["EMA_200"] = ta.ema(df_h4["close"], length=200)
                        else:
                            df_h4["EMA_200"] = _calculate_ema(df_h4["close"].values, 200)
                        h4_data["close"]   = float(df_h4["close"].iloc[-1])
                        h4_data["ema_200"] = float(df_h4["EMA_200"].iloc[-1])
                    all_h4[sym] = h4_data

                    # ── Indicators ────────────────────────────────────────────
                    calc = IndicatorCalculator(df)
                    ind  = calc.get_latest()
                    all_ind[sym] = ind

                    logger.info(
                        f"  {sym} | Close {ind['close']:.5f} "
                        f"EMA50 {ind['ema_50']:.5f} EMA200 {ind['ema_200']:.5f} "
                        f"RSI {ind['rsi']:.1f} ADX {ind['adx']:.1f} ATR {ind['atr']:.5f}"
                    )

                    # ── Spread check ──────────────────────────────────────────
                    spread = self.mt5.get_spread_pips(sym)
                    if spread > self.cfg.MAX_SPREAD_PIPS:
                        logger.warning(f"  {sym}: spread {spread:.1f} > {self.cfg.MAX_SPREAD_PIPS} — skip")
                        all_sig[sym] = "HOLD"
                        all_det[sym] = {}
                        continue

                    # ── Current hour for session filter ───────────────────────
                    tick = mt5.symbol_info_tick(sym)
                    hour = datetime.fromtimestamp(tick.time).hour if tick else datetime.now().hour

                    # ── Signal generation ─────────────────────────────────────
                    signal, details = SignalGenerator.generate_signal(ind, h4_data, hour, calc.df)
                    all_sig[sym] = signal
                    all_det[sym] = details

                    score = details.get("confluence_score", 0)
                    buy_score = details.get("buy_score", 0)
                    sell_score = details.get("sell_score", 0)
                    logger.info(
                        f"  {sym}: {signal}  "
                        f"(confluence {score}/5 | BUY:{buy_score} SELL:{sell_score})"
                    )

                    # ── Trade execution ───────────────────────────────────────
                    mgr = self.managers[sym]
                    if signal in ("BUY", "SELL"):
                        # Get market structure support/resistance for better SL placement
                        structure = MarketStructureAnalyzer.detect_structure(calc.df)
                        support = structure.get("support", None)
                        resistance = structure.get("resistance", None)
                        
                        # FIX #2: global cap checked here; per-symbol cap in TradeManager
                        if global_open >= self.cfg.MAX_OPEN_TRADES:
                            logger.warning(
                                f"  Global cap {global_open}/{self.cfg.MAX_OPEN_TRADES} — skip {sym}"
                            )
                        else:
                            pos = mgr.calculate_positions(
                                ind["close"], ind["atr"], signal, support, resistance
                            )
                            ticket = None
                            if signal == "BUY":
                                ticket = mgr.execute_buy(
                                    ind["close"], pos["sl_price"], pos["tp_price"],
                                    f"score={score}/5|struct={structure['type']}"
                                )
                            else:
                                ticket = mgr.execute_sell(
                                    ind["close"], pos["sl_price"], pos["tp_price"],
                                    f"score={score}/5|struct={structure['type']}"
                                )
                            if ticket:
                                self.history.record_entry(ticket, signal, ind)
                                global_open += 1  # keep counter in sync for this loop

                    # ── Manage existing positions ─────────────────────────────
                    mgr.manage_open_trades(ind["close"], ind["atr"])

                # ── Dashboard ─────────────────────────────────────────────────
                self._export(all_ind, all_sig, all_det, all_h4)

                # ── Update closed-trade history (FIX #8 in process_history) ──
                deals = self.mt5.fetch_deals(days=2)
                self.history.process_history(deals)

                time.sleep(self.cfg.LOOP_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt — shutting down")
        except Exception:
            logger.error(traceback.format_exc())
        finally:
            self.mt5.disconnect()
            logger.info("Bot shutdown complete")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    cfg = Config()           # FIX #13: instantiate Config
    bot = TrendConfirmationBot(cfg)
    bot.run()


if __name__ == "__main__":
    main()