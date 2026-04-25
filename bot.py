"""
╔══════════════════════════════════════════════════════════════╗
║           FOREX TRADING BOT — ALL-IN-ONE                    ║
║   Technical Analysis + ML Prediction + Risk Management      ║
║   MetaTrader 5 Integration                                   ║
╚══════════════════════════════════════════════════════════════╝

SETUP:
  1. Install:  pip install MetaTrader5 pandas numpy scikit-learn joblib
  2. Set credentials via environment variables (NEVER hardcode!):
       Windows:  set MT5_LOGIN=YOUR_LOGIN
                 set MT5_PASSWORD=YOUR_PASSWORD
                 set MT5_SERVER=YOUR_SERVER
       Mac/Linux: export MT5_LOGIN=YOUR_LOGIN  (etc.)
  3. Run:      python forex_bot_all_in_one.py

⚠  Always test on a DEMO account first. Forex carries substantial risk.
"""

# ══════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════
import os
import time
import argparse
import logging
import warnings
import joblib
import requests
import csv
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    import MetaTrader5 as mt5
except ImportError:
    raise SystemExit("MetaTrader5 not installed. Run:  pip install MetaTrader5")

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler("forex_bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ForexBot")


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════════
class Config:
    """
    All bot settings in one place.
    Credentials are loaded from environment variables for safety.
    """

    # ── MT5 Credentials (set via env vars!) ───────────────────
    LOGIN    = int(os.getenv("MT5_LOGIN",    "413646889"))
    PASSWORD = os.getenv("MT5_PASSWORD",     "Anoma@0822")
    SERVER   = os.getenv("MT5_SERVER",       "Exness-MT5Trial6")

    # ── Symbols to trade ──────────────────────────────────────
    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD"]

    # ── Timeframes ────────────────────────────────────────────
    TIMEFRAMES      = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1]
    TREND_TIMEFRAME = mt5.TIMEFRAME_H4   # Secondary filter (default H4)
    USE_TREND_FILTER = True             # Enabled for safety

    # ── Risk Management ───────────────────────────────────────
    RISK_PCT           = 1.0    # % of balance per trade
    MAX_DRAWDOWN_PCT   = 10.0   # stop bot if total drawdown hits this
    MAX_DAILY_TRADES   = 10     # max orders per day
    MAX_DAILY_LOSS_PCT = 3.0    # daily loss cap (% of balance)

    # ── Trade Execution (Dynamic SL/TP) ───────────────────────
    ATR_SL_MULT  = 2.0   # Stop Loss = 2.0 * ATR
    ATR_TP_MULT  = 3.0   # Take Profit = 3.0 * ATR (better win rate)
    BREAK_EVEN_ATR_MULT = 1.2 # Move to break-even after 1.2 ATR profit
    TRAILING_SL  = True  # Enable Trailing Stop Loss
    
    DEFAULT_LOT  = 0.01   # fallback lot size
    MAX_LOT      = 0.05   # hard cap per trade

    # ── News Filter ───────────────────────────────────────────
    USE_NEWS_FILTER  = True
    NEWS_URL         = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    NEWS_BUFFER_MINS = 15  # Stop trading 30m before/after high impact news

    # ── ML Settings ───────────────────────────────────────────
    ML_LOOKBACK             = 20    # feature window (bars)
    ML_CONFIDENCE_THRESHOLD = 0.60  # min confidence to act on ML signal
    ML_RETRAIN_CANDLES      = 1000  # retrain when this many new candles seen
    LOG_FILE_TRADES         = "trade_logs.csv"
    
    # ── Anti-Noise Filter ─────────────────────────────────────
    MIN_ADX                 = 22    # Only trade if trend strength is > 22
    EMA_SQUEEZE_THRESHOLD   = 0.0010 # 10 pips max gap between EMAs (avoid sideways)

    # ── Bot Loop ──────────────────────────────────────────────
    LOOP_INTERVAL = 60    # seconds between each market scan

    # ── Misc ──────────────────────────────────────────────────
    MAGIC_NUMBER  = 202401
    DATA_HISTORY  = 500   # candles fetched per scan
    TRAIN_HISTORY = 2000  # candles used for initial ML training (was 1000)
    MODEL_DIR     = "models"

    # ── Timeframe Mapping ─────────────────────────────────────
    TF_MAP = {
        "M1":  1,      # mt5.TIMEFRAME_M1
        "M5":  5,      # mt5.TIMEFRAME_M5
        "M15": 15,     # mt5.TIMEFRAME_M15
        "M30": 30,     # mt5.TIMEFRAME_M30
        "H1":  16385,  # mt5.TIMEFRAME_H1
        "H4":  16388,  # mt5.TIMEFRAME_H4
        "D1":  16408   # mt5.TIMEFRAME_D1
    }

    # -- ML Labeling --
    # Threshold for a "significant" move to label as BUY/SELL
    # M15: 0.0003 (3 pips), H1: 0.0010 (10 pips)
    LABEL_THRESHOLD = 0.0010 


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════
class TechnicalAnalysis:
    """
    Computes 8 indicators and returns a voted BUY/SELL/HOLD signal.

    Indicators:
      EMA 20/50/200  — trend direction
      RSI (14)       — overbought / oversold
      MACD (12,26,9) — momentum crossover
      Bollinger (20) — volatility mean-reversion
      Stochastic(14) — short-term momentum
      ADX (14)       — trend strength filter
      Ichimoku       — cloud trend & S/R
      ATR (14)       — used for sizing (not voting)
    """

    @staticmethod
    def _ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _sma(s: pd.Series, n: int) -> pd.Series:
        return s.rolling(n).mean()

    def _macd(self, close, fast=12, slow=26, sig=9):
        m = self._ema(close, fast) - self._ema(close, slow)
        s = self._ema(m, sig)
        return m, s, m - s

    @staticmethod
    def _rsi(close: pd.Series, n=14) -> pd.Series:
        d = close.diff()
        g = d.clip(lower=0).ewm(span=n, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(span=n, adjust=False).mean()
        return 100 - 100 / (1 + g / l.replace(0, np.nan))

    @staticmethod
    def _atr(df: pd.DataFrame, n=14) -> pd.Series:
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        return tr.ewm(span=n, adjust=False).mean()

    def get_atr(self, df: pd.DataFrame, n=14) -> float:
        return self._atr(df, n).iloc[-1]

    @staticmethod
    def _adx(df: pd.DataFrame, n=14) -> pd.Series:
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        pdm = h.diff().clip(lower=0)
        ndm = (-l.diff()).clip(lower=0)
        pdm[pdm < (-l.diff()).clip(lower=0)] = 0
        ndm[ndm < h.diff().clip(lower=0)] = 0
        atr = tr.ewm(span=n, adjust=False).mean()
        pdi = 100 * pdm.ewm(span=n, adjust=False).mean() / atr
        ndi = 100 * ndm.ewm(span=n, adjust=False).mean() / atr
        dx  = 100 * (pdi - ndi).abs() / (pdi + ndi)
        return dx.ewm(span=n, adjust=False).mean()

    @staticmethod
    def _stoch(df: pd.DataFrame, k=14, d=3):
        lo = df["Low"].rolling(k).min()
        hi = df["High"].rolling(k).max()
        K  = 100 * (df["Close"] - lo) / (hi - lo + 1e-10)
        return K, K.rolling(d).mean()

    @staticmethod
    def _ichimoku(df: pd.DataFrame) -> dict:
        h, l = df["High"], df["Low"]
        ten = (h.rolling(9).max()  + l.rolling(9).min())  / 2
        kij = (h.rolling(26).max() + l.rolling(26).min()) / 2
        sa  = ((ten + kij) / 2).shift(26)
        sb  = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
        return {"tenkan": ten, "kijun": kij, "senkou_a": sa, "senkou_b": sb}

    def _get_candle_patterns(self, df: pd.DataFrame) -> dict:
        """Identifies key price action patterns from the last 2 candles."""
        c1, c2 = df.iloc[-2], df.iloc[-1] # c1 = previous, c2 = current
        
        o1, h1, l1, cl1 = c1["Open"], c1["High"], c1["Low"], c1["Close"]
        o2, h2, l2, cl2 = c2["Open"], c2["High"], c2["Low"], c2["Close"]
        
        body1 = abs(cl1 - o1)
        body2 = abs(cl2 - o2)
        range1 = h1 - l1
        range2 = h2 - l2
        
        res = {"bull": [], "bear": []}
        
        # 1. Engulfing
        if cl2 > o2 and cl1 < o1 and o2 < cl1 and cl2 > o1:
            res["bull"].append("ENGULFING")
        elif cl2 < o2 and cl1 > o1 and o2 > cl1 and cl2 < o1:
            res["bear"].append("ENGULFING")
            
        # 2. Pin Bar / Hammer
        lower_wick2 = min(o2, cl2) - l2
        upper_wick2 = h2 - max(o2, cl2)
        if lower_wick2 > (body2 * 2) and upper_wick2 < (body2 * 0.5):
            res["bull"].append("HAMMER")
        elif upper_wick2 > (body2 * 2) and lower_wick2 < (body2 * 0.5):
            res["bear"].append("SHOOTING_STAR")
            
        # 3. Doji (Indecision)
        if body2 < (range2 * 0.1):
            res["neutral"] = "DOJI"
            
        return res

    def get_signal(self, df: pd.DataFrame) -> Tuple[str, dict]:
        """Return (signal, details_dict). signal ∈ {'BUY','SELL','HOLD'}"""
        close = df["Close"]
        buy, sell = 0, 0
        d = {}

        # 1. EMA trend
        e20, e50, e200 = [self._ema(close, n).iloc[-1] for n in (20, 50, 200)]
        price = close.iloc[-1]
        d.update(ema20=round(e20,5), ema50=round(e50,5), ema200=round(e200,5))
        if e20 > e50 > e200:   buy  += 2; d["ema"] = "BULL"
        elif e20 < e50 < e200: sell += 2; d["ema"] = "BEAR"
        else:                              d["ema"] = "NEUTRAL"

        # 2. RSI
        rsi = self._rsi(close).iloc[-1]
        d["rsi"] = round(rsi, 2)
        if rsi < 30:   buy  += 2; d["rsi_sig"] = "OVERSOLD"
        elif rsi > 70: sell += 2; d["rsi_sig"] = "OVERBOUGHT"
        elif rsi < 45: buy  += 1; d["rsi_sig"] = "LEAN_BULL"
        elif rsi > 55: sell += 1; d["rsi_sig"] = "LEAN_BEAR"
        else:                     d["rsi_sig"] = "NEUTRAL"

        # 3. MACD
        ml, sl, hist = self._macd(close)
        d.update(macd=round(ml.iloc[-1],6), macd_hist=round(hist.iloc[-1],6))
        if ml.iloc[-1] > sl.iloc[-1] and hist.iloc[-1] > 0:  buy  += 2; d["macd_sig"] = "BUY"
        elif ml.iloc[-1] < sl.iloc[-1] and hist.iloc[-1] < 0: sell += 2; d["macd_sig"] = "SELL"
        else:                                                              d["macd_sig"] = "NEUTRAL"

        # 4. Bollinger Bands
        mid = self._sma(close, 20)
        std = close.rolling(20).std()
        bb_u, bb_l = (mid + 2*std).iloc[-1], (mid - 2*std).iloc[-1]
        d.update(bb_upper=round(bb_u,5), bb_lower=round(bb_l,5))
        if price <= bb_l:   buy  += 1; d["bb"] = "LOWER_TOUCH"
        elif price >= bb_u: sell += 1; d["bb"] = "UPPER_TOUCH"
        else:                          d["bb"] = "INSIDE"

        # 5. Stochastic
        K, D = self._stoch(df)
        kv, dv = K.iloc[-1], D.iloc[-1]
        d.update(stoch_k=round(kv,2), stoch_d=round(dv,2))
        if kv < 20 and dv < 20:   buy  += 1; d["stoch"] = "OVERSOLD"
        elif kv > 80 and dv > 80: sell += 1; d["stoch"] = "OVERBOUGHT"
        elif kv > dv:              buy  += 1; d["stoch"] = "BULL_X"
        elif kv < dv:              sell += 1; d["stoch"] = "BEAR_X"
        else:                                 d["stoch"] = "NEUTRAL"

        # 6. ADX — trend strength (HARD FILTER for noise)
        adx_series = self._adx(df)
        adx = adx_series.iloc[-1]
        adx_slope = adx - adx_series.iloc[-3]
        d["adx"] = round(adx, 2)
        
        is_noisy = False
        if adx < Config.MIN_ADX:
            is_noisy = True
            d["noise_reason"] = "LOW_ADX"
        elif adx_slope < -2: # ADX falling sharply
            is_noisy = True
            d["noise_reason"] = "FALLING_TREND"

        # 7. EMA Squeeze Detection (Sideways Noise)
        ema_gap = abs(e20 - e50) / price
        if ema_gap < Config.EMA_SQUEEZE_THRESHOLD:
            is_noisy = True
            d["noise_reason"] = "EMA_SQUEEZE"

        # 8. Ichimoku
        ichi = self._ichimoku(df)
        cloud_top = max(ichi["senkou_a"].iloc[-1], ichi["senkou_b"].iloc[-1])
        cloud_bot = min(ichi["senkou_a"].iloc[-1], ichi["senkou_b"].iloc[-1])
        if price > cloud_top:   buy  += 1; d["ichi"] = "ABOVE"
        elif price < cloud_bot: sell += 1; d["ichi"] = "BELOW"
        else:                              
            is_noisy = True; d["ichi"] = "IN_CLOUD"
            d["noise_reason"] = "IN_CLOUD"

        # 9. Candlestick Patterns
        patterns = self._get_candle_patterns(df)
        d["patterns"] = patterns
        if patterns["bull"]:
            buy += 2 * len(patterns["bull"])
        if patterns["bear"]:
            sell += 2 * len(patterns["bear"])

        d.update(buy_votes=buy, sell_votes=sell, is_noisy=is_noisy)

        if is_noisy:
            return "HOLD", d

        MIN_VOTES = 5
        if buy >= MIN_VOTES and buy > sell * 1.3:   return "BUY",  d
        elif sell >= MIN_VOTES and sell > buy * 1.3: return "SELL", d
        else:                                        return "HOLD", d

    def get_trend_direction(self, df: pd.DataFrame) -> str:
        """Determines overall trend: BULL, BEAR, or NEUTRAL."""
        if df is None or len(df) < 50:
            return "NEUTRAL"
        
        close = df["Close"]
        e50 = self._ema(close, 50).iloc[-1]
        e200 = self._ema(close, 200).iloc[-1]
        
        if e50 > e200:
            return "BULL"
        elif e50 < e200:
            return "BEAR"
        return "NEUTRAL"

    def get_swing_levels(self, df: pd.DataFrame, n=20) -> Tuple[float, float]:
        """Finds recent swing high and low for SL placement."""
        low = df["Low"].iloc[-n:].min()
        high = df["High"].iloc[-n:].max()
        return low, high


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — ML CANDLE PREDICTOR
# ══════════════════════════════════════════════════════════════
class CandlePredictor:
    """
    Gradient Boosting classifier that predicts the direction
    of the NEXT candle using 25+ engineered features.

    Features include:
      - Multi-lag returns (1,2,3,5,8,13 bars)
      - RSI value & slope
      - MACD histogram & slope
      - Bollinger Band position
      - ATR% (volatility normalised)
      - Stochastic K & D
      - EMA distances (10/20/50)
      - Candle body, wick ratios
      - Volume ratio & slope
      - Doji / engulfing pattern flags
    """

    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        self.models:  Dict[Tuple[str, int], XGBClassifier] = {}
        self.scalers: Dict[Tuple[str, int], StandardScaler] = {}
        self.trained: Dict[Tuple[str, int], bool] = {}

    # ── Feature Engineering ───────────────────────────────────
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        c, h, l, o, v = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]

        # Returns
        for lag in [1, 2, 3, 5, 8, 13]:
            f[f"ret_{lag}"] = c.pct_change(lag)

        # Candle shape
        rng = (h - l).replace(0, np.nan)
        f["body"]        = (c - o) / rng
        f["upper_wick"]  = (h - c.clip(upper=o)) / rng
        f["lower_wick"]  = (o.clip(upper=c) - l) / rng
        f["range_norm"]  = rng / c

        # RSI
        d  = c.diff()
        g  = d.clip(lower=0).ewm(span=14, adjust=False).mean()
        ls = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean()
        rsi = 100 - 100 / (1 + g / ls.replace(0, np.nan))
        f["rsi"]       = rsi
        f["rsi_slope"] = rsi.diff(3)

        # MACD
        e12 = c.ewm(span=12, adjust=False).mean()
        e26 = c.ewm(span=26, adjust=False).mean()
        macd = e12 - e26
        sig  = macd.ewm(span=9, adjust=False).mean()
        f["macd_hist"]  = macd - sig
        f["macd_slope"] = f["macd_hist"].diff(2)

        # ATR%
        tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
        f["atr_pct"] = tr.ewm(span=14, adjust=False).mean() / c

        # Stochastic
        l14 = l.rolling(14).min()
        h14 = h.rolling(14).max()
        K   = 100 * (c - l14) / (h14 - l14 + 1e-10)
        f["stoch_k"] = K
        f["stoch_d"] = K.rolling(3).mean()

        # Bollinger position
        sma = c.rolling(20).mean()
        std = c.rolling(20).std()
        f["bb_pos"] = (c - sma) / (2 * std + 1e-10)

        # EMA distances
        for span in [10, 20, 50]:
            f[f"d_ema{span}"] = (c - c.ewm(span=span, adjust=False).mean()) / c

        # Volume Refinement
        f["vol_roc"]   = v.pct_change(3)  # Rate of Change
        f["rvol"]      = v / v.rolling(20).mean()  # Relative Volume
        f["vol_slope"] = f["rvol"].diff(3)

        # Candle patterns
        f["doji"]           = (f["body"].abs() < 0.1).astype(int)
        f["bull_engulf"]    = ((o<c) & (o<c.shift(1)) & (c>o.shift(1))).astype(int)
        f["bear_engulf"]    = ((o>c) & (o>c.shift(1)) & (c<o.shift(1))).astype(int)

        return f

    def _labels(self, df: pd.DataFrame, threshold: float = None) -> pd.Series:
        # Predict the return over the next 3 candles to smooth out noise
        fut = df["Close"].shift(-3) / df["Close"] - 1
        
        if threshold is None:
            # Dynamic threshold based on ATR (adapts to Crypto vs Forex volatility)
            h, l, c = df["High"], df["Low"], df["Close"]
            tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
            atr_pct = tr.ewm(span=14, adjust=False).mean() / c
            threshold = atr_pct * 0.5  # Must move at least half an ATR to be a valid BUY/SELL

        lab = pd.Series(1, index=df.index)  # 1 is HOLD
        lab[fut >  threshold] =  2          # 2 is BUY
        lab[fut < -threshold] =  0          # 0 is SELL
        return lab

    # ── Training ──────────────────────────────────────────────
    def train(self, df: pd.DataFrame, symbol: str = "default", tf: int = 16385) -> float:
        feat   = self._features(df)
        labels = self._labels(df)

        data = pd.concat([feat, labels.rename("y")], axis=1).dropna().iloc[:-1]
        if len(data) < 100:
            log.warning("[%s|%d] Not enough rows for ML training.", symbol, tf)
            return 0.0

        X = data.drop("y", axis=1).values
        y = data["y"].values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Validation split to log accuracy
        Xtr, Xvl, ytr, yvl = train_test_split(Xs, y, test_size=0.2, shuffle=False)
        cw_val = compute_sample_weight(class_weight='balanced', y=ytr)
        val_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.05, objective='multi:softprob', random_state=42)
        val_model.fit(Xtr, ytr, sample_weight=cw_val)
        acc = accuracy_score(yvl, val_model.predict(Xvl))

        # Train final deployed model on ALL data (including newest)
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            objective='multi:softprob', random_state=42
        )
        cw = compute_sample_weight(class_weight='balanced', y=y)
        weights = cw * np.linspace(1.0, 5.0, len(y))
        model.fit(Xs, y, sample_weight=weights)
        
        self.models[(symbol, tf)] = model
        self.scalers[(symbol, tf)] = scaler
        self.trained[(symbol, tf)] = True
        
        # Save model
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        joblib.dump(model,  f"{Config.MODEL_DIR}/{symbol}_{tf}_model.pkl")
        joblib.dump(scaler, f"{Config.MODEL_DIR}/{symbol}_{tf}_scaler.pkl")
        
        R_MAP = {v: k for k, v in Config.TF_MAP.items()}
        tf_str = R_MAP.get(tf, str(tf))
        log.info("[%s] ML trained (%s) | val accuracy: %.1f%%", symbol, tf_str, acc * 100)
        return acc

    def load(self, symbol: str, tf: int) -> bool:
        try:
            m_path = f"{Config.MODEL_DIR}/{symbol}_{tf}_model.pkl"
            s_path = f"{Config.MODEL_DIR}/{symbol}_{tf}_scaler.pkl"
            if os.path.exists(m_path):
                self.models[(symbol, tf)]  = joblib.load(m_path)
                self.scalers[(symbol, tf)] = joblib.load(s_path)
                self.trained[(symbol, tf)] = True
                return True
            return False
        except: return False

    def predict(self, df: pd.DataFrame, symbol: str, tf: int) -> Tuple[str, float]:
        if (symbol, tf) not in self.trained:
            if not self.load(symbol, tf):
                return "HOLD", 0.0

        feat   = self._features(df)
        latest = feat.iloc[[-1]]
        if latest.isnull().values.any():
            return "HOLD", 0.0

        try:
            X    = self.scalers[(symbol, tf)].transform(latest.values)
            prob = self.models[(symbol, tf)].predict_proba(X)[0]
            cls  = self.models[(symbol, tf)].classes_
            best = int(np.argmax(prob))
            sig  = {0: "SELL", 1: "HOLD", 2: "BUY"}.get(cls[best], "HOLD")
            return sig, float(prob[best])
        except Exception as e:
            log.error("ML predict error [%s|%d]: %s", symbol, tf, e)
            return "HOLD", 0.0


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — RISK MANAGER
# ══════════════════════════════════════════════════════════════
class RiskManager:
    """
    Position sizing + guard rails:
      - 1% risk model (lot = risk$ / (SL_pips × pip_value))
      - Max drawdown kill-switch
      - Daily trade count cap
      - Daily loss cap
    """

    def __init__(self):
        self._daily_trades = 0
        self._daily_pnl    = 0.0
        self._last_reset   = date.today()

    def _daily_reset(self):
        if date.today() != self._last_reset:
            log.info("Daily counters reset.")
            self._daily_trades = 0
            self._daily_pnl    = 0.0
            self._last_reset   = date.today()

    def can_trade(self, balance: float) -> bool:
        self._daily_reset()
        if self._daily_trades >= Config.MAX_DAILY_TRADES:
            log.warning("Daily trade cap (%d) reached.", Config.MAX_DAILY_TRADES)
            return False
        loss_limit = balance * Config.MAX_DAILY_LOSS_PCT / 100
        if self._daily_pnl <= -loss_limit:
            log.warning("Daily loss cap hit (%.2f).", self._daily_pnl)
            return False
        return True

    def record(self, pnl: float = 0.0):
        self._daily_reset()
        self._daily_trades += 1
        self._daily_pnl    += pnl

    def lot_size(self, balance: float, sl_pips: float,
                 pip_val: float, min_lot: float, max_lot: float,
                 confidence: float = 0.0) -> float:
        if sl_pips <= 0 or pip_val <= 0:
            return round(max(min_lot, min(Config.DEFAULT_LOT, max_lot, Config.MAX_LOT)), 2)
        confidence = max(0.0, min(1.0, confidence))
        if confidence < 0.70:
            scaled_lot = 0.01
        elif confidence < 0.80:
            scaled_lot = 0.02
        elif confidence < 0.90:
            scaled_lot = 0.03
        elif confidence < 0.95:
            scaled_lot = 0.04
        else:
            scaled_lot = Config.MAX_LOT

        lot = min(scaled_lot, max_lot, Config.MAX_LOT)
        return round(max(min_lot, lot), 2)

    @staticmethod
    def drawdown_ok(equity_now: float, equity_start: float) -> bool:
        if equity_start <= 0:
            return True
        dd = (equity_start - equity_now) / equity_start * 100
        if dd >= Config.MAX_DRAWDOWN_PCT:
            log.error("Max drawdown exceeded: %.2f%%", dd)
            return False
        return True


# ══════════════════════════════════════════════════════════════
#  SECTION 4.5 — NEWS FILTER
# ══════════════════════════════════════════════════════════════
class NewsFilter:
    """
    Checks for high-impact economic news from Forex Factory (unofficial JSON).
    Prevents trading within a buffer window around news events.
    """
    def __init__(self):
        self.news_events = []
        self.last_update = datetime.min.replace(tzinfo=timezone.utc)

    def update_news(self):
        """Fetches the latest news if more than 6 hours since last update."""
        now = datetime.now(timezone.utc)
        if (now - self.last_update).total_seconds() < 21600: # 6 hours
            return

        try:
            log.info("Updating economic calendar...")
            res = requests.get(Config.NEWS_URL, timeout=10)
            if res.status_code == 200:
                self.news_events = res.json()
                self.last_update = now
                log.info("News calendar updated. %d events found.", len(self.news_events))
        except Exception as e:
            log.error("Failed to update news: %s", e)

    def is_news_time(self, symbol: str) -> bool:
        """Returns True if high-impact news is nearby for relevant currencies."""
        if not Config.USE_NEWS_FILTER:
            return False

        self.update_news()
        now = datetime.now(timezone.utc)
        buffer = timedelta(minutes=Config.NEWS_BUFFER_MINS)

        # Identify relevant currencies for this symbol (e.g. EURUSD -> EUR, USD)
        currencies = [symbol[:3], symbol[3:6]]
        
        for event in self.news_events:
            if event.get('impact') != 'High':
                continue
            
            event_curr = event.get('country') # In FF JSON, 'country' is often the currency code
            if event_curr not in currencies:
                continue

            try:
                # FF JSON date format example: "2024-05-23T12:30:00-04:00"
                # But some formats differ. We'll try common ones.
                date_str = event.get('date')
                if not date_str: continue
                
                # Replace Z with +00:00 for fromisoformat
                if date_str.endswith('Z'): date_str = date_str[:-1] + '+00:00'
                event_time = datetime.fromisoformat(date_str)
                
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)

                if (event_time - buffer) <= now <= (event_time + buffer):
                    log.warning("Trading blocked by News: %s (%s) at %s", 
                                event.get('title'), event_curr, date_str)
                    return True
            except Exception as e:
                # log.debug("News date parse error: %s", e)
                continue

        return False

# ══════════════════════════════════════════════════════════════
#  SECTION 5 — MAIN TRADING BOT
# ══════════════════════════════════════════════════════════════
class ForexBot:
    """
    Orchestrates everything:
      1. Connects to MT5
      2. Trains ML models
      3. Every LOOP_INTERVAL seconds:
         a. Fetch candles
         b. Compute TA signal (vote-based)
         c. Get ML prediction
         d. Combine → BUY / SELL / HOLD
         e. Place order if conditions met
         f. Monitor drawdown
    """

    def __init__(self):
        self.ta        = TechnicalAnalysis()
        self.ml        = CandlePredictor(Config.ML_LOOKBACK)
        self.risk      = RiskManager()
        self.news      = NewsFilter()
        self.running   = False
        self.eq_start  = 0.0
        self.loop_count = 0  # For online learning feedback loop
        self.symbol_map: Dict[str, str] = {}
        
        # Initialize log file with header if not exists
        if not os.path.exists(Config.LOG_FILE_TRADES):
            with open(Config.LOG_FILE_TRADES, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Symbol", "Action", "Confidence", "TA_Signal", "ML_Signal", "ATR", "Actual_Close"])

    # ── MT5 Connection ────────────────────────────────────────
    def _connect(self) -> bool:
        if not mt5.initialize():
            log.error("MT5 initialize() failed: %s", mt5.last_error())
            return False
        ok = mt5.login(Config.LOGIN, password=Config.PASSWORD, server=Config.SERVER)
        if not ok:
            log.error("MT5 login failed: %s", mt5.last_error())
            mt5.shutdown(); return False
        info = mt5.account_info()
        self.eq_start = info.equity
        self._refresh_symbol_map()
        log.info("Connected | Account #%s | Balance: %.2f | Equity: %.2f",
                 info.login, info.balance, info.equity)
        return True

    def _refresh_symbol_map(self):
        names = [s.name for s in (mt5.symbols_get() or [])]
        self.symbol_map = {}
        for base in Config.SYMBOLS:
            if base in names:
                self.symbol_map[base] = base
                continue
            matches = [n for n in names if n.startswith(base)]
            if not matches:
                matches = [n for n in names if base in n]
            if matches:
                self.symbol_map[base] = matches[0]
                log.info("Mapped symbol %s -> %s", base, matches[0])
            else:
                self.symbol_map[base] = base
                log.warning("No broker symbol match for %s; using as-is.", base)

    def _broker_symbol(self, sym: str) -> str:
        return self.symbol_map.get(sym, sym)

    # ── Data ──────────────────────────────────────────────────
    def _candles(self, sym: str, tf: int, n: int = 500) -> Optional[pd.DataFrame]:
        broker_sym = self._broker_symbol(sym)
        mt5.symbol_select(broker_sym, True)
        rates = mt5.copy_rates_from_pos(broker_sym, tf, 0, n)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"open":"Open","high":"High","low":"Low",
                            "close":"Close","tick_volume":"Volume"}, inplace=True)
        return df[["Open","High","Low","Close","Volume"]]

    def _price(self, sym: str) -> Tuple[float, float]:
        broker_sym = self._broker_symbol(sym)
        mt5.symbol_select(broker_sym, True)
        t = mt5.symbol_info_tick(broker_sym)
        if t is None:
            raise RuntimeError(f"No tick data for {sym}")
        return t.bid, t.ask

    def _combine_signal_from_df(self, df: pd.DataFrame, sym: str, tf: int, trend: str = "NEUTRAL") -> Tuple[str, str, str, float, dict]:
        ta_sig, ta_det = self.ta.get_signal(df)
        ml_sig, ml_con = self.ml.predict(df, sym, tf)

        # Both agree + ML confident -> strong signal
        if ta_sig == ml_sig and ml_con >= Config.ML_CONFIDENCE_THRESHOLD:
            sig = ta_sig
        elif ml_con >= 0.80 and ml_sig != "HOLD":
            sig = ml_sig
        elif ta_sig != "HOLD" and ml_sig == "HOLD":
            # Allow TA signal to go through if ML is just predicting HOLD
            # (ML only predicts the single next candle, which is often HOLD)
            sig = ta_sig
        else:
            sig = "HOLD"

        # Apply noise filter from TA
        if ta_det.get("is_noisy", False):
            # log.info("[%s] Signal blocked by Market Noise filter (%s).", sym, ta_det.get("noise_reason"))
            sig = "HOLD"

        # Apply trend filter
        if Config.USE_TREND_FILTER and trend != "NEUTRAL":
            if sig == "BUY" and trend != "BULL":
                log.info("[%s|%d] BUY signal blocked by %s trend filter.", sym, tf, trend)
                sig = "HOLD"
            elif sig == "SELL" and trend != "BEAR":
                log.info("[%s|%d] SELL signal blocked by %s trend filter.", sym, tf, trend)
                sig = "HOLD"

        return sig, ta_sig, ml_sig, ml_con, ta_det

    def _signal(self, sym: str, tf: int) -> Tuple[str, float, str, str, float]:
        # News Filter Check
        if self.news.is_news_time(sym):
            return "HOLD", 0.0, "HOLD", "HOLD", 0.0

        # Entry timeframe data
        df = self._candles(sym, tf, Config.DATA_HISTORY)
        if df is None or len(df) < 100:
            return "HOLD", 0.0, "HOLD", "HOLD", 0.0

        # Trend timeframe data
        trend = "NEUTRAL"
        if Config.USE_TREND_FILTER:
            df_trend = self._candles(sym, Config.TREND_TIMEFRAME, Config.DATA_HISTORY)
            trend = self.ta.get_trend_direction(df_trend)

        sig, ta_sig, ml_sig, ml_con, ta_det = self._combine_signal_from_df(df, sym, tf, trend)
        atr = self.ta.get_atr(df)

        if sig != "HOLD":
            log.info("[%s|%d] Trend=%s | TA=%s | ML=%s(%.0f%%) | ATR=%.5f | FINAL=%s",
                    sym, tf, trend, ta_sig, ml_sig, ml_con * 100, atr, sig)

        return sig, ml_con, ta_sig, ml_sig, atr

    # ── Order Execution ───────────────────────────────────────
    def _lot(self, sym: str, sl_points: int, confidence: float = 0.0) -> float:
        broker_sym = self._broker_symbol(sym)
        acc = mt5.account_info()
        si = mt5.symbol_info(broker_sym)
        if acc is None or si is None or sl_points <= 0:
            return Config.DEFAULT_LOT

        # Risk amount in account currency
        risk_amount = acc.balance * (Config.RISK_PCT / 100)
        
        # Lot = Risk / ((SL_Points * (Point/TickSize)) * TickValue)
        # si.point / si.trade_tick_size is usually 1.0 but handles non-standard brokers
        points_per_tick = si.point / si.trade_tick_size
        lot = risk_amount / (sl_points * points_per_tick * si.trade_tick_value)

        # Margin-aware clamp
        step = si.volume_step if si.volume_step and si.volume_step > 0 else 0.01
        decimals = max(0, len(str(step).split(".")[-1].rstrip("0")))
        lot = round(max(si.volume_min, min(lot, si.volume_max, Config.MAX_LOT)), decimals)
        
        _, ask = self._price(sym)
        free_margin = getattr(acc, "margin_free", 0.0)
        while lot >= si.volume_min:
            margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, broker_sym, lot, ask)
            if margin is None: break
            if free_margin > 0 and margin <= free_margin * 0.9:
                return lot
            lot = round(lot - step, decimals)

        return si.volume_min

    def _build_order_request(self, sym: str, tf: int, side: str, lot: float, atr: float = 0.0) -> Optional[dict]:
        broker_sym = self._broker_symbol(sym)
        si = mt5.symbol_info(broker_sym)
        if si is None: return None
        bid, ask = self._price(sym)
        pt = si.point

        # Expert Logic: Structure-based SL + ATR-based TP
        df = self._candles(sym, tf, 30)
        swing_low, swing_high = self.ta.get_swing_levels(df) if df is not None else (0, 0)

        if side == "BUY":
            price = ask
            # SL is the lower of (ATR-based SL) or (Swing Low - 5 pips)
            atr_sl = bid - (atr * Config.ATR_SL_MULT)
            struct_sl = swing_low - (5 * pt)
            sl = max(atr_sl, struct_sl) if swing_low > 0 else atr_sl
            
            tp = ask + (atr * Config.ATR_TP_MULT)
            ot = mt5.ORDER_TYPE_BUY
        else:
            price = bid
            # SL is the higher of (ATR-based SL) or (Swing High + 5 pips)
            atr_sl = ask + (atr * Config.ATR_SL_MULT)
            struct_sl = swing_high + (5 * pt)
            sl = min(atr_sl, struct_sl) if swing_high > 0 else atr_sl
            
            tp = bid - (atr * Config.ATR_TP_MULT)
            ot = mt5.ORDER_TYPE_SELL

        # Ensure SL/TP are not too close (broker limits)
        min_stop_dist = (si.trade_stops_level + 5) * pt
        if abs(price - sl) < min_stop_dist:
            sl = price - min_stop_dist if side == "BUY" else price + min_stop_dist
        if abs(price - tp) < min_stop_dist:
            tp = price + min_stop_dist if side == "BUY" else price - min_stop_dist

        # TF-based Magic Number
        magic = Config.MAGIC_NUMBER + (tf % 100)

        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_sym,
            "volume": lot,
            "type": ot,
            "price": price,
            "sl": round(sl, si.digits),
            "tp": round(tp, si.digits),
            "deviation": 10,
            "magic": magic,
            "comment": f"ExpertV1_{tf}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

    def _trailing_stop(self):
        """Expert-level Trailing Stop: Combines Break-even, ATR, and Market Structure."""
        if not Config.TRAILING_SL: return
        
        try:
            positions = mt5.positions_get()
            if not positions: return

            for p in positions:
                # Filter by our magic number base range
                if not (Config.MAGIC_NUMBER <= p.magic <= Config.MAGIC_NUMBER + 100):
                    continue
                    
                try:
                    sym = p.symbol
                    # Decode TF from magic
                    tf_offset = p.magic - Config.MAGIC_NUMBER
                    tf = 16385 # fallback
                    for k, v in Config.TF_MAP.items():
                        if v % 100 == tf_offset:
                            tf = v; break

                    si = mt5.symbol_info(sym)
                    if not si: continue
                    
                    df = self._candles(sym, tf, 30)
                    if df is None or len(df) < 14: continue
                    atr = self.ta.get_atr(df)
                    swing_low, swing_high = self.ta.get_swing_levels(df)
                    
                    bid, ask = self._price(sym)
                    new_sl = 0.0

                    if p.type == mt5.POSITION_TYPE_BUY:
                        # 1. Break-Even
                        if p.sl < p.price_open and bid - p.price_open > atr * Config.BREAK_EVEN_ATR_MULT:
                            new_sl = p.price_open + (2 * si.point)
                        
                        # 2. Structure-based Trailing (Pro)
                        elif swing_low > p.sl + (5 * si.point) and bid > swing_low + atr:
                            new_sl = swing_low - (2 * si.point)
                            
                    elif p.type == mt5.POSITION_TYPE_SELL:
                        # 1. Break-Even
                        if p.sl > p.price_open and p.price_open - ask > atr * Config.BREAK_EVEN_ATR_MULT:
                            new_sl = p.price_open - (2 * si.point)
                        
                        # 2. Structure-based Trailing (Pro)
                        elif swing_high < p.sl - (5 * si.point) and ask < swing_high - atr:
                            new_sl = swing_high + (2 * si.point)

                    if new_sl > 0:
                        # Only update if it actually improves protection
                        is_better = (p.type == mt5.POSITION_TYPE_BUY and new_sl > p.sl) or \
                                    (p.type == mt5.POSITION_TYPE_SELL and new_sl < p.sl)
                        if is_better:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": sym,
                                "position": p.ticket,
                                "sl": round(new_sl, si.digits),
                                "tp": p.tp,
                                "magic": p.magic
                            }
                            mt5.order_send(request)
                except Exception as e:
                    log.error("Expert trailing error [%s]: %s", p.symbol, e)
        except Exception as e:
            log.error("Critical error in expert trailing loop: %s", e)

    def _log_trade(self, sym: str, side: str, conf: float, ta_sig: str, ml_sig: str, atr: float):
        """Logs trade attempt details to CSV."""
        with open(Config.LOG_FILE_TRADES, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                sym, side, round(conf, 2), ta_sig, ml_sig, round(atr, 6), ""
            ])

    @staticmethod
    def _order_check_passed(retcode: int) -> bool:
        # Some MT5 builds return 0 for successful order_check.
        return retcode in (0, mt5.TRADE_RETCODE_DONE)

    def validate_order(self, sym: str, side: str) -> bool:
        if not self._connect():
            return False
        try:
            df = self._candles(sym, Config.TIMEFRAME, 20)
            atr = self.ta.get_atr(df) if df is not None else 0.0
            
            # Estimate SL points for lot calc
            si = mt5.symbol_info(self._broker_symbol(sym))
            sl_points = int((atr * Config.ATR_SL_MULT) / si.point) if atr > 0 else 300
            
            lot = self._lot(sym, sl_points)
            req = self._build_order_request(sym, side, lot, atr)
            if req is None:
                log.error("[%s] Could not build order request.", sym)
                return False
            check = mt5.order_check(req)
            if check is None:
                log.error("[%s] order_check failed: %s", sym, mt5.last_error())
                return False
            ok = self._order_check_passed(check.retcode)
            log.info("[%s] order_check retcode=%s | comment=%s", sym, check.retcode, getattr(check, "comment", ""))
            return ok
        finally:
            mt5.shutdown()

    def _order(self, sym: str, tf: int, side: str, lot: float, atr: float = 0.0) -> bool:
        req = self._build_order_request(sym, tf, side, lot, atr)
        if req is None:
            return False

        check = mt5.order_check(req)
        if check is None:
            log.error("Order check failed [%s]: %s", sym, mt5.last_error())
            return False
        if not self._order_check_passed(check.retcode):
            log.error("Order check rejected [%s]: %s (code %d)", sym, check.comment, check.retcode)
            return False

        res = mt5.order_send(req)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("Order failed [%s]: %s (code %d)", sym, res.comment, res.retcode)
            return False
        self.risk.record()
        log.info("ORDER OK %s %s | tf=%d | lot=%.2f | price=%.5f | SL=%.5f | TP=%.5f",
                 side, sym, tf, lot, req["price"], req["sl"], req["tp"])
        return True

    def _close_position(self, p):
        """Closes an open position."""
        tick = mt5.symbol_info_tick(p.symbol)
        if tick is None: return False
        
        type_dict = {
            mt5.POSITION_TYPE_BUY: mt5.ORDER_TYPE_SELL,
            mt5.POSITION_TYPE_SELL: mt5.ORDER_TYPE_BUY
        }
        price_dict = {
            mt5.POSITION_TYPE_BUY: tick.bid,
            mt5.POSITION_TYPE_SELL: tick.ask
        }
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": p.volume,
            "type": type_dict[p.type],
            "position": p.ticket,
            "price": price_dict[p.type],
            "deviation": 10,
            "magic": p.magic,
            "comment": "Expert Exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("[%s] Close failed: %s", p.symbol, res.comment)
            return False
        log.info("[%s] Position closed successfully.", p.symbol)
        return True

    def _has_position(self, sym: str, tf: int) -> bool:
        broker_sym = self._broker_symbol(sym)
        magic = Config.MAGIC_NUMBER + (tf % 100)
        pos = mt5.positions_get(symbol=broker_sym, magic=magic)
        return pos is not None and len(pos) > 0

    def backtest(self, history: int = 2500, test_ratio: float = 0.30) -> Dict[str, dict]:
        """Run a lightweight walk-forward backtest on recent MT5 candles."""
        results: Dict[str, dict] = {}
        if not self._connect():
            return results

        try:
            for sym in Config.SYMBOLS:
                df = self._candles(sym, Config.TIMEFRAME, history)
                if df is None or len(df) < 220:
                    log.warning("[%s] Not enough candles for backtest.", sym)
                    continue

                split_idx = int(len(df) * (1 - test_ratio))
                train_df = df.iloc[:split_idx].copy()
                test_df = df.iloc[split_idx:].copy()
                if len(train_df) < 120 or len(test_df) < 60:
                    log.warning("[%s] Backtest split too small.", sym)
                    continue

                self.ml.train(train_df, sym)

                y_true = []
                y_pred = []
                action_true = []
                action_pred = []
                trade_count = 0

                # Fetch trend data for entire backtest if needed
                trend_map = {}
                if Config.USE_TREND_FILTER:
                    df_trend = self._candles(sym, Config.TREND_TIMEFRAME, history)
                    if df_trend is not None:
                        for idx, row in df_trend.iterrows():
                            # Simple trend at each point
                            # Note: This is a bit naive but works for a lightweight backtest
                            pass # We'll do it point-in-time below for accuracy

                start_i = min(120, max(30, len(test_df) // 3))
                for i in range(start_i, len(test_df) - 1):
                    window = pd.concat([train_df, test_df.iloc[: i + 1]])
                    
                    trend = "NEUTRAL"
                    if Config.USE_TREND_FILTER:
                        # Get trend at this point in time from HTF
                        current_time = test_df.index[i]
                        # Fetch recent HTF candles up to this time
                        df_trend = self._candles(sym, Config.TREND_TIMEFRAME, 250) # Fetch some history
                        # In a real backtest we'd use a faster method, but this ensures correctness with current MT5 connection
                        trend = self.ta.get_trend_direction(df_trend)

                    sig, _, _, _, _ = self._combine_signal_from_df(window, sym, trend)

                    nxt_ret = (test_df["Close"].iloc[i + 1] / test_df["Close"].iloc[i]) - 1
                    true_dir = 1 if nxt_ret > 0 else (-1 if nxt_ret < 0 else 0)
                    pred_dir = {"BUY": 1, "SELL": -1, "HOLD": 0}[sig]

                    y_true.append(true_dir)
                    y_pred.append(pred_dir)

                    if pred_dir != 0:
                        trade_count += 1
                        action_true.append(true_dir)
                        action_pred.append(pred_dir)

                overall_acc = accuracy_score(y_true, y_pred) if y_true else 0.0
                action_acc = accuracy_score(action_true, action_pred) if action_true else 0.0

                results[sym] = {
                    "bars": len(df),
                    "train": len(train_df),
                    "test": len(test_df),
                    "signals": len(y_pred),
                    "trades": trade_count,
                    "overall_accuracy": round(overall_acc, 4),
                    "trade_direction_accuracy": round(action_acc, 4),
                }

                log.info(
                    "[BT %s] overall_acc=%.2f%% | trade_acc=%.2f%% | trades=%d/%d",
                    sym,
                    overall_acc * 100,
                    action_acc * 100,
                    trade_count,
                    len(y_pred),
                )

            return results
        finally:
            mt5.shutdown()

    # ── Main Loop ─────────────────────────────────────────────
    def run(self):
        # Create reverse map for logging
        R_MAP = {v: k for k, v in Config.TF_MAP.items()}
        tf_str = R_MAP.get(Config.TIMEFRAME, str(Config.TIMEFRAME))
        trend_tf_str = R_MAP.get(Config.TREND_TIMEFRAME, str(Config.TREND_TIMEFRAME))

        log.info("=" * 60)
        log.info("  FOREX BOT STARTING")
        log.info("  Symbols      : %s", Config.SYMBOLS)
        log.info("  Entry TF     : %s", tf_str)
        if Config.USE_TREND_FILTER:
            log.info("  Trend Filter : %s (ENABLED)", trend_tf_str)
        else:
            log.info("  Trend Filter : DISABLED")
        log.info("  Risk/trade   : %.1f%%  |  Max DD: %.1f%%", Config.RISK_PCT, Config.MAX_DRAWDOWN_PCT)
        log.info("=" * 60)

        if not self._connect():
            return

        # ── Initial ML training ───────────────────────────────
        log.info("Training ML models (this may take a moment)...")
        for sym in Config.SYMBOLS:
            df = self._candles(sym, Config.TIMEFRAME, Config.TRAIN_HISTORY)
            if df is not None and len(df) >= 100:
                self.ml.train(df, sym)
            else:
                log.warning("[%s] Insufficient history for training.", sym)
        log.info("ML models ready.")

        self.running = True
        try:
            while self.running:
                try:
                    acc = mt5.account_info()
                    if acc is None:
                        log.error("Lost MT5 connection. Retrying in 10s...")
                        time.sleep(10)
                        if not self._connect(): continue
                        acc = mt5.account_info()

                    # Drawdown guard
                    if not RiskManager.drawdown_ok(acc.equity, self.eq_start):
                        log.error("Bot stopping due to drawdown limit.")
                        self.running = False
                        break
                    
                    # Update Trailing SL for existing positions
                    self._trailing_stop()

                    # Process each symbol and timeframe
                    for sym in Config.SYMBOLS:
                        for tf in Config.TIMEFRAMES:
                            try:
                                if not self.risk.can_trade(acc.balance):
                                    break
                                
                                sig, confidence, ta_sig, ml_sig, atr = self._signal(sym, tf)
                                
                                # EXPERT LOGIC: Signal Reversal Exit
                                magic = Config.MAGIC_NUMBER + (tf % 100)
                                pos = mt5.positions_get(symbol=self._broker_symbol(sym), magic=magic)
                                if pos:
                                    p = pos[0]
                                    if (p.type == mt5.POSITION_TYPE_BUY and sig == "SELL") or \
                                       (p.type == mt5.POSITION_TYPE_SELL and sig == "BUY"):
                                        log.warning("[%s|%d] Signal reversal detected. Closing position.", sym, tf)
                                        self._close_position(p)
                                    continue 

                                if self._has_position(sym, tf):
                                    continue
                                if sig in ("BUY", "SELL"):
                                    si = mt5.symbol_info(self._broker_symbol(sym))
                                    sl_points = int((atr * Config.ATR_SL_MULT) / si.point) if atr > 0 else 300
                                    lot = self._lot(sym, sl_points, confidence)
                                    if self._order(sym, tf, sig, lot, atr):
                                        self._log_trade(sym, sig, confidence, ta_sig, ml_sig, atr)

                            except Exception as e:
                                log.error("[%s|%d] Error in scan loop: %s", sym, tf, e)

                except Exception as e:
                    log.error("Main loop error: %s. Continuing...", e)
                    time.sleep(5)

                # ONLINE LEARNING: Continuous Feedback Loop
                self.loop_count += 1
                if self.loop_count >= 240:  # Every ~4 hours
                    log.info("Continuous Feedback Loop: Retraining models for all TFs...")
                    for sym in Config.SYMBOLS:
                        for tf in Config.TIMEFRAMES:
                            df_retrain = self._candles(sym, tf, Config.TRAIN_HISTORY)
                            if df_retrain is not None and len(df_retrain) >= 100:
                                self.ml.train(df_retrain, sym, tf)
                    self.loop_count = 0

                time.sleep(Config.LOOP_INTERVAL)

        except KeyboardInterrupt:
            log.info("Bot stopped by user (Ctrl+C).")
        finally:
            mt5.shutdown()
            log.info("MT5 disconnected. Goodbye.")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forex bot with run, backtest and order validation modes.")
    parser.add_argument("--mode", choices=["run", "backtest", "check-order"], default="run")
    parser.add_argument("--tf", default="H1", help="Primary entry timeframe (M1, M5, M15, M30, H1, H4, D1)")
    parser.add_argument("--trend-tf", default="H4", help="Trend filter timeframe (used only if --use-filter is passed)")
    parser.add_argument("--use-filter", action="store_true", help="Enable higher timeframe trend filter")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol for check-order mode")
    parser.add_argument("--side", choices=["BUY", "SELL"], default="BUY", help="Side for check-order mode")
    parser.add_argument("--history", type=int, default=2500, help="Backtest candles")
    args = parser.parse_args()

    # Apply command-line overrides
    if args.tf in Config.TF_MAP:
        Config.TIMEFRAME = Config.TF_MAP[args.tf]
        # Adjust threshold based on timeframe
        if args.tf == "M15":
            Config.LABEL_THRESHOLD = 0.0003
        elif args.tf in ("H1", "H4"):
            Config.LABEL_THRESHOLD = 0.0010
        elif args.tf == "D1":
            Config.LABEL_THRESHOLD = 0.0050
            
    if args.trend_tf in Config.TF_MAP:
        Config.TREND_TIMEFRAME = Config.TF_MAP[args.trend_tf]
    if args.use_filter:
        Config.USE_TREND_FILTER = True

    # Validate credentials
    if Config.LOGIN == 0 or Config.PASSWORD in ("", "CHANGE_ME"):
        print("\nCredentials not set!")
        print("Set environment variables before running:")
        print("   set MT5_LOGIN=YOUR_LOGIN_NUMBER")
        print("   set MT5_PASSWORD=YOUR_PASSWORD")
        print("   set MT5_SERVER=YOUR_SERVER\n")
    else:
        bot = ForexBot()
        if args.mode == "run":
            bot.run()
        elif args.mode == "backtest":
            summary = bot.backtest(history=args.history)
            if not summary:
                print("Backtest did not produce results.")
            else:
                print("\nBacktest summary")
                for sym, stats in summary.items():
                    print(
                        f"{sym}: overall={stats['overall_accuracy']*100:.2f}% | "
                        f"trade_acc={stats['trade_direction_accuracy']*100:.2f}% | "
                        f"trades={stats['trades']}/{stats['signals']}"
                    )
        else:
            ok = bot.validate_order(args.symbol, args.side)
            print(f"order_check {'PASS' if ok else 'FAIL'} for {args.side} {args.symbol}")