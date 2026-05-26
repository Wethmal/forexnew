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
    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"]

    # ── Timeframes ────────────────────────────────────────────
    # List of (Entry Timeframe, Trend Timeframe)
    TIMEFRAMES = [
        (mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1),
        (mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30),
        (mt5.TIMEFRAME_M5,  mt5.TIMEFRAME_M15)
    ]
    USE_TREND_FILTER = True             # Enabled for safety

    # ── Risk Management ───────────────────────────────────────
    RISK_PCT           = 1.0    # % of balance per trade
    MAX_DRAWDOWN_PCT   = 10.0   # stop bot if total drawdown hits this
    MAX_DAILY_TRADES   = 10     # max orders per day
    MAX_DAILY_LOSS_PCT = 3.0    # daily loss cap (% of balance)

    # ── Trade Execution (Dynamic SL/TP) ───────────────────────
    ATR_SL_MULT  = 1.8   # Base Stop Loss = 1.8 * ATR
    ATR_TP_MULT  = 3.0   # Base Take Profit = 3.0 * ATR
    TRAILING_SL  = True  # Enable Trailing Stop Loss
    REVERSE_SIGNALS = False # Turned off, we use Break-Even strategy instead
    
    DEFAULT_LOT  = 0.01   # fallback lot size
    MAX_LOT      = 0.05   # hard cap per trade

    # ── News Filter ───────────────────────────────────────────
    USE_NEWS_FILTER  = True
    NEWS_URL         = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    NEWS_BUFFER_MINS = 15  # Stop trading 30m before/after high impact news

    # ── ML Settings ───────────────────────────────────────────
    ML_LOOKBACK             = 20    # feature window (bars)
    # Raising this generally improves win-rate by skipping weak ML predictions.
    ML_CONFIDENCE_THRESHOLD = 0.60
    ML_RETRAIN_CANDLES      = 1000  # retrain when this many new candles seen

    # Final trade gating (reduces overtrading / low-quality entries)
    MIN_FINAL_CONFIDENCE_TO_TRADE = 0.58
    TRADE_COOLDOWN_MINS           = 10

    # Spread filter (skip trades when spread is too large vs ATR)
    MAX_SPREAD_ATR_RATIO_FX     = 0.15
    MAX_SPREAD_ATR_RATIO_CRYPTO = 0.35

    # Logging
    LOG_FILE_TRADES   = "trade_signals.csv"  # clean, consistent schema going forward
    LOG_FILE_RESULTS  = "trade_results.csv"  # closed-trade outcomes (synced from MT5 history)

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

        # 6. ADX — reduces votes if trend is weak
        adx = self._adx(df).iloc[-1]
        d["adx"] = round(adx, 2)
        if adx < 20:
            buy  = max(0, buy  - 2)
            sell = max(0, sell - 2)
            d["adx_sig"] = "WEAK"
        else:
            d["adx_sig"] = "STRONG"

        # 7. Ichimoku
        ichi = self._ichimoku(df)
        cloud_top = max(ichi["senkou_a"].iloc[-1], ichi["senkou_b"].iloc[-1])
        cloud_bot = min(ichi["senkou_a"].iloc[-1], ichi["senkou_b"].iloc[-1])
        if price > cloud_top:   buy  += 1; d["ichi"] = "ABOVE"
        elif price < cloud_bot: sell += 1; d["ichi"] = "BELOW"
        else:                              d["ichi"] = "IN_CLOUD"

        d.update(buy_votes=buy, sell_votes=sell)

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
        self.models:  Dict[str, XGBClassifier] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.trained: Dict[str, bool] = {}

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
    def train(self, df: pd.DataFrame, symbol: str = "default", tf: int = None) -> float:
        feat   = self._features(df)
        labels = self._labels(df)

        data = pd.concat([feat, labels.rename("y")], axis=1).dropna().iloc[:-1]
        if len(data) < 100:
            log.warning("[%s] Not enough rows for ML training.", symbol)
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
        # ONLINE LEARNING FEEDBACK: Add higher weight to recent data (rewards/penalties)
        cw = compute_sample_weight(class_weight='balanced', y=y)
        weights = cw * np.linspace(1.0, 5.0, len(y))
        model.fit(Xs, y, sample_weight=weights)
        
        # Get TF string for filename
        R_MAP = {v: k for k, v in Config.TF_MAP.items()}
        tf_str = R_MAP.get(tf, "UNK")
        model_key = f"{symbol}_{tf_str}"
        
        log.info("[%s] ML trained (%s) | val accuracy: %.1f%%", symbol, tf_str, acc * 100)

        self.models[model_key]  = model
        self.scalers[model_key] = scaler
        self.trained[model_key] = True

        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        joblib.dump(model,  f"{Config.MODEL_DIR}/{model_key}_model.pkl")
        joblib.dump(scaler, f"{Config.MODEL_DIR}/{model_key}_scaler.pkl")
        return acc

    def load(self, symbol: str, tf: int = None) -> bool:
        try:
            R_MAP = {v: k for k, v in Config.TF_MAP.items()}
            tf_str = R_MAP.get(tf, "UNK")
            model_key = f"{symbol}_{tf_str}"
            self.models[model_key]  = joblib.load(f"{Config.MODEL_DIR}/{model_key}_model.pkl")
            self.scalers[model_key] = joblib.load(f"{Config.MODEL_DIR}/{model_key}_scaler.pkl")
            self.trained[model_key] = True
            log.info("[%s] Loaded saved ML model (%s).", symbol, tf_str)
            return True
        except FileNotFoundError:
            return False

    # ── Prediction ────────────────────────────────────────────
    def predict(self, df: pd.DataFrame, symbol: str = "default", tf: int = None) -> Tuple[str, float]:
        R_MAP = {v: k for k, v in Config.TF_MAP.items()}
        tf_str = R_MAP.get(tf, "UNK")
        model_key = f"{symbol}_{tf_str}"

        if not self.trained.get(model_key):
            if not self.load(symbol, tf):
                return "HOLD", 0.0

        feat   = self._features(df)
        latest = feat.iloc[[-1]]
        if latest.isnull().values.any():
            return "HOLD", 0.0

        try:
            X    = self.scalers[model_key].transform(latest.values)
            prob = self.models[model_key].predict_proba(X)[0]
            cls  = self.models[model_key].classes_
            best = int(np.argmax(prob))
            sig  = {0: "SELL", 1: "HOLD", 2: "BUY"}.get(cls[best], "HOLD")
            return sig, float(prob[best])
        except Exception as e:
            log.error("ML predict error: %s", e)
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
        self._last_trade_time: Dict[str, datetime] = {}
        self._last_history_sync: datetime = datetime.min.replace(tzinfo=timezone.utc)
        
        self._init_logs()

    @staticmethod
    def _ensure_csv_header(path: str, header: List[str]) -> None:
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(header)
            return

        try:
            with open(path, 'r', newline='') as f:
                first = f.readline().strip("\n\r")
            if not first:
                with open(path, 'w', newline='') as f:
                    csv.writer(f).writerow(header)
        except Exception:
            # If we can't read the file, don't overwrite it.
            pass

    def _init_logs(self) -> None:
        self._ensure_csv_header(
            Config.LOG_FILE_TRADES,
            [
                "Timestamp", "Symbol", "Timeframe", "Side",
                "Final_Confidence", "TA_Signal", "ML_Signal", "ML_Confidence",
                "ATR", "Bid", "Ask", "Spread",
                "Volume", "SL", "TP", "Order", "Deal", "Magic",
            ],
        )
        self._ensure_csv_header(
            Config.LOG_FILE_RESULTS,
            [
                "Position_ID", "Symbol", "Side", "Volume",
                "Time_Open", "Time_Close", "Price_Open", "Price_Close",
                "Profit", "Commission", "Swap", "Magic",
            ],
        )

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

    @staticmethod
    def _is_crypto(sym: str) -> bool:
        s = (sym or "").upper()
        return "BTC" in s or "ETH" in s

    def _max_spread_atr_ratio(self, sym: str) -> float:
        return Config.MAX_SPREAD_ATR_RATIO_CRYPTO if self._is_crypto(sym) else Config.MAX_SPREAD_ATR_RATIO_FX

    def _combine_signal_from_df(
        self,
        df: pd.DataFrame,
        sym: str,
        trend: str = "NEUTRAL",
        tf: int = None,
    ) -> Tuple[str, str, str, float, float, dict]:
        ta_sig, ta_det = self.ta.get_signal(df)
        ml_sig, ml_conf = self.ml.predict(df, sym, tf)

        buy_votes = ta_det.get("buy_votes", 0)
        sell_votes = ta_det.get("sell_votes", 0)
        adx = ta_det.get("adx", 0.0)
        vote_edge = abs(buy_votes - sell_votes)
        ta_votes = max(buy_votes, sell_votes)
        ta_strength = min(1.0, ta_votes / 8.0)
        edge_strength = min(1.0, vote_edge / 4.0)

        if ta_sig == "HOLD" or adx < 18:
            sig = "HOLD"
        elif ta_sig == ml_sig and ml_conf >= Config.ML_CONFIDENCE_THRESHOLD:
            # Case 1: TA and ML agree with decent confidence
            sig = ta_sig
        elif ml_sig == ta_sig and ml_conf >= 0.35 and vote_edge >= 3:
            # Case 2: Strong TA agreement, even with lower ML confidence
            sig = ta_sig
        elif ta_sig != "HOLD" and ml_sig == "HOLD" and ta_votes >= 6 and vote_edge >= 3 and adx >= 22:
            # Case 3: Very strong TA with Neutral ML
            sig = ta_sig
        else:
            sig = "HOLD"

        # Reverse Strategy (Since the bot was losing 80% of the time)
        if getattr(Config, "REVERSE_SIGNALS", False):
            if sig == "BUY":
                sig = "SELL"
                # log.debug("[%s] Signal REVERSED: BUY -> SELL", sym)
            elif sig == "SELL":
                sig = "BUY"
                # log.debug("[%s] Signal REVERSED: SELL -> BUY", sym)

        # Apply trend filter (after reversal)
        if Config.USE_TREND_FILTER and trend != "NEUTRAL":
            if sig == "BUY" and trend != "BULL":
                # log.debug("[%s] BUY signal blocked by %s trend filter.", sym, trend)
                sig = "HOLD"
            elif sig == "SELL" and trend != "BEAR":
                # log.debug("[%s] SELL signal blocked by %s trend filter.", sym, trend)
                sig = "HOLD"

        # Calculate final confidence (even if HOLD)
        trend_bonus = 0.15 if ((sig == "BUY" and trend == "BULL") or (sig == "SELL" and trend == "BEAR")) else 0.0
        final_conf = (0.55 * ml_conf) + (0.25 * ta_strength) + (0.20 * edge_strength) + trend_bonus
        final_conf = max(0.0, min(1.0, final_conf))

        # Quality gate: only trade if the overall setup is strong enough.
        if sig in ("BUY", "SELL") and final_conf < Config.MIN_FINAL_CONFIDENCE_TO_TRADE:
            sig = "HOLD"

        return sig, ta_sig, ml_sig, ml_conf, final_conf, ta_det

    def _signal(self, sym: str, tf: int, trend_tf: int) -> Tuple[str, float, float, str, str, float]:
        # News Filter Check
        if self.news.is_news_time(sym):
            return "HOLD", 0.0, 0.0, "HOLD", "HOLD", 0.0

        # Entry timeframe data
        df = self._candles(sym, tf, Config.DATA_HISTORY)
        if df is None or len(df) < 100:
            return "HOLD", 0.0, 0.0, "HOLD", "HOLD", 0.0

        # Trend timeframe data
        trend = "NEUTRAL"
        if Config.USE_TREND_FILTER:
            df_trend = self._candles(sym, trend_tf, Config.DATA_HISTORY)
            trend = self.ta.get_trend_direction(df_trend)

        sig, ta_sig, ml_sig, ml_conf, final_conf, ta_det = self._combine_signal_from_df(df, sym, trend, tf)
        atr = self.ta.get_atr(df)

        R_MAP = {v: k for k, v in Config.TF_MAP.items()}
        tf_str = R_MAP.get(tf, "UNK")
        log.info("[%s|%s] Trend=%s | TA=%s(B:%d/S:%d) | ML=%s(%.0f%%) | FinalConf=%.0f%% | ATR=%.5f | FINAL=%s",
                 sym, tf_str, trend, ta_sig, ta_det.get("buy_votes", 0),
             ta_det.get("sell_votes", 0), ml_sig, ml_conf * 100, final_conf * 100, atr, sig)

        return sig, ml_conf, final_conf, ta_sig, ml_sig, atr

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

    def _build_order_request(self, sym: str, side: str, lot: float, atr: float = 0.0, confidence: float = 0.0) -> Optional[dict]:
        broker_sym = self._broker_symbol(sym)
        si = mt5.symbol_info(broker_sym)
        if si is None: return None
        bid, ask = self._price(sym)
        pt = si.point

        # Spread filter (common cause of instant drawdown / SL hits)
        spread = abs(ask - bid)
        if atr and atr > 0:
            ratio_limit = self._max_spread_atr_ratio(sym)
            if spread > atr * ratio_limit:
                log.info("[%s] Skip trade: spread %.5f too high vs ATR %.5f (limit %.2f×ATR)", sym, spread, atr, ratio_limit)
                return None

        confidence = max(0.0, min(1.0, confidence))
        if confidence >= 0.90:
            sl_mult, rr_mult = 1.55, 2.45
        elif confidence >= 0.80:
            sl_mult, rr_mult = 1.70, 2.20
        else:
            sl_mult, rr_mult = 1.90, 1.90

        # Dynamic SL/TP based on ATR and signal quality.
        if atr > 0:
            atr_pct = atr / max(ask, pt)
            if atr_pct > 0.004:
                sl_mult += 0.20
                rr_mult += 0.15
            elif atr_pct < 0.0015:
                sl_mult -= 0.10
            sl_dist = atr * max(1.35, sl_mult)
            tp_dist = sl_dist * rr_mult
        else:
            # Fallback to points if ATR is missing.
            pip_size = 10 * pt if si.digits in (3, 5) else pt
            sl_dist = 24 * pip_size
            tp_dist = 48 * pip_size

        min_stop_dist = (si.trade_stops_level + 5) * pt
        sl_dist = max(sl_dist, min_stop_dist)
        tp_dist = max(tp_dist, min_stop_dist)

        if side == "BUY":
            price = ask
            sl = bid - sl_dist
            tp = ask + tp_dist
            ot = mt5.ORDER_TYPE_BUY
        else:
            price = bid
            sl = ask + sl_dist
            tp = bid - tp_dist
            ot = mt5.ORDER_TYPE_SELL

        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": broker_sym,
            "volume": lot,
            "type": ot,
            "price": price,
            "sl": round(sl, si.digits),
            "tp": round(tp, si.digits),
            "deviation": 10,
            "magic": Config.MAGIC_NUMBER,
            "comment": f"ATR:{round(atr,5)}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

    def _trailing_stop(self):
        """Adjusts SL for profitable positions."""
        if not Config.TRAILING_SL: return
        
        try:
            positions = mt5.positions_get(magic=Config.MAGIC_NUMBER)
            if not positions: return

            for p in positions:
                try:
                    sym = p.symbol
                    si = mt5.symbol_info(sym)
                    if not si: continue
                    
                    # Fetch latest ATR for dynamic trailing step
                    atr = 0.0
                    if p.comment and p.comment.startswith("ATR:"):
                        try:
                            atr = float(p.comment.split(":")[1])
                        except: pass
                    if atr <= 0.0:
                        base_tf = mt5.TIMEFRAME_M15
                        df = self._candles(sym, base_tf, 20)
                        if df is None or len(df) < 14: continue
                        atr = self.ta.get_atr(df)
                    
                    trail_step = atr * 0.5 # Move SL every 0.5 ATR profit
                    bid, ask = self._price(sym)
                    
                    new_sl = 0.0
                    if p.type == mt5.POSITION_TYPE_BUY:
                        profit_dist = bid - p.price_open
                        initial_risk = max((p.price_open - p.sl) if p.sl > 0 else 0.0, atr * Config.ATR_SL_MULT, 10 * si.point)
                        if profit_dist >= initial_risk and p.sl < p.price_open:
                            new_sl = p.price_open + max(5 * si.point, initial_risk * 0.10)
                        elif profit_dist >= initial_risk * 1.5:
                            potential_sl = bid - (initial_risk * 0.85)
                            if potential_sl > p.sl + trail_step:
                                new_sl = potential_sl
                    elif p.type == mt5.POSITION_TYPE_SELL:
                        profit_dist = p.price_open - ask
                        initial_risk = max((p.sl - p.price_open) if p.sl > 0 else 0.0, atr * Config.ATR_SL_MULT, 10 * si.point)
                        if profit_dist >= initial_risk and (p.sl > p.price_open or p.sl == 0):
                            new_sl = p.price_open - max(5 * si.point, initial_risk * 0.10)
                        elif profit_dist >= initial_risk * 1.5:
                            potential_sl = ask + (initial_risk * 0.85)
                            if potential_sl < p.sl - trail_step or p.sl == 0:
                                new_sl = potential_sl

                    if new_sl > 0:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": sym,
                            "position": p.ticket,
                            "sl": round(new_sl, si.digits),
                            "tp": p.tp,
                            "magic": Config.MAGIC_NUMBER
                        }
                        res = mt5.order_send(request)
                        if res.retcode == mt5.TRADE_RETCODE_DONE:
                            log.info("[%s] Trailing SL updated to %.5f", sym, new_sl)
                except Exception as e:
                    log.error("Error in trailing stop for %s: %s", p.symbol, e)
        except Exception as e:
            log.error("Critical error in trailing stop loop: %s", e)

    def _log_trade(
        self,
        sym: str,
        tf: int,
        side: str,
        final_conf: float,
        ta_sig: str,
        ml_sig: str,
        ml_conf: float,
        atr: float,
        lot: float,
        req: dict,
        order_id: int,
        deal_id: int,
    ) -> None:
        """Logs executed trade details to CSV (for later analysis)."""
        bid, ask = self._price(sym)
        spread = abs(ask - bid)
        R_MAP = {v: k for k, v in Config.TF_MAP.items()}
        tf_str = R_MAP.get(tf, "UNK")
        with open(Config.LOG_FILE_TRADES, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                sym,
                tf_str,
                side,
                round(float(final_conf), 4),
                ta_sig,
                ml_sig,
                round(float(ml_conf), 4),
                round(float(atr), 6),
                bid,
                ask,
                spread,
                round(float(lot), 2),
                req.get("sl"),
                req.get("tp"),
                int(order_id) if order_id else "",
                int(deal_id) if deal_id else "",
                Config.MAGIC_NUMBER,
            ])

    @staticmethod
    def _order_check_passed(retcode: int) -> bool:
        # Some MT5 builds return 0 for successful order_check.
        return retcode in (0, mt5.TRADE_RETCODE_DONE)

    def validate_order(self, sym: str, side: str) -> bool:
        if not self._connect():
            return False
        try:
            df = self._candles(sym, Config.TIMEFRAMES[0][0], 20)
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

    def _order(self, sym: str, side: str, lot: float, atr: float = 0.0, confidence: float = 0.0) -> Optional[Tuple[int, int, dict]]:
        req = self._build_order_request(sym, side, lot, atr, confidence)
        if req is None:
            return None

        check = mt5.order_check(req)
        if check is None:
            log.error("Order check failed [%s]: %s", sym, mt5.last_error())
            return None
        if not self._order_check_passed(check.retcode):
            log.error("Order check rejected [%s]: %s (code %d)", sym, check.comment, check.retcode)
            return None

        res = mt5.order_send(req)
        if res.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("Order failed [%s]: %s (code %d)", sym, res.comment, res.retcode)
            return None
        self.risk.record()
        log.info("ORDER OK %s %s | lot=%.2f | price=%.5f | SL=%.5f | TP=%.5f",
                 side, sym, lot, req["price"], req["sl"], req["tp"])
        return int(getattr(res, "order", 0) or 0), int(getattr(res, "deal", 0) or 0), req

    def _has_position(self, sym: str) -> bool:
        broker_sym = self._broker_symbol(sym)
        # Only skip if the BOT itself has an open position for this symbol
        pos = mt5.positions_get(symbol=broker_sym, magic=Config.MAGIC_NUMBER)
        return pos is not None and len(pos) > 0

    def _sync_trade_results_if_needed(self, now_utc: datetime, days_back: int = 7) -> None:
        # Keep it lightweight; sync at most every 15 minutes.
        if (now_utc - self._last_history_sync).total_seconds() < 900:
            return
        self._last_history_sync = now_utc
        try:
            self._sync_trade_results(days_back=days_back)
        except Exception as e:
            log.error("Trade result sync failed: %s", e)

    def _sync_trade_results(self, days_back: int = 7) -> int:
        """Append newly-closed bot positions to Config.LOG_FILE_RESULTS.

        This enables real win-rate tracking from MT5 profit, commission, and swap.
        """
        # MT5 history API typically expects naive datetimes in terminal timezone.
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days_back)

        deals = mt5.history_deals_get(from_dt, to_dt)
        if deals is None:
            return 0

        bot_deals = [d for d in deals if int(getattr(d, "magic", 0) or 0) == int(Config.MAGIC_NUMBER)]
        if not bot_deals:
            return 0

        existing = set()
        try:
            with open(Config.LOG_FILE_RESULTS, "r", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    pid = (row.get("Position_ID") or "").strip()
                    if pid:
                        existing.add(pid)
        except Exception:
            pass

        # Group by position_id (fallback to order id if missing)
        grouped: Dict[str, List] = {}
        for d in bot_deals:
            pos_id = int(getattr(d, "position_id", 0) or 0)
            order_id = int(getattr(d, "order", 0) or 0)
            key = str(pos_id or order_id or getattr(d, "ticket", ""))
            grouped.setdefault(key, []).append(d)

        rows_to_append: List[List] = []
        for key, ds in grouped.items():
            if key in existing:
                continue
            ds_sorted = sorted(ds, key=lambda x: int(getattr(x, "time", 0) or 0))

            # Determine if position is closed (has any OUT deal)
            has_out = False
            for d in ds_sorted:
                entry = int(getattr(d, "entry", -1) or -1)
                if entry in (getattr(mt5, "DEAL_ENTRY_OUT", 1), getattr(mt5, "DEAL_ENTRY_OUT_BY", 2)):
                    has_out = True
                    break
            if not has_out:
                continue

            first = ds_sorted[0]
            last = ds_sorted[-1]
            symbol = getattr(first, "symbol", "")

            # Side: use the first IN deal type if available.
            side = ""
            for d in ds_sorted:
                entry = int(getattr(d, "entry", -1) or -1)
                if entry == getattr(mt5, "DEAL_ENTRY_IN", 0):
                    dtype = int(getattr(d, "type", -1) or -1)
                    side = "BUY" if dtype == getattr(mt5, "DEAL_TYPE_BUY", 0) else "SELL"
                    break
            if not side:
                dtype = int(getattr(first, "type", -1) or -1)
                side = "BUY" if dtype == getattr(mt5, "DEAL_TYPE_BUY", 0) else "SELL"

            volume = float(getattr(first, "volume", 0.0) or 0.0)
            t_open = datetime.fromtimestamp(int(getattr(first, "time", 0) or 0)).strftime("%Y-%m-%d %H:%M:%S")
            t_close = datetime.fromtimestamp(int(getattr(last, "time", 0) or 0)).strftime("%Y-%m-%d %H:%M:%S")
            price_open = float(getattr(first, "price", 0.0) or 0.0)
            price_close = float(getattr(last, "price", 0.0) or 0.0)

            profit = float(sum(float(getattr(d, "profit", 0.0) or 0.0) for d in ds_sorted))
            commission = float(sum(float(getattr(d, "commission", 0.0) or 0.0) for d in ds_sorted))
            swap = float(sum(float(getattr(d, "swap", 0.0) or 0.0) for d in ds_sorted))

            rows_to_append.append([
                key,
                symbol,
                side,
                volume,
                t_open,
                t_close,
                price_open,
                price_close,
                profit,
                commission,
                swap,
                Config.MAGIC_NUMBER,
            ])

        if not rows_to_append:
            return 0

        with open(Config.LOG_FILE_RESULTS, "a", newline="") as f:
            w = csv.writer(f)
            for row in rows_to_append:
                w.writerow(row)

        log.info("Synced %d closed positions into %s", len(rows_to_append), Config.LOG_FILE_RESULTS)
        return len(rows_to_append)

    def backtest(self, history: int = 2500, test_ratio: float = 0.30) -> Dict[str, dict]:
        """Run a lightweight walk-forward backtest on recent MT5 candles.

        Notes:
          - This is a *bar-based* simulator (uses OHLC), not tick-accurate.
          - It evaluates realistic outcomes by simulating SL/TP hits using bar High/Low.
          - It still isn't a substitute for forward testing (spread/slippage/partial fills).
        """
        results: Dict[str, dict] = {}
        if not self._connect():
            return results

        try:
            for sym in Config.SYMBOLS:
                df = self._candles(sym, Config.TIMEFRAMES[0][0], history)
                if df is None or len(df) < 220:
                    log.warning("[%s] Not enough candles for backtest.", sym)
                    continue

                split_idx = int(len(df) * (1 - test_ratio))
                train_df = df.iloc[:split_idx].copy()
                test_df = df.iloc[split_idx:].copy()
                if len(train_df) < 120 or len(test_df) < 60:
                    log.warning("[%s] Backtest split too small.", sym)
                    continue

                self.ml.train(train_df, sym, Config.TIMEFRAMES[0][0])

                trades = 0
                wins = 0
                losses = 0
                r_mults: List[float] = []

                # conservative equity curve in R-units
                eq = 0.0
                peak = 0.0
                max_dd = 0.0

                start_i = min(200, max(60, len(test_df) // 3))
                # i is the index of the *signal evaluation* bar; entry happens on i+1 open.
                for i in range(start_i, len(test_df) - 2):
                    window = pd.concat([train_df, test_df.iloc[: i + 1]])

                    trend = self.ta.get_trend_direction(window) if Config.USE_TREND_FILTER else "NEUTRAL"
                    sig, _, _, ml_conf, final_conf, _ = self._combine_signal_from_df(window, sym, trend, Config.TIMEFRAMES[0][0])
                    if sig not in ("BUY", "SELL"):
                        continue

                    # Entry at next bar open (more realistic than same-bar close)
                    entry_i = i + 1
                    entry = float(test_df["Open"].iloc[entry_i])

                    atr = float(self.ta.get_atr(window))
                    if not np.isfinite(atr) or atr <= 0:
                        continue

                    conf = float(final_conf)
                    if conf >= 0.90:
                        sl_mult, rr_mult = 1.55, 2.45
                    elif conf >= 0.80:
                        sl_mult, rr_mult = 1.70, 2.20
                    else:
                        sl_mult, rr_mult = 1.90, 1.90

                    # Small volatility-based adjustment (mirrors live behavior)
                    atr_pct = atr / max(abs(entry), 1e-9)
                    if atr_pct > 0.004:
                        sl_mult += 0.20
                        rr_mult += 0.15
                    elif atr_pct < 0.0015:
                        sl_mult -= 0.10

                    sl_dist = atr * max(1.35, sl_mult)
                    tp_dist = sl_dist * rr_mult

                    if sig == "BUY":
                        sl = entry - sl_dist
                        tp = entry + tp_dist
                    else:
                        sl = entry + sl_dist
                        tp = entry - tp_dist

                    outcome_r: Optional[float] = None
                    # Walk forward until SL/TP hit
                    for j in range(entry_i, len(test_df)):
                        hi = float(test_df["High"].iloc[j])
                        lo = float(test_df["Low"].iloc[j])

                        if sig == "BUY":
                            hit_sl = lo <= sl
                            hit_tp = hi >= tp
                        else:
                            hit_sl = hi >= sl
                            hit_tp = lo <= tp

                        # If both hit in same bar, assume worst-case (prevents optimistic bias)
                        if hit_sl and hit_tp:
                            outcome_r = -1.0
                            break
                        if hit_tp:
                            outcome_r = rr_mult
                            break
                        if hit_sl:
                            outcome_r = -1.0
                            break

                    if outcome_r is None:
                        # Close at end of test (partial R)
                        close_end = float(test_df["Close"].iloc[-1])
                        if sig == "BUY":
                            outcome_r = (close_end - entry) / sl_dist
                        else:
                            outcome_r = (entry - close_end) / sl_dist

                    trades += 1
                    r_mults.append(float(outcome_r))
                    if outcome_r > 0:
                        wins += 1
                    else:
                        losses += 1

                    eq += float(outcome_r)
                    peak = max(peak, eq)
                    max_dd = max(max_dd, peak - eq)

                win_rate = (wins / trades) if trades else 0.0
                avg_r = float(np.mean(r_mults)) if r_mults else 0.0
                gross_win = float(sum(r for r in r_mults if r > 0))
                gross_loss = float(sum(abs(r) for r in r_mults if r < 0))
                profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float('inf') if gross_win > 0 else 0.0)

                results[sym] = {
                    "bars": len(df),
                    "train": len(train_df),
                    "test": len(test_df),
                    "trades": trades,
                    "win_rate": round(win_rate, 4),
                    "avg_r": round(avg_r, 4),
                    "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else float('inf'),
                    "max_drawdown_r": round(float(max_dd), 4),
                }

                log.info("[BT %s] win=%.1f%% | avgR=%.2f | PF=%.2f | maxDD(R)=%.2f | trades=%d",
                         sym, win_rate * 100, avg_r, profit_factor if np.isfinite(profit_factor) else 0.0, max_dd, trades)

            return results
        finally:
            mt5.shutdown()

    # ── Main Loop ─────────────────────────────────────────────
    def run(self):
        # Create reverse map for logging
        R_MAP = {v: k for k, v in Config.TF_MAP.items()}
        tfs_str = ", ".join([f"{R_MAP.get(entry, 'UNK')}->{R_MAP.get(trend, 'UNK')}" for entry, trend in Config.TIMEFRAMES])

        log.info("=" * 60)
        log.info("  FOREX BOT STARTING")
        log.info("  Symbols      : %s", Config.SYMBOLS)
        log.info("  Timeframes   : %s", tfs_str)
        if Config.USE_TREND_FILTER:
            log.info("  Trend Filter : ENABLED")
        else:
            log.info("  Trend Filter : DISABLED")
        log.info("  Risk/trade   : %.1f%%  |  Max DD: %.1f%%", Config.RISK_PCT, Config.MAX_DRAWDOWN_PCT)
        log.info("=" * 60)

        if not self._connect():
            return

        # ── Initial ML training ───────────────────────────────
        log.info("Training ML models (this may take a moment)...")
        for sym in Config.SYMBOLS:
            for tf, _ in Config.TIMEFRAMES:
                df = self._candles(sym, tf, Config.TRAIN_HISTORY)
                if df is not None and len(df) >= 100:
                    self.ml.train(df, sym, tf)
                else:
                    log.warning("[%s|%s] Insufficient history for training.", sym, R_MAP.get(tf, "UNK"))
        log.info("ML models ready.")

        self.running = True
        try:
            while self.running:
                try:
                    now_utc = datetime.now(timezone.utc)
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

                    # Sync closed trade results periodically (to measure real win-rate)
                    self._sync_trade_results_if_needed(now_utc)

                    # Process each symbol
                    for sym in Config.SYMBOLS:
                        try:
                            if not self.risk.can_trade(acc.balance):
                                break

                            last_t = self._last_trade_time.get(sym)
                            if last_t is not None:
                                cooldown_s = Config.TRADE_COOLDOWN_MINS * 60
                                if (now_utc - last_t).total_seconds() < cooldown_s:
                                    continue

                            if self._has_position(sym):
                                # log.info("[%s] Position open — skipping.", sym)
                                continue

                            # Evaluate each timeframe pair
                            for tf, trend_tf in Config.TIMEFRAMES:
                                sig, ml_conf, final_conf, ta_sig, ml_sig, atr = self._signal(sym, tf, trend_tf)
                                if sig in ("BUY", "SELL"):
                                    si = mt5.symbol_info(self._broker_symbol(sym))
                                    sl_points = int((atr * Config.ATR_SL_MULT) / si.point) if atr > 0 else 300
                                    lot = self._lot(sym, sl_points, final_conf)
                                    order_info = self._order(sym, sig, lot, atr, final_conf)
                                    if order_info is not None:
                                        order_id, deal_id, req = order_info
                                        self._log_trade(sym, tf, sig, final_conf, ta_sig, ml_sig, ml_conf, atr, lot, req, order_id, deal_id)
                                        self._last_trade_time[sym] = now_utc
                                    # If trade taken, skip other timeframes for this symbol
                                    break

                        except Exception as e:
                            log.error("[%s] Error in symbol loop: %s", sym, e)

                except Exception as e:
                    log.error("Main loop error: %s. Continuing...", e)
                    time.sleep(5)

                # log.info("— scan complete — sleeping %ds —", Config.LOOP_INTERVAL)
                
                # ONLINE LEARNING: Continuous Feedback Loop
                self.loop_count += 1
                if self.loop_count >= 240:  # Every ~4 hours
                    log.info("Continuous Feedback Loop: Retraining models with latest data + penalties...")
                    for sym in Config.SYMBOLS:
                        for tf, _ in Config.TIMEFRAMES:
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
    parser.add_argument("--tf", default=None, help="Primary entry timeframe (M1, M5, M15, M30, H1, H4, D1)")
    parser.add_argument("--trend-tf", default=None, help="Trend filter timeframe (used only if --use-filter is passed)")
    parser.add_argument("--use-filter", action="store_true", help="Enable higher timeframe trend filter")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol for check-order mode")
    parser.add_argument("--side", choices=["BUY", "SELL"], default="BUY", help="Side for check-order mode")
    parser.add_argument("--history", type=int, default=2500, help="Backtest candles")
    args = parser.parse_args()

    # Apply command-line overrides
    if args.tf in Config.TF_MAP:
        # Override the first timeframe pair
        Config.TIMEFRAMES = [(Config.TF_MAP[args.tf], Config.TIMEFRAMES[0][1])]
        # Adjust threshold based on timeframe
        if args.tf == "M15":
            Config.LABEL_THRESHOLD = 0.0003
        elif args.tf in ("H1", "H4"):
            Config.LABEL_THRESHOLD = 0.0010
        elif args.tf == "D1":
            Config.LABEL_THRESHOLD = 0.0050
            
    if args.trend_tf in Config.TF_MAP:
        # Override the trend TF of the first pair (or the one just set)
        Config.TIMEFRAMES = [(Config.TIMEFRAMES[0][0], Config.TF_MAP[args.trend_tf])]

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
                        f"{sym}: win_rate={stats.get('win_rate', 0)*100:.1f}% | "
                        f"avgR={stats.get('avg_r', 0):.2f} | "
                        f"PF={stats.get('profit_factor', 0):.2f} | "
                        f"maxDD(R)={stats.get('max_drawdown_r', 0):.2f} | "
                        f"trades={stats.get('trades', 0)}"
                    )
        else:
            ok = bot.validate_order(args.symbol, args.side)
            print(f"order_check {'PASS' if ok else 'FAIL'} for {args.side} {args.symbol}")