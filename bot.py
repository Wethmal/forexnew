"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           GOLDEN BOT v5 — SIGNAL SCANNER + PAPER/LIVE TRADER               ║
║                                                                              ║
║  DEFAULT MODE: PAPER TRADING (no real money at risk)                        ║
║  TO ENABLE LIVE: Set LIVE_TRADING = True in Config AND set MT5 credentials  ║
║                                                                              ║
║  Strategies:                                                                 ║
║    S1 — EMA Cross + RSI Pullback (trend-momentum)                           ║
║    S2 — Bollinger Band + Stochastic (mean reversion)                        ║
║    S3 — MACD Divergence + Price Action (confluence)                         ║
║                                                                              ║
║  Safety rules built-in:                                                     ║
║    - Max 1% risk per trade                                                   ║
║    - Max 3% daily loss (bot pauses if hit)                                  ║
║    - Max 3 open positions                                                    ║
║    - Stop after 3 consecutive losses                                         ║
║    - Spread check before every entry                                         ║
║    - News filter (Finnhub / Forex Factory)                                  ║
║                                                                              ║
║  Install:  pip install yfinance pandas numpy requests                        ║
║  MT5 live: pip install MetaTrader5  (Windows only)                          ║
║                                                                              ║
║  Run:                                                                        ║
║    python golden_bot_v5.py --paper       (safe, no real money)              ║
║    python golden_bot_v5.py --signal      (print signals once and exit)      ║
║    python golden_bot_v5.py --backtest    (historical test)                  ║
║    python golden_bot_v5.py --stats       (show trade journal stats)         ║
║    python golden_bot_v5.py --live        (REAL MONEY — requires MT5 setup)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import os, sys, json, time, logging, warnings, traceback, requests
from copy import deepcopy
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Optional dependencies
try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False

try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score
    HAS_SKL = True
except ImportError:
    HAS_SKL = False


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stdout)
log = logging.getLogger("GoldenBot")

class ColorLog(logging.Formatter):
    RESET = "\033[0m"
    MAP   = {
        logging.DEBUG:    "\033[90m",
        logging.INFO:     "\033[0m",
        logging.WARNING:  "\033[93m",
        logging.ERROR:    "\033[91m",
        logging.CRITICAL: "\033[95m",
    }
    def format(self, record):
        c   = self.MAP.get(record.levelno, self.RESET)
        msg = super().format(record)
        msg = msg.replace("BUY",  "\033[92mBUY\033[0m")
        msg = msg.replace("SELL", "\033[91mSELL\033[0m")
        msg = msg.replace("WIN",  "\033[92mWIN\033[0m")
        msg = msg.replace("LOSS", "\033[91mLOSS\033[0m")
        return f"{c}{msg}{self.RESET}"

for h in logging.root.handlers:
    h.setFormatter(ColorLog(LOG_FORMAT))


# ─────────────────────────────────────────────
#  CONFIGURATION  ← edit here
# ─────────────────────────────────────────────
@dataclass
class Config:

    # ── Symbols ─────────────────────────────
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"
    ])
    mt5_suffix: str = ""          # e.g. "m" if broker appends 'm' to symbols

    # ── Timeframes ──────────────────────────
    tf_entry:   str = "15m"
    tf_trend:   str = "1h"
    tf_confirm: str = "4h"

    # ── EMA / RSI ───────────────────────────
    ema_fast:  int   = 9
    ema_slow:  int   = 21
    ema_50:    int   = 50
    ema_200:   int   = 200
    rsi_period: int  = 14
    rsi_bull_lo: float = 40.0    # RSI pullback zone for BUY
    rsi_bull_hi: float = 55.0
    rsi_ob:     float = 70.0     # Overbought
    rsi_os:     float = 30.0     # Oversold

    # ── Bollinger / Stochastic ───────────────
    bb_period: int   = 20
    bb_std:    float = 2.0
    stoch_k:   int   = 14
    stoch_d:   int   = 3
    stoch_smooth: int = 3
    stoch_ob:  float = 80.0
    stoch_os:  float = 20.0
    bb_touch:  float = 0.002     # within 0.2% counts as "touch"

    # ── MACD / Divergence / S/R ──────────────
    macd_fast:    int   = 12
    macd_slow:    int   = 26
    macd_sig:     int   = 9
    div_lookback: int   = 30
    sr_lookback:  int   = 50
    sr_tol:       float = 0.003  # 0.3% zone
    pin_ratio:    float = 0.60
    engulf_ratio: float = 1.10

    # ── Signal scoring ───────────────────────
    w1: float = 0.35             # S1 weight
    w2: float = 0.30             # S2 weight
    w3: float = 0.35             # S3 weight
    min_score: float = 6.0       # minimum combined score to trade (raised from 5.5)

    # ── Volume ───────────────────────────────
    vol_period: int   = 20
    vol_min:    float = 1.0

    # ── ATR / Risk ───────────────────────────
    atr_period:   int   = 14
    sl_atr_mult:  float = 2.0
    rr_target:    float = 2.5
    rr_min:       float = 2.0
    risk_pct:     float = 0.01   # 1% per trade
    max_positions: int  = 3
    daily_loss_limit: float = 0.03   # 3% daily loss stops bot
    consec_loss_limit: int  = 3      # stop after N losses in a row
    lot_min:  float = 0.01
    lot_max:  float = 0.10
    lot_step: float = 0.01
    spread_max: float = 30.0     # pips

    # ── News filter ──────────────────────────
    news_enabled: bool  = True
    finnhub_key:  str   = ""     # paste your free key from finnhub.io
    news_cache:   str   = "news_cache.json"

    # ── ML filter (optional) ────────────────
    use_ml:      bool  = False   # enable after bot proves itself in paper
    ml_min_prob: float = 0.65

    # ── Execution ────────────────────────────
    #
    #  ⚠️  LIVE_TRADING IS FALSE BY DEFAULT.
    #  Only set True after 2+ weeks of paper results show profit.
    #
    live_trading: bool = False
    mt5_magic:   int  = 20250418
    mt5_comment: str  = "GoldenBot v5"

    # ── Sessions (UTC hour ranges to trade) ──
    sessions: List[Tuple[int, int]] = field(default_factory=lambda: [
        (7, 16), (13, 22)        # London + NY overlap
    ])

    # ── Files ────────────────────────────────
    log_file:       str = "trades.json"
    dashboard_file: str = "dashboard.json"
    interval:       int = 60     # seconds between loops

    def spread_limit(self, tf: str) -> float:
        return {
            "1m": 2.0, "5m": 3.0, "15m": 5.0,
            "30m": 10.0, "1h": 15.0, "4h": 25.0
        }.get(tf, self.spread_max)


# ─────────────────────────────────────────────
#  ENUMS & DATA CLASSES
# ─────────────────────────────────────────────
class Dir(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class Strat(Enum):
    S1 = "S1_EMACross"
    S2 = "S2_MeanRevert"
    S3 = "S3_Divergence"
    COMBO = "COMBO"

@dataclass
class Score:
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    r1: List[str] = field(default_factory=list)
    r2: List[str] = field(default_factory=list)
    r3: List[str] = field(default_factory=list)

    def combined(self, cfg: Config) -> float:
        return self.s1 * cfg.w1 + self.s2 * cfg.w2 + self.s3 * cfg.w3

    def dominant(self) -> Strat:
        m = max([(self.s1, Strat.S1), (self.s2, Strat.S2), (self.s3, Strat.S3)],
                key=lambda x: x[0])
        return m[1]

@dataclass
class Signal:
    symbol:     str
    direction:  Dir
    entry:      float
    sl:         float
    tp:         float
    sl_dist:    float
    tp_dist:    float
    rr:         float
    lots:       float
    confidence: float
    score:      Score
    strategy:   str
    rsi:        float
    macd_hist:  float
    atr:        float
    spread:     float
    reason:     str
    timestamp:  str = field(default_factory=lambda: datetime.now().isoformat())
    ml_prob:    float = 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["score_combined"] = round(self.score.combined(Config()), 2)
        return d


# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────
class Ind:

    @staticmethod
    def ema(s: pd.Series, n: int) -> pd.Series:
        return s.ewm(span=n, adjust=False).mean()

    @staticmethod
    def rsi(s: pd.Series, n: int = 14) -> pd.Series:
        d = s.diff()
        g = d.clip(lower=0).rolling(n).mean()
        l = (-d.clip(upper=0)).rolling(n).mean()
        return 100 - (100 / (1 + g / l.replace(0, np.nan)))

    @staticmethod
    def macd(s: pd.Series, f=12, sl=26, sig=9):
        ml  = s.ewm(span=f, adjust=False).mean() - s.ewm(span=sl, adjust=False).mean()
        msl = ml.ewm(span=sig, adjust=False).mean()
        return ml, msl, ml - msl

    @staticmethod
    def bb(s: pd.Series, n=20, std=2.0):
        mid = s.rolling(n).mean()
        sd  = s.rolling(n).std()
        return mid + sd*std, mid, mid - sd*std

    @staticmethod
    def stoch(df: pd.DataFrame, k=14, d=3, sm=3):
        lo = df["Low"].rolling(k).min()
        hi = df["High"].rolling(k).max()
        sk = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
        sk = sk.rolling(sm).mean()
        return sk, sk.rolling(d).mean()

    @staticmethod
    def atr(df: pd.DataFrame, n=14) -> pd.Series:
        hl = df["High"] - df["Low"]
        hc = (df["High"] - df["Close"].shift()).abs()
        lc = (df["Low"]  - df["Close"].shift()).abs()
        return pd.concat([hl, hc, lc], axis=1).max(axis=1).ewm(span=n, adjust=False).mean()

    @staticmethod
    def adx(df: pd.DataFrame, n=14):
        hi, lo, cl = df["High"], df["Low"], df["Close"]
        pdm = hi.diff().clip(lower=0)
        ndm = (-lo.diff()).clip(lower=0)
        pdm[pdm < ndm] = 0
        ndm[ndm < pdm] = 0
        tr  = pd.concat([hi-lo, (hi-cl.shift()).abs(), (lo-cl.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=n, adjust=False).mean()
        pdi = 100 * pdm.ewm(span=n, adjust=False).mean() / atr
        ndi = 100 * ndm.ewm(span=n, adjust=False).mean() / atr
        dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
        return dx.ewm(span=n, adjust=False).mean(), pdi, ndi

    @staticmethod
    def volratio(df: pd.DataFrame, n=20) -> pd.Series:
        if "Volume" not in df or (df["Volume"] == 0).all():
            return pd.Series(1.0, index=df.index)
        return df["Volume"] / df["Volume"].rolling(n).mean().replace(0, np.nan)

    @classmethod
    def compute(cls, df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        df  = df.copy()
        c   = df["Close"]
        df["e_fast"] = cls.ema(c, cfg.ema_fast)
        df["e_slow"] = cls.ema(c, cfg.ema_slow)
        df["e50"]    = cls.ema(c, cfg.ema_50)
        df["e200"]   = cls.ema(c, cfg.ema_200)
        df["rsi"]    = cls.rsi(c, cfg.rsi_period)
        df["macd"], df["macd_s"], df["macd_h"] = cls.macd(c, cfg.macd_fast, cfg.macd_slow, cfg.macd_sig)
        df["bb_u"], df["bb_m"], df["bb_l"]     = cls.bb(c, cfg.bb_period, cfg.bb_std)
        df["bb_w"]   = (df["bb_u"] - df["bb_l"]) / df["bb_m"]
        df["bb_pos"] = (c - df["bb_l"]) / (df["bb_u"] - df["bb_l"]).replace(0, np.nan)
        df["sk"], df["sd"] = cls.stoch(df, cfg.stoch_k, cfg.stoch_d, cfg.stoch_smooth)
        df["atr"]    = cls.atr(df, cfg.atr_period)
        df["adx"], df["pdi"], df["ndi"] = cls.adx(df, cfg.atr_period)
        df["vr"]     = cls.volratio(df, cfg.vol_period)
        df["body"]   = (df["Close"] - df["Open"]).abs()
        df["rng"]    = df["High"] - df["Low"]
        df["ush"]    = df["High"] - df[["Open","Close"]].max(axis=1)
        df["lsh"]    = df[["Open","Close"]].min(axis=1) - df["Low"]
        df["brat"]   = df["body"] / df["rng"].replace(0, np.nan)
        df["bull"]   = (df["Close"] > df["Open"]).astype(int)
        df["ret"]    = c.pct_change()
        df.dropna(inplace=True)
        return df


# ─────────────────────────────────────────────
#  PRICE ACTION
# ─────────────────────────────────────────────
class PA:

    @staticmethod
    def pin(row, direction: str, ratio=0.6) -> bool:
        if row["rng"] == 0:
            return False
        if direction == "bull":
            return row["lsh"]/row["rng"] >= ratio and row["body"]/row["rng"] <= 0.35
        return row["ush"]/row["rng"] >= ratio and row["body"]/row["rng"] <= 0.35

    @staticmethod
    def engulf(cur, prev, direction: str, ratio=1.1) -> bool:
        if prev["body"] == 0:
            return False
        br = cur["body"] / prev["body"]
        if direction == "bull":
            return cur["bull"] == 1 and prev["bull"] == 0 and br >= ratio
        return cur["bull"] == 0 and prev["bull"] == 1 and br >= ratio

    @staticmethod
    def doji(row, max_pct=0.1) -> bool:
        return row["rng"] > 0 and row["body"]/row["rng"] <= max_pct

    @staticmethod
    def sr_levels(df: pd.DataFrame, lb=50, tol=0.003):
        hi = df["High"].values[-lb:]
        lo = df["Low"].values[-lb:]
        res, sup = [], []
        for i in range(2, len(hi)-2):
            if hi[i] == max(hi[i-2:i+3]):
                res.append(hi[i])
            if lo[i] == min(lo[i-2:i+3]):
                sup.append(lo[i])
        def cluster(lvls):
            if not lvls: return []
            lvls = sorted(lvls)
            out  = [lvls[0]]
            for v in lvls[1:]:
                if abs(v - out[-1]) / out[-1] > tol:
                    out.append(v)
                else:
                    out[-1] = (out[-1] + v) / 2
            return out
        return cluster(sup), cluster(res)

    @staticmethod
    def near(price, levels, tol=0.003) -> bool:
        return any(abs(price - l)/l <= tol for l in levels)


# ─────────────────────────────────────────────
#  DIVERGENCE ENGINE
# ─────────────────────────────────────────────
class Div:

    @staticmethod
    def pivots(s: pd.Series, left=5, right=5):
        v  = s.values
        ph, pl = [], []
        for i in range(left, len(v)-right):
            w = v[i-left:i+right+1]
            if v[i] == max(w): ph.append(i)
            if v[i] == min(w): pl.append(i)
        return ph, pl

    @classmethod
    def detect(cls, df: pd.DataFrame, lb=40) -> dict:
        res = {"rb": False, "bb_div": False, "hb": False, "hb_bear": False, "strength": 0.0}
        if len(df) < lb+10:
            return res
        w = df.tail(lb)
        p, h = w["Close"], w["macd_h"]
        pph, ppl = cls.pivots(p)
        hph, hpl = cls.pivots(h)
        if len(pph) >= 2 and len(hph) >= 2:
            pp1, pp2 = pph[-2], pph[-1]
            hp1, hp2 = hph[-2], hph[-1]
            if p.iloc[pp2] > p.iloc[pp1] and h.iloc[hp2] < h.iloc[hp1] and h.iloc[hp2] < 0:
                res["bb_div"] = True
                res["strength"] = max(res["strength"], min(abs(h.iloc[hp1]-h.iloc[hp2])*500, 1.0))
            if p.iloc[pp2] < p.iloc[pp1] and h.iloc[hp2] > h.iloc[hp1]:
                res["hb_bear"] = True
        if len(ppl) >= 2 and len(hpl) >= 2:
            pp1, pp2 = ppl[-2], ppl[-1]
            lp1, lp2 = hpl[-2], hpl[-1]
            if p.iloc[pp2] < p.iloc[pp1] and h.iloc[lp2] > h.iloc[lp1] and h.iloc[lp2] < 0:
                res["rb"] = True
                res["strength"] = max(res["strength"], min(abs(h.iloc[lp1]-h.iloc[lp2])*500, 1.0))
            if p.iloc[pp2] > p.iloc[pp1] and h.iloc[lp2] < h.iloc[lp1]:
                res["hb"] = True
        return res


# ─────────────────────────────────────────────
#  STRATEGY 1 — EMA Cross + RSI Pullback
# ─────────────────────────────────────────────
class S1:
    @staticmethod
    def score(df: pd.DataFrame, cfg: Config, d: str) -> Tuple[float, List[str]]:
        if len(df) < 3: return 0.0, []
        cur, prev, prev2 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
        s, r = 0.0, []

        e50, e200 = cur["e50"], cur["e200"]
        ef, es    = cur["e_fast"], cur["e_slow"]
        rsi       = cur["rsi"]
        rp, rp2   = prev["rsi"], prev2["rsi"]
        adx       = cur["adx"]
        cl        = cur["Close"]
        vr        = cur["vr"]

        if d == "BUY":
            if e50 > e200:
                s += 2.0; r.append("Golden Cross: EMA50 > EMA200")
            elif e50 > e200 * 0.999:
                s += 0.5; r.append("EMA50 near EMA200 (forming)")
            if cl > e50:    s += 1.5; r.append("Price above EMA50")
            if ef > es:     s += 1.0; r.append(f"EMA{cfg.ema_fast} > EMA{cfg.ema_slow}")
            if cfg.rsi_bull_lo <= rsi <= cfg.rsi_bull_hi and rp <= rsi:
                s += 2.0; r.append(f"RSI pullback {rsi:.0f} → rising (momentum reset)")
            elif cfg.rsi_bull_lo <= rsi <= 60:
                s += 1.0; r.append(f"RSI {rsi:.0f} in bullish zone")
            if rsi < cfg.rsi_ob:    s += 1.0; r.append(f"RSI {rsi:.0f} not overbought")
            if adx > 25:            s += 1.0; r.append(f"ADX {adx:.0f} strong trend")
            elif adx > 20:          s += 0.5; r.append(f"ADX {adx:.0f} moderate trend")
            if vr >= cfg.vol_min:   s += 0.5; r.append(f"Volume {vr:.1f}x average")
        else:
            if e50 < e200:
                s += 2.0; r.append("Death Cross: EMA50 < EMA200")
            elif e50 < e200 * 1.001:
                s += 0.5; r.append("EMA50 near EMA200 (death forming)")
            if cl < e50:    s += 1.5; r.append("Price below EMA50")
            if ef < es:     s += 1.0; r.append(f"EMA{cfg.ema_fast} < EMA{cfg.ema_slow}")
            if 45.0 <= rsi <= 60.0 and rp >= rsi:
                s += 2.0; r.append(f"RSI {rsi:.0f} dead-cat bounce → falling")
            elif 35 <= rsi <= 60:
                s += 1.0; r.append(f"RSI {rsi:.0f} in sell zone")
            if rsi > cfg.rsi_os:    s += 1.0; r.append(f"RSI {rsi:.0f} not oversold")
            if adx > 25:            s += 1.0; r.append(f"ADX {adx:.0f} strong downtrend")
            elif adx > 20:          s += 0.5
            if vr >= cfg.vol_min:   s += 0.5; r.append(f"Volume {vr:.1f}x average")

        return min(s, 10.0), r


# ─────────────────────────────────────────────
#  STRATEGY 2 — Bollinger + Stochastic
# ─────────────────────────────────────────────
class S2:
    @staticmethod
    def score(df: pd.DataFrame, cfg: Config, d: str) -> Tuple[float, List[str]]:
        if len(df) < 3: return 0.0, []
        cur, prev = df.iloc[-1], df.iloc[-2]
        s, r = 0.0, []

        cl     = cur["Close"]
        bbu    = cur["bb_u"]
        bbl    = cur["bb_l"]
        bbw    = cur["bb_w"]
        sk     = cur["sk"]
        sd     = cur["sd"]
        skp    = prev["sk"]
        sdp    = prev["sd"]
        rsi    = cur["rsi"]
        vr     = cur["vr"]
        ld     = abs(cl - bbl) / bbl
        ud     = abs(cl - bbu) / bbu

        if d == "BUY":
            if cl <= bbl:
                s += 2.5; r.append("Price BELOW lower Bollinger Band")
            elif ld <= cfg.bb_touch:
                s += 2.0; r.append(f"Price touching lower BB ({ld*100:.2f}% away)")
            elif ld <= cfg.bb_touch*2:
                s += 1.0; r.append("Price near lower BB")
            cross_up = skp <= sdp and sk > sd
            if sk < cfg.stoch_os and cross_up:
                s += 2.5; r.append(f"Stoch {sk:.0f} oversold + crossing UP")
            elif sk < cfg.stoch_os:
                s += 1.5; r.append(f"Stoch {sk:.0f} oversold")
            elif sk < 30 and cross_up:
                s += 1.0; r.append(f"Stoch crossing up at {sk:.0f}")
            if rsi < 35:    s += 1.5; r.append(f"RSI {rsi:.0f} oversold")
            elif rsi < 40:  s += 0.75; r.append(f"RSI {rsi:.0f} near oversold")
            if PA.pin(cur, "bull", cfg.pin_ratio):
                s += 2.0; r.append("Bullish pin bar at lower band")
            elif PA.engulf(cur, prev, "bull", cfg.engulf_ratio):
                s += 2.0; r.append("Bullish engulfing candle")
            elif cur["bull"] == 1 and cur["brat"] > 0.5:
                s += 0.5; r.append("Bullish close")
            if bbw > df["bb_w"].rolling(20).mean().iloc[-1]:
                s += 0.5; r.append("Bollinger Band widening")
        else:
            if cl >= bbu:
                s += 2.5; r.append("Price ABOVE upper Bollinger Band")
            elif ud <= cfg.bb_touch:
                s += 2.0; r.append(f"Price touching upper BB ({ud*100:.2f}% away)")
            elif ud <= cfg.bb_touch*2:
                s += 1.0; r.append("Price near upper BB")
            cross_dn = skp >= sdp and sk < sd
            if sk > cfg.stoch_ob and cross_dn:
                s += 2.5; r.append(f"Stoch {sk:.0f} overbought + crossing DOWN")
            elif sk > cfg.stoch_ob:
                s += 1.5; r.append(f"Stoch {sk:.0f} overbought")
            elif sk > 70 and cross_dn:
                s += 1.0; r.append(f"Stoch crossing down at {sk:.0f}")
            if rsi > 65:    s += 1.5; r.append(f"RSI {rsi:.0f} overbought")
            elif rsi > 60:  s += 0.75; r.append(f"RSI {rsi:.0f} elevated")
            if PA.pin(cur, "bear", cfg.pin_ratio):
                s += 2.0; r.append("Bearish pin bar at upper band")
            elif PA.engulf(cur, prev, "bear", cfg.engulf_ratio):
                s += 2.0; r.append("Bearish engulfing candle")
            elif cur["bull"] == 0 and cur["brat"] > 0.5:
                s += 0.5; r.append("Bearish close")
            if bbw > df["bb_w"].rolling(20).mean().iloc[-1]:
                s += 0.5; r.append("Bollinger Band widening")

        return min(s, 10.0), r


# ─────────────────────────────────────────────
#  STRATEGY 3 — MACD Divergence + S/R
# ─────────────────────────────────────────────
class S3:
    @staticmethod
    def score(df: pd.DataFrame, cfg: Config, d: str) -> Tuple[float, List[str]]:
        if len(df) < 40: return 0.0, []
        cur, prev = df.iloc[-1], df.iloc[-2]
        s, r = 0.0, []

        cl  = cur["Close"]
        vr  = cur["vr"]
        mh  = cur["macd_h"]
        mhp = prev["macd_h"]
        rsi = cur["rsi"]

        sup, res = PA.sr_levels(df, cfg.sr_lookback, cfg.sr_tol)
        div      = Div.detect(df, cfg.div_lookback)

        if d == "BUY":
            if div["rb"]:
                s += 3.0 + div["strength"]
                r.append(f"REGULAR BULLISH DIVERGENCE (strength {div['strength']:.2f})")
            elif div["hb"]:
                s += 1.5; r.append("Hidden bullish divergence (trend continuation)")
            if PA.near(cl, sup, cfg.sr_tol):
                s += 2.0
                nearest = min(sup, key=lambda x: abs(x-cl)) if sup else cl
                r.append(f"Price at support {nearest:.5f}")
            if PA.pin(cur, "bull", cfg.pin_ratio):
                s += 2.0; r.append("Bullish pin bar at support")
            elif PA.engulf(cur, prev, "bull", cfg.engulf_ratio):
                s += 2.0; r.append("Bullish engulfing at support")
            elif PA.doji(cur):
                s += 0.75; r.append("Doji at support (indecision)")
            if mh > mhp and mhp < 0:
                s += 1.0; r.append("MACD histogram turning UP (momentum shift)")
            if vr >= 1.5:
                s += 0.5; r.append(f"Volume spike {vr:.1f}x")
            elif vr >= cfg.vol_min:
                s += 0.25
            if rsi < 60: s += 0.5; r.append(f"RSI {rsi:.0f} has room to run")
        else:
            if div["bb_div"]:
                s += 3.0 + div["strength"]
                r.append(f"REGULAR BEARISH DIVERGENCE (strength {div['strength']:.2f})")
            elif div["hb_bear"]:
                s += 1.5; r.append("Hidden bearish divergence (trend continuation)")
            if PA.near(cl, res, cfg.sr_tol):
                s += 2.0
                nearest = min(res, key=lambda x: abs(x-cl)) if res else cl
                r.append(f"Price at resistance {nearest:.5f}")
            if PA.pin(cur, "bear", cfg.pin_ratio):
                s += 2.0; r.append("Bearish pin bar at resistance")
            elif PA.engulf(cur, prev, "bear", cfg.engulf_ratio):
                s += 2.0; r.append("Bearish engulfing at resistance")
            elif PA.doji(cur):
                s += 0.75; r.append("Doji at resistance (potential reversal)")
            if mh < mhp and mhp > 0:
                s += 1.0; r.append("MACD histogram turning DOWN")
            if vr >= 1.5:
                s += 0.5; r.append(f"Volume spike {vr:.1f}x")
            elif vr >= cfg.vol_min:
                s += 0.25
            if rsi > 40: s += 0.5; r.append(f"RSI {rsi:.0f} has room to fall")

        return min(s, 10.0), r


# ─────────────────────────────────────────────
#  DATA FETCHER
# ─────────────────────────────────────────────
class Fetcher:
    YF_INTERVAL = {"1m":"1m","5m":"5m","15m":"15m","30m":"30m",
                   "1h":"1h","4h":"1h","1d":"1d"}
    YF_PERIOD   = {"1m":"7d","5m":"30d","15m":"60d","30m":"60d",
                   "1h":"90d","4h":"180d","1d":"2y"}

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _yf_sym(self, sym: str) -> str:
        if len(sym) == 6 and sym.isalpha():
            return f"{sym[:3]}{sym[3:]}=X"
        return {"XAUUSD":"GC=F"}.get(sym, sym)

    def yf(self, sym: str, tf: str) -> pd.DataFrame:
        ysym = self._yf_sym(sym)
        iv   = self.YF_INTERVAL.get(tf, "1h")
        per  = self.YF_PERIOD.get(tf, "90d")
        try:
            df = yf.download(ysym, period=per, interval=iv,
                             progress=False, auto_adjust=True)
            if df.empty: return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            cols = ["Open","High","Low","Close","Volume"]
            for c in cols:
                if c not in df.columns: return pd.DataFrame()
            return df[cols].dropna().copy()
        except Exception as e:
            log.error(f"YF error {ysym}: {e}")
            return pd.DataFrame()

    def mt5_fetch(self, sym: str, tf: str, count=500) -> pd.DataFrame:
        if not HAS_MT5: raise RuntimeError("MT5 not installed")
        TF = {"1m":mt5.TIMEFRAME_M1,"5m":mt5.TIMEFRAME_M5,
              "15m":mt5.TIMEFRAME_M15,"30m":mt5.TIMEFRAME_M30,
              "1h":mt5.TIMEFRAME_H1,"4h":mt5.TIMEFRAME_H4,"1d":mt5.TIMEFRAME_D1}
        rates = mt5.copy_rates_from_pos(sym + self.cfg.mt5_suffix, TF.get(tf, mt5.TIMEFRAME_H1), 0, count)
        if rates is None or len(rates) == 0: return pd.DataFrame()
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"open":"Open","high":"High","low":"Low",
                            "close":"Close","tick_volume":"Volume"}, inplace=True)
        return df[["Open","High","Low","Close","Volume"]].copy()

    def fetch(self, sym: str, tf: str, use_mt5=False) -> pd.DataFrame:
        if use_mt5 and HAS_MT5:
            return self.mt5_fetch(sym, tf)
        return self.yf(sym, tf)

    def spread(self, sym: str) -> float:
        if not HAS_MT5: return 1.0
        tick = mt5.symbol_info_tick(sym + self.cfg.mt5_suffix)
        if not tick: return 99.0
        info = mt5.symbol_info(sym + self.cfg.mt5_suffix)
        pip  = 0.0001 if (info and info.digits >= 4) else 0.01
        return (tick.ask - tick.bid) / pip


# ─────────────────────────────────────────────
#  NEWS FILTER
# ─────────────────────────────────────────────
class News:
    def __init__(self, cfg: Config):
        self.cfg        = cfg
        self.events     = []
        self.last_fetch = None
        self._load()
        if not self.events or self._stale():
            self.refresh()

    def _load(self):
        if os.path.exists(self.cfg.news_cache):
            try:
                data = json.load(open(self.cfg.news_cache))
                self.events = data.get("events", [])
                lf = data.get("last_fetch")
                if lf: self.last_fetch = datetime.fromisoformat(lf)
                log.info(f"News cache: {len(self.events)} events loaded")
            except: pass

    def _save(self):
        try:
            json.dump({"last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
                       "events": self.events},
                      open(self.cfg.news_cache, "w"), indent=2)
        except: pass

    def _stale(self) -> bool:
        return not self.last_fetch or \
               (datetime.utcnow() - self.last_fetch).total_seconds() > 21600

    def refresh(self):
        if not self.cfg.news_enabled: return
        if self.cfg.finnhub_key and self._fetch_finnhub(): return
        self._fetch_ff()

    def _fetch_finnhub(self) -> bool:
        try:
            url = f"https://finnhub.io/api/v1/calendar/economic?token={self.cfg.finnhub_key}"
            r   = requests.get(url, timeout=15)
            if r.status_code == 429:
                log.warning("Finnhub rate limit"); return False
            if r.status_code != 200: return False
            data = r.json()
            evs  = data.get("economicCalendar", data if isinstance(data, list) else [])
            self.events = [{"title": e.get("event",""),
                            "currency": e.get("country","").upper(),
                            "time": e.get("time",""),
                            "impact": e.get("impact","low").lower()} for e in evs]
            self.last_fetch = datetime.utcnow()
            self._save()
            log.info(f"Finnhub: {len(self.events)} events")
            return True
        except Exception as e:
            log.error(f"Finnhub: {e}"); return False

    def _fetch_ff(self):
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            r   = requests.get(url, headers={
                "User-Agent":"Mozilla/5.0","Accept":"application/json",
                "Referer":"https://www.forexfactory.com/"}, timeout=15)
            if r.status_code != 200: return
            self.events = [{"title": e.get("title",""),
                            "currency": e.get("country","").upper(),
                            "time": e.get("date",""),
                            "impact": e.get("impact","low").lower()} for e in r.json()]
            self.last_fetch = datetime.utcnow()
            self._save()
            log.info(f"ForexFactory: {len(self.events)} events")
        except Exception as e:
            log.error(f"FF: {e}")

    def penalty(self, sym: str) -> float:
        if not self.cfg.news_enabled or not self.events: return 0.0
        if self._stale(): self.refresh()
        curs = [c for c in ["USD","EUR","GBP","JPY","AUD","CAD","CHF","NZD"] if c in sym]
        now  = datetime.utcnow()
        mx   = 0.0
        for ev in self.events:
            if ev["currency"] not in curs: continue
            try:
                ts   = ev["time"]
                fmt  = "%Y-%m-%d %H:%M:%S"
                try:
                    evt = datetime.fromisoformat(ts.replace("Z","+00:00"))
                    if evt.tzinfo:
                        evt = evt.replace(tzinfo=None) - timedelta(hours=evt.utcoffset().seconds//3600 if evt.utcoffset() else 0)
                except:
                    evt = datetime.strptime(ts[:19], fmt)
                if evt - timedelta(minutes=30) <= now <= evt + timedelta(minutes=60):
                    p = {"high":4.0,"medium":2.0}.get(ev["impact"],0.0)
                    if p > mx:
                        mx = p
                        log.warning(f"NEWS: {ev['currency']} {ev['impact'].upper()} — penalty -{p:.0f}")
            except: continue
        return mx


# ─────────────────────────────────────────────
#  SIGNAL ENGINE
# ─────────────────────────────────────────────
class Engine:
    def __init__(self, cfg: Config, fetcher: Fetcher, news: News):
        self.cfg     = cfg
        self.fetcher = fetcher
        self.news    = news

    def _session_ok(self) -> bool:
        h = datetime.utcnow().hour
        return any(s <= h < e for s, e in self.cfg.sessions)

    def _macro(self, sym: str, use_mt5: bool) -> Optional[str]:
        df = self.fetcher.fetch(sym, self.cfg.tf_confirm, use_mt5)
        if df.empty or len(df) < 210: return None
        df = Ind.compute(df, self.cfg)
        if df.empty: return None
        r = df.iloc[-1]
        if r["Close"] > r["e200"] and r["e50"] > r["e200"]: return "BUY"
        if r["Close"] < r["e200"] and r["e50"] < r["e200"]: return "SELL"
        return None

    def _trend(self, sym: str, use_mt5: bool) -> Optional[str]:
        df = self.fetcher.fetch(sym, self.cfg.tf_trend, use_mt5)
        if df.empty or len(df) < 60: return None
        df = Ind.compute(df, self.cfg)
        if df.empty: return None
        r = df.iloc[-1]
        if r["Close"] > r["e50"] and r["e_fast"] > r["e_slow"]: return "BUY"
        if r["Close"] < r["e50"] and r["e_fast"] < r["e_slow"]: return "SELL"
        return None

    def _sl_tp(self, d: str, entry: float, atr: float, df15: pd.DataFrame):
        sl_d = atr * self.cfg.sl_atr_mult
        if d == "BUY":
            swing = entry - df15["Low"].tail(15).min()
            sl_d  = max(sl_d, swing * 1.05)
            return entry - sl_d, entry + sl_d * self.cfg.rr_target, sl_d
        else:
            swing = df15["High"].tail(15).max() - entry
            sl_d  = max(sl_d, swing * 1.05)
            return entry + sl_d, entry - sl_d * self.cfg.rr_target, sl_d

    def _lots(self, equity: float, sl_dist: float, sym: str) -> float:
        risk    = equity * self.cfg.risk_pct
        pip     = 0.01 if "JPY" in sym else 0.0001
        sl_pips = sl_dist / pip
        pip_val = 9.0 if "JPY" in sym else 10.0
        if sl_pips <= 0: return 0.0
        lots = risk / (sl_pips * pip_val)
        lots = round(lots / self.cfg.lot_step) * self.cfg.lot_step
        return round(max(self.cfg.lot_min, min(lots, self.cfg.lot_max)), 2)

    def generate(self, sym: str, equity: float,
                 use_mt5=False) -> Optional[Signal]:

        if not self._session_ok():
            return None

        sp     = self.fetcher.spread(sym)
        sp_lim = self.cfg.spread_limit(self.cfg.tf_entry)
        if sp > sp_lim:
            log.warning(f"{sym}: spread {sp:.1f} > {sp_lim:.0f} pips limit")
            return None

        macro = self._macro(sym, use_mt5)
        trend = self._trend(sym, use_mt5)
        if trend is None:
            log.info(f"{sym}: no clear trend")
            return None
        if macro is not None and macro != trend:
            log.info(f"{sym}: macro ({macro}) conflicts with trend ({trend})")
            return None

        df15 = self.fetcher.fetch(sym, self.cfg.tf_entry, use_mt5)
        if df15.empty or len(df15) < 80:
            log.warning(f"{sym}: insufficient 15m data")
            return None
        df15 = Ind.compute(df15, self.cfg)
        if df15.empty: return None

        sc = Score()
        sc.s1, sc.r1 = S1.score(df15, self.cfg, trend)
        sc.s2, sc.r2 = S2.score(df15, self.cfg, trend)
        sc.s3, sc.r3 = S3.score(df15, self.cfg, trend)

        combo   = sc.combined(self.cfg)
        penalty = self.news.penalty(sym)
        final   = combo - penalty

        if final < self.cfg.min_score:
            log.info(f"{sym}: score {final:.2f} (tech={combo:.2f} news=-{penalty:.1f}) < {self.cfg.min_score}")
            return None

        cur   = df15.iloc[-1]
        entry = float(cur["Close"])
        atr   = float(cur["atr"])
        rsi   = float(cur["rsi"])
        mh    = float(cur["macd_h"])

        sl, tp, sl_d = self._sl_tp(trend, entry, atr, df15)
        tp_d = abs(tp - entry)
        rr   = round(tp_d / sl_d, 2) if sl_d > 0 else 0

        if rr < self.cfg.rr_min:
            log.info(f"{sym}: RR {rr:.2f} below min {self.cfg.rr_min}")
            return None

        lots = self._lots(equity, sl_d, sym)
        if lots <= 0: return None

        dom   = sc.dominant()
        lines = (
            [f"[S1-EMA]   {r}" for r in sc.r1] +
            [f"[S2-BB]    {r}" for r in sc.r2] +
            [f"[S3-MACD]  {r}" for r in sc.r3]
        )
        reason = (
            f"Trend={trend} Macro={macro} | "
            f"Score={final:.1f}/10 (tech={combo:.1f} news=-{penalty:.1f}) | "
            f"Dominant={dom.value}\n" +
            "\n".join(f"    {l}" for l in lines)
        )

        sig = Signal(
            symbol    = sym,
            direction = Dir.BUY if trend == "BUY" else Dir.SELL,
            entry     = round(entry, 5),
            sl        = round(sl, 5),
            tp        = round(tp, 5),
            sl_dist   = round(sl_d, 5),
            tp_dist   = round(tp_d, 5),
            rr        = rr,
            lots      = lots,
            confidence= round(final / 10.0, 3),
            score     = sc,
            strategy  = dom.value,
            rsi       = round(rsi, 2),
            macd_hist = round(mh, 6),
            atr       = round(atr, 5),
            spread    = round(sp, 1),
            reason    = reason,
        )

        log.info(
            f"SIGNAL > {sig.direction.value} {sym} | "
            f"Entry={sig.entry} SL={sig.sl} TP={sig.tp} | "
            f"RR={sig.rr} Score={final:.1f}/10"
        )
        return sig


# ─────────────────────────────────────────────
#  OPTIONAL ML FILTER
# ─────────────────────────────────────────────
class ML:
    FEATS = ["e_fast","e_slow","e50","rsi","macd","macd_h","bb_pos","bb_w",
             "sk","sd","atr","adx","pdi","ndi","vr","ret","brat","ush","lsh"]

    def __init__(self):
        self.models  = {}
        self.scalers = {}

    def train(self, sym: str, df: pd.DataFrame) -> float:
        if not HAS_SKL: return 0.0
        df = df.copy()
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        df.dropna(inplace=True)
        av = [c for c in self.FEATS if c in df.columns]
        X, y = df[av].values, df["target"].values
        if len(X) < 200: return 0.0
        n = int(len(X) * 0.8)
        sc = RobustScaler()
        Xtr = sc.fit_transform(X[:n])
        Xte = sc.transform(X[n:])
        m = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                        learning_rate=0.05, random_state=42)
        m.fit(Xtr, y[:n])
        acc = accuracy_score(y[n:], m.predict(Xte))
        self.models[sym]  = m
        self.scalers[sym] = sc
        log.info(f"ML {sym}: acc={acc:.3f}")
        return acc

    def predict(self, sym: str, df: pd.DataFrame, d: str) -> float:
        if not HAS_SKL or sym not in self.models: return 0.5
        av = [c for c in self.FEATS if c in df.columns]
        X  = self.scalers[sym].transform(df[av].iloc[-1:].values)
        p  = self.models[sym].predict_proba(X)[0]
        return float(p[1]) if d == "BUY" else float(p[0])


# ─────────────────────────────────────────────
#  EXECUTOR  (paper + live)
# ─────────────────────────────────────────────
class Executor:
    def __init__(self, cfg: Config, fetcher: Fetcher):
        self.cfg     = cfg
        self.fetcher = fetcher

    def _sym(self, s: str) -> str:
        return s + self.cfg.mt5_suffix

    def _fill_mode(self, sym: str) -> int:
        info = mt5.symbol_info(sym)
        if not info: return mt5.ORDER_FILLING_IOC
        fm  = getattr(info, "filling_mode", 0)
        fok = getattr(mt5, "SYMBOL_FILLING_FOK", 1)
        ioc = getattr(mt5, "SYMBOL_FILLING_IOC", 2)
        if fm & fok: return mt5.ORDER_FILLING_FOK
        if fm & ioc: return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    def _normalize_volume(self, volume: float, info: Any) -> float:
        step = float(getattr(info, "volume_step", self.cfg.lot_step) or self.cfg.lot_step)
        vmin = float(getattr(info, "volume_min", self.cfg.lot_min) or self.cfg.lot_min)
        vmax = float(getattr(info, "volume_max", self.cfg.lot_max) or self.cfg.lot_max)
        if step <= 0:
            step = self.cfg.lot_step
        volume = max(vmin, min(volume, vmax))
        volume = round(round(volume / step) * step, 2)
        return max(vmin, min(volume, vmax))

    def _retcode_text(self, result: Any) -> str:
        code = getattr(result, "retcode", None)
        comment = getattr(result, "comment", "")
        return f"retcode={code} comment={comment}"

    def positions(self) -> List[dict]:
        if not HAS_MT5: return []
        ps = mt5.positions_get()
        if ps is None: return []
        return [{"ticket":p.ticket,"symbol":p.symbol,
                 "type":"BUY" if p.type==0 else "SELL",
                 "volume":p.volume,"price_open":p.price_open,
                 "price_cur":p.price_current,"sl":p.sl,"tp":p.tp,
                 "profit":p.profit,"magic":p.magic} for p in ps]

    def has_position(self, sym: str) -> bool:
        mt5s = self._sym(sym)
        return any(p["symbol"]==mt5s and p["magic"]==self.cfg.mt5_magic
                   for p in self.positions())

    def live(self, sig: Signal) -> bool:
        if not HAS_MT5: return False
        mt5s = self._sym(sig.symbol)
        if not mt5.symbol_select(mt5s, True):
            log.error(f"Cannot select {mt5s}"); return False
        info = mt5.symbol_info(mt5s)
        if not info:
            log.error(f"No symbol info for {mt5s}")
            return False
        trade_mode = getattr(info, "trade_mode", None)
        full_mode = getattr(mt5, "SYMBOL_TRADE_MODE_FULL", 4)
        if trade_mode is not None and trade_mode != full_mode:
            log.error(f"{mt5s} trading not allowed by broker (trade_mode={trade_mode})")
            return False
        tick = mt5.symbol_info_tick(mt5s)
        if not tick: return False
        ot, price = (mt5.ORDER_TYPE_BUY, tick.ask) if sig.direction == Dir.BUY \
                 else (mt5.ORDER_TYPE_SELL, tick.bid)
        digits = int(getattr(info, "digits", 5) or 5)
        point = float(getattr(info, "point", 0.0) or (0.01 if "JPY" in sig.symbol else 0.0001))
        volume = self._normalize_volume(sig.lots, info)
        price = round(float(price), digits)
        sl = round(float(sig.sl), digits)
        tp = round(float(sig.tp), digits)
        stops_level = int(getattr(info, "trade_stops_level", 0) or 0)
        if stops_level > 0:
            min_dist = stops_level * point
            if abs(price - sl) < min_dist or abs(tp - price) < min_dist:
                log.error(f"{mt5s} stops too close for broker: min_dist={min_dist:.5f}")
                return False
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": mt5s,
               "volume": float(volume), "type": ot,
               "price": price, "sl": sl, "tp": tp,
               "magic": self.cfg.mt5_magic, "comment": self.cfg.mt5_comment,
               "deviation": 20,
               "type_time": mt5.ORDER_TIME_GTC,
               "type_filling": self._fill_mode(mt5s)}
        check = mt5.order_check(req)
        if check and getattr(check, "retcode", None) not in (None, mt5.TRADE_RETCODE_DONE):
            log.error(f"MT5 order check failed for {mt5s}: {self._retcode_text(check)}")
            return False
        res = mt5.order_send(req)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"MT5 ORDER PLACED: {sig.direction.value} {mt5s} @ {price:.5f}")
            return True
        if res:
            log.error(f"MT5 order failed for {mt5s}: {self._retcode_text(res)} last_error={mt5.last_error()}")
        else:
            log.error(f"MT5 order failed for {mt5s}: last_error={mt5.last_error()}")
        return False

    def paper(self, sig: Signal) -> bool:
        log.info(
            f"[PAPER] {sig.direction.value} {sig.symbol} "
            f"{sig.lots}L @ {sig.entry:.5f} | "
            f"SL={sig.sl:.5f}  TP={sig.tp:.5f} | "
            f"RR={sig.rr}  Score={sig.confidence*10:.1f}/10"
        )
        print(f"\n  {'▲' if sig.direction==Dir.BUY else '▼'}  {sig.direction.value} {sig.symbol}")
        print(f"     Entry : {sig.entry:.5f}")
        print(f"     SL    : {sig.sl:.5f}  ({round(sig.sl_dist/(0.01 if 'JPY' in sig.symbol else 0.0001))} pips)")
        print(f"     TP    : {sig.tp:.5f}  ({round(sig.tp_dist/(0.01 if 'JPY' in sig.symbol else 0.0001))} pips)")
        print(f"     RR    : 1:{sig.rr}    Score: {sig.confidence*10:.1f}/10")
        print(f"     Why   : {sig.score.r1[:1] + sig.score.r2[:1] + sig.score.r3[:1]}")
        return True

    def execute(self, sig: Signal) -> bool:
        if self.cfg.live_trading and HAS_MT5:
            return self.live(sig)
        return self.paper(sig)

    def trail_sl(self):
        if not HAS_MT5: return
        for p in self.positions():
            if p["magic"] != self.cfg.mt5_magic: continue
            df = self.fetcher.mt5_fetch(p["symbol"].replace(self.cfg.mt5_suffix,""), "15m")
            if df.empty: continue
            atr  = float(Ind.compute(df, self.cfg)["atr"].iloc[-1])
            dist = atr * 1.0
            nsl  = None
            if p["type"] == "BUY":
                nsl = p["price_cur"] - dist
                if nsl <= p["sl"] or nsl >= p["price_cur"]: nsl = None
            else:
                nsl = p["price_cur"] + dist
                if nsl >= p["sl"] or nsl <= p["price_cur"]: nsl = None
            if nsl:
                r = mt5.order_send({"action":mt5.TRADE_ACTION_SLTP,
                                    "symbol":p["symbol"],"sl":nsl,"tp":p["tp"],
                                    "position":p["ticket"]})
                if r and r.retcode == mt5.TRADE_RETCODE_DONE:
                    log.info(f"Trail SL → {nsl:.5f} (ticket {p['ticket']})")


# ─────────────────────────────────────────────
#  RISK MANAGER
# ─────────────────────────────────────────────
class Risk:
    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self.day_equity   = 0.0
        self.today        = datetime.utcnow().date()
        self.consec_loss  = 0
        self.paused       = False
        self.pause_reason = ""

    def new_day(self, equity: float):
        today = datetime.utcnow().date()
        if today != self.today:
            log.info(f"New trading day. Equity {self.day_equity:.2f} → {equity:.2f}")
            self.day_equity  = equity
            self.today       = today
            self.paused      = False
            self.consec_loss = 0
        elif self.day_equity == 0:
            self.day_equity = equity

    def record_result(self, won: bool):
        if won:
            self.consec_loss = 0
        else:
            self.consec_loss += 1
            if self.consec_loss >= self.cfg.consec_loss_limit:
                self.paused       = True
                self.pause_reason = f"{self.consec_loss} consecutive losses — stopping today"
                log.warning(f"⛔ PAUSED: {self.pause_reason}")

    def can_trade(self, equity: float, n_open: int) -> Tuple[bool, str]:
        if self.paused:
            return False, self.pause_reason
        if self.day_equity > 0:
            dd = (equity - self.day_equity) / self.day_equity
            if dd <= -self.cfg.daily_loss_limit:
                self.paused       = True
                self.pause_reason = f"Daily loss {dd:.1%} hit {self.cfg.daily_loss_limit:.0%} limit"
                log.warning(f"⛔ PAUSED: {self.pause_reason}")
                return False, self.pause_reason
        if n_open >= self.cfg.max_positions:
            return False, f"Max {self.cfg.max_positions} positions open"
        return True, ""


# ─────────────────────────────────────────────
#  TRADE LOGGER
# ─────────────────────────────────────────────
class Logger:
    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.trades = []
        self._load()

    def _load(self):
        if os.path.exists(self.cfg.log_file):
            try:
                self.trades = json.load(open(self.cfg.log_file))
                log.info(f"Loaded {len(self.trades)} trade records")
            except: pass

    def save(self):
        tmp = self.cfg.log_file + ".tmp"
        json.dump(self.trades, open(tmp, "w"), indent=2)
        os.replace(tmp, self.cfg.log_file)

    def log_signal(self, sig: Signal):
        entry = sig.to_dict()
        entry["type"] = "signal"
        self.trades.append(entry)
        self.save()

    def stats(self) -> dict:
        closed = [t for t in self.trades if t.get("type") == "result"]
        if not closed: return {"total": 0}
        pnls   = [t["pnl_usd"] for t in closed]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gw     = sum(wins) or 0
        gl     = abs(sum(losses)) or 0.001
        cum    = np.cumsum(pnls)
        pk     = np.maximum.accumulate(cum)
        dd     = float(np.max(pk - cum)) if len(cum) else 0
        by_s   = {}
        for t in closed:
            s = t.get("strategy","?")
            if s not in by_s: by_s[s] = {"wins":0,"losses":0,"pnl":0.0}
            if t["pnl_usd"] > 0: by_s[s]["wins"] += 1
            else: by_s[s]["losses"] += 1
            by_s[s]["pnl"] += t["pnl_usd"]
        return {
            "total":          len(closed),
            "wins":           len(wins),
            "losses":         len(losses),
            "win_rate_%":     round(len(wins)/len(pnls)*100, 1),
            "total_pnl_usd":  round(sum(pnls), 2),
            "avg_win_usd":    round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss_usd":   round(float(np.mean(losses)), 2) if losses else 0,
            "profit_factor":  round(gw/gl, 2),
            "expectancy_usd": round(float(np.mean(pnls)), 2),
            "max_drawdown":   round(dd, 2),
            "by_strategy":    by_s,
        }


# ─────────────────────────────────────────────
#  BACKTESTER
# ─────────────────────────────────────────────
class Backtest:
    def __init__(self, cfg: Config):
        self.cfg     = cfg
        self.fetcher = Fetcher(cfg)

    def run(self, sym: str, capital=10_000.0) -> dict:
        log.info(f"Backtesting {sym} | Capital=${capital:,.0f}")
        df = self.fetcher.yf(sym, "1d")
        if df.empty or len(df) < 150:
            log.error(f"Not enough data for {sym}")
            return {}
        df = Ind.compute(df, self.cfg)
        log.info(f"{sym}: {len(df)} daily bars")
        cap, pos, trades, equity = capital, None, [], [capital]

        for i in range(100, len(df)-1):
            bar    = df.iloc[i]
            window = df.iloc[max(0, i-100):i+1].copy()

            if pos:
                hi, lo    = bar["High"], bar["Low"]
                closed, outcome, xp = False, "", 0.0
                if pos["side"] == "BUY":
                    if lo <= pos["sl"]:   xp, outcome, closed = pos["sl"],  "LOSS", True
                    elif hi >= pos["tp"]: xp, outcome, closed = pos["tp"],  "WIN",  True
                else:
                    if hi >= pos["sl"]:   xp, outcome, closed = pos["sl"],  "LOSS", True
                    elif lo <= pos["tp"]: xp, outcome, closed = pos["tp"],  "WIN",  True
                if closed:
                    pip  = 0.01 if "JPY" in sym else 0.0001
                    pp   = (xp - pos["entry"]) if pos["side"]=="BUY" else (pos["entry"] - xp)
                    pips = pp / pip
                    usd  = pips * 10.0 * pos["lots"]
                    cap += usd
                    trades.append({"side":pos["side"],"strategy":pos["strat"],
                                   "pnl_pips":round(pips,1),"pnl_usd":round(usd,2),"outcome":outcome})
                    pos = None

            if pos is None:
                d, sc, strat = self._signal(window)
                if d and sc.combined(self.cfg) >= self.cfg.min_score:
                    entry = float(bar["Close"])
                    atr   = float(bar["atr"])
                    sl_d  = atr * self.cfg.sl_atr_mult
                    sl = entry - sl_d if d=="BUY" else entry + sl_d
                    tp = entry + sl_d*self.cfg.rr_target if d=="BUY" else entry - sl_d*self.cfg.rr_target
                    pip  = 0.01 if "JPY" in sym else 0.0001
                    lots = max(self.cfg.lot_min, min(
                        round((cap * self.cfg.risk_pct) / (sl_d/pip*10), 2), self.cfg.lot_max))
                    pos = {"side":d,"entry":entry,"sl":sl,"tp":tp,"lots":lots,"strat":strat}
            equity.append(cap)

        return self._stats(trades, capital, cap, equity, sym)

    def _signal(self, w: pd.DataFrame):
        if len(w) < 60: return None, Score(), ""
        r = w.iloc[-1]
        if r["Close"] > r["e50"] and r["e_fast"] > r["e_slow"]: d = "BUY"
        elif r["Close"] < r["e50"] and r["e_fast"] < r["e_slow"]: d = "SELL"
        else: return None, Score(), ""
        sc = Score()
        sc.s1, sc.r1 = S1.score(w, self.cfg, d)
        sc.s2, sc.r2 = S2.score(w, self.cfg, d)
        sc.s3, sc.r3 = S3.score(w, self.cfg, d)
        return d, sc, sc.dominant().value

    def _stats(self, trades, init, final, equity, sym) -> dict:
        if not trades: return {"total":0,"note":"No trades"}
        pnls   = [t["pnl_usd"] for t in trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        eq     = np.array(equity)
        pk     = np.maximum.accumulate(eq)
        dd     = float(np.max((pk-eq)/pk.clip(min=1e-9)))*100
        gw     = sum(wins) or 0
        gl     = abs(sum(losses)) or 0.001
        by_s   = {}
        for t in trades:
            s = t.get("strategy","?")
            if s not in by_s: by_s[s] = {"trades":0,"wins":0,"pnl":0}
            by_s[s]["trades"] += 1
            by_s[s]["pnl"]    += t["pnl_usd"]
            if t["pnl_usd"] > 0: by_s[s]["wins"] += 1
        st = {
            "symbol":           sym,
            "initial_capital":  round(init, 2),
            "final_capital":    round(final, 2),
            "return_%":         round((final-init)/init*100, 2),
            "total_trades":     len(trades),
            "wins":             len(wins),
            "losses":           len(losses),
            "win_rate_%":       round(len(wins)/len(trades)*100, 1),
            "profit_factor":    round(gw/gl, 2),
            "avg_win_usd":      round(float(np.mean(wins)), 2) if wins else 0,
            "avg_loss_usd":     round(float(np.mean(losses)), 2) if losses else 0,
            "best_trade_usd":   round(max(pnls), 2),
            "worst_trade_usd":  round(min(pnls), 2),
            "max_drawdown_%":   round(dd, 2),
            "expectancy_usd":   round(float(np.mean(pnls)), 2),
            "by_strategy":      by_s,
        }
        log.info("="*55)
        log.info(f"  BACKTEST: {sym}")
        log.info("="*55)
        for k, v in st.items():
            if k == "by_strategy":
                log.info("  Strategy breakdown:")
                for s2, sd in v.items():
                    wr = round(sd["wins"]/sd["trades"]*100,1) if sd["trades"] else 0
                    log.info(f"    {s2:<35} trades={sd['trades']} win%={wr} pnl=${sd['pnl']:.2f}")
            else:
                log.info(f"  {k:<30}: {v}")
        log.info("="*55)
        return st


# ─────────────────────────────────────────────
#  MT5 CONNECTION MANAGER
# ─────────────────────────────────────────────
class MT5Mgr:
    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.connected = False

    def connect(self) -> bool:
        if not HAS_MT5:
            log.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
            return False
        login    = int(os.environ.get("MT5_LOGIN", "0"))
        password = os.environ.get("MT5_PASSWORD", "")
        server   = os.environ.get("MT5_SERVER", "")
        if login == 0:
            log.error("MT5_LOGIN env var not set")
            return False
        if not mt5.initialize():
            log.error(f"mt5.initialize() failed: {mt5.last_error()}")
            return False
        if not mt5.login(login=login, password=password, server=server):
            log.error(f"MT5 login failed: {mt5.last_error()}")
            mt5.shutdown()
            return False
        acc = mt5.account_info()
        log.info(f"MT5 connected | Balance={acc.balance:.2f} Equity={acc.equity:.2f} Server={server}")
        self.connected = True
        return True

    def ensure(self) -> bool:
        if not HAS_MT5: return False
        if mt5.terminal_info() is None: return self.connect()
        return True

    def equity(self) -> float:
        if not HAS_MT5 or not self.ensure(): return 10_000.0
        acc = mt5.account_info()
        return float(acc.equity) if acc else 10_000.0

    def disconnect(self):
        if HAS_MT5: mt5.shutdown()
        self.connected = False


# ─────────────────────────────────────────────
#  MAIN BOT
# ─────────────────────────────────────────────
class Bot:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg      = cfg or Config()
        self.fetcher  = Fetcher(self.cfg)
        self.news     = News(self.cfg)
        self.engine   = Engine(self.cfg, self.fetcher, self.news)
        self.ml       = ML()
        self.executor = Executor(self.cfg, self.fetcher)
        self.risk     = Risk(self.cfg)
        self.logger   = Logger(self.cfg)
        self.mt5      = MT5Mgr(self.cfg)

        mode = "LIVE (MT5)" if self.cfg.live_trading else "PAPER (no real money)"
        print("\n" + "="*60)
        print(f"  GOLDEN BOT v5  |  Mode: {mode}")
        print(f"  Symbols : {', '.join(self.cfg.symbols)}")
        print(f"  Min score: {self.cfg.min_score}/10   RR: {self.cfg.rr_target}R")
        print(f"  Risk/trade: {self.cfg.risk_pct:.0%}   Max positions: {self.cfg.max_positions}")
        print("="*60 + "\n")

    def setup(self):
        if self.cfg.live_trading:
            if not self.mt5.connect():
                log.warning("MT5 connect failed → switching to paper mode")
                self.cfg.live_trading = False
        if self.cfg.use_ml and HAS_SKL:
            log.info("Training ML filters…")
            for sym in self.cfg.symbols:
                try:
                    df = self.fetcher.yf(sym, "1d")
                    if not df.empty:
                        df = Ind.compute(df, self.cfg)
                        self.ml.train(sym, df)
                except Exception as e:
                    log.error(f"ML train {sym}: {e}")

    def process(self, sym: str, equity: float, n_open: int) -> Optional[Signal]:
        use_mt5 = self.cfg.live_trading and HAS_MT5
        ok, reason = self.risk.can_trade(equity, n_open)
        if not ok:
            log.debug(f"{sym}: {reason}")
            return None
        if use_mt5 and self.executor.has_position(sym):
            return None
        sig = self.engine.generate(sym, equity, use_mt5)
        if sig is None: return None
        if self.cfg.use_ml and sym in self.ml.models:
            df15 = self.fetcher.fetch(sym, self.cfg.tf_entry, use_mt5)
            if not df15.empty:
                df15 = Ind.compute(df15, self.cfg)
                prob = self.ml.predict(sym, df15, sig.direction.value)
                sig.ml_prob = prob
                if prob < self.cfg.ml_min_prob:
                    log.info(f"{sym}: ML prob {prob:.2f} too low")
                    return None
        self.logger.log_signal(sig)
        self.executor.execute(sig)
        return sig

    def run(self):
        self.setup()
        loop = 0
        log.info("Trading loop started. Press Ctrl+C to stop.")
        while True:
            try:
                loop += 1
                log.info(f"\n─── Loop #{loop} | {datetime.utcnow():%Y-%m-%d %H:%M UTC} ───")
                if self.cfg.live_trading: self.mt5.ensure()
                eq = self.mt5.equity()
                self.risk.new_day(eq)
                ok, reason = self.risk.can_trade(eq, 0)
                if not ok:
                    log.warning(f"⛔ {reason}")
                    time.sleep(self.cfg.interval)
                    continue
                positions = self.executor.positions()
                n_open    = len([p for p in positions if p.get("magic")==self.cfg.mt5_magic])
                log.info(f"Equity=${eq:.2f} | Open={n_open}/{self.cfg.max_positions}")
                if self.cfg.live_trading: self.executor.trail_sl()
                for sym in self.cfg.symbols:
                    try:
                        self.process(sym, eq, n_open)
                    except Exception as e:
                        log.error(f"Error {sym}: {e}")
                st = self.logger.stats()
                if st.get("total", 0) > 0:
                    log.info(
                        f"Stats | Trades={st['total']}  "
                        f"WR={st.get('win_rate_%',0)}%  "
                        f"PnL=${st.get('total_pnl_usd',0):.2f}  "
                        f"PF={st.get('profit_factor',0):.2f}"
                    )
                log.info(f"Sleeping {self.cfg.interval}s…")
                time.sleep(self.cfg.interval)
            except KeyboardInterrupt:
                log.info("Stopped by user.")
                break
            except Exception as e:
                log.error(f"Loop error: {e}\n{traceback.format_exc()}")
                time.sleep(30)
        self.logger.save()
        if self.cfg.live_trading: self.mt5.disconnect()

    def signals_now(self) -> List[dict]:
        eq = self.mt5.equity()
        out = []
        for sym in self.cfg.symbols:
            sig = self.engine.generate(sym, eq, False)
            out.append(sig.to_dict() if sig else {"symbol":sym,"signal":"HOLD"})
        return out

    def backtest_all(self, capital=10_000.0) -> dict:
        bt = Backtest(self.cfg)
        return {sym: bt.run(sym, capital) for sym in self.cfg.symbols}


# ─────────────────────────────────────────────
#  CONFIG LOADER (reads 'config' file if exists)
# ─────────────────────────────────────────────
def load_env_config():
    if not os.path.exists("config"): return
    try:
        for line in open("config"):
            if "=" in line:
                k, v = line.strip().split("=", 1)
                k = k.strip().upper()
                v = v.strip().strip('"').strip("'")
                if k == "LOGIN":    os.environ["MT5_LOGIN"]    = v
                if k == "PASSWORD": os.environ["MT5_PASSWORD"] = v
                if k == "SERVER":   os.environ["MT5_SERVER"]   = v
    except Exception as e:
        log.error(f"Config file error: {e}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
def main():
    load_env_config()

    cfg = Config(
        symbols     = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"],
        mt5_suffix  = "",          # change to "m" if your broker adds suffix

        tf_entry    = "15m",
        tf_trend    = "1h",
        tf_confirm  = "4h",

        # ── Tighter signal quality ───────────
        min_score   = 6.0,         # was 5.5 — now stricter
        w1=0.35, w2=0.30, w3=0.35,

        # ── Risk: very conservative ──────────
        risk_pct          = 0.01,  # 1% per trade
        sl_atr_mult       = 2.0,
        rr_target         = 2.5,
        rr_min            = 2.0,
        max_positions     = 3,
        daily_loss_limit  = 0.03,  # 3% daily stop
        consec_loss_limit = 3,     # pause after 3 losses in a row
        lot_min           = 0.01,
        lot_max           = 0.10,

        # ── Sessions ─────────────────────────
        sessions    = [(7, 16), (13, 22)],   # London + NY sessions

        # ── News filter ──────────────────────
        news_enabled = True,
        finnhub_key  = "",    # get free key at https://finnhub.io

        # ── ML (off by default) ──────────────
        use_ml       = False,
        ml_min_prob  = 0.65,

        # ────────────────────────────────────
        #  ⚠️  LIVE TRADING OFF BY DEFAULT
        #  To enable: set live_trading=True AND set these env vars:
        #    export MT5_LOGIN=<your_account_number>
        #    export MT5_PASSWORD=<your_password>
        #    export MT5_SERVER=<your_broker_server>
        #
        #  Only enable after 2+ weeks of paper results show consistent profit.
        # ────────────────────────────────────
        live_trading = False,

        interval     = 60,
    )

    args = sys.argv[1:]

    # ── --live flag overrides config ────────
    if "--live" in args:
        print("\n" + "!"*60)
        print("  WARNING: You are switching to LIVE trading mode.")
        print("  Real money will be used. Losses are real.")
        print("  Only continue if you have tested this in paper mode")
        print("  for at least 2 weeks with positive results.")
        print("!"*60)
        confirm = input("\n  Type 'YES I UNDERSTAND' to continue: ").strip()
        if confirm != "YES I UNDERSTAND":
            print("  Cancelled. Running in paper mode instead.")
        else:
            cfg.live_trading = True

    # ── --paper flag (explicit paper mode) ──
    if "--paper" in args:
        cfg.live_trading = False

    # ── Run modes ───────────────────────────
    if "--backtest" in args:
        print("\n" + "="*60)
        print("  BACKTEST MODE")
        print("="*60)
        bot = Bot(cfg)
        results = bot.backtest_all(capital=10_000.0)
        for sym, res in results.items():
            print(f"\n{'─'*50}")
            print(f"  {sym}")
            print(f"{'─'*50}")
            for k, v in res.items():
                if k == "by_strategy":
                    print("  By strategy:")
                    for s, sd in v.items():
                        wr = round(sd["wins"]/sd["trades"]*100,1) if sd["trades"] else 0
                        print(f"    {s:<35} trades={sd['trades']} win%={wr} pnl=${sd['pnl']:.2f}")
                else:
                    print(f"  {k:<28}: {v}")

    elif "--signal" in args:
        print("\n" + "="*60)
        print("  CURRENT SIGNALS  (scan once and exit)")
        print("="*60)
        bot     = Bot(cfg)
        signals = bot.signals_now()
        found   = False
        for sig in signals:
            sym = sig.get("symbol","?")
            d   = sig.get("signal","HOLD")
            if hasattr(d, "value"): d = d.value
            if d in ("BUY","SELL"):
                found = True
                arrow = "▲" if d == "BUY" else "▼"
                print(f"\n  {arrow} {d} {sym}")
                print(f"    Entry  : {sig.get('entry')}")
                print(f"    SL     : {sig.get('sl')}")
                print(f"    TP     : {sig.get('tp')}")
                print(f"    RR     : 1:{sig.get('rr')}")
                print(f"    Score  : {sig.get('score_combined',0):.1f}/10")
            else:
                print(f"  ─  HOLD {sym}")
        if not found:
            print("\n  No high-quality signals right now. Wait for better setups.")

    elif "--stats" in args:
        bot = Bot(cfg)
        st  = bot.logger.stats()
        print("\n" + "="*60)
        print("  PERFORMANCE STATS")
        print("="*60)
        for k, v in st.items():
            if k == "by_strategy":
                print("  By strategy:")
                for s, sd in v.items(): print(f"    {s}: {sd}")
            else:
                print(f"  {k:<28}: {v}")

    else:
        # Default: paper trading loop
        cfg.live_trading = False   # safety: paper unless --live explicitly confirmed
        print("\n  No mode specified. Running in PAPER mode (safe).")
        print("  Available modes:")
        print("    python golden_bot_v5.py --paper      ← safe, paper trading loop")
        print("    python golden_bot_v5.py --signal     ← scan once, print signals")
        print("    python golden_bot_v5.py --backtest   ← historical test")
        print("    python golden_bot_v5.py --stats      ← show trade journal")
        print("    python golden_bot_v5.py --live       ← REAL MONEY (MT5 required)\n")
        bot = Bot(cfg)
        bot.run()


if __name__ == "__main__":
    main()