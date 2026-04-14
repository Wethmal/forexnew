import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
from datetime import datetime, timedelta
import os

# ============================================================================
# CONFIGURATION (Mirrored from bot.py)
# ============================================================================

class Config:
    SYMBOL = "EURUSDm"
    TIMEFRAME = mt5.TIMEFRAME_M15
    TIMEFRAME_NAME = "M15"
    H1_TIMEFRAME = mt5.TIMEFRAME_H1
    LOT_SIZE = 0.01

    # Session filter (MT5 server hour)
    START_HOUR = 8
    END_HOUR = 17

    EMA_FAST = 50
    EMA_SLOW = 200
    RSI_PERIOD = 14
    RSI_BUY_MIN = 50
    RSI_BUY_MAX = 65
    RSI_SELL_MIN = 35
    RSI_SELL_MAX = 50

    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    BB_PERIOD = 20
    BB_STD_DEV = 2
    BB_MIN_WIDTH = 0.0005

    ATR_PERIOD = 14
    ADX_PERIOD = 14
    ADX_THRESHOLD = 25

    # Risk-to-Reward 1:2
    SL_MULTIPLIER = 1.0
    TP_MULTIPLIER = 2.0

    # Break-even and trailing stop
    BE_ACTIVATION_ATR = 1.0
    TRAILING_STOP_ATR = 1.5

    LOGIN = 413646889
    PASSWORD = "Anoma@0822"
    SERVER = "Exness-MT5Trial6"

    MIN_CANDLES_REQUIRED = 200
    BACKTEST_CANDLES = 2000       # Number of M15 candles to backtest
    H1_BUFFER_CANDLES = 250       # Extra H1 candles to ensure EMA200 coverage
    H1_LOOKBACK_HOURS = 25        # Max hours to look back when aligning M15→H1 EMA200

# ============================================================================
# INDICATOR CALCULATOR (Backtesting — M15 + H1)
# ============================================================================

class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_all_indicators()

    def _calculate_all_indicators(self):
        self.df['EMA_50'] = ta.ema(self.df['close'], length=Config.EMA_FAST)
        self.df['EMA_200'] = ta.ema(self.df['close'], length=Config.EMA_SLOW)
        self.df['RSI'] = ta.rsi(self.df['close'], length=Config.RSI_PERIOD)

        macd_result = ta.macd(self.df['close'], fast=Config.MACD_FAST,
                              slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
        if macd_result is not None and not macd_result.empty:
            macd_col = [c for c in macd_result.columns
                        if c.startswith('MACD_') and 's' not in c and 'h' not in c]
            signal_col = [c for c in macd_result.columns if c.startswith('MACDs_')]
            self.df['MACD'] = macd_result[macd_col[0]] if macd_col else np.nan
            self.df['MACD_Signal'] = macd_result[signal_col[0]] if signal_col else np.nan
        else:
            self.df['MACD'] = np.nan
            self.df['MACD_Signal'] = np.nan

        bb_result = ta.bbands(self.df['close'], length=Config.BB_PERIOD, std=Config.BB_STD_DEV)
        if bb_result is not None and not bb_result.empty:
            upper_col = [c for c in bb_result.columns if c.startswith('BBU_')]
            lower_col = [c for c in bb_result.columns if c.startswith('BBL_')]
            self.df['BB_Upper'] = bb_result[upper_col[0]] if upper_col else np.nan
            self.df['BB_Lower'] = bb_result[lower_col[0]] if lower_col else np.nan
        else:
            self.df['BB_Upper'] = np.nan
            self.df['BB_Lower'] = np.nan

        self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'],
                                length=Config.ATR_PERIOD)

        adx_res = ta.adx(self.df['high'], self.df['low'], self.df['close'],
                         length=Config.ADX_PERIOD)
        if adx_res is not None and not adx_res.empty:
            adx_col = [c for c in adx_res.columns if c.startswith('ADX_')]
            self.df['ADX'] = adx_res[adx_col[0]] if adx_col else np.nan
        else:
            self.df['ADX'] = np.nan


# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    @staticmethod
    def generate_signal(row, h1_ema200: float) -> str:
        """Generate BUY/SELL/HOLD using the full upgraded filter stack."""
        close = row['close']

        # 1. Session filter
        hour = row['time'].hour
        if not (Config.START_HOUR <= hour < Config.END_HOUR):
            return 'HOLD'

        # 2. Trend direction (M15 EMA 50/200)
        if close > row['EMA_50'] > row['EMA_200']:
            trend_dir = 'UP'
        elif close < row['EMA_50'] < row['EMA_200']:
            trend_dir = 'DOWN'
        else:
            return 'HOLD'

        # 3. H1 MTF filter (EMA 200)
        if not pd.isna(h1_ema200):
            if trend_dir == 'UP' and close <= h1_ema200:
                return 'HOLD'
            if trend_dir == 'DOWN' and close >= h1_ema200:
                return 'HOLD'

        # 4. Strict RSI filter
        rsi = row['RSI']
        if pd.isna(rsi):
            return 'HOLD'
        if trend_dir == 'UP' and not (Config.RSI_BUY_MIN < rsi < Config.RSI_BUY_MAX):
            return 'HOLD'
        if trend_dir == 'DOWN' and not (Config.RSI_SELL_MIN < rsi < Config.RSI_SELL_MAX):
            return 'HOLD'

        # 5. MACD zero-line confirmation
        macd = row['MACD']
        macd_signal = row['MACD_Signal']
        if pd.isna(macd) or pd.isna(macd_signal):
            return 'HOLD'
        if trend_dir == 'UP' and not (macd > macd_signal and macd > 0):
            return 'HOLD'
        if trend_dir == 'DOWN' and not (macd < macd_signal and macd < 0):
            return 'HOLD'

        # 6. ADX trend strength
        adx = row['ADX']
        if pd.isna(adx) or adx <= Config.ADX_THRESHOLD:
            return 'HOLD'

        # 7. Bollinger Bands width
        if pd.isna(row['BB_Upper']) or pd.isna(row['BB_Lower']):
            return 'HOLD'
        if (row['BB_Upper'] - row['BB_Lower']) < Config.BB_MIN_WIDTH:
            return 'HOLD'

        # 8. ATR validity
        if pd.isna(row['ATR']) or row['ATR'] <= 0:
            return 'HOLD'

        return 'BUY' if trend_dir == 'UP' else 'SELL'


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def _build_h1_ema200_map(df_h1: pd.DataFrame) -> dict:
    """Build a time→H1 EMA200 lookup aligned to M15 candle times."""
    df_h1 = df_h1.copy()
    df_h1['EMA_200'] = ta.ema(df_h1['close'], length=200)
    # Key: H1 candle open time → EMA200 value
    return dict(zip(df_h1['time'], df_h1['EMA_200']))


def _lookup_h1_ema200(h1_map: dict, candle_time: pd.Timestamp) -> float:
    """Return the H1 EMA200 value that was valid at the given M15 candle time."""
    # Find the most recent H1 candle whose open time <= candle_time
    h1_floor = candle_time.floor('h')
    # Walk back at most 24 hours to find a valid value
    for delta in range(0, Config.H1_LOOKBACK_HOURS + 1):
        key = h1_floor - pd.Timedelta(hours=delta)
        val = h1_map.get(key, np.nan)
        if not pd.isna(val):
            return val
    return np.nan


def run_backtest():
    print("Starting Backtest (Upgraded Strategy)...")

    if not mt5.initialize():
        print("MT5 Initialization failed")
        return

    if not mt5.login(Config.LOGIN, Config.PASSWORD, Config.SERVER):
        print("Failed to login to MT5 account")
        print(f"Error: {mt5.last_error()}")
        mt5.shutdown()
        return

    print(f"Logged into {Config.SERVER}")

    # Fetch M15 candles
    rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, Config.BACKTEST_CANDLES)
    if rates is None:
        print(f"Failed to fetch M15 data for {Config.SYMBOL}")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Fetch H1 candles for MTF filter (need enough history to cover M15 window)
    h1_count = Config.BACKTEST_CANDLES // 4 + Config.H1_BUFFER_CANDLES
    rates_h1 = mt5.copy_rates_from_pos(Config.SYMBOL, Config.H1_TIMEFRAME, 0, h1_count)
    h1_map = {}
    if rates_h1 is not None:
        df_h1 = pd.DataFrame(rates_h1)
        df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
        h1_map = _build_h1_ema200_map(df_h1)

    # Calculate M15 indicators
    calc = IndicatorCalculator(df)
    df = calc.df

    # Simulation variables
    trades = []
    active_trade = None
    balance = 10000.0

    for i in range(Config.MIN_CANDLES_REQUIRED, len(df)):
        row = df.iloc[i]
        h1_ema200 = _lookup_h1_ema200(h1_map, row['time'])

        if active_trade is None:
            signal = SignalGenerator.generate_signal(row, h1_ema200)

            if signal in ['BUY', 'SELL']:
                entry_price = row['close']
                atr = row['ATR']

                if signal == 'BUY':
                    sl = entry_price - atr * Config.SL_MULTIPLIER
                    tp = entry_price + atr * Config.TP_MULTIPLIER
                else:
                    sl = entry_price + atr * Config.SL_MULTIPLIER
                    tp = entry_price - atr * Config.TP_MULTIPLIER

                active_trade = {
                    'type': signal,
                    'entry_time': str(row['time']),
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': tp,
                    'atr': atr,
                    'be_activated': False,
                }
        else:
            trade_type = active_trade['type']
            entry = active_trade['entry_price']
            atr = active_trade['atr']
            current_sl = active_trade['sl']
            tp = active_trade['tp']
            high = row['high']
            low = row['low']

            # --- Break-even and trailing stop simulation ---
            if trade_type == 'BUY':
                be_threshold = entry + Config.BE_ACTIVATION_ATR * atr
                if not active_trade['be_activated'] and high >= be_threshold:
                    active_trade['sl'] = max(active_trade['sl'], entry)
                    active_trade['be_activated'] = True

                # Trailing stop: use candle close as proxy for current price
                candle_close = row['close']
                new_trail_sl = candle_close - Config.TRAILING_STOP_ATR * atr
                if active_trade['be_activated'] and new_trail_sl > active_trade['sl']:
                    active_trade['sl'] = new_trail_sl

            else:  # SELL
                be_threshold = entry - Config.BE_ACTIVATION_ATR * atr
                if not active_trade['be_activated'] and low <= be_threshold:
                    active_trade['sl'] = min(active_trade['sl'], entry)
                    active_trade['be_activated'] = True

                candle_close = row['close']
                new_trail_sl = candle_close + Config.TRAILING_STOP_ATR * atr
                if active_trade['be_activated'] and new_trail_sl < active_trade['sl']:
                    active_trade['sl'] = new_trail_sl

            # --- Check for SL/TP hit ---
            hit_sl = False
            hit_tp = False

            if trade_type == 'BUY':
                if low <= active_trade['sl']:
                    hit_sl = True
                elif high >= tp:
                    hit_tp = True
            else:
                if high >= active_trade['sl']:
                    hit_sl = True
                elif low <= tp:
                    hit_tp = True

            if hit_sl or hit_tp:
                exit_price = active_trade['sl'] if hit_sl else tp

                if trade_type == 'BUY':
                    profit_pips = (exit_price - entry) / 0.0001
                else:
                    profit_pips = (entry - exit_price) / 0.0001

                # 0.01 lot EURUSD ≈ $0.10 per pip
                profit_usd = profit_pips * 0.10
                balance += profit_usd

                outcome = "WIN" if profit_pips > 0 else "LOSS"

                trades.append({
                    'type': trade_type,
                    'entry_time': active_trade['entry_time'],
                    'exit_time': str(row['time']),
                    'entry_price': entry,
                    'exit_price': exit_price,
                    'sl_original': entry - atr * Config.SL_MULTIPLIER if trade_type == 'BUY'
                                   else entry + atr * Config.SL_MULTIPLIER,
                    'be_activated': active_trade['be_activated'],
                    'profit_pips': round(profit_pips, 1),
                    'profit_usd': round(profit_usd, 2),
                    'outcome': outcome,
                })
                active_trade = None

    # Calculate metrics
    total_trades = len(trades)
    wins = len([t for t in trades if t['outcome'] == "WIN"])
    losses = total_trades - wins
    accuracy = (wins / total_trades * 100) if total_trades > 0 else 0
    total_profit_usd = sum(t['profit_usd'] for t in trades)

    results = {
        "summary": {
            "symbol": Config.SYMBOL,
            "period": Config.TIMEFRAME_NAME,
            "strategy": "Upgraded (Strict RSI, MACD Zero-Line, ADX, H1 MTF, Session, BE/Trail)",
            "rr_ratio": f"1:{int(Config.TP_MULTIPLIER / Config.SL_MULTIPLIER)}",
            "candles_analyzed": len(df),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "accuracy_percent": round(accuracy, 2),
            "total_profit_usd": round(total_profit_usd, 2),
            "final_balance": round(balance, 2),
            "start_time": str(df['time'].iloc[0]),
            "end_time": str(df['time'].iloc[-1]),
        },
        "trades": trades,
    }

    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n--- BACKTEST COMPLETE ---")
    print(f"Total Trades : {total_trades}")
    print(f"Wins / Losses: {wins} / {losses}")
    print(f"Accuracy     : {accuracy:.2f}%")
    print(f"Profit/Loss  : {total_profit_usd:.2f} USD")
    print("Results saved to backtest_results.json")

    mt5.shutdown()
    return results

if __name__ == "__main__":
    run_backtest()
