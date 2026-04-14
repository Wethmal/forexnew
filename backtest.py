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
    LOT_SIZE = 0.01
    
    EMA_FAST = 50
    EMA_SLOW = 200
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 65
    RSI_OVERSOLD = 45
    
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    BB_PERIOD = 20
    BB_STD_DEV = 2
    BB_MIN_WIDTH = 0.0005
    
    ATR_PERIOD = 14
    SL_MULTIPLIER = 1.5
    TP_MULTIPLIER = 3.0
    
    LOGIN = 413646889
    PASSWORD = "Anoma@0822"
    SERVER = "Exness-MT5Trial6"
    
    MIN_CANDLES_REQUIRED = 200
    BACKTEST_CANDLES = 2000  # Number of candles to backtest

# ============================================================================
# INDICATOR CALCULATOR (Simplified for Backtesting)
# ============================================================================

class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_all_indicators()
    
    def _calculate_all_indicators(self):
        self.df['EMA_50'] = ta.ema(self.df['close'], length=Config.EMA_FAST)
        self.df['EMA_200'] = ta.ema(self.df['close'], length=Config.EMA_SLOW)
        self.df['RSI'] = ta.rsi(self.df['close'], length=Config.RSI_PERIOD)
        
        macd_result = ta.macd(self.df['close'], fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
        if macd_result is not None and not macd_result.empty:
            macd_col = [c for c in macd_result.columns if c.startswith('MACD_') and 's' not in c and 'h' not in c]
            hist_col = [c for c in macd_result.columns if c.startswith('MACDh_')]
            self.df['MACD'] = macd_result[macd_col[0]] if macd_col else np.nan
            self.df['MACD_Hist'] = macd_result[hist_col[0]] if hist_col else np.nan
        
        bb_result = ta.bbands(self.df['close'], length=Config.BB_PERIOD, std=Config.BB_STD_DEV)
        if bb_result is not None and not bb_result.empty:
            upper_col = [c for c in bb_result.columns if c.startswith('BBU_')]
            lower_col = [c for c in bb_result.columns if c.startswith('BBL_')]
            self.df['BB_Upper'] = bb_result[upper_col[0]] if upper_col else np.nan
            self.df['BB_Lower'] = bb_result[lower_col[0]] if lower_col else np.nan
            
        self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=Config.ATR_PERIOD)

# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    @staticmethod
    def generate_signal(row):
        # 1. Trend Direction
        if row['close'] > row['EMA_50'] > row['EMA_200']:
            trend_dir = 'UP'
        elif row['close'] < row['EMA_50'] < row['EMA_200']:
            trend_dir = 'DOWN'
        else:
            return 'HOLD'

        # 2. RSI Filter
        if not (Config.RSI_OVERSOLD < row['RSI'] < Config.RSI_OVERBOUGHT):
            return 'HOLD'
        
        # 3. MACD Confirmation
        if trend_dir == 'UP' and row['MACD_Hist'] <= 0:
            return 'HOLD'
        if trend_dir == 'DOWN' and row['MACD_Hist'] >= 0:
            return 'HOLD'
        
        # 4. Bollinger Bands Filter
        band_width = row['BB_Upper'] - row['BB_Lower']
        if band_width < Config.BB_MIN_WIDTH:
            return 'HOLD'
        
        # 5. ATR Validity
        if pd.isna(row['ATR']) or row['ATR'] <= 0:
            return 'HOLD'
        
        return 'BUY' if trend_dir == 'UP' else 'SELL'

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_backtest():
    print("Starting Backtest...")
    
    if not mt5.initialize():
        print("MT5 Initialization failed")
        return
    
    # Login to account
    if not mt5.login(Config.LOGIN, Config.PASSWORD, Config.SERVER):
        print("Failed to login to MT5 account")
        print(f"Error: {mt5.last_error()}")
        mt5.shutdown()
        return
    
    print(f"Logged into {Config.SERVER}")
    
    # Fetch candles
    rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, Config.BACKTEST_CANDLES)
    if rates is None:
        print(f"Failed to fetch data for {Config.SYMBOL}")
        mt5.shutdown()
        return
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Calculate indicators
    calc = IndicatorCalculator(df)
    df = calc.df
    
    # Simulation variables
    trades = []
    active_trade = None
    balance = 10000.0
    
    # Iterate through candles
    for i in range(Config.MIN_CANDLES_REQUIRED, len(df)):
        row = df.iloc[i]
        
        if active_trade is None:
            # Check for signal
            signal = SignalGenerator.generate_signal(row)
            
            if signal in ['BUY', 'SELL']:
                entry_price = row['close']
                atr = row['ATR']
                
                if signal == 'BUY':
                    sl = entry_price - (atr * Config.SL_MULTIPLIER)
                    tp = entry_price + (atr * Config.TP_MULTIPLIER)
                else:
                    sl = entry_price + (atr * Config.SL_MULTIPLIER)
                    tp = entry_price - (atr * Config.TP_MULTIPLIER)
                
                active_trade = {
                    'type': signal,
                    'entry_time': str(row['time']),
                    'entry_price': entry_price,
                    'sl': sl,
                    'tp': tp,
                }
        else:
            # Check for exit (using high/low of current candle)
            trade_type = active_trade['type']
            hit_sl = False
            hit_tp = False
            
            if trade_type == 'BUY':
                if row['low'] <= active_trade['sl']:
                    hit_sl = True
                elif row['high'] >= active_trade['tp']:
                    hit_tp = True
            else: # SELL
                if row['high'] >= active_trade['sl']:
                    hit_sl = True
                elif row['low'] <= active_trade['tp']:
                    hit_tp = True
            
            if hit_sl or hit_tp:
                # Close trade
                exit_price = active_trade['sl'] if hit_sl else active_trade['tp']
                
                # Simple profit calculation in pips (ignoring lot size for accuracy %)
                # For EURUSD, 1 pip = 0.0001
                if trade_type == 'BUY':
                    profit_pips = (exit_price - active_trade['entry_price']) / 0.0001
                else:
                    profit_pips = (active_trade['entry_price'] - exit_price) / 0.0001
                
                # Approximate monetary profit (for 0.01 lot)
                # 0.01 lot EURUSD = $0.10 per pip
                profit_usd = profit_pips * 0.10
                balance += profit_usd
                
                outcome = "WIN" if profit_pips > 0 else "LOSS"
                
                trades.append({
                    'type': trade_type,
                    'entry_time': active_trade['entry_time'],
                    'exit_time': str(row['time']),
                    'entry_price': active_trade['entry_price'],
                    'exit_price': exit_price,
                    'profit_pips': profit_pips,
                    'profit_usd': profit_usd,
                    'outcome': outcome
                })
                active_trade = None

    # Calculate metrics
    total_trades = len(trades)
    wins = len([t for t in trades if t['outcome'] == "WIN"])
    losses = total_trades - wins
    accuracy = (wins / total_trades * 100) if total_trades > 0 else 0
    
    total_profit_usd = sum([t['profit_usd'] for t in trades])
    
    results = {
        "summary": {
            "symbol": Config.SYMBOL,
            "period": Config.TIMEFRAME_NAME,
            "candles_analyzed": len(df),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "accuracy_percent": round(accuracy, 2),
            "total_profit_usd": round(total_profit_usd, 2),
            "final_balance": round(balance, 2),
            "start_time": str(df['time'].iloc[0]),
            "end_time": str(df['time'].iloc[-1])
        },
        "trades": trades
    }
    
    # Output as JSON
    with open("backtest_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n--- BACKTEST COMPLETE ---")
    print(f"Total Trades: {total_trades}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Profit/Loss: {total_profit_usd:.2f} USD")
    print("Results saved to backtest_results.json")
    
    mt5.shutdown()
    return results

if __name__ == "__main__":
    run_backtest()
