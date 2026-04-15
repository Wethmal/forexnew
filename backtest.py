import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    return logging.getLogger(__name__)

logger = setup_logging()

class Config:
    SYMBOL = "EURUSDm"
    TIMEFRAME = mt5.TIMEFRAME_H1
    
    # EMAs
    EMA_FAST = 9
    EMA_PULLBACK = 21
    EMA_TREND = 200
    
    # 1:1 RR for >50% Accuracy Goal
    # We will use Fixed Pips or ATR for RR 1:1
    RR_RATIO = 1.0 # 1:1
    
    BACKTEST_CANDLES = 5000
    LOGIN = 413646889
    PASSWORD = "Anoma@0822"
    SERVER = "Exness-MT5Trial6"

class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calc()
    
    def _calc(self):
        self.df['EMA_9'] = ta.ema(self.df['close'], length=9)
        self.df['EMA_21'] = ta.ema(self.df['close'], length=21)
        self.df['EMA_200'] = ta.ema(self.df['close'], length=200)
        self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'])

class TrendPullbackBacktest:
    def __init__(self):
        self.trades = []
        self.balance = 10000.0
        
    def run(self):
        mt5.initialize()
        mt5.login(Config.LOGIN, Config.PASSWORD, Config.SERVER)
        
        r = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, Config.BACKTEST_CANDLES)
        df = pd.DataFrame(r)
        calc = IndicatorCalculator(df)
        df = calc.df
        
        dr = mt5.copy_rates_from_pos(Config.SYMBOL, mt5.TIMEFRAME_D1, 0, 500)
        ddf = pd.DataFrame(dr)
        ddf['EMA_50'] = ta.ema(ddf['close'], length=50)
        ddf['time'] = pd.to_datetime(ddf['time'], unit='s')
        
        active = None
        for i in range(200, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            if active:
                hit = False
                if active['type'] == 'BUY':
                    if row['low'] <= active['sl']: exit_p, outcome, hit = active['sl'], 'LOSS', True
                    elif row['high'] >= active['tp']: exit_p, outcome, hit = active['tp'], 'WIN', True
                else:
                    if row['high'] >= active['sl']: exit_p, outcome, hit = active['sl'], 'LOSS', True
                    elif row['low'] <= active['tp']: exit_p, outcome, hit = active['tp'], 'WIN', True
                
                if hit:
                    profit = ((exit_p - active['entry']) / 0.0001 if active['type'] == 'BUY' else (active['entry'] - exit_p) / 0.0001) * 0.1
                    self.balance += profit
                    active.update({'exit': exit_p, 'outcome': outcome, 'profit': profit})
                    self.trades.append(active)
                    active = None
            
            if not active:
                dt = pd.to_datetime(row['time'], unit='s').normalize()
                d_row = ddf[ddf['time'] <= dt].tail(1)
                if d_row.empty: continue
                d_trend = 'UP' if d_row['close'].values[0] > d_row['EMA_50'].values[0] else 'DOWN'
                
                sig = 'HOLD'
                if d_trend == 'UP' and row['close'] > row['EMA_200']:
                    # Pullback: Price low touched EMA_21 in last 3 candles
                    touched = (df.iloc[i-3:i]['low'] <= df.iloc[i-3:i]['EMA_21']).any()
                    # Trigger: Current close > EMA_9
                    if touched and row['close'] > row['EMA_9'] and prev['close'] <= prev['EMA_9']:
                        sig = 'BUY'
                elif d_trend == 'DOWN' and row['close'] < row['EMA_200']:
                    touched = (df.iloc[i-3:i]['high'] >= df.iloc[i-3:i]['EMA_21']).any()
                    if touched and row['close'] < row['EMA_9'] and prev['close'] >= prev['EMA_9']:
                        sig = 'SELL'
                
                if sig != 'HOLD':
                    atr = row['ATR']
                    dist = atr * 2.0
                    active = {
                        'type': sig, 'entry': row['close'],
                        'sl': row['close'] - dist if sig == 'BUY' else row['close'] + dist,
                        'tp': row['close'] + (dist * Config.RR_RATIO) if sig == 'BUY' else row['close'] - (dist * Config.RR_RATIO)
                    }
        
        counts = len(self.trades)
        wins = len([t for t in self.trades if t['outcome'] == 'WIN'])
        acc = (wins / counts * 100) if counts else 0
        logger.info(f"FINAL SNIPER RESULTS: Trades: {counts} | Accuracy: {acc:.2f}% | P/L: {self.balance - 10000:.2f}")
        mt5.shutdown()

if __name__ == "__main__":
    TrendPullbackBacktest().run()
