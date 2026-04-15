import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from datetime import datetime

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    return logging.getLogger(__name__)

logger = setup_logging()

class Config:
    SYMBOLS = ["EURUSDm", "USDJPYm", "GBPUSDm", "AUDUSDm"]
    TIMEFRAME = mt5.TIMEFRAME_H1
    BACKTEST_CANDLES = 2000
    EMA_FAST = 50
    EMA_SLOW = 200
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ATR_PERIOD = 14
    BB_PERIOD = 20
    BB_STD_DEV = 2
    LOGIN = 413646889
    PASSWORD = "Anoma@0822"
    SERVER = "Exness-MT5Trial6"

class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calc()
    
    def _calc(self):
        self.df['EMA_50'] = ta.ema(self.df['close'], length=Config.EMA_FAST)
        self.df['EMA_200'] = ta.ema(self.df['close'], length=Config.EMA_SLOW)
        self.df['RSI'] = ta.rsi(self.df['close'], length=Config.RSI_PERIOD)
        self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=Config.ATR_PERIOD)
        
        macd = ta.macd(self.df['close'], fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
        if macd is not None:
            self.df['MACD'] = macd.iloc[:, 0]
            self.df['MACD_Signal'] = macd.iloc[:, 1]
            self.df['MACD_Hist'] = macd.iloc[:, 2]
            
        bb = ta.bbands(self.df['close'], length=Config.BB_PERIOD, std=Config.BB_STD_DEV)
        if bb is not None:
            self.df['BB_Upper'] = bb.iloc[:, 0]
            self.df['BB_Lower'] = bb.iloc[:, 2]

class MultiSymbolBacktest:
    def __init__(self):
        self.results = {}
        
    def run(self):
        if not mt5.initialize():
            logger.error("MT5 init failed")
            return
            
        if not mt5.login(Config.LOGIN, Config.PASSWORD, Config.SERVER):
            logger.error("MT5 login failed")
            return

        for symbol in Config.SYMBOLS:
            logger.info(f"Backtesting {symbol}...")
            trades = self.backtest_symbol(symbol)
            
            if trades:
                wins = len([t for t in trades if t['outcome'] == 'WIN'])
                accuracy = (wins / len(trades)) * 100
                self.results[symbol] = {
                    'trades': len(trades),
                    'wins': wins,
                    'accuracy': accuracy,
                    'profit': sum([t['profit'] for t in trades])
                }
                logger.info(f"Result for {symbol}: {len(trades)} trades, {accuracy:.2f}% accuracy")
            else:
                logger.info(f"No trades for {symbol}")
                self.results[symbol] = {'trades': 0, 'wins': 0, 'accuracy': 0, 'profit': 0}
        
        mt5.shutdown()
        self.summary()

    def backtest_symbol(self, symbol):
        rates = mt5.copy_rates_from_pos(symbol, Config.TIMEFRAME, 0, Config.BACKTEST_CANDLES)
        if rates is None: return []
        
        df = pd.DataFrame(rates)
        calc = IndicatorCalculator(df)
        df = calc.df
        
        trades = []
        active = None
        
        for i in range(200, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]
            
            if active:
                hit = False
                if active['type'] == 'BUY':
                    if row['low'] <= active['sl']: 
                        exit_p, outcome, hit = active['sl'], 'LOSS', True
                    elif row['high'] >= active['tp']: 
                        exit_p, outcome, hit = active['tp'], 'WIN', True
                else:
                    if row['high'] >= active['sl']: 
                        exit_p, outcome, hit = active['sl'], 'LOSS', True
                    elif row['low'] <= active['tp']: 
                        exit_p, outcome, hit = active['tp'], 'WIN', True
                
                if hit:
                    multiplier = 100 if 'JPY' in symbol else 10000
                    profit = (exit_p - active['entry']) * multiplier if active['type'] == 'BUY' else (active['entry'] - exit_p) * multiplier
                    active.update({'exit': exit_p, 'outcome': outcome, 'profit': profit})
                    trades.append(active)
                    active = None
            
            if not active:
                sig = 'HOLD'
                if row['close'] > row['EMA_50'] > row['EMA_200'] and row['RSI'] > 65:
                    body = row['close'] - row['open']
                    if row['close'] > prev['high'] and body > (prev['high'] - prev['low']) * 0.2: sig = 'BUY'
                elif row['close'] < row['EMA_50'] < row['EMA_200'] and row['RSI'] < 35:
                    body = row['open'] - row['close']
                    if row['close'] < prev['low'] and body > (prev['high'] - prev['low']) * 0.2: sig = 'SELL'
                
                if sig != 'HOLD':
                    entry = row['close']
                    atr = row['ATR']
                    # Using 1.0 multiplier for TP and 1.5 for SL to favor win rate
                    active = {'type': sig, 'entry': entry, 'time': row['time'], 'sl': entry - (atr * 1.5) if sig=='BUY' else entry + (atr * 1.5), 'tp': entry + (atr * 1.0) if sig=='BUY' else entry - (atr * 1.0)}
        return trades

    def summary(self):
        print("\n" + "="*40)
        print("BACKTEST SUMMARY")
        print("="*40)
        total_trades = 0
        total_wins = 0
        for s, r in self.results.items():
            print(f"{s}: {r['accuracy']:.2f}% ({r['wins']}/{r['trades']}) | Profit: {r['profit']:.1f} pips")
            total_trades += r['trades']
            total_wins += r['wins']
        
        avg_acc = (total_wins / total_trades * 100) if total_trades > 0 else 0
        print("-" * 40)
        print(f"OVERALL ACCURACY: {avg_acc:.2f}%")
        print("="*40)

if __name__ == "__main__":
    bt = MultiSymbolBacktest()
    bt.run()
