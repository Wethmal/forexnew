import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np

mt5.initialize()
# Use a common symbols name or EURUSDm
rates = mt5.copy_rates_from_pos("EURUSDm", mt5.TIMEFRAME_M15, 0, 100)
df = pd.DataFrame(rates)

print("--- MACD ---")
macd = ta.macd(df['close'])
print(macd.columns.tolist())

print("--- BBANDS ---")
bb = ta.bbands(df['close'])
print(bb.columns.tolist())

print("--- ADX ---")
adx = ta.adx(df['high'], df['low'], df['close'])
print(adx.columns.tolist())

print("--- STOCH ---")
stoch = ta.stoch(df['high'], df['low'], df['close'])
print(stoch.columns.tolist())

mt5.shutdown()
