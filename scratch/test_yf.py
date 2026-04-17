import yfinance as yf
import pandas as pd

def test_yf():
    symbol = "EURUSD=X"
    print(f"Testing download for {symbol}...")
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
        print(f"DataFrame empty: {df.empty}")
        if not df.empty:
            print(f"Columns: {df.columns}")
            print(f"Head:\n{df.head()}")
        else:
            print("No data received.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_yf()
