
import MetaTrader5 as mt5
import os

def check_symbols():
    # Load config manually
    creds = {}
    if os.path.exists("config"):
        with open("config", "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, val = line.split("=", 1)
                    creds[key.strip().upper()] = val.strip().strip('"')
    
    login = int(creds.get("LOGIN", 0))
    password = creds.get("PASSWORD", "")
    server = creds.get("SERVER", "")
    
    mt5.initialize()
    mt5.login(login=login, password=password, server=server)
    
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    suffixes = ["", "m", "+", "."]
    
    print("Checking Symbols:")
    for sym in symbols:
        found = False
        for suffix in suffixes:
            full_sym = sym + suffix
            info = mt5.symbol_info(full_sym)
            if info:
                print(f"  FOUND: {full_sym} (Digits: {info.digits})")
                found = True
                break
        if not found:
            print(f"  NOT FOUND: {sym} with common suffixes")

    mt5.shutdown()

if __name__ == "__main__":
    check_symbols()
