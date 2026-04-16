
import MetaTrader5 as mt5
import os

def check_mt5_status():
    print("--- MT5 Connection Check ---")
    
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
    
    if not mt5.initialize():
        print(f"FAILED: mt5.initialize() error: {mt5.last_error()}")
        return

    if not mt5.login(login=login, password=password, server=server):
        print(f"FAILED: mt5.login() error: {mt5.last_error()}")
        mt5.shutdown()
        return

    acc = mt5.account_info()
    term = mt5.terminal_info()
    
    if acc:
        print(f"SUCCESS: Connected to {acc.server}")
        print(f"Account: {acc.login}")
        print(f"Balance: {acc.balance}")
        print(f"Equity: {acc.equity}")
        print(f"Trade Mode: {acc.trade_mode} (0=Demo, 1=Contest, 2=Real)")
        print(f"Trade Allowed (Account): {acc.trade_allowed}")
    
    if term:
        print(f"Trade Allowed (Terminal/Button): {term.trade_allowed}")
        print(f"Connected: {term.connected}")

    mt5.shutdown()

if __name__ == "__main__":
    check_mt5_status()
