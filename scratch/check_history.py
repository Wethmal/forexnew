
import MetaTrader5 as mt5
import os
from datetime import datetime, timedelta

def check_history():
    # Load config
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

    # From 30 days ago to now
    from_date = datetime.now() - timedelta(days=30)
    to_date = datetime.now()
    
    # Get history deals
    deals = mt5.history_deals_get(from_date, to_date)
    if deals is None:
        print(f"No deals found or error: {mt5.last_error()}")
    else:
        print(f"Found {len(deals)} history deals")
        for d in deals:
            # Entry 0 = In, 1 = Out, 2 = In/Out
            # Profit is only on Entry Out or In/Out
            if d.entry in [1, 2] and d.magic == 20250415: # Use bot's magic
                print(f"Trade: Ticket={d.ticket}, Symbol={d.symbol}, Profit={d.profit}, Pips={(d.price_current - d.price_open) if d.entry==1 else 0}")
            elif d.entry in [1, 2]:
                 print(f"Trade (External Magic {d.magic}): Ticket={d.ticket}, Symbol={d.symbol}, Profit={d.profit}")

    mt5.shutdown()

if __name__ == "__main__":
    check_history()
