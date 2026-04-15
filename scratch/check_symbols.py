import MetaTrader5 as mt5
import os

def load_mt5_config(file_path):
    config = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip().strip('"').strip("'")
    return config

mt5_conf = load_mt5_config('config')

if not mt5.initialize():
    print("initialize() failed")
    quit()

login = int(mt5_conf.get('LOGIN'))
password = mt5_conf.get('PASSWORD')
server = mt5_conf.get('SERVER')

authorized = mt5.login(login=login, password=password, server=server)
if authorized:
    print(f"Logged in to {login}")
    symbols = mt5.symbols_get()
    print(f"Total symbols: {len(symbols)}")
    # Print first 20 symbols
    for s in symbols[:50]:
        print(s.name)
else:
    print("Login failed")

mt5.shutdown()
