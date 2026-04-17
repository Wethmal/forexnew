from bot import GoldenConfig, GoldenTradingBot, Indicators, HAS_SKL, DataFetcher
import logging

# Setup logging to see the output
logging.basicConfig(level=logging.INFO)

def check_ml_accuracy():
    if not HAS_SKL:
        print("Scikit-learn not installed.")
        return

    cfg = GoldenConfig()
    bot = GoldenTradingBot(cfg)
    fetcher = DataFetcher(cfg)
    
    print("\nTraining ML models and checking accuracy...")
    for sym in cfg.symbols:
        try:
            df = fetcher.fetch_yf(sym, "1d")
            if not df.empty:
                df = Indicators.add_all(df, cfg)
                acc = bot.ml.train(sym, df)
                # Accuracy is already logged in bot.ml.train
        except Exception as e:
            print(f"Error training {sym}: {e}")

if __name__ == "__main__":
    check_ml_accuracy()
