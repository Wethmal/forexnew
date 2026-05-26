import pandas as pd


def _profit_factor(pnl: pd.Series) -> float:
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def main() -> None:
    results_path = "trade_results.csv"
    signals_path = "trade_signals.csv"

    print("\n=== Trade Results ===")
    try:
        res = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Missing {results_path}. Run the bot so it can sync MT5 history.")
        return

    if res.empty:
        print(f"{results_path} is empty (no closed bot trades synced yet).")
    else:
        pnl = res["Profit"].fillna(0) + res.get("Commission", 0).fillna(0) + res.get("Swap", 0).fillna(0)
        n = len(res)
        win_rate = (pnl > 0).mean() * 100
        print(f"Trades: {n} | Win rate: {win_rate:.1f}% | Net PnL: {pnl.sum():.2f} | PF: {_profit_factor(pnl):.2f}")
        by_sym = (
            res.assign(Net=pnl)
            .groupby("Symbol", dropna=False)
            .agg(Trades=("Net", "size"), WinRate=("Net", lambda s: (s > 0).mean() * 100), NetPnL=("Net", "sum"))
            .sort_values("NetPnL", ascending=False)
        )
        print("\nBy symbol:")
        print(by_sym.to_string(float_format=lambda x: f"{x:.2f}"))

    print("\n=== Signal Quality (no PnL join) ===")
    try:
        sig = pd.read_csv(signals_path)
    except FileNotFoundError:
        print(f"Missing {signals_path}. It will be created when the bot places trades.")
        return

    if sig.empty:
        print(f"{signals_path} is empty.")
        return

    if "Final_Confidence" in sig.columns:
        buckets = pd.cut(sig["Final_Confidence"].astype(float), bins=[0, 0.58, 0.65, 0.75, 0.85, 1.0])
        print("Signals by Final_Confidence bucket:")
        print(sig.groupby(buckets).size().to_string())

    print("\nSignals by symbol:")
    if "Symbol" in sig.columns:
        print(sig.groupby("Symbol").size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
