"""
================================================================================
FOREX BOT BACKTEST ENGINE
================================================================================
Tests trading logic on historical data to validate accuracy and profitability.

Usage:
    python backtest.py [days] [symbols]
    
Example:
    python backtest.py 30 EURUSD GBPUSD  # Last 30 days, 2 symbols
    python backtest.py 60  # Last 60 days, all configured symbols
    python backtest.py     # Last 30 days (default), all symbols

================================================================================
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# Try to import pandas_ta, but it's optional
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from bot import (
    Config, IndicatorCalculator, SignalGenerator, TradeManager,
    MarketStructureAnalyzer, SafeEncoder, MT5Manager, setup_logging, HAS_PANDAS_TA
)
import bot as bot_module

logger = setup_logging()


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """Simulates trading strategy on historical data."""

    def __init__(self, cfg: Config, mt5_manager: MT5Manager):
        self.cfg = cfg
        self.mt5 = mt5_manager
        self.trades: List[dict] = []
        self.daily_pnl: Dict[str, float] = defaultdict(float)

    def run_backtest(self, symbols: List[str], days: int) -> dict:
        """
        Run backtest on historical data.
        
        Args:
            symbols: List of symbols to test
            days: Number of days of historical data to fetch
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("=" * 70)
        logger.info(f"BACKTEST: {days} days | Symbols: {', '.join(symbols)}")
        logger.info("=" * 70)

        results = {
            "start_date": [],
            "end_date": [],
            "symbols": symbols,
            "days": days,
            "trades": [],
            "metrics": {},
        }

        for symbol in symbols:
            logger.info(f"\n--- Testing {symbol} ---")
            
            # Fetch H1 data
            df = self.mt5.fetch_candles(
                symbol, self.cfg.TIMEFRAME, 
                self.cfg.MIN_CANDLES_REQUIRED + (days * 24)
            )
            if df is None or len(df) == 0:
                logger.warning(f"  No data for {symbol}")
                continue

            # Fetch H4 data for trend filter
            df_h4 = self.mt5.fetch_candles(symbol, self.cfg.TREND_TIMEFRAME, 250)
            
            logger.info(f"  Data: {len(df)} H1 candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
            
            # Simulate trading
            symbol_trades = self._simulate_symbol(symbol, df, df_h4)
            logger.info(f"  Signals: {len(symbol_trades)} trades")
            
            results["trades"].extend(symbol_trades)
            
            if not results["start_date"]:
                results["start_date"] = df['time'].iloc[0].strftime("%Y-%m-%d")
            results["end_date"] = df['time'].iloc[-1].strftime("%Y-%m-%d")

        # Calculate overall metrics
        results["metrics"] = self._calculate_metrics(results["trades"])
        
        return results

    def _simulate_symbol(self, symbol: str, df: pd.DataFrame, 
                        df_h4: Optional[pd.DataFrame]) -> List[dict]:
        """Simulate trading for a single symbol across all historical data."""
        
        trades = []
        
        # Process each candle
        for i in range(self.cfg.MIN_CANDLES_REQUIRED, len(df)):
            current_time = df['time'].iloc[i]
            hour = current_time.hour
            
            # Get data up to this point (lookahead prevention)
            hist_df = df.iloc[: i + 1]
            
            # Calculate H4 context
            h4_data = self._get_h4_context(symbol, current_time, df_h4)
            
            # Calculate indicators
            calc = IndicatorCalculator(hist_df)
            ind = calc.get_latest()
            
            # Skip if insufficient data
            if pd.isna(ind.get("close", np.nan)):
                continue
            
            # Generate signal
            signal, details = SignalGenerator.generate_signal(
                ind, h4_data, hour, calc.df
            )
            
            # Skip non-trades and out-of-session
            if signal == "HOLD":
                continue
            
            confluence = details.get("confluence_score", 0)
            
            # Simulate trade entry and exit
            entry_price = ind["close"]
            entry_time = current_time
            entry_index = i
            
            # Look ahead to find exit
            trade_result = self._find_trade_exit(
                symbol, signal, entry_price, ind["atr"], 
                hist_df, entry_index, df
            )
            
            if trade_result:
                trade_result.update({
                    "symbol": symbol,
                    "signal": signal,
                    "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"),
                    "entry_price": entry_price,
                    "confluence": confluence,
                })
                trades.append(trade_result)
        
        return trades

    def _get_h4_context(self, symbol: str, current_time: pd.Timestamp, 
                       df_h4: Optional[pd.DataFrame]) -> dict:
        """Get H4 trend context at current time."""
        if df_h4 is None or len(df_h4) == 0:
            return {"close": np.nan, "ema_200": np.nan}
        
        # Find closest H4 candle before or at current time
        h4_subset = df_h4[df_h4['time'] <= current_time]
        if len(h4_subset) == 0:
            return {"close": np.nan, "ema_200": np.nan}
        
        latest = h4_subset.iloc[-1]
        
        # Calculate EMA on this subset
        if len(h4_subset) >= 200:
            if HAS_PANDAS_TA:
                h4_ema = ta.ema(h4_subset["close"], length=200)
                ema_200 = float(h4_ema.iloc[-1])
            else:
                h4_ema = bot_module._calculate_ema(h4_subset["close"].values, 200)
                ema_200 = float(h4_ema[-1])
        else:
            ema_200 = np.nan
        
        return {
            "close": float(latest["close"]),
            "ema_200": ema_200,
        }

    def _find_trade_exit(self, symbol: str, signal: str, entry_price: float,
                        atr: float, hist_df: pd.DataFrame, entry_index: int,
                        full_df: pd.DataFrame, max_bars: int = 240) -> Optional[dict]:
        """
        Simulate trade by looking ahead for exit (TP hit, SL hit, time-based).
        
        Args:
            symbol: Trading pair
            signal: BUY or SELL
            entry_price: Entry price
            atr: ATR at entry
            hist_df: Historical data up to entry
            entry_index: Index in full_df where entry occurred
            full_df: Full historical dataframe
            max_bars: Max bars to hold trade (default 240 H1 = 10 days)
            
        Returns:
            Trade result dict or None
        """
        
        # Calculate SL/TP
        mgr = TradeManager(symbol, self.cfg.LOT_SIZE)
        
        # Get support/resistance for better SL
        structure = MarketStructureAnalyzer.detect_structure(hist_df)
        support = structure.get("support", None)
        resistance = structure.get("resistance", None)
        
        pos = mgr.calculate_positions(entry_price, atr, signal, support, resistance)
        
        sl_price = pos["sl_price"]
        tp_price = pos["tp_price"]
        
        # Look ahead for exit
        max_look_ahead = min(max_bars, len(full_df) - entry_index - 1)
        if max_look_ahead <= 0:
            return None  # Not enough data after entry
        
        for j in range(1, max_look_ahead + 1):
            bar_index = entry_index + j
            if bar_index >= len(full_df):
                break
            
            bar = full_df.iloc[bar_index]
            exit_time = bar["time"]
            
            # Check for exit conditions
            if signal == "BUY":
                # TP hit
                if bar["high"] >= tp_price:
                    return {
                        "exit_time": exit_time.strftime("%Y-%m-%d %H:%M"),
                        "exit_price": tp_price,
                        "profit_pips": (tp_price - entry_price) / self.cfg.LOT_SIZE * 10000,
                        "profit": (tp_price - entry_price) * 100000,  # Approximate
                        "outcome": "WIN",
                        "bars_held": j,
                        "reason": "TP_HIT",
                        "sl": sl_price,
                        "tp": tp_price,
                        "rr_ratio": pos["rr_ratio"],
                    }
                # SL hit
                elif bar["low"] <= sl_price:
                    return {
                        "exit_time": exit_time.strftime("%Y-%m-%d %H:%M"),
                        "exit_price": sl_price,
                        "profit_pips": (sl_price - entry_price) / self.cfg.LOT_SIZE * 10000,
                        "profit": (sl_price - entry_price) * 100000,
                        "outcome": "LOSS",
                        "bars_held": j,
                        "reason": "SL_HIT",
                        "sl": sl_price,
                        "tp": tp_price,
                        "rr_ratio": pos["rr_ratio"],
                    }
            else:  # SELL
                # TP hit
                if bar["low"] <= tp_price:
                    return {
                        "exit_time": exit_time.strftime("%Y-%m-%d %H:%M"),
                        "exit_price": tp_price,
                        "profit_pips": (entry_price - tp_price) / self.cfg.LOT_SIZE * 10000,
                        "profit": (entry_price - tp_price) * 100000,
                        "outcome": "WIN",
                        "bars_held": j,
                        "reason": "TP_HIT",
                        "sl": sl_price,
                        "tp": tp_price,
                        "rr_ratio": pos["rr_ratio"],
                    }
                # SL hit
                elif bar["high"] >= sl_price:
                    return {
                        "exit_time": exit_time.strftime("%Y-%m-%d %H:%M"),
                        "exit_price": sl_price,
                        "profit_pips": (entry_price - sl_price) / self.cfg.LOT_SIZE * 10000,
                        "profit": (entry_price - sl_price) * 100000,
                        "outcome": "LOSS",
                        "bars_held": j,
                        "reason": "SL_HIT",
                        "sl": sl_price,
                        "tp": tp_price,
                        "rr_ratio": pos["rr_ratio"],
                    }
        
        # Exit on timeout (reached max bars)
        if max_look_ahead >= max_bars:
            final_bar = full_df.iloc[entry_index + max_bars - 1]
            exit_price = final_bar["close"]
            profit = (exit_price - entry_price) * 100000 if signal == "BUY" else (entry_price - exit_price) * 100000
            
            return {
                "exit_time": final_bar["time"].strftime("%Y-%m-%d %H:%M"),
                "exit_price": exit_price,
                "profit_pips": (exit_price - entry_price) / self.cfg.LOT_SIZE * 10000 if signal == "BUY" else (entry_price - exit_price) / self.cfg.LOT_SIZE * 10000,
                "profit": profit,
                "outcome": "WIN" if profit > 0 else "LOSS",
                "bars_held": max_bars,
                "reason": "TIMEOUT",
                "sl": sl_price,
                "tp": tp_price,
                "rr_ratio": pos["rr_ratio"],
            }
        
        return None

    def _calculate_metrics(self, trades: List[dict]) -> dict:
        """Calculate comprehensive backtest metrics."""
        
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "gross_profit": 0,
                "gross_loss": 0,
                "net_profit": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "largest_win": 0,
                "largest_loss": 0,
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "average_bars_held": 0,
                "payoff_ratio": 0,
                "sharp_ratio_approx": 0,
                "by_confluence": {},
                "by_symbol": {},
            }
        
        df_trades = pd.DataFrame(trades)
        
        # Basic stats
        total = len(trades)
        wins = len(df_trades[df_trades['outcome'] == 'WIN'])
        losses = len(df_trades[df_trades['outcome'] == 'LOSS'])
        win_rate = (wins / total * 100) if total > 0 else 0
        
        # Profitability
        gross_profit = df_trades[df_trades['outcome'] == 'WIN']['profit'].sum()
        gross_loss = abs(df_trades[df_trades['outcome'] == 'LOSS']['profit'].sum())
        net_profit = gross_profit - gross_loss
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Trade statistics
        win_trades = df_trades[df_trades['outcome'] == 'WIN']
        loss_trades = df_trades[df_trades['outcome'] == 'LOSS']
        
        avg_win = win_trades['profit'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['profit'].mean() if len(loss_trades) > 0 else 0
        
        largest_win = df_trades['profit'].max()
        largest_loss = df_trades['profit'].min()
        
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Consecutive wins/losses
        max_consecutive_wins = self._max_consecutive(df_trades['outcome'], 'WIN')
        max_consecutive_losses = self._max_consecutive(df_trades['outcome'], 'LOSS')
        
        # Average bars held
        avg_bars = df_trades['bars_held'].mean()
        
        # Sharpe ratio approximation (simplified - assumes 252 trading days)
        returns = df_trades['profit'].pct_change().dropna() if len(df_trades) > 1 else pd.Series()
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 and returns.std() > 0 else 0
        
        # By confluence score
        by_confluence = {}
        for conf in sorted(df_trades['confluence'].unique()):
            conf_trades = df_trades[df_trades['confluence'] == conf]
            conf_wins = len(conf_trades[conf_trades['outcome'] == 'WIN'])
            conf_wr = (conf_wins / len(conf_trades) * 100) if len(conf_trades) > 0 else 0
            conf_profit = conf_trades['profit'].sum()
            
            by_confluence[int(conf)] = {
                "count": len(conf_trades),
                "wins": conf_wins,
                "win_rate": round(conf_wr, 2),
                "profit": round(conf_profit, 2),
                "avg_profit": round(conf_trades['profit'].mean(), 2),
            }
        
        # By symbol
        by_symbol = {}
        for sym in sorted(df_trades['symbol'].unique()):
            sym_trades = df_trades[df_trades['symbol'] == sym]
            sym_wins = len(sym_trades[sym_trades['outcome'] == 'WIN'])
            sym_wr = (sym_wins / len(sym_trades) * 100) if len(sym_trades) > 0 else 0
            sym_profit = sym_trades['profit'].sum()
            
            by_symbol[sym] = {
                "count": len(sym_trades),
                "wins": sym_wins,
                "win_rate": round(sym_wr, 2),
                "profit": round(sym_profit, 2),
                "avg_profit": round(sym_trades['profit'].mean(), 2),
            }
        
        return {
            "total_trades": total,
            "winning_trades": wins,
            "losing_trades": losses,
            "win_rate": round(win_rate, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_profit": round(net_profit, 2),
            "profit_factor": round(profit_factor, 3),
            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "largest_win": round(largest_win, 2),
            "largest_loss": round(largest_loss, 2),
            "consecutive_wins": max_consecutive_wins,
            "consecutive_losses": max_consecutive_losses,
            "average_bars_held": round(avg_bars, 1),
            "payoff_ratio": round(payoff_ratio, 3),
            "sharpe_ratio_approx": round(sharpe, 3),
            "by_confluence": by_confluence,
            "by_symbol": by_symbol,
        }

    @staticmethod
    def _max_consecutive(series, value):
        """Find maximum consecutive occurrences of a value in a series."""
        max_count = 0
        current_count = 0
        for v in series:
            if v == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count


# ============================================================================
# BACKTEST REPORTER
# ============================================================================

class BacktestReporter:
    """Generates detailed backtest reports."""
    
    @staticmethod
    def generate_report(results: dict) -> str:
        """Generate human-readable backtest report."""
        
        trades = results["trades"]
        metrics = results["metrics"]
        
        report = []
        report.append("=" * 80)
        report.append("FOREX BOT BACKTEST REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Header
        report.append(f"Period: {results['start_date']} to {results['end_date']} ({results['days']} days)")
        report.append(f"Symbols: {', '.join(results['symbols'])}")
        report.append("")
        
        # Key metrics
        report.append("KEY PERFORMANCE INDICATORS")
        report.append("-" * 80)
        report.append(f"Total Trades:              {metrics['total_trades']}")
        report.append(f"Winning Trades:            {metrics['winning_trades']}")
        report.append(f"Losing Trades:             {metrics['losing_trades']}")
        report.append(f"")
        report.append(f"Win Rate:                  {metrics['win_rate']:.2f}%  {'✓ GOOD' if metrics['win_rate'] >= 55 else '⚠ NEEDS IMPROVEMENT'}")
        report.append(f"Profit Factor:             {metrics['profit_factor']:.3f}  {'✓ GOOD' if metrics['profit_factor'] >= 1.5 else '⚠ NEEDS IMPROVEMENT'}")
        report.append(f"")
        report.append(f"Gross Profit:              ${metrics['gross_profit']:.2f}")
        report.append(f"Gross Loss:                ${metrics['gross_loss']:.2f}")
        report.append(f"Net Profit:                ${metrics['net_profit']:.2f}  {'✓ PROFITABLE' if metrics['net_profit'] > 0 else '✗ LOSS'}")
        report.append(f"")
        report.append(f"Average Win:               ${metrics['average_win']:.2f}")
        report.append(f"Average Loss:              ${metrics['average_loss']:.2f}")
        report.append(f"Payoff Ratio:              {metrics['payoff_ratio']:.3f}  (avg win / avg loss)")
        report.append(f"")
        report.append(f"Largest Win:               ${metrics['largest_win']:.2f}")
        report.append(f"Largest Loss:              ${metrics['largest_loss']:.2f}")
        report.append(f"")
        report.append(f"Consecutive Wins:          {metrics['consecutive_wins']}")
        report.append(f"Consecutive Losses:        {metrics['consecutive_losses']}")
        report.append(f"")
        report.append(f"Avg Bars Held:             {metrics['average_bars_held']:.1f}")
        report.append(f"Sharpe Ratio (approx):     {metrics['sharpe_ratio_approx']:.3f}")
        report.append("")
        
        # By Confluence
        report.append("PERFORMANCE BY CONFLUENCE SCORE")
        report.append("-" * 80)
        for conf_score in sorted(metrics['by_confluence'].keys(), reverse=True):
            data = metrics['by_confluence'][conf_score]
            report.append(
                f"  {conf_score}/5: {data['count']:3d} trades | "
                f"{data['wins']:2d} wins ({data['win_rate']:5.2f}%) | "
                f"Profit: ${data['profit']:10.2f}"
            )
        report.append("")
        
        # By Symbol
        report.append("PERFORMANCE BY SYMBOL")
        report.append("-" * 80)
        for symbol in sorted(metrics['by_symbol'].keys()):
            data = metrics['by_symbol'][symbol]
            report.append(
                f"  {symbol:10s}: {data['count']:3d} trades | "
                f"{data['wins']:2d} wins ({data['win_rate']:5.2f}%) | "
                f"Profit: ${data['profit']:10.2f}"
            )
        report.append("")
        
        # Recommendations
        report.append("ANALYSIS & RECOMMENDATIONS")
        report.append("-" * 80)
        
        if metrics['win_rate'] >= 60:
            report.append("✓ Excellent win rate! Ready for live trading.")
        elif metrics['win_rate'] >= 55:
            report.append("✓ Good win rate. Could start live trading with caution.")
        elif metrics['win_rate'] >= 50:
            report.append("⚠ Break-even range. Consider parameter tuning.")
        else:
            report.append("✗ Below break-even. Significant adjustments needed.")
        
        if metrics['profit_factor'] >= 2.0:
            report.append("✓ Excellent profit factor. Strategy is robust.")
        elif metrics['profit_factor'] >= 1.5:
            report.append("✓ Good profit factor. Strategy is profitable.")
        elif metrics['profit_factor'] >= 1.0:
            report.append("⚠ Marginal profit factor. Risk/reward needs improvement.")
        else:
            report.append("✗ Negative expectancy. Strategy is unprofitable.")
        
        # Confluence analysis
        best_confluence = max(metrics['by_confluence'].items(), 
                             key=lambda x: x[1]['win_rate'])[0]
        report.append(f"✓ Best results at {best_confluence}/5 confluence - prioritize these setups.")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Forex Bot Backtest")
    parser.add_argument("days", nargs="?", type=int, default=30,
                       help="Number of days of historical data (default: 30)")
    parser.add_argument("symbols", nargs="*", default=None,
                       help="Symbols to test (default: all from Config)")
    
    args = parser.parse_args()
    
    days = args.days
    symbols = args.symbols if args.symbols else Config.SYMBOLS
    
    logger.info(f"Backtesting {len(symbols)} symbols for {days} days...")
    
    # Initialize MT5
    cfg = Config()
    mt5_mgr = MT5Manager(cfg.LOGIN, cfg.PASSWORD, cfg.SERVER)
    
    if not mt5_mgr.connect():
        logger.error("Cannot connect to MT5")
        return
    
    # Run backtest
    engine = BacktestEngine(cfg, mt5_mgr)
    results = engine.run_backtest(symbols, days)
    
    # Generate report
    report = BacktestReporter.generate_report(results)
    print(report)
    
    # Save results
    try:
        report_file = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w") as f:
            f.write(report)
        logger.info(f"Report saved: {report_file}")
        
        # Save JSON for further analysis
        json_file = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, cls=SafeEncoder)
        logger.info(f"Detailed results saved: {json_file}")
        
    except Exception as exc:
        logger.error(f"Error saving report: {exc}")
    
    mt5_mgr.disconnect()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Backtest interrupted")
    except Exception:
        logger.error(traceback.format_exc())
