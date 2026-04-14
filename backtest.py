import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
from datetime import datetime, timedelta
import os
import logging
from typing import Optional, Tuple, Dict
import traceback

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Configure logging with timestamps and color coding."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# CONFIGURATION (Mirrored from bot.py)
# ============================================================================

class Config:
    SYMBOL = "EURUSDm"
    TIMEFRAME = mt5.TIMEFRAME_M15
    TIMEFRAME_NAME = "M15"
    H1_TIMEFRAME = mt5.TIMEFRAME_H1
    H1_TIMEFRAME_NAME = "H1"
    LOT_SIZE = 0.01
    
    EMA_FAST = 50
    EMA_SLOW = 200
    RSI_PERIOD = 14
    RSI_BUY_MIN = 50
    RSI_BUY_MAX = 65
    RSI_SELL_MIN = 35
    RSI_SELL_MAX = 50
    
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    BB_PERIOD = 20
    BB_STD_DEV = 2
    BB_MIN_WIDTH = 0.0005
    
    ATR_PERIOD = 14
    ADX_PERIOD = 14
    ADX_THRESHOLD = 25
    
    SL_MULTIPLIER = 1.0  
    TP_MULTIPLIER = 2.0  
    
    BE_ACTIVATION_ATR = 1.0
    TRAILING_STOP_ATR = 1.5
    
    START_HOUR = 8
    END_HOUR = 17
    
    LOGIN = 413646889
    PASSWORD = "Anoma@0822"
    SERVER = "Exness-MT5Trial6"
    
    MIN_CANDLES_REQUIRED = 200
    BACKTEST_CANDLES = 2000  # Number of candles to backtest
    HISTORY_EXCEL = "backtest_history.xlsx"
    ACTIVE_TRADES_JSON = "active_trades_features.json"


# ============================================================================
# TRADE HISTORY MANAGER (Adapted for Backtest)
# ============================================================================

class TradeHistoryManager:
    """Manages recording and saving simulated trade history to Excel."""
    
    def __init__(self, excel_file: str):
        """Initialize history manager."""
        self.excel_file = excel_file
        self.trades = []
        
    def record_trade(self, trade_data: dict):
        """Record a completed trade."""
        self.trades.append(trade_data)
        self._save_to_excel()
        
    def _save_to_excel(self):
        """Save trades to Excel file."""
        try:
            df_new = pd.DataFrame(self.trades)
            
            if os.path.exists(self.excel_file):
                df_existing = pd.read_excel(self.excel_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
                
            df_combined.to_excel(self.excel_file, index=False)
            logger.info(f"✓ Updated backtest history in {self.excel_file} with {len(self.trades)} trades")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

# ============================================================================
# BACKTEST TRADE MANAGER
# ============================================================================

class BacktestTradeManager:
    """Manages simulated trades in backtest."""
    
    def __init__(self, lot_size: float):
        """Initialize trade manager."""
        self.lot_size = lot_size
        self.active_trade = None
        
    def has_open_trade(self) -> bool:
        """Check if a trade is already open."""
        return self.active_trade is not None
    
    def open_trade(self, signal: str, entry_price: float, sl_price: float, tp_price: float, entry_time: str, indicators: dict):
        """Open a simulated trade."""
        if self.has_open_trade():
            return None
        
        self.active_trade = {
            'type': signal,
            'entry_time': entry_time,
            'entry_price': entry_price,
            'sl': sl_price,
            'tp': tp_price,
            'indicators': indicators
        }
        return self.active_trade
    
    def manage_open_trade(self, current_price: float, atr: float) -> None:
        """Manage open trade: break-even and trailing stop logic."""
        if not self.active_trade:
            return
        
        trade = self.active_trade
        pip_value = 0.0001  # For EURUSD
        
        if trade['type'] == 'BUY':
            profit_pips = (current_price - trade['entry_price']) / pip_value
            be_threshold = Config.BE_ACTIVATION_ATR * atr / pip_value

            # Move SL to break-even when profit >= 1.0 × ATR
            if profit_pips >= be_threshold and trade['sl'] < trade['entry_price']:
                trade['sl'] = trade['entry_price']
                logger.info(f"✓ Break-Even Set: SL moved to {trade['sl']:.5f}")

            # Trailing stop: keep SL at 1.5 × ATR below current price
            elif profit_pips > 0:
                trailing_sl = current_price - (Config.TRAILING_STOP_ATR * atr)
                # Only update if new SL is higher than current SL
                if trailing_sl > trade['sl']:
                    trade['sl'] = trailing_sl
                    logger.info(f"✓ Trailing Stop Updated: New SL: {trailing_sl:.5f}")

        else:  # SELL
            profit_pips = (trade['entry_price'] - current_price) / pip_value
            be_threshold = Config.BE_ACTIVATION_ATR * atr / pip_value

            # Move SL to break-even when profit >= 1.0 × ATR
            if profit_pips >= be_threshold and trade['sl'] > trade['entry_price']:
                trade['sl'] = trade['entry_price']
                logger.info(f"✓ Break-Even Set: SL moved to {trade['sl']:.5f}")

            # Trailing stop: keep SL at 1.5 × ATR above current price
            elif profit_pips > 0:
                trailing_sl = current_price + (Config.TRAILING_STOP_ATR * atr)
                # Only update if new SL is lower than current SL
                if trailing_sl < trade['sl']:
                    trade['sl'] = trailing_sl
                    logger.info(f"✓ Trailing Stop Updated: New SL: {trailing_sl:.5f}")
    
    def check_exit(self, high: float, low: float) -> Optional[dict]:
        """Check if trade should exit based on high/low."""
        if not self.active_trade:
            return None
        
        trade = self.active_trade
        hit_sl = False
        hit_tp = False
        
        if trade['type'] == 'BUY':
            if low <= trade['sl']:
                hit_sl = True
                exit_price = trade['sl']
            elif high >= trade['tp']:
                hit_tp = True
                exit_price = trade['tp']
        else:  # SELL
            if high >= trade['sl']:
                hit_sl = True
                exit_price = trade['sl']
            elif low <= trade['tp']:
                hit_tp = True
                exit_price = trade['tp']
        
        if hit_sl or hit_tp:
            # Close trade
            outcome = "WIN" if (hit_tp) else "LOSS"
            
            # Simple profit calculation in pips
            if trade['type'] == 'BUY':
                profit_pips = (exit_price - trade['entry_price']) / 0.0001
            else:
                profit_pips = (trade['entry_price'] - exit_price) / 0.0001
            
            # Approximate monetary profit
            profit_usd = profit_pips * 0.10
            
            closed_trade = {
                'type': trade['type'],
                'entry_time': trade['entry_time'],
                'exit_time': str(datetime.now()),  # Placeholder
                'entry_price': trade['entry_price'],
                'exit_price': exit_price,
                'profit_pips': profit_pips,
                'profit_usd': profit_usd,
                'outcome': outcome,
                **trade['indicators']
            }
            
            self.active_trade = None
            return closed_trade
        
        return None

# ============================================================================
# INDICATOR CALCULATOR (Simplified for Backtesting)
# ============================================================================

class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_all_indicators()
    
    def _calculate_all_indicators(self):
        self.df['EMA_50'] = ta.ema(self.df['close'], length=Config.EMA_FAST)
        self.df['EMA_200'] = ta.ema(self.df['close'], length=Config.EMA_SLOW)
        self.df['RSI'] = ta.rsi(self.df['close'], length=Config.RSI_PERIOD)
        
        macd_result = ta.macd(self.df['close'], fast=Config.MACD_FAST, slow=Config.MACD_SLOW, signal=Config.MACD_SIGNAL)
        if macd_result is not None and not macd_result.empty:
            macd_col = [c for c in macd_result.columns if c.startswith('MACD_') and 's' not in c and 'h' not in c]
            signal_col = [c for c in macd_result.columns if c.startswith('MACDs_')]
            hist_col = [c for c in macd_result.columns if c.startswith('MACDh_')]
            self.df['MACD'] = macd_result[macd_col[0]] if macd_col else np.nan
            self.df['MACD_Signal'] = macd_result[signal_col[0]] if signal_col else np.nan
            self.df['MACD_Hist'] = macd_result[hist_col[0]] if hist_col else np.nan
        
        bb_result = ta.bbands(self.df['close'], length=Config.BB_PERIOD, std=Config.BB_STD_DEV)
        if bb_result is not None and not bb_result.empty:
            upper_col = [c for c in bb_result.columns if c.startswith('BBU_')]
            middle_col = [c for c in bb_result.columns if c.startswith('BBM_')]
            lower_col = [c for c in bb_result.columns if c.startswith('BBL_')]
            self.df['BB_Upper'] = bb_result[upper_col[0]] if upper_col else np.nan
            self.df['BB_Middle'] = bb_result[middle_col[0]] if middle_col else np.nan
            self.df['BB_Lower'] = bb_result[lower_col[0]] if lower_col else np.nan
            
        self.df['ATR'] = ta.atr(self.df['high'], self.df['low'], self.df['close'], length=Config.ATR_PERIOD)
        
        # ADX
        adx_res = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=Config.ADX_PERIOD)
        if adx_res is not None and not adx_res.empty:
            adx_col = [c for c in adx_res.columns if c.startswith('ADX_')]
            self.df['ADX'] = adx_res[adx_col[0]] if adx_col else np.nan
        else:
            self.df['ADX'] = np.nan
    
    def get_latest_values(self) -> Dict:
        """Get latest indicator values from the last candle."""
        if self.df.empty:
            return {}
            
        last_idx = len(self.df) - 1
        
        def safe_get(col):
            if col in self.df.columns:
                val = self.df[col].iloc[last_idx]
                return val if not pd.isna(val) else np.nan
            return np.nan

        return {
            'close': safe_get('close'),
            'high': safe_get('high'),
            'low': safe_get('low'),
            'ema_50': safe_get('EMA_50'),
            'ema_200': safe_get('EMA_200'),
            'rsi': safe_get('RSI'),
            'macd': safe_get('MACD'),
            'macd_signal': safe_get('MACD_Signal'),
            'macd_hist': safe_get('MACD_Hist'),
            'bb_upper': safe_get('BB_Upper'),
            'bb_middle': safe_get('BB_Middle'),
            'bb_lower': safe_get('BB_Lower'),
            'atr': safe_get('ATR'),
            'adx': safe_get('ADX'),
        }

# ============================================================================
# SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """Generates trading signals based on advanced logic."""
    
    @staticmethod
    def check_trend_direction(indicators: dict) -> tuple:
        close = indicators['close']
        ema_50 = indicators['ema_50']
        ema_200 = indicators['ema_200']
        if pd.isna(ema_50) or pd.isna(ema_200): return 'NEUTRAL', False
        if close > ema_50 and ema_50 > ema_200: return 'UP', True
        elif close < ema_50 and ema_50 < ema_200: return 'DOWN', True
        return 'NEUTRAL', False
    
    @staticmethod
    def check_rsi_filter(indicators: dict, trend_dir: str) -> tuple:
        rsi = indicators['rsi']
        if pd.isna(rsi): return 'INVALID', False
        if trend_dir == 'UP' and Config.RSI_BUY_MIN < rsi < Config.RSI_BUY_MAX:
            return 'VALID (BUY)', True
        if trend_dir == 'DOWN' and Config.RSI_SELL_MIN < rsi < Config.RSI_SELL_MAX:
            return 'VALID (SELL)', True
        return f'INVALID ({rsi:.1f})', False
    
    @staticmethod
    def check_macd_confirmation(indicators: dict, trend_dir: str) -> tuple:
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        if pd.isna(macd) or pd.isna(macd_signal): return 'INVALID', False
        if trend_dir == 'UP' and macd > macd_signal and macd > 0: return 'CONFIRM', True
        elif trend_dir == 'DOWN' and macd < macd_signal and macd < 0: return 'CONFIRM', True
        return 'REJECT', False
    
    @staticmethod
    def check_adx_filter(indicators: dict) -> tuple:
        adx = indicators.get('adx', np.nan)
        if pd.isna(adx): return 'INVALID', False
        if adx > Config.ADX_THRESHOLD: return 'STRONG', True
        return 'WEAK', False
    
    @staticmethod
    def check_bollinger_bands_filter(indicators: dict) -> tuple:
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        if pd.isna(bb_upper) or pd.isna(bb_lower): return 'INVALID', False
        if (bb_upper - bb_lower) >= Config.BB_MIN_WIDTH: return 'VALID', True
        return 'SQUEEZE', False
    
    @staticmethod
    def check_atr_validity(indicators: dict) -> tuple:
        atr = indicators['atr']
        if pd.isna(atr) or atr <= 0: return 'INVALID', False
        return 'VALID', True

    @staticmethod
    def check_mtf_filter(h1_data: dict, trend_dir: str) -> tuple:
        h1_close = h1_data.get('close', np.nan)
        h1_ema200 = h1_data.get('ema_200', np.nan)
        if pd.isna(h1_close) or pd.isna(h1_ema200): return 'INVALID', False
        if trend_dir == 'UP' and h1_close > h1_ema200: return 'UPTREND (H1)', True
        if trend_dir == 'DOWN' and h1_close < h1_ema200: return 'DOWNTREND (H1)', True
        return 'MISMATCH', False

    @staticmethod
    def check_session_filter(hour: int) -> tuple:
        if Config.START_HOUR <= hour < Config.END_HOUR:
            return 'IN SESSION', True
        return 'OUT OF SESSION', False

    @staticmethod
    def check_engulfing_pattern(df, trend_dir: str) -> tuple:
        if len(df) < 3: return 'INVALID', False
        prev = df.iloc[-3]
        curr = df.iloc[-2]
        if trend_dir == 'UP':
            prev_bear = prev['close'] < prev['open']
            curr_bull = curr['close'] > curr['open']
            engulfing = curr['close'] > prev['open'] and curr['open'] < prev['close']
            if prev_bear and curr_bull and engulfing: return 'BULLISH ENGULFING', True
        elif trend_dir == 'DOWN':
            prev_bull = prev['close'] > prev['open']
            curr_bear = curr['close'] < curr['open']
            engulfing = curr['close'] < prev['open'] and curr['open'] > prev['close']
            if prev_bull and curr_bear and engulfing: return 'BEARISH ENGULFING', True
        return 'NO PATTERN', True  

    @staticmethod
    def generate_signal(indicators: dict, h1_data: dict, hour: int, df) -> tuple:
        signal_details = {}
        
        trend_dir, trend_valid = SignalGenerator.check_trend_direction(indicators)
        signal_details['trend'] = (trend_dir, trend_valid)
        signal_details['rsi'] = SignalGenerator.check_rsi_filter(indicators, trend_dir)
        signal_details['macd'] = SignalGenerator.check_macd_confirmation(indicators, trend_dir)
        signal_details['adx'] = SignalGenerator.check_adx_filter(indicators)
        signal_details['bb'] = SignalGenerator.check_bollinger_bands_filter(indicators)
        signal_details['atr'] = SignalGenerator.check_atr_validity(indicators)
        signal_details['mtf'] = SignalGenerator.check_mtf_filter(h1_data, trend_dir)
        signal_details['session'] = SignalGenerator.check_session_filter(hour)
        eng_msg, eng_valid = SignalGenerator.check_engulfing_pattern(df, trend_dir)
        signal_details['engulfing'] = (eng_msg, eng_valid)
        
        all_conditions_met = (
            trend_valid and 
            signal_details['rsi'][1] and 
            signal_details['macd'][1] and 
            signal_details['adx'][1] and 
            signal_details['bb'][1] and 
            signal_details['atr'][1] and 
            signal_details['mtf'][1] and 
            signal_details['session'][1]
        )
        
        if all_conditions_met:
            if trend_dir == 'UP': return 'BUY', signal_details
            elif trend_dir == 'DOWN': return 'SELL', signal_details
            
        return 'HOLD', signal_details

# ============================================================================
# BACKTEST BOT CLASS
# ============================================================================

class BacktestBot:
    """Main backtest orchestrator."""
    
    def __init__(self, config):
        """Initialize the backtest bot."""
        self.config = config
        self.trade_manager = BacktestTradeManager(config.LOT_SIZE)
        self.history_manager = TradeHistoryManager(config.HISTORY_EXCEL)
        self.balance = 10000.0
        self.trades = []
        
    def run_backtest(self):
        """Run the backtest simulation."""
        logger.info("Starting Backtest...")
        
        if not mt5.initialize():
            logger.error("MT5 Initialization failed")
            return
        
        # Login to account
        if not mt5.login(self.config.LOGIN, self.config.PASSWORD, self.config.SERVER):
            logger.error("Failed to login to MT5 account")
            logger.error(f"Error: {mt5.last_error()}")
            mt5.shutdown()
            return
        
        logger.info(f"Logged into {self.config.SERVER}")
        
        # Fetch M15 candles
        rates = mt5.copy_rates_from_pos(self.config.SYMBOL, self.config.TIMEFRAME, 0, self.config.BACKTEST_CANDLES)
        if rates is None:
            logger.error(f"Failed to fetch M15 data for {self.config.SYMBOL}")
            mt5.shutdown()
            return
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Fetch H1 candles for MTF
        h1_rates = mt5.copy_rates_from_pos(self.config.SYMBOL, self.config.H1_TIMEFRAME, 0, self.config.BACKTEST_CANDLES // 4)
        if h1_rates is not None:
            df_h1 = pd.DataFrame(h1_rates)
            df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
            df_h1['EMA_200'] = ta.ema(df_h1['close'], length=200)
        else:
            df_h1 = pd.DataFrame()
            logger.warning("Failed to fetch H1 data, MTF filter will be invalid")
        
        # Calculate indicators
        calc = IndicatorCalculator(df)
        df = calc.df
        
        logger.info(f"Backtesting {len(df)} candles from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
        
        # Iterate through candles
        for i in range(self.config.MIN_CANDLES_REQUIRED, len(df)):
            row = df.iloc[i]
            
            # Get indicators for this candle
            indicators = {
                'close': row['close'],
                'high': row['high'],
                'low': row['low'],
                'ema_50': row['EMA_50'],
                'ema_200': row['EMA_200'],
                'rsi': row['RSI'],
                'macd': row['MACD'],
                'macd_signal': row['MACD_Signal'],
                'macd_hist': row['MACD_Hist'],
                'bb_upper': row['BB_Upper'],
                'bb_middle': row['BB_Middle'],
                'bb_lower': row['BB_Lower'],
                'atr': row['ATR'],
                'adx': row['ADX'],
            }
            
            # Get H1 data (approximate by index)
            h1_data = {'close': np.nan, 'ema_200': np.nan}
            if not df_h1.empty:
                h1_idx = min(i // 4, len(df_h1) - 1)
                h1_row = df_h1.iloc[h1_idx]
                h1_data['close'] = h1_row['close']
                h1_data['ema_200'] = h1_row['EMA_200']
            
            # Get hour
            hour = row['time'].hour
            
            # Generate signal
            signal, details = SignalGenerator.generate_signal(indicators, h1_data, hour, df.iloc[:i+1])
            
            # If no open trade and signal is BUY/SELL, open trade
            if not self.trade_manager.has_open_trade() and signal in ['BUY', 'SELL']:
                atr = indicators['atr']
                entry_price = indicators['close']
                
                if signal == 'BUY':
                    sl_price = entry_price - (atr * self.config.SL_MULTIPLIER)
                    tp_price = entry_price + (atr * self.config.TP_MULTIPLIER)
                else:
                    sl_price = entry_price + (atr * self.config.SL_MULTIPLIER)
                    tp_price = entry_price - (atr * self.config.TP_MULTIPLIER)
                
                trade = self.trade_manager.open_trade(signal, entry_price, sl_price, tp_price, str(row['time']), indicators)
                if trade:
                    logger.info(f"Opened {signal} trade at {entry_price:.5f}, SL: {sl_price:.5f}, TP: {tp_price:.5f}")
            
            # Manage open trade
            if self.trade_manager.has_open_trade():
                self.trade_manager.manage_open_trade(indicators['close'], indicators['atr'])
                
                # Check for exit
                closed_trade = self.trade_manager.check_exit(row['high'], row['low'])
                if closed_trade:
                    closed_trade['exit_time'] = str(row['time'])
                    self.balance += closed_trade['profit_usd']
                    self.trades.append(closed_trade)
                    self.history_manager.record_trade(closed_trade)
                    logger.info(f"Closed {closed_trade['type']} trade: {closed_trade['outcome']}, P/L: {closed_trade['profit_usd']:.2f} USD")
        
        # Calculate metrics
        total_trades = len(self.trades)
        wins = len([t for t in self.trades if t['outcome'] == "WIN"])
        losses = total_trades - wins
        accuracy = (wins / total_trades * 100) if total_trades > 0 else 0
        
        total_profit_usd = sum([t['profit_usd'] for t in self.trades])
        
        results = {
            "summary": {
                "symbol": self.config.SYMBOL,
                "period": self.config.TIMEFRAME_NAME,
                "candles_analyzed": len(df),
                "total_trades": total_trades,
                "wins": wins,
                "losses": losses,
                "accuracy_percent": round(accuracy, 2),
                "total_profit_usd": round(total_profit_usd, 2),
                "final_balance": round(self.balance, 2),
                "start_time": str(df['time'].iloc[0]),
                "end_time": str(df['time'].iloc[-1])
            },
            "trades": self.trades
        }
        
        # Output as JSON
        with open("backtest_results.json", "w") as f:
            json.dump(results, f, indent=4)
        
        logger.info("\n--- BACKTEST COMPLETE ---")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        logger.info(f"Profit/Loss: {total_profit_usd:.2f} USD")
        logger.info("Results saved to backtest_results.json")
        
        mt5.shutdown()
        return results

if __name__ == "__main__":
    bot = BacktestBot(Config)
    bot.run_backtest()
