"""
================================================================================
EXNESS EUR/USD TREND-CONFIRMATION ALGORITHMIC TRADING BOT
================================================================================
Production-Ready High-Frequency Trading System

Strategy: Multi-Indicator Trend Confirmation (5-Indicator Alignment)
Indicators: EMA 200/50, RSI(14), MACD(12,26,9), Bollinger Bands(20,2), ATR(14)
Timeframe: M15 (15-minute candles)
Symbol: EURUSD
Risk per Trade: 1% (Fixed lot 0.01)
Risk-to-Reward: 1:2 (SL: 1.5xATR, TP: 3xATR)

Author: Senior Quantitative Developer
Date: 2024
================================================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict
import traceback
import json
import os

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
# CONFIGURATION SECTION
# ============================================================================

class Config:
    """Trading configuration constants."""
    
    # MT5 Connection
    LOGIN = 413646889
    PASSWORD = "Anoma@0822"
    SERVER = "Exness-MT5Trial6"
    
    # Trading Parameters
    SYMBOL = "EURUSDm"
    TIMEFRAME = mt5.TIMEFRAME_M15
    TIMEFRAME_NAME = "M15"
    H1_TIMEFRAME = mt5.TIMEFRAME_H1
    H1_TIMEFRAME_NAME = "H1"
    LOT_SIZE = 0.01
    
    # Session Control
    START_HOUR = 8
    END_HOUR = 17
    
    # Indicator Parameters
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
    
    # Risk Management
    MAX_SPREAD = 5
    LOOP_INTERVAL = 60
    MIN_CANDLES_REQUIRED = 200
    
    # Position Management
    MAX_OPEN_TRADES = 1
    
    # History Management
    HISTORY_EXCEL = "trade_history.xlsx"
    ACTIVE_TRADES_JSON = "active_trades_features.json"


# ============================================================================
# TRADE HISTORY MANAGER
# ============================================================================

class TradeHistoryManager:
    """Manages recording and saving trade history to Excel with indicators."""
    
    def __init__(self, excel_file: str, features_file: str):
        """Initialize history manager."""
        self.excel_file = excel_file
        self.features_file = features_file
        self.active_features = self._load_features()
        
    def _load_features(self) -> Dict:
        """Load features of active trades from JSON."""
        if os.path.exists(self.features_file):
            try:
                with open(self.features_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading features JSON: {e}")
                return {}
        return {}

    def _save_features(self):
        """Save features of active trades to JSON."""
        try:
            with open(self.features_file, "w") as f:
                json.dump(self.active_features, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving features JSON: {e}")

    def record_entry(self, ticket: int, signal: str, indicators: Dict):
        """Record indicators at entry for a specific ticket."""
        self.active_features[str(ticket)] = {
            "ticket": ticket,
            "type": signal,
            "entry_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **indicators
        }
        self._save_features()
        logger.info(f"Recorded features for trade entry: Ticket {ticket}")

    def process_history(self, deals: list):
        """Process closed deals and merge with recorded entry features."""
        if not deals:
            return
            
        new_entries = []
        for deal in deals:
            # We only care about deals that close a position (DEAL_ENTRY_OUT)
            if deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
                
            pos_id = str(deal.position_id)
            if pos_id in self.active_features:
                entry_data = self.active_features.pop(pos_id)
                
                # Create history row
                history_row = {
                    **entry_data,
                    "exit_time": datetime.fromtimestamp(deal.time).strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_price": deal.price,
                    "profit": deal.profit,
                    "commission": deal.commission,
                    "swap": deal.swap,
                    "outcome": "WIN" if deal.profit > 0 else "LOSS"
                }
                new_entries.append(history_row)
        
        if new_entries:
            self._save_to_excel(new_entries)
            self._save_features()

    def _save_to_excel(self, new_entries: list):
        """Save/Append entries to Excel file."""
        try:
            df_new = pd.DataFrame(new_entries)
            
            if os.path.exists(self.excel_file):
                df_existing = pd.read_excel(self.excel_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new
                
            df_combined.to_excel(self.excel_file, index=False)
            logger.info(f"✓ Updated trade history in {self.excel_file} with {len(new_entries)} new entries")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {e}")

    def get_recent_history(self, limit: int = 15) -> list:
        """Get recent trade history from Excel."""
        if os.path.exists(self.excel_file):
            try:
                # Use openpyxl engine specifically if needed
                df = pd.read_excel(self.excel_file)
                if df.empty:
                    return []
                # Ensure we handle NaN values for JSON serialization
                df = df.fillna("")
                # Take last N rows and convert to list of dicts
                recent = df.tail(limit).to_dict('records')
                # Reverse to show newest first
                recent.reverse()
                return recent
            except Exception as e:
                logger.error(f"Error reading history for export: {e}")
                return []
        return []

# ============================================================================
# TRADE MANAGER CLASS
# ============================================================================


class TradeManager:
    """Manages trade execution and position tracking."""
    
    def __init__(self, symbol: str, lot_size: float):
        """Initialize trade manager."""
        self.symbol = symbol
        self.lot_size = lot_size
        self.open_trades = []
        self.last_candle_time = None
        
    def get_open_trades(self) -> list:
        """Fetch open trades from MT5."""
        try:
            positions = mt5.positions_get(symbol=self.symbol)
            if positions:
                return list(positions)
            return []
        except Exception as e:
            logger.error(f"Error fetching open trades: {e}")
            return []
    
    def has_open_trade(self) -> bool:
        """Check if a trade is already open."""
        return len(self.get_open_trades()) > 0
    
    def get_account_balance(self) -> float:
        """Get current account balance."""
        try:
            account_info = mt5.account_info()
            if account_info:
                return account_info.balance
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return 0.0
    
    def calculate_lot_size(self, symbol_info) -> float:
        """Calculate lot size based on risk management."""
        return self.lot_size
    
    def get_filling_mode(self) -> int:
        """Dynamically determine the supported filling mode for the symbol."""
        try:
            symbol_info = mt5.symbol_info(self.symbol)
            if not symbol_info:
                return mt5.ORDER_FILLING_IOC
                
            filling = symbol_info.filling_mode
            if filling & mt5.SYMBOL_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            elif filling & mt5.SYMBOL_FILLING_FOK:
                return mt5.ORDER_FILLING_FOK
            else:
                return mt5.ORDER_FILLING_RETURN
        except Exception:
            return mt5.ORDER_FILLING_IOC
    
    def execute_buy_order(self, entry_price: float, sl_price: float, 
                          tp_price: float, order_comment: str = "") -> Optional[int]:
        """Execute a BUY order with SL and TP. Returns ticket if successful."""
        try:
            if self.has_open_trade():
                logger.warning("Trade already open. Skipping new BUY order.")
                return None

            
            symbol_info = mt5.symbol_info(self.symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {self.symbol}")
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select {self.symbol}")
                    return False
            
            # Determine filling mode
            filling_mode = self.get_filling_mode()
            
            # Create buy order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_BUY,
                "price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 99991,
                "comment": f"BUY - {order_comment}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"BUY order failed. Retcode: {result.retcode}")
                logger.error(f"Retcode description: {mt5.last_error()}")
                return None
            
            logger.info(f"✓ BUY ORDER EXECUTED")
            logger.info(f"  Entry: {entry_price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
            logger.info(f"  Lot Size: {self.lot_size} | Ticket: {result.order}")
            return result.order

            
        except Exception as e:
            logger.error(f"Exception in execute_buy_order: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def _modify_position_sl(self, ticket: int, new_sl: float, tp: float) -> bool:
        """Modify the stop loss of an open position."""
        try:
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
                "tp": tp,
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to modify SL for ticket {ticket}. Retcode: {result.retcode}")
                return False
            return True
        except Exception as e:
            logger.error(f"Exception modifying SL for ticket {ticket}: {e}")
            return False

    def manage_open_trades(self, current_price: float, atr: float):
        """Apply break-even and trailing stop logic to all open positions."""
        if atr <= 0 or pd.isna(atr):
            return

        positions = self.get_open_trades()
        if not positions:
            return

        for pos in positions:
            entry = pos.price_open
            current_sl = pos.sl
            tp = pos.tp
            ticket = pos.ticket
            is_buy = pos.type == mt5.ORDER_TYPE_BUY

            if is_buy:
                be_threshold = entry + Config.BE_ACTIVATION_ATR * atr
                be_active = current_sl >= entry  # SL at or above entry means BE was already applied

                if not be_active and current_price >= be_threshold:
                    # First time: activate break-even
                    if self._modify_position_sl(ticket, entry, tp):
                        logger.info(f"[BE] Ticket {ticket}: SL moved to break-even ({entry:.5f})")
                    be_active = True

                if be_active:
                    # Trailing stop: trail 1.5×ATR below current price
                    new_sl = current_price - Config.TRAILING_STOP_ATR * atr
                    if new_sl > current_sl:
                        if self._modify_position_sl(ticket, new_sl, tp):
                            logger.info(f"[TS] Ticket {ticket}: Trailing SL moved to {new_sl:.5f}")

            else:  # SELL
                be_threshold = entry - Config.BE_ACTIVATION_ATR * atr
                be_active = current_sl <= entry  # SL at or below entry means BE was already applied

                if not be_active and current_price <= be_threshold:
                    # First time: activate break-even
                    if self._modify_position_sl(ticket, entry, tp):
                        logger.info(f"[BE] Ticket {ticket}: SL moved to break-even ({entry:.5f})")
                    be_active = True

                if be_active:
                    # Trailing stop: trail 1.5×ATR above current price
                    new_sl = current_price + Config.TRAILING_STOP_ATR * atr
                    if new_sl < current_sl:
                        if self._modify_position_sl(ticket, new_sl, tp):
                            logger.info(f"[TS] Ticket {ticket}: Trailing SL moved to {new_sl:.5f}")

    def execute_sell_order(self, entry_price: float, sl_price: float, 
                           tp_price: float, order_comment: str = "") -> Optional[int]:
        """Execute a SELL order with SL and TP. Returns ticket if successful."""
        try:
            if self.has_open_trade():
                logger.warning("Trade already open. Skipping new SELL order.")
                return None

            
            symbol_info = mt5.symbol_info(self.symbol)
            if not symbol_info:
                logger.error(f"Failed to get symbol info for {self.symbol}")
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(self.symbol, True):
                    logger.error(f"Failed to select {self.symbol}")
                    return False
            
            # Determine filling mode
            filling_mode = self.get_filling_mode()
            
            # Create sell order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": self.lot_size,
                "type": mt5.ORDER_TYPE_SELL,
                "price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": 99992,
                "comment": f"SELL - {order_comment}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"SELL order failed. Retcode: {result.retcode}")
                logger.error(f"Retcode description: {mt5.last_error()}")
                return None
            
            logger.info(f"✓ SELL ORDER EXECUTED")
            logger.info(f"  Entry: {entry_price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
            logger.info(f"  Lot Size: {self.lot_size} | Ticket: {result.order}")
            return result.order

            
        except Exception as e:
            logger.error(f"Exception in execute_sell_order: {e}")
            logger.error(traceback.format_exc())
            return False

# ============================================================================
# INDICATOR CALCULATOR CLASS
# ============================================================================

class IndicatorCalculator:
    """Calculates all technical indicators."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with OHLCV dataframe."""
        self.df = df.copy()
        self._calculate_all_indicators()
    
    def _calculate_all_indicators(self):
        """Calculate all indicators on the dataframe."""
        # EMA 50 and EMA 200
        self.df['EMA_50'] = ta.ema(self.df['close'], length=Config.EMA_FAST)
        self.df['EMA_200'] = ta.ema(self.df['close'], length=Config.EMA_SLOW)
        
        # RSI
        self.df['RSI'] = ta.rsi(self.df['close'], length=Config.RSI_PERIOD)
        
        # MACD
        macd_result = ta.macd(
            self.df['close'],
            fast=Config.MACD_FAST,
            slow=Config.MACD_SLOW,
            signal=Config.MACD_SIGNAL
        )
        
        if macd_result is not None and not macd_result.empty:
            # Flexible column matching for MACD
            macd_col = [c for c in macd_result.columns if c.startswith('MACD_') and 's' not in c and 'h' not in c]
            signal_col = [c for c in macd_result.columns if c.startswith('MACDs_')]
            hist_col = [c for c in macd_result.columns if c.startswith('MACDh_')]
            
            self.df['MACD'] = macd_result[macd_col[0]] if macd_col else np.nan
            self.df['MACD_Signal'] = macd_result[signal_col[0]] if signal_col else np.nan
            self.df['MACD_Hist'] = macd_result[hist_col[0]] if hist_col else np.nan
        else:
            self.df['MACD'] = np.nan
            self.df['MACD_Signal'] = np.nan
            self.df['MACD_Hist'] = np.nan
        
        # Bollinger Bands
        bb_result = ta.bbands(
            self.df['close'],
            length=Config.BB_PERIOD,
            std=Config.BB_STD_DEV
        )
        if bb_result is not None and not bb_result.empty:
            upper_col = [c for c in bb_result.columns if c.startswith('BBU_')]
            middle_col = [c for c in bb_result.columns if c.startswith('BBM_')]
            lower_col = [c for c in bb_result.columns if c.startswith('BBL_')]
            
            self.df['BB_Upper'] = bb_result[upper_col[0]] if upper_col else np.nan
            self.df['BB_Middle'] = bb_result[middle_col[0]] if middle_col else np.nan
            self.df['BB_Lower'] = bb_result[lower_col[0]] if lower_col else np.nan
        else:
            self.df['BB_Upper'] = np.nan
            self.df['BB_Middle'] = np.nan
            self.df['BB_Lower'] = np.nan
        
        # ATR
        self.df['ATR'] = ta.atr(
            self.df['high'],
            self.df['low'],
            self.df['close'],
            length=Config.ATR_PERIOD
        )
        
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
# SIGNAL GENERATOR CLASS
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
# MT5 CONNECTION MANAGER
# ============================================================================

class MT5Manager:
    """Manages MetaTrader5 connection and data retrieval."""
    
    def __init__(self, login: int, password: str, server: str):
        """Initialize MT5 manager."""
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to MT5 with robust error handling."""
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error("Failed to initialize MetaTrader5")
                logger.error(f"Error: {mt5.last_error()}")
                return False
            
            logger.info("✓ MetaTrader5 initialized successfully")
            
            # Login to account
            if not mt5.login(self.login, self.password, self.server):
                logger.error("Failed to login to MT5 account")
                logger.error(f"Error: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            account_info = mt5.account_info()
            logger.info(f"✓ Successfully logged into account {self.login}")
            logger.info(f"  Broker: {account_info.company}")
            logger.info(f"  Balance: {account_info.balance:.2f}")
            logger.info(f"  Equity: {account_info.equity:.2f}")
            logger.info(f"  Leverage: 1:{account_info.leverage}")
            
            self.connected = True
            return True
            
        except Exception as e:
            logger.error(f"Exception during MT5 connection: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def disconnect(self):
        """Disconnect from MT5."""
        try:
            mt5.shutdown()
            self.connected = False
            logger.info("✓ MetaTrader5 disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting from MT5: {e}")
    
    def find_symbol(self, symbol: str) -> Optional[str]:
        """Find the correct symbol name with possible suffixes (Exness specific)."""
        # Try exact match first
        if mt5.symbol_info(symbol):
            return symbol
            
        # Common suffixes: m, !, #, c, k, .
        base_symbol = symbol.replace("m", "").replace("!", "").replace("#", "").replace("c", "")
        suffixes = ["", "m", "!", "#", "c", "k", "pro", "raw"]
        
        for suffix in suffixes:
            test_symbol = base_symbol + suffix
            if mt5.symbol_info(test_symbol):
                logger.info(f"Found symbol variant: {test_symbol}")
                return test_symbol
                
        return None

    def fetch_candles(self, symbol: str, timeframe, count: int) -> Optional[pd.DataFrame]:
        """Fetch candles from MT5."""
        try:
            if not self.connected:
                logger.error("MT5 not connected. Cannot fetch candles.")
                return None
            
            # Find correct symbol variant
            actual_symbol = self.find_symbol(symbol)
            if not actual_symbol:
                logger.error(f"Could not find symbol {symbol} or any variant.")
                return None
            
            # Select symbol
            if not mt5.symbol_select(actual_symbol, True):
                logger.error(f"Failed to select symbol {actual_symbol}")
                return None
            
            # Fetch rates
            rates = mt5.copy_rates_from_pos(actual_symbol, timeframe, 0, count)
            
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to fetch candles for {symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'
            })
            
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            logger.error(f"Exception fetching candles: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def fetch_history_deals(self, days: int = 1) -> list:
        """Fetch history deals for the last X days."""
        try:
            from_date = datetime.now() - timedelta(days=days)
            deals = mt5.history_deals_get(from_date, datetime.now())
            if deals is None:
                return []
            return list(deals)
        except Exception as e:
            logger.error(f"Error fetching history deals: {e}")
            return []

    
    def get_spread(self, symbol: str) -> float:
        """Get current spread in pips."""
        try:
            actual_symbol = self.find_symbol(symbol)
            if not actual_symbol:
                return float('inf')
                
            symbol_info = mt5.symbol_info(actual_symbol)
            if symbol_info:
                # spread is in points. Convert to pips.
                # Usually points / 10 for 5-digit brokers
                digits = symbol_info.digits
                spread_points = symbol_info.spread
                
                if digits == 3 or digits == 5:
                    return spread_points / 10.0
                else:
                    return float(spread_points)
            return float('inf')
        except Exception as e:
            logger.error(f"Error getting spread: {e}")
            return float('inf')
    
    def get_symbol_info(self, symbol: str) -> Optional:
        """Get symbol information."""
        try:
            return mt5.symbol_info(symbol)
        except Exception as e:
            logger.error(f"Error getting symbol info: {e}")
            return None

# ============================================================================
# MAIN TRADING BOT CLASS
# ============================================================================

class TrendConfirmationBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config: Config):
        """Initialize the bot."""
        self.config = config
        self.mt5 = MT5Manager(config.LOGIN, config.PASSWORD, config.SERVER)
        self.trade_manager = TradeManager(config.SYMBOL, config.LOT_SIZE)
        self.history_manager = TradeHistoryManager(config.HISTORY_EXCEL, config.ACTIVE_TRADES_JSON)
        self.last_signal_time = None
        self.is_running = False
        self.data_file = "bot_data.json"

    
    def export_data(self, indicators: Dict, signal: str, details: Dict):
        """Export current bot state for the dashboard."""
        try:
            account_info = mt5.account_info()
            positions = mt5.positions_get(symbol=self.config.SYMBOL)
            
            data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "account": {
                    "balance": account_info.balance if account_info else 0,
                    "equity": account_info.equity if account_info else 0,
                    "company": account_info.company if account_info else "Unknown",
                    "currency": account_info.currency if account_info else "USD",
                },
                "config": {
                    "symbol": self.config.SYMBOL,
                    "timeframe": self.config.TIMEFRAME_NAME,
                    "h1_data": h1_data if 'h1_data' in locals() else {},
                    "lot_size": self.config.LOT_SIZE,
                },
                "indicators": indicators,
                "signal": {
                    "type": signal,
                    "details": {k: [v[0], bool(v[1])] for k, v in details.items()}
                },
                "open_trades": [],
                "history": self.history_manager.get_recent_history(20)
            }
            
            if positions:
                for p in positions:
                    data["open_trades"].append({
                        "ticket": p.ticket,
                        "type": "BUY" if p.type == 0 else "SELL",
                        "volume": p.volume,
                        "price_open": p.price_open,
                        "price_current": p.price_current,
                        "profit": p.profit,
                        "sl": p.sl,
                        "tp": p.tp
                    })
            
            # Save to temporary file first then rename to prevent corruption
            temp_file = self.data_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=4)
            
            if os.path.exists(self.data_file):
                os.remove(self.data_file)
            os.rename(temp_file, self.data_file)
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
    
    def log_indicator_values(self, indicators: Dict):
        """Log all indicator values in a formatted way."""
        logger.info("=" * 80)
        logger.info("INDICATOR VALUES (Latest Candle)")
        logger.info("=" * 80)
        logger.info(f"Price (Close): {indicators['close']:.5f}")
        logger.info(f"EMA 50:        {indicators['ema_50']:.5f} | EMA 200: {indicators['ema_200']:.5f}")
        logger.info(f"RSI(14):       {indicators['rsi']:.2f}")
        logger.info(f"MACD:          {indicators['macd']:.6f} | Signal: {indicators['macd_signal']:.6f} | Histogram: {indicators['macd_hist']:.6f}")
        logger.info(f"BB Upper:      {indicators['bb_upper']:.5f} | Middle: {indicators['bb_middle']:.5f} | Lower: {indicators['bb_lower']:.5f}")
        logger.info(f"ATR(14):       {indicators['atr']:.5f}")
        logger.info("=" * 80)
    
    def log_signal_analysis(self, signal: str, details: Dict):
        """Log signal generation analysis."""
        logger.info("=" * 80)
        logger.info("SIGNAL ANALYSIS")
        logger.info("=" * 80)
        
        trend_status = "✓" if details['trend'][1] else "✗"
        logger.info(f"{trend_status} Trend (EMA): {details['trend'][0]}")
        
        rsi_status = "✓" if details['rsi'][1] else "✗"
        logger.info(f"{rsi_status} RSI Filter: {details['rsi'][0]}")
        
        macd_status = "✓" if details['macd'][1] else "✗"
        logger.info(f"{macd_status} MACD: {details['macd'][0]}")
        
        bb_status = "✓" if details['bb'][1] else "✗"
        logger.info(f"{bb_status} Bollinger Bands: {details['bb'][0]}")
        
        atr_status = "✓" if details['atr'][1] else "✗"
        logger.info(f"{atr_status} ATR: {details['atr'][0]}")
        
        logger.info("-" * 80)
        logger.info(f"FINAL SIGNAL: {signal}")
        logger.info("=" * 80)
    
    def run(self):
        """Main trading loop."""
        logger.info("\n" + "=" * 80)
        logger.info("EXNESS EUR/USD TREND-CONFIRMATION BOT STARTING")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Symbol: {self.config.SYMBOL} | Timeframe: {self.config.TIMEFRAME_NAME}")
        logger.info(f"  Lot Size: {self.config.LOT_SIZE}")
        logger.info(f"  Loop Interval: {self.config.LOOP_INTERVAL}s")
        logger.info("=" * 80 + "\n")
        
        # Connect to MT5
        if not self.mt5.connect():
            logger.error("Failed to connect to MetaTrader5. Exiting.")
            return
        
        self.is_running = True
        loop_count = 0
        
        try:
            while self.is_running:
                loop_count += 1
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(f"\n>>> LOOP #{loop_count} - {current_time}")
                
                # Fetch candles
                df = self.mt5.fetch_candles(
                    self.config.SYMBOL,
                    self.config.TIMEFRAME,
                    self.config.MIN_CANDLES_REQUIRED + 50
                )
                
                if df is None:
                    logger.warning("Failed to fetch candles. Retrying...")
                    time.sleep(self.config.LOOP_INTERVAL)
                    continue
                
                # Check if enough data
                if len(df) < self.config.MIN_CANDLES_REQUIRED:
                    logger.warning(f"Not enough data. Got {len(df)}, need {self.config.MIN_CANDLES_REQUIRED}")
                    time.sleep(self.config.LOOP_INTERVAL)
                    continue
                
                # Fetch H1 data for MTF filter
                df_h1 = self.mt5.fetch_candles(self.config.SYMBOL, self.config.H1_TIMEFRAME, 250)
                h1_data = {'close': np.nan, 'ema_200': np.nan}
                if df_h1 is not None and not df_h1.empty:
                    df_h1['EMA_200'] = ta.ema(df_h1['close'], length=200)
                    h1_data['close'] = df_h1['close'].iloc[-1]
                    h1_data['ema_200'] = df_h1['EMA_200'].iloc[-1]
                
                # Get current server hour
                last_tick = mt5.symbol_info_tick(self.config.SYMBOL)
                if last_tick:
                    hour = datetime.fromtimestamp(last_tick.time).hour
                else:
                    hour = datetime.now().hour

                # Calculate indicators
                try:
                    calc = IndicatorCalculator(df)
                    indicators = calc.get_latest_values()
                except Exception as e:
                    logger.error(f"Error calculating indicators: {e}")
                    logger.error(traceback.format_exc())
                    time.sleep(self.config.LOOP_INTERVAL)
                    continue
                
                # Log indicator values
                self.log_indicator_values(indicators)
                
                # Check spread
                spread = self.mt5.get_spread(self.config.SYMBOL)
                logger.info(f"Current Spread: {spread} pips")
                
                if spread > self.config.MAX_SPREAD:
                    logger.warning(f"Spread too wide ({spread} > {self.config.MAX_SPREAD}). Skipping trade.")
                    time.sleep(self.config.LOOP_INTERVAL)
                    continue
                
                # Generate signal
                signal, details = SignalGenerator.generate_signal(indicators, h1_data, hour, df)
                self.log_signal_analysis(signal, details)
                
                # Export data for dashboard
                self.export_data(indicators, signal, details)
                
                # Execute trade if signal is BUY or SELL
                if signal in ['BUY', 'SELL']:
                    atr = indicators['atr']
                    current_price = indicators['close']
                    
                    if signal == 'BUY':
                        sl_price = current_price - (atr * self.config.SL_MULTIPLIER)
                        tp_price = current_price + (atr * self.config.TP_MULTIPLIER)
                        
                        logger.info(f"\n*** BUY SIGNAL CONFIRMED ***")
                        logger.info(f"ATR: {atr:.5f}")
                        logger.info(f"Calculated SL: {sl_price:.5f} ({atr * self.config.SL_MULTIPLIER:.5f} from entry)")
                        logger.info(f"Calculated TP: {tp_price:.5f} ({atr * self.config.TP_MULTIPLIER:.5f} from entry)")
                        
                        ticket = self.trade_manager.execute_buy_order(
                            current_price, sl_price, tp_price,
                            "TrendConfirmation-5Ind"
                        )
                        
                        if ticket:
                            self.history_manager.record_entry(ticket, 'BUY', indicators)

                    
                    elif signal == 'SELL':
                        sl_price = current_price + (atr * self.config.SL_MULTIPLIER)
                        tp_price = current_price - (atr * self.config.TP_MULTIPLIER)
                        
                        logger.info(f"\n*** SELL SIGNAL CONFIRMED ***")
                        logger.info(f"ATR: {atr:.5f}")
                        logger.info(f"Calculated SL: {sl_price:.5f} ({atr * self.config.SL_MULTIPLIER:.5f} from entry)")
                        logger.info(f"Calculated TP: {tp_price:.5f} ({atr * self.config.TP_MULTIPLIER:.5f} from entry)")
                        
                        ticket = self.trade_manager.execute_sell_order(
                            current_price, sl_price, tp_price,
                            "TrendConfirmation-5Ind"
                        )
                        
                        if ticket:
                            self.history_manager.record_entry(ticket, 'SELL', indicators)

                else:
                    logger.info("No signal. Waiting for next candle...")
                
                # Manage SL / TP 
                self.trade_manager.manage_open_trades(indicators['close'], indicators['atr'])
                
                # Check open trades
                open_trades = self.trade_manager.get_open_trades()
                logger.info(f"\nOpen Trades: {len(open_trades)}")
                if open_trades:
                    for trade in open_trades:
                        logger.info(f"  - Ticket: {trade.ticket} | Type: {'BUY' if trade.type == 0 else 'SELL'} | "
                                  f"Volume: {trade.volume} | Entry: {trade.price_open:.5f}")
                
                # Update history (process closed trades)
                deals = self.mt5.fetch_history_deals(days=1)
                self.history_manager.process_history(deals)

                
                # Wait before next loop
                logger.info(f"Waiting {self.config.LOOP_INTERVAL}s before next check...\n")
                time.sleep(self.config.LOOP_INTERVAL)
        
        except KeyboardInterrupt:
            logger.info("\n*** BOT STOPPED BY USER ***")
            self.is_running = False
        except Exception as e:
            logger.error(f"Critical error in main loop: {e}")
            logger.error(traceback.format_exc())
            self.is_running = False
        finally:
            self.mt5.disconnect()
            logger.info("\n*** BOT SHUTDOWN COMPLETE ***\n")

# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    try:
        # Initialize bot
        bot = TrendConfirmationBot(Config)
        
        # Run bot
        bot.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
