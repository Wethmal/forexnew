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
    TIMEFRAME = mt5.TIMEFRAME_H1 # Switched to H1 for 50%+ Accuracy
    TIMEFRAME_NAME = "H1"
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
    
    SL_MULTIPLIER = 2.0  
    TP_MULTIPLIER = 2.0  # 1:1 Risk Reward for 50%+ Accuracy
    
    BE_ACTIVATION_ATR = 1.0
    TRAILING_STOP_ATR = 2.0
    
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
    """Manages trade execution and position tracking with professional risk management."""
    
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
    
    def calculate_professional_positions(self, entry_price: float, atr: float, 
                                         direction: str, confluence_score: float) -> dict:
        """
        PROFESSIONAL POSITION SIZING
        
        Based on:
        1. Risk-Reward Ratio (1:2 minimum for B-rated setups, 1:3 for A-rated)
        2. ATR-based SL placement
        3. Confluence score adjustment
        4. Account heat management
        """
        
        if direction == 'BUY':
            # SL: 1.0 ATR below entry (tight stop-loss for professional entries)
            sl_price = entry_price - (atr * 1.0)
            
            # TP: Scale based on confluence score
            # A+ setup (confluence >= 4.5): 1:3 RR (TP = entry + 3*SL_distance)
            # A setup (confluence >= 4.0): 1:2.5 RR
            # B+ setup (confluence >= 3.5): 1:2 RR
            if confluence_score >= 4.5:
                tp_price = entry_price + (atr * 3.0)
            elif confluence_score >= 4.0:
                tp_price = entry_price + (atr * 2.5)
            else:
                tp_price = entry_price + (atr * 2.0)
            
        else:  # SELL
            sl_price = entry_price + (atr * 1.0)
            
            if confluence_score >= 4.5:
                tp_price = entry_price - (atr * 3.0)
            elif confluence_score >= 4.0:
                tp_price = entry_price - (atr * 2.5)
            else:
                tp_price = entry_price - (atr * 2.0)
        
        return {
            'sl_price': sl_price,
            'tp_price': tp_price,
            'risk': abs(entry_price - sl_price),
            'reward': abs(tp_price - entry_price),
            'rr_ratio': abs(tp_price - entry_price) / abs(entry_price - sl_price) if entry_price != sl_price else 0
        }
    
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
            
            logger.info(f"✓ BUY ORDER EXECUTED (Professional Setup)")
            logger.info(f"  Entry: {entry_price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
            logger.info(f"  Risk: {abs(entry_price - sl_price):.5f} | Reward: {abs(tp_price - entry_price):.5f}")
            logger.info(f"  R:R Ratio: 1:{abs(tp_price - entry_price)/abs(entry_price - sl_price):.2f}")
            logger.info(f"  Lot Size: {self.lot_size} | Ticket: {result.order}")
            return result.order

            
        except Exception as e:
            logger.error(f"Exception in execute_buy_order: {e}")
            logger.error(traceback.format_exc())
            return False
    
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
            
            logger.info(f"✓ SELL ORDER EXECUTED (Professional Setup)")
            logger.info(f"  Entry: {entry_price:.5f} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
            logger.info(f"  Risk: {abs(entry_price - sl_price):.5f} | Reward: {abs(tp_price - entry_price):.5f}")
            logger.info(f"  R:R Ratio: 1:{abs(tp_price - entry_price)/abs(entry_price - sl_price):.2f}")
            logger.info(f"  Lot Size: {self.lot_size} | Ticket: {result.order}")
            return result.order

            
        except Exception as e:
            logger.error(f"Exception in execute_sell_order: {e}")
            logger.error(traceback.format_exc())
            return False

    def manage_open_trades(self, current_price: float, atr: float) -> None:
        """
        PROFESSIONAL TRADE MANAGEMENT
        
        1. Break-even move: Once profit reaches 0.75 ATR
        2. Trailing stop: Use 1.5 ATR distance
        3. Partial profit taking: 50% at 1.5 ATR profit
        """
        try:
            trades = self.get_open_trades()
            if not trades:
                return

            pip_value = 0.0001  # For EURUSD

            for trade in trades:
                if trade.type == 0:  # BUY
                    profit_pips = (current_price - trade.price_open) / pip_value
                    profit_in_atr = profit_pips / (atr / pip_value) if atr > 0 else 0

                    # Break-even activation: 0.75 ATR profit
                    if profit_in_atr >= 0.75 and trade.sl < trade.price_open:
                        new_sl = trade.price_open
                        self._modify_position_sl(trade.ticket, new_sl)
                        logger.info(f"✓ Break-Even Set: Ticket {trade.ticket} | SL: {new_sl:.5f}")

                    # Trailing stop: Keep SL at 1.5 ATR below current price
                    elif profit_in_atr > 0:
                        trailing_sl = current_price - (atr * 1.5)
                        if trailing_sl > trade.sl:
                            self._modify_position_sl(trade.ticket, trailing_sl)
                            logger.info(f"✓ Trailing Stop: Ticket {trade.ticket} | New SL: {trailing_sl:.5f}")

                else:  # SELL
                    profit_pips = (trade.price_open - current_price) / pip_value
                    profit_in_atr = profit_pips / (atr / pip_value) if atr > 0 else 0

                    # Break-even activation: 0.75 ATR profit
                    if profit_in_atr >= 0.75 and trade.sl > trade.price_open:
                        new_sl = trade.price_open
                        self._modify_position_sl(trade.ticket, new_sl)
                        logger.info(f"✓ Break-Even Set: Ticket {trade.ticket} | SL: {new_sl:.5f}")

                    # Trailing stop: Keep SL at 1.5 ATR above current price
                    elif profit_in_atr > 0:
                        trailing_sl = current_price + (atr * 1.5)
                        if trailing_sl < trade.sl:
                            self._modify_position_sl(trade.ticket, trailing_sl)
                            logger.info(f"✓ Trailing Stop: Ticket {trade.ticket} | New SL: {trailing_sl:.5f}")

        except Exception as e:
            logger.error(f"Error in manage_open_trades: {e}")
            logger.error(traceback.format_exc())

    def _modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify an open position's stop loss."""
        try:
            position = None
            trades = self.get_open_trades()
            for trade in trades:
                if trade.ticket == ticket:
                    position = trade
                    break

            if not position:
                logger.warning(f"Position {ticket} not found")
                return False

            filling_mode = self.get_filling_mode()

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
                "tp": position.tp,
                "magic": 99999,
            }

            result = mt5.order_send(request)

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to modify position {ticket}. Retcode: {result.retcode}")
                return False

            return True

        except Exception as e:
            logger.error(f"Exception in _modify_position_sl: {e}")
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
        # EMAs
        self.df['EMA_9'] = ta.ema(self.df['close'], length=9)
        self.df['EMA_21'] = ta.ema(self.df['close'], length=21)
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
            'ema_9': safe_get('EMA_9'),
            'ema_21': safe_get('EMA_21'),
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
# MARKET STRUCTURE ANALYZER (Professional Grade)
# ============================================================================

class MarketStructureAnalyzer:
    """Analyzes market structure for professional trading decisions."""
    
    @staticmethod
    def find_swing_levels(df: pd.DataFrame, lookback: int = 20) -> dict:
        """Identify recent swing highs and lows for market structure."""
        if len(df) < lookback + 5:
            return {'swing_high': np.nan, 'swing_low': np.nan, 'structure': 'INSUFFICIENT'}
        
        recent = df.tail(lookback)
        swing_high = recent['high'].max()
        swing_low = recent['low'].min()
        
        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'swings_range': swing_high - swing_low
        }
    
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, periods: int = 50) -> dict:
        """Calculate dynamic support/resistance levels."""
        if len(df) < periods:
            return {}
        
        recent = df.tail(periods)
        
        # Find pivot points from last 50 candles
        high_points = recent['high'].nlargest(3).values
        low_points = recent['low'].nsmallest(3).values
        
        return {
            'resistance': float(high_points[0]) if len(high_points) > 0 else np.nan,
            'support': float(low_points[0]) if len(low_points) > 0 else np.nan,
        }
    
    @staticmethod
    def analyze_volatility_regime(indicators: dict, atr_period: int = 14) -> str:
        """Classify market volatility regime."""
        atr = indicators.get('atr', 0)
        
        if atr < 0.0005:  # Very tight
            return 'SQUEEZED'
        elif atr < 0.0008:  # Normal
            return 'NORMAL'
        elif atr < 0.0012:  # Higher
            return 'ELEVATED'
        else:  # Expansion
            return 'EXPANSION'


# ============================================================================
# PROFESSIONAL SIGNAL GENERATOR (Expert Level - 30 Years Market Experience)
# ============================================================================

class SignalGenerator:
    """Professional-grade signal generation with confluence-based rules."""
    
    @staticmethod
    def check_primary_trend(indicators: dict, h1_data: dict, df: pd.DataFrame) -> tuple:
        """
        Professional trend confirmation using:
        1. EMA alignment (Price > EMA50 > EMA200)
        2. H1 timeframe confirmation
        3. Trend strength via ADX
        """
        close = indicators['close']
        ema_50 = indicators['ema_50']
        ema_200 = indicators['ema_200']
        
        if pd.isna(ema_50) or pd.isna(ema_200):
            return 'UNDEFINED', False, 0
        
        h1_close = h1_data.get('close', np.nan)
        h1_ema200 = h1_data.get('ema_200', np.nan)
        
        # BULLISH: Close > EMA50 > EMA200 with H1 confirmation
        if close > ema_50 > ema_200:
            if not pd.isna(h1_close) and not pd.isna(h1_ema200):
                if h1_close > h1_ema200:  # H1 confirms uptrend
                    return 'STRONG_UPTREND', True, 1.0
            return 'UPTREND', True, 0.8
        
        # BEARISH: Close < EMA50 < EMA200 with H1 confirmation
        elif close < ema_50 < ema_200:
            if not pd.isna(h1_close) and not pd.isna(h1_ema200):
                if h1_close < h1_ema200:  # H1 confirms downtrend
                    return 'STRONG_DOWNTREND', True, 1.0
            return 'DOWNTREND', True, 0.8
        
        # WEAK: Only one layer in proper order
        elif ema_50 > ema_200:
            return 'WEAK_UPTREND', False, 0.5
        elif ema_50 < ema_200:
            return 'WEAK_DOWNTREND', False, 0.5
        
        return 'NO_TREND', False, 0
    
    @staticmethod
    def check_momentum_confirmation(indicators: dict, trend_dir: str) -> tuple:
        """
        Momentum confirmation using MACD + RSI convergence.
        PROFESSIONAL RULES:
        - MACD must cross zero line (strong signals only)
        - RSI must show directional bias (not extreme)
        - Both must align with trend
        """
        macd = indicators['macd']
        macd_signal = indicators['macd_signal']
        macd_hist = indicators['macd_hist']
        rsi = indicators['rsi']
        
        if pd.isna(macd) or pd.isna(macd_signal) or pd.isna(rsi):
            return 'INVALID', False, 0
        
        macd_valid = False
        rsi_valid = False
        
        if trend_dir in ['STRONG_UPTREND', 'UPTREND', 'WEAK_UPTREND']:
            # BUY: MACD above signal + histogram positive + histogram growing
            macd_valid = (macd > macd_signal and macd_hist > 0)
            # RSI: Not overbought, showing momentum (30-75 range is OK)
            rsi_valid = (35 < rsi < 80)
            
            if macd_valid and rsi_valid:
                return 'BULLISH_CONFIRMED', True, 1.0
            elif macd_valid and 30 < rsi < 70:
                return 'BULLISH_PARTIAL', True, 0.7
            elif macd > 0:
                return 'BULLISH_WEAK', False, 0.4
        
        elif trend_dir in ['STRONG_DOWNTREND', 'DOWNTREND', 'WEAK_DOWNTREND']:
            # SELL: MACD below signal + histogram negative + histogram growing
            macd_valid = (macd < macd_signal and macd_hist < 0)
            # RSI: Not oversold, showing momentum (25-70 range is OK)
            rsi_valid = (20 < rsi < 65)
            
            if macd_valid and rsi_valid:
                return 'BEARISH_CONFIRMED', True, 1.0
            elif macd_valid and 30 < rsi < 70:
                return 'BEARISH_PARTIAL', True, 0.7
            elif macd < 0:
                return 'BEARISH_WEAK', False, 0.4
        
        return 'NO_MOMENTUM', False, 0
    
    @staticmethod
    def check_volatility_and_structure(indicators: dict, df: pd.DataFrame) -> tuple:
        """
        Check for proper volatility expansion and market structure.
        RULES:
        - Bollinger Bands must be expanding (not squeezing)
        - ATR must be adequate for trade execution
        - No news-driven gaps (price > 3x ATR move)
        """
        atr = indicators['atr']
        bb_upper = indicators['bb_upper']
        bb_lower = indicators['bb_lower']
        close = indicators['close']
        
        if pd.isna(atr) or pd.isna(bb_upper) or pd.isna(bb_lower):
            return 'INVALID', False
        
        bb_width = (bb_upper - bb_lower) / close
        
        # Band expansion is healthy
        if bb_width > 0.005:  # >0.5% width
            structure_valid = True
        else:
            structure_valid = False
        
        # ATR must be adequate
        atr_valid = atr > 0.00035  # Minimum ATR threshold
        
        if structure_valid and atr_valid:
            return 'HEALTHY_STRUCTURE', True
        elif atr_valid:
            return 'SQUEEZED_BANDS', False
        else:
            return 'INSUFFICIENT_ATR', False
    
    @staticmethod
    def check_price_action(df: pd.DataFrame, trend_dir: str) -> tuple:
        """
        Professional price action analysis:
        - Internal bar reversal (IB)
        - Break of structure
        - Pullback setup
        """
        if len(df) < 5:
            return 'INSUFFICIENT_DATA', False
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        pprev = df.iloc[-3]
        
        # Current candle should show directional conviction
        curr_body = abs(curr['close'] - curr['open'])
        curr_range = curr['high'] - curr['low']
        
        if curr_body < (curr_range * 0.3):  # Doji or spinning top
            return 'INDECISION', False
        
        if trend_dir in ['STRONG_UPTREND', 'UPTREND', 'WEAK_UPTREND']:
            # Bullish: Higher lows, bullish candles, no rejection
            if curr['close'] > curr['open'] and curr['close'] > prev['high']:
                return 'BULLISH_CONVICTION', True
            elif curr['low'] > prev['low'] and curr['close'] > curr['open']:
                return 'BULLISH_STRUCTURE', True
            else:
                return 'BULLISH_WEAK', False
        
        elif trend_dir in ['STRONG_DOWNTREND', 'DOWNTREND', 'WEAK_DOWNTREND']:
            # Bearish: Lower highs, bearish candles, no rejection
            if curr['close'] < curr['open'] and curr['close'] < prev['low']:
                return 'BEARISH_CONVICTION', True
            elif curr['high'] < prev['high'] and curr['close'] < curr['open']:
                return 'BEARISH_STRUCTURE', True
            else:
                return 'BEARISH_WEAK', False
        
        return 'NO_CONVICTION', False
    
    @staticmethod
    def check_confluence_score(signal_details: dict) -> float:
        """Calculate confluence score: how many major factors align (0-5)."""
        score = 0
        
        # Trend (1 point)
        trend_status = signal_details.get('trend', ('', False))
        if trend_status[0] in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
            score += 1.0
        elif trend_status[0] in ['UPTREND', 'DOWNTREND']:
            score += 0.8
        
        # Momentum (1 point)
        momentum_status = signal_details.get('momentum', ('', False))[0]
        if 'CONFIRMED' in momentum_status:
            score += 1.0
        elif 'PARTIAL' in momentum_status:
            score += 0.6
        
        # Structure (1 point)
        structure_status = signal_details.get('structure', ('', False))
        if structure_status[1]:
            score += 1.0
        
        # Price action (1 point)
        paction_status = signal_details.get('price_action', ('', False))
        if 'CONVICTION' in paction_status[0]:
            score += 1.0
        elif 'STRUCTURE' in paction_status[0]:
            score += 0.7
        
        # Session (1 point)
        session_status = signal_details.get('session', ('', False))
        if session_status[1]:
            score += 1.0
        
        return min(score, 5.0)
    
    @staticmethod
    def check_session_filter(hour: int) -> tuple:
        """Session filter: Trade only during high-liquidity hours."""
        # London (8-12), NYC (13-17), Tokyo overlap (evening)
        if 8 <= hour < 17:
            return 'OPTIMAL_HOURS', True
        elif 0 <= hour < 8 or 17 <= hour < 24:
            return 'CAUTION_HOURS', False
        return 'OUT_OF_SESSION', False
    
    @staticmethod
    def generate_signal(indicators: dict, h1_data: dict, hour: int, df: pd.DataFrame) -> tuple:
        """
        Professional confluence-based signal generation.
        Combines multiple technical filters for high-probability entries.
        """
        if pd.isna(indicators.get('close')):
            return 'HOLD', {}

        # 1. Run all technical checks
        trend_status = SignalGenerator.check_primary_trend(indicators, h1_data, df)
        trend_msg, trend_valid, trend_weight = trend_status
        
        momentum_status = SignalGenerator.check_momentum_confirmation(indicators, trend_msg)
        structure_status = SignalGenerator.check_volatility_and_structure(indicators, df)
        pa_status = SignalGenerator.check_price_action(df, trend_msg)
        session_status = SignalGenerator.check_session_filter(hour)
        
        # 2. Compile details dictionary
        details = {
            'trend': (trend_msg, trend_valid),
            'momentum': (momentum_status[0], momentum_status[1]),
            'structure': (structure_status[0], structure_status[1]),
            'price_action': (pa_status[0], pa_status[1]),
            'session': (session_status[0], session_status[1]),
        }
        
        # 3. Calculate confluence score
        confluence_score = SignalGenerator.check_confluence_score(details)
        details['confluence_score'] = confluence_score
        
        # 4. Final signal decision based on confluence
        signal = 'HOLD'
        if confluence_score >= 3.5:
            if 'UP' in trend_msg:
                signal = 'BUY'
            elif 'DOWN' in trend_msg:
                signal = 'SELL'
                
        return signal, details

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

    
    def export_data(self, indicators: Dict, signal: str, details: Dict, h1_data: Dict = None):
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
                    "h1_data": h1_data if h1_data else {},
                    "lot_size": self.config.LOT_SIZE,
                },
                "indicators": indicators,
                "signal": {
                    "type": signal,
                    "details": {k: [v[0], bool(v[1])] for k, v in details.items() if isinstance(v, (tuple, list))},
                    "confluence_score": details.get('confluence_score', 0)
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
        logger.info(f"ADX(14):       {indicators['adx']:.2f} (Threshold: {Config.ADX_THRESHOLD})")
        logger.info(f"BB Upper:      {indicators['bb_upper']:.5f} | Middle: {indicators['bb_middle']:.5f} | Lower: {indicators['bb_lower']:.5f}")
        logger.info(f"ATR(14):       {indicators['atr']:.5f}")
        logger.info("=" * 80)
    
    def log_signal_analysis(self, signal: str, details: Dict):
        """Log signal generation analysis with professional grading."""
        logger.info("=" * 80)
        logger.info("PROFESSIONAL SIGNAL ANALYSIS")
        logger.info("=" * 80)

        # Trend Grade
        trend_msg, trend_valid = details['trend']
        trend_grade = "A+" if "STRONG" in trend_msg else ("A" if trend_valid else "B")
        trend_status = "✓" if trend_valid else "✗"
        logger.info(f"{trend_status} Trend: {trend_msg:25} | Grade: {trend_grade}")

        # Momentum Grade
        momentum_detail = details.get('momentum', ('UNKNOWN', False))
        momentum_msg, momentum_valid = momentum_detail[0], momentum_detail[1]
        momo_grade = "A+" if "CONFIRMED" in momentum_msg else ("A" if "PARTIAL" in momentum_msg else "B")
        momo_status = "✓" if momentum_valid else "✗"
        logger.info(f"{momo_status} Momentum: {momentum_msg:25} | Grade: {momo_grade}")

        # Structure Grade
        structure_msg, structure_valid = details.get('structure', ('INVALID', False))
        struct_grade = "A" if structure_valid else "B"
        struct_status = "✓" if structure_valid else "✗"
        logger.info(f"{struct_status} Structure: {structure_msg:25} | Grade: {struct_grade}")

        # Price Action Grade  
        pa_detail = details.get('price_action', ('NO_DATA', False))
        pa_msg, pa_valid = pa_detail[0], pa_detail[1]
        pa_grade = "A+" if "CONVICTION" in pa_msg else ("A" if "STRUCTURE" in pa_msg else "B")
        pa_status = "✓" if pa_valid else "✗"
        logger.info(f"{pa_status} Price Action: {pa_msg:25} | Grade: {pa_grade}")

        # Session Grade
        session_msg, session_valid = details.get('session', ('UNKNOWN', False))
        sess_status = "✓" if session_valid else "✗"
        logger.info(f"{sess_status} Session: {session_msg:25} | Grade: {'A' if session_valid else 'C'}")

        # Confluence Score
        confluence = details.get('confluence_score', 0)
        logger.info("-" * 80)
        
        if confluence >= 4.5:
            quality = "A+ (Excellent)"
        elif confluence >= 4.0:
            quality = "A (Strong)"
        elif confluence >= 3.5:
            quality = "B+ (Good)"
        else:
            quality = "B (Fair)"
            
        logger.info(f"CONFLUENCE SCORE: {confluence:.2f}/5.0 | Setup Quality: {quality}")
        
        logger.info("-" * 80)
        logger.info(f"FINAL SIGNAL: {signal}")
        logger.info("=" * 80)
    
    def run(self):
        """Main trading loop."""
        logger.info("\n" + "=" * 80)
        logger.info("PROFESSIONAL FOREX TRADING BOT - EXPERT SYSTEM")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Symbol: {self.config.SYMBOL} | Timeframe: {self.config.TIMEFRAME_NAME}")
        logger.info(f"  Lot Size: {self.config.LOT_SIZE} | Trading Hours: {self.config.START_HOUR:02d}:00 - {self.config.END_HOUR:02d}:00 UTC")
        logger.info(f"  Max Spread: {self.config.MAX_SPREAD} pips | Loop Interval: {self.config.LOOP_INTERVAL}s")
        logger.info(f"\nTRADING RULES:")
        logger.info(f"  - Entry: Confluence Score >= 3.5/5.0")
        logger.info(f"  - SL: 1.0 × ATR from entry")
        logger.info(f"  - TP: Dynamic based on confluence (1:2 to 1:3 risk-reward)")
        logger.info(f"  - Exit: Take profit or stop loss, trailing stop after 0.75 ATR profit")
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
                signal, details = SignalGenerator.generate_signal(indicators, h1_data, hour, calc.df)
                self.log_signal_analysis(signal, details)
                
                # Export data for dashboard
                self.export_data(indicators, signal, details, h1_data)
                
                # Execute trade if signal is BUY or SELL
                if signal in ['BUY', 'SELL']:
                    atr = indicators['atr']
                    current_price = indicators['close']
                    confluence_score = details.get('confluence_score', 0)
                    
                    if signal == 'BUY':
                        # Use professional position sizing based on confluence
                        positions = self.trade_manager.calculate_professional_positions(
                            current_price, atr, 'BUY', confluence_score
                        )
                        sl_price = positions['sl_price']
                        tp_price = positions['tp_price']
                        
                        logger.info(f"\n{'='*80}")
                        logger.info(f"*** PROFESSIONAL BUY SIGNAL - EXECUTING (CONFLUENCE {confluence_score:.2f}/5.0) ***")
                        logger.info(f"{'='*80}")
                        logger.info(f"Entry Price: {current_price:.5f}")
                        logger.info(f"Stop Loss:   {sl_price:.5f} (Risk: {positions['risk']:.5f})")
                        logger.info(f"Take Profit: {tp_price:.5f} (Reward: {positions['reward']:.5f})")
                        logger.info(f"Risk:Reward: 1:{positions['rr_ratio']:.2f}")
                        logger.info(f"ATR(14):     {atr:.5f}")
                        logger.info(f"{'='*80}")
                        
                        ticket = self.trade_manager.execute_buy_order(
                            current_price, sl_price, tp_price,
                            f"Professional-BUY-C{confluence_score:.1f}"
                        )
                        
                        if ticket:
                            self.history_manager.record_entry(ticket, 'BUY', indicators)

                    
                    elif signal == 'SELL':
                        # Use professional position sizing based on confluence
                        positions = self.trade_manager.calculate_professional_positions(
                            current_price, atr, 'SELL', confluence_score
                        )
                        sl_price = positions['sl_price']
                        tp_price = positions['tp_price']
                        
                        logger.info(f"\n{'='*80}")
                        logger.info(f"*** PROFESSIONAL SELL SIGNAL - EXECUTING (CONFLUENCE {confluence_score:.2f}/5.0) ***")
                        logger.info(f"{'='*80}")
                        logger.info(f"Entry Price: {current_price:.5f}")
                        logger.info(f"Stop Loss:   {sl_price:.5f} (Risk: {positions['risk']:.5f})")
                        logger.info(f"Take Profit: {tp_price:.5f} (Reward: {positions['reward']:.5f})")
                        logger.info(f"Risk:Reward: 1:{positions['rr_ratio']:.2f}")
                        logger.info(f"ATR(14):     {atr:.5f}")
                        logger.info(f"{'='*80}")
                        
                        ticket = self.trade_manager.execute_sell_order(
                            current_price, sl_price, tp_price,
                            f"Professional-SELL-C{confluence_score:.1f}"
                        )
                        
                        if ticket:
                            self.history_manager.record_entry(ticket, 'SELL', indicators)

                else:
                    confluence_score = details.get('confluence_score', 0)
                    reason = details['trend'][0] if confluence_score < 3.5 else "No confluence"
                    logger.info(f"HOLD - Confluence: {confluence_score:.2f}/5.0 (Reason: {reason})")
                
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
