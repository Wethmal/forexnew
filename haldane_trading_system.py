import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import alpaca_trade_api as tradeapi
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import time
import logging
from datetime import datetime

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HaldaneTradingSystem:
    def __init__(self, symbol='SPY', api_key=None, api_secret=None, base_url=None):
        self.symbol = symbol
        self.scaler = StandardScaler()
        self.model = None
        self.rf_model = None
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.api = None
        self.data = None
        self.features = [
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_lower', 'Stoch_K', 'Stoch_D',
            'ATR', 'Volume_ratio', 'Price_change', 'Price_change_5',
            'Price_change_10', 'Volatility'
        ]

        if api_key and api_secret and base_url:
            self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')

    def fetch_data(self, period='2y'):
        """Fetch historical data from Yahoo Finance"""
        logger.info(f"Fetching {period} of data for {self.symbol}")
        self.data = yf.download(self.symbol, period=period, interval='1d')
        logger.info(f"Fetched {len(self.data)} rows of data")
        return self.data

    def add_technical_indicators(self):
        """Add technical indicators to the dataset"""
        df = self.data.copy()

        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)

        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']

        # Price change indicators
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_5'] = df['Close'].pct_change(5)
        df['Price_change_10'] = df['Close'].pct_change(10)

        # Volatility
        df['Volatility'] = df['Price_change'].rolling(window=20).std()

        # Drop NaN values
        df.dropna(inplace=True)

        self.data = df
        logger.info(f"Added technical indicators. Dataset has {len(df)} rows")
        return df

    def prepare_data(self, look_back=60):
        """Prepare data for ML model"""
        df = self.data.copy()

        # Create target variable (1 if price goes up next day, 0 otherwise)
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

        # Create sequences for LSTM
        X = []
        y = []

        for i in range(look_back, len(df)):
            X.append(df[self.features].iloc[i - look_back:i].values)
            y.append(df['Target'].iloc[i])

        X, y = np.array(X), np.array(y)

        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Scale features
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)

        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_lstm_model(self, input_shape):
        """Build LSTM model for high accuracy predictions"""
        model = Sequential()

        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))

        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(32))
        model.add(Dropout(0.2))

        model.add(Dense(16, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model = model
        return model

    def train_ensemble_model(self, X_train, y_train):
        """Train a Random Forest model as an ensemble companion to LSTM"""
        # Flatten LSTM sequences to 2D for Random Forest (use the last time-step)
        X_train_flat = X_train[:, -1, :]
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train_flat, y_train)
        logger.info("Random Forest ensemble model trained")

    def train_model(self, epochs=50, batch_size=32):
        """Train the ML model"""
        X_train, X_test, y_train, y_test = self.prepare_data()

        # Build LSTM model
        self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Callbacks for better training
        early_stop = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )

        # Train LSTM model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Train Random Forest ensemble
        self.train_ensemble_model(X_train, y_train)

        # Evaluate LSTM model
        y_pred_lstm = (self.model.predict(X_test) > 0.5).astype(int).flatten()

        # Evaluate Random Forest model
        X_test_flat = X_test[:, -1, :]
        y_pred_rf = self.rf_model.predict(X_test_flat)

        # Ensemble prediction (agree-on-buy: both models must predict 1)
        y_pred_ensemble = ((y_pred_lstm + y_pred_rf) >= 2).astype(int)

        lstm_accuracy = accuracy_score(y_test, y_pred_lstm)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

        logger.info(f"LSTM Accuracy: {lstm_accuracy:.4f}")
        logger.info(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        logger.info(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        print(f"\nLSTM Accuracy: {lstm_accuracy:.4f}")
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
        print("\nEnsemble Classification Report:")
        print(classification_report(y_test, y_pred_ensemble))

        return history

    def generate_signal(self):
        """Generate trading signal based on latest data using ensemble prediction"""
        if self.model is None:
            logger.warning("Model not trained yet. Please train the model first.")
            return None

        # Get latest data window
        latest_data = self.data[self.features].iloc[-60:].copy()

        X = latest_data.values
        X = X.reshape(1, X.shape[0], X.shape[1])
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        # LSTM prediction
        lstm_prob = self.model.predict(X_scaled, verbose=0)[0][0]
        lstm_pred = int(lstm_prob > 0.5)

        # Random Forest prediction
        rf_pred = 0
        rf_prob = 0.5
        if self.rf_model is not None:
            X_flat = X_scaled[:, -1, :]
            rf_pred = int(self.rf_model.predict(X_flat)[0])
            rf_prob = self.rf_model.predict_proba(X_flat)[0][1]

        # Ensemble confidence: average of both model probabilities
        ensemble_prob = (lstm_prob + rf_prob) / 2.0

        # Technical indicator confirmation
        last_row = self.data.iloc[-1]
        rsi = last_row['RSI']
        macd_hist = last_row['MACD_hist']
        stoch_k = last_row['Stoch_K']
        close = last_row['Close']
        bb_upper = last_row['BB_upper']
        bb_lower = last_row['BB_lower']

        # Build a composite technical score (-1 to +1 range)
        tech_score = 0.0

        # RSI: oversold favours buy, overbought favours sell
        if rsi < 30:
            tech_score += 0.3
        elif rsi > 70:
            tech_score -= 0.3

        # MACD histogram direction
        if macd_hist > 0:
            tech_score += 0.2
        else:
            tech_score -= 0.2

        # Stochastic: oversold favours buy, overbought favours sell
        if stoch_k < 20:
            tech_score += 0.2
        elif stoch_k > 80:
            tech_score -= 0.2

        # Bollinger Band position
        if close <= bb_lower:
            tech_score += 0.3
        elif close >= bb_upper:
            tech_score -= 0.3

        # Final signal determination
        # Combine ML ensemble probability with technical score
        combined_score = ensemble_prob + (tech_score * 0.5)

        if combined_score > 0.6:
            signal = 'BUY'
        elif combined_score < 0.4:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        confidence = abs(combined_score - 0.5) * 2  # 0-1 scale

        result = {
            'signal': signal,
            'confidence': round(confidence, 4),
            'lstm_probability': round(float(lstm_prob), 4),
            'rf_probability': round(float(rf_prob), 4),
            'ensemble_probability': round(float(ensemble_prob), 4),
            'technical_score': round(tech_score, 4),
            'combined_score': round(float(combined_score), 4),
            'rsi': round(float(rsi), 2),
            'macd_hist': round(float(macd_hist), 4),
            'current_price': round(float(close), 2),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(
            f"Signal: {signal} | Confidence: {confidence:.2%} | "
            f"Price: {close:.2f} | RSI: {rsi:.1f}"
        )
        return result

    def calculate_position_size(self, signal_info, risk_per_trade=0.02):
        """Calculate position size based on risk management rules.

        Args:
            signal_info: dict from generate_signal()
            risk_per_trade: fraction of account equity to risk per trade (default 2%)

        Returns:
            Number of shares to trade
        """
        if self.api is None:
            logger.warning("Alpaca API not configured, cannot calculate position size")
            return 0

        account = self.api.get_account()
        equity = float(account.equity)
        current_price = signal_info['current_price']
        atr = float(self.data.iloc[-1]['ATR'])

        # Risk amount for this trade
        risk_amount = equity * risk_per_trade

        # Use 2x ATR as the stop-loss distance
        stop_distance = atr * 2.0
        if stop_distance <= 0:
            return 0

        # Position size = risk amount / stop distance
        shares = int(risk_amount / stop_distance)

        # Cap at 10% of equity in a single position
        max_shares = int((equity * 0.10) / current_price)
        shares = min(shares, max_shares)

        # Scale by confidence – trade smaller when confidence is low
        confidence = signal_info['confidence']
        shares = int(shares * min(confidence, 1.0))

        return max(shares, 0)

    def execute_trade(self, signal_info, risk_per_trade=0.02):
        """Execute a trade via the Alpaca API.

        Args:
            signal_info: dict returned by generate_signal()
            risk_per_trade: fraction of equity to risk (default 2%)

        Returns:
            Alpaca order object or None
        """
        if self.api is None:
            logger.warning("Alpaca API not configured. Skipping trade execution.")
            return None

        signal = signal_info['signal']
        if signal == 'HOLD':
            logger.info("Signal is HOLD – no trade executed")
            return None

        qty = self.calculate_position_size(signal_info, risk_per_trade)
        if qty <= 0:
            logger.info("Position size is 0 – no trade executed")
            return None

        try:
            # Check for existing position
            try:
                position = self.api.get_position(self.symbol)
                current_qty = int(position.qty)
                side = position.side
            except tradeapi.rest.APIError:
                current_qty = 0
                side = None

            order = None

            if signal == 'BUY':
                if side == 'short':
                    # Close short before going long
                    self.api.close_position(self.symbol)
                    logger.info(f"Closed short position of {current_qty} shares")
                order = self.api.submit_order(
                    symbol=self.symbol,
                    qty=qty,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"BUY order submitted: {qty} shares of {self.symbol}")

            elif signal == 'SELL':
                if side == 'long' and current_qty > 0:
                    # Close the long position
                    self.api.close_position(self.symbol)
                    logger.info(f"Closed long position of {current_qty} shares")
                else:
                    order = self.api.submit_order(
                        symbol=self.symbol,
                        qty=qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    logger.info(f"SELL order submitted: {qty} shares of {self.symbol}")

            return order

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None

    def run_backtest(self, look_back=60, initial_capital=100_000):
        """Run a simple backtest over the historical dataset.

        Returns:
            dict with backtest performance metrics
        """
        if self.model is None:
            logger.warning("Model not trained. Train the model before backtesting.")
            return None

        df = self.data.copy()
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

        capital = initial_capital
        position = 0  # number of shares held
        entry_price = 0.0
        trades = []

        for i in range(look_back, len(df) - 1):
            window = df[self.features].iloc[i - look_back:i].values
            X = window.reshape(1, window.shape[0], window.shape[1])
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

            lstm_prob = float(self.model.predict(X_scaled, verbose=0)[0][0])
            rf_prob = 0.5
            if self.rf_model is not None:
                X_flat = X_scaled[:, -1, :]
                rf_prob = float(self.rf_model.predict_proba(X_flat)[0][1])

            ensemble_prob = (lstm_prob + rf_prob) / 2.0

            current_price = float(df['Close'].iloc[i])
            next_price = float(df['Close'].iloc[i + 1])

            if ensemble_prob > 0.55 and position == 0:
                # Buy signal
                shares = int(capital / current_price)
                if shares > 0:
                    position = shares
                    entry_price = current_price
                    capital -= shares * current_price

            elif ensemble_prob < 0.45 and position > 0:
                # Sell signal
                capital += position * current_price
                pnl = (current_price - entry_price) * position
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': position,
                    'pnl': pnl
                })
                position = 0
                entry_price = 0.0

        # Close any remaining position at the end
        if position > 0:
            last_price = float(df['Close'].iloc[-1])
            capital += position * last_price
            pnl = (last_price - entry_price) * position
            trades.append({
                'entry_price': entry_price,
                'exit_price': last_price,
                'shares': position,
                'pnl': pnl
            })

        total_return = (capital - initial_capital) / initial_capital
        winning = [t for t in trades if t['pnl'] > 0]
        losing = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(winning) / len(trades) if trades else 0

        results = {
            'initial_capital': initial_capital,
            'final_capital': round(capital, 2),
            'total_return': round(total_return * 100, 2),
            'total_trades': len(trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': round(win_rate * 100, 2),
            'avg_win': round(
                np.mean([t['pnl'] for t in winning]), 2
            ) if winning else 0,
            'avg_loss': round(
                np.mean([t['pnl'] for t in losing]), 2
            ) if losing else 0,
        }

        logger.info("=== Backtest Results ===")
        for key, val in results.items():
            logger.info(f"  {key}: {val}")

        return results

    def run_live(self, interval_seconds=60, risk_per_trade=0.02, min_confidence=0.15):
        """Run the trading system in a live loop.

        Fetches fresh data, generates signals, and executes trades at the
        specified interval.

        Args:
            interval_seconds: seconds between each iteration
            risk_per_trade: fraction of equity to risk per trade
            min_confidence: minimum signal confidence to execute a trade
        """
        logger.info(f"Starting live trading for {self.symbol}")

        while True:
            try:
                # Refresh market data
                self.fetch_data(period='2y')
                self.add_technical_indicators()

                # Generate signal
                signal_info = self.generate_signal()
                if signal_info is None:
                    logger.warning("No signal generated, skipping cycle")
                else:
                    logger.info(
                        f"Signal: {signal_info['signal']} | "
                        f"Confidence: {signal_info['confidence']:.2%} | "
                        f"Price: {signal_info['current_price']}"
                    )

                    # Execute trade if confidence meets minimum threshold
                    if signal_info['confidence'] >= min_confidence:
                        self.execute_trade(signal_info, risk_per_trade)
                    else:
                        logger.info("Confidence too low – skipping trade")

            except Exception as e:
                logger.error(f"Error in live loop: {e}")

            logger.info(f"Sleeping {interval_seconds}s until next cycle")
            time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import os

    symbol = os.environ.get('TRADE_SYMBOL', 'SPY')
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_API_SECRET')
    base_url = os.environ.get(
        'ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'
    )

    trader = HaldaneTradingSystem(
        symbol=symbol,
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url
    )

    # 1. Fetch data and compute indicators
    trader.fetch_data(period='2y')
    trader.add_technical_indicators()

    # 2. Train models
    trader.train_model(epochs=50, batch_size=32)

    # 3. Generate a signal
    signal = trader.generate_signal()
    if signal:
        print("\n=== Latest Trading Signal ===")
        for k, v in signal.items():
            print(f"  {k}: {v}")

    # 4. Run backtest
    print("\n=== Backtest ===")
    backtest = trader.run_backtest()
    if backtest:
        for k, v in backtest.items():
            print(f"  {k}: {v}")

    # 5. Optionally start live trading (uncomment to enable)
    # trader.run_live(interval_seconds=300)
