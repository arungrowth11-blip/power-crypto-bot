import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import ccxt
import gc
import tracemalloc
import psutil
import sqlite3
import pickle
import asyncio
import logging
import csv
import time
import requests
import shutil
from datetime import datetime, timedelta, time as dtime, timezone, date
from zoneinfo import ZoneInfo
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode
from dotenv import load_dotenv
from flask import Flask, request
import threading
import functools
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# Initialize Flask app for health checks
app = Flask(__name__)

@app.route('/health')
def health_check():
    try:
        # Check database
        with get_db_connection() as conn:
            conn.execute("SELECT 1 FROM signals LIMIT 1")
        
        # Check exchange connection
        exchange.fetch_time()
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return {'status': 'warning', 'memory': f'{memory.percent}%'}, 200
            
        return {
            'status': 'healthy', 
            'timestamp': datetime.now().isoformat(),
            'memory': f'{memory.percent}%'
        }
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}, 500

@app.route('/')
def home():
    return {'message': 'Crypto Trading Bot is running', 'status': 'active'}

# ================================= LOGGING ==================================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# =================== NEW DEPENDENCIES (PyTorch) ======================
# Initialize variables first
PYTORCH_AVAILABLE = False
torch = None
nn = None

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    logger.info("PyTorch found. Hybrid model functionality will be enabled.")
except ImportError:
    logger.warning("PyTorch not found. The new HybridModel will be disabled. Install with 'pip install torch'.")

# Define the load_hybrid_model function outside of the conditional block
def load_hybrid_model(path, model_class):
    if not PYTORCH_AVAILABLE:
        logger.warning("Cannot load Hybrid Model because PyTorch is not installed.")
        return None
    
    if not os.path.exists(path):
        logger.warning(f"Hybrid model not found at {path}. Confidence check will be skipped.")
        return None
        
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading Hybrid Model onto device: {device}")
        
        model = model_class()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        logger.info(f"Successfully loaded Hybrid Model from {path}")
        return model
    except Exception as e:
        logger.error(f"Could not load Hybrid Model: {e}")
        return None

# Only define the HybridModel class if PyTorch is available
if PYTORCH_AVAILABLE:
    class HybridModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=64, nhead=4):
            super().__init__()
            self.recurrent_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, 
                                        batch_first=True, num_layers=2, dropout=0.2)
            transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, 
                                                         dim_feedforward=256, dropout=0.2, activation='relu', batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self.fc2 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            recurrent_out, _ = self.recurrent_layer(x)
            trans_out = self.transformer_encoder(recurrent_out)
            last_step_out = trans_out[:, -1, :]
            out = self.dropout(self.relu(self.fc1(last_step_out)))
            return self.sigmoid(self.fc2(out))
else:
    # Create a dummy class when PyTorch is not available
    class HybridModel:
        def __init__(self, *args, **kwargs):
            pass
        
        def __call__(self, *args, **kwargs):
            return None
        
        def eval(self):
            pass
        
        def to(self, device):
            return self

# ================================== CONFIG ==================================
BOT_TOKEN = os.environ.get("CRYPTO_BOT_TOKEN")
OWNER_CHAT_ID = os.environ.get("CRYPTO_OWNER_ID")

# --- Bot & Strategy Parameters ---
TIMEFRAME = os.environ.get("TIMEFRAME", "1h")  # Changed from "60m" to "1h"
TOP_N_MARKETS = int(os.environ.get("TOP_N_MARKETS", 30))
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL", 15 * 60))
MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", 30))
DB_PATH = os.environ.get("DB_PATH", "/tmp/power_crypto_bot.db")
CACHE_TTL = int(os.environ.get("CACHE_TTL", 90))
CHART_CANDLES = int(os.environ.get("CHART_CANDLES", 100))
HYBRID_MODEL_PATH = os.environ.get("HYBRID_MODEL_PATH", './hybrid_model.pth')
PORTFOLIO_VALUE = float(os.environ.get("PORTFOLIO_VALUE", 10000))

# --- Strategy Fine-Tuning ---
ATR_PERIOD = int(os.environ.get("ATR_PERIOD", 14))
RSI_PERIOD = int(os.environ.get("RSI_PERIOD", 14))
TP_MULT = [float(x) for x in os.environ.get("TP_MULT", "0.75,1.5,3.0").split(",")]
SL_MULT = float(os.environ.get("SL_MULT", 1.5))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", 0.50))

# --- Risk Management ---
MAX_DAILY_LOSS = float(os.environ.get("MAX_DAILY_LOSS", 0.02))
MAX_CONCURRENT_TRADES = int(os.environ.get("MAX_CONCURRENT_TRADES", 10))
MAX_POSITION_SIZE = float(os.environ.get("MAX_POSITION_SIZE", 0.02))

# --- Timezone for Daily Reports ---
REPORT_TIMEZONE = ZoneInfo(os.environ.get("REPORT_TIMEZONE", "Asia/Kolkata"))
REPORT_TIME = dtime(
    hour=int(os.environ.get("REPORT_TIME_HOUR", 9)),
    minute=int(os.environ.get("REPORT_TIME_MINUTE", 0))
)

# --- Parameter Optimization Grid ---
PARAMETER_GRID = {
    'RSI_BUY': [50, 52, 55],
    'RSI_SELL': [48, 45, 42],
    'CONFIDENCE_THRESHOLD': [0.45, 0.50, 0.55],
    'TP_MULT_1': [0.5, 0.75, 1.0],
    'TP_MULT_2': [1.5, 2.0, 2.5],
    'TP_MULT_3': [3.0, 4.0, 5.0],
    'SL_MULT': [1.0, 1.25, 1.5]
}

# Problematic symbols to skip (updated with more symbols)
SKIP_SYMBOLS = ['FARTCOIN/USDT', 'FARTCOINUSDT', 'XPIN/USDT', 'MOODENG/USDT', 
                'DOLO/USDT', 'HOLO/USDT', 'WLFI/USDT', 'CUDIS/USDT', 'HYPE/USDT']

# ========================== PRE-RUN CHECKS ===========================
if not BOT_TOKEN or not OWNER_CHAT_ID:
    logger.critical("CRITICAL: BOT_TOKEN or CRYPTO_OWNER_ID environment variable not set. Exiting.")
    exit()
try:
    OWNER_CHAT_ID_INT = int(OWNER_CHAT_ID)
except ValueError:
    logger.critical("CRITICAL: CRYPTO_OWNER_ID must be a valid integer. Exiting.")
    exit()

# ======================= ENHANCED SYSTEMS IMPLEMENTATION =======================

# Database connection context manager
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
    try:
        yield conn
    finally:
        conn.close()

# Safe database operation wrapper
def safe_db_operation(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        return None

# 1. Performance Tracker
class PerformanceTracker:
    def __init__(self):
        self.today = date.today().isoformat()
        self.filename = f"performance_{self.today}.csv"
        self._init_csv()
    
    def _init_csv(self):
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'symbol', 'direction', 'entry', 
                'exit', 'pnl_percent', 'confidence', 'duration',
                'market_regime', 'volume_ratio', 'rsi', 'atr_ratio'
            ])
    
    def record_trade(self, trade_data):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_data.get('timestamp', datetime.now().isoformat()),
                trade_data.get('symbol', ''),
                trade_data.get('direction', ''),
                trade_data.get('entry_price', ''),
                trade_data.get('exit_price', ''),
                trade_data.get('pnl_percent', ''),
                trade_data.get('confidence', ''),
                trade_data.get('duration_minutes', ''),
                trade_data.get('market_regime', ''),
                trade_data.get('volume_ratio', ''),
                trade_data.get('rsi', ''),
                trade_data.get('atr_ratio', '')
            ])

# 2. Risk Manager
class RiskManager:
    def __init__(self, portfolio_value):
        self.portfolio_value = portfolio_value
        self.daily_pnl = 0
        self.max_daily_loss = portfolio_value * MAX_DAILY_LOSS
        self.open_positions = {}
        self.max_concurrent_trades = MAX_CONCURRENT_TRADES
    
    def add_position(self, symbol, size_usd):
        self.open_positions[symbol] = size_usd
    
    def remove_position(self, symbol):
        if symbol in self.open_positions:
            del self.open_positions[symbol]
    
    def can_open_trade(self, symbol, proposed_size_usd):
        if self.daily_pnl <= -self.max_daily_loss:
            return False, "Daily loss limit exceeded"
        
        if len(self.open_positions) >= self.max_concurrent_trades:
            return False, "Max concurrent trades reached"
        
        if symbol in self.open_positions:
            return False, "Already in this symbol"
        
        if proposed_size_usd > self.portfolio_value * MAX_POSITION_SIZE:
            return False, "Position size too large"
        
        return True, "OK"
    
    def update_daily_pnl(self, pnl):
        self.daily_pnl += pnl
    
    def reset_daily_pnl(self):
        self.daily_pnl = 0
        logger.info("Daily PnL reset")

# 3. Model Validator
class ModelValidator:
    def __init__(self):
        self.predictions = []
        self.actuals = []
    
    def record_prediction(self, confidence, actual_pnl):
        self.predictions.append(confidence)
        self.actuals.append(1 if actual_pnl > 0 else 0)
    
    def calculate_model_performance(self):
        if len(self.predictions) < 10:
            return "Insufficient data"
        
        correct = sum(1 for p, a in zip(self.predictions, self.actuals) 
                     if (p > 0.5 and a == 1) or (p <= 0.5 and a == 0))
        accuracy = correct / len(self.predictions)
        
        return {
            'accuracy': accuracy,
            'total_predictions': len(self.predictions),
            'precision': self.calculate_precision(),
            'recall': self.calculate_recall()
        }
    
    def calculate_precision(self):
        true_positives = sum(1 for p, a in zip(self.predictions, self.actuals) 
                            if p > 0.5 and a == 1)
        false_positives = sum(1 for p, a in zip(self.predictions, self.actuals) 
                             if p > 0.5 and a == 0)
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    def calculate_recall(self):
        true_positives = sum(1 for p, a in zip(self.predictions, self.actuals) 
                            if p > 0.5 and a == 1)
        false_negatives = sum(1 for p, a in zip(self.predictions, self.actuals) 
                             if p <= 0.5 and a == 1)
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

# 4. Multi-Timeframe Analyzer
class MultiTimeframeAnalyzer:
    def __init__(self):
        self.timeframes = ['15m', '30m', '1h', '4h']
        self.weights = {'15m': 0.15, '30m': 0.25, '1h': 0.35, '4h': 0.25}
    
    async def analyze_multi_timeframe(self, symbol, exchange):
        analysis = {}
        
        for tf in self.timeframes:
            df = await fetch_ohlcv_cached(symbol, tf, limit=100, exchange=exchange)
            if df is not None and len(df) > 20:
                analysis[tf] = self.analyze_timeframe(df)
        
        return self.calculate_confluence_score(analysis)
    
    def calculate_confluence_score(self, analysis):
        if not analysis:
            return 0.5  # Neutral score if no data
            
        score = 0
        for tf, data in analysis.items():
            if tf in self.weights:
                score += data['trend_score'] * self.weights[tf]
                score += data['momentum_score'] * self.weights[tf] * 0.5
        
        return max(0, min(1, score))  # Normalize to 0-1 range
    
    def analyze_timeframe(self, df):
        # Calculate trend score (EMA cross)
        ema20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        trend_score = 1.0 if ema20 > ema50 else 0.0
        
        # Calculate momentum score (RSI)
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean().iloc[-1]
        loss = -delta.clip(upper=0).rolling(window=14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
        momentum_score = (rsi - 50) / 50  # Normalize to -1 to 1, then scale to 0-1
        
        # Calculate volume score
        volume_avg = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / volume_avg if volume_avg > 0 else 1
        volume_score = min(1.0, volume_ratio / 2.0)  # Cap at 1.0
        
        return {
            'trend_score': trend_score,
            'momentum_score': (momentum_score + 1) / 2,  # Convert to 0-1 range
            'volume_score': volume_score
        }

# 5. Exchange Factory with improved connection settings
class ExchangeFactory:
    def __init__(self):
        self.exchanges = {}
        
    def get_exchange(self, exchange_name='binance'):
        if exchange_name not in self.exchanges:
            exchange_config = {
                'enableRateLimit': True,
                'options': {'defaultType': 'future'},
                'timeout': 30000,
                'rateLimit': 1000,
            }
            
            if exchange_name == 'binance':
                exchange_config.update({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True,
                    }
                })
                self.exchanges[exchange_name] = ccxt.binanceusdm(exchange_config)
            elif exchange_name == 'bybit':
                self.exchanges[exchange_name] = ccxt.bybit({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
            elif exchange_name == 'okx':
                self.exchanges[exchange_name] = ccxt.okx({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
            else:
                raise ValueError(f"Unsupported exchange: {exchange_name}")
                
            # Manage connection pool
            self.manage_connection_pool(self.exchanges[exchange_name])
            
        return self.exchanges[exchange_name]
    
    def manage_connection_pool(self, exchange):
        # Increase connection pool size
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        exchange.session.mount('https://', adapter)
        exchange.session.mount('http://', adapter)

# 6. Portfolio Optimizer (Simplified)
class PortfolioOptimizer:
    def __init__(self):
        self.correlation_matrix = None
    
    async def optimize_allocation(self, signals, portfolio_value):
        if not signals:
            return {}
        
        # Simple allocation based on confidence score
        total_confidence = sum(signal.get('confidence', 0.5) for signal in signals)
        if total_confidence == 0:
            return {}
            
        allocations = {}
        for signal in signals:
            symbol = signal['symbol']
            confidence = signal.get('confidence', 0.5)
            allocations[symbol] = (confidence / total_confidence) * portfolio_value * 0.8  # Use 80% of portfolio
        
        return allocations

# Initialize enhanced systems
exchange_factory = ExchangeFactory()
performance_tracker = PerformanceTracker()
risk_manager = RiskManager(PORTFOLIO_VALUE)
model_validator = ModelValidator()
multi_timeframe_analyzer = MultiTimeframeAnalyzer()
portfolio_optimizer = PortfolioOptimizer()

# Initialize database
def init_db(path=DB_PATH):
    with get_db_connection() as conn:
        # Create tables if they don't exist
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                market_id TEXT, 
                symbol TEXT, 
                direction TEXT,
                entry_price REAL, 
                entry_time TEXT, 
                tp1 REAL, 
                tp2 REAL, 
                tp3 REAL, 
                sl REAL,
                position_size REAL,
                tp1_hit INTEGER DEFAULT 0, 
                tp2_hit INTEGER DEFAULT 0, 
                tp3_hit INTEGER DEFAULT 0,
                exit_price REAL, 
                exit_time TEXT, 
                pnl REAL, 
                status TEXT DEFAULT 'open',
                confidence REAL,
                market_regime TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                market_id TEXT, timeframe TEXT, fetched_at TEXT, blob BLOB,
                PRIMARY KEY(market_id, timeframe)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                total_predictions INTEGER
            )
        """)
        
        # Check for missing columns and add them
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(signals)")
        existing_columns = [column[1] for column in cursor.fetchall()]
        
        if 'confidence' not in existing_columns:
            conn.execute("ALTER TABLE signals ADD COLUMN confidence REAL")
            logger.info("Added confidence column to signals table")
            
        if 'market_regime' not in existing_columns:
            conn.execute("ALTER TABLE signals ADD COLUMN market_regime TEXT")
            logger.info("Added market_regime column to signals table")
            
    logger.info("Database initialized successfully.")

init_db()

# Initialize primary exchange
exchange = exchange_factory.get_exchange('binance')
hybrid_model = load_hybrid_model(HYBRID_MODEL_PATH, HybridModel)

# ============================= AUTHORIZATION & UTILITIES ===============================
def is_authorized(update: Update) -> bool:
    user = update.effective_user
    if user and str(user.id) == str(OWNER_CHAT_ID):
        return True
        
    logger.warning(f"Unauthorized access attempt by user ID: {user.id if user else 'Unknown'}")
    
    if user and update.message:
        update.message.reply_text(
            "🚫 You are not authorized to use this bot.\n\n"
            f"Your ID: {user.id}\n"
            f"Authorized ID: {OWNER_CHAT_ID}"
        )
        
    return False

async def to_thread(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

def _pickle_df(df: pd.DataFrame) -> bytes: 
    return pickle.dumps(df)

def _unpickle_df(blob: bytes):
    try: 
        return pickle.loads(blob)
    except Exception: 
        return None

async def send_notification(context: ContextTypes.DEFAULT_TYPE, message: str):
    await context.bot.send_message(chat_id=OWNER_CHAT_ID_INT, text=message, parse_mode=ParseMode.MARKDOWN)

# Rate limiting decorator
def rate_limited(max_per_second):
    min_interval = 1.0 / max_per_second
    def decorate(func):
        last_time_called = 0.0
        @functools.wraps(func)
        def rate_limited_function(*args, **kwargs):
            nonlocal last_time_called
            elapsed = time.time() - last_time_called
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_time_called = time.time()
            return ret
        return rate_limited_function
    return decorate

# Retry mechanism
async def fetch_with_retry(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"Attempt {attempt+1} failed, retrying in {wait_time}s: {e}")
            await asyncio.sleep(wait_time)
    return None

# ================= UPGRADED: OHLCV, INDICATORS & FEATURES =======================
@rate_limited(3)  # Reduced from 5 to 3 requests per second
async def fetch_ohlcv_cached(market_id: str, timeframe: str, limit: int = 100, exchange=exchange):  # Reduced default limit
    now = datetime.now(timezone.utc)
    
    with get_db_connection() as conn:
        row = conn.execute("SELECT fetched_at, blob FROM ohlcv_cache WHERE market_id=? AND timeframe=?", 
                          (market_id, timeframe)).fetchone()
        
        if row:
            fetched_at = datetime.fromisoformat(row[0])
            cache_age = (now - fetched_at).total_seconds()
            
            if cache_age < CACHE_TTL:
                df = _unpickle_df(row[1])
                if df is not None and len(df) >= 50:  # Reduced requirement from 80% to 50 candles
                    return df.copy()
    
    # If cache is stale or insufficient, fetch new data
    try:
        bars = await fetch_with_retry(to_thread, exchange.fetch_ohlcv, market_id, timeframe, limit=limit)
        if not bars or len(bars) < 20:  # Require minimum bars
            logger.warning(f"Insufficient data for {market_id}: {len(bars) if bars else 0} bars")
            return None
            
        df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        # Add data validation
        if df['close'].isna().any() or df['volume'].isna().any():
            logger.warning(f"NaN values detected in {market_id} data")
            return None
            
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        
        blob = _pickle_df(df)
        with get_db_connection() as conn:
            conn.execute("REPLACE INTO ohlcv_cache(market_id, timeframe, fetched_at, blob) VALUES (?,?,?,?)",
                         (market_id, timeframe, now.isoformat(), blob))
        
        return df
    except ccxt.BadSymbol as e:
        logger.warning(f"Invalid symbol {market_id}: {e}")
        return None
    except ccxt.BadRequest as e:
        logger.warning(f"Bad request for {market_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {market_id}: {e}")
        return None

def compute_advanced_features(df):
    df['candle_vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3)
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    hurst_exp = log_returns.rolling(window=100).std()
    df['hurst'] = hurst_exp
    df['market_regime'] = np.where(df['hurst'] > hurst_exp.quantile(0.7), 'trending', 
                                   np.where(df['hurst'] < hurst_exp.quantile(0.3), 'mean_reverting', 'choppy'))
    df['spread_ratio'] = (df['high'] - df['low']) / df['volume'].replace(0, 1e-6)
    df['absorption'] = (df['volume'] * (df['high'] - df['low'])).rolling(14).sum()
    return df

def compute_indicators(df: pd.DataFrame):
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(window=RSI_PERIOD).mean()
    loss = -delta.clip(upper=0).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = df["close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(ATR_PERIOD, min_periods=ATR_PERIOD).mean()
    
    df = compute_advanced_features(df)
    return df

def predict_signal_confidence(df: pd.DataFrame, seq_len=50) -> float:  # Reduced seq_len from 60 to 50
    if hybrid_model is None or not PYTORCH_AVAILABLE:
        return calculate_fallback_confidence(df)

    if len(df) < seq_len:
        logger.warning(f"Not enough data ({len(df)} candles) for prediction. Need {seq_len}.")
        return calculate_fallback_confidence(df)

    try:
        feature_cols = ['close', 'volume', 'rsi', 'macd', 'macd_sig', 'atr', 'candle_vwap', 'hurst', 'spread_ratio', 'absorption']
        df_features = df[feature_cols].dropna()

        if len(df_features) < 30:  # Reduced requirement from seq_len to 30
            logger.warning(f"Not enough feature data after dropna ({len(df_features)}) for prediction.")
            return calculate_fallback_confidence(df)
        
        # Use whatever data we have, even if less than seq_len
        actual_seq_len = min(seq_len, len(df_features))
        data_subset = df_features.tail(actual_seq_len).values
        scaled_data = (data_subset - np.mean(data_subset, axis=0)) / (np.std(data_subset, axis=0) + 1e-7)

        device = next(hybrid_model.parameters()).device
        data_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = hybrid_model(data_tensor)
        
        return prediction.item()
    except Exception as e:
        logger.error(f"Hybrid Model prediction failed: {e}. Using fallback confidence.")
        return calculate_fallback_confidence(df)

def calculate_fallback_confidence(df):
    """Fallback confidence calculation when model fails"""
    try:
        # Simple logic based on recent price action and volume
        recent_returns = df['close'].pct_change().tail(5)
        avg_return = recent_returns.mean() if not pd.isna(recent_returns.mean()) else 0
        
        # Safe volume ratio calculation
        vol_mean = df['volume'].rolling(20).mean()
        if len(vol_mean) > 1 and vol_mean.iloc[-2] > 0:
            vol_ratio = df['volume'].iloc[-1] / vol_mean.iloc[-2]
        else:
            vol_ratio = 1.0  # Default neutral value
        
        # Combine factors for a simple confidence score (0-1)
        confidence = 0.5  # Neutral starting point
        confidence += min(0.2, max(-0.2, avg_return * 10))  # Scale returns
        confidence += min(0.2, max(-0.2, (vol_ratio - 1) * 0.1))  # Scale volume
        
        return max(0.1, min(0.9, confidence))  # Keep within reasonable bounds
    except Exception as e:
        logger.error(f"Fallback confidence calculation failed: {e}")
        return 0.5  # Return neutral confidence as last resort

# ================= NEW: ORDER BOOK & DYNAMIC SIZING ===================
async def fetch_orderbook_features(symbol, exchange=exchange):
    try:
        ob = await to_thread(exchange.fetch_order_book, symbol, limit=10)
        bids = ob.get('bids', [])
        asks = ob.get('asks', [])
        if not bids or not asks: 
            return {}

        mid_price = (bids[0][0] + asks[0][0]) / 2
        spread = asks[0][0] - bids[0][0]
        
        bid_volume = sum(b[1] for b in bids[:5])
        ask_volume = sum(a[1] for a in asks[:5])
        total_volume = bid_volume + ask_volume
        
        imbalance = (bid_volume - ask_volume) / total_volume if total_volume > 0 else 0
        
        return {
            'mid_price': mid_price,
            'spread': spread,
            'imbalance': imbalance,
            'micro_alpha': (imbalance * spread) / mid_price if mid_price > 0 else 0
        }
    except Exception as e:
        logger.warning(f"Could not fetch order book for {symbol}: {e}")
        return {}

def dynamic_position_sizing(portfolio_value, volatility_usd, confidence):
    if volatility_usd <= 0: 
        return 0

    edge = confidence - 0.5
    if edge <= 0: 
        return 0

    max_risk_fraction = MAX_DAILY_LOSS
    suggested_risk_fraction = edge * 0.1
    
    risk_fraction = min(max_risk_fraction, suggested_risk_fraction)
    
    usd_to_risk = portfolio_value * risk_fraction
    position_size = usd_to_risk / volatility_usd
    return position_size

# ================================ CHARTING =================================
async def plot_annotated_chart(df: pd.DataFrame, display_symbol: str, entry: float, sl: float, tps: list) -> str:
    def _plot():
        df_plot = df.tail(CHART_CANDLES).copy()
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='#F0F0F0')
        ax.set_facecolor('#FFFFFF')
        up = df_plot[df_plot.close >= df_plot.open]
        down = df_plot[df_plot.close < df_plot.open]
        width, width2 = 0.005, 0.0005
        ax.bar(up.index, up.close-up.open, width, bottom=up.open, color='#26a69a')
        ax.bar(up.index, up.high-up.close, width2, bottom=up.close, color='#26a69a')
        ax.bar(up.index, up.low-up.open, width2, bottom=up.open, color='#26a69a')
        ax.bar(down.index, down.close-down.open, width, bottom=down.open, color='#ef5350')
        ax.bar(down.index, down.high-down.open, width2, bottom=down.open, color='#ef5350')
        ax.bar(down.index, down.low-down.close, width2, bottom=down.close, color='#ef5350')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=30)
        plt.ylabel("Price (USDT)")
        plt.title(f"{display_symbol} Signal ({TIMEFRAME})", fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)
        x_min, x_max = mdates.date2num(df_plot.index[0]), mdates.date2num(df_plot.index[-1])
        ax.hlines(entry, x_min, x_max, colors='green', linestyles="--", label=f"Entry {entry:,.4f}")
        ax.hlines(sl, x_min, x_max, colors='red', linestyles="--", label=f"SL {sl:,.4f}")
        for i, tp in enumerate(tps, start=1):
            ax.hlines(tp, x_min, x_max, colors='blue', alpha=0.6, linestyles="--", label=f"TP{i} {tp:,.4f}")
        plt.legend()
        plt.tight_layout()
        fname = f"chart_{display_symbol.replace('/', '')}_{int(datetime.now(timezone.utc).timestamp())}.png"
        plt.savefig(fname, dpi=150)
        
        # Clean up
        plt.close(fig)
        del df_plot, fig, ax
        gc.collect()
        
        return fname
    return await to_thread(_plot)

# ================= UPGRADED: SIGNAL GENERATION & ALERTS ============================
async def generate_signal(market_id: str, display_symbol: str, cooldowns: dict, exchange=exchange):
    try:
        # Skip problematic symbols
        if any(skip in display_symbol for skip in SKIP_SYMBOLS):
            return None
            
        # Fetch data: OHLCV and new Order Book features
        df = await fetch_ohlcv_cached(market_id, TIMEFRAME, limit=400, exchange=exchange)
        ob_features = await fetch_orderbook_features(display_symbol, exchange=exchange)
        
        if df is None or len(df) < 100: 
            return None
        df = compute_indicators(df)

        last = df.iloc[-2]
        last_time = df.index[-2]
        if market_id in cooldowns and cooldowns[market_id] >= last_time: 
            return None

        # --- Model Confidence Check ---
        confidence = await to_thread(predict_signal_confidence, df)
        
        # --- Multi-Timeframe Analysis ---
        confluence_score = await multi_timeframe_analyzer.analyze_multi_timeframe(market_id, exchange)
        combined_confidence = (confidence + confluence_score) / 2
        
        if combined_confidence < CONFIDENCE_THRESHOLD:
            logger.info(f"Signal for {display_symbol} skipped. Confidence {combined_confidence:.2f} < {CONFIDENCE_THRESHOLD}")
            return None

        # --- Base Strategy Conditions ---
        is_uptrend = last["ema20"] > last["ema50"]
        is_downtrend = last["ema20"] < last["ema50"]
        
        # --- NEW Advanced Conditions ---
        is_trending_regime = last["market_regime"] == 'trending'
        has_volume_spike = last["volume"] > df["volume"].rolling(20).mean().iloc[-2] * 1.5
        has_pos_imbalance = ob_features.get('imbalance', 0) > 0.1
        has_neg_imbalance = ob_features.get('imbalance', 0) < -0.1
        
        signal = None
        if is_uptrend and is_trending_regime and last["rsi"] > 52 and has_volume_spike and has_pos_imbalance:
            signal = "Long"
        elif is_downtrend and is_trending_regime and last["rsi"] < 48 and has_volume_spike and has_neg_imbalance:
            signal = "Short"

        if signal:
            entry_price = float(last["close"])
            atr = float(last["atr"]) if not pd.isna(last["atr"]) else entry_price * 0.02
            
            # --- NEW: Dynamic Position Sizing Calculation ---
            sl_distance_usd = atr * SL_MULT
            position_size_coin = dynamic_position_sizing(PORTFOLIO_VALUE, sl_distance_usd, combined_confidence)
            
            # --- Risk Management Check ---
            position_value = position_size_coin * entry_price
            can_trade, reason = risk_manager.can_open_trade(display_symbol, position_value)
            if not can_trade:
                logger.info(f"Risk management blocked trade for {display_symbol}: {reason}")
                return None
            
            # Add position to risk manager
            risk_manager.add_position(display_symbol, position_value)
            
            if signal == "Long":
                sl = entry_price - sl_distance_usd
                tps = [entry_price + atr * m for m in TP_MULT]
            else:
                sl = entry_price + sl_distance_usd
                tps = [entry_price - atr * m for m in TP_MULT]

            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO signals (market_id, symbol, direction, entry_price, entry_time, tp1, tp2, tp3, sl, position_size, confidence, market_regime, status)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')",
                    (
                        market_id,
                        display_symbol,
                        signal.lower(),
                        entry_price,
                        pd.Timestamp(last_time).isoformat(),
                        tps[0],
                        tps[1],
                        tps[2],
                        sl,
                        position_size_coin,
                        combined_confidence,
                        last['market_regime']
                    ),
                )
                conn.commit()
                signal_id = cursor.lastrowid
            
            cooldowns[market_id] = last_time
            
            chart_file = await plot_annotated_chart(df, display_symbol, entry_price, sl, tps)
            alert_text = format_alert(display_symbol, signal, entry_price, sl, tps, combined_confidence, position_size_coin, last['market_regime'])
            
            return {"text": alert_text, "chart": chart_file, "signal_id": signal_id}

    except Exception as e:
        logger.exception(f"Error generating signal for {market_id}: {e}")
    return None

def format_alert(symbol, side, entry, sl, tps, confidence, position_size, regime):
    return (
        f"📩 *{symbol} Signal ({TIMEFRAME})*\n\n"
        f"{'🚀' if side == 'Long' else '📉'} *Trade Type:* {side.upper()}\n"
        f"🧠 *Confidence:* {confidence*100:.1f}%\n"
        f"📈 *Market Regime:* {regime.title()}\n\n"
        f"*Trade Parameters:*\n"
        f"  - Entry: {entry:,.4f}\n"
        f"  - Stop-loss: {sl:,.4f}\n\n"
        f"*Take-Profit Targets:*\n"
        f"  - TP1: {tps[0]:,.4f}\n"
        f"  - TP2: {tps[1]:,.4f}\n"
        f"  - TP3: {tps[2]:,.4f}\n\n"
        f"*Sizing & Risk (Based on ${PORTFOLIO_VALUE:,} portfolio):*\n"
        f"  - Suggested Size: {position_size:.4f} {symbol.split('/')[0]}\n"
        f"  - Position Value: ${(position_size * entry):,.2f}\n"
        f"💡 Move SL to entry after TP1 is hit."
    )

async def send_alert(context: ContextTypes.DEFAULT_TYPE, alert_data: dict):
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Move SL to BE", callback_data=f"be:{alert_data['signal_id']}"),
        InlineKeyboardButton("Close Manually", callback_data=f"close:{alert_data['signal_id']}")
    ]])
    chart_path = alert_data["chart"]
    try:
        with open(chart_path, "rb") as chart_photo:
            await context.bot.send_photo(
                chat_id=OWNER_CHAT_ID_INT, photo=chart_photo, caption=alert_data["text"],
                reply_markup=kb, parse_mode=ParseMode.MARKDOWN
            )
    except Exception as e:
        logger.error(f"Failed to send photo alert: {e}")
        await send_notification(context, "⚠️ Error sending chart image:\n\n" + alert_data["text"])
    finally:
        if os.path.exists(chart_path): 
            os.remove(chart_path)

# ====================== BACKGROUND JOBS & TELEGRAM ======================
async def scan_markets(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Scanning for new signals with upgraded logic...")
    try:
        markets = await to_thread(exchange.load_markets)
        tickers = await to_thread(exchange.fetch_tickers)
        
        usdt_futures = [t for t in tickers.values() 
                       if 'USDT' in t['symbol'] 
                       and t.get('quoteVolume') is not None
                       and not any(skip in t['symbol'] for skip in SKIP_SYMBOLS)]  # Filter out problematic symbols
        
        # Get top markets but process in smaller batches
        top_markets = sorted(usdt_futures, key=lambda t: t['quoteVolume'], reverse=True)[:TOP_N_MARKETS]
        
        cooldowns = defaultdict(lambda: datetime.min.replace(tzinfo=timezone.utc))
        
        # Process markets in smaller batches to reduce memory pressure
        batch_size = 5
        for i in range(0, len(top_markets), batch_size):
            batch = top_markets[i:i+batch_size]
            tasks = [generate_signal(t['info']['symbol'], t['symbol'], cooldowns, exchange) for t in batch]
            
            for alert_data in await asyncio.gather(*tasks):
                if alert_data:
                    await send_alert(context, alert_data)
            
            # Clear memory between batches
            if i + batch_size < len(top_markets):
                await cleanup_memory(context)
                
    except Exception as e:
        logger.error(f"Error during market scan: {e}")
        await send_notification(context, f"⚠️ An error occurred during the market scan:\n`{e}`")

async def monitor_signals(context: ContextTypes.DEFAULT_TYPE):
    with get_db_connection() as conn:
        open_signals = conn.execute("SELECT id, market_id, symbol, direction, entry_price, tp1, tp2, tp3, sl, tp1_hit, tp2_hit FROM signals WHERE status='open'").fetchall()
    if not open_signals: 
        return
    try:
        market_ids = list(set([s[1] for s in open_signals]))
        symbols = [exchange.market(mid)['symbol'] for mid in market_ids]
        tickers = await to_thread(exchange.fetch_tickers, symbols)
    except Exception as e:
        logger.error(f"Could not fetch tickers for monitoring: {e}")
        return
        
    for sig in open_signals:
        sig_id, m_id, sym, direction, entry, tp1, tp2, tp3, sl, tp1_hit, tp2_hit = sig
        ticker = tickers.get(exchange.market(mid)['symbol'])
        if not ticker or 'last' not in ticker or ticker['last'] is None: 
            continue
            
        price = ticker['last']
        now_iso = datetime.now(timezone.utc).isoformat()
        
        async def close_position(exit_price, pnl_percent, status_msg):
            with get_db_connection() as conn_update:
                conn_update.execute("UPDATE signals SET status='closed', exit_price=?, exit_time=?, pnl=? WHERE id=?", 
                                  (exit_price, now_iso, pnl_percent, sig_id))
            
            # Remove position from risk manager
            risk_manager.remove_position(sym)
            
            # Record trade for performance tracking
            trade_data = {
                'symbol': sym,
                'direction': direction,
                'entry_price': entry,
                'exit_price': exit_price,
                'pnl_percent': pnl_percent,
                'timestamp': now_iso
            }
            performance_tracker.record_trade(trade_data)
            
            # Update risk manager
            risk_manager.update_daily_pnl(pnl_percent)
            
            await send_notification(context, 
                f"✅ *Position Closed*\nSymbol: {sym}\nReason: {status_msg}\nExit Price: {exit_price:,.4f}\nPnL: {pnl_percent:.2f}%")
            
        try:
            if direction == 'long':
                if price <= sl: 
                    await close_position(sl, (sl - entry) / entry * 100, "Stop-Loss Hit")
                    continue
                if price >= tp3 and not tp2_hit: 
                    await close_position(tp3, (tp3 - entry) / entry * 100, "TP3 Hit")
                    continue
                if not tp2_hit and price >= tp2:
                    with get_db_connection() as c: 
                        c.execute("UPDATE signals SET tp2_hit=1 WHERE id=?", (sig_id,))
                    await send_notification(context, f"🎯 *TP2 Hit* for {sym}!")
                if not tp1_hit and price >= tp1:
                    with get_db_connection() as c: 
                        c.execute("UPDATE signals SET tp1_hit=1, sl=? WHERE id=?", (entry, sig_id))
                    await send_notification(context, f"🎯 *TP1 Hit* for {sym}! SL moved to BE ({entry:,.4f}).")
            elif direction == 'short':
                if price >= sl: 
                    await close_position(sl, (entry - sl) / entry * 100, "Stop-Loss Hit")
                    continue
                if price <= tp3 and not tp2_hit: 
                    await close_position(tp3, (entry - tp3) / entry * 100, "TP3 Hit")
                    continue
                if not tp2_hit and price <= tp2:
                    with get_db_connection() as c: 
                        c.execute("UPDATE signals SET tp2_hit=1 WHERE id=?", (sig_id,))
                    await send_notification(context, f"🎯 *TP2 Hit* for {sym}!")
                if not tp1_hit and price <= tp1:
                    with get_db_connection() as c: 
                        c.execute("UPDATE signals SET tp1_hit=1, sl=? WHERE id=?", (entry, sig_id))
                    await send_notification(context, f"🎯 *TP1 Hit* for {sym}! SL moved to BE ({entry:,.4f}).")
        except Exception as e: 
            logger.error(f"Error monitoring signal {sig_id}: {e}")

async def daily_report(context: ContextTypes.DEFAULT_TYPE):
    logger.info("Generating daily report...")
    report = get_pnl_summary(days=1)
    
    # Add model performance to report
    model_perf = model_validator.calculate_model_performance()
    if isinstance(model_perf, dict):
        model_report = (f"\n🤖 *Model Performance:*\n"
                       f"  - Accuracy: {model_perf['accuracy']*100:.2f}%\n"
                       f"  - Precision: {model_perf['precision']*100:.2f}%\n"
                       f"  - Recall: {model_perf['recall']*100:.2f}%\n"
                       f"  - Predictions: {model_perf['total_predictions']}")
        report += model_report
    
    # Add risk management status
    risk_report = (f"\n⚠️ *Risk Status:*\n"
                   f"  - Daily PnL: {risk_manager.daily_pnl:.2f}%\n"
                   f"  - Daily Limit: -{MAX_DAILY_LOSS*100:.0f}%\n"
                   f"  - Open Positions: {len(risk_manager.open_positions)}/{MAX_CONCURRENT_TRADES}")
    report += risk_report
    
    await send_notification(context, f"📅 *Daily PNL Report*\n\n{report}")
    
    # Reset daily PnL
    risk_manager.reset_daily_pnl()

def get_pnl_summary(days=None):
    query = "SELECT pnl FROM signals WHERE status='closed' AND pnl IS NOT NULL"
    params = []
    if days:
        query += " AND exit_time >= ?"
        params.append((datetime.now(timezone.utc) - timedelta(days=days)).isoformat())
    with get_db_connection() as conn:
        pnls = [row[0] for row in conn.execute(query, params).fetchall()]
    if not pnls: 
        return "No closed trades found for this period."
    total_trades, wins = len(pnls), [p for p in pnls if p > 0]
    return f"  - Total Trades: {total_trades}\n  - Win Rate: {(len(wins) / total_trades * 100):.2f}%\n  - Total PNL: {sum(pnls):.2f}%"

# Add a function to clear memory-intensive objects
def clear_large_objects():
    """Clear large objects from memory"""
    large_vars = [var for var in globals().items() if 
                 isinstance(var[1], (pd.DataFrame, np.ndarray)) and 
                 hasattr(var[1], 'nbytes') and var[1].nbytes > 1000000]  # 1MB threshold
    
    for name, obj in large_vars:
        if name not in ['exchange', 'performance_tracker', 'risk_manager']:  # Keep essential objects
            logger.info(f"Clearing large object: {name} ({obj.nbytes/1000000:.2f}MB)")
            globals()[name] = None
    
    gc.collect()

# Performance monitoring function
async def monitor_performance(context: ContextTypes.DEFAULT_TYPE):
    try:
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        if memory_usage > 300:  # Reduced from 400 to 300MB
            logger.warning(f"High memory usage: {memory_usage:.2f}MB, running garbage collection")
            gc.collect()
            clear_large_objects()
            
        if memory_usage > 400:  # Reduced from 500 to 400MB
            logger.warning(f"Critical memory usage: {memory_usage:.2f}MB")
            # Clear cache if memory is too high
            with get_db_connection() as conn:
                conn.execute("DELETE FROM ohlcv_cache WHERE fetched_at < datetime('now', '-1 hour')")
            
        if memory_usage > 500:  # Emergency measures
            logger.warning(f"Emergency memory usage: {memory_usage:.2f}MB, restarting may be needed")
            
        cpu_percent = process.cpu_percent()
        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent}%")
            
        # Log performance metrics
        logger.info(f"Performance - Memory: {memory_usage:.2f}MB, CPU: {cpu_percent}%")
    except Exception as e:
        logger.error(f"Error monitoring performance: {e}")

# Memory cleanup function (fixed to accept context parameter)
async def cleanup_memory(context: ContextTypes.DEFAULT_TYPE = None):
    logger.info("Running memory cleanup...")
    gc.collect()
    if PYTORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clear OHLCV cache for older entries
    with get_db_connection() as conn:
        conn.execute("DELETE FROM ohlcv_cache WHERE fetched_at < datetime('now', '-2 hours')")
    
    logger.info("Memory cleanup completed")

# Database backup function
async def backup_database(context: ContextTypes.DEFAULT_TYPE = None):
    if os.path.exists(DB_PATH):
        backup_path = f"{DB_PATH}.backup.{datetime.now().strftime('%Y%m%d')}"
        shutil.copy2(DB_PATH, backup_path)
        logger.info(f"Database backed up to {backup_path}")

# Deployment notification
async def send_deployment_notification(context: ContextTypes.DEFAULT_TYPE):
    try:
        commit_hash = os.environ.get('RENDER_GIT_COMMIT', 'unknown')
        await send_notification(
            context, 
            f"🚀 Bot deployed successfully!\nCommit: {commit_hash}\nTime: {datetime.now()}"
        )
    except Exception as e:
        logger.error(f"Failed to send deployment notification: {e}")

# ============================== TELEGRAM COMMANDS ==============================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
    await update.message.reply_text("👋 *Power Crypto Bot (Upgraded)* Activated\n\nCommands:\n`/start`, /help, /status, /pnl, /forcescan, /testalert", parse_mode=ParseMode.MARKDOWN)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
    with get_db_connection() as conn:
        open_signals = conn.execute("SELECT symbol, direction, entry_price, sl, tp1, tp2, tp3 FROM signals WHERE status='open'").fetchall()
    if not open_signals: 
        await update.message.reply_text("No open positions.")
        return
    message = "📊 *Current Open Positions:\n\n" + "".join([f"{s[0]} ({s[1].upper()})*\n - Entry: {s[2]:.4f}\n - SL: {s[3]:.4f}\n\n" for s in open_signals])
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

async def pnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
    await update.message.reply_text(f"📈 *Performance Summary\n\nLast 24 Hours:\n{get_pnl_summary(days=1)}\n\nAll Time:*\n{get_pnl_summary()}", parse_mode=ParseMode.MARKDOWN)

async def model_performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
    perf = model_validator.calculate_model_performance()
    if isinstance(perf, dict):
        message = (f"🤖 *Model Performance Report*\n\n"
                  f"Accuracy: {perf['accuracy']*100:.2f}%\n"
                  f"Precision: {perf['precision']*100:.2f}%\n"
                  f"Recall: {perf['recall']*100:.2f}%\n"
                  f"Total Predictions: {perf['total_predictions']}")
    else:
        message = perf
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

async def risk_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
    message = (f"⚠️ *Risk Management Status*\n\n"
               f"Daily PnL: {risk_manager.daily_pnl:.2f}%\n"
               f"Daily Limit: -{MAX_DAILY_LOSS*100:.0f}%\n"
               f"Open Positions: {len(risk_manager.open_positions)}/{MAX_CONCURRENT_TRADES}\n"
               f"Max Position Size: {MAX_POSITION_SIZE*100:.0f}% of portfolio")
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not is_authorized(update): 
        await query.edit_message_text(text="Unauthorized.")
        return
    action, signal_id_str = query.data.split(":")
    signal_id = int(signal_id_str)
    with get_db_connection() as conn:
        sig = conn.execute("SELECT market_id, symbol, direction, entry_price, status FROM signals WHERE id=?", (signal_id,)).fetchone()
    if not sig or sig[4] != 'open': 
        await query.edit_message_caption(caption=query.message.caption_markdown + "\n\n*Action failed: Trade closed.*", parse_mode=ParseMode.MARKDOWN)
        return

    market_id, symbol, direction, entry_price, _ = sig
    if action == "be":
        with get_db_connection() as c: 
            c.execute("UPDATE signals SET sl=? WHERE id=?", (entry_price, signal_id))
        msg = f"🛠️ *Manual:* SL for {symbol} moved to BE ({entry_price:,.4f})."
        await send_notification(context, msg)
        await query.edit_message_caption(caption=query.message.caption_markdown + f"\n\n*{msg}*", parse_mode=ParseMode.MARKDOWN)
    elif action == "close":
        try:
            ticker = await to_thread(exchange.fetch_ticker, exchange.market(market_id)['symbol'])
            price = ticker['last']
            pnl = ((price-entry_price)/entry_price*100) if direction=='long' else ((entry_price-price)/entry_price*100)
            with get_db_connection() as c: 
                c.execute("UPDATE signals SET status='closed', exit_price=?, exit_time=?, pnl=? WHERE id=?",(price,datetime.now(timezone.utc).isoformat(),pnl,signal_id))
            # Remove from risk manager
            risk_manager.remove_position(symbol)
            msg = f"🛠️ *Manual Close:* {symbol} closed at {price:,.4f}. PNL: {pnl:.2f}%."
            await send_notification(context, msg)
            await query.edit_message_caption(caption=query.message.caption_markdown + f"\n\n*{msg}*", parse_mode=ParseMode.MARKDOWN)
        except Exception as e: 
            await send_notification(context, f"⚠️ Error closing {symbol}: {e}")

async def testalert_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
    
    message = await update.message.reply_text("🧪 Sending test alert...")
    
    try:
        await send_notification(context, "🧪 Test alert: Bot is connected and working properly.")
        await message.edit_text("✅ Test notification sent successfully.")
    except Exception as e:
        logger.error(f"Test alert failed: {e}")
        await message.edit_text(f"❌ Test alert failed: {str(e)}")

async def forcescan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
    
    message = await update.message.reply_text("🔎 Forcing a one-time market scan...")
    
    try:
        await scan_markets(context)
        await message.edit_text("✅ One-time scan completed. Check for new signals.")
    except Exception as e:
        logger.error(f"Forced scan failed: {e}")
        await message.edit_text(f"❌ Scan failed: {str(e)}")

# ============================== BOT INITIALIZATION ==============================
def main():
    # Get port from environment variable (required for Render)
    port = int(os.environ.get("PORT", 10000))
    
    # Start Flask app in a separate thread for health checks
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False))
    flask_thread.daemon = True
    flask_thread.start()
    
    logger.info(f"Flask health check server started on port {port}")
    
    # Initialize and start the Telegram bot
    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(CommandHandler(["start", "help"], start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("pnl", pnl_command))
    application.add_handler(CommandHandler("model", model_performance_command))
    application.add_handler(CommandHandler("risk", risk_status_command))
    application.add_handler(CommandHandler("testalert", testalert_cmd))
    application.add_handler(CommandHandler("forcescan", forcescan_cmd))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    job_queue = application.job_queue
    job_queue.run_repeating(scan_markets, interval=SCAN_INTERVAL, first=10)
    job_queue.run_repeating(monitor_signals, interval=MONITOR_INTERVAL, first=5)
    job_queue.run_repeating(monitor_performance, interval=300, first=60)  # Every 5 minutes
    job_queue.run_repeating(cleanup_memory, interval=1800, first=120)  # Every 30 minutes
    job_queue.run_daily(backup_database, time=dtime(hour=2, minute=0, tzinfo=timezone.utc))  # Daily backup at 2 AM UTC
    
    # Daily report and risk reset
    report_time_aware = dtime(hour=REPORT_TIME.hour, minute=REPORT_TIME.minute, tzinfo=REPORT_TIMEZONE)
    job_queue.run_daily(daily_report, time=report_time_aware)
    
    # Daily risk reset (at midnight UTC)
    job_queue.run_daily(lambda ctx: risk_manager.reset_daily_pnl(), time=dtime(hour=0, minute=0, tzinfo=timezone.utc))
    
    async def post_init(app: Application):
        await app.bot.send_message(chat_id=OWNER_CHAT_ID_INT, text="🚀 *Bot Upgraded & Live!*\nNew features:\n- Advanced Risk Management\n- Multi-Timeframe Analysis\n- Performance Tracking\n- Model Validation")
        await send_deployment_notification(app)
        logger.info("Startup notification sent to owner.")
    application.post_init = post_init

    logger.info("Bot starting polling...")
    application.run_polling()

if __name__ == "__main__":
    main()

