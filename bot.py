import os
import gc
import csv
import hmac
import json
import time
import math
import asyncio
import logging
import pickle
import random
import string
import hashlib
import requests
import numpy as np
import pandas as pd
import psutil
import pytest
import structlog
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime, timedelta, time as dtime, timezone, date
from collections import defaultdict
from functools import wraps

# Cryptography for secure API key storage
try:
    from cryptography.fernet import Fernet
except ImportError:
    Fernet = None

# Timezones
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# ML (optional)
PYTORCH_AVAILABLE = False
torch = None
nn = None
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except Exception:
    PYTORCH_AVAILABLE = False

# Crypto API
import ccxt

# Web server for health checks & webhook
from flask import Flask, request, jsonify

# Telegram bot (async)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# Env
from dotenv import load_dotenv

# Async DB & Caching
from sqlalchemy import (
    MetaData, Table, Column, Integer, Float, String, LargeBinary, Index
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlalchemy.sql import text as sql_text
from sqlalchemy.pool import AsyncAdaptedQueuePool

# Redis optional
try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

# ------------------------------------------------------------------------------
# Configuration & Validation
# ------------------------------------------------------------------------------

load_dotenv()

class Config:
    def __init__(self):
        self.bot_token = os.environ.get("CRYPTO_BOT_TOKEN")
        self.owner_id = os.environ.get("CRYPTO_OWNER_ID")
        self.owner_id_int = int(self.owner_id) if self.owner_id and self.owner_id.isdigit() else None
        
        self.timeframe = os.environ.get("TIMEFRAME", "1h")
        self.top_n_markets = int(os.environ.get("TOP_N_MARKETS", 60))
        self.scan_interval = int(os.environ.get("SCAN_INTERVAL", 10 * 60))
        self.monitor_interval = int(os.environ.get("MONITOR_INTERVAL", 30))
        self.cache_ttl = int(os.environ.get("CACHE_TTL", 90))
        self.chart_candles = int(os.environ.get("CHART_CANDLES", 100))
        self.hybrid_model_path = os.environ.get("HYBRID_MODEL_PATH", './hybrid_model.pth')
        self.portfolio_value = float(os.environ.get("PORTFOLIO_VALUE", 10000.0))
        
        self.atr_period = int(os.environ.get("ATR_PERIOD", 14))
        self.rsi_period = int(os.environ.get("RSI_PERIOD", 14))
        self.tp_mult = [float(x) for x in os.environ.get("TP_MULT", "0.75,1.5,3.0").split(",")]
        self.sl_mult = float(os.environ.get("SL_MULT", 1.5))
        
        self.max_daily_loss = float(os.environ.get("MAX_DAILY_LOSS", 0.02))
        self.max_concurrent_trades = int(os.environ.get("MAX_CONCURRENT_TRADES", 12))
        self.max_position_size = float(os.environ.get("MAX_POSITION_SIZE", 0.02))
        
        self.report_timezone = ZoneInfo(os.environ.get("REPORT_TIMEZONE", "Asia/Kolkata"))
        self.webhook_mode = os.environ.get("WEBHOOK_MODE", "false").lower() == "true"
        self.render_external_url = os.environ.get("RENDER_EXTERNAL_URL", "")
        self.webhook_secret = os.environ.get("WEBHOOK_SECRET", ''.join(random.choices(string.ascii_letters + string.digits, k=16)))
        
        self.db_url = os.environ.get("DATABASE_URL", "").strip()
        self.db_path = os.environ.get("DB_PATH", "/tmp/power_crypto_bot.db")
        if not self.db_url:
            self.db_url = f"sqlite+aiosqlite:///{self.db_path}"
            
        self.redis_url = os.environ.get("REDIS_URL", "").strip()
        self.skip_symbols = os.environ.get("SKIP_SYMBOLS", "XPIN/USDT,DOLO/USDT").split(",")
        
        self.target_daily_signals = int(os.environ.get("TARGET_DAILY_SIGNALS", 10))
        self.max_daily_signals = self.target_daily_signals
        
        # Adaptive signal controls
        self.base_min_quality_score = float(os.environ.get("BASE_MIN_QUALITY_SCORE", 0.78))
        self.base_require_mtf_score = float(os.environ.get("BASE_REQUIRE_MTF_SCORE", 0.75))
        self.base_confidence_floor = float(os.environ.get("BASE_CONFIDENCE_FLOOR", 0.70))
        self.base_min_vol_zscore = float(os.environ.get("BASE_MIN_VOL_ZSCORE", 2.5))
        self.base_min_obs_imbalance = float(os.environ.get("BASE_MIN_OBS_IMBALANCE", 0.25))
        self.base_ema_atr_min = float(os.environ.get("BASE_EMA_ATR_MIN", 0.35))
        self.base_atr_pctl_min = float(os.environ.get("BASE_ATR_PCTL_MIN", 0.35))
        self.base_atr_pctl_max = float(os.environ.get("BASE_ATR_PCTL_MAX", 0.80))
        
        # Relaxation parameters
        self.relax_max_quality = float(os.environ.get("RELAX_MAX_QUALITY", 0.12))
        self.relax_max_conf = float(os.environ.get("RELAX_MAX_CONF", 0.08))
        self.relax_max_mtf = float(os.environ.get("RELAX_MAX_MTF", 0.10))
        self.relax_max_volz = float(os.environ.get("RELAX_MAX_VOLZ", 1.0))
        self.relax_max_imb = float(os.environ.get("RELAX_MAX_IMB", 0.10))
        self.relax_max_atr_p_band = float(os.environ.get("RELAX_MAX_ATR_P_BAND", 0.10))
        self.relax_max_ema_atr = float(os.environ.get("RELAX_MAX_EMA_ATR", 0.10))
        
        # Near-miss parameters
        self.near_miss_quality = float(os.environ.get("NEAR_MISS_QUALITY", 0.03))
        self.near_miss_conf = float(os.environ.get("NEAR_MISS_CONF", 0.03))
        self.near_miss_mtf = float(os.environ.get("NEAR_MISS_MTF", 0.05))
        self.near_miss_volz = float(os.environ.get("NEAR_MISS_VOLZ", 0.5))
        self.near_miss_imb = float(os.environ.get("NEAR_MISS_IMB", 0.05))
        self.near_miss_ema_atr = float(os.environ.get("NEAR_MISS_EMA_ATR", 0.05))
        self.day_end_catchup_hours = int(os.environ.get("DAY_END_CATCHUP_HOURS", 3))
        
        self.max_retries = int(os.environ.get("MAX_RETRIES", 3))
        self.retry_delay = float(os.environ.get("RETRY_DELAY", 1.0))
        self.retry_backoff = float(os.environ.get("RETRY_BACKOFF", 2.0))
        
        self.flask_port = int(os.environ.get("FLASK_PORT", "8081"))
        self.environment = os.environ.get("ENVIRONMENT", "development")
        
        # Validate required configuration
        self.validate()
        
    def validate(self):
        """Validate required configuration"""
        required_vars = ['CRYPTO_BOT_TOKEN', 'CRYPTO_OWNER_ID']
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
            
    @property
    def is_production(self):
        return self.environment == "production"

config = Config()

# ------------------------------------------------------------------------------
# Structured Logging
# ------------------------------------------------------------------------------

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# ------------------------------------------------------------------------------
# Secure Configuration Management
# ------------------------------------------------------------------------------

class SecureConfig:
    def __init__(self, encryption_key=None):
        self.cipher = Fernet(encryption_key) if encryption_key and Fernet else None
        
    def encrypt_api_key(self, key):
        if self.cipher and key:
            return self.cipher.encrypt(key.encode()).decode()
        return key
        
    def decrypt_api_key(self, encrypted_key):
        if self.cipher and encrypted_key:
            return self.cipher.decrypt(encrypted_key.encode()).decode()
        return encrypted_key

# Generate or use existing encryption key
encryption_key = os.environ.get('CONFIG_ENCRYPTION_KEY')
if not encryption_key and Fernet:
    encryption_key = Fernet.generate_key().decode()
    
secure_config = SecureConfig(encryption_key)

# ------------------------------------------------------------------------------
# Performance Tracking Decorator
# ------------------------------------------------------------------------------

def track_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info("function_performance", 
                       function=func.__name__,
                       execution_time=execution_time,
                       status="success")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("function_performance",
                        function=func.__name__,
                        execution_time=execution_time,
                        error=str(e),
                        status="error")
            raise
    return wrapper

# ------------------------------------------------------------------------------
# Security Manager
# ------------------------------------------------------------------------------

class SecurityManager:
    @staticmethod
    def validate_webhook_secret(req: request, expected: str) -> bool:
        token = req.headers.get('X-Telegram-Bot-Api-Secret-Token', '')
        return hmac.compare_digest(token, expected)

# ------------------------------------------------------------------------------
# Enhanced Memory Manager (PyTorch)
# ------------------------------------------------------------------------------

class EnhancedMemoryManager:
    def __init__(self, max_memory_percent: int = 85):
        self.max_memory_percent = max_memory_percent
        self.usage_history = []
        
    def cleanup(self):
        gc.collect()
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Log GPU memory usage if available
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info("gpu_memory_usage", 
                           allocated_gb=allocated, 
                           cached_gb=cached)
                
    def guard(self):
        memory_percent = psutil.virtual_memory().percent
        self.usage_history.append(memory_percent)
        # Keep only last 100 readings
        self.usage_history = self.usage_history[-100:]
        
        if memory_percent > self.max_memory_percent:
            logger.warning("high_memory_usage", percent=memory_percent)
            self.cleanup()

memory_manager = EnhancedMemoryManager()

# ------------------------------------------------------------------------------
# Retry Decorator
# ------------------------------------------------------------------------------

def with_retry(max_retries=config.max_retries, delay=config.retry_delay, backoff=config.retry_backoff):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error("max_retries_exceeded", error=str(e), max_retries=max_retries)
                        raise
                    wait_time = delay * (backoff ** (retries - 1))
                    logger.warning("retry_attempt", 
                                  attempt=retries, 
                                  max_retries=max_retries, 
                                  wait_time=wait_time,
                                  error=str(e))
                    await asyncio.sleep(wait_time)
            return None
        return wrapper
    return decorator

# ------------------------------------------------------------------------------
# Exchange Rate Limiter
# ------------------------------------------------------------------------------

class ExchangeRateLimiter:
    LIMITS = {
        'binance': {'window_sec': 60, 'weight_limit': 2400},
        'bybit': {'window_sec': 60, 'weight_limit': 600},
        'okx': {'window_sec': 60, 'weight_limit': 300},
    }
    
    def __init__(self):
        self.usage = defaultdict(int)
        self.window_start = defaultdict(lambda: time.time())
        self._lock = asyncio.Lock()
        
    async def acquire(self, exchange_name: str, weight: int = 1):
        async with self._lock:
            now = time.time()
            limits = self.LIMITS.get(exchange_name, {'window_sec': 60, 'weight_limit': 1200})
            window = limits['window_sec']
            limit = limits['weight_limit']
            
            if now - self.window_start[exchange_name] >= window:
                self.window_start[exchange_name] = now
                self.usage[exchange_name] = 0
                
            if self.usage[exchange_name] + weight > limit:
                sleep_for = window - (now - self.window_start[exchange_name])
                sleep_for = max(0.0, sleep_for)
                logger.warning("rate_limit_exceeded", 
                              exchange=exchange_name, 
                              sleep_time=sleep_for)
                await asyncio.sleep(sleep_for)
                self.window_start[exchange_name] = time.time()
                self.usage[exchange_name] = 0
                
            self.usage[exchange_name] += weight

rate_limiter = ExchangeRateLimiter()

# ------------------------------------------------------------------------------
# Exchange Factory
# ------------------------------------------------------------------------------

class ExchangeFactory:
    def __init__(self):
        self.exchanges: Dict[str, Any] = {}
        
    def get_exchange(self, name: str = 'binance'):
        if name in self.exchanges:
            return self.exchanges[name]
            
        exchange_config = {
            'enableRateLimit': True,
            'timeout': 30000,
            'rateLimit': 1000,
        }
        
        if name == 'binance':
            exchange_config['options'] = {'defaultType': 'future', 'adjustForTimeDifference': True}
            ex = ccxt.binanceusdm(exchange_config)
        elif name == 'bybit':
            exchange_config['options'] = {'defaultType': 'future'}
            ex = ccxt.bybit(exchange_config)
        elif name == 'okx':
            exchange_config['options'] = {'defaultType': 'future'}
            ex = ccxt.okx(exchange_config)
        else:
            raise ValueError(f"Unsupported exchange {name}")
            
        adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=20)
        ex.session.mount('https://', adapter)
        ex.session.mount('http://', adapter)
        
        self.exchanges[name] = ex
        return ex

exchange_factory = ExchangeFactory()
exchange = exchange_factory.get_exchange('binance')

# ------------------------------------------------------------------------------
# Async Database Setup (SQLAlchemy)
# ------------------------------------------------------------------------------

metadata = MetaData()

signals_table = Table(
    "signals", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("market_id", String(64)),
    Column("symbol", String(64), index=True),
    Column("direction", String(8)),
    Column("entry_price", Float),
    Column("entry_time", String(64)),
    Column("tp1", Float),
    Column("tp2", Float),
    Column("tp3", Float),
    Column("sl", Float),
    Column("position_size", Float),
    Column("tp1_hit", Integer, default=0),
    Column("tp2_hit", Integer, default=0),
    Column("tp3_hit", Integer, default=0),
    Column("exit_price", Float),
    Column("exit_time", String(64)),
    Column("pnl", Float),
    Column("status", String(16), index=True),
    Column("confidence", Float),
    Column("market_regime", String(32)),
    Index("idx_signals_status", "status"),
    Index("idx_signals_symbol", "symbol"),
)

ohlcv_cache_table = Table(
    "ohlcv_cache", metadata,
    Column("market_id", String(64), primary_key=True),
    Column("timeframe", String(16), primary_key=True),
    Column("fetched_at", String(64)),
    Column("blob", LargeBinary),
    Index("idx_ohlcv_cache", "market_id", "timeframe"),
)

model_performance_table = Table(
    "model_performance", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", String(64)),
    Column("accuracy", Float),
    Column("precision", Float),
    Column("recall", Float),
    Column("total_predictions", Integer),
)

# Use connection pooling for better performance
engine: AsyncEngine = create_async_engine(
    config.db_url, 
    poolclass=AsyncAdaptedQueuePool,
    pool_size=20, 
    max_overflow=30, 
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True
)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

# ------------------------------------------------------------------------------
# Redis Cache (optional)
# ------------------------------------------------------------------------------

redis_client = None
if config.redis_url and aioredis is not None:
    try:
        redis_client = aioredis.from_url(config.redis_url, decode_responses=False)
        logger.info("redis_enabled")
    except Exception as e:
        logger.warning("redis_init_failed", error=str(e))
        redis_client = None

# ------------------------------------------------------------------------------
# Performance Tracker
# ------------------------------------------------------------------------------

class PerformanceTracker:
    def __init__(self):
        self.today = date.today().isoformat()
        self.filename = f"performance_{self.today}.csv"
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'symbol', 'direction', 'entry',
                    'exit', 'pnl_percent', 'confidence', 'duration',
                    'market_regime', 'volume_ratio', 'rsi', 'atr_ratio'
                ])
                
    @track_performance
    def record_trade(self, trade_data: Dict[str, Any]):
        try:
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
        except Exception as e:
            logger.error("performance_tracker_error", error=str(e))

performance_tracker = PerformanceTracker()

# ------------------------------------------------------------------------------
# Enhanced Risk Manager
# ------------------------------------------------------------------------------

class EnhancedRiskManager:
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.daily_pnl = 0.0
        self.max_daily_loss = portfolio_value * config.max_daily_loss
        self.open_positions: Dict[str, float] = {}
        self.max_concurrent_trades = config.max_concurrent_trades
        self.position_correlations = {}
        self.circuit_breaker = False
        self.circuit_breaker_time = None
        
    def add_position(self, symbol: str, size_usd: float):
        self.open_positions[symbol] = size_usd
        
    def remove_position(self, symbol: str):
        self.open_positions.pop(symbol, None)
        
    async def check_correlation(self, new_symbol: str, existing_positions: List[str]) -> bool:
        """Check if new symbol is highly correlated with existing positions"""
        if len(existing_positions) == 0:
            return True
            
        # Simple correlation check - avoid multiple positions in similar cryptocurrencies
        base_currency = new_symbol.split('/')[0]
        for position in existing_positions:
            pos_currency = position.split('/')[0]
            if base_currency == pos_currency:
                logger.warning("correlation_check_failed", 
                              new_symbol=new_symbol, 
                              existing_position=position)
                return False
                
        return True
        
    async def can_open_trade(self, symbol: str, proposed_size_usd: float) -> Tuple[bool, str]:
        # Check circuit breaker
        if self.circuit_breaker:
            if self.circuit_breaker_time and (datetime.now() - self.circuit_breaker_time).total_seconds() > 3600:
                self.circuit_breaker = False
                self.circuit_breaker_time = None
            else:
                return False, "Circuit breaker active"
                
        if self.daily_pnl <= -self.max_daily_loss:
            return False, "Daily loss limit exceeded"
            
        if len(self.open_positions) >= self.max_concurrent_trades:
            return False, "Max concurrent trades reached"
            
        if symbol in self.open_positions:
            return False, "Already in this symbol"
            
        if proposed_size_usd > self.portfolio_value * config.max_position_size:
            return False, "Position size too large"
            
        # Add correlation check
        existing_positions = list(self.open_positions.keys())
        if not await self.check_correlation(symbol, existing_positions):
            return False, "Highly correlated with existing positions"
            
        return True, "OK"
        
    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl
        
    def reset_daily_pnl(self):
        self.daily_pnl = 0.0
        
    def activate_circuit_breaker(self):
        """Activate circuit breaker to pause trading"""
        self.circuit_breaker = True
        self.circuit_breaker_time = datetime.now()
        logger.warning("circuit_breaker_activated")

risk_manager = EnhancedRiskManager(config.portfolio_value)

# ------------------------------------------------------------------------------
# Model Validator
# ------------------------------------------------------------------------------

class ModelValidator:
    def __init__(self):
        self.predictions: List[float] = []
        self.actuals: List[int] = []
        
    def record_prediction(self, confidence: float, actual_pnl: float):
        self.predictions.append(confidence)
        self.actuals.append(1 if actual_pnl > 0 else 0)
        
    def calculate_model_performance(self):
        if len(self.predictions) < 10:
            return "Insufficient data"
            
        correct = sum(1 for p, a in zip(self.predictions, self.actuals)
                      if (p > 0.5 and a == 1) or (p <= 0.5 and a == 0))
        accuracy = correct / len(self.predictions)
        precision = self._precision()
        recall = self._recall()
        
        return {
            'accuracy': accuracy, 
            'total_predictions': len(self.predictions),
            'precision': precision, 
            'recall': recall
        }
        
    def _precision(self):
        tp = sum(1 for p, a in zip(self.predictions, self.actuals) if p > 0.5 and a == 1)
        fp = sum(1 for p, a in zip(self.predictions, self.actuals) if p > 0.5 and a == 0)
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
    def _recall(self):
        tp = sum(1 for p, a in zip(self.predictions, self.actuals) if p > 0.5 and a == 1)
        fn = sum(1 for p, a in zip(self.predictions, self.actuals) if p <= 0.5 and a == 1)
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

model_validator = ModelValidator()

# ------------------------------------------------------------------------------
# Multi-Timeframe Analyzer
# ------------------------------------------------------------------------------

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.timeframes = ['15m', '30m', '1h', '4h']
        self.weights = {'15m': 0.15, '30m': 0.25, '1h': 0.35, '4h': 0.25}
        
    @track_performance
    async def analyze_multi_timeframe(self, symbol: str, exchange_name: str):
        analysis = {}
        for tf in self.timeframes:
            df = await fetch_ohlcv_cached(symbol, tf, limit=100, exchange_name=exchange_name)
            
            if df is not None and len(df) > 20:
                analysis[tf] = self._analyze_timeframe(df)
        return self._confluence_score(analysis)
        
    def _confluence_score(self, analysis: Dict[str, Dict[str, float]]) -> float:
        if not analysis:
            return 0.5
            
        score = 0.0
        for tf, data in analysis.items():
            if tf in self.weights:
                score += data['trend_score'] * self.weights[tf]
                score += data['momentum_score'] * self.weights[tf] * 0.5
                
        return max(0.0, min(1.0, score))
        
    def _analyze_timeframe(self, df: pd.DataFrame):
        ema20 = df['close'].ewm(span=20).mean().iloc[-1]
        ema50 = df['close'].ewm(span=50).mean().iloc[-1]
        trend_score = 1.0 if ema20 > ema50 else 0.0
        
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean().iloc[-1]
        loss = -delta.clip(upper=0).rolling(window=14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 0.0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
        momentum_score = (rsi - 50) / 50
        
        volume_avg = df['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = df['volume'].iloc[-1] / volume_avg if volume_avg > 0 else 1.0
        volume_score = min(1.0, volume_ratio / 2.0)
        
        return {
            'trend_score': trend_score, 
            'momentum_score': (momentum_score + 1) / 2, 
            'volume_score': volume_score
        }

multi_timeframe_analyzer = MultiTimeframeAnalyzer()

# ------------------------------------------------------------------------------
# Portfolio Optimizer
# ------------------------------------------------------------------------------

class PortfolioOptimizer:
    @track_performance
    async def optimize_allocation(self, signals: List[Dict[str, Any]], portfolio_value: float):
        if not signals:
            return {}
            
        total_conf = sum(s.get('confidence', 0.5) for s in signals)
        if total_conf == 0:
            return {}
            
        allocations = {}
        for s in signals:
            symbol = s['symbol']
            conf = s.get('confidence', 0.5)
            allocations[symbol] = (conf / total_conf) * portfolio_value * 0.8
            
        return allocations

portfolio_optimizer = PortfolioOptimizer() 

# ---------------------------- Indicators & Features ----------------------------

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['candle_vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3.0)
    
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    hurst_exp = log_returns.rolling(window=100).std()
    df['hurst'] = hurst_exp
    
    if len(hurst_exp) > 0:
        q70 = hurst_exp.quantile(0.7)
        q30 = hurst_exp.quantile(0.3)
        df['market_regime'] = np.where(
            df['hurst'] > q70, 'trending',
            np.where(df['hurst'] < q30, 'mean_reverting', 'choppy')
        )
    else:
        df['market_regime'] = 'choppy'
        
    df['spread_ratio'] = (df['high'] - df['low']) / df['volume'].replace(0, 1e-6)
    df['absorption'] = (df['volume'] * (df['high'] - df['low'])).rolling(14).sum()
    
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=config.rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(window=config.rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(config.atr_period, min_periods=config.atr_period).mean()
    
    df = compute_advanced_features(df)
    return df

# --------------------- Quality Metrics & Volatility Filters --------------------

def _volume_zscore(df: pd.DataFrame, lookback: int = 20) -> float:
    v = df['volume']
    if len(v) < lookback + 2:
        return 0.0
        
    base = v.tail(lookback + 1).iloc[:-1]
    mu, sigma = base.mean(), base.std() + 1e-9
    return float((v.iloc[-1] - mu) / sigma)

def _ema_spread_vs_atr(df: pd.DataFrame) -> float:
    ema20 = df['ema20'].iloc[-2]
    ema50 = df['ema50'].iloc[-2]
    atr = df['atr'].iloc[-2]
    
    if pd.isna(ema20) or pd.isna(ema50) or pd.isna(atr) or atr <= 0:
        return 0.0
        
    return float(abs(ema20 - ema50) / atr)

def _atr_percentile(df: pd.DataFrame, window: int = 200) -> float:
    atr_series = df['atr'].dropna()
    if len(atr_series) < window + 2:
        return 0.5
        
    base = atr_series.tail(window + 1).iloc[:-1]
    cur = atr_series.iloc[-1]
    rank = (base < cur).sum()
    
    return float(rank / max(1, len(base)))

def _signal_quality_score(confidence: float, mtf_score: float, ema_atr_score: float,
                          vol_z: float, ob_micro_alpha: float) -> float:
    # Blend confidence (35%), MTF (25%), EMA/ATR (20%), volume z (10%), orderbook micro-alpha (10%)
    vol_term = max(0.0, min(1.0, vol_z / 3.0))
    ob_term = max(0.0, min(1.0, (ob_micro_alpha * 100)))
    
    return float(
        0.35 * confidence +
        0.25 * mtf_score +
        0.20 * max(0.0, min(1.0, ema_atr_score)) +
        0.10 * vol_term +
        0.10 * ob_term
    )

# ------------------------------ Adaptive Policy -------------------------------

def _day_progress(tz: ZoneInfo) -> float:
    now = datetime.now(tz)
    start = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=tz)
    end = start + timedelta(days=1)
    total = (end - start).total_seconds()
    elapsed = (now - start).total_seconds()
    
    return max(0.0, min(1.0, elapsed / total))

def _expected_by_now(target: int, tz: ZoneInfo) -> int:
    prog = _day_progress(tz)
    adj = math.sqrt(prog)  # slight front-loading to catch active sessions
    return int(round(target * adj))

def _regime_adjustments(regime: str) -> Dict[str, float]:
    if regime == 'trending':
        return {'vol_z_nudge': -0.2, 'imb_nudge': 0.0, 'quality_nudge': -0.0}
    if regime == 'mean_reverting':
        return {'vol_z_nudge': 0.1, 'imb_nudge': -0.05, 'quality_nudge': 0.02}
    return {'vol_z_nudge': 0.0, 'imb_nudge': 0.0, 'quality_nudge': 0.03}

def _compute_dynamic_policy(published: int, target: int, tz: ZoneInfo) -> Dict[str, Any]:
    exp_now = _expected_by_now(target, tz)
    deficit = max(0, exp_now - published)
    prog = _day_progress(tz)
    base_relax = 0.0 if exp_now == 0 else min(1.0, deficit / max(1.0, exp_now))
    time_scale = min(1.0, max(0.25, prog * 1.5))
    relax_alpha = base_relax * time_scale
    
    now = datetime.now(tz)
    hours_left = 24 - now.hour - now.minute / 60.0
    
    if hours_left <= config.day_end_catchup_hours:
        relax_alpha = min(1.0, max(relax_alpha, 0.5))
        
    thresh = {
        'min_quality': max(0.5, config.base_min_quality_score - config.relax_max_quality * relax_alpha),
        'min_conf': max(0.5, config.base_confidence_floor - config.relax_max_conf * relax_alpha),
        'min_mtf': max(0.55, config.base_require_mtf_score - config.relax_max_mtf * relax_alpha),
        'min_volz': max(0.5, config.base_min_vol_zscore - config.relax_max_volz * relax_alpha),
        'min_imb': max(0.05, config.base_min_obs_imbalance - config.relax_max_imb * relax_alpha),
        'min_ema_atr': max(0.15, config.base_ema_atr_min - config.relax_max_ema_atr * relax_alpha),
        'atr_p_min': max(0.15, config.base_atr_pctl_min - config.relax_max_atr_p_band * relax_alpha),
        'atr_p_max': min(0.95, config.base_atr_pctl_max + config.relax_max_atr_p_band * relax_alpha),
    }
    
    near = {
        'dq': config.near_miss_quality * relax_alpha,
        'dc': config.near_miss_conf * relax_alpha,
        'dm': config.near_miss_mtf * relax_alpha,
        'dvz': config.near_miss_volz * relax_alpha,
        'dimb': config.near_miss_imb * relax_alpha,
        'dema': config.near_miss_ema_atr * relax_alpha,
        'active': (relax_alpha > 0.2) or (hours_left <= config.day_end_catchup_hours)
    }
    
    return {'thresh': thresh, 'near': near, 'exp_now': exp_now, 'relax_alpha': relax_alpha}

# ---------------------- Hybrid Model & Confidence Scoring ----------------------

class HybridModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, nhead=4):
        super().__init__()
        self.recurrent_layer = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size,
            batch_first=True, 
            num_layers=2, 
            dropout=0.2
        )
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead,
            dim_feedforward=256, 
            dropout=0.2,
            activation='relu', 
            batch_first=True
        )
        
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

hybrid_model: Optional[nn.Module] = None

def load_hybrid_model(path: str) -> Optional[nn.Module]:
    if not PYTORCH_AVAILABLE:
        logger.warning("pytorch_not_available")
        return None
        
    if not os.path.exists(path):
        logger.warning("model_not_found", path=path)
        return None
        
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybridModel()
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        logger.info("model_loaded", device=str(device))
        return model
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        return None

hybrid_model = load_hybrid_model(config.hybrid_model_path)

def calculate_fallback_confidence(df: pd.DataFrame) -> float:
    try:
        recent_returns = df['close'].pct_change().tail(5)
        avg_return = recent_returns.mean() if not pd.isna(recent_returns.mean()) else 0.0
        
        vol_mean = df['volume'].rolling(20).mean()
        if len(vol_mean) > 1 and vol_mean.iloc[-2] > 0:
            vol_ratio = df['volume'].iloc[-1] / vol_mean.iloc[-2]
        else:
            vol_ratio = 1.0
            
        confidence = 0.5
        confidence += min(0.2, max(-0.2, avg_return * 10))
        confidence += min(0.2, max(-0.2, (vol_ratio - 1.0) * 0.1))
        
        return max(0.1, min(0.9, confidence))
    except Exception:
        return 0.5

def predict_signal_confidence(df: pd.DataFrame, seq_len: int = 50) -> float:
    if hybrid_model is None or not PYTORCH_AVAILABLE:
        return calculate_fallback_confidence(df)
        
    if len(df) < 30:
        return calculate_fallback_confidence(df)
        
    try:
        feature_cols = ['close', 'volume', 'rsi', 'macd', 'macd_sig', 'atr',
                        'candle_vwap', 'hurst', 'spread_ratio', 'absorption']
        df_features = df[feature_cols].dropna()
        
        if len(df_features) < 30:
            return calculate_fallback_confidence(df)
            
        actual_seq_len = min(seq_len, len(df_features))
        data_subset = df_features.tail(actual_seq_len).values
        scaled = (data_subset - np.mean(data_subset, axis=0)) / (np.std(data_subset, axis=0) + 1e-7)
        
        device = next(hybrid_model.parameters()).device
        with torch.no_grad():
            memory_manager.guard()
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
            pred = hybrid_model(tensor)
            return float(pred.item())
    except Exception as e:
        logger.error("ml_prediction_failed", error=str(e))
        return calculate_fallback_confidence(df)

# --------------------------- Cache & Serialization -----------------------------

def _pickle_df(df: pd.DataFrame) -> bytes:
    return pickle.dumps(df)

def _unpickle_df(blob: bytes) -> Optional[pd.DataFrame]:
    try:
        return pickle.loads(blob)
    except Exception:
        return None

async def cache_set(key: str, value: bytes, ttl: int):
    if redis_client:
        try:
            await redis_client.setex(key, ttl, value)
            return
        except Exception as e:
            logger.warning("redis_set_failed", error=str(e))

async def cache_get(key: str) -> Optional[bytes]:
    if redis_client:
        try:
            data = await redis_client.get(key)
            return data
        except Exception as e:
            logger.warning("redis_get_failed", error=str(e))
    return None

# ------------------------------- CCXT Wrappers --------------------------------

@with_retry()
@track_performance
async def ccxt_call(exchange_name: str, method: str, weight: int, *args, **kwargs):
    await rate_limiter.acquire(exchange_name, weight)
    ex = exchange_factory.get_exchange(exchange_name)
    func = getattr(ex, method)
    return await asyncio.to_thread(func, *args, **kwargs)

# -------------------------- OHLCV & Order Book Fetchers -----------------------

@with_retry()
@track_performance
async def fetch_ohlcv_cached(symbol: str, timeframe: str, limit: int = 100,
                             exchange_name: str = 'binance') -> Optional[pd.DataFrame]:
    cache_key = f"ohlcv:{exchange_name}:{symbol}:{timeframe}:{limit}"
    blob = await cache_get(cache_key)
    
    if blob:
        df = _unpickle_df(blob)
        if df is not None and len(df) >= 20:
            return df
            
    try:
        async with engine.begin() as conn:
            res = await conn.execute(
                sql_text("SELECT fetched_at, blob FROM ohlcv_cache WHERE market_id = :m AND timeframe = :t"),
                {"m": symbol, "t": timeframe}
            )
            row = res.fetchone()
            
            if row:
                fetched_at = datetime.fromisoformat(row[0])
                age = (datetime.now(timezone.utc) - fetched_at).total_seconds()
                
                if age < config.cache_ttl:
                    df = _unpickle_df(row[1])
                    if df is not None and len(df) >= 20:
                        await cache_set(cache_key, row[1], config.cache_ttl)
                        return df
    except Exception as e:
        logger.warning("db_cache_read_failed", error=str(e))
        
    try:
        bars = await ccxt_call(exchange_name, 'fetch_ohlcv', 1, symbol, timeframe, limit=limit)
        if not bars or len(bars) < 20:
            return None
            
        df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        if df['close'].isna().any() or df['volume'].isna().any():
            return None
            
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        
        blob = _pickle_df(df)
        await cache_set(cache_key, blob, config.cache_ttl)
        
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    sql_text("""
                        INSERT INTO ohlcv_cache(market_id, timeframe, fetched_at, blob) 
                        VALUES (:m, :t, :f, :b) 
                        ON CONFLICT(market_id, timeframe) DO UPDATE SET fetched_at = :f, blob = :b
                    """),
                    {"m": symbol, "t": timeframe, "f": datetime.now(timezone.utc).isoformat(), "b": blob}
                )
        except Exception as e:
            logger.warning("db_cache_write_failed", error=str(e))
            
        return df
    except Exception as e:
        logger.error("fetch_ohlcv_error", symbol=symbol, timeframe=timeframe, error=str(e))
        return None

@with_retry()
@track_performance
async def fetch_orderbook_features(symbol: str, exchange_name: str = 'binance') -> Dict[str, float]:
    try:
        ob = await ccxt_call(exchange_name, 'fetch_order_book', 1, symbol, 10)
        bids = ob.get('bids', [])
        asks = ob.get('asks', [])
        
        if not bids or not asks:
            return {}
            
        mid_price = (bids[0][0] + asks[0][0]) / 2.0
        spread = asks[0][0] - bids[0][0]
        bid_volume = sum(b[1] for b in bids[:5])
        ask_volume = sum(a[1] for a in asks[:5])
        tot = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / tot if tot > 0 else 0.0
        
        return {
            'mid_price': mid_price,
            'spread': spread,
            'imbalance': imbalance,
            'micro_alpha': (imbalance * spread) / mid_price if mid_price > 0 else 0.0
        }
    except Exception as e:
        logger.warning("orderbook_fetch_failed", symbol=symbol, error=str(e))
        return {}

# ------------------------- Dynamic Sizing & Parameters -------------------------

def dynamic_position_sizing(portfolio_value: float, volatility_usd: float, confidence: float) -> float:
    if volatility_usd <= 0:
        return 0.0
        
    edge = confidence - 0.5
    if edge <= 0:
        return 0.0
        
    max_risk_fraction = config.max_daily_loss
    suggested_risk_fraction = edge * 0.1
    risk_fraction = min(max_risk_fraction, suggested_risk_fraction)
    usd_to_risk = portfolio_value * risk_fraction
    size = usd_to_risk / volatility_usd
    
    return max(0.0, size)

def get_optimal_parameters(market_regime: str) -> Dict[str, float]:
    regimes = {
        'trending': {'RSI_BUY': 52, 'RSI_SELL': 48},
        'mean_reverting': {'RSI_BUY': 45, 'RSI_SELL': 55},
        'choppy': {'RSI_BUY': 50, 'RSI_SELL': 50},
    }
    return regimes.get(market_regime, regimes['choppy'])

# ----------------------------- Signal Validation ------------------------------

@track_performance
async def validate_signal(signal: Dict[str, Any]) -> bool:
    """Perform additional validation on generated signals"""
    if not signal:
        return False
        
    # Check if recent price action confirms the signal
    try:
        df = await fetch_ohlcv_cached(signal['market_id'], '5m', limit=10, exchange_name='binance')
        if df is None or len(df) < 5:
            return True  # Can't validate, but don't reject
            
        recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        if signal['side'] == 'Long' and recent_trend < -0.01:  # Down 1% in last 5 candles
            logger.warning("rejecting_long_signal", recent_trend=recent_trend)
            return False
            
        if signal['side'] == 'Short' and recent_trend > 0.01:  # Up 1% in last 5 candles
            logger.warning("rejecting_short_signal", recent_trend=recent_trend)
            return False
            
    except Exception as e:
        logger.error("signal_validation_error", error=str(e))
        
    return True

# ----------------------------- Signal Generation ------------------------------

@track_performance
async def generate_signal(market_id: str, display_symbol: str, cooldowns: Dict[str, datetime],
                          exchange_name: str, dyn_policy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        if any(skip in display_symbol for skip in config.skip_symbols):
            return None

        df = await fetch_ohlcv_cached(market_id, config.timeframe, limit=400, exchange_name=exchange_name)
        ob_features = await fetch_orderbook_features(display_symbol, exchange_name=exchange_name)
        
        if df is None or len(df) < 120:
            return None

        df = compute_indicators(df)
        last = df.iloc[-2]
        last_time = df.index[-2]
        
        if market_id in cooldowns and cooldowns[market_id] >= last_time:
            return None

        # MTF confluence and confidence
        mtf_score = await multi_timeframe_analyzer.analyze_multi_timeframe(market_id, exchange_name)
        confidence = predict_signal_confidence(df)

        # Derived metrics
        ema_atr = _ema_spread_vs_atr(df)
        atr_pctl = _atr_percentile(df, window=200)
        vol_z = _volume_zscore(df, lookback=20)
        ob_imb = ob_features.get('imbalance', 0.0)
        ob_micro_alpha = ob_features.get('micro_alpha', 0.0)

        regime = str(last.get("market_regime", "choppy"))
        r_adj = _regime_adjustments(regime)

        # Dynamic thresholds
        T = dyn_policy['thresh']
        min_quality = max(0.0, T['min_quality'] + r_adj['quality_nudge'])
        min_conf = T['min_conf']
        min_mtf = T['min_mtf']
        min_volz = max(0.0, T['min_volz'] + r_adj['vol_z_nudge'])
        min_imb = max(0.0, T['min_imb'] + r_adj['imb_nudge'])
        min_ema_atr = T['min_ema_atr']
        atr_p_min, atr_p_max = T['atr_p_min'], T['atr_p_max']

        # Trend/RSI gate
        is_uptrend = last["ema20"] > last["ema50"]
        is_downtrend = last["ema20"] < last["ema50"]
        params = get_optimal_parameters(regime)

        side = None
        if is_uptrend and regime == 'trending' and last["rsi"] > params['RSI_BUY'] and ob_imb > 0:
            side = "Long"
        elif is_downtrend and regime == 'trending' and last["rsi"] < params['RSI_SELL'] and ob_imb < 0:
            side = "Short"
            
        if not side:
            return None

        # Hard gates
        if not (atr_p_min <= atr_pctl <= atr_p_max):
            return None
            
        if ema_atr < min_ema_atr:
            return None

        # Quality score and acceptance
        q = _signal_quality_score(confidence, mtf_score, ema_atr, vol_z, ob_micro_alpha)
        strict_ok = (
            (confidence >= min_conf) and
            (mtf_score >= min_mtf) and
            (vol_z >= min_volz) and
            (abs(ob_imb) >= min_imb) and
            (q >= min_quality)
        )

        # Near-miss for deficit catch-up
        near = dyn_policy['near']
        near_ok = False
        
        if near['active'] and not strict_ok:
            near_ok = (
                (confidence >= max(0.5, min_conf - near['dc'])) and
                (mtf_score >= max(0.5, min_mtf - near['dm'])) and
                (vol_z >= max(0.0, min_volz - near['dvz'])) and
                (abs(ob_imb) >= max(0.0, min_imb - near['dimb'])) and
                (q >= max(0.5, min_quality - near['dq'])) and
                (ema_atr >= max(0.1, min_ema_atr - near['dema']))
            )

        if not (strict_ok or near_ok):
            return None

        entry_price = float(last["close"])
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else entry_price * 0.02
        sl_dist = atr * config.sl_mult
        position_size_coin = dynamic_position_sizing(config.portfolio_value, sl_dist, confidence)
        position_value = position_size_coin * entry_price

        can_trade, _ = await risk_manager.can_open_trade(display_symbol, position_value)
        if not can_trade:
            return None

        if side == "Long":
            sl = entry_price - sl_dist
            tps = [entry_price + atr * m for m in config.tp_mult]
        else:
            sl = entry_price + sl_dist
            tps = [entry_price - atr * m for m in config.tp_mult]

        alert = format_alert(display_symbol, side, entry_price, sl, tps, confidence, position_size_coin, regime)
        
        signal_data = {
            "symbol": display_symbol,
            "market_id": market_id,
            "side": side,
            "quality": q,
            "confidence": confidence,
            "mtf_score": mtf_score,
            "vol_z": vol_z,
            "ob_imb": ob_imb,
            "ob_micro_alpha": ob_micro_alpha,
            "entry": entry_price,
            "sl": sl,
            "tps": tps,
            "regime": regime,
            "position_size_coin": position_size_coin,
            "position_value": position_value,
            "last_time": last_time,
            "text": alert,
            "strict": strict_ok,
            "near_miss": (not strict_ok) and near_ok
        }
        
        # Additional validation
        if not await validate_signal(signal_data):
            return None
            
        return signal_data
    except Exception as e:
        logger.exception("signal_generation_error", market_id=market_id, error=str(e))
        return None

# --------------------------------- Charting -----------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

@track_performance
async def plot_annotated_chart(df: pd.DataFrame, display_symbol: str, entry: float, sl: float, tps: list) -> str:
    def _plot():
        try:
            df_plot = df.tail(config.chart_candles).copy()
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
            plt.title(f"{display_symbol} Signal ({config.timeframe})", fontsize=16, fontweight='bold')
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
            plt.close(fig)
            return fname
        except Exception as e:
            logger.error("charting_failed", error=str(e))
            return ""
            
    return await asyncio.to_thread(_plot)

def format_alert(symbol: str, side: str, entry: float, sl: float, tps: list, confidence: float, position_size: float, regime: str) -> str:
    return (
        f"📈 {symbol} Signal ({config.timeframe})\n\n"
        f"{'🚀' if side == 'Long' else '📉'} Trade Type: {side.upper()}\n"
        f"🧠  Confidence: {confidence*100:.1f}%\n"
        f"📊 Market Regime: {regime.title()}\n\n"
        f"Trade Parameters:\n"
        f"  - Entry: {entry:,.4f}\n"
        f"  - Stop-loss: {sl:,.4f}\n\n"
        f"Take-Profit Targets:\n"
        f"  - TP1: {tps[0]:,.4f}\n"
        f"  - TP2: {tps[1]:,.4f}\n"
        f"  - TP3: {tps[2]:,.4f}\n\n"
        f"Sizing & Risk (Based on ${config.portfolio_value:,} portfolio):\n"
        f"  - Suggested Size: {position_size:.4f} {symbol.split('/')[0]}\n"
        f"  - Position Value: ${(position_size * entry):,.2f}\n"
        f"⚡ Move SL to entry after TP1 is hit."
    )

# ---------------------------- Daily Quota Helpers -----------------------------

async def _get_daily_count() -> int:
    key = f"signals:day:{date.today().isoformat()}"
    if redis_client:
        try:
            v = await redis_client.get(key)
            return int(v) if v else 0
        except Exception:
            return 0
    return 0

async def _incr_daily_count(n: int):
    key = f"signals:day:{date.today().isoformat()}"
    if redis_client:
        try:
            ttl = 24 * 3600
            pipe = redis_client.pipeline()
            pipe.incrby(key, n)
            pipe.expire(key, ttl)
            await pipe.execute()
        except Exception:
            pass

# ---------------------- Batch Processing for Market Scanning --------------------

async def batch_process_markets(markets: List[Tuple[str, Any]], batch_size: int = 10, 
                               cooldowns: Dict[str, datetime], exchange_name: str, 
                               dyn_policy: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Process markets in batches to manage rate limiting"""
    strict_signals = []
    near_miss_signals = []
    
    for i in range(0, len(markets), batch_size):
        batch = markets[i:i+batch_size]
        batch_tasks = []
        
        for symbol, _ in batch:
            batch_tasks.append(
                generate_signal(symbol, symbol, cooldowns, exchange_name, dyn_policy)
            )
        
        # Process batch with rate limiting
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                logger.warning("batch_processing_error", error=str(result))
                continue
                
            if result:
                if result['strict']:
                    strict_signals.append(result)
                elif result['near_miss']:
                    near_miss_signals.append(result)
        
        # Rate limiting between batches
        if i + batch_size < len(markets):
            await asyncio.sleep(0.5)
    
    return strict_signals, near_miss_signals

# ---------------------- Adaptive Selection Scanner (Daily) --------------------

@track_performance
async def scan_markets(context: Optional[ContextTypes.DEFAULT_TYPE] = None, exchange_name: str = 'binance'):
    logger.info("market_scan_started")
    cooldowns: Dict[str, datetime] = {}
    
    try:
        current_count = await _get_daily_count()
        remaining_total = max(0, config.target_daily_signals - current_count)
        
        if remaining_total <= 0:
            logger.info("daily_quota_reached")
            return

        dyn_policy = _compute_dynamic_policy(current_count, config.target_daily_signals, config.report_timezone)

        tickers = await ccxt_call(exchange_name, 'fetch_tickers', 1)
        if not tickers:
            logger.warning("no_tickers_received")
            return
            
        futures_tickers = {k: v for k, v in tickers.items() if k.endswith('/USDT')}
        sorted_tickers = sorted(
            futures_tickers.items(),
            key=lambda x: x[1].get('quoteVolume', 0) or 0,
            reverse=True
        )

        # Process markets in batches
        strict, near_miss = await batch_process_markets(
            sorted_tickers[:config.top_n_markets], 
            10,  # batch size
            cooldowns, 
            exchange_name, 
            dyn_policy
        )

        if not strict and not near_miss:
            logger.info("no_qualified_candidates")
            return

        # Sort by quality
        strict.sort(key=lambda c: c['quality'], reverse=True)
        near_miss.sort(key=lambda c: c['quality'], reverse=True)

        selected: List[Dict[str, Any]] = []

        # Prefer diversity: at least one long and one short from strict if available
        longs = [c for c in strict if c['side'] == 'Long']
        shorts = [c for c in strict if c['side'] == 'Short']
        
        if longs:
            selected.append(longs[0])
        if shorts and len(selected) < remaining_total:
            selected.append(shorts[0])

        # Fill from remaining strict
        used = set((s['symbol'], s['side']) for s in selected)
        for c in strict:
            if len(selected) >= remaining_total:
                break
            key = (c['symbol'], c['side'])
            if key not in used:
                selected.append(c)
                used.add(key)

        # If still deficit, fill from near-miss
        if len(selected) < remaining_total:
            for c in near_miss:
                if len(selected) >= remaining_total:
                    break
                key = (c['symbol'], c['side'])
                if key not in used:
                    selected.append(c)
                    used.add(key)

        if not selected:
            logger.info("no_candidates_final_selection")
            return

        committed = 0
        async with engine.begin() as conn:
            for s in selected:
                # Final risk check and commit
                can_trade, _ = await risk_manager.can_open_trade(s['symbol'], s['position_value'])
                if not can_trade:
                    continue
                    
                risk_manager.add_position(s['symbol'], s['position_value'])
                res = await conn.execute(
                    signals_table.insert().values(
                        market_id=s['market_id'], symbol=s['symbol'], direction=s['side'].lower(),
                        entry_price=s['entry'], entry_time=pd.Timestamp(s['last_time']).isoformat(),
                        tp1=s['tps'][0], tp2=s['tps'][1], tp3=s['tps'][2], sl=s['sl'],
                        position_size=s['position_size_coin'], confidence=s['confidence'],
                        market_regime=s['regime'], status='open'
                    )
                )
                
                signal_id = res.inserted_primary_key[0] if res.inserted_primary_key else None
                
                # Chart + alert if Telegram context is available       
                chart_df = await fetch_ohlcv_cached(s['market_id'], config.timeframe, 400, exchange_name)
                chart_path = await plot_annotated_chart(chart_df, s['symbol'], s['entry'], s['sl'], s['tps'])
                
                if context:
                    alert_payload = {"text": s['text'], "chart": chart_path, "signal_id": signal_id}
                    try:
                        await send_alert(context, alert_payload)
                    except Exception as e:
                        logger.error("alert_send_failed", error=str(e))
                        
                committed += 1

        if committed > 0:
            await _incr_daily_count(committed)
            logger.info("signals_published", count=committed)
        else:
            logger.info("no_signals_published")
            
    except Exception as e:
        logger.error("scan_error", error=str(e))

# ------------------------------- Position Monitor -----------------------------

@track_performance
async def monitor_positions(context: Optional[ContextTypes.DEFAULT_TYPE] = None, exchange_name: str = 'binance'):
    try:
        async with engine.begin() as conn:
            res = await conn.execute(sql_text("SELECT * FROM signals WHERE status='open'"))
            rows = res.fetchall()
            
        if not rows:
            return
            
        for r in rows:
            try:
                symbol = r[2]
                direction = r[3]
                entry_price = float(r[4])
                tp1, tp2, tp3 = float(r[6]), float(r[7]), float(r[8])
                sl = float(r[9])
                size = float(r[10] or 0.0)
                tp1_hit, tp2_hit, tp3_hit = int(r[11] or 0), int(r[12] or 0), int(r[13] or 0)
                
                cur_ticker = await ccxt_call(exchange_name, 'fetch_ticker', 1, symbol)
                price = float(cur_ticker.get('last') or cur_ticker.get('close') or 0.0)
                
                if price <= 0:
                    continue
                    
                position_closed = False
                exit_reason = ""
                
                if direction == 'long':
                    if price <= sl:
                        position_closed = True
                        exit_reason = "Stop Loss"
                    elif price >= tp3 and not tp3_hit:
                        position_closed = True
                        exit_reason = "TP3"
                    elif price >= tp2 and not tp2_hit:
                        async with engine.begin() as conn:
                            await conn.execute(
                                sql_text("UPDATE signals SET tp2_hit=1 WHERE id=:id"), 
                                {"id": r[0]}
                            )
                    elif price >= tp1 and not tp1_hit:
                        async with engine.begin() as conn:
                            await conn.execute(
                                sql_text("UPDATE signals SET tp1_hit=1 WHERE id=:id"), 
                                {"id": r[0]}
                            )
                else:
                    if price >= sl:
                        position_closed = True
                        exit_reason = "Stop Loss"
                    elif price <= tp3 and not tp3_hit:
                        position_closed = True
                        exit_reason = "TP3"
                    elif price <= tp2 and not tp2_hit:
                        async with engine.begin() as conn:
                            await conn.execute(
                                sql_text("UPDATE signals SET tp2_hit=1 WHERE id=:id"), 
                                {"id": r[0]}
                            )
                    elif price <= tp1 and not tp1_hit:
                        async with engine.begin() as conn:
                            await conn.execute(
                                sql_text("UPDATE signals SET tp1_hit=1 WHERE id=:id"), 
                                {"id": r[0]}
                            )
                            
                if position_closed:
                    pnl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                    
                    async with engine.begin() as conn:
                        await conn.execute(
                            sql_text("""
                                UPDATE signals SET status='closed', exit_price=:p, exit_time=:t, pnl=:pl WHERE id=:id
                            """),
                            {"p": price, "t": datetime.now(timezone.utc).isoformat(), "pl": pnl, "id": r[0]}
                        )
                        
                    risk_manager.remove_position(symbol)
                    risk_manager.update_daily_pnl(pnl)
                    
                    try:
                        pnl_pct = (pnl / (entry_price * size)) * 100 if entry_price > 0 and size > 0 else 0.0
                        performance_tracker.record_trade({
                            'timestamp': datetime.now().isoformat(),
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'pnl_percent': pnl_pct,
                            'confidence': float(r[18] or 0.5),
                            'duration_minutes': None,
                            'market_regime': r[19],
                            'volume_ratio': None, 'rsi': None, 'atr_ratio': None
                        })
                    except Exception as e:
                        logger.warning("performance_recording_failed", error=str(e))
                        
                    if context:
                        note = (
                            f"🎯 Position Closed\n\n"
                            f"📊 {symbol} {direction.upper()}\n"
                            f"💰 PnL: ${pnl:.2f}\n"
                            f"🏁 Reason: {exit_reason}\n"
                            f"📈 Entry: ${entry_price:.4f}\n"
                            f"📉 Exit: ${price:.4f}"
                        )
                        try:
                            await context.bot.send_message(
                                chat_id=config.owner_id_int, 
                                text=note, 
                                parse_mode=ParseMode.MARKDOWN
                            )
                        except Exception as e:
                            logger.error("close_notify_failed", error=str(e))
            except Exception as e:
                logger.error("monitor_error", signal_id=r[0], error=str(e))
                continue
                
    except Exception as e:
        logger.error("monitor_loop_error", error=str(e))

# ------------------------------ Telegram Helpers ------------------------------

application: Optional[Application] = None

def is_authorized(update: Update) -> bool:
    user = update.effective_user
    return bool(user and config.owner_id and str(user.id) == str(config.owner_id))

async def send_notification(context: ContextTypes.DEFAULT_TYPE, message: str):
    try:
        await context.bot.send_message(
            chat_id=config.owner_id_int, 
            text=message, 
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error("notification_failed", error=str(e))

async def send_alert(context: ContextTypes.DEFAULT_TYPE, alert_data: dict):
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Move SL to BE", callback_data=f"be:{alert_data['signal_id']}"),
        InlineKeyboardButton("Close Manually", callback_data=f"close:{alert_data['signal_id']}")
    ]])
    
    chart_path = alert_data.get("chart", "")
    try:
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, "rb") as chart_photo:
                await context.bot.send_photo(
                    chat_id=config.owner_id_int,
                    photo=chart_photo,
                    caption=alert_data["text"],
                    reply_markup=kb,
                    parse_mode=ParseMode.MARKDOWN
                )
            try:
                os.remove(chart_path)
            except Exception as e:
                logger.warning("chart_cleanup_failed", error=str(e))
        else:
            await context.bot.send_message(
                chat_id=config.owner_id_int,
                text=alert_data["text"],
                reply_markup=kb,
                parse_mode=ParseMode.MARKDOWN
            )
    except Exception as e:
        logger.error("alert_send_failed", error=str(e))

# --------------------------------- Handlers -----------------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
        
    msg = (
        "🤖 Crypto Trading Bot Started\n\n"
        f"📊 Monitoring {config.top_n_markets} top markets\n"
        f"⏱️ Timeframe: {config.timeframe}\n"
        f"🔄 Scan Interval: {config.scan_interval//60} minutes\n"
        f"🎯 Target Daily Signals: {config.target_daily_signals}\n"
        f"💰 Portfolio: ${config.portfolio_value:,}\n\n"
        "Commands:\n"
        "/status - Bot status\n"
        "/performance - Trading performance\n"
        "/stats - Trading statistics\n"
        "/positions - Open positions\n"
        "/stop - Stop bot\n"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
        
    try:
        async with engine.begin() as conn:
            res1 = await conn.execute(sql_text("SELECT COUNT(*) FROM signals WHERE status='open'"))
            open_signals = res1.scalar() or 0
            res2 = await conn.execute(sql_text("SELECT COUNT(*) FROM signals"))
            total_signals = res2.scalar() or 0
            
        model_status = "✅ Loaded" if hybrid_model is not None else "❌ Fallback Mode"
        memory_percent = psutil.virtual_memory().percent
        
        status_message = (
            f"🤖 Bot Status\n\n"
            f"🟢 Status: Active\n"
            f"🧠 Model: {model_status}\n"
            f"📊 Open Positions: {open_signals}\n"
            f"📈 Total Signals: {total_signals}\n"
            f"💰 Daily PnL: ${risk_manager.daily_pnl:.2f}\n"
            f"⚠️ Risk Level: {len(risk_manager.open_positions)}/{config.max_concurrent_trades}\n"
            f"🧠 Memory Usage: {memory_percent:.1f}%\n"
        )
        
        await update.message.reply_text(status_message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error("status_error", error=str(e))
        await update.message.reply_text("❌ Error retrieving status")

async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
        
    try:
        async with engine.begin() as conn:
            res = await conn.execute(sql_text("""
                SELECT 
                    COUNT(*) as total_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM signals
                WHERE status = 'closed'
            """))
            row = res.fetchone()
            
        if row and row[0] > 0:
            total, avg_pnl, wins, best, worst = row
            win_rate = (wins / total) * 100 if total > 0 else 0
            
            msg = (
                f"📊 Performance Summary\n\n"
                f"🎯 Total Trades: {total}\n"
                f"✅ Win Rate: {win_rate:.1f}%\n"
                f"💰 Avg PnL: ${avg_pnl:.2f}\n"
                f"🚀 Best Trade: ${best:.2f}\n"
                f"📉 Worst Trade: ${worst:.2f}\n"
            )
            
            perf = model_validator.calculate_model_performance()
            if isinstance(perf, dict):
                msg += (
                    f"\n🧠 Model Metrics\n"
                    f"🎯 Accuracy: {perf['accuracy']:.1%}\n"
                    f"📈 Precision: {perf['precision']:.1%}\n"
                    f"🔄 Recall: {perf['recall']:.1%}\n"
                )
        else:
            msg = "📊 No closed trades yet"
            
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error("performance_error", error=str(e))
        await update.message.reply_text("❌ Error retrieving performance data")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
        
    try:
        async with engine.begin() as conn:
            res = await conn.execute(sql_text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winners
                FROM signals 
                WHERE status = 'closed'
            """))
            row = res.fetchone()
            win_rate = (row[1] / row[0] * 100) if row[0] > 0 else 0
            
            res2 = await conn.execute(sql_text("""
                SELECT 
                    AVG((julianday(exit_time) - julianday(entry_time)) * 24 * 60) as avg_minutes
                FROM signals 
                WHERE status = 'closed' AND exit_time IS NOT NULL
            """))
            avg_hold = res2.scalar() or 0
            
        msg = (
            f"📊 Trading Statistics\n\n"
            f"🏆 Win Rate: {win_rate:.1f}%\n"
            f"⏱️ Avg Holding Time: {avg_hold:.1f} minutes\n"
            f"💰 Daily PnL: ${risk_manager.daily_pnl:.2f}\n"
            f"📈 Open Positions: {len(risk_manager.open_positions)}\n"
        )
        
        memory_percent = psutil.virtual_memory().percent
        msg += f"🧠 Memory Usage: {memory_percent:.1f}%\n"
        
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error("stats_error", error=str(e))
        await update.message.reply_text("❌ Error retrieving statistics")

async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        return
        
    try:
        async with engine.begin() as conn:
            res = await conn.execute(
                sql_text("SELECT symbol, entry_price, sl, tp1, tp2, tp3, status FROM signals WHERE status='open'")
            )
            rows = res.fetchall()
            
        if not rows:
            await update.message.reply_text("No open positions")
            return
            
        lines = ["📌 Open Positions:"]
        for r in rows:
            sym, entry, sl, tp1, tp2, tp3, st = r
            lines.append(f"- {sym} | Entry {entry:.4f} | SL {sl:.4f} | TP1 {tp1:.4f} | TP2 {tp2:.4f} | TP3 {tp3:.4f}")
            
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        logger.error("positions_error", error=str(e))
        await update.message.reply_text("❌ Error retrieving positions")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query:
        return
        
    if not is_authorized(update):
        await query.answer("Unauthorized")
        return
        
    await query.answer()
    
    try:
        action, signal_id = query.data.split(":")
        signal_id = int(signal_id)
        
        async with engine.begin() as conn:
            if action == "be":
                res = await conn.execute(
                    sql_text("UPDATE signals SET sl = entry_price WHERE id = :id AND status='open'"), 
                    {"id": signal_id}
                )
                
                if res.rowcount and res.rowcount > 0:
                    await query.edit_message_caption(
                        caption=(query.message.caption or "") + "\n\n✅ Stop Loss moved to Breakeven",
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await query.edit_message_caption(
                        caption=(query.message.caption or "") + "\n\n❌ Signal not found or already closed",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    
            elif action == "close":
                res = await conn.execute(
                    sql_text("UPDATE signals SET status='closed', exit_time = :t WHERE id = :id AND status='open'"),
                    {"id": signal_id, "t": datetime.now(timezone.utc).isoformat()}
                )
                
                if res.rowcount and res.rowcount > 0:
                    row2 = await conn.execute(
                        sql_text("SELECT symbol FROM signals WHERE id=:id"), 
                        {"id": signal_id}
                    )
                    symrow = row2.fetchone()
                    
                    if symrow:
                        risk_manager.remove_position(symrow[0])
                        
                    await query.edit_message_caption(
                        caption=(query.message.caption or "") + "\n\n✅ Position closed manually",
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await query.edit_message_caption(
                        caption=(query.message.caption or "") + "\n\n❌ Signal not found or already closed",
                        parse_mode=ParseMode.MARKDOWN
                    )
                    
    except Exception as e:
        logger.error("callback_error", error=str(e))
        await query.edit_message_caption(
            caption=(query.message.caption or "") + "\n\n❌ Error processing request",
            parse_mode=ParseMode.MARKDOWN
        )

# ------------------------------ Health Endpoints ------------------------------

flask_app = Flask(__name__)

@flask_app.route('/health')
def health_check():
    try:
        ok_ex = True
        try:
            _ = exchange.fetch_time()
            ok_ex = True
        except Exception:
            ok_ex = False
            
        memory_percent = psutil.virtual_memory().percent
        status = 'healthy' if ok_ex and memory_percent < 90 else 'degraded'
        
        return jsonify({
            'status': status, 
            'timestamp': datetime.now().isoformat(),
            'memory_usage': memory_percent,
            'exchange_connected': ok_ex
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@flask_app.route('/')
def home():
    return jsonify({
        'message': 'Crypto Trading Bot is running', 
        'status': 'active',
        'environment': config.environment
    })

@flask_app.route('/webhook', methods=['POST'])
def telegram_webhook():
    if not SecurityManager.validate_webhook_secret(request, config.webhook_secret):
        return 'Forbidden', 403
        
    try:
        update = Update.de_json(request.get_json(), application.bot)
        application.process_update(update)
        return 'OK'
    except Exception as e:
        logger.error("webhook_error", error=str(e))
        return 'Error', 500

# ------------------------------- Telegram App ---------------------------------

def build_telegram_app() -> Application:
    app = Application.builder().token(config.bot_token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("performance", performance_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("positions", positions_command))
    app.add_handler(CallbackQueryHandler(callback_handler))
    return app

# --------------------------------- Scheduler ----------------------------------

async def scheduled_runner(app: Application):
    try:
        scan_timer = 0
        while True:
            if scan_timer <= 0:
                await scan_markets(app)
                scan_timer = config.scan_interval
                
            await monitor_positions(app)
            await asyncio.sleep(config.monitor_interval)
            scan_timer -= config.monitor_interval
            
            # Reset daily PnL and Redis counter at local midnight
            now = datetime.now(config.report_timezone)
            if now.hour == 0 and now.minute == 0:
                risk_manager.reset_daily_pnl()
                
    except asyncio.CancelledError:
        logger.info("scheduler_cancelled")

# ----------------------------------- Main -------------------------------------

def start_flask(port: int):
    flask_app.run(host='0.0.0.0', port=port)

async def main():
    global application
    
    # Validate environment
    config.validate()
    
    await init_db()
    application = build_telegram_app()

    # Health server on separate port to avoid webhook conflict
    if config.webhook_mode and config.render_external_url:
        # Start Flask health in background
        asyncio.get_running_loop().run_in_executor(None, start_flask, config.flask_port)
        webhook_url = f"{config.render_external_url}/webhook"
        
        application.run_webhook(
            listen="0.0.0.0",
            port=8080,
            url_path="/webhook",
            webhook_url=webhook_url,
            secret_token=config.webhook_secret
        )
    else:
        # Polling mode: start Flask in background, then run app + scheduler
        asyncio.get_running_loop().run_in_executor(None, start_flask, config.flask_port)
        await application.initialize()
        await application.start()
        
        scheduler_task = asyncio.create_task(scheduled_runner(application))
        
        try:
            # Start polling updates
            await application.updater.start_polling()
            await scheduler_task
        except KeyboardInterrupt:
            logger.info("bot_stopped_by_user")
        finally:
            scheduler_task.cancel()
            await application.stop()
    
if __name__ == "__main__" and not config.webhook_mode:
    flask_app.run(host='0.0.0.0', port=config.flask_port)


