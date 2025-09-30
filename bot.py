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
import structlog
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime, timedelta, time as dtime, timezone, date
from collections import defaultdict
from functools import wraps
from scipy.stats import pearsonr

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
import ccxt.async_support as ccxt_async

# Web server for health checks & webhook
from flask import Flask, request, jsonify

# Telegram bot (async)
# ... (other imports)
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode
# ... (other imports)
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

# THREADING FIX: Import threading libraries and force matplotlib backend
import concurrent.futures
import matplotlib
# Force matplotlib to use Agg backend BEFORE importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
MATPLOTLIB_AVAILABLE = True

# ------------------------------------------------------------------------------

# Network connection error handling
import socket
from urllib.error import URLError
from requests.exceptions import ConnectionError, Timeout, RequestException

def is_network_available() -> bool:
    """Check if network is available"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

async def wait_for_network():
    """Wait for network to become available"""
    while not is_network_available():
        logger.warning("network_unavailable_waiting")
        await asyncio.sleep(10)
    logger.info("network_available")

# Configuration & Validation
# ------------------------------------------------------------------------------

load_dotenv()

class Config:
    def __init__(self):
        self.bot_token = os.environ.get("CRYPTO_BOT_TOKEN")
        self.owner_id = os.environ.get("CRYPTO_OWNER_ID")
        self.owner_id_int = int(self.owner_id) if self.owner_id and self.owner_id.isdigit() else None
        
        # --- FIX: Added group_chat_id attribute ---
        self.group_chat_id = os.environ.get("CRYPTO_GROUP_ID")

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
        
        self.max_slippage_percent = float(os.environ.get("MAX_SLIPPAGE_PERCENT", 0.05))
        self.max_portfolio_correlation = float(os.environ.get("MAX_PORTFOLIO_CORRELATION", 0.70))
        
        self.base_min_quality_score = float(os.environ.get("BASE_MIN_QUALITY_SCORE", 0.45))
        self.base_require_mtf_score = float(os.environ.get("BASE_REQUIRE_MTF_SCORE", 0.35))
        self.base_confidence_floor = float(os.environ.get("BASE_CONFIDENCE_FLOOR", 0.50))
        self.base_min_vol_zscore = float(os.environ.get("BASE_MIN_VOL_ZSCORE", 1.0))
        self.base_min_obs_imbalance = float(os.environ.get("BASE_MIN_OBS_IMBALANCE", 0.25))
        self.base_ema_atr_min = float(os.environ.get("BASE_EMA_ATR_MIN", 0.35))
        self.base_atr_pctl_min = float(os.environ.get("BASE_ATR_PCTL_MIN", 0.35))
        self.base_atr_pctl_max = float(os.environ.get("BASE_ATR_PCTL_MAX", 0.80))
        
        self.relax_max_quality = float(os.environ.get("RELAX_MAX_QUALITY", 0.12))
        self.relax_max_conf = float(os.environ.get("RELAX_MAX_CONF", 0.08))
        self.relax_max_mtf = float(os.environ.get("RELAX_MAX_MTF", 0.10))
        self.relax_max_volz = float(os.environ.get("RELAX_MAX_VOLZ", 1.0))
        self.relax_max_imb = float(os.environ.get("RELAX_MAX_IMB", 0.10))
        self.relax_max_atr_p_band = float(os.environ.get("RELAX_MAX_ATR_P_BAND", 0.10))
        self.relax_max_ema_atr = float(os.environ.get("RELAX_MAX_EMA_ATR", 0.10))
        
        self.near_miss_quality = float(os.environ.get("NEAR_MISS_QUALITY", 0.03))
        self.near_miss_conf = float(os.environ.get("NEAR_MISS_CONF", 0.03))
        self.near_miss_mtf = float(os.environ.get("NEAR_MISS_MTF", 0.05))
        self.near_miss_volz = float(os.environ.get("NEAR_MISS_VOLZ", 0.5))
        self.near_miss_imb = float(os.environ.get("NEAR_MISS_IMB", 0.05))
        self.near_miss_ema_atr = float(os.environ.get("NEAR_MISS_EMA_ATR", 0.05))
        self.day_end_catchup_hours = int(os.environ.get("DAY_END_CATCHUP_HOURS", 3))
        
        self.report_timezone = ZoneInfo(os.environ.get("REPORT_TIMEZONE", "Asia/Kolkata"))
        self.webhook_mode = os.environ.get("WEBHOOK_MODE", "false").lower() == "true"
        self.render_external_url = os.environ.get("RENDER_EXTERNAL_URL", "")
        self.webhook_secret = os.environ.get("WEBHOOK_SECRET", ''.join(random.choices(string.ascii_letters + string.digits, k=16)))
        
        self.db_url = os.environ.get('DB_URL', '')
        self.db_path = os.environ.get("DB_PATH", "./power_crypto_bot.db")
        if not self.db_url:
            self.db_url = f"sqlite+aiosqlite:///{self.db_path}"
        self.redis_url = os.environ.get("REDIS_URL", "").strip()
        self.skip_symbols = os.environ.get("SKIP_SYMBOLS", "XPIN/USDT,DOLO/USDT").split(",")
        
        self.max_retries = int(os.environ.get("MAX_RETRIES", 3))
        self.retry_delay = float(os.environ.get("RETRY_DELAY", 1.0))
        self.retry_backoff = float(os.environ.get("RETRY_BACKOFF", 2.0))
        
        self.flask_port = int(os.environ.get("FLASK_PORT", "8081"))
        self.environment = os.environ.get("ENVIRONMENT", "development")

        # --- FIX: Added use_ml attribute ---
        self.use_ml = os.environ.get("USE_ML", "true").lower() == "true"
        
        self.validate()
        
    def validate(self):
        required_vars = ['CRYPTO_BOT_TOKEN', 'CRYPTO_OWNER_ID', 'CRYPTO_GROUP_ID']
        missing = [var for var in required_vars if not os.environ.get(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
            
    @property
    def is_production(self):
        return self.environment == "production"

config = Config()


# ===================================================================
# UNIVERSAL GROUP FORWARDING SYSTEM WITH TIMESTAMPS
# ===================================================================

async def send_to_group(context: ContextTypes.DEFAULT_TYPE, message: str, message_type: str = "general"):
    """Universal function to send messages to group with proper formatting"""
    try:
        # Current timestamp in IST
        ist_now = datetime.now(config.report_timezone)
        timestamp_str = f"🕒 {ist_now.strftime('%d-%m-%Y %H:%M:%S IST')}"

        emoji_map = {
            "signal": "🚨",
            "position_closed": "🎯", 
            "position_update": "📊",
            "status": "📊",
            "health": "💚",
            "performance": "📈",
            "daily_summary": "📅",
            "notification": "📢",
            "error": "⚠️",
            "scan_result": "🔍",
            "system": "🤖",
            "general": "🤖"
        }

        emoji = emoji_map.get(message_type, "🤖")

        # Add timestamp to all group messages
        if message_type in ["signal", "position_closed", "position_update"]:
            group_message = f"{emoji} {message}\n\n{timestamp_str}\n📢 _Automated {message_type.replace('_', ' ').title()} from Trading Bot_"
        else:
            group_message = f"{emoji} {message}\n\n📢 _Automated {message_type.replace('_', ' ').title()} from Trading Bot_"

        await context.bot.send_message(
            chat_id=config.group_chat_id,
            text=group_message,
            parse_mode=ParseMode.MARKDOWN
        )
        logger.info("Message forwarded to group", message_type=message_type, group_id=config.group_chat_id)

    except Exception as e:
        logger.error(f"Failed to forward {message_type} to group", group_id=config.group_chat_id, error=str(e))


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
logger.info("database_config", db_url=config.db_url, redis_url=config.redis_url)

IS_PAUSED = False

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
                       execution_time=round(execution_time, 4),
                       status="success")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("function_performance",
                        function=func.__name__,
                        execution_time=round(execution_time, 4),
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
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info("gpu_memory_usage", 
                           allocated_gb=allocated, 
                           cached_gb=cached)
    
    # THREADING FIX: Add matplotlib cleanup
    def cleanup_matplotlib(self):
        """Clean up matplotlib resources"""
        try:
            plt.close('all')
            for i in plt.get_fignums():
                plt.close(i)
        except Exception:
            pass
                
    def guard(self):
        memory_percent = psutil.virtual_memory().percent
        self.usage_history.append(memory_percent)
        self.usage_history = self.usage_history[-100:]
        
        if memory_percent > self.max_memory_percent:
            logger.warning("high_memory_usage", percent=memory_percent)
            self.cleanup()
            self.cleanup_matplotlib()  # Add matplotlib cleanup

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
                        logger.error("max_retries_exceeded", function=func.__name__, error=str(e), max_retries=max_retries)
                        raise
                    wait_time = delay * (backoff ** (retries - 1))
                    logger.warning("retry_attempt", 
                                  function=func.__name__,
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
        'binanceusdm': {'window_sec': 60, 'weight_limit': 2400},
        'bybit': {'window_sec': 60, 'weight_limit': 600}, 
        'okx': {'window_sec': 60, 'weight_limit': 300}
    }
    
    def __init__(self):
        self.usage = defaultdict(int)
        self.window_start = defaultdict(lambda: time.time())
        self._lock = asyncio.Lock()
        
    async def acquire(self, exchange_name: str, weight: int = 1):
        async with self._lock:
            now = time.time()
            limits = self.LIMITS.get(exchange_name, {'window_sec': 60, 'weight_limit': 1200})
            window, limit = limits['window_sec'], limits['weight_limit']
            
            if now - self.window_start[exchange_name] >= window:
                self.window_start[exchange_name] = now
                self.usage[exchange_name] = 0
                
            if self.usage[exchange_name] + weight > limit:
                sleep_for = window - (now - self.window_start[exchange_name])
                sleep_for = max(0.0, sleep_for)
                logger.warning("rate_limit_exceeded", exchange=exchange_name, sleep_time=sleep_for)
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
        
    async def get_exchange(self, name: str = 'binance', use_futures: bool = True):
        cache_key = f"{name}{'_futures' if use_futures else ''}"
        if cache_key in self.exchanges:
            return self.exchanges[cache_key]
        
        if use_futures and name == 'binance':
            try:
                exchange_class = getattr(ccxt_async, 'binanceusdm', None)
                if exchange_class:
                    logger.info("Using dedicated binanceusdm for futures trading")
                    exchange_config = {
                        'enableRateLimit': True, 
                        'timeout': 30000,
                        'options': {
                            'adjustForTimeDifference': True,
                        },
                    }
                    
                    ex = exchange_class(exchange_config)
                    await ex.load_markets()
                    
                    if ex.markets:
                        logger.info("binanceusdm initialized successfully", 
                                   market_count=len(ex.markets))
                        self.exchanges[cache_key] = ex
                        return ex
                    else:
                        await ex.close()
                        
            except Exception as e:
                logger.warning("binanceusdm_init_failed", error=str(e))
        
        exchange_class = getattr(ccxt_async, name, None)
        if not exchange_class:
            raise ValueError(f"Exchange {name} not found in ccxt")
        
        exchange_config = {
            'enableRateLimit': True, 
            'timeout': 30000,
            'options': {
                'adjustForTimeDifference': True,
            },
        }
        
        if name == 'binance' and use_futures:
            exchange_config['options']['defaultType'] = 'future'
            exchange_config['options']['warnOnFetchOHLCVLimitArgument'] = False
        
        ex = exchange_class(exchange_config)
        
        try:
            await ex.load_markets()
            logger.info(f"Exchange {name} initialized successfully", 
                       market_count=len(ex.markets))
            
        except Exception as e:
            logger.error(f"Failed to initialize exchange {name}", error=str(e))
            await ex.close()
            raise
        
        self.exchanges[cache_key] = ex
        return ex

    # --- FIX: Implemented close_all method ---
    async def close_all(self):
        """Iterates through and cleanly closes all active exchange connections."""
        logger.info("closing_all_exchange_connections")
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                logger.info("closed_exchange", exchange=exchange_name)
            except Exception as e:
                logger.warning("failed_to_close_exchange", exchange=exchange_name, error=str(e))
        self.exchanges.clear()

exchange_factory = ExchangeFactory()

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

engine: AsyncEngine = create_async_engine(
    config.db_url, 
    poolclass=AsyncAdaptedQueuePool, 
    pool_size=30, 
    max_overflow=50, 
    pool_timeout=30,
    pool_recycle=1800, 
    pool_pre_ping=True, 
    echo=False
)

async def init_db():
    global engine
    try:
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        logger.info("database_initialized_successfully")
    except Exception as e:
        logger.error("database_initialization_failed", error=str(e))
        # This fallback logic is excellent for resilience.
        if "postgres" in config.db_url or "psycopg" in config.db_url:
            logger.info("falling_back_to_sqlite")
            config.db_url = f"sqlite+aiosqlite:///{config.db_path}"
            engine = create_async_engine(
                config.db_url, 
                poolclass=AsyncAdaptedQueuePool, 
                pool_size=30, 
                max_overflow=50, 
                pool_timeout=30,
                pool_recycle=1800, 
                pool_pre_ping=True, 
                echo=False
            )
            try:
                async with engine.begin() as conn:
                    await conn.run_sync(metadata.create_all)
                logger.info("sqlite_database_initialized_successfully")
            except Exception as e2:
                logger.critical("sqlite_also_failed", error=str(e2))
                raise

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
        self._init_file()
        
    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'symbol', 'direction', 'entry', 'exit', 
                               'pnl_percent', 'confidence', 'duration', 'market_regime'])
                
    def record_trade(self, trade_data: Dict[str, Any]):
        try:
            today_str = date.today().isoformat()
            if self.today != today_str:
                self.today = today_str
                self.filename = f"performance_{self.today}.csv"
                self._init_file()
                
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
                    trade_data.get('market_regime', '')
                ])
        except Exception as e:
            logger.error("performance_tracker_error", error=str(e))

performance_tracker = PerformanceTracker()

# ------------------------------------------------------------------------------
# Enhanced Risk Manager (With Correlation Check)
# ------------------------------------------------------------------------------

class EnhancedRiskManager:
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.daily_pnl = 0.0
        self.max_daily_loss = portfolio_value * config.max_daily_loss
        self.open_positions: Dict[str, float] = {}
        self.max_concurrent_trades = config.max_concurrent_trades
        self.circuit_breaker = False
        self.circuit_breaker_time = None
        
    def add_position(self, symbol: str, size_usd: float): 
        self.open_positions[symbol] = size_usd
        
    def remove_position(self, symbol: str): 
        self.open_positions.pop(symbol, None)
        
    @track_performance
    async def check_portfolio_correlation(self, new_symbol: str) -> Tuple[bool, str]:
        open_symbols = list(self.open_positions.keys())
        if not open_symbols: 
            return True, "OK"
            
        symbols_to_fetch = list(set(open_symbols + [new_symbol]))
        price_data = {}
        
        for symbol in symbols_to_fetch:
            df = await fetch_ohlcv_cached(symbol, '1h', limit=100, exchange_name='binance')
            if df is not None and not df.empty: 
                price_data[symbol] = df['close']
                
        if new_symbol not in price_data or len(price_data) < 2:
            return True, "Not enough data for correlation check"
            
        price_df = pd.DataFrame(price_data).ffill().dropna()
        if len(price_df) < 20:
             return True, "Not enough overlapping data for correlation check"
             
        for existing_symbol in open_symbols:
            if existing_symbol in price_df.columns:
                corr, _ = pearsonr(price_df[new_symbol], price_df[existing_symbol])
                if corr > config.max_portfolio_correlation:
                    return False, f"High correlation ({corr:.2f}) with open position {existing_symbol}"
                    
        return True, "OK"
        
    async def can_open_trade(self, symbol: str, proposed_size_usd: float) -> Tuple[bool, str]:
        if self.circuit_breaker:
            if self.circuit_breaker_time and (datetime.now() - self.circuit_breaker_time).total_seconds() > 3600:
                self.circuit_breaker, self.circuit_breaker_time = False, None
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
            
        is_safe, reason = await self.check_portfolio_correlation(symbol)
        if not is_safe: 
            return False, reason
            
        return True, "OK"
        
    def update_daily_pnl(self, pnl: float): 
        self.daily_pnl += pnl
        
    def reset_daily_pnl(self): 
        self.daily_pnl = 0.0
        
    def activate_circuit_breaker(self):
        self.circuit_breaker, self.circuit_breaker_time = True, datetime.now()
        logger.warning("circuit_breaker_activated")

risk_manager = EnhancedRiskManager(config.portfolio_value)

# ------------------------------------------------------------------------------
# Scan Analyzer
# ------------------------------------------------------------------------------

class ScanAnalyzer:
    def __init__(self):
        self.rejection_counts = defaultdict(int)
        
    def add_rejection(self, symbol: str, reason: str):
        self.rejection_counts[reason] += 1
        
    def get_summary(self, top_n=3):
        if not self.rejection_counts:
            return "Scan complete. No symbols were processed or all passed initial checks but failed later."
            
        total_rejections = sum(self.rejection_counts.values())
        summary = f"Scan Summary: Total rejections logged: {total_rejections}.\nTop reasons for rejection:\n"
        
        sorted_reasons = sorted(self.rejection_counts.items(), key=lambda item: item[1], reverse=True)
        for reason, count in sorted_reasons[:top_n]:
            percentage = (count / total_rejections) * 100
            summary += f"- {reason}: {count} times ({percentage:.1f}%)\n"
            
        return summary

# ------------------------------------------------------------------------------
# Model Validator, MTF Analyzer, Portfolio Optimizer, etc.
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
            
        score, total_weight = 0.0, 0.0
        for tf, data in analysis.items():
            if tf in self.weights:
                weight = self.weights[tf]
                score += (data['trend_score'] * weight) + (data['momentum_score'] * weight * 0.5)
                total_weight += weight * 1.5
                
        return score / total_weight if total_weight > 0 else 0.5
        
    def _analyze_timeframe(self, df: pd.DataFrame):
        ema20, ema50 = (df['close'].ewm(span=20).mean().iloc[-1], 
                       df['close'].ewm(span=50).mean().iloc[-1])
        trend_score = 1.0 if ema20 > ema50 else 0.0
        
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean().iloc[-1]
        loss = -delta.clip(upper=0).rolling(window=14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 0.0
        rsi = 100 - (100 / (1 + rs)) if rs != 0 else 50
        momentum_score = (rsi - 50) / 50
        
        return {
            'trend_score': trend_score, 
            'momentum_score': (momentum_score + 1) / 2
        }

multi_timeframe_analyzer = MultiTimeframeAnalyzer()

class PortfolioOptimizer:
    @track_performance
    async def optimize_allocation(self, signals: List[Dict[str, Any]], portfolio_value: float):
        if not signals: 
            return {}
            
        total_conf = sum(s.get('confidence', 0.5) for s in signals)
        if total_conf == 0: 
            return {}
            
        allocations = {
            s['symbol']: (s.get('confidence', 0.5) / total_conf) * portfolio_value * 0.8 
            for s in signals
        }
        
        return allocations

portfolio_optimizer = PortfolioOptimizer()

def compute_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['candle_vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3.0)
    
    log_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
    if not log_returns.empty:
        hurst_exp = log_returns.rolling(window=100).std()
        df['hurst'] = hurst_exp
        if not hurst_exp.isna().all():
            q70, q30 = hurst_exp.quantile(0.7), hurst_exp.quantile(0.3)
            df['market_regime'] = np.where(df['hurst'] > q70, 'trending', 
                                         np.where(df['hurst'] < q30, 'mean_reverting', 'choppy'))
        else: 
            df['market_regime'] = 'choppy'
    else:
        df['hurst'], df['market_regime'] = np.nan, 'choppy'
        
    df['spread_ratio'] = (df['high'] - df['low']) / df['volume'].replace(0, 1e-6)
    df['absorption'] = (df['volume'] * (df['high'] - df['low'])).rolling(14).sum()
    df['price_change_pct'] = df['close'].pct_change()
    
    rolling_vol = df['volume'].rolling(window=20)
    df['volume_zscore'] = (df['volume'] - rolling_vol.mean()) / (rolling_vol.std() + 1e-9)
    df['liquidation_spike'] = ((df['volume_zscore'] > 3.0) & 
                              (abs(df['price_change_pct']) > 0.025)).astype(int)
    
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=config.rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(window=config.rsi_period).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 1e-9))))
    
    ema_fast = df['close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_fast - ema_slow
    df['macd_sig'] = (ema_fast - ema_slow).ewm(span=9, adjust=False).mean()
    
    tr = pd.concat([
        (df['high'] - df['low']), 
        (df['high'] - df['close'].shift()).abs(), 
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(config.atr_period, min_periods=config.atr_period).mean()
    
    return compute_advanced_features(df)

def _volume_zscore(df: pd.DataFrame, lookback: int = 20) -> float:
    v = df['volume']
    if len(v) < lookback + 2: 
        return 0.0
        
    base = v.tail(lookback + 1).iloc[:-1]
    mu, sigma = base.mean(), base.std()
    if pd.isna(mu) or pd.isna(sigma) or sigma == 0: 
        return 0.0
        
    return float((v.iloc[-1] - mu) / sigma)

def _ema_spread_vs_atr(df: pd.DataFrame) -> float:
    if len(df) < 2: 
        return 0.0
        
    ema20, ema50, atr = df['ema20'].iloc[-2], df['ema50'].iloc[-2], df['atr'].iloc[-2]
    if pd.isna(ema20) or pd.isna(ema50) or pd.isna(atr) or atr <= 0: 
        return 0.0
        
    return float(abs(ema20 - ema50) / atr)

def _atr_percentile(df: pd.DataFrame, window: int = 200) -> float:
    atr_series = df['atr'].dropna()
    if len(atr_series) < window + 2: 
        return 0.5
        
    base, cur = atr_series.tail(window + 1).iloc[:-1], atr_series.iloc[-1]
    if pd.isna(cur): 
        return 0.5
        
    return float((base < cur).sum() / max(1, len(base)))

def _signal_quality_score(confidence: float, mtf_score: float, ema_atr_score: float, 
                         vol_z: float, ob_micro_alpha: float) -> float:
    vol_term = max(0.0, min(1.0, vol_z / 3.0))
    ob_term = max(0.0, min(1.0, (ob_micro_alpha * 100)))
    
    return float(0.35*confidence + 0.25*mtf_score + 0.20*max(0.0, min(1.0, ema_atr_score)) + 
                0.10*vol_term + 0.10*ob_term)

def _day_progress(tz: ZoneInfo) -> float:
    now = datetime.now(tz)
    start = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=tz)
    end = start + timedelta(days=1)
    return max(0.0, min(1.0, (now - start).total_seconds() / (end - start).total_seconds()))

def _expected_by_now(target: int, tz: ZoneInfo) -> int:
    return int(round(target * math.sqrt(_day_progress(tz))))

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
    if (24 - now.hour - now.minute / 60.0) <= config.day_end_catchup_hours:
        relax_alpha = min(1.0, max(relax_alpha, 0.5))

    # FIXED: lower hard-coded minimum caps so your BASE_* env values are respected
    thresh = {
        "min_quality": max(0.25, config.base_min_quality_score - config.relax_max_quality * relax_alpha),
        "min_conf":    max(0.25, config.base_confidence_floor - config.relax_max_conf   * relax_alpha),
        "min_mtf":     max(0.25, config.base_require_mtf_score - config.relax_max_mtf   * relax_alpha),
        "min_volz":    max(0.00, config.base_min_vol_zscore   - config.relax_max_volz  * relax_alpha),
        "min_imb":     max(0.02, config.base_min_obs_imbalance - config.relax_max_imb   * relax_alpha),
        "min_ema_atr": max(0.05, config.base_ema_atr_min      - config.relax_max_ema_atr* relax_alpha),
        "atr_p_min":   max(0.05, config.base_atr_pctl_min     - config.relax_max_atr_p_band * relax_alpha),
        "atr_p_max":   min(0.98, config.base_atr_pctl_max     + config.relax_max_atr_p_band * relax_alpha),
    }

    near = {
        "dq":     config.near_miss_quality  * relax_alpha,
        "dc":     config.near_miss_conf     * relax_alpha,
        "dm":     config.near_miss_mtf      * relax_alpha,
        "dvz":    config.near_miss_volz     * relax_alpha,
        "dimb":   config.near_miss_imb      * relax_alpha,
        "dema":   config.near_miss_ema_atr  * relax_alpha,
        "active": (relax_alpha > 0.2)
                  or ((24 - now.hour - now.minute / 60.0) <= config.day_end_catchup_hours),
    }

    return {
        "thresh":      thresh,
        "near":        near,
        "exp_now":     exp_now,
        "relax_alpha": relax_alpha,
    }

class HybridModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, nhead=4):
        super().__init__()
        self.recurrent_layer = nn.GRU(input_size, hidden_size, batch_first=True, 
                                     num_layers=2, dropout=0.2)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=256, 
            dropout=0.2, activation='relu', batch_first=True
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
        out = self.dropout(self.relu(self.fc1(trans_out[:, -1, :])))
        return self.sigmoid(self.fc2(out))

hybrid_model: Optional[nn.Module] = None

def load_hybrid_model(path: str) -> Optional[nn.Module]:
    if not PYTORCH_AVAILABLE: 
        return None
        
    if not os.path.exists(path):
        logger.warning("model_not_found", path=path)
        return None
        
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybridModel()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        logger.info("model_loaded", device=str(device))
        return model
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        return None

# The model is loaded in the main() function to ensure config is ready
# hybrid_model = load_hybrid_model(config.hybrid_model_path)

def calculate_fallback_confidence(df: pd.DataFrame) -> float:
    try:
        recent_returns = df['close'].pct_change().tail(5)
        avg_return = recent_returns.mean() if not pd.isna(recent_returns.mean()) else 0.0
        
        vol_mean = df['volume'].rolling(20).mean()
        vol_ratio = (df['volume'].iloc[-1] / vol_mean.iloc[-2] 
                    if len(vol_mean) > 1 and vol_mean.iloc[-2] > 0 else 1.0)
        
        confidence = (0.5 + min(0.2, max(-0.2, avg_return * 10)) + 
                     min(0.2, max(-0.2, (vol_ratio - 1.0) * 0.1)))
        
        return max(0.1, min(0.9, confidence))
    except Exception: 
        return 0.5

def predict_signal_confidence(df: pd.DataFrame, seq_len: int = 50) -> float:
    if hybrid_model is None or not PYTORCH_AVAILABLE or len(df) < 30:
        return calculate_fallback_confidence(df)
        
    try:
        feature_cols = ['close', 'volume', 'rsi', 'macd', 'macd_sig', 'atr', 
                       'candle_vwap', 'hurst', 'spread_ratio', 'absorption']
        df_features = df[feature_cols].dropna()
        
        if len(df_features) < 30: 
            return calculate_fallback_confidence(df)
            
        data_subset = df_features.tail(min(seq_len, len(df_features))).values
        scaled = (data_subset - np.mean(data_subset, axis=0)) / (np.std(data_subset, axis=0) + 1e-7)
        
        device = next(hybrid_model.parameters()).device
        with torch.no_grad():
            memory_manager.guard()
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0).to(device)
            return float(hybrid_model(tensor).item())
            
    except Exception as e:
        logger.error("ml_prediction_failed", error=str(e))
        return calculate_fallback_confidence(df)

def _pickle_df(df: pd.DataFrame) -> bytes: 
    return pickle.dumps(df)

def _unpickle_df(blob: bytes) -> Optional[pd.DataFrame]:
    try: 
        return pickle.loads(blob)
    except Exception: 
        return None

async def cache_set(key: str, value: bytes, ttl: int = None):
    if redis_client:
        try: 
            await redis_client.setex(key, ttl or config.cache_ttl, value)
        except Exception as e: 
            logger.warning("redis_set_failed", error=str(e))

async def cache_get(key: str) -> Optional[bytes]:
    if redis_client:
        try:
            if data := await redis_client.get(key):
                await redis_client.expire(key, config.cache_ttl)
                return data
        except Exception as e: 
            logger.warning("redis_get_failed", error=str(e))
    return None

@with_retry()
@track_performance
async def ccxt_call(exchange_name: str, method: str, weight: int, *args, **kwargs):
    ex = await exchange_factory.get_exchange(exchange_name, use_futures=True)
    await rate_limiter.acquire(exchange_name, weight)
    return await getattr(ex, method)(*args, **kwargs)

@with_retry()
@track_performance
async def fetch_ohlcv_cached(symbol: str, timeframe: str, limit: int = 100, 
                           exchange_name: str = 'binance') -> Optional[pd.DataFrame]:
    cache_key = f"ohlcv:{exchange_name}:{symbol}:{timeframe}:{limit}"
    
    if blob := await cache_get(cache_key):
        if (df := _unpickle_df(blob)) is not None and len(df) >= 20: 
            return df
            
    try:
        async with engine.begin() as conn:
            res = await conn.execute(
                sql_text("SELECT fetched_at, blob FROM ohlcv_cache WHERE market_id = :m AND timeframe = :t"), 
                {"m": symbol, "t": timeframe}
            )
            if (row := res.fetchone()) and (datetime.now(timezone.utc) - 
                                          datetime.fromisoformat(row[0])).total_seconds() < config.cache_ttl:
                if (df := _unpickle_df(row[1])) is not None and len(df) >= 20:
                    await cache_set(cache_key, row[1])
                    return df
    except Exception as e: 
        logger.warning("db_cache_read_failed", error=str(e))
        
    try:
        bars = await ccxt_call(exchange_name, 'fetch_ohlcv', 1, symbol, timeframe, limit=limit)
        if not bars or len(bars) < 20: 
            return None
            
        df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        if df.isnull().values.any(): 
            return None
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], downcast='float')
            
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        
        blob = _pickle_df(df)
        await cache_set(cache_key, blob)
        
        try:
            async with engine.begin() as conn:
                await conn.execute(sql_text(
                    "INSERT INTO ohlcv_cache(market_id, timeframe, fetched_at, blob) "
                    "VALUES (:m, :t, :f, :b) ON CONFLICT(market_id, timeframe) "
                    "DO UPDATE SET fetched_at = :f, blob = :b"
                ), {
                    "m": symbol, "t": timeframe, 
                    "f": datetime.now(timezone.utc).isoformat(), "b": blob
                })
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
        bids, asks = ob.get('bids', []), ob.get('asks', [])
        
        if not bids or not asks: 
            return {}
            
        mid_price = (bids[0][0] + asks[0][0]) / 2.0
        spread = asks[0][0] - bids[0][0]
        bid_volume, ask_volume = sum(b[1] for b in bids[:5]), sum(a[1] for a in asks[:5])
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

@with_retry(max_retries=2, delay=0.5)
@track_performance
async def fetch_funding_rate(symbol: str, exchange_name: str = 'binance') -> Optional[float]:
    try:
        rate_data = await ccxt_call(exchange_name, 'fetch_funding_rate', 1, symbol)
        return float(rate_data['fundingRate'])
    except Exception as e:
        logger.warning("funding_rate_fetch_failed", symbol=symbol, error=str(e))
        return None

def dynamic_position_sizing(portfolio_value: float, volatility_usd: float, confidence: float) -> float:
    if volatility_usd <= 0 or (edge := confidence - 0.5) <= 0: 
        return 0.0
        
    risk_fraction = min(config.max_daily_loss, edge * 0.1)
    usd_to_risk = portfolio_value * risk_fraction
    return max(0.0, usd_to_risk / volatility_usd)

def get_optimal_parameters(market_regime: str) -> Dict[str, float]:
    regimes = {
        'trending': {'RSI_BUY': 52, 'RSI_SELL': 48}, 
        'mean_reverting': {'RSI_BUY': 45, 'RSI_SELL': 55}, 
        'choppy': {'RSI_BUY': 50, 'RSI_SELL': 50}
    }
    return regimes.get(market_regime, regimes['choppy'])

@track_performance
async def validate_signal(signal: Dict[str, Any]) -> bool:
    if not signal: 
        return False
        
    try:
        df = await fetch_ohlcv_cached(signal['market_id'], '5m', limit=10, exchange_name='binance')
        if df is None or len(df) < 5: 
            return True
            
        recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        if signal['side'] == 'Long' and recent_trend < -0.01:
            logger.warning("rejecting_long_signal", recent_trend=recent_trend)
            return False
            
        if signal['side'] == 'Short' and recent_trend > 0.01:
            logger.warning("rejecting_short_signal", recent_trend=recent_trend)
            return False
            
    except Exception as e:
        logger.error("signal_validation_error", error=str(e))
        
    return True

@track_performance
async def generate_signal(market_id: str, display_symbol: str, cooldowns: Dict[str, datetime],
                          exchange_name: str, dyn_policy: Dict[str, Any], 
                          analyzer: ScanAnalyzer) -> Optional[Dict[str, Any]]:
    df = None
    try:
        if any(skip in display_symbol for skip in config.skip_symbols): 
            return None
            
        df = await fetch_ohlcv_cached(market_id, config.timeframe, limit=200, exchange_name=exchange_name)

        if df is None or len(df) < 120:
            analyzer.add_rejection(display_symbol, "Insufficient historical data")
            return None
            
        df = compute_indicators(df)
        if df.iloc[-2].isna().any():
            analyzer.add_rejection(display_symbol, "Indicators contain NaN")
            return None
            
        last = df.iloc[-2]
        
        if market_id in cooldowns and cooldowns[market_id] >= df.index[-2]:
            analyzer.add_rejection(display_symbol, "On cooldown")
            return None
            
        if last['liquidation_spike'] == 1:
            analyzer.add_rejection(display_symbol, "Recent liquidation spike")
            return None
            
        is_uptrend, is_downtrend = last["ema20"] > last["ema50"], last["ema20"] < last["ema50"]
        regime = str(last.get("market_regime", "choppy"))
        params = get_optimal_parameters(regime)
        
        side = ("Long" if is_uptrend and last["rsi"] > params['RSI_BUY'] else 
                "Short" if is_downtrend and last["rsi"] < params['RSI_SELL'] else None)
                
        if not side:
            analyzer.add_rejection(display_symbol, "Primary condition (EMA/RSI) not met")
            return None
            
        funding_rate = await fetch_funding_rate(display_symbol, exchange_name)
        if funding_rate is not None:
            if side == "Long" and funding_rate > 0.00075:
                analyzer.add_rejection(display_symbol, "High funding rate")
                return None
            if side == "Short" and funding_rate < -0.00075:
                analyzer.add_rejection(display_symbol, "Negative funding rate")
                return None
                
        ticker = await ccxt_call(exchange_name, 'fetch_ticker', 1, display_symbol)
        bid, ask = ticker.get('bid'), ticker.get('ask')
        if bid and ask and ask > 0 and ((ask - bid) / ask * 100) > config.max_slippage_percent:
            analyzer.add_rejection(display_symbol, "High slippage (wide spread)")
            return None
            
        mtf_score = await multi_timeframe_analyzer.analyze_multi_timeframe(market_id, exchange_name)
        confidence = predict_signal_confidence(df)
        ob_features = await fetch_orderbook_features(display_symbol, exchange_name)
        
        ema_atr, atr_pctl, vol_z = (_ema_spread_vs_atr(df), _atr_percentile(df), 
                                   _volume_zscore(df))
        
        T = dyn_policy['thresh']
        if not (T['atr_p_min'] <= atr_pctl <= T['atr_p_max']):
            analyzer.add_rejection(display_symbol, "ATR percentile out of band")
            return None
            
        if ema_atr < T['min_ema_atr']:
            analyzer.add_rejection(display_symbol, "EMA/ATR spread too low")
            return None
            
        q = _signal_quality_score(confidence, mtf_score, ema_atr, vol_z, 
                                 ob_features.get('micro_alpha', 0.0))
        
        strict_ok = (confidence >= T['min_conf'] and mtf_score >= T['min_mtf'] and 
                    vol_z >= T['min_volz'] and abs(ob_features.get('imbalance', 0.0)) >= T['min_imb'] and 
                    q >= T['min_quality'])
        
        near, near_ok = dyn_policy['near'], False
        if near['active'] and not strict_ok:
            near_ok = (confidence >= max(0.5, T['min_conf'] - near['dc']) and 
                      mtf_score >= max(0.5, T['min_mtf'] - near['dm']) and
                      vol_z >= max(0.0, T['min_volz'] - near['dvz']) and 
                      abs(ob_features.get('imbalance', 0.0)) >= max(0.0, T['min_imb'] - near['dimb']) and
                      q >= max(0.5, T['min_quality'] - near['dq']) and 
                      ema_atr >= max(0.1, T['min_ema_atr'] - near['dema']))

        # --- DEBUGGING BLOCK ADDED AS REQUESTED ---
        if not (strict_ok or near_ok):
            rejection_details = {
                "symbol": display_symbol,
                "quality_score": round(q, 2),
                "conf": round(confidence, 2),
                "mtf": round(mtf_score, 2),
                "vol_z": round(vol_z, 2),
                "imbalance": round(ob_features.get('imbalance', 0.0), 2),
                "ema_atr": round(ema_atr, 2),
                "atr_pctl": round(atr_pctl, 2)
            }
            logger.info("final_check_rejection_details", **rejection_details)
        # --- END OF DEBUGGING BLOCK ---

        if not (strict_ok or near_ok):
            analyzer.add_rejection(display_symbol, "Final quality check failed")
            return None
            
        entry_price = float(last["close"])
        atr = float(last["atr"]) if not pd.isna(last["atr"]) else entry_price * 0.02
        sl_dist = atr * config.sl_mult
        position_size_coin = dynamic_position_sizing(config.portfolio_value, sl_dist, confidence)
        position_value = position_size_coin * entry_price
        
        can_trade, reason = await risk_manager.can_open_trade(display_symbol, position_value)
        if not can_trade:
            analyzer.add_rejection(display_symbol, f"Risk Manager: {reason}")
            return None
            
        logger.info("signal_generated_successfully", symbol=display_symbol, 
                   quality_score=round(q, 2), side=side)
        
        sl = entry_price - sl_dist if side == "Long" else entry_price + sl_dist
        tps = ([entry_price + atr * m for m in config.tp_mult] if side == "Long" else 
               [entry_price - atr * m for m in config.tp_mult])
        
        signal_data = {
            "symbol": display_symbol, "market_id": market_id, "side": side, 
            "quality": q, "confidence": confidence, "mtf_score": mtf_score,
            "vol_z": vol_z, "ob_imb": ob_features.get('imbalance', 0.0), 
            "ob_micro_alpha": ob_features.get('micro_alpha', 0.0),
            "entry": entry_price, "sl": sl, "tps": tps, "regime": regime, 
            "position_size_coin": position_size_coin, "position_value": position_value,
            "last_time": df.index[-2], 
            "text": format_alert(display_symbol, side, entry_price, sl, tps, 
                           confidence, position_size_coin, regime, datetime.now(timezone.utc)),
            "strict": strict_ok, "near_miss": (not strict_ok) and near_ok
        }
        
        if not await validate_signal(signal_data):
            analyzer.add_rejection(display_symbol, "Final validation failed")
            return None
            
        return signal_data
        
    except Exception as e:
        logger.exception("signal_generation_error", market_id=market_id, error=str(e))
        analyzer.add_rejection(display_symbol, f"Error: {e}")
        return None
    finally:
        del df

# THREADING FIX: Completely new plot_annotated_chart function with proper threading
@track_performance
async def plot_annotated_chart(df: pd.DataFrame, display_symbol: str, 
                             entry: float, sl: float, tps: list) -> str:
    """
    Generate annotated trading chart with proper thread handling
    Eliminates RuntimeError: main thread is not in main loop warnings
    """
    if not MATPLOTLIB_AVAILABLE:
        return ""
    
    def plot_chart():
        """Thread-safe plotting function that runs in executor"""
        try:
            # Create a copy of data for plotting
            df_plot = df.tail(config.chart_candles).copy()
            
            # Create figure with Agg backend (thread-safe)
            fig, ax = plt.subplots(figsize=(12, 7), facecolor='#F0F0F0')
            ax.set_facecolor('#FFFFFF')
            
            # Separate up and down candles
            up = df_plot[df_plot.close >= df_plot.open]
            down = df_plot[df_plot.close < df_plot.open]
            
            # Plot candlesticks
            ax.vlines(up.index, up.low, up.high, color='#26a69a', linewidth=1)
            ax.vlines(down.index, down.low, down.high, color='#ef5350', linewidth=1)
            
            # Calculate candle width
            candle_width = (df_plot.index[1] - df_plot.index[0]) * 0.7
            
            # Plot candle bodies
            ax.bar(up.index, up.close - up.open, width=candle_width, bottom=up.open, color='#26a69a')
            ax.bar(down.index, down.close - down.open, width=candle_width, bottom=down.open, color='#ef5350')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=30)
            
            # Labels and title
            plt.ylabel('Price (USDT)')
            plt.title(f'{display_symbol} Signal ({config.timeframe})', fontsize=16, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # Add trading lines
            xmin, xmax = mdates.date2num(df_plot.index[0]), mdates.date2num(df_plot.index[-1])
            
            # Entry line
            ax.hlines(entry, xmin, xmax, colors='green', linestyles='--', label=f'Entry: {entry:.4f}')
            
            # Stop loss line
            ax.hlines(sl, xmin, xmax, colors='red', linestyles='--', label=f'SL: {sl:.4f}')
            
            # Take profit lines
            for i, tp in enumerate(tps, start=1):
                ax.hlines(tp, xmin, xmax, colors='blue', alpha=0.6, linestyles='--', label=f'TP{i}: {tp:.4f}')
            
            plt.legend()
            plt.tight_layout()
            
            # Save chart with unique filename
            fname = f'chart_{display_symbol.replace("/", "")}_{int(time.time())}.png'
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            
            # Important: Close the figure to free memory and avoid threading issues
            plt.close(fig)
            
            return fname
            
        except Exception as e:
            logger.error("charting_failed", error=str(e))
            return ""
    
    # Run chart generation in thread pool to avoid GUI threading issues
    try:
        loop = asyncio.get_event_loop()
        # Use thread pool executor for matplotlib operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(executor, plot_chart)
            chart_filename = await future
            return chart_filename
    except Exception as e:
        logger.error("async_chart_generation_failed", error=str(e))
        return ""

def format_alert(symbol: str, side: str, entry: float, sl: float, tps: list, 
                confidence: float, position_size: float, regime: str, scan_time: datetime = None) -> str:
    # Add scan timestamp to signal
    if scan_time:
        ist_time = scan_time.astimezone(config.report_timezone)
        timestamp_str = ist_time.strftime('%d-%m-%Y %H:%M:%S IST')
        timestamp_line = f"🕒 *Scan Time:* `{timestamp_str}`\n"
    else:
        timestamp_line = ""

    return (f"📈 *{symbol} Signal ({config.timeframe})*\n\n"
            f"{timestamp_line}"
            f"{'🚀' if side == 'Long' else '📉'} *Trade Type:* `{side.upper()}`\n"
            f"🧠 *Confidence:* `{confidence*100:.1f}%`\n"
            f"📊 *Market Regime:* `{regime.title()}`\n\n"
            f"*Trade Parameters:*\n  - Entry: `{entry:,.4f}`\n  - Stop-loss: `{sl:,.4f}`\n\n"
            f"*Take-Profit Targets:*\n  - TP1: `{tps[0]:,.4f}`\n  - TP2: `{tps[1]:,.4f}`\n"
            f"  - TP3: `{tps[2]:,.4f}`\n\n"
            f"*Sizing & Risk (Based on ${config.portfolio_value:,} portfolio):*\n"
            f"  - Suggested Size: `{position_size:.4f} {symbol.split('/')[0]}`\n"
            f"  - Position Value: `${(position_size * entry):,.2f}`\n\n"
            f"⚡ _Move SL to entry after TP1 is hit._")

async def _get_daily_count() -> int:
    key = f"signals:day:{date.today().isoformat()}"
    if redis_client:
        try: 
            return int(v) if (v := await redis_client.get(key)) else 0
        except Exception: 
            return 0
    # Fallback to DB if redis fails
    try:
        async with engine.begin() as conn:
            db_type = engine.dialect.name
            date_func = "DATE(entry_time)" if db_type == 'sqlite' else "DATE(entry_time::timestamp)"
            today_str = date.today().isoformat()
            
            res = await conn.execute(sql_text(f"SELECT COUNT(*) FROM signals WHERE {date_func} = :today"), {"today": today_str})
            return res.scalar_one_or_none() or 0
    except Exception as e:
        logger.warning("db_daily_count_failed", error=str(e))
        return 0


async def batch_process_markets(
    markets: List[Tuple[str, Any]], cooldowns: Dict[str, datetime], 
    exchange_name: str, dyn_policy: Dict[str, Any], analyzer: ScanAnalyzer
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    
    tasks = [
        generate_signal(market[0], market[0], cooldowns, exchange_name, dyn_policy, analyzer)
        for market in markets
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    strict_signals, near_miss_signals = [], []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error("unhandled_exception_in_generate_signal", symbol=markets[i][0], error=str(result))
            analyzer.add_rejection(markets[i][0], f"Unhandled Error: {result}")
        elif result:
            (strict_signals if result['strict'] else near_miss_signals).append(result)

    return strict_signals, near_miss_signals

async def debug_markets():
    """Enhanced debug function to see what markets are available and their structure"""
    try:
        logger.info("=== ENHANCED MARKET DEBUG INFO ===")
        
        logger.info("Testing Method 1: binanceusdm")
        try:
            exchange_futures = await exchange_factory.get_exchange('binance', use_futures=True)
            markets_futures = exchange_futures.markets
            
            logger.info("binanceusdm markets loaded", total=len(markets_futures))
            
            usdt_futures = [s for s in markets_futures.keys() if '/USDT' in s]
            logger.info("USDT futures found", count=len(usdt_futures))
            
            if usdt_futures:
                sample = usdt_futures[0]
                sample_market = markets_futures[sample]
                logger.info("Sample futures market structure", 
                           symbol=sample,
                           type=sample_market.get('type'),
                           swap=sample_market.get('swap'),
                           future=sample_market.get('future'),
                           settle=sample_market.get('settle'),
                           active=sample_market.get('active'))
                
        except Exception as e:
            logger.error("Method 1 (binanceusdm) failed", error=str(e))
        
        logger.info("Testing Method 2: regular binance")
        try:
            exchange_regular = await exchange_factory.get_exchange('binance', use_futures=False)
            markets_regular = exchange_regular.markets
            
            logger.info("Regular binance markets loaded", total=len(markets_regular))
            
            market_types = defaultdict(int)
            contract_symbols = []
            for symbol, market in markets_regular.items():
                market_types[market.get('type', 'unknown')] += 1
                if ':' in symbol:
                    contract_symbols.append(symbol)
            
            logger.info("Market types breakdown", types=dict(market_types))
            logger.info("Contract symbols found", count=len(contract_symbols), 
                       samples=contract_symbols[:5] if contract_symbols else [])
            
            usdt_spot = [s for s in markets_regular.keys() 
                        if '/USDT' in s and markets_regular[s].get('active')]
            logger.info("Active USDT spot markets", count=len(usdt_spot))
            
        except Exception as e:
            logger.error("Method 2 (regular binance) failed", error=str(e))
                
    except Exception as e:
        logger.error("Debug markets failed", error=str(e))

@track_performance
async def scan_markets(context: Optional[ContextTypes.DEFAULT_TYPE] = None, exchange_name: str = 'binance'):
    logger.info("!!! RUNNING FULLY OPTIMIZED VERSION - SCANNER V1.13 !!!")
    
    if IS_PAUSED:
        logger.info("scan_skipped_bot_is_paused")
        return
        
    logger.info("market_scan_started")
    analyzer = ScanAnalyzer()
    cooldowns: Dict[str, datetime] = {}
    
    try:
        # --- FIX: Calculate dynamic policy at the start of the scan ---
        published_today = await _get_daily_count()
        # Set a reasonable target, e.g., max concurrent trades * 2, or make it a config variable
        daily_target = config.max_concurrent_trades * 2 
        dyn_policy = _compute_dynamic_policy(published_today, daily_target, config.report_timezone)
        logger.info("dynamic_policy_calculated", 
                    published=published_today, 
                    target=daily_target, 
                    relax_alpha=dyn_policy['relax_alpha'])
        
        # Determine how many new signals can be opened
        open_slots = config.max_concurrent_trades - len(risk_manager.open_positions)
        if open_slots <= 0:
            logger.info("scan_skipped_max_positions_reached")
            return

        exchange = await exchange_factory.get_exchange(exchange_name, use_futures=True)
        await exchange.load_markets(True) 
        markets = exchange.markets
        logger.info("Fetched markets from exchange", total_markets=len(markets))
        
        futures_markets = {}
        for symbol, market in markets.items():
            is_futures = (
                market.get('swap', False) or market.get('future', False) or
                market.get('type') in ['swap', 'future'] or
                ':' in symbol or market.get('linear', False) or market.get('inverse', False)
            )
            is_usdt_margined = ('/USDT' in symbol and (market.get('settle') == 'USDT' or market.get('quote') == 'USDT'))
            
            if (is_futures or 'usdm' in str(type(exchange).__name__.lower())) and is_usdt_margined and market.get('active'):
                futures_markets[symbol] = market
                
        logger.info("Filtered for USDT futures markets", futures_count=len(futures_markets))
        
        if not futures_markets:
            logger.error("No tradable markets found")
            await debug_markets()
            return

        # --- CORRECTED MEMORY OPTIMIZATION ---
        all_symbols = list(futures_markets.keys())
        swap_symbols = [s for s in all_symbols if futures_markets.get(s, {}).get('type') == 'swap']
        future_symbols = [s for s in all_symbols if futures_markets.get(s, {}).get('type') == 'future']
        
        logger.info("Separated market types", swap_count=len(swap_symbols), future_count=len(future_symbols))
        
        tickers = {}
        ticker_batch_size = 100

        async def fetch_tickers_by_type(symbols_list: List[str], market_type: str):
            if not symbols_list: return
            logger.info(f"Fetching tickers for {market_type} symbols", count=len(symbols_list))
            for i in range(0, len(symbols_list), ticker_batch_size):
                batch = symbols_list[i:i + ticker_batch_size]
                try:
                    batch_tickers = await ccxt_call(exchange_name, 'fetch_tickers', 2, symbols=batch)
                    tickers.update(batch_tickers)
                    logger.info(f"Fetched {market_type} ticker batch {i//ticker_batch_size + 1}/{-(-len(symbols_list)//ticker_batch_size)}")
                    await asyncio.sleep(0.2)
                except Exception as e:
                    logger.error(f"Error fetching {market_type} ticker batch {i//ticker_batch_size + 1}", error=str(e))
        
        await fetch_tickers_by_type(swap_symbols, "swap")
        await fetch_tickers_by_type(future_symbols, "future")
        
        logger.info("Total live tickers fetched", ticker_count=len(tickers))
        
        if not tickers:
            logger.error("No live tickers fetched")
            return

        sorted_live_tickers = []
        for symbol, ticker in tickers.items():
            volume = ticker.get('quoteVolume', 0) or 0
            if volume > 0:
                sorted_live_tickers.append((symbol, ticker))
                
        sorted_live_tickers.sort(key=lambda x: x[1].get('quoteVolume', 0), reverse=True)
        logger.info("Sorted live tickers by current volume", count=len(sorted_live_tickers))
        
        if sorted_live_tickers:
            logger.info("Top 5 tickers by current volume", 
                       top_5=[(s, t.get('quoteVolume', 0)) for s, t in sorted_live_tickers[:5]])

        # Create the final list of markets to process
        markets_to_process = sorted_live_tickers[:config.top_n_markets]

        # Explicitly delete the large, now-unnecessary data structures
        del tickers
        del sorted_live_tickers
        gc.collect() 
        logger.info("Cleaned up full ticker list from memory.")

        strict, near_miss = [], []
        scan_batch_size = 15
        
        logger.info("Starting final analysis in batches", batch_size=scan_batch_size, total_markets=len(markets_to_process))

        for i in range(0, len(markets_to_process), scan_batch_size):
            batch = markets_to_process[i:i + scan_batch_size]
            logger.info(f"Processing batch {(i // scan_batch_size) + 1}/{-(-len(markets_to_process) // scan_batch_size)}")

            s_batch, nm_batch = await batch_process_markets(
                batch, cooldowns, exchange_name, dyn_policy, analyzer
            )
            strict.extend(s_batch)
            near_miss.extend(nm_batch)

            memory_manager.cleanup()
            await asyncio.sleep(1)
        
        if not strict and not near_miss:
            logger.info("no_qualified_candidates_found", summary=analyzer.get_summary())
            return

        strict.sort(key=lambda c: c['quality'], reverse=True)
        near_miss.sort(key=lambda c: c['quality'], reverse=True)
        selected: List[Dict[str, Any]] = []
        
        longs = [c for c in strict if c['side'] == 'Long']
        shorts = [c for c in strict if c['side'] == 'Short']
        
        # --- FIX: Use `open_slots` instead of undefined `remaining_total` ---
        if longs: 
            selected.append(longs[0])
            logger.info("Selected long signal", symbol=longs[0]['symbol'], quality=longs[0]['quality'])
            
        if shorts and len(selected) < open_slots: 
            selected.append(shorts[0])
            logger.info("Selected short signal", symbol=shorts[0]['symbol'], quality=shorts[0]['quality'])
        
        used = {(s['symbol'], s['side']) for s in selected}
        for c in strict:
            if len(selected) >= open_slots: 
                break
            if (c['symbol'], c['side']) not in used:
                selected.append(c)
                used.add((c['symbol'], c['side']))
                logger.info("Added strict signal", symbol=c['symbol'], side=c['side'], quality=c['quality'])
                
        if len(selected) < open_slots:
            for c in near_miss:
                if len(selected) >= open_slots: 
                    break
                if (c['symbol'], c['side']) not in used:
                    selected.append(c)
                    used.add((c['symbol'], c['side']))
                    logger.info("Added near-miss signal", symbol=c['symbol'], side=c['side'], quality=c['quality'])
                    
        if not selected:
            logger.info("no_candidates_for_final_selection", summary=analyzer.get_summary())
            return
            
        logger.info("Final selected signals", count=len(selected), 
                   symbols=[s['symbol'] for s in selected])
            
        committed = 0
        for s in selected:
            can_trade, reason = await risk_manager.can_open_trade(s['symbol'], s['position_value'])
            if not can_trade:
                logger.info("risk_manager_rejected_signal", symbol=s['symbol'], reason=reason)
                continue
                
            risk_manager.add_position(s['symbol'], s['position_value'])
            
            async with engine.begin() as conn:
                res = await conn.execute(signals_table.insert().values(
                    market_id=s['market_id'], symbol=s['symbol'], direction=s['side'].lower(), 
                    entry_price=s['entry'], entry_time=pd.Timestamp(s['last_time']).isoformat(), 
                    tp1=s['tps'][0], tp2=s['tps'][1], tp3=s['tps'][2], sl=s['sl'], 
                    position_size=s['position_size_coin'], confidence=s['confidence'],
                    market_regime=s['regime'], status='open'
                ))
                signal_id = res.inserted_primary_key[0] if res.inserted_primary_key else None
                
            logger.info("Signal stored in database", symbol=s['symbol'], signal_id=signal_id)
                
            chart_path = ""
            try:
                chart_df = await fetch_ohlcv_cached(s['market_id'], config.timeframe, 200, exchange_name)
                if chart_df is not None and not chart_df.empty:
                    chart_path = await plot_annotated_chart(chart_df, s['symbol'], s['entry'], s['sl'], s['tps'])
            except Exception as e:
                logger.error("chart_generation_failed", symbol=s['symbol'], error=str(e))
                
            if context:
                try: 
                    await send_alert(context, {"text": s['text'], "chart": chart_path, "signal_id": signal_id})
                    logger.info("Alert sent successfully", symbol=s['symbol'])
                except Exception as e: 
                    logger.error("alert_send_failed", error=str(e))
                    
            committed += 1
            
        if committed > 0:
            logger.info("signals_published", count=committed)
        else:
            logger.info("no_signals_published_after_risk_checks")
            
    except Exception as e:
        logger.error("scan_markets_main_error", error=str(e), exc_info=True)

@track_performance
async def monitor_positions(context: Optional[ContextTypes.DEFAULT_TYPE] = None, exchange_name: str = 'binance'):
    try:
        async with engine.begin() as conn:
            rows = (await conn.execute(sql_text("SELECT * FROM signals WHERE status='open'"))).fetchall()
            
        if not rows: 
            return
        
        for r in rows:
            try:
                (signal_id, _, symbol, direction, entry_price, entry_time_str, tp1, tp2, tp3, sl, size, 
                 tp1_hit, tp2_hit, tp3_hit, _, _, _, _, confidence, market_regime) = r
                entry_time = datetime.fromisoformat(entry_time_str)
                
                cur_ticker = await ccxt_call(exchange_name, 'fetch_ticker', 1, symbol)
                price = float(cur_ticker.get('last', 0.0))
                if price <= 0: 
                    continue
                
                position_closed, exit_reason = False, ""
                
                if direction == 'long':
                    if price <= sl: 
                        position_closed, exit_reason = True, "Stop Loss"
                    elif price >= tp3 and not tp3_hit: 
                        position_closed, exit_reason = True, "TP3"
                    elif price >= tp2 and not tp2_hit: 
                        async with engine.begin() as conn:
                            await conn.execute(sql_text("UPDATE signals SET tp2_hit=1 WHERE id=:id"), {"id": signal_id})
                        # Send TP2 hit notification
                        if context:
                            tp_msg = f"🎯 *TP2 Hit!*\n\n📊 `{symbol} {direction.upper()}`\nTP2 Price: `${tp2:,.4f}`\nCurrent Price: `${price:,.4f}`"
                            try:
                                await context.bot.send_message(chat_id=config.owner_id_int, text=tp_msg, parse_mode=ParseMode.MARKDOWN)
                                await send_to_group(context, tp_msg, "position_update")
                            except Exception as e:
                                logger.error("tp2_notify_failed", error=str(e))
                    elif price >= tp1 and not tp1_hit: 
                        async with engine.begin() as conn:
                            await conn.execute(sql_text("UPDATE signals SET tp1_hit=1, sl=:new_sl WHERE id=:id"), 
                                             {"id": signal_id, "new_sl": entry_price})
                        # Send TP1 hit notification
                        if context:
                            tp_msg = f"🎯 *TP1 Hit - SL Moved to BE!*\n\n📊 `{symbol} {direction.upper()}`\nTP1 Price: `${tp1:,.4f}`\nCurrent Price: `${price:,.4f}`\n\n✅ Stop Loss moved to Entry Price for risk-free trade!"
                            try:
                                await context.bot.send_message(chat_id=config.owner_id_int, text=tp_msg, parse_mode=ParseMode.MARKDOWN)
                                await send_to_group(context, tp_msg, "position_update")
                            except Exception as e:
                                logger.error("tp1_notify_failed", error=str(e))
                else:
                    if price >= sl: 
                        position_closed, exit_reason = True, "Stop Loss"
                    elif price <= tp3 and not tp3_hit: 
                        position_closed, exit_reason = True, "TP3"
                    elif price <= tp2 and not tp2_hit: 
                        async with engine.begin() as conn:
                            await conn.execute(sql_text("UPDATE signals SET tp2_hit=1 WHERE id=:id"), {"id": signal_id})
                        # Send TP2 hit notification
                        if context:
                            tp_msg = f"🎯 *TP2 Hit!*\n\n📊 `{symbol} {direction.upper()}`\nTP2 Price: `${tp2:,.4f}`\nCurrent Price: `${price:,.4f}`"
                            try:
                                await context.bot.send_message(chat_id=config.owner_id_int, text=tp_msg, parse_mode=ParseMode.MARKDOWN)
                                await send_to_group(context, tp_msg, "position_update")
                            except Exception as e:
                                logger.error("tp2_notify_failed", error=str(e))
                    elif price <= tp1 and not tp1_hit: 
                        async with engine.begin() as conn:
                            await conn.execute(sql_text("UPDATE signals SET tp1_hit=1, sl=:new_sl WHERE id=:id"), 
                                             {"id": signal_id, "new_sl": entry_price})
                        # Send TP1 hit notification
                        if context:
                            tp_msg = f"🎯 *TP1 Hit - SL Moved to BE!*\n\n📊 `{symbol} {direction.upper()}`\nTP1 Price: `${tp1:,.4f}`\nCurrent Price: `${price:,.4f}`\n\n✅ Stop Loss moved to Entry Price for risk-free trade!"
                            try:
                                await context.bot.send_message(chat_id=config.owner_id_int, text=tp_msg, parse_mode=ParseMode.MARKDOWN)
                                await send_to_group(context, tp_msg, "position_update")
                            except Exception as e:
                                logger.error("tp1_notify_failed", error=str(e))
                        # Send TP1 hit notification
                        if context:
                            tp_msg = f"🎯 *TP1 Hit - SL Moved to BE!*\n\n📊 `{symbol} {direction.upper()}`\nTP1 Price: `${tp1:,.4f}`\nCurrent Price: `${price:,.4f}`\n\n✅ Stop Loss moved to Entry Price for risk-free trade!"
                            try:
                                await context.bot.send_message(chat_id=config.owner_id_int, text=tp_msg, parse_mode=ParseMode.MARKDOWN)
                                await send_to_group(context, tp_msg, "position_update")
                            except Exception as e:
                                logger.error("tp1_notify_failed", error=str(e))
                
                if position_closed:
                    pnl = (price - entry_price) * size if direction == 'long' else (entry_price - price) * size
                    exit_time = datetime.now(timezone.utc)
                    duration_minutes = (exit_time.replace(tzinfo=None) - entry_time).total_seconds() / 60
                    
                    async with engine.begin() as conn:
                        await conn.execute(sql_text("UPDATE signals SET status='closed', exit_price=:p, exit_time=:t, pnl=:pl WHERE id=:id"),
                                         {"p": price, "t": exit_time.isoformat(), "pl": pnl, "id": signal_id})
                    
                    risk_manager.remove_position(symbol)
                    risk_manager.update_daily_pnl(pnl)
                    model_validator.record_prediction(float(confidence or 0.5), pnl)
                    
                    try:
                        pnl_pct = (pnl / (entry_price * size)) * 100 if entry_price > 0 and size > 0 else 0.0
                        performance_tracker.record_trade({
                            'timestamp': exit_time.isoformat(), 'symbol': symbol, 'direction': direction, 
                            'entry_price': entry_price, 'exit_price': price, 'pnl_percent': pnl_pct, 
                            'confidence': float(confidence or 0.5), 'duration_minutes': duration_minutes, 
                            'market_regime': market_regime
                        })
                    except Exception as e: 
                        logger.warning("performance_recording_failed", error=str(e))
                    
                    # Send comprehensive position closed notification to both owner and group
                    if context:
                        pnl_emoji = "💚" if pnl > 0 else "❌"
                        duration_str = f"{duration_minutes:.0f}m" if duration_minutes < 60 else f"{duration_minutes/60:.1f}h"

                        note = (f"🎯 *Position Closed*\n\n"
                               f"📊 `{symbol} {direction.upper()}`\n"
                               f"{pnl_emoji} *PnL:* `${pnl:,.2f}` ({(pnl/(entry_price*size)*100):+.1f}%)\n"
                               f"🏁 *Reason:* {exit_reason}\n"
                               f"📈 *Entry:* `${entry_price:,.4f}`\n"
                               f"📉 *Exit:* `${price:,.4f}`\n"
                               f"⏱️ *Duration:* `{duration_str}`")
                        try: 
                            await context.bot.send_message(chat_id=config.owner_id_int, text=note, parse_mode=ParseMode.MARKDOWN)
                            await send_to_group(context, note, "position_closed")
                        except Exception as e: 
                            logger.error("close_notify_failed", error=str(e))
                            
            except Exception as e:
                logger.error("monitor_error", signal_id=r[0], error=str(e))
                
    except Exception as e:
        logger.error("monitor_loop_error", error=str(e))

# ------------------------------------------------------------------------------
# Telegram Bot Handlers
# ------------------------------------------------------------------------------

application: Optional[Application] = None

def is_authorized(update: Update) -> bool: 
    return bool(update.effective_user and config.owner_id and str(update.effective_user.id) == str(config.owner_id))

async def send_notification(context: ContextTypes.DEFAULT_TYPE, message: str):
    try: 
        await context.bot.send_message(chat_id=config.owner_id_int, text=message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e: 
        logger.error("notification_failed", error=str(e))

async def send_alert(context: ContextTypes.DEFAULT_TYPE, alert_data: dict):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("Move SL to BE", callback_data=f"be:{alert_data['signal_id']}"),
         InlineKeyboardButton("Close Manually", callback_data=f"close:{alert_data['signal_id']}")]
    ])
    
    if (chart_path := alert_data.get("chart", "")) and os.path.exists(chart_path):
        with open(chart_path, "rb") as chart_photo:
            await context.bot.send_photo(chat_id=config.owner_id_int, photo=chart_photo, 
                                       caption=alert_data["text"], reply_markup=kb, parse_mode=ParseMode.MARKDOWN)
        os.remove(chart_path)
    else:
        await context.bot.send_message(chat_id=config.owner_id_int, text=alert_data["text"], 
                                     reply_markup=kb, parse_mode=ParseMode.MARKDOWN)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    await update.message.reply_text(
        "🤖 *Crypto Trading Bot Started*\n\n"
        f"📊 Monitoring `{config.top_n_markets}` top markets\n⏱️ Timeframe: `{config.timeframe}`\n"
        f"💰 Portfolio: `${config.portfolio_value:,}`\n\n*Commands:*\n"
        "`/status` - Bot status & performance.\n`/positions` - Show open positions.\n"
        "`/daily_summary` - Today's trading PnL.\n`/scan_now` - Trigger a market scan now.\n"
        "`/market_info <SYMBOL>` - Get info (e.g. `BTC/USDT`).\n`/export_trades` - Get today's trade log file.\n"
        "`/config` - Show current bot settings.\n`/health` - Check system health.\n"
        "`/pause` - Pause new trade scanning.\n`/resume` - Resume trade scanning.\n"
        "`/close_all` - *Emergency close all positions.*\n",
        parse_mode=ParseMode.MARKDOWN
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    try:
        async with engine.begin() as conn:
            open_signals = (await conn.execute(sql_text("SELECT COUNT(*) FROM signals WHERE status='open'"))).scalar_one()
            total_closed = (await conn.execute(sql_text("SELECT COUNT(*) FROM signals WHERE status='closed'"))).scalar_one()
            
        status_message = (f"🤖 *Bot Status*\n\n*{'⏸️ Paused' if IS_PAUSED else '▶️ Running'}*\n"
                          f"🧠 *Model:* {'✅ Loaded' if hybrid_model else '❌ Fallback'}\n"
                          f"📊 *Open Positions:* {open_signals}\n📈 *Total Closed Trades:* {total_closed}\n"
                          f"💰 *Daily PnL:* `${risk_manager.daily_pnl:,.2f}`\n"
                          f"⚠️ *Risk Level:* {len(risk_manager.open_positions)}/{config.max_concurrent_trades}\n"
                          f"💾 *Memory Usage:* {psutil.virtual_memory().percent:.1f}%")
        await update.message.reply_text(status_message, parse_mode=ParseMode.MARKDOWN)
        
        # Forward to group
        await send_to_group(context, status_message, "status")
        
    except Exception as e:
        logger.error("status_error", error=str(e))
        await update.message.reply_text("❌ Error retrieving status")

async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    try:
        async with engine.begin() as conn:
            row = (await conn.execute(sql_text("SELECT COUNT(*), AVG(pnl), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), MAX(pnl), MIN(pnl) FROM signals WHERE status = 'closed'"))).fetchone()
            
        if row and row[0] > 0:
            total, avg_pnl, wins, best, worst = row
            win_rate = (wins / total) * 100 if wins and total > 0 else 0
            msg = (f"📊 *Lifetime Performance Summary*\n\n🎯 *Total Trades:* {total}\n✅ *Win Rate:* {win_rate:.1f}%\n"
                   f"💰 *Avg PnL:* `${avg_pnl or 0:,.2f}`\n🚀 *Best Trade:* `${best or 0:,.2f}`\n📉 *Worst Trade:* `${worst or 0:,.2f}`\n")
            
            if isinstance(perf := model_validator.calculate_model_performance(), dict):
                msg += f"\n🧠 *Model Metrics*\n🎯 *Accuracy:* {perf['accuracy']:.1%}\n📈 *Precision:* {perf['precision']:.1%}\n🔄 *Recall:* {perf['recall']:.1%}\n"
        else: 
            msg = "📊 No closed trades yet"
            
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error("performance_error", error=str(e))
        await update.message.reply_text("❌ Error retrieving performance data")

async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    try:
        async with engine.begin() as conn:
            rows = (await conn.execute(sql_text("SELECT symbol, direction, entry_price, sl, tp1 FROM signals WHERE status='open'"))).fetchall()
            
        if not rows: 
            await update.message.reply_text("No open positions.")
        else:
            lines = ["*📌 Open Positions:*\n"] + [f"- `{s}` ({d.upper()}) | E: `{e:.4f}` | SL: `{sl:.4f}` | TP1: `{t1:.4f}`" 
                                               for s, d, e, sl, t1 in rows]
            await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
            
    except Exception as e:
        logger.error("positions_error", error=str(e))
        await update.message.reply_text("❌ Error retrieving positions")

async def scan_now_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    await update.message.reply_text("🤖 Manual scan initiated...")
    logger.info("manual_scan_triggered_by_user")
    
    try:
        await scan_markets(context)
        await update.message.reply_text("✅ Manual scan complete.")
    except Exception as e:
        logger.error("manual_scan_failed", error=str(e))
        await update.message.reply_text(f"❌ Manual scan failed: {e}")

async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global IS_PAUSED
    if not is_authorized(update): 
        return
        
    IS_PAUSED = True
    logger.info("bot_paused_by_user")
    await update.message.reply_text("⏸️ Bot scanning paused.")

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global IS_PAUSED
    if not is_authorized(update): 
        return
        
    IS_PAUSED = False
    logger.info("bot_resumed_by_user")
    await update.message.reply_text("▶️ Bot scanning resumed.")

async def close_all_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    await update.message.reply_text("🚨 Closing all open positions...")
    
    try:
        async with engine.begin() as conn:
            open_positions = (await conn.execute(sql_text("SELECT id, symbol, direction, entry_price, position_size FROM signals WHERE status='open'"))).fetchall()
            
        if not open_positions:
            await update.message.reply_text("No open positions to close.")
            return
            
        tickers = await ccxt_call('binance', 'fetch_tickers', 1, [pos[1] for pos in open_positions])
        results = []
        
        for pos_id, symbol, direction, entry, size in open_positions:
            if ticker := tickers.get(symbol):
                price = float(ticker['last'])
                pnl = (price - entry) * size if direction == 'long' else (entry - price) * size
                
                async with engine.begin() as conn:
                    await conn.execute(sql_text("UPDATE signals SET status='closed', exit_price=:p, exit_time=:t, pnl=:pl WHERE id=:id"), 
                                       {"p": price, "t": datetime.now(timezone.utc).isoformat(), "pl": pnl, "id": pos_id})
                
                risk_manager.remove_position(symbol)
                risk_manager.update_daily_pnl(pnl)
                results.append(f"✅ Closed `{symbol}` for PnL: `${pnl:,.2f}`")
            else: 
                results.append(f"⚠️ Could not fetch price for `{symbol}`. Please close manually.")
                
        await update.message.reply_text("\n".join(results), parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error("close_all_failed", error=str(e))
        await update.message.reply_text(f"❌ Error during close_all: {e}")

async def daily_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    try:
        today_str = date.today().isoformat()
        db_type = engine.dialect.name
        date_func = "DATE(entry_time)" if db_type == 'sqlite' else "entry_time::date"
        
        async with engine.begin() as conn:
            row = (await conn.execute(sql_text(f"SELECT COUNT(*), SUM(pnl), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) FROM signals WHERE status = 'closed' AND {date_func} = '{today_str}'"))).fetchone()
            
        if row and row[0] > 0:
            total, total_pnl, winners = (row[0] or 0), (row[1] or 0), (row[2] or 0)
            win_rate = (winners / total) * 100 if total > 0 else 0
            msg = (f"📈 *Daily Summary for {today_str}*\n\n💰 *Net PnL:* `${total_pnl:,.2f}`\n"
                   f"📊 Trades: {total} | ✅ Winners: {winners} | ❌ Losers: {total - winners}\n🎯 Win Rate: {win_rate:.1f}%")
        else: 
            msg = "No trades closed today."
            
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error("daily_summary_failed", error=str(e))
        await update.message.reply_text(f"❌ Error fetching daily summary: {e}")

async def export_trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    try:
        filename = f"performance_{date.today().isoformat()}.csv"
        if os.path.exists(filename):
            await update.message.reply_document(document=open(filename, 'rb'))
        else: 
            await update.message.reply_text("No trades recorded today; file does not exist.")
            
    except Exception as e:
        logger.error("export_trades_failed", error=str(e))
        await update.message.reply_text(f"❌ Could not export trades: {e}")

async def market_info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    if not context.args:
        await update.message.reply_text("Usage: `/market_info BTC/USDT`")
        return
        
    symbol = context.args[0].upper()
    try:
        ticker = await ccxt_call('binance', 'fetch_ticker', 1, symbol)
        msg = (f"ℹ️ *Market Info for `{symbol}`*\n\n"
               f"💰 *Price:* `${ticker.get('last', 0):,.4f}`\n"
               f"📈 *24h Change:* `{ticker.get('percentage', 0):.2f}%`\n"
               f"📊 *24h High/Low:* `${ticker.get('high', 0):,.4f}` / `${ticker.get('low', 0):,.4f}`\n"
               f"📦 *24h Volume:* `${ticker.get('quoteVolume', 0):,.2f}`")
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error("market_info_failed", error=str(e))
        await update.message.reply_text(f"❌ Error fetching market info for `{symbol}`: {e}", parse_mode=ParseMode.MARKDOWN)

async def config_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    msg = (f"⚙️ *Current Bot Configuration*\n\n"
           f"📊 *Timeframe:* `{config.timeframe}`\n"
           f"🔍 *Markets Scanned:* `{config.top_n_markets}`\n"
           f"⚖️ *Max Concurrent Trades:* `{config.max_concurrent_trades}`\n"
           f"💰 *Max Position Size:* `{config.max_position_size * 100}%`\n"
           f"🚨 *Max Daily Loss:* `{config.max_daily_loss * 100}%`\n"
           f"📉 *Slippage Limit:* `{config.max_slippage_percent}%`\n"
           f"🔗 *Correlation Limit:* `{config.max_portfolio_correlation}`")
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): 
        return
        
    try: 
        exchange = await exchange_factory.get_exchange('binance', use_futures=True)
        await exchange.fetch_time()
        ex_status = "✅ Connected"
    except Exception: 
        ex_status = "❌ Disconnected"
        
    if redis_client:
        try: 
            await redis_client.ping()
            redis_status = "✅ Connected"
        except Exception: 
            redis_status = "❌ Disconnected"
    else: 
        redis_status = "⚪️ Disabled"
        
    mem_percent = psutil.virtual_memory().percent
    msg = (f"❤️ *System Health*\n\n"
           f"🔗 *Exchange:* {ex_status}\n💾 *Database:* `{engine.url.drivername}`\n"
           f"⚡ *Redis Cache:* {redis_status}\n🧠 *Memory Usage:* {'✅' if mem_percent < 90 else '⚠️'} {mem_percent}%")
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        
    # Forward to group
    await send_to_group(context, msg, "health")

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not (query and is_authorized(update)): 
        return
        
    await query.answer()
    
    try:
        action, signal_id_str = query.data.split(":")
        signal_id = int(signal_id_str)
        
        async with engine.begin() as conn:
            if action == "be":
                res = await conn.execute(sql_text("UPDATE signals SET sl = entry_price WHERE id = :id AND status='open'"), {"id": signal_id})
                msg = "\n\n✅ SL moved to Breakeven" if res.rowcount else "\n\n❌ Signal not found or already closed"
                
            elif action == "close":
                pos = (await conn.execute(sql_text("SELECT symbol, direction, entry_price, position_size FROM signals WHERE id=:id AND status='open'"), {"id": signal_id})).fetchone()
                
                if not pos: 
                    msg = "\n\n❌ Signal not found or already closed"
                else:
                    symbol, direction, entry, size = pos
                    price = float((await ccxt_call('binance', 'fetch_ticker', 1, symbol))['last'])
                    pnl = (price - entry) * size if direction == 'long' else (entry - price) * size
                    
                    await conn.execute(sql_text("UPDATE signals SET status='closed', exit_price=:p, exit_time=:t, pnl=:pl WHERE id=:id"), 
                                       {"p": price, "t": datetime.now(timezone.utc).isoformat(), "pl": pnl, "id": signal_id})
                    
                    risk_manager.remove_position(symbol)
                    risk_manager.update_daily_pnl(pnl)
                    msg = f"\n\n✅ Manually closed for PnL: ${pnl:,.2f}"
                    
        await query.edit_message_caption(caption=(query.message.caption or "") + msg, parse_mode=ParseMode.MARKDOWN)
        
    except Exception as e:
        logger.error("callback_error", error=str(e))
        await query.edit_message_caption(caption=(query.message.caption or "") + "\n\n❌ Error processing request", parse_mode=ParseMode.MARKDOWN)

# ------------------------------------------------------------------------------
# Flask Web Server
# ------------------------------------------------------------------------------

flask_app = Flask(__name__)

@flask_app.route('/health')
def health_check():
    try:
        import sqlite3
        if 'sqlite' in config.db_url:
            conn = sqlite3.connect(config.db_path)
            conn.close()
            db_ok = True
        else:
            db_ok = True
    except Exception:
        db_ok = False
        
    mem_percent = psutil.virtual_memory().percent
    status = 'healthy' if db_ok and mem_percent < 90 else 'degraded'
    
    return jsonify({
        'status': status, 
        'timestamp': datetime.now().isoformat(), 
        'memory_usage': mem_percent, 
        'database_connected': db_ok
    })

@flask_app.route('/')
def home():
    return jsonify({
        'message': 'Crypto Trading Bot is running', 
        'status': 'active', 
        'environment': config.environment
    })

def build_telegram_app() -> Application:
    app = Application.builder().token(config.bot_token).build()
    
    commands = {
        'start': start_command,
        'status': status_command,
        'performance': performance_command,
        'positions': positions_command,
        'scan_now': scan_now_command,
        'pause': pause_command,
        'resume': resume_command,
        'close_all': close_all_command,
        'daily_summary': daily_summary_command,
        'export_trades': export_trades_command,
        'market_info': market_info_command,
        'config': config_command,
        'health': health_command,
    }
    
    for command, handler in commands.items():
        app.add_handler(CommandHandler(command, handler))
    
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    return app

# ------------------------------------------------------------------------------
# Main Application Loop
# ------------------------------------------------------------------------------

# --- FIX: `scheduled_runner` now accepts the Telegram application instance ---
async def scheduled_runner(app: Application):
    try:
        scan_timer = 0
        while True:
            now = datetime.now(config.report_timezone)
            
            if now.hour == 0 and now.minute == 0 and now.second < config.monitor_interval:
                risk_manager.reset_daily_pnl()
                logger.info("daily_pnl_reset_for_new_day")
                
            if scan_timer <= 0:
                # Pass the application context to the scanner
                await scan_markets(app)
                scan_timer = config.scan_interval
                
            await monitor_positions(app)
            
            await asyncio.sleep(config.monitor_interval)
            scan_timer -= config.monitor_interval
            
    except asyncio.CancelledError:
        logger.info("scheduler_cancelled")
    finally:
        # Cleanup within the scheduler is not needed, handled in main
        pass

# --- FIX: Rewritten main function for clarity and robustness ---
async def main():
    """Main application entry point."""
    global hybrid_model
    logger.info("starting_advanced_crypto_bot", version="2.0-fixed")

    scheduler_task = None
    telegram_app = None

    try:
        await wait_for_network()
        await init_db()

        if config.use_ml:
            hybrid_model = load_hybrid_model(config.hybrid_model_path)
            logger.info("ml_model_status", loaded=bool(hybrid_model))

        telegram_app = build_telegram_app()
        # Pass the application instance to the scheduler
        scheduler_task = asyncio.create_task(scheduled_runner(telegram_app))

        logger.info("bot_initialization_complete", 
                   ml_enabled=config.use_ml,
                   redis_enabled=bool(redis_client),
                   environment=config.environment)

        # Run the bot and the scheduler concurrently
        async with telegram_app:
            await telegram_app.start()
            await telegram_app.updater.start_polling()
            logger.info("telegram_bot_started_successfully")
            await scheduler_task

    except (ConnectionError, URLError, TimeoutError) as e:
        logger.critical("network_error_preventing_bot_start", error=str(e))
        logger.info("Cannot start Telegram bot due to network issues. Shutting down.")
    except KeyboardInterrupt:
        logger.info("bot_shutdown_requested_by_user")
    except Exception as e:
        logger.critical("fatal_error_in_main_loop", error=str(e), exc_info=True)
    finally:
        logger.info("initiating_graceful_shutdown")
        if scheduler_task and not scheduler_task.done():
            scheduler_task.cancel()
            await asyncio.sleep(1) # Give it a moment to cancel

        if telegram_app and telegram_app.updater and telegram_app.updater.is_running:
            await telegram_app.updater.stop()
            await telegram_app.stop()
            logger.info("telegram_bot_stopped")
        
        await exchange_factory.close_all()

        if redis_client:
            await redis_client.close()
            logger.info("redis_connection_closed")

        logger.info("cleanup_completed_application_exiting")


if __name__ == "__main__":
    try: 
        asyncio.run(main())
    except KeyboardInterrupt: 
        logger.info("application_shutdown_via_keyboard_interrupt")
    except Exception as e:
        logger.critical("application_failed_to_run", error=str(e), exc_info=True)
