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
import signal
import contextlib
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime, timedelta, time as dtime, timezone, date
from collections import defaultdict
from functools import wraps
from scipy.stats import pearsonr
from dateutil import parser
from enum import Enum
from dataclasses import dataclass

# New ML imports
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

# Suppress HMM warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*converge.*")

# --- ### NEW: Imports for XAI and RL ### ---
try:
    import joblib
    import shap
    JOBLIB_SHAP_AVAILABLE = True
except ImportError:
    JOBLIB_SHAP_AVAILABLE = False

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
# --- ### END NEW ### ---

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
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.constants import ParseMode

# Env
from dotenv import load_dotenv

# Async DB & Caching
from sqlalchemy import (
    MetaData, Table, Column, Integer, Float, String, LargeBinary, Index,
    DateTime
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, AsyncSession
from sqlalchemy.sql import text as sql_text
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy.orm import sessionmaker

# Redis optional
try:
    import redis.asyncio as aioredis
except Exception:
    aioredis = None

# THREADING FIX: Import threading libraries and force matplotlib backend
import concurrent.futures
import matplotlib
matplotlib.use('Agg') # Force non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
MATPLOTLIB_AVAILABLE = True

import structlog

# Configure structlog (example)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# ------------------------------------------------------------------------------
# Network connection error handling
# ------------------------------------------------------------------------------
import socket
from urllib.error import URLError
from requests.exceptions import ConnectionError, Timeout, RequestException

load_dotenv()
SKIP_NETWORK_CHECK = os.environ.get("SKIP_NETWORK_CHECK", "false").lower() == "true"

def is_network_available() -> bool:
    """Fixed network check that works with your IPv6/HTTPS setup"""
    if SKIP_NETWORK_CHECK:
        print("⚠️  Network check bypassed via SKIP_NETWORK_CHECK=true")
        return True
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=10)
        if response.status_code == 200:
            print("✅ Binance API connectivity: OK")
            return True
    except Exception as e:
        print(f"❌ Binance API test failed: {str(e)}")
    try:
        response = requests.get("https://www.google.com", timeout=10)
        if response.status_code == 200:
            print("✅ Google HTTPS connectivity: OK")
            return True
    except Exception as e:
        print(f"❌ Google HTTPS test failed: {str(e)}")
    print("❌ All network tests failed")
    return False

async def wait_for_network():
    """Enhanced network waiting that won't get stuck"""
    if SKIP_NETWORK_CHECK:
        print("⚠️  Network check skipped via environment variable")
        return True
    max_attempts = 10
    max_wait_time = 180
    start_time = time.time()
    for attempt in range(1, max_attempts + 1):
        print(f"📡 Network Check Attempt {attempt}/{max_attempts}")
        if (time.time() - start_time) > max_wait_time:
            print(f"⏰ Maximum wait time ({max_wait_time}s) exceeded. Proceeding.")
            return True
        if is_network_available():
            print("✅ Network connectivity confirmed")
            return True
        if attempt < max_attempts:
            wait_time = min(20, 5 + attempt * 2)
            print(f"⏳ Waiting {wait_time} seconds before retry...")
            await asyncio.sleep(wait_time)
    print("⚠️ Network check failed after all attempts. Proceeding anyway.")
    return True

# ------------------------------------------------------------------------------
# GRACEFUL SHUTDOWN IMPLEMENTATION
# ------------------------------------------------------------------------------

SIGNAL_TRANSLATION_MAP = {
    signal.SIGINT: 'SIGINT',
    signal.SIGTERM: 'SIGTERM',
}

class DelayedKeyboardInterrupt:
    """
    Protects critical code sections from immediate interruption.
    Delays SIGINT and SIGTERM signals until critical section completes.
    """
    def __init__(self):
        self._sig = None
        self._frame = None
        self._old_signal_handler_map = None

    def __enter__(self):
        self._old_signal_handler_map = {
            sig: signal.signal(sig, self._handler)
            for sig in SIGNAL_TRANSLATION_MAP.keys()
        }

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original signal handlers
        for sig, handler in self._old_signal_handler_map.items():
            signal.signal(sig, handler)
        # If signal was received during critical section, handle it now
        if self._sig is not None:
            self._old_signal_handler_map[self._sig](self._sig, self._frame)

    def _handler(self, sig, frame):
        self._sig = sig
        self._frame = frame
        print(f'!!! {SIGNAL_TRANSLATION_MAP[sig]} received; delaying KeyboardInterrupt')

class GracefulShutdown:
    """
    Manages graceful shutdown process for the trading bot
    """
    def __init__(self):
        self.shutdown_requested = False
        self.shutdown_event = asyncio.Event()
        # Register signal handlers
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        """Signal handler for graceful shutdown"""
        print(f"🛑 Received signal {SIGNAL_TRANSLATION_MAP.get(signum, signum)}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.shutdown_event.set()

    def is_shutdown_requested(self):
        """Check if shutdown has been requested"""
        return self.shutdown_requested

# Global graceful shutdown manager
shutdown_manager = GracefulShutdown()

async def perform_cleanup():
    """
    Comprehensive cleanup function for trading bot resources
    """
    with DelayedKeyboardInterrupt():
        print("🛑 Starting comprehensive cleanup...")

        # Send shutdown notification to owner
        if application and config.owner_id_int:
            try:
                # Directly build and send the shutdown notification.
                shutdown_message = "<b>🛑 Bot Shutdown Initiated</b>\nGracefully closing all connections."
                await application.bot.send_message(
                    chat_id=config.owner_id_int,
                    text=shutdown_message,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )

            except Exception as e:
                logger.error("Failed to send shutdown notification", error=str(e))

        # Close exchange connections with critical section protection
        try:
            with DelayedKeyboardInterrupt():
                await exchange_factory.close_all()
                print("✅ Exchange connections closed")
        except Exception as e:
            logger.error("Error closing exchange connections", error=str(e))

        # Close Redis connection
        if redis_client:
            try:
                with DelayedKeyboardInterrupt():
                    await redis_client.close()
                    print("✅ Redis connection closed")
            except Exception as e:
                logger.error("Error closing Redis connection", error=str(e))

        # Close database engine
        try:
            with DelayedKeyboardInterrupt():
                await engine.dispose()
                print("✅ Database connections closed")
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))

        # Clean up matplotlib figures and thread pool
        try:
            plt.close('all')
            if plotter and hasattr(plotter, '_executor'):
                plotter._executor.shutdown(wait=False)
            print("✅ Matplotlib figures closed")
        except Exception as e:
            logger.error("Error closing matplotlib figures", error=str(e))

        # Force garbage collection
        gc.collect()
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✅ Graceful shutdown completed successfully")

# ------------------------------------------------------------------------------
# HEALTH MONITOR CLASS
# ------------------------------------------------------------------------------

class HealthMonitor:
    """Monitors system health and trading bot performance metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            'last_successful_scan': None,
            'last_successful_trade': None,
            'memory_usage': 0,
            'cpu_usage': 0,
            'errors_last_hour': 0,
            'scan_success_rate': 1.0
        }
        self.error_counts = []
        self.scan_results = []
        
    async def update_metrics(self):
        """Update system health metrics"""
        try:
            self.metrics['memory_usage'] = psutil.virtual_memory().percent
            self.metrics['cpu_usage'] = psutil.cpu_percent()
            
            # Cleanup old error counts
            now = datetime.now()
            self.error_counts = [ts for ts in self.error_counts 
                               if ts > now - timedelta(hours=1)]
            self.metrics['errors_last_hour'] = len(self.error_counts)
            
            # Calculate scan success rate
            if len(self.scan_results) > 0:
                success_count = sum(1 for x in self.scan_results if x)
                self.metrics['scan_success_rate'] = success_count / len(self.scan_results)
                
        except Exception as e:
            logger.error("health_metrics_update_failed", error=str(e))
    
    def record_scan_result(self, success: bool):
        """Record the result of a market scan"""
        self.scan_results.append(success)
        if len(self.scan_results) > 100:
            self.scan_results.pop(0)
        if success:
            self.metrics['last_successful_scan'] = datetime.now()
            
    def record_error(self):
        """Record an error occurrence"""
        self.error_counts.append(datetime.now())
        
    def record_successful_trade(self):
        """Record a successful trade"""
        self.metrics['last_successful_trade'] = datetime.now()
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        status = self.metrics.copy()
        status['status'] = 'HEALTHY'
        
        # Check for warning conditions
        if self.metrics['memory_usage'] > 85:
            status['status'] = 'WARNING'
            status['warning'] = 'High memory usage'
        elif self.metrics['errors_last_hour'] > 10:
            status['status'] = 'WARNING'
            status['warning'] = 'High error rate'
        elif self.metrics['scan_success_rate'] < 0.7:
            status['status'] = 'WARNING'
            status['warning'] = 'Low scan success rate'
            
        # Check for critical conditions
        if self.metrics['memory_usage'] > 95:
            status['status'] = 'CRITICAL'
            status['critical'] = 'Critical memory usage'
        elif self.metrics['errors_last_hour'] > 20:
            status['status'] = 'CRITICAL' 
            status['critical'] = 'Critical error rate'
            
        return status

    async def monitor_loop(self, recovery_callback: Optional[Callable] = None):
        """Continuous monitoring loop"""
        while True:
            try:
                await self.update_metrics()
                status = self.get_health_status()
                
                if status['status'] == 'CRITICAL' and recovery_callback:
                    logger.warning("health_critical_recovery_initiated", 
                                 reason=status.get('critical'))
                    await recovery_callback()
                    
                elif status['status'] == 'WARNING':
                    logger.warning("health_warning", reason=status.get('warning'))
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error("health_monitor_error", error=str(e))
                await asyncio.sleep(60)

# ------------------------------------------------------------------------------
# ENHANCED ML FEATURES CLASS (NEW IMPLEMENTATION)
# ------------------------------------------------------------------------------

class EnhancedMLFeatures:
    def __init__(self):
        self.feature_importance = {}
        self.feature_engineering = AdvancedFeatureEngineer()
        
        # Define coin categories for 1h timeframe
        self.major_coins = {'BTC/USDT', 'ETH/USDT', 'BNB/USDT'}
        
        self.tier1_altcoins = {
            'SOL/USDT', 'ADA/USDT', 'XRP/USDT', 'DOT/USDT', 
            'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT', 'UNI/USDT',
            'DOGE/USDT', 'LTC/USDT', 'ATOM/USDT', 'ETC/USDT'
        }
        
        # Sector classifications for crypto
        self.coin_sectors = {
            'DEFI': ['UNI', 'AAVE', 'SUSHI', 'CAKE', 'CRV', 'COMP', 'MKR'],
            'L1': ['SOL', 'ADA', 'AVAX', 'DOT', 'MATIC', 'NEAR', 'ALGO', 'FTM'],
            'GAMING': ['AXS', 'SAND', 'MANA', 'ENJ', 'GALA', 'ILV'],
            'ORACLE': ['LINK', 'BAND', 'TRB', 'API3'],
            'CEX': ['BNB', 'FTT', 'KCS', 'OKB', 'HT'],
            'MEME': ['DOGE', 'SHIB', 'PEPE', 'FLOKI'],
            'AI': ['AGIX', 'FET', 'OCEAN', 'NMR']
        }
        
        # 1h timeframe specific parameters
        self.hourly_volatility_multipliers = {
            'major': 0.8,      # Lower volatility adjustment for majors
            'tier1': 1.0,      # Standard for tier1
            'altcoin': 1.3     # Higher for altcoins due to more volatility
        }

    async def generate_features(self, df: pd.DataFrame, market_data: Dict) -> pd.DataFrame:
        """Generate comprehensive feature set with coin-specific adjustments for 1h timeframe"""
        
        # Get symbol from market data
        symbol = market_data.get('symbol', '')
        
        # Determine coin category
        category = self._get_coin_category(symbol)
        
        # Generate base features with category-specific parameters
        features = self.feature_engineering.generate_all_features(df)
        
        # Add enhanced regime features
        features = await self.add_enhanced_regime_features(features, category)
        
        # Add market microstructure features
        features = await self.add_advanced_microstructure_features(features, market_data, category)
        
        # Add cross-market correlations and sector metrics
        features = await self.add_enhanced_cross_market_features(features, symbol, category)
        
        # Add category-specific features
        features = self.add_category_specific_features(features, category)
        
        # Add 1h timeframe specific features
        features = self.add_hourly_timeframe_features(features)
        
        # Ensure no NaN values
        features = features.ffill().bfill().fillna(0)
        
        return features

    def _get_coin_category(self, symbol: str) -> str:
        """Determine coin category for feature adjustments"""
        if symbol in self.major_coins:
            return 'major'
        elif symbol in self.tier1_altcoins:
            return 'tier1'
        return 'altcoin'

    async def add_enhanced_regime_features(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Add regime features with category-specific thresholds for 1h timeframe"""
        df = df.copy()
        
        # Adjust volatility thresholds based on category for 1h
        vol_multiplier = self.hourly_volatility_multipliers.get(category, 1.0)
        
        # Enhanced volatility regime classification for 1h
        df['volatility_1h'] = df['close'].pct_change().rolling(20).std() * np.sqrt(24 * 365)  # Annualized
        df['relative_volatility'] = df['volatility_1h'] / df['volatility_1h'].rolling(100).mean()
        
        # Classify volatility regimes with dynamic thresholds for 1h
        df['volatility_regime'] = pd.cut(
            df['relative_volatility'],
            bins=[-np.inf, 0.6, 1.4 * vol_multiplier, np.inf],
            labels=['low', 'normal', 'high']
        )
        
        # Enhanced trend regime detection for 1h
        df['trend_strength'] = self._calculate_enhanced_trend_strength(df)
        df['trend_regime'] = pd.cut(
            df['trend_strength'],
            bins=[-np.inf, 0.25, 0.65, np.inf],
            labels=['ranging', 'weak_trend', 'strong_trend']
        )
        
        # Liquidity regime with volume analysis for 1h
        df['volume_sma_1h'] = df['volume'].rolling(24).mean()  # 24 hours
        df['volume_ratio'] = df['volume'] / df['volume_sma_1h']
        df['liquidity_regime'] = pd.cut(
            df['volume_ratio'],
            bins=[-np.inf, 0.6, 1.4, np.inf],
            labels=['low', 'normal', 'high']
        )
        
        # Market regime composite score
        df['regime_composite_score'] = (
            (df['trend_strength'] * 0.4) + 
            ((df['relative_volatility'] / 2) * 0.3) + 
            (np.minimum(df['volume_ratio'], 2) * 0.3)
        )
        
        return df

    async def add_advanced_microstructure_features(self, 
                                                 df: pd.DataFrame, 
                                                 market_data: Dict,
                                                 category: str) -> pd.DataFrame:
        """Add advanced microstructure features with category adjustments for 1h"""
        df = df.copy()
        
        # Get order book data from market_data
        order_book = market_data.get('orderbook', {})
        trades = market_data.get('trades', [])
        
        if order_book:
            # Category-specific depth analysis for 1h
            depth_factor = {
                'major': 1.0,
                'tier1': 0.7,
                'altcoin': 0.4
            }.get(category, 0.4)
            
            # Calculate order book imbalance at multiple levels
            for level in [5, 10, 15]:
                imbalance = self._calculate_ob_imbalance(order_book, level, depth_factor)
                df[f'ob_imbalance_{level}'] = imbalance
            
            # Market depth features
            df['market_depth_score'] = self._calculate_market_depth(order_book, depth_factor)
            df['spread_analysis'] = self._analyze_spreads(order_book)
            
            # Support/resistance levels from order book
            support_resistance = self._calculate_support_resistance_levels(order_book)
            df['support_strength'] = support_resistance.get('support_strength', 0)
            df['resistance_strength'] = support_resistance.get('resistance_strength', 0)
        
        if trades:
            # Advanced trade flow analysis for 1h
            df['smart_money_flow'] = self._calculate_smart_money_flow(trades)
            df['retail_flow'] = self._calculate_retail_flow(trades)
            df['large_trade_ratio'] = self._calculate_large_trade_ratio(trades)
            
        # Price impact features
        df['price_impact'] = self._calculate_price_impact(df)
        df['efficiency_ratio'] = self._calculate_efficiency_ratio(df)
        
        return df

    async def add_enhanced_cross_market_features(self, 
                                               df: pd.DataFrame, 
                                               symbol: str,
                                               category: str) -> pd.DataFrame:
        """Add enhanced cross-market features with sector analysis for 1h"""
        df = df.copy()
        
        # Get sector for the symbol
        symbol_sector = self._get_symbol_sector(symbol)
        
        try:
            # Correlation with major coins (adjusted by category for 1h)
            correlation_windows = {
                'major': 72,    # 3 days for majors
                'tier1': 48,    # 2 days for tier1  
                'altcoin': 24   # 1 day for altcoins
            }
            
            corr_window = correlation_windows.get(category, 48)
            
            for major in ['BTC/USDT', 'ETH/USDT']:
                if symbol != major:
                    major_df = await fetch_ohlcv_cached(major, '1h', corr_window + 10, 'binance')
                    if major_df is not None and len(major_df) >= corr_window:
                        corr = self._calculate_dynamic_correlation(df, major_df, corr_window)
                        df[f'{major.split("/")[0].lower()}_correlation'] = corr
            
            # Sector-specific metrics
            if symbol_sector:
                sector_data = await self._fetch_sector_data(symbol_sector)
                if sector_data is not None:
                    df['sector_strength'] = self._calculate_sector_strength(sector_data)
                    df['sector_momentum'] = self._calculate_sector_momentum(sector_data)
                    df['relative_sector_performance'] = self._calculate_relative_performance(df, sector_data)
            
            # Market breadth features for 1h
            df['market_breadth'] = await self._calculate_market_breadth()
            df['altcoin_dominance'] = await self._calculate_altcoin_dominance()
            
            # Category-specific momentum for 1h
            lookback_windows = {
                'major': [4, 12, 24],    # Shorter windows for majors (4h, 12h, 1d)
                'tier1': [8, 24, 48],    # Medium windows for tier1 (8h, 1d, 2d)
                'altcoin': [12, 48, 96]  # Longer windows for altcoins (12h, 2d, 4d)
            }.get(category, [8, 24, 48])
            
            for window in lookback_windows:
                df[f'momentum_{window}h'] = df['close'].pct_change(window)
                # Add momentum quality score
                df[f'momentum_quality_{window}h'] = self._calculate_momentum_quality(df, window)
            
            # Volatility clustering features
            df['volatility_clustering'] = self._calculate_volatility_clustering(df)
            df['volatility_regime_persistence'] = self._calculate_volatility_persistence(df)
            
        except Exception as e:
            logger.error(f"Error in cross-market features for {symbol}: {e}")
            # Fill with default values
            self._fill_missing_cross_market_features(df)
            
        return df

    def add_category_specific_features(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Add features specific to each coin category for 1h timeframe"""
        df = df.copy()
        
        if category == 'major':
            # Major coins - focus on institutional flows and macro impact
            df['institutional_flow'] = self._calculate_institutional_flow(df)
            df['macro_correlation_strength'] = self._calculate_macro_correlation(df)
            df['market_leadership'] = self._calculate_market_leadership(df)
            
        elif category == 'tier1':
            # Tier 1 altcoins - focus on ecosystem and adoption
            df['ecosystem_momentum'] = self._calculate_ecosystem_momentum(df)
            df['adoption_velocity'] = self._calculate_adoption_velocity(df)
            df['developer_activity_proxy'] = self._calculate_developer_activity_proxy(df)
            
        else:
            # Other altcoins - focus on momentum, sentiment, and risk
            df['sentiment_momentum'] = self._calculate_sentiment_momentum(df)
            df['risk_adjusted_momentum'] = self._calculate_risk_adjusted_momentum(df)
            df['liquidity_quality'] = self._calculate_liquidity_quality(df)
            
        return df

    def add_hourly_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features specific to 1-hour timeframe"""
        df = df.copy()
        
        # Intraday seasonality patterns
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Asian (0-8), European (8-16), US (16-24) session flags
        df['asian_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        df['european_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['us_session'] = ((df.index.hour >= 16) | (df.index.hour < 0)).astype(int)
        
        # Session strength indicators
        df['session_strength'] = self._calculate_session_strength(df)
        
        # Overnight gap analysis
        df['overnight_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_fill_probability'] = self._calculate_gap_fill_probability(df)
        
        # Hourly momentum cycles
        df['hourly_momentum_cycle'] = self._calculate_hourly_momentum_cycle(df)
        
        # Volume profile by hour
        df['volume_profile_hourly'] = self._calculate_volume_profile_by_hour(df)
        
        # NEW: Add adaptive RSI
        df['adaptive_rsi'] = self.calculate_adaptive_rsi(df['close'], df['volatility_1h'])
        
        return df

    def calculate_adaptive_rsi(self, prices: pd.Series, volatility_series: pd.Series) -> pd.Series:
        """Calculate RSI with adaptive periods based on volatility"""
        # Determine periods based on volatility
        avg_volatility = volatility_series.mean()
        rsi_periods = np.where(
            volatility_series > avg_volatility * 1.5, 
            10,  # Faster RSI during high volatility
            np.where(
                volatility_series < avg_volatility * 0.7,
                21,  # Slower RSI during low volatility
                14   # Standard period
            )
        )
        
        # Calculate RSI with standard period (adaptive calculation would be more complex)
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    # ===== IMPLEMENTATION OF HELPER METHODS =====

    def _calculate_enhanced_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate enhanced trend strength using multiple indicators for 1h"""
        # ADX calculation
        high, low, close = df['high'], df['low'], df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up = high - high.shift()
        down = low.shift() - low
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        # Smoothed values (14-period EMA)
        atr = tr.ewm(span=14).mean()
        plus_di = (pd.Series(plus_dm, index=df.index).ewm(span=14).mean() / atr) * 100
        minus_di = (pd.Series(minus_dm, index=df.index).ewm(span=14).mean() / atr) * 100
        
        # DX and ADX
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(span=14).mean()
        
        # Price movement consistency (Z-score of returns)
        returns = df['close'].pct_change()
        consistency = abs(returns.rolling(20).mean()) / (returns.rolling(20).std() + 1e-9)
        
        # EMA alignment strength
        ema_short = df['close'].ewm(span=20).mean()
        ema_medium = df['close'].ewm(span=50).mean()
        ema_alignment = np.where(ema_short > ema_medium, 1, -1) * abs(ema_short - ema_medium) / ema_medium
        
        # Final trend strength score (0-1)
        trend_strength = (
            0.4 * (adx / 100) + 
            0.3 * np.minimum(consistency, 2) / 2 +
            0.3 * np.minimum(abs(ema_alignment) * 100, 1)
        )
        
        return trend_strength.fillna(0)

    def _calculate_ob_imbalance(self, order_book: Dict, levels: int, depth_factor: float) -> float:
        """Calculate order book imbalance"""
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            return 0.0
            
        bids = order_book['bids'][:levels] if len(order_book['bids']) >= levels else order_book['bids']
        asks = order_book['asks'][:levels] if len(order_book['asks']) >= levels else order_book['asks']
        
        if not bids or not asks:
            return 0.0
            
        total_bid_volume = sum(float(bid[1]) for bid in bids)
        total_ask_volume = sum(float(ask[1]) for ask in asks)
        
        if total_bid_volume + total_ask_volume == 0:
            return 0.0
            
        imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        return imbalance * depth_factor

    def _calculate_market_depth(self, order_book: Dict, depth_factor: float) -> float:
        """Calculate market depth score"""
        if not order_book:
            return 0.0
            
        # Calculate depth at different price levels (0.5%, 1%, 2%)
        if not order_book['bids'] or not order_book['asks']:
            return 0.0
            
        mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
        
        depth_levels = [0.005, 0.01, 0.02]  # 0.5%, 1%, 2%
        depth_scores = []
        
        for level in depth_levels:
            price_range = mid_price * level
            
            # Bid depth
            bid_depth = 0
            for price, volume in order_book['bids']:
                if price >= mid_price - price_range:
                    bid_depth += volume * price
            
            # Ask depth  
            ask_depth = 0
            for price, volume in order_book['asks']:
                if price <= mid_price + price_range:
                    ask_depth += volume * price
            
            total_depth = bid_depth + ask_depth
            if total_depth > 0:
                depth_score = min(total_depth / 100000, 1.0)  # Normalize
                depth_scores.append(depth_score)
        
        return np.mean(depth_scores) if depth_scores else 0.0

    def _analyze_spreads(self, order_book: Dict) -> float:
        """Analyze spread quality"""
        if not order_book or not order_book['bids'] or not order_book['asks']:
            return 0.0
            
        best_bid = order_book['bids'][0][0]
        best_ask = order_book['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2
        
        spread = (best_ask - best_bid) / mid_price
        
        # Convert to quality score (lower spread = higher quality)
        spread_quality = 1.0 / (spread * 1000 + 1e-9)
        return min(spread_quality, 10.0)  # Cap at 10

    def _calculate_support_resistance_levels(self, order_book: Dict) -> Dict[str, float]:
        """Calculate support and resistance strength from order book"""
        if not order_book:
            return {'support_strength': 0, 'resistance_strength': 0}
            
        # Analyze top 10 levels for support/resistance
        bid_levels = order_book['bids'][:10] if len(order_book['bids']) >= 10 else order_book['bids']
        ask_levels = order_book['asks'][:10] if len(order_book['asks']) >= 10 else order_book['asks']
        
        support_strength = sum(float(bid[1]) for bid in bid_levels) if bid_levels else 0
        resistance_strength = sum(float(ask[1]) for ask in ask_levels) if ask_levels else 0
        
        # Normalize
        max_strength = max(support_strength, resistance_strength, 1e-9)
        
        return {
            'support_strength': support_strength / max_strength,
            'resistance_strength': resistance_strength / max_strength
        }

    def _calculate_smart_money_flow(self, trades: List[Dict]) -> float:
        """Calculate smart money flow from large trades"""
        if not trades:
            return 0.0
            
        try:
            df_trades = pd.DataFrame(trades)
            if 'amount' not in df_trades.columns or 'side' not in df_trades.columns:
                return 0.0
                
            # Define large trades as top 20% by amount
            large_trade_threshold = df_trades['amount'].quantile(0.8)
            large_trades = df_trades[df_trades['amount'] >= large_trade_threshold]
            
            if large_trades.empty:
                return 0.0
                
            buy_volume = large_trades[large_trades['side'] == 'buy']['amount'].sum()
            sell_volume = large_trades[large_trades['side'] == 'sell']['amount'].sum()
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0.0
                
            return (buy_volume - sell_volume) / total_volume
            
        except Exception:
            return 0.0

    def _calculate_retail_flow(self, trades: List[Dict]) -> float:
        """Calculate retail flow from small trades"""
        if not trades:
            return 0.0
            
        try:
            df_trades = pd.DataFrame(trades)
            if 'amount' not in df_trades.columns or 'side' not in df_trades.columns:
                return 0.0
                
            # Define small trades as bottom 50% by amount
            small_trade_threshold = df_trades['amount'].quantile(0.5)
            small_trades = df_trades[df_trades['amount'] <= small_trade_threshold]
            
            if small_trades.empty:
                return 0.0
                
            buy_volume = small_trades[small_trades['side'] == 'buy']['amount'].sum()
            sell_volume = small_trades[small_trades['side'] == 'sell']['amount'].sum()
            total_volume = buy_volume + sell_volume
            
            if total_volume == 0:
                return 0.0
                
            return (buy_volume - sell_volume) / total_volume
            
        except Exception:
            return 0.0

    def _calculate_large_trade_ratio(self, trades: List[Dict]) -> float:
        """Calculate ratio of large trades to total volume"""
        if not trades:
            return 0.0
            
        try:
            df_trades = pd.DataFrame(trades)
            if 'amount' not in df_trades.columns:
                return 0.0
                
            large_trade_threshold = df_trades['amount'].quantile(0.8)
            large_trades = df_trades[df_trades['amount'] >= large_trade_threshold]
            
            total_volume = df_trades['amount'].sum()
            large_volume = large_trades['amount'].sum()
            
            if total_volume == 0:
                return 0.0
                
            return large_volume / total_volume
            
        except Exception:
            return 0.0

    def _calculate_price_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate price impact of volume"""
        # Price change relative to volume
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        
        # Rolling correlation between absolute price change and volume
        impact = price_change.abs().rolling(20).corr(volume_change)
        return impact.fillna(0)

    def _calculate_efficiency_ratio(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market efficiency ratio"""
        # Absolute price change over period
        net_change = abs(df['close'] - df['close'].shift(20))
        total_change = abs(df['close'] - df['close'].shift(1)).rolling(20).sum()
        
        efficiency = net_change / (total_change + 1e-9)
        return efficiency.fillna(0)

    def _get_symbol_sector(self, symbol: str) -> Optional[str]:
        """Get the sector for a given symbol"""
        if not symbol or '/' not in symbol:
            return None
            
        symbol_base = symbol.split('/')[0]
        
        for sector, coins in self.coin_sectors.items():
            if symbol_base in coins:
                return sector
                
        return None

    async def _fetch_sector_data(self, sector: str) -> Optional[pd.DataFrame]:
        """Fetch aggregate sector data"""
        try:
            # This would typically fetch from an API or database
            # For now, return None - implement based on your data sources
            return None
        except Exception:
            return None

    def _calculate_sector_strength(self, sector_data: pd.DataFrame) -> pd.Series:
        """Calculate sector strength indicator"""
        # Placeholder implementation
        return pd.Series(0.5, index=range(len(sector_data))) if sector_data is not None else pd.Series(0.0)

    def _calculate_sector_momentum(self, sector_data: pd.DataFrame) -> pd.Series:
        """Calculate sector momentum"""
        # Placeholder implementation  
        return pd.Series(0.5, index=range(len(sector_data))) if sector_data is not None else pd.Series(0.0)

    def _calculate_relative_performance(self, df: pd.DataFrame, sector_data: pd.DataFrame) -> pd.Series:
        """Calculate relative performance to sector"""
        # Placeholder implementation
        return pd.Series(0.0, index=df.index)

    def _calculate_dynamic_correlation(self, df1: pd.DataFrame, df2: pd.DataFrame, window: int) -> pd.Series:
        """Calculate dynamic correlation between two assets"""
        if len(df1) < window or len(df2) < window:
            return pd.Series(0.0, index=df1.index)
            
        # Align the two dataframes
        common_index = df1.index.intersection(df2.index)
        if len(common_index) < window:
            return pd.Series(0.0, index=df1.index)
            
        returns1 = df1.loc[common_index, 'close'].pct_change()
        returns2 = df2.loc[common_index, 'close'].pct_change()
        
        # Rolling correlation
        correlation = returns1.rolling(window).corr(returns2)
        
        # Reindex to original index
        correlation = correlation.reindex(df1.index).fillna(0)
        return correlation

    async def _calculate_market_breadth(self) -> float:
        """Calculate market breadth (placeholder)"""
        # Implement based on your market data
        return 0.5

    async def _calculate_altcoin_dominance(self) -> float:
        """Calculate altcoin dominance (placeholder)"""
        # Implement based on your market data  
        return 0.3

    def _calculate_momentum_quality(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate momentum quality score"""
        returns = df['close'].pct_change(window)
        volatility = df['close'].pct_change().rolling(window).std()
        
        # Sharpe-like ratio for momentum quality
        quality = returns / (volatility + 1e-9)
        return quality.fillna(0)

    def _calculate_volatility_clustering(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility clustering indicator"""
        returns = df['close'].pct_change()
        vol_clustering = returns.rolling(20).std().rolling(20).std()
        return vol_clustering.fillna(0)

    def _calculate_volatility_persistence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility regime persistence"""
        volatility = df['close'].pct_change().rolling(20).std()
        vol_persistence = volatility.rolling(50).apply(lambda x: x.autocorr(), raw=False)
        return vol_persistence.fillna(0)

    def _fill_missing_cross_market_features(self, df: pd.DataFrame):
        """Fill missing cross-market features with default values"""
        cross_market_features = [
            'btc_correlation', 'eth_correlation', 'sector_strength',
            'sector_momentum', 'relative_sector_performance', 'market_breadth',
            'altcoin_dominance'
        ]
        
        for feature in cross_market_features:
            if feature not in df.columns:
                df[feature] = 0.0

    def _calculate_institutional_flow(self, df: pd.DataFrame) -> pd.Series:
        """Calculate institutional flow proxy"""
        # Use large volume candles with small price movement as institutional accumulation
        volume_z = (df['volume'] - df['volume'].rolling(50).mean()) / df['volume'].rolling(50).std()
        price_change = df['close'].pct_change().abs()
        
        institutional_flow = volume_z * (1 - price_change * 10)
        return institutional_flow.fillna(0)

    def _calculate_macro_correlation(self, df: pd.DataFrame) -> pd.Series:
        """Calculate macro correlation strength (placeholder)"""
        return pd.Series(0.5, index=df.index)

    def _calculate_market_leadership(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market leadership score (placeholder)"""
        return pd.Series(0.5, index=df.index)

    def _calculate_ecosystem_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ecosystem momentum (placeholder)"""
        return pd.Series(0.5, index=df.index)

    def _calculate_adoption_velocity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate adoption velocity (placeholder)"""
        return pd.Series(0.5, index=df.index)

    def _calculate_developer_activity_proxy(self, df: pd.DataFrame) -> pd.Series:
        """Calculate developer activity proxy (placeholder)"""
        return pd.Series(0.5, index=df.index)

    def _calculate_sentiment_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate sentiment momentum proxy"""
        # Use volume and price relationship as sentiment proxy
        price_momentum = df['close'].pct_change(5)
        volume_momentum = df['volume'].pct_change(5)
        
        sentiment = price_momentum * volume_momentum
        return sentiment.fillna(0)

    def _calculate_risk_adjusted_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Calculate risk-adjusted momentum"""
        returns = df['close'].pct_change(5)
        volatility = df['close'].pct_change().rolling(20).std()
        
        risk_adjusted = returns / (volatility + 1e-9)
        return risk_adjusted.fillna(0)

    def _calculate_liquidity_quality(self, df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity quality score"""
        volume_stability = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
        liquidity_quality = 1 / (volume_stability + 1e-9)
        return liquidity_quality.fillna(0)

    def _calculate_session_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trading session strength"""
        # Compare current volume to session average
        session_volume_avg = df.groupby(df.index.hour)['volume'].transform('mean')
        session_strength = df['volume'] / (session_volume_avg + 1e-9)
        return session_strength.fillna(1)

    def _calculate_gap_fill_probability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate gap fill probability"""
        gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Simple heuristic: larger gaps have lower fill probability
        fill_probability = 1 - abs(gap) * 10
        return fill_probability.clip(0, 1).fillna(0.5)

    def _calculate_hourly_momentum_cycle(self, df: pd.DataFrame) -> pd.Series:
        """Calculate hourly momentum cycle"""
        # Average returns by hour of day
        hour_returns = df.groupby(df.index.hour)['close'].pct_change().mean()
        current_hour_avg = df.index.hour.map(hour_returns.to_dict())
        return pd.Series(current_hour_avg, index=df.index).fillna(0)

    def _calculate_volume_profile_by_hour(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile by hour"""
        hour_volume_avg = df.groupby(df.index.hour)['volume'].mean()
        current_hour_avg = df.index.hour.map(hour_volume_avg.to_dict())
        current_volume_ratio = df['volume'] / (current_hour_avg + 1e-9)
        return current_volume_ratio.fillna(1)

# ------------------------------------------------------------------------------
# ADVANCED FEATURE ENGINEERING
# ------------------------------------------------------------------------------

class AdvancedFeatureEngineer:
    """Enhanced feature engineering with ML-ready features"""

    def __init__(self):
        self.feature_cache = {}

    def compute_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-horizon momentum and acceleration features"""
        df = df.copy()

        # Multi-horizon momentum (3, 5, 10, 20 periods)
        for k in [3, 5, 10, 20]:
            df[f'momentum_{k}'] = df['close'].pct_change(k)
            df[f'acceleration_{k}'] = df[f'momentum_{k}'] - df[f'momentum_{k}'].shift(k)

        # Fractional differencing for stationarity
        df['frac_diff_0_7'] = self.fractional_difference(df['close'], 0.7)

        # Z-score normalized momentum
        for k in [5, 10, 20]:
            df[f'momentum_z_{k}'] = (
                df[f'momentum_{k}'] - df[f'momentum_{k}'].rolling(50).mean()
            ) / (df[f'momentum_{k}'].rolling(50).std() + 1e-9)

        return df

    def fractional_difference(self, series: pd.Series, d: float) -> pd.Series:
        """Fractional differencing for stationarity preservation"""
        from scipy.special import binom
        n = len(series)
        diff_series = np.zeros(n)

        for t in range(n):
            weighted_sum = 0
            for k in range(t + 1):
                weight = (-1) ** k * binom(d, k)
                if t - k >= 0:
                    weighted_sum += weight * series.iloc[t - k]
            diff_series[t] = weighted_sum

        return pd.Series(diff_series, index=series.index)

    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced volatility clustering features"""
        df = df.copy()
        returns = df['close'].pct_change().fillna(0)

        # EWMA volatility (multiple spans)
        for span in [10, 20, 50]:
            df[f'ewma_vol_{span}'] = returns.pow(2).ewm(span=span).mean().pow(0.5)

        # GARCH-like features (simplified)
        df['volatility_regime'] = self.compute_volatility_regime(returns)

        # Realized volatility (rolling)
        df['realized_vol_20'] = returns.rolling(20).std()
        df['vol_of_vol'] = df['realized_vol_20'].rolling(50).std()

        # Volatility ratio (short-term vs long-term)
        df['vol_ratio'] = df['ewma_vol_10'] / (df['ewma_vol_50'] + 1e-9)

        return df

    def compute_volatility_regime(self, returns: pd.Series) -> pd.Series:
        """HMM-based volatility regime detection"""
        try:
            # Simple 2-state HMM for high/low volatility
            returns_2d = returns.values.reshape(-1, 1)
            model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
            model.fit(returns_2d[:1000])  # Fit on first 1000 points

            # Predict regimes
            if len(returns) > 1000:
                states = model.predict(returns_2d[1000:])
                # Pad with first state for initial period
                states_full = np.concatenate([np.zeros(1000), states])
            else:
                states_full = model.predict(returns_2d)

            return pd.Series(states_full, index=returns.index)
        except Exception:
            # Fallback: simple standard deviation based regime
            vol = returns.rolling(20).std()
            return (vol > vol.rolling(100).median()).astype(int)

    def compute_market_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hurst exponent and market efficiency features"""
        df = df.copy()

        # Calculate Hurst exponent on rolling window
        hurst_values = []
        prices = df['close'].values

        for i in range(len(prices)):
            if i < 100:  # Minimum window size
                hurst_values.append(0.5)
                continue

            window = prices[max(0, i-100):i+1]
            try:
                hurst = self.hurst_exponent_rs(window)
                hurst_values.append(hurst)
            except:
                hurst_values.append(0.5)

        df['hurst_exponent'] = hurst_values

        # Market regime based on Hurst
        df['market_regime_hurst'] = np.select(
            [
                df['hurst_exponent'] > 0.6,
                df['hurst_exponent'] < 0.4
            ],
            ['trending', 'mean_reverting'],
            default='efficient'
        )

        return df

    def hurst_exponent_rs(self, series: np.ndarray) -> float:
        """Rescaled Range Hurst exponent calculation"""
        if len(series) < 16:
            return 0.5

        min_window = 16
        max_window = min(256, len(series) // 4)

        if max_window <= min_window:
            return 0.5

        windows = np.logspace(np.log10(min_window), np.log10(max_window), num=8, dtype=int)
        windows = np.unique(windows)

        rs_values = []

        for w in windows:
            if w > len(series):
                continue

            chunks = len(series) // w
            if chunks < 2:
                continue

            rs_chunks = []
            for i in range(chunks):
                chunk = series[i*w:(i+1)*w]
                if len(chunk) < 2:
                    continue

                # Calculate R/S
                mean_chunk = np.mean(chunk)
                deviations = chunk - mean_chunk
                z = np.cumsum(deviations)
                r = np.max(z) - np.min(z)
                s = np.std(chunk)

                if s > 0:
                    rs_chunks.append(r / s)

            if rs_chunks:
                rs_values.append(np.log(np.mean(rs_chunks)))
            else:
                rs_values.append(np.nan)

        valid_rs = [x for x in rs_values if not np.isnan(x)]
        valid_windows = windows[:len(valid_rs)]

        if len(valid_rs) < 2:
            return 0.5

        # Linear regression to get Hurst
        coeffs = np.polyfit(np.log(valid_windows), valid_rs, 1)
        return coeffs[0]

    def compute_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seasonal and temporal pattern features"""
        df = df.copy()
        df_index = pd.to_datetime(df.index)

        # Intraday patterns
        df['minute_of_day'] = df_index.hour * 60 + df_index.minute
        df['hour_of_day'] = df_index.hour
        df['day_of_week'] = df_index.dayofweek
        df['week_of_month'] = (df_index.day - 1) // 7
        df['month'] = df_index.month

        # Market session flags
        df['asian_session'] = ((df_index.hour >= 0) & (df_index.hour < 8)).astype(int)
        df['european_session'] = ((df_index.hour >= 8) & (df_index.hour < 16)).astype(int)
        df['us_session'] = ((df_index.hour >= 16) | (df_index.hour < 0)).astype(int)

        # Time since major market opens
        df['minutes_since_market_open'] = (df_index.hour - 9) * 60 + df_index.minute
        df['minutes_since_market_open'] = df['minutes_since_market_open'].clip(0, 60*6.5)

        return df

    def compute_technical_confluence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Composite technical signal features"""
        df = df.copy()

        # Signal confluence count
        bull_signals = []
        bear_signals = []

        # EMA signals
        if 'ema20' in df.columns and 'ema50' in df.columns:
            bull_signals.append(df['ema20'] > df['ema50'])
            bear_signals.append(df['ema20'] < df['ema50'])

        # RSI signals
        if 'rsi' in df.columns:
            bull_signals.append(df['rsi'] > 50)
            bear_signals.append(df['rsi'] < 50)

        # MACD signals
        if 'macd' in df.columns and 'macd_sig' in df.columns:
            bull_signals.append(df['macd'] > df['macd_sig'])
            bear_signals.append(df['macd'] < df['macd_sig'])

        # Enhanced indicator signals
        enhanced_bull = ['ichimoku_bull_signal', 'stoch_rsi_bull', 'cci_bull_signal', 'williams_r_bull']
        enhanced_bear = ['ichimoku_bear_signal', 'stoch_rsi_bear', 'cci_bear_signal', 'williams_r_bear']

        for col in enhanced_bull:
            if col in df.columns:
                bull_signals.append(df[col] > 0)

        for col in enhanced_bear:
            if col in df.columns:
                bear_signals.append(df[col] > 0)

        # Confluence scores
        if bull_signals:
            df['bull_confluence'] = sum(bull_signals) / len(bull_signals)
        if bear_signals:
            df['bear_confluence'] = sum(bear_signals) / len(bear_signals)

        # Distance to signal thresholds
        if 'rsi' in df.columns:
            df['rsi_distance_50'] = (df['rsi'] - 50) / 50
            df['rsi_distance_oversold'] = (30 - df['rsi']) / 30
            df['rsi_distance_overbought'] = (df['rsi'] - 70) / 30

        # Recent signal performance (simulated - would need historical data)
        df['recent_signal_strength'] = df['bull_confluence'].rolling(10).mean()

        return df

    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate complete feature set"""
        df = self.compute_momentum_features(df)
        df = self.compute_volatility_features(df)
        df = self.compute_market_efficiency_features(df)
        df = self.compute_temporal_features(df)
        df = self.compute_technical_confluence(df)

        # Fill any remaining NaN values
        df = df.ffill().bfill()

        return df

# ------------------------------------------------------------------------------
# ENHANCED MARKET REGIME CLASSIFICATION
# ------------------------------------------------------------------------------

class MarketRegime(Enum):
    TRENDING_STABLE = "TRENDING_STABLE"
    TRENDING_VOLATILE = "TRENDING_VOLATILE"
    CHOPPY_VOLATILE = "CHOPPY_VOLATILE"
    RANGE_BOUND = "RANGE_BOUND"
    NEUTRAL = "NEUTRAL"

@dataclass
class RegimeMetrics:
    volatility: float
    trend_strength: float
    volume_profile: float
    regime: MarketRegime
    confidence: float

class EnhancedMarketRegimeClassifier:
    """Enhanced market regime classification that works alongside existing Hurst analysis"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'lookback_window': 100,
            'volatility_threshold': 1.5,
            'trend_strength_threshold': 0.6,
            'range_bound_threshold': 0.3,
            'volume_threshold': 1.2
        }
        
    def classify_regime(self, df: pd.DataFrame) -> RegimeMetrics:
        """Classifies market regime using multiple indicators alongside existing analysis"""
        # Calculate base metrics
        volatility = self._calculate_volatility(df)
        trend_strength = self._calculate_trend_strength(df)
        volume_profile = self._analyze_volume_profile(df)
        
        # Determine regime
        regime, confidence = self._determine_regime(
            volatility, trend_strength, volume_profile
        )
        
        return RegimeMetrics(
            volatility=volatility,
            trend_strength=trend_strength,
            volume_profile=volume_profile,
            regime=regime,
            confidence=confidence
        )
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculates normalized volatility using ATR alongside existing volatility features"""
        returns = df['close'].pct_change()
        current_vol = returns.rolling(20).std() * np.sqrt(252)
        historical_vol = returns.rolling(100).std() * np.sqrt(252)
        
        return (current_vol.iloc[-1] / historical_vol.iloc[-1]) if historical_vol.iloc[-1] != 0 else 1.0
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculates trend strength using ADX and price movement consistency"""
        # ADX calculation
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(14).mean()
        
        # Price movement consistency
        returns = df['close'].pct_change()
        consistency = abs(returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-9))
        
        # Combine metrics
        adx_score = min(adx.iloc[-1] / 100, 1.0)
        consistency_score = min(consistency.iloc[-1], 1.0)
        
        return 0.7 * adx_score + 0.3 * consistency_score
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> float:
        """Analyzes volume profile to detect accumulation/distribution"""
        volume = df['volume']
        close = df['close']
        
        # Volume trend
        vol_sma = volume.rolling(20).mean()
        vol_trend = vol_sma.iloc[-1] / vol_sma.iloc[-20] if len(vol_sma) >= 20 else 1.0
        
        # Price-volume correlation
        returns = close.pct_change()
        volume_returns = volume.pct_change()
        corr = returns.corr(volume_returns)
        
        # Normalize and combine
        vol_score = min(vol_trend / self.config['volume_threshold'], 1.0)
        corr_score = (corr + 1) / 2  # Normalize correlation to 0-1
        
        return 0.6 * vol_score + 0.4 * corr_score
    
    def _determine_regime(self, volatility: float, trend_strength: float, 
                         volume_profile: float) -> Tuple[MarketRegime, float]:
        """Determines market regime based on calculated metrics"""
        # Base confidence calculation
        confidence = (trend_strength + volume_profile) / 2
        
        # Regime classification with enhanced logic
        if trend_strength > self.config['trend_strength_threshold']:
            if volatility > self.config['volatility_threshold']:
                return MarketRegime.TRENDING_VOLATILE, confidence
            return MarketRegime.TRENDING_STABLE, confidence
            
        if volatility > self.config['volatility_threshold']:
            return MarketRegime.CHOPPY_VOLATILE, confidence * 0.8
            
        if volume_profile < self.config['range_bound_threshold']:
            return MarketRegime.RANGE_BOUND, confidence * 0.9
            
        return MarketRegime.NEUTRAL, confidence * 0.7

# ------------------------------------------------------------------------------
# ENHANCED DYNAMIC RISK MANAGEMENT
# ------------------------------------------------------------------------------

@dataclass
class PositionSizeResult:
    size_usd: float
    size_coins: float
    leverage: int
    risk_score: float
    max_drawdown: float

class EnhancedDynamicRiskManager:
    """Enhanced risk management that works with your existing risk managers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.portfolio_value = float(config.get('portfolio_value', 10000))
        self.max_position_size = float(config.get('max_position_size', 0.1))
        self.max_leverage = int(config.get('max_leverage', 20))
        self.base_risk_per_trade = float(config.get('base_risk_per_trade', 0.01))
        
        # Risk scaling factors
        self.regime_risk_factors = {
            MarketRegime.TRENDING_STABLE: 1.2,
            MarketRegime.TRENDING_VOLATILE: 0.8,
            MarketRegime.CHOPPY_VOLATILE: 0.6,
            MarketRegime.RANGE_BOUND: 0.9,
            MarketRegime.NEUTRAL: 1.0
        }
        
    async def calculate_position_size(self, 
                                    entry_price: float,
                                    stop_loss: float,
                                    regime: MarketRegime,
                                    volatility: float,
                                    confidence: float,
                                    current_exposure: float) -> PositionSizeResult:
        """Calculates optimal position size considering multiple factors"""
        # Calculate base risk amount
        risk_amount = self.portfolio_value * self.base_risk_per_trade
        
        # Apply regime-based scaling
        regime_factor = self.regime_risk_factors.get(regime, 1.0)
        risk_amount *= regime_factor
        
        # Apply volatility scaling
        vol_scale = self._calculate_volatility_scalar(volatility)
        risk_amount *= vol_scale
        
        # Apply confidence scaling
        conf_scale = self._calculate_confidence_scalar(confidence)
        risk_amount *= conf_scale
        
        # Calculate position size based on stop loss distance
        sl_dist_pct = abs(entry_price - stop_loss) / entry_price
        if sl_dist_pct == 0:
            return self._create_zero_position()
            
        # Calculate initial position size
        position_size_usd = risk_amount / sl_dist_pct
        
        # Apply portfolio heat limits
        adjusted_size = self._apply_portfolio_limits(
            position_size_usd, current_exposure
        )
        
        # Calculate optimal leverage
        leverage = self._calculate_optimal_leverage(sl_dist_pct)
        
        # Calculate final position sizes
        position_coins = adjusted_size / entry_price
        
        return PositionSizeResult(
            size_usd=adjusted_size,
            size_coins=position_coins,
            leverage=leverage,
            risk_score=self._calculate_risk_score(
                regime, volatility, confidence, current_exposure
            ),
            max_drawdown=self._estimate_max_drawdown(
                adjusted_size, sl_dist_pct, leverage
            )
        )
    
    def _calculate_volatility_scalar(self, volatility: float) -> float:
        """Adjusts position size based on volatility"""
        target_vol = 0.20  # 20% annualized vol target
        vol_scale = target_vol / (volatility + 1e-9)
        return np.clip(vol_scale, 0.5, 2.0)
    
    def _calculate_confidence_scalar(self, confidence: float) -> float:
        """Adjusts position size based on signal confidence"""
        # Non-linear scaling that rewards high confidence
        conf_scale = np.power(confidence, 1.5)
        return np.clip(conf_scale, 0.5, 1.5)
    
    def _apply_portfolio_limits(self, size_usd: float, 
                              current_exposure: float) -> float:
        """Applies portfolio-level position limits"""
        # Maximum single position size
        max_position = self.portfolio_value * self.max_position_size
        
        # Available risk capacity
        available_risk = self.portfolio_value - current_exposure
        
        # Apply limits
        adjusted_size = min(
            size_usd,
            max_position,
            available_risk * 0.5  # Don't use more than 50% of available risk
        )
        
        return max(adjusted_size, 0.0)
    
    def _calculate_optimal_leverage(self, sl_dist_pct: float) -> int:
        """Calculates optimal leverage based on stop distance"""
        # Base leverage calculation
        base_leverage = int(1 / (sl_dist_pct * 1.5))  # 1.5x safety factor
        
        # Apply limits
        return min(base_leverage, self.max_leverage)
    
    def _calculate_risk_score(self, regime: MarketRegime, 
                            volatility: float, confidence: float,
                            current_exposure: float) -> float:
        """Calculates overall risk score for the position"""
        # Component scores
        regime_score = 1.0 - (self.regime_risk_factors[regime] / 1.2)
        vol_score = min(volatility / 0.4, 1.0)  # Cap at 40% annualized vol
        exposure_score = current_exposure / self.portfolio_value
        
        # Weighted combination
        risk_score = (
            0.4 * regime_score +
            0.3 * vol_score +
            0.2 * (1.0 - confidence) +  # Lower confidence = higher risk
            0.1 * exposure_score
        )
        
        return np.clip(risk_score, 0.0, 1.0)
    
    def _estimate_max_drawdown(self, position_size: float,
                             sl_dist_pct: float, leverage: int) -> float:
        """Estimates maximum potential drawdown for the position"""
        max_loss = position_size * sl_dist_pct * leverage
        return max_loss / self.portfolio_value  # As percentage of portfolio
    
    def _create_zero_position(self) -> PositionSizeResult:
        """Creates a zero position result for invalid scenarios"""
        return PositionSizeResult(
            size_usd=0.0,
            size_coins=0.0,
            leverage=1,
            risk_score=0.0,
            max_drawdown=0.0
        )

# ------------------------------------------------------------------------------
# SMART ORDER EXECUTION
# ------------------------------------------------------------------------------

@dataclass
class OrderBookAnalysis:
    bid_ask_imbalance: float
    depth_score: float
    spread: float
    volatility: float
    suggested_slip: float

@dataclass
class ExecutionPlan:
    base_price: float
    suggested_price: float
    order_type: str
    time_in_force: str
    post_only: bool
    execution_style: str
    estimated_fill_time: float

class EnhancedSmartOrderExecutor:
    """Smart order execution that integrates with your existing microstructure analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.slippage_buffer = float(config.get('slippage_buffer', 0.001))
        self.max_spread = float(config.get('max_spread', 0.003))
        self.min_depth = float(config.get('min_depth', 10000))
        self.execution_patience = int(config.get('execution_patience', 60))
        
    async def optimize_entry(self, 
                           signal: Dict, 
                           order_book: Dict,
                           recent_trades: List[Dict]) -> ExecutionPlan:
        """Creates optimized execution plan based on market microstructure"""
        # Analyze order book
        ob_analysis = self._analyze_order_book(order_book, signal['position_size_coin'])
        
        # Analyze recent trades
        trade_analysis = self._analyze_recent_trades(recent_trades)
        
        # Determine execution strategy
        if ob_analysis.spread > self.max_spread:
            return self._create_passive_entry(signal, ob_analysis)
        
        if ob_analysis.depth_score < 0.5:
            return self._create_scaled_entry(signal, ob_analysis)
            
        if ob_analysis.volatility > 0.002:  # 0.2% per minute
            return self._create_twap_entry(signal, ob_analysis)
            
        return self._create_immediate_entry(signal, ob_analysis)
    
    def _analyze_order_book(self, order_book: Dict, 
                           size: float) -> OrderBookAnalysis:
        """Analyzes order book for optimal execution"""
        bids = order_book['bids']
        asks = order_book['asks']
        
        # Calculate basic metrics
        spread = (asks[0][0] - bids[0][0]) / bids[0][0]
        mid_price = (asks[0][0] + bids[0][0]) / 2
        
        # Calculate depth up to size
        bid_depth = self._calculate_cumulative_depth(bids, size)
        ask_depth = self._calculate_cumulative_depth(asks, size)
        
        # Calculate imbalance
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
        
        # Calculate depth score
        depth_score = min(total_depth / self.min_depth, 1.0)
        
        # Estimate volatility from book updates (simplified)
        volatility = spread * 2  # Simple approximation
        
        # Estimate required slippage
        suggested_slip = self._estimate_required_slippage(
            size, depth_score, volatility
        )
        
        return OrderBookAnalysis(
            bid_ask_imbalance=imbalance,
            depth_score=depth_score,
            spread=spread,
            volatility=volatility,
            suggested_slip=suggested_slip
        )
    
    def _analyze_recent_trades(self, trades: List[Dict]) -> Dict:
        """Analyzes recent trades for execution insights"""
        if not trades:
            return {'avg_size': 0, 'avg_impact': 0, 'volatility': 0}
            
        df = pd.DataFrame(trades)
        
        return {
            'avg_size': df['size'].mean(),
            'avg_impact': df['price'].pct_change().abs().mean(),
            'volatility': df['price'].pct_change().std()
        }
    
    def _create_immediate_entry(self, signal: Dict, 
                              analysis: OrderBookAnalysis) -> ExecutionPlan:
        """Creates a plan for immediate execution"""
        base_price = signal['entry']
        suggested_price = base_price * (1 + analysis.suggested_slip)
        
        return ExecutionPlan(
            base_price=base_price,
            suggested_price=suggested_price,
            order_type='LIMIT',
            time_in_force='GTC',
            post_only=False,
            execution_style='IMMEDIATE',
            estimated_fill_time=1.0
        )
    
    def _create_passive_entry(self, signal: Dict, 
                            analysis: OrderBookAnalysis) -> ExecutionPlan:
        """Creates a plan for passive execution"""
        base_price = signal['entry']
        suggested_price = base_price * (1 - self.slippage_buffer)
        
        return ExecutionPlan(
            base_price=base_price,
            suggested_price=suggested_price,
            order_type='LIMIT',
            time_in_force='GTC',
            post_only=True,
            execution_style='PASSIVE',
            estimated_fill_time=self.execution_patience
        )
    
    def _create_scaled_entry(self, signal: Dict, 
                           analysis: OrderBookAnalysis) -> ExecutionPlan:
        """Creates a plan for scaled execution"""
        base_price = signal['entry']
        suggested_price = base_price * (1 + analysis.suggested_slip * 0.5)
        
        return ExecutionPlan(
            base_price=base_price,
            suggested_price=suggested_price,
            order_type='LIMIT',
            time_in_force='GTC',
            post_only=True,
            execution_style='SCALED',
            estimated_fill_time=self.execution_patience * 0.5
        )
    
    def _create_twap_entry(self, signal: Dict, 
                          analysis: OrderBookAnalysis) -> ExecutionPlan:
        """Creates a plan for time-weighted average price execution"""
        base_price = signal['entry']
        suggested_price = base_price * (1 + analysis.suggested_slip * 0.7)
        
        return ExecutionPlan(
            base_price=base_price,
            suggested_price=suggested_price,
            order_type='LIMIT',
            time_in_force='GTD',
            post_only=False,
            execution_style='TWAP',
            estimated_fill_time=self.execution_patience * 0.3
        )
    
    def _calculate_cumulative_depth(self, levels: List[Tuple], 
                                  size: float) -> float:
        """Calculates cumulative depth up to required size"""
        cumulative = 0
        for price, quantity in levels:
            cumulative += price * quantity
            if cumulative >= size:
                break
        return cumulative
    
    def _estimate_required_slippage(self, size: float, depth_score: float,
                                  volatility: float) -> float:
        """Estimates required slippage based on order size and market conditions"""
        # Base slippage using square root law
        base_slip = 0.0001 * np.sqrt(size / depth_score)
        
        # Adjust for volatility
        vol_adjustment = volatility * 2
        
        # Combine with minimum slippage
        return max(base_slip + vol_adjustment, self.slippage_buffer)

# ------------------------------------------------------------------------------
# ENSEMBLE ML MODEL ARCHITECTURE (CORRECTED)
# ------------------------------------------------------------------------------

class DynamicEnsembleModel:
    """Advanced ensemble with dynamic weight allocation"""

    def __init__(self):
        self.models = {}
        self.weights = {}
        self.performance_history = defaultdict(list)
        self.feature_importance_history = []
        self.calibrators = {}
        # ### NEW: Store feature names for XAI ###
        self.feature_names = []
        # ### NEW: Track training status ###
        self.is_trained = False
        self.scaler = None # ADDED: To store the scaler

    def initialize_models(self):
        """Initialize base learners with appropriate parameters"""

        # Random Forest - robust to noise
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )

        # XGBoost - strong tabular performance
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        # LightGBM - fast with large feature sets
        self.models['lgb'] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        # Initialize equal weights
        for model_name in self.models.keys():
            self.weights[model_name] = 1.0 / len(self.models)
        self.is_trained = False

    def fit_scaler(self, X: pd.DataFrame):
        """Fit scaler on training data"""
        self.scaler = StandardScaler()
        self.scaler.fit(X[self.feature_names])

    def time_series_cv_split(self, X, y, n_splits=5):
        """Time-series aware cross-validation splits"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []

        for train_idx, val_idx in tscv.split(X):
            # Add embargo period to prevent data leakage
            embargo_size = int(0.1 * len(val_idx))
            if embargo_size > 0:
                val_idx = val_idx[embargo_size:]

            splits.append((train_idx, val_idx))

        return splits

    def train_ensemble(self, X, y, feature_names):
        """Train ensemble with time-series CV and calibration"""
        self.feature_names = feature_names
        
        # Convert to DataFrame if not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        
        # Fit scaler
        self.fit_scaler(X)
        X_scaled = self.scaler.transform(X)

        # Generate CV splits
        cv_splits = self.time_series_cv_split(X_scaled, y)

        for model_name, model in self.models.items():
            logger.info(f"Training {model_name}")

            # Store out-of-fold predictions for calibration
            oof_predictions = np.zeros(len(X))
            feature_importances = []

            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train model
                model.fit(X_train, y_train)

                # Get predictions
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model.predict(X_val)

                oof_predictions[val_idx] = y_pred

                # Track feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)

            # Calibrate model using out-of-fold predictions
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            
            # Create out-of-fold predictions for calibration
            oof_preds_for_calib = np.zeros_like(y, dtype=float)
            for train_idx_cal, val_idx_cal in cv_splits:
                # Create a fresh model instance for each fold
                fresh_model = self.models[model_name].__class__(**self.models[model_name].get_params())
                fresh_model.fit(X_scaled[train_idx_cal], y[train_idx_cal])
                oof_preds_for_calib[val_idx_cal] = fresh_model.predict_proba(X_scaled[val_idx_cal])[:, 1]
            
            # Fit the calibrator
            calibrated_model.fit(X_scaled, y)

            self.models[model_name] = calibrated_model
            self.calibrators[model_name] = calibrated_model

            # Store feature importance
            if feature_importances:
                avg_importance = np.mean(feature_importances, axis=0)
                self.feature_importance_history.append({
                    'model': model_name,
                    'importance': dict(zip(feature_names, avg_importance)),
                    'timestamp': datetime.now()
                })
    
        self.is_trained = True
        
    def save_model(self, path: str = './ensemble_model.joblib'):
        """Saves the trained ensemble model and feature names."""
        if not JOBLIB_SHAP_AVAILABLE:
            logger.warning("Joblib not available, cannot save model.")
            return
        try:
            model_data = {
                'models': self.models,
                'calibrators': self.calibrators,
                'weights': self.weights,
                'feature_names': self.feature_names,
                'scaler': self.scaler
            }
            joblib.dump(model_data, path)
            logger.info("Ensemble model saved successfully", path=path)
        except Exception as e:
            logger.error("Failed to save ensemble model", error=str(e))

    def load_model(self, path: str = './ensemble_model.joblib'):
        """
        Loads a pre-trained ensemble model. This version ensures that the
        loaded model is fully functional for both prediction and SHAP analysis.
        """
        if not (JOBLIB_SHAP_AVAILABLE and os.path.exists(path)):
            logger.warning("Joblib not available or model file not found, cannot load model.", path=path)
            self.is_trained = False
            return False
        try:
            model_data = joblib.load(path)
            
            # Directly assign the loaded models. The key is how we access them later.
            self.models = model_data.get('models', {})
            
            # Post-load validation to ensure models are usable
            for name, model in self.models.items():
                if not (hasattr(model, 'predict_proba') and hasattr(model, 'classes_')):
                     raise AttributeError(f"Model {name} loaded incorrectly and is missing critical attributes.")

            self.calibrators = model_data.get('calibrators', self.models)
            self.weights = model_data.get('weights', {})
            self.feature_names = model_data.get('feature_names', [])
            self.scaler = model_data.get('scaler', None) # ADDED: Load the scaler
            self.is_trained = True # Set status to true
            
            if self.scaler is None:
                logger.warning("Scaler not found in model file. Predictions may be inaccurate.")
            
            logger.info("Ensemble model loaded successfully", path=path)
            return True
            
        except Exception as e:
            logger.error("Failed to load ensemble model", error=str(e), exc_info=True)
            self.is_trained = False
            return False
            
    def update_weights(self, recent_performance: Dict[str, float], beta: float = 2.0, ema_alpha: float = 0.8):
        """Update model weights based on recent performance"""

        if not recent_performance:
            return

        # Convert performance scores to weights using softmax
        performance_scores = np.array([recent_performance.get(m, 0.5) for m in self.models.keys()])
        new_weights = self.softmax(performance_scores, beta)

        # EMA smoothing with previous weights
        prev_weights = np.array([self.weights[m] for m in self.models.keys()])
        blended_weights = ema_alpha * prev_weights + (1 - ema_alpha) * new_weights

        # Normalize and apply floor/ceiling
        blended_weights = np.clip(blended_weights, 0.1, 0.8)  # No model below 10% or above 80%
        blended_weights = blended_weights / blended_weights.sum()

        # Update weights
        for i, model_name in enumerate(self.models.keys()):
            self.weights[model_name] = blended_weights[i]

        logger.info("ensemble_weights_updated", weights=dict(zip(self.models.keys(), blended_weights)))

    def softmax(self, x: np.ndarray, beta: float = 1.0) -> np.ndarray:
        """Softmax function for weight conversion"""
        x = np.array(x)
        ex = np.exp(beta * (x - np.max(x)))
        return ex / ex.sum()

    def predict(self, X: pd.DataFrame) -> np.ndarray: # CHANGED: Takes DataFrame
        """Ensemble prediction with dynamic weights"""
        if not self.models or not self.is_trained or self.scaler is None:
            return np.full(X.shape[0], 0.5)

        # Reorder columns to match training order and scale the data
        X_prepared = X[self.feature_names]
        X_scaled = self.scaler.transform(X_prepared)

        predictions = {}

        for model_name, model in self.models.items():
            try:
                pred = model.predict_proba(X_scaled)[:, 1]
                predictions[model_name] = pred
            except Exception as e:
                logger.warning(f"Model {model_name} prediction failed: {e}")
                predictions[model_name] = np.full(X_scaled.shape[0], 0.5)

        # Weighted ensemble prediction
        ensemble_pred = np.zeros(X_scaled.shape[0])
        for model_name, pred in predictions.items():
            ensemble_pred += self.weights.get(model_name, 0) * pred

        return ensemble_pred
    
    def track_performance(self, model_name: str, actual: np.ndarray, predicted: np.ndarray):
        """Track model performance for weight updates"""

        if len(actual) == 0 or len(predicted) == 0:
            return

        # Use Brier score for calibration assessment
        brier_score = brier_score_loss(actual, predicted)

        # Convert to performance score (higher is better)
        performance_score = 1.0 - brier_score

        self.performance_history[model_name].append(performance_score)

        # Keep only recent history
        if len(self.performance_history[model_name]) > 100:
            self.performance_history[model_name] = self.performance_history[model_name][-100:]

    def get_recent_performance(self, window: int = 20) -> Dict[str, float]:
        """Get recent performance metrics for weight calculation"""

        recent_performance = {}

        for model_name, scores in self.performance_history.items():
            if len(scores) >= window:
                recent_scores = scores[-window:]
                recent_performance[model_name] = np.mean(recent_scores)
            elif scores:
                recent_performance[model_name] = np.mean(scores)
            else:
                recent_performance[model_name] = 0.5  # Default score

        return recent_performance

# ------------------------------------------------------------------------------
# ### NEW: Explainable AI (XAI) Analyzer (CORRECTED & FINAL) ###
# ------------------------------------------------------------------------------
class XAIAnalyzer:
    """
    Generates explanations for model predictions using SHAP.
    This version contains the definitive fix for loading calibrated models.
    """
    def __init__(self, ensemble_model: DynamicEnsembleModel):
        self.explainer = None
        self.ensemble_model = ensemble_model # Keep a reference
        self.feature_names = ensemble_model.feature_names if hasattr(ensemble_model, 'feature_names') else []
        
        if not JOBLIB_SHAP_AVAILABLE:
            logger.warning("SHAP or Joblib not installed. XAI features will be disabled.")
            return

        tree_model = self._extract_tree_model(ensemble_model)
        if tree_model:
            try:
                self.explainer = shap.TreeExplainer(tree_model)
                logger.info("SHAP TreeExplainer initialized successfully.")
                return
            except Exception as e:
                logger.warning(f"SHAP TreeExplainer failed, will attempt fallback.", error=str(e))
        
        logger.warning("No suitable tree model for SHAP, XAI will be disabled.")

    def _extract_tree_model(self, ensemble_model: DynamicEnsembleModel):
        """
        Extracts the raw, FITTED tree-based model from within a loaded
        CalibratedClassifierCV wrapper. This is the definitive fix for the SHAP error.
        """
        tree_model_names = ['xgb', 'lgb', 'rf']
        for name in tree_model_names:
            if name in ensemble_model.models:
                calibrated_model_wrapper = ensemble_model.models[name]
                if isinstance(calibrated_model_wrapper, CalibratedClassifierCV):
                    if hasattr(calibrated_model_wrapper, 'calibrated_classifiers_') and calibrated_model_wrapper.calibrated_classifiers_:
                        # <<< BUG FIX WAS HERE <<<
                        # The internal object holds the base model in the '.estimator' attribute after being loaded.
                        base_model = calibrated_model_wrapper.calibrated_classifiers_[0].estimator
                        return base_model
                else:
                    return calibrated_model_wrapper # It's already the base model
        return None

    def get_top_contributors(self, features_df: pd.DataFrame, top_n: int = 3) -> str:
        """Calculates SHAP values and returns top contributing features."""
        if self.explainer is None or features_df.empty or not self.feature_names:
            return "XAI not available"

        try:
            # Reorder columns to match training order for the explainer
            recent_features = features_df[self.feature_names].iloc[-1:]

            shap_values = self.explainer.shap_values(recent_features)
            
            if isinstance(shap_values, list): # For classifiers
                shap_values = shap_values[1]
            
            contributions = pd.Series(shap_values.flatten(), index=self.feature_names)
            top_features = contributions.abs().nlargest(top_n)

            explanation = []
            for feature, value in top_features.items():
                effect = "bullish" if contributions[feature] > 0 else "bearish"
                explanation.append(f"{feature} ({effect})")
            
            return ", ".join(explanation) if explanation else "No strong contributors"
        except Exception as e:
            logger.warning("SHAP explanation failed", error=str(e))
            return "XAI analysis failed"

# ------------------------------------------------------------------------------
# ### NEW: Reinforcement Learning (RL) Agent Structure ###
# ------------------------------------------------------------------------------
class RLAgent:
    """
    A wrapper for a pre-trained Stable-Baselines3 RL agent.
    This class provides the structure to load and use an agent, but
    the agent itself must be trained separately by the user.
    """
    def __init__(self, model_path: str = './rl_agent_ppo.zip'):
        self.model = None
        self.model_path = model_path

        if not STABLE_BASELINES_AVAILABLE:
            logger.warning("Stable-Baselines3 or Gymnasium not installed. RL features disabled.")
            return

        self.load_agent()

    def load_agent(self):
        """Loads a pre-trained PPO agent from disk."""
        if os.path.exists(self.model_path):
            try:
                self.model = PPO.load(self.model_path)
                logger.info(f"RL agent loaded successfully from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load RL agent: {e}")
        else:
            logger.warning(f"RL agent model not found at {self.model_path}. RL agent is inactive.")

    def predict_action(self, features_df: pd.DataFrame) -> Optional[Tuple[str, float]]:
        """
        Uses the loaded RL agent to predict the next action (Buy, Sell, Hold).

        Returns:
            A tuple of (action_string, confidence) or None if no action.
            Example: ('Long', 0.85)
        """
        if self.model is None or features_df.empty:
            return None

        try:
            # IMPORTANT: The features must be preprocessed and scaled
            # exactly as they were during the training of the RL agent.
            # This is a placeholder for the user's specific preprocessing.
            obs = features_df.iloc[-1].values.astype(np.float32)

            action, _states = self.model.predict(obs, deterministic=True)

            # The mapping from action (integer) to trade decision depends on the
            # action space defined when the agent was trained.
            # Common action space: 0=Hold, 1=Buy, 2=Sell
            if action == 1:
                return ('Long', 0.75) # Confidence can be fixed or derived from action probabilities
            elif action == 2:
                return ('Short', 0.75)
            else: # Action 0 or any other
                return None
        except Exception as e:
            logger.error("RL agent failed to predict action", error=str(e))
            return None

# ------------------------------------------------------------------------------
# MARKET MICROSTRUCTURE ANALYSIS
# ------------------------------------------------------------------------------

class MarketMicrostructureAnalyzer:
    """Advanced order book and market microstructure analysis"""

    def __init__(self):
        self.order_book_cache = {}
        self.manipulation_patterns = {}

    async def get_order_book_data(self, symbol: str, exchange, depth: int = 20):
        """Fetch and cache order book data"""
        cache_key = f"{symbol}_{depth}"

        if cache_key in self.order_book_cache:
            cached_data = self.order_book_cache[cache_key]
            if time.time() - cached_data['timestamp'] < 2.0:  # 2 second cache
                return cached_data['data']

        try:
            order_book = await exchange.fetch_order_book(symbol, depth)
            self.order_book_cache[cache_key] = {
                'data': order_book,
                'timestamp': time.time()
            }
            return order_book
        except Exception as e:
            logger.warning(f"Failed to fetch order book for {symbol}: {e}")
            return None

    def calculate_order_book_imbalance(self, order_book, levels: int = 5) -> float:
        """Calculate Order Book Imbalance (OBI)"""
        if not order_book:
            return 0.0

        bid_volumes = sum(order_book['bids'][i][1] for i in range(min(levels, len(order_book['bids']))))
        ask_volumes = sum(order_book['asks'][i][1] for i in range(min(levels, len(order_book['asks']))))

        total_volume = bid_volumes + ask_volumes
        if total_volume == 0:
            return 0.0

        return (bid_volumes - ask_volumes) / total_volume

    def calculate_liquidity_depth(self, order_book, price_levels: int = 10) -> Dict[str, float]:
        """Analyze liquidity depth at different price levels"""
        if not order_book:
            return {}

        # Cumulative depth
        bid_depth = sum(level[1] for level in order_book['bids'][:price_levels])
        ask_depth = sum(level[1] for level in order_book['asks'][:price_levels])

        # Depth ratio
        total_depth = bid_depth + ask_depth
        depth_ratio = bid_depth / (ask_depth + 1e-9)

        # Slippage estimation (simplified square-root model)
        mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
        slippage_bps = 10.0 / np.sqrt(total_depth)  # Simplified model

        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'depth_ratio': depth_ratio,
            'slippage_bps': slippage_bps,
            'total_liquidity': total_depth
        }

    def detect_large_walls(self, order_book, z_threshold: float = 2.5) -> Dict[str, List]:
        """Detect large buy/sell walls in order book"""
        if not order_book:
            return {'bid_walls': [], 'ask_walls': []}

        bid_walls = []
        ask_walls = []

        # Analyze bid side
        bid_sizes = [level[1] for level in order_book['bids'][:10]]
        if bid_sizes:
            bid_mean = np.mean(bid_sizes)
            bid_std = np.std(bid_sizes)
            for i, size in enumerate(bid_sizes):
                if bid_std > 0 and (size - bid_mean) / bid_std > z_threshold:
                    bid_walls.append((order_book['bids'][i][0], size))

        # Analyze ask side
        ask_sizes = [level[1] for level in order_book['asks'][:10]]
        if ask_sizes:
            ask_mean = np.mean(ask_sizes)
            ask_std = np.std(ask_sizes)
            for i, size in enumerate(ask_sizes):
                if ask_std > 0 and (size - ask_mean) / ask_std > z_threshold:
                    ask_walls.append((order_book['asks'][i][0], size))

        return {'bid_walls': bid_walls, 'ask_walls': ask_walls}

    def calculate_spread_metrics(self, order_book) -> Dict[str, float]:
        """Calculate spread-based quality metrics"""
        if not order_book or len(order_book['bids']) == 0 or len(order_book['asks']) == 0:
            return {}

        best_bid = order_book['bids'][0][0]
        best_ask = order_book['asks'][0][0]
        mid_price = (best_bid + best_ask) / 2

        relative_spread = (best_ask - best_bid) / mid_price

        # Quote volatility (simplified - would need historical data)
        spread_quality = 1.0 / (relative_spread + 1e-9)  # Higher is better

        return {
            'relative_spread': relative_spread,
            'spread_quality': spread_quality,
            'best_bid': best_bid,
            'best_ask': best_ask
        }

    def detect_manipulation_patterns(self, symbol: str, order_book, recent_trades) -> Dict[str, bool]:
        """Detect potential market manipulation patterns"""
        patterns = {
            'spoofing': False,
            'layering': False,
            'quote_stuffing': False,
            'wash_trading': False
        }

        if not order_book:
            return patterns

        # Simplified heuristics - would need more sophisticated detection
        walls = self.detect_large_walls(order_book)

        # Spoofing detection: large orders far from mid price
        mid_price = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
        for bid_wall in walls['bid_walls']:
            if bid_wall[0] < mid_price * 0.995:  # More than 0.5% below mid
                patterns['spoofing'] = True
                break

        for ask_wall in walls['ask_walls']:
            if ask_wall[0] > mid_price * 1.005:  # More than 0.5% above mid
                patterns['spoofing'] = True
                break

        # Layering: multiple large orders at different price levels
        if len(walls['bid_walls']) >= 2 or len(walls['ask_walls']) >= 2:
            patterns['layering'] = True

        return patterns

    async def get_microstructure_features(self, symbol: str, exchange) -> Dict[str, Any]:
        """Get comprehensive microstructure features"""

        order_book = await self.get_order_book_data(symbol, exchange)
        if not order_book:
            return {}

        # Calculate all microstructure metrics
        obi = self.calculate_order_book_imbalance(order_book)
        liquidity = self.calculate_liquidity_depth(order_book)
        walls = self.detect_large_walls(order_book)
        spread_metrics = self.calculate_spread_metrics(order_book)
        manipulation = self.detect_manipulation_patterns(symbol, order_book, [])

        # Combine all features
        features = {
            'order_book_imbalance': obi,
            'liquidity_ratio': liquidity.get('depth_ratio', 1.0),
            'total_liquidity': liquidity.get('total_liquidity', 0.0),
            'slippage_estimate': liquidity.get('slippage_bps', 10.0),
            'relative_spread': spread_metrics.get('relative_spread', 0.0),
            'spread_quality': spread_metrics.get('spread_quality', 0.0),
            'has_bid_walls': len(walls['bid_walls']) > 0,
            'has_ask_walls': len(walls['ask_walls']) > 0,
            'suspected_manipulation': any(manipulation.values()),
            **manipulation
        }

        return features

# ------------------------------------------------------------------------------
# ENHANCED HYBRID TRADING STRATEGY INTEGRATION
# ------------------------------------------------------------------------------

class EnhancedHybridTradingStrategy:
    """Enhanced strategy that incorporates all advanced features"""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.ensemble_model = DynamicEnsembleModel()
        self.microstructure_analyzer = MarketMicrostructureAnalyzer()
        self.regime_classifier = EnhancedMarketRegimeClassifier()
        self.enhanced_ml_features = EnhancedMLFeatures()  # NEW: Added enhanced ML features
        self.enhanced_risk_manager = EnhancedDynamicRiskManager({
            'portfolio_value': config.portfolio_value,
            'max_position_size': config.max_position_size,
            'max_leverage': 20,
            'base_risk_per_trade': config.base_risk_per_trade
        })
        self.smart_executor = EnhancedSmartOrderExecutor({
            'slippage_buffer': config.slippage_buffer,
            'max_spread': config.max_spread,
            'min_depth': config.min_depth,
            'execution_patience': config.execution_patience
        })

    async def prepare_ml_features(self, df: pd.DataFrame, symbol: str, exchange) -> Optional[pd.DataFrame]:
        """Prepare comprehensive features for ML prediction with enhanced features"""
        try:
            # Get market data for microstructure features
            market_data = {}
            try:
                order_book = await exchange.fetch_order_book(symbol, 20)
                trades = await exchange.fetch_trades(symbol, limit=100)
                market_data = {
                    'symbol': symbol,
                    'orderbook': order_book,
                    'trades': trades
                }
            except Exception as e:
                logger.debug(f"Could not fetch market data for {symbol}: {e}")
            
            # Generate enhanced features if enabled
            if config.enhanced_ml_features:
                features = await self.enhanced_ml_features.generate_features(df, market_data)
            else:
                # Fallback to original feature engineering
                features = self.feature_engineer.generate_all_features(df)
            
            # Add microstructure features if available
            if config.microstructure_enabled:
                micro_features = await self.microstructure_analyzer.get_microstructure_features(symbol, exchange)
                if micro_features:
                    for feature_name, value in micro_features.items():
                        features[feature_name] = value

            # Ensure all columns required by the model are present
            if hasattr(self.ensemble_model, 'feature_names') and self.ensemble_model.feature_names:
                required_cols = self.ensemble_model.feature_names
                for col in required_cols:
                    if col not in features.columns:
                        features[col] = 0
            
            return features.fillna(0)
            
        except Exception as e:
            logger.error(f"Feature preparation failed for {symbol}: {e}")
            # Fallback to basic features
            return self.feature_engineer.generate_all_features(df).fillna(0)

    def calculate_ml_confidence(self, features: pd.DataFrame) -> float:
        """Calculate ML-based confidence score"""
        if not self.ensemble_model.is_trained:
            return 0.5  # Default confidence if model not ready

        try:
            # The predict method now handles scaling and column ordering
            ml_confidence = self.ensemble_model.predict(features.iloc[-1:])[0]
            return float(ml_confidence)
        except Exception as e:
            logger.warning(f"ML confidence calculation failed: {e}", exc_info=True)
            return 0.5

    def adjust_position_with_ml(self, technical_signal: Dict, ml_confidence: float) -> Dict:
        """Adjust technical signal with ML confidence"""

        # ML adjustment factor
        confidence_factor = np.clip((ml_confidence - 0.5) / 0.5, -1.0, 1.0)

        # Adjust position size based on ML confidence
        base_size = technical_signal.get('position_size_coin', 0)
        adjusted_size = base_size * (1.0 + confidence_factor * 0.5)  # Up to ±50% adjustment

        # Create enhanced signal
        enhanced_signal = technical_signal.copy()
        enhanced_signal['position_size_coin'] = adjusted_size
        enhanced_signal['ml_confidence'] = ml_confidence
        enhanced_signal['confidence_factor'] = confidence_factor
        enhanced_signal['is_ml_enhanced'] = True

        return enhanced_signal

    def should_allow_trade(self, technical_signal: Dict, microstructure_features: Dict) -> Tuple[bool, str]:
        """Microstructure-based trade gating"""

        symbol = technical_signal.get('symbol', '')
        side = technical_signal.get('side', '').lower()

        # Spread quality check
        spread_quality = microstructure_features.get('spread_quality', 0)
        if spread_quality < 1000:  # Threshold needs calibration
            return False, f"Poor spread quality: {spread_quality:.2f}"

        # Order book imbalance alignment
        obi = microstructure_features.get('order_book_imbalance', 0)
        if side == 'long' and obi < -0.3:  # Heavy selling pressure
            return False, f"OBI against direction: {obi:.3f}"
        if side == 'short' and obi > 0.3:  # Heavy buying pressure
            return False, f"OBI against direction: {obi:.3f}"

        # Manipulation detection
        if microstructure_features.get('suspected_manipulation', False):
            return False, "Suspected market manipulation"

        # Liquidity check
        liquidity = microstructure_features.get('total_liquidity', 0)
        position_size = technical_signal.get('position_value', 0)
        if liquidity > 0 and position_size > liquidity * 0.1:  # Don't take more than 10% of liquidity
            return False, f"Insufficient liquidity: {liquidity:.0f} vs position {position_size:.0f}"

        return True, "OK"

    async def generate_enhanced_signal(self, market_id: str, df: pd.DataFrame, 
                                     analyzer: 'ScanAnalyzer') -> Optional[Dict[str, Any]]:
        """Generate enhanced trading signal with all advanced features"""
        try:
            # Get base signal from parent class
            base_signal = await generate_trend_momentum_signal(market_id, df, analyzer)
            if not base_signal:
                return None

            # Enhanced regime classification
            if config.enhanced_regime_classification:
                regime_metrics = self.regime_classifier.classify_regime(df)
                base_signal['enhanced_regime'] = regime_metrics.regime.value
                base_signal['regime_confidence'] = regime_metrics.confidence
            else:
                # Fallback to existing regime detection
                base_signal['enhanced_regime'] = df['market_regime_hurst'].iloc[-2]
                base_signal['regime_confidence'] = 0.7

            # Enhanced risk management
            if config.enhanced_risk_enabled:
                current_exposure = sum(advanced_risk_manager.open_positions_usd.values())
                risk_result = await self.enhanced_risk_manager.calculate_position_size(
                    entry_price=base_signal['entry'],
                    stop_loss=base_signal['sl'],
                    regime=MarketRegime(base_signal['enhanced_regime']),
                    volatility=df['ewma_vol_20'].iloc[-2] if 'ewma_vol_20' in df.columns else 0.2,
                    confidence=base_signal.get('confidence', 0.5),
                    current_exposure=current_exposure
                )
                
                # Update position sizing
                base_signal['position_size_coin'] = risk_result.size_coins
                base_signal['position_value'] = risk_result.size_usd
                base_signal['suggested_leverage'] = risk_result.leverage
                base_signal['risk_score'] = risk_result.risk_score
                base_signal['max_drawdown'] = risk_result.max_drawdown

            # ML confidence integration (existing)
            if config.use_ml and self.ensemble_model.is_trained:
                exchange = await exchange_factory.get_exchange()
                ml_features = await self.prepare_ml_features(df.copy(), market_id, exchange)
                if ml_features is not None:
                    ml_confidence = self.calculate_ml_confidence(ml_features)
                    base_signal['ml_confidence'] = ml_confidence
                    base_signal['confidence'] = (base_signal.get('confidence', 0.5) + ml_confidence) / 2

            return base_signal

        except Exception as e:
            logger.error("enhanced_signal_generation_failed", symbol=market_id, error=str(e))
            return None

    async def get_smart_execution_plan(self, signal: Dict, exchange) -> ExecutionPlan:
        """Get smart execution plan for the signal"""
        if not config.smart_order_execution:
            # Return default execution plan
            return ExecutionPlan(
                base_price=signal['entry'],
                suggested_price=signal['entry'],
                order_type='LIMIT',
                time_in_force='GTC',
                post_only=True,
                execution_style='IMMEDIATE',
                estimated_fill_time=1.0
            )

        try:
            # Get order book data
            order_book = await exchange.fetch_order_book(signal['symbol'], 20)
            recent_trades = await exchange.fetch_trades(signal['symbol'], limit=50)
            
            # Get optimized execution plan
            execution_plan = await self.smart_executor.optimize_entry(
                signal, order_book, recent_trades
            )
            
            return execution_plan
            
        except Exception as e:
            logger.warning("smart_execution_failed", symbol=signal['symbol'], error=str(e))
            # Fallback to default
            return ExecutionPlan(
                base_price=signal['entry'],
                suggested_price=signal['entry'],
                order_type='LIMIT',
                time_in_force='GTC',
                post_only=True,
                execution_style='IMMEDIATE',
                estimated_fill_time=1.0
            )

# ------------------------------------------------------------------------------
# Configuration & Validation (UPDATED FOR 1H TIMEFRAME)
# ------------------------------------------------------------------------------
class Config:
    def __init__(self):
        self.bot_token = os.environ.get("CRYPTO_BOT_TOKEN")
        self.owner_id = os.environ.get("CRYPTO_OWNER_ID")
        self.owner_id_int = int(self.owner_id) if self.owner_id and self.owner_id.isdigit() else None
        
        # FIXED: Enhanced group chat ID handling
        self.group_chat_id = os.environ.get("CRYPTO_GROUP_ID", "").strip()
        self.group_chat_id_int = None
        if self.group_chat_id:
            try:
                self.group_chat_id_int = int(self.group_chat_id)
            except (ValueError, TypeError):
                logger.warning("Invalid group chat ID format, group notifications disabled")

        # --- OPTIMIZED FOR 1H TIMEFRAME ---
        self.timeframe = "1h"  # Changed from 30m to 1h
        self.top_n_markets = int(os.environ.get("TOP_N_MARKETS", 30))  # Reduced for 1h
        self.portfolio_value = float(os.environ.get("PORTFOLIO_VALUE", 10000.0))
        
        # Technical indicator periods optimized for 1h
        self.atr_period = int(os.environ.get("ATR_PERIOD", 20))  # Increased for 1h
        self.rsi_period = int(os.environ.get("RSI_PERIOD", 16))  # Adjusted for 1h
        
        # Profit targets optimized for 1h (more conservative)
        self.tp_mult = [float(x) for x in os.environ.get("TP_MULT", "1.5,2.8,4.5").split(",")]
        self.sl_mult = float(os.environ.get("SL_MULT", "2.0"))  # Wider SL for 1h
        
        # Trailing stop loss for 1h
        self.trailing_stop_loss_enabled = os.environ.get("TRAILING_STOP_LOSS_ENABLED", "true").lower() == "true"
        self.trailing_stop_loss_percent = float(os.environ.get("TRAILING_STOP_LOSS_PERCENT", 0.025))  # 2.5% for 1h

        # Risk Management Params optimized for 1h
        self.max_daily_loss = float(os.environ.get("MAX_DAILY_LOSS", 0.015))  # More conservative
        self.max_concurrent_trades = int(os.environ.get("MAX_CONCURRENT_TRADES", 12))  # Reduced for 1h
        self.max_position_size = float(os.environ.get("MAX_POSITION_SIZE", 0.015))  # Smaller positions
        self.correlation_threshold = float(os.environ.get("CORRELATION_THRESHOLD", 0.65))  # Stricter

        # Quality thresholds for 1h
        self.base_require_mtf_score = float(os.environ.get("BASE_REQUIRE_MTF_SCORE", 0.30))  # Higher quality
        self.base_confidence_floor = float(os.environ.get("BASE_CONFIDENCE_FLOOR", 0.35))  # Higher confidence
        
        # Enhanced features for 1h
        self.enhanced_ml_features = os.environ.get("ENHANCED_ML_FEATURES", "true").lower() == "true"

        # Phase 1 Enhancement Settings
        self.enhanced_indicators = os.environ.get("ENHANCED_INDICATORS", "true").lower() == "true"
        self.kelly_criterion_enabled = os.environ.get("KELLY_CRITERION_ENABLED", "true").lower() == "true"
        self.max_portfolio_heat = float(os.environ.get("MAX_PORTFOLIO_HEAT", 0.20))  # Reduced for 1h
        self.dynamic_sizing_enabled = os.environ.get("USE_DYNAMIC_POSITION_SIZING", "true").lower() == "true"

        # ### NEW: Strategy & AI Model Flags ###
        self.dynamic_strategy_selection = os.environ.get("DYNAMIC_STRATEGY_SELECTION", "true").lower() == "true"
        self.use_xai_explanations = os.environ.get("USE_XAI_EXPLANATIONS", "true").lower() == "true"
        self.use_rl_agent = os.environ.get("USE_RL_AGENT", "false").lower() == "true"
        self.ensemble_model_path = os.environ.get("ENSEMBLE_MODEL_PATH", "./ensemble_model.joblib")
        self.rl_agent_model_path = os.environ.get("RL_AGENT_MODEL_PATH", "./rl_agent_ppo.zip")
        
        # Enhanced Features
        self.enhanced_risk_enabled = os.environ.get("ENHANCED_RISK_ENABLED", "true").lower() == "true"
        self.enhanced_regime_classification = os.environ.get("ENHANCED_REGIME_CLASSIFICATION", "true").lower() == "true"
        self.smart_order_execution = os.environ.get("SMART_ORDER_EXECUTION", "true").lower() == "true"
        self.base_risk_per_trade = float(os.environ.get("BASE_RISK_PER_TRADE", 0.008))  # Reduced for 1h

        # Order Execution
        self.slippage_buffer = float(os.environ.get("SLIPPAGE_BUFFER", 0.0015))  # Increased for 1h
        self.max_spread = float(os.environ.get("MAX_SPREAD", 0.004))  # Increased for 1h
        self.min_depth = float(os.environ.get("MIN_DEPTH", 15000))  # Increased for 1h
        self.execution_patience = int(os.environ.get("EXECUTION_PATIENCE", 90))  # Increased for 1h
        # ### END NEW ###

        # System & Bot Settings
        self.scan_interval = int(os.environ.get("SCAN_INTERVAL", 1800))  # 30 minutes for 1h
        self.monitor_interval = int(os.environ.get("MONITOR_INTERVAL", 300))  # 5 minutes
        self.cache_ttl = int(os.environ.get("CACHE_TTL", 300))  # Increased for 1h
        self.chart_candles = int(os.environ.get("CHART_CANDLES", 150))  # Increased for 1h
        self.report_timezone = ZoneInfo(os.environ.get("REPORT_TIMEZONE", "Asia/Kolkata"))
        self.db_path = os.environ.get("DB_PATH", "./power_crypto_bot.db")
        self.db_url = os.environ.get('DB_URL', f"sqlite+aiosqlite:///{self.db_path}")
        self.redis_url = os.environ.get("REDIS_URL", "").strip()
        self.skip_symbols = os.environ.get("SKIP_SYMBOLS", " ").split(",")
        self.max_retries = int(os.environ.get("MAX_RETRIES", 3))
        self.environment = os.environ.get("ENVIRONMENT", "development")
        self.use_ml = os.environ.get("USE_ML", "true").lower() == "true"
        self.hybrid_model_path = os.environ.get("HYBRID_MODEL_PATH", './hybrid_model.pth')

        # ML and Ensemble settings
        self.ml_enabled = os.environ.get("ML_ENABLED", "true").lower() == "true"
        self.ensemble_enabled = os.environ.get("ENSEMBLE_ENABLED", "true").lower() == "true"
        self.ml_confidence_threshold = float(os.environ.get("ML_CONFIDENCE_THRESHOLD", 0.45))  # Increased for 1h
        self.min_ml_confidence = float(os.environ.get("MIN_ML_CONFIDENCE", 0.35))  # Increased for 1h

        # Microstructure settings
        self.microstructure_enabled = os.environ.get("MICROSTRUCTURE_ENABLED", "true").lower() == "true"
        self.min_spread_quality = float(os.environ.get("MIN_SPREAD_QUALITY", 800.0))  # Adjusted for 1h
        self.max_obi_against = float(os.environ.get("MAX_OBI_AGAINST", 0.25))  # Stricter for 1h
        self.min_liquidity_ratio = float(os.environ.get("MIN_LIQUIDITY_ratio", 0.15))  # Increased for 1h

        self.validate()

    def validate(self):
        """Enhanced validation with better error messages."""
        required = {
            'CRYPTO_BOT_TOKEN': 'Telegram Bot Token from @BotFather',
            'CRYPTO_OWNER_ID': 'Your Telegram User ID',
        }

        missing = []
        for var, description in required.items():
            if not os.environ.get(var):
                missing.append(f"  • {var}: {description}")

        if missing:
            error_msg = "Missing required environment variables:\n" + "\n".join(missing)
            error_msg += "\n\nPlease set these in your .env file or environment."
            raise ValueError(error_msg)

        # Validate numeric values
        try:
            int(self.owner_id)
        except (ValueError, TypeError):
            raise ValueError("CRYPTO_OWNER_ID must be a valid integer (Telegram User ID)")

        # Validate group chat ID if provided
        if self.group_chat_id:
            try:
                group_id = int(self.group_chat_id)
                # Group IDs typically start with -100
                if group_id > 0:
                    logger.warning("Group chat ID appears to be a user ID, not a group ID")
            except (ValueError, TypeError):
                logger.warning(f"Invalid group chat ID format: {self.group_chat_id}")
                self.group_chat_id_int = None

        # Validate portfolio value
        if self.portfolio_value <= 0:
            raise ValueError("PORTFOLIO_VALUE must be greater than 0")

        # Validate risk parameters
        if not 0 < self.max_position_size < 1:
            raise ValueError("MAX_POSITION_SIZE must be between 0 and 1")

        if not 0 < self.max_daily_loss < 1:
            raise ValueError("MAX_DAILY_LOSS must be between 0 and 1")

        logger.info(f"Configuration validated - 1H Timeframe Optimized - Group Chat: {'Enabled' if self.group_chat_id_int else 'Disabled'}")


config = Config()

# ------------------------------------------------------------------------------
# GLOBAL MANAGER INSTANCES
# ------------------------------------------------------------------------------

class EnhancedMemoryManager:
    """Simple memory cleanup manager."""
    def guard(self):
        # Called after each scan to clear garbage
        import gc
        gc.collect()
        return

    def cleanup(self):
        # Called on shutdown for full cleanup
        import gc
        gc.collect()
        return

memory_manager = EnhancedMemoryManager()

class ErrorRecoveryManager:
    """Handles progressive error recovery."""
    def __init__(self):
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5

    async def handle_error(self, error: Exception, context: str = ""):
        import structlog
        logger = structlog.get_logger()
        self.consecutive_errors += 1
        logger.error("error_recovery", context=context, error=str(error), count=self.consecutive_errors)
        # Simple recovery: reset error count after threshold
        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.critical("max_errors_reached", count=self.consecutive_errors)
            raise RuntimeError("Too many consecutive errors")
        # Otherwise, perform trivial backoff
        await asyncio.sleep(min(30, 5 * self.consecutive_errors))
    
    def reset_error_count(self):
        self.consecutive_errors = 0


error_recovery = ErrorRecoveryManager()

import time
from collections import defaultdict
import numpy as np

class PerformanceMonitor:
    """Tracks performance metrics over time."""
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = time.time()

    def record_metric(self, name: str, value: float):
        timestamp = time.time()
        self.metrics[name].append((timestamp, value))
        # Keep last 1000 entries
        if len(self.metrics[name]) > 1000:
            self.metrics[name].pop(0)

    def get_summary(self) -> dict:
        summary = {}
        for name, entries in self.metrics.items():
            values = [v for _, v in entries]
            if values:
                summary[name] = {
                    'count': len(values),
                    'avg': float(np.mean(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'last': float(values[-1])
                }
        summary['uptime_hours'] = (time.time() - self.start_time) / 3600
        return summary


performance_monitor = PerformanceMonitor()
feature_engineer = AdvancedFeatureEngineer()
hybrid_strategy = EnhancedHybridTradingStrategy()
microstructure_analyzer = MarketMicrostructureAnalyzer()
enhanced_ml_features = EnhancedMLFeatures()  # NEW: Global instance


import threading
import matplotlib.pyplot as plt

class ThreadSafePlotter:
    """Thread-safe chart plotting with candlestick charts."""
    def __init__(self):
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._lock = asyncio.Lock()

    async def plot_annotated_chart(self, df: pd.DataFrame, display_symbol: str,
                                 entry: float, sl: float, tps: list) -> str:
        """Thread-safe chart plotting with candlesticks"""
        async with self._lock:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self._plot_sync,
                df, display_symbol, entry, sl, tps
            )

    def _plot_sync(self, df: pd.DataFrame, display_symbol: str,
                  entry: float, sl: float, tps: list) -> str:
        """Synchronous plotting function with candlesticks."""
        try:
            matplotlib.use('Agg')
            plt.close('all')

            df_plot = df.tail(config.chart_candles).copy()

            # Create a layout with 3 subplots
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(15, 12), 
                gridspec_kw={'height_ratios': [3, 1, 1]}, 
                sharex=True
            )
            fig.patch.set_facecolor('#F0F0F0')

            # --- Main Price Chart (ax1) with Candlesticks ---
            ax1.set_facecolor('#FFFFFF')
            
            # Plot candlesticks
            width = np.min(np.diff(mdates.date2num(df_plot.index))) * 0.8
            up = df_plot[df_plot.close >= df_plot.open]
            down = df_plot[df_plot.close < df_plot.open]

            # Up candlesticks
            ax1.bar(up.index, up.close - up.open, width, bottom=up.open, 
                   color='#26a69a', edgecolor='black', linewidth=0.5)
            ax1.bar(up.index, up.high - up.close, width*0.1, bottom=up.close, 
                   color='#26a69a', edgecolor='black', linewidth=0.5)
            ax1.bar(up.index, up.low - up.open, width*0.1, bottom=up.open, 
                   color='#26a69a', edgecolor='black', linewidth=0.5)

            # Down candlesticks
            ax1.bar(down.index, down.close - down.open, width, bottom=down.open, 
                   color='#ef5350', edgecolor='black', linewidth=0.5)
            ax1.bar(down.index, down.high - down.open, width*0.1, bottom=down.open, 
                   color='#ef5350', edgecolor='black', linewidth=0.5)
            ax1.bar(down.index, down.low - down.close, width*0.1, bottom=down.close, 
                   color='#ef5350', edgecolor='black', linewidth=0.5)

            # Add EMAs
            ax1.plot(df_plot.index, df_plot['ema20'], color='blue', 
                    linestyle='--', linewidth=1, label='EMA(20)')
            ax1.plot(df_plot.index, df_plot['ema50'], color='orange', 
                    linestyle='--', linewidth=1, label='EMA(50)')

            # Chart title and styling
            ax1.set_ylabel('Price (USDT)')
            ax1.set_title(f'{display_symbol} Signal ({config.timeframe})', 
                         fontsize=18, fontweight='bold')
            ax1.grid(True, linestyle='--', alpha=0.5)

            # Add entry, SL and TP lines
            xmin, xmax = mdates.date2num(df_plot.index[0]), mdates.date2num(df_plot.index[-1])
            ax1.hlines(entry, xmin, xmax, colors='green', 
                      linestyles='-', label=f'Entry: {entry:.4f}', lw=2)
            ax1.hlines(sl, xmin, xmax, colors='red', 
                      linestyles='-', label=f'SL: {sl:.4f}', lw=2)

            for i, tp in enumerate(tps, start=1):
                ax1.hlines(tp, xmin, xmax, colors='purple', alpha=0.8, 
                          linestyles='--', label=f'TP{i}: {tp:.4f}')
            
            ax1.legend()

            # --- RSI Subplot (ax2) ---
            ax2.plot(df_plot.index, df_plot['rsi'], color='purple', 
                    label=f'RSI({config.rsi_period})')
            ax2.axhline(70, linestyle='--', color='red', alpha=0.5)
            ax2.axhline(30, linestyle='--', color='green', alpha=0.5)
            ax2.axhline(50, linestyle='--', color='gray', alpha=0.5)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('RSI')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.3)

            # --- MACD Subplot (ax3) ---
            ax3.plot(df_plot.index, df_plot['macd'], color='blue', label='MACD')
            ax3.plot(df_plot.index, df_plot['macd_sig'], color='orange', label='Signal')
            
            # MACD histogram
            macd_hist = df_plot['macd'] - df_plot['macd_sig']
            ax3.bar(df_plot.index, macd_hist, 
                   color=np.where(macd_hist >= 0, '#26a69a', '#ef5350'),
                   width=width, alpha=0.5)
            
            ax3.set_ylabel('MACD')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.3)

            # X-axis formatting
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=30)
            plt.xlabel('Time')
            plt.tight_layout()

            # Save and return
            fname = f'chart_{display_symbol.replace("/", "")}_{int(time.time())}.png'
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return fname

        except Exception as e:
            logger.error("charting_failed", error=str(e))
            plt.close('all')
            return ""


plotter = ThreadSafePlotter()

# ------------------------------------------------------------------------------
# NEW: HEALTH MONITOR INSTANCE
# ------------------------------------------------------------------------------
health_monitor = HealthMonitor()

# ------------------------------------------------------------------------------
# Structured Logging, Decorators & Utilities
# ------------------------------------------------------------------------------
structlog.configure(processors=[structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer()])
logger = structlog.get_logger()
IS_PAUSED = False

def track_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            logger.info("function_performance", function=func.__name__, execution_time=f"{time.time() - start_time:.4f}s")
            return result
        except Exception as e:
            logger.error("function_error", function=func.__name__, error=str(e), exc_info=True)
            raise
    return wrapper

def with_retry(max_retries=config.max_retries, delay=1.0, backoff=2.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt + 1 == max_retries:
                        logger.error("max_retries_exceeded", function=func.__name__, error=str(e))
                        raise
                    wait = delay * (backoff ** attempt)
                    logger.warning("retry_attempt", function=func.__name__, attempt=attempt+1, wait=f"{wait:.2f}s", error=str(e))
                    await asyncio.sleep(wait)
        return wrapper
    return decorator

class EnhancedMemoryManager:
    def __init__(self, max_percent: int = 85):
        self.max_percent = max_percent
    def cleanup(self):
        gc.collect()
        if PYTORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        plt.close('all')
    def guard(self):
        if psutil.virtual_memory().percent > self.max_percent:
            logger.warning("high_memory_usage", percent=psutil.virtual_memory().percent)
            self.cleanup()


# ------------------------------------------------------------------------------
# NEW: Advanced Error Recovery & Performance Monitoring
# ------------------------------------------------------------------------------
class ErrorRecoveryManager:
    def __init__(self):
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        self.recovery_strategies = [
            self.restart_exchange_connection,
            self.clear_cache,
            self.garbage_collect,
            self.reset_risk_managers
        ]

    async def handle_error(self, error: Exception, context: str = "unknown"):
        """Handle errors with progressive recovery strategies."""
        self.consecutive_errors += 1
        logger.error(f"error_recovery_triggered",
                    error=str(error),
                    context=context,
                    consecutive=self.consecutive_errors)

        if self.consecutive_errors <= len(self.recovery_strategies):
            strategy = self.recovery_strategies[self.consecutive_errors - 1]
            try:
                await strategy()
                logger.info(f"recovery_strategy_applied", strategy=strategy.__name__)
            except Exception as recovery_error:
                logger.error(f"recovery_strategy_failed",
                           strategy=strategy.__name__,
                           error=str(recovery_error))

        if self.consecutive_errors >= self.max_consecutive_errors:
            logger.critical("max_consecutive_errors_reached", count=self.consecutive_errors)
            # Optionally send a critical alert via Telegram here
            raise Exception(f"System unstable: {self.consecutive_errors} consecutive errors")

    def reset_error_count(self):
        """Reset error count after successful operation."""
        if self.consecutive_errors > 0:
            logger.info("Error count reset", last_count=self.consecutive_errors)
            self.consecutive_errors = 0

    async def restart_exchange_connection(self):
        """Recovery strategy: restart exchange connections."""
        await exchange_factory.close_all()
        await asyncio.sleep(2)

    async def clear_cache(self):
        """Recovery strategy: clear Redis cache."""
        if redis_client:
            await redis_client.flushdb()
            logger.info("Redis cache flushed")

    async def garbage_collect(self):
        """Recovery strategy: force garbage collection."""
        memory_manager.cleanup()

    async def reset_risk_managers(self):
        """Recovery strategy: reset risk manager state."""
        risk_manager.open_positions.clear()
        advanced_risk_manager.open_positions_usd.clear()
        logger.info("Risk managers have been reset")

# ------------------------------------------------------------------------------
# NEW: LEVERAGE CALCULATION & ADVANCED INDICATORS
# ------------------------------------------------------------------------------
def calculate_leverage(entry_price: float, sl_price: float, max_leverage: int = 20) -> int:
    """
    Suggests an appropriate leverage based on the stop-loss percentage.
    Aims to limit risk and avoid liquidation.
    """
    if entry_price == 0:
        return 1

    sl_percentage = abs(entry_price - sl_price) / entry_price
    if sl_percentage == 0:
        # Avoid division by zero if SL is at entry
        return max_leverage

    # Formula: Leverage = 1 / Stop-Loss Percentage (with a safety factor)
    # A safety factor of 1.2 means we are slightly more conservative.
    suggested_leverage = int(1 / (sl_percentage * 1.2))

    # Ensure the leverage is clamped between 1x and the maximum.
    return max(1, min(suggested_leverage, max_leverage))

def calculate_ichimoku_cloud(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    high_9 = df['high'].rolling(9).max()
    low_9 = df['low'].rolling(9).min()
    df['tenkan_sen'] = (high_9 + low_9) / 2
    high_26 = df['high'].rolling(26).max()
    low_26 = df['low'].rolling(26).min()
    df['kijun_sen'] = (high_26 + low_26) / 2
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    high_52 = df['high'].rolling(52).max()
    low_52 = df['low'].rolling(52).min()
    df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
    df['ichimoku_bull_signal'] = ((df['tenkan_sen'] > df['kijun_sen']) & (df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b'])).astype(int)
    df['ichimoku_bear_signal'] = ((df['tenkan_sen'] < df['kijun_sen']) & (df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b'])).astype(int)
    return df

def calculate_stochastic_rsi(df: pd.DataFrame, period=14, k=3, d=3) -> pd.DataFrame:
    df = df.copy()
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - (100 / (1 + rs))
    min_rsi = rsi.rolling(period).min()
    max_rsi = rsi.rolling(period).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, 1e-9)
    df['stoch_rsi_k'] = stoch_rsi.rolling(k).mean()
    df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(d).mean()
    df['stoch_rsi_bull'] = ((df['stoch_rsi_k'] > df['stoch_rsi_d']) & (df['stoch_rsi_k'].shift(1) <= df['stoch_rsi_d'].shift(1))).astype(int)
    df['stoch_rsi_bear'] = ((df['stoch_rsi_k'] < df['stoch_rsi_d']) & (df['stoch_rsi_k'].shift(1) >= df['stoch_rsi_d'].shift(1))).astype(int)
    return df

def calculate_cci(df: pd.DataFrame, period=20) -> pd.DataFrame:
    df = df.copy()
    tp = (df['high'] + df['low'] + df['close']) / 3
    tp_sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: (x - x.mean()).abs().mean())
    df['cci'] = (tp - tp_sma) / (0.015 * mad.replace(0, 1e-9))
    df['cci_bull_signal'] = ((df['cci'] > -100) & (df['cci'].shift(1) <= -100)).astype(int)
    df['cci_bear_signal'] = ((df['cci'] < 100) & (df['cci'].shift(1) >= 100)).astype(int)
    return df

def calculate_williams_r(df: pd.DataFrame, period=14) -> pd.DataFrame:
    df = df.copy()
    high_h = df['high'].rolling(period).max()
    low_l = df['low'].rolling(period).min()
    df['williams_r'] = -100 * (high_h - df['close']) / (high_h - low_l).replace(0, 1e-9)
    df['williams_r_bull'] = ((df['williams_r'] > -80) & (df['williams_r'].shift(1) <= -80)).astype(int)
    df['williams_r_bear'] = ((df['williams_r'] < -20) & (df['williams_r'].shift(1) >= -20)).astype(int)  # <-- FIXED
    return df

# ### NEW: Bollinger Bands for Volatility Breakout Strategy ###
def calculate_bollinger_bands(df: pd.DataFrame, period=20, std_dev=2.0) -> pd.DataFrame:
    """Calculates Bollinger Bands and Band Width."""
    df = df.copy()
    df['bb_mid'] = df['close'].rolling(window=period).mean()
    df['bb_std'] = df['close'].rolling(window=period).std()
    df['bb_upper'] = df['bb_mid'] + (df['bb_std'] * std_dev)
    df['bb_lower'] = df['bb_mid'] - (df['bb_std'] * std_dev)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    return df

# ------------------------------------------------------------------------------
# NEW: ENHANCED INDICATOR COMPUTATION
# ------------------------------------------------------------------------------
def compute_base_indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    tr = pd.concat([(df['high'] - df['low']), (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(config.atr_period).mean()
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(window=20).mean()) / (df['volume'].rolling(window=20).std() + 1e-9)
    return df

def compute_enhanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Combines base and advanced indicators based on config."""
    df = compute_base_indicators(df)
    df = calculate_bollinger_bands(df) # BBands are useful for multiple strategies
    df = feature_engineer.compute_market_efficiency_features(df.copy()) # Hurst exponent is needed for dispatch
    if config.enhanced_indicators:
        df = calculate_ichimoku_cloud(df)
        df = calculate_stochastic_rsi(df)
        df = calculate_cci(df)
        df = calculate_williams_r(df)
    return df

# ------------------------------------------------------------------------------
# NEW: ADVANCED MULTI-TIMEFRAME ANALYZER
# ------------------------------------------------------------------------------
class AdvancedMultiTimeframeAnalyzer:
    def __init__(self):
        self.timeframes = {
            '5m': {'weight': 0.05}, '15m': {'weight': 0.15}, '30m': {'weight': 0.20},
            '1h': {'weight': 0.30}, '4h': {'weight': 0.20}, '1d': {'weight': 0.10}
        }

    async def analyze(self, symbol: str, exchange_name: str) -> Dict[str, Any]:
        tasks = [self._analyze_tf(symbol, tf, cfg, exchange_name) for tf, cfg in self.timeframes.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid = [r for r in results if r and not isinstance(r, Exception)]
        if not valid:
            return {'mtf_score': 0.5, 'confidence': 'LOW', 'total_timeframes': 0}

        total_w = sum(r['weight'] for r in valid)
        if total_w == 0:
            return {'mtf_score': 0.5, 'confidence': 'LOW'}

        trend = sum(r['trend'] * r['weight'] for r in valid) / total_w
        momentum = sum(r['momentum'] * r['weight'] for r in valid) / total_w

        score = trend * 0.6 + momentum * 0.4
        return {
            'mtf_score': score,
            'confidence': 'HIGH' if score >= 0.7 else 'MEDIUM' if score >= 0.55 else 'LOW',
            'total_timeframes': len(valid)
        }

    async def _analyze_tf(self, symbol, tf, cfg, exchange):
        try:
            df = await fetch_ohlcv_cached(symbol, tf, 300, exchange)
            if df is None or len(df) < 100:
                return None
            df = compute_enhanced_indicators(df.copy())
            if df.iloc[-1].isna().any():
                return None

            last = df.iloc[-1]
            trend_score = 0.5
            if config.enhanced_indicators and 'ichimoku_bull_signal' in last:
                is_bull = last.get('ichimoku_bull_signal', 0) > 0
                is_bear = last.get('ichimoku_bear_signal', 0) > 0
                trend_score = 1 if last.ema20 > last.ema50 and is_bull else 0 if last.ema20 < last.ema50 and is_bear else 0.5
            else:
                trend_score = 1 if last.ema20 > last.ema50 else 0

            mom_scores = [1 if last.rsi > 50 else 0]
            if config.enhanced_indicators and 'stoch_rsi_k' in last:
                is_bull_stoch = last.get('stoch_rsi_k', 0) > last.get('stoch_rsi_d', 0)
                mom_scores.extend([1 if is_bull_stoch else 0, 1 if last.macd > last.macd_sig else 0])
            momentum_score = sum(mom_scores) / len(mom_scores)

            return {'trend': trend_score, 'momentum': momentum_score, 'weight': cfg['weight']}
        except Exception as e:
            logger.debug("mtf_analysis_failed", symbol=symbol, timeframe=tf, error=str(e))
            return None

advanced_mtf_analyzer = AdvancedMultiTimeframeAnalyzer()


# ------------------------------------------------------------------------------
# NEW: MODULARIZED RISK MANAGERS
# ------------------------------------------------------------------------------
class AdvancedRiskManager: # For dynamic position sizing and portfolio heat
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.open_positions_usd: Dict[str, float] = {}

    async def calculate_dynamic_size(self, confidence: float, mtf_score: float) -> Dict[str, Any]:
        if config.kelly_criterion_enabled:
            win_rate = 0.5 + (confidence - 0.5) * 0.25
            rr_ratio = 1.5 + (mtf_score - 0.5) * 1.0
            kelly_f = win_rate - ((1 - win_rate) / rr_ratio)
            risk_fraction = max(0, kelly_f) * 0.5 # Half-Kelly
        else:
            risk_fraction = config.max_position_size * confidence

        capped_risk = min(risk_fraction, config.max_position_size)
        position_size_usd = self.portfolio_value * capped_risk

        current_heat = sum(self.open_positions_usd.values()) / self.portfolio_value
        available_heat_usd = self.portfolio_value * (config.max_portfolio_heat - current_heat)

        final_size_usd = max(self.portfolio_value * 0.005, min(position_size_usd, available_heat_usd))

        return {'position_size_usd': final_size_usd}

    def add_position(self, symbol: str, size_usd: float):
        self.open_positions_usd[symbol] = size_usd

    def remove_position(self, symbol: str):
        self.open_positions_usd.pop(symbol, None)

advanced_risk_manager = AdvancedRiskManager(config.portfolio_value)


class EnhancedRiskManager: # For trade permissions (daily loss, correlation etc.)
    def __init__(self, portfolio_value: float):
        self.portfolio_value = portfolio_value
        self.daily_pnl = 0.0
        self.max_daily_loss_usd = portfolio_value * config.max_daily_loss
        self.open_positions: Dict[str, float] = {}

    def add_position(self, symbol: str, size_usd: float):
        self.open_positions[symbol] = size_usd

    def remove_position(self, symbol: str):
        self.open_positions.pop(symbol, None)

    async def can_open_trade(self, symbol: str) -> Tuple[bool, str]:
        if self.daily_pnl <= -self.max_daily_loss_usd:
            return False, "Daily loss limit exceeded"
        if len(self.open_positions) >= config.max_concurrent_trades:
            return False, "Max concurrent trades reached"
        if symbol in self.open_positions:
            return False, "Position already open for this symbol"

        is_safe, reason = await self.check_portfolio_correlation(symbol)
        if not is_safe:
            return False, reason
        return True, "OK"

    async def check_portfolio_correlation(self, new_symbol: str) -> Tuple[bool, str]:
        open_symbols = list(self.open_positions.keys())
        if not open_symbols:
            return True, "OK"
        try:
            symbols = list(set(open_symbols + [new_symbol]))
            tasks = [fetch_ohlcv_cached(s, '4h', 100, 'binance') for s in symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            price_data = {}
            for s, r in zip(symbols, results):
                if r is not None and not isinstance(r, Exception) and not r.empty:
                    price_data[s] = r['close']

            if new_symbol not in price_data or len(price_data) < 2:
                return True, "Not enough data for correlation"

            df = pd.DataFrame(price_data).ffill().dropna()
            if len(df) < 20:
                return True, "Not enough overlapping data for correlation"

            for existing in open_symbols:
                if existing in df.columns:
                    try:
                        corr, _ = pearsonr(df[new_symbol], df[existing])
                        if corr > config.correlation_threshold:
                            return False, f"High correlation ({corr:.2f}) with {existing}"
                    except Exception:
                        continue
            return True, "OK"
        except Exception as e:
            logger.error("correlation_check_failed", error=str(e))
            return True, "Correlation check failed, proceeding cautiously"

    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl

    def reset_daily_pnl(self):
        self.daily_pnl = 0.0

risk_manager = EnhancedRiskManager(config.portfolio_value)

# ------------------------------------------------------------------------------
# Exchange, DB, Caching, ML Model, and Other Utilities
# ------------------------------------------------------------------------------
class ExchangeFactory:
    def __init__(self):
        self.exchanges: Dict[str, Any] = {}

    async def get_exchange(self, name: str = 'binance', use_futures: bool = True):
        cache_key = f"{name}{'_futures' if use_futures else ''}"
        if cache_key in self.exchanges:
            return self.exchanges[cache_key]
        exchange_class = getattr(ccxt_async, 'binanceusdm' if use_futures and name == 'binance' else name)
        ex = exchange_class({
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {'defaultType': 'future' if use_futures else 'spot'}
        })
        try:
            await ex.load_markets()
            self.exchanges[cache_key] = ex
            return ex
        except Exception as e:
            await ex.close()
            raise e

    async def close_all(self):
        for ex in self.exchanges.values():
            await ex.close()
        self.exchanges.clear()

# Global exchange factory
exchange_factory = ExchangeFactory()

# Database Schema
metadata = MetaData()

signals_table = Table(
    'signals', metadata,
    Column('id', Integer, primary_key=True, autoincrement=True),
    Column('market_id', String(20), nullable=False),
    Column('symbol', String(20), nullable=False),
    Column('direction', String(10), nullable=False),
    Column('entry_price', Float, nullable=False),
    Column('entry_time', DateTime, nullable=False),
    Column('tp1', Float, nullable=False),
    Column('tp2', Float, nullable=False),
    Column('tp3', Float, nullable=False),
    Column('sl', Float, nullable=False),
    Column('position_size', Float, nullable=False),
    Column('confidence', Float, default=0.5),
    Column('market_regime', String(20), default='unknown'),
    Column('strategy_name', String(30), default='unknown'),
    Column('status', String(10), default='open'),
    Column('exit_price', Float, nullable=True),
    Column('exit_time', DateTime, nullable=True),
    Column('exit_reason', String(50), nullable=True),
    Column('pnl', Float, default=0.0),
    Column('tp1_hit', Integer, default=0),
    Column('tp2_hit', Integer, default=0),
    Column('tp3_hit', Integer, default=0),
    Index('idx_symbol_status', 'symbol', 'status'),
    Index('idx_entry_time', 'entry_time'),
)


# Database Engine
engine = create_async_engine(
    config.db_url,
    poolclass=AsyncAdaptedQueuePool,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    echo=False
)

# Redis Setup (Optional)
redis_client = None
if aioredis and config.redis_url:
    try:
        redis_client = aioredis.from_url(
            config.redis_url,
            decode_responses=False,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
    except Exception as e:
        logger.warning("Redis connection failed", error=str(e))
        redis_client = None

async def init_database():
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise

# ------------------------------------------------------------------------------
# NEW: UPDATED HYBRID MODEL CLASS
# ------------------------------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, input_size=50, hidden_size=128, num_layers=2, nhead=8):
        super(HybridModel, self).__init__()
        # Use the exact same layer names as in your saved model
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers, batch_first=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers
        )
        self.classifier = nn.Linear(hidden_size, 3)  # output: Buy, Sell, Hold

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        gru_out, _ = self.gru(x)                 # (batch, seq_len, hidden)
        transformer_out = self.transformer(gru_out)  # (batch, seq_len, hidden)
        last = transformer_out[:, -1, :]             # (batch, hidden)
        return self.classifier(last)                 # (batch, 3)

def load_hybrid_model(path: str, device: torch.device) -> Optional[HybridModel]:
    """
    Loads a HybridModel from checkpoint with robust error handling.
    """
    if not PYTORCH_AVAILABLE:
        logger.warning("PyTorch not available, cannot load hybrid model")
        return None
        
    if not os.path.exists(path):
        logger.warning(f"Hybrid model file not found: {path}")
        return None

    try:
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            config_dict = checkpoint.get("model_config", {})
        else:
            state_dict = checkpoint
            config_dict = {}
            
        # Extract model parameters
        input_size = config_dict.get("input_size", 50)
        hidden_size = config_dict.get("hidden_size", 128)
        num_layers = config_dict.get("num_layers", 2)
        nhead = config_dict.get("nhead", 8)
        
        # Create model
        model = HybridModel(input_size, hidden_size, num_layers, nhead)
        
        # Load state dict with flexible key matching
        try:
            model.load_state_dict(state_dict, strict=True)
            logger.info("Hybrid model loaded successfully with strict loading")
        except Exception as e:
            logger.warning(f"Strict loading failed, trying flexible loading: {e}")
            # Try flexible loading
            model_dict = model.state_dict()
            
            # 1. Try direct mapping
            pretrained_dict = {k: v for k, v in state_dict.items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            # 2. If that fails, try key renaming
            if len(pretrained_dict) == 0:
                key_mapping = {
                    'recurrent_layer.': 'gru.',
                    'transformer_encoder.': 'transformer.',
                    'fc.': 'classifier.'
                }
                for old_key, new_key in key_mapping.items():
                    pretrained_dict.update({k.replace(old_key, new_key): v 
                                          for k, v in state_dict.items() 
                                          if old_key in k})
            
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            
            loaded_keys = set(pretrained_dict.keys())
            expected_keys = set(model_dict.keys())
            missing_keys = expected_keys - loaded_keys
            unexpected_keys = set(state_dict.keys()) - set(pretrained_dict.keys())
            
            if missing_keys:
                logger.warning(f"Missing keys after flexible loading: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys after flexible loading: {unexpected_keys}")
        
        model.to(device)
        model.eval()
        logger.info(f"HybridModel loaded from {path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load hybrid model: {e}")
        return None
    
@with_retry(max_retries=5, delay=2.0, backoff=2.0)
async def fetch_ohlcv_cached(symbol: str, timeframe: str, limit: int, exchange_name: str) -> Optional[pd.DataFrame]:
    cache_key = f"ohlcv:{exchange_name}:{symbol}:{timeframe}:{limit}"

    # Check cache first
    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.warning("cache_read_failed", error=str(e))

    # Fetch from exchange
    try:
        ex = await exchange_factory.get_exchange(exchange_name, use_futures=True)
        bars = await ex.fetch_ohlcv(symbol, timeframe, limit=limit)

        if not bars or len(bars) < 20:
            logger.warning("insufficient_ohlcv_data", symbol=symbol, timeframe=timeframe, count=len(bars) if bars else 0)
            return None

        df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        
        # FIX: Ensure the index is unique before setting it to prevent indicator calculation errors
        df = df.drop_duplicates(subset='ts')

        df = df.set_index("ts")

        # Cache the result
        if redis_client:
            try:
                await redis_client.set(cache_key, pickle.dumps(df), ex=config.cache_ttl)
            except Exception as e:
                logger.warning("cache_write_failed", error=str(e))

        return df

    except Exception as e:
        logger.error("fetch_ohlcv_failed", symbol=symbol, error=str(e))
        raise

class PerformanceTracker:
    def __init__(self):
        self.today = date.today().isoformat()
        self.filename = f"performance_{self.today}.csv"
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'symbol', 'direction', 'entry', 'exit', 'pnl_percent', 'confidence', 'duration', 'market_regime'])

    def record_trade(self, trade_data: Dict[str, Any]):
        today_str = date.today().isoformat()
        if self.today != today_str:
            self.today = today_str
            self.filename = f"performance_{self.today}.csv"
            self._init_file()
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_data.get('timestamp'), trade_data.get('symbol'), trade_data.get('direction'),
                trade_data.get('entry_price'), trade_data.get('exit_price'), trade_data.get('pnl_percent'),
                trade_data.get('confidence'), trade_data.get('duration_minutes'), trade_data.get('market_regime')
            ])

performance_tracker = PerformanceTracker()

def initialize_xai_analyzer(ensemble_model: DynamicEnsembleModel) -> Optional[XAIAnalyzer]:
    """Safely initialize XAI analyzer with proper error handling"""
    if not config.use_xai_explanations:
        return None
        
    if not ensemble_model.models:
        logger.warning("No models in ensemble, cannot initialize XAI")
        return None
        
    analyzer = XAIAnalyzer(ensemble_model)
    if analyzer.explainer is None:
        logger.warning("XAI analyzer could not be initialized, disabling XAI features")
        return None
        
    return analyzer

rl_agent: Optional[RLAgent] = None

# ------------------------------------------------------------------------------
# MODULAR STRATEGY DEFINITIONS
# ------------------------------------------------------------------------------
async def generate_trend_momentum_signal(market_id: str, df: pd.DataFrame, analyzer: 'ScanAnalyzer') -> Optional[Dict[str, Any]]:
    """Trend and momentum strategy with a '2 out of 3' logic for higher signal frequency."""
    if len(df) < 3: return None
    last = df.iloc[-2]

    ema_bullish = last.ema20 > last.ema50
    ema_bearish = last.ema20 < last.ema50
    rsi_bullish = last.rsi > 52
    rsi_bearish = last.rsi < 48
    volume_spike = last.volume > df['volume'].rolling(20).mean().iloc[-2] * 1.15
    
    side = None
    bullish_score = sum([ema_bullish, rsi_bullish, volume_spike])
    bearish_score = sum([ema_bearish, rsi_bearish, volume_spike])

    if bullish_score >= 2: side = "Long"
    elif bearish_score >= 2: side = "Short"

    if not side: return None

    entry = float(last.close)
    atr = float(last.atr)
    if atr == 0 or pd.isna(atr): return None

    sl_dist = atr * config.sl_mult
    if side == "Long":
        sl = entry - sl_dist
        tps = [entry + sl_dist * m for m in config.tp_mult]
    else: # Short
        sl = entry + sl_dist
        tps = [entry - sl_dist * m for m in config.tp_mult]
        
    return {"side": side, "entry": entry, "sl": sl, "tps": tps, "strategy_name": "TrendMomentum_2/3"}

async def generate_mean_reversion_signal(market_id: str, df: pd.DataFrame, analyzer: 'ScanAnalyzer') -> Optional[Dict[str, Any]]:
    """Strategy for mean-reverting markets with wider bands."""
    last = df.iloc[-2]
    side = None
    if last.close < last.bb_lower and last.rsi < 30: side = "Long"
    elif last.close > last.bb_upper and last.rsi > 70: side = "Short"

    if not side: return None
    entry = float(last.close)
    atr = float(last.atr)
    if atr == 0 or pd.isna(atr): return None

    if side == "Long":
        sl = entry - (atr * (config.sl_mult * 0.8))
        tps = [last.bb_mid, last.bb_upper]
    else:
        sl = entry + (atr * (config.sl_mult * 0.8))
        tps = [last.bb_mid, last.bb_lower]

    if len(tps) < 3: tps.append(tps[-1] * (1.005 if side == "Long" else 0.995))
    return {"side": side, "entry": entry, "sl": sl, "tps": tps, "strategy_name": "MeanReversion_Tuned"}

async def generate_volatility_breakout_signal(market_id: str, df: pd.DataFrame, analyzer: 'ScanAnalyzer') -> Optional[Dict[str, Any]]:
    """Strategy for breakouts after low volatility with a more frequent trigger."""
    is_in_squeeze = df['bb_width'][-20:].min() < df['bb_width'][-100:].quantile(0.25)
    if not is_in_squeeze: return None

    last = df.iloc[-2]
    prev = df.iloc[-3]
    side = None
    if prev.close < prev.bb_upper and last.close > last.bb_upper and last.volume_zscore > 0.8: side = "Long"
    elif prev.close > prev.bb_lower and last.close < last.bb_lower and last.volume_zscore > 0.8: side = "Short"

    if not side: return None
    entry = float(last.close)
    atr = float(last.atr)
    if atr == 0 or pd.isna(atr): return None

    sl_dist = atr * config.sl_mult
    if side == "Long":
        sl = min(last.low, last.bb_mid)
        tps = [entry + sl_dist * m for m in config.tp_mult]
    else:
        sl = max(last.high, last.bb_mid)
        tps = [entry - sl_dist * m for m in config.tp_mult]
    return {"side": side, "entry": entry, "sl": sl, "tps": tps, "strategy_name": "VolatilityBreakout_Tuned"}


# --- ### DISPATCHER FUNCTION (REPLACE ENTIRE FUNCTION) ### ---
@track_performance
async def dispatch_strategy_scan(market_id: str, analyzer: 'ScanAnalyzer') -> Optional[Dict[str, Any]]:
    """
    Main signal generation function with corrected ML confidence logic and regime logging.
    """
    start_time = time.time()
    try:
        if any(skip in market_id for skip in config.skip_symbols): return None
        if market_id in risk_manager.open_positions:
            analyzer.add_rejection(market_id, "Position already open")
            return None

        df = await fetch_ohlcv_cached(market_id, config.timeframe, 300, 'binance')
        if df is None or len(df) < 120:
            analyzer.add_rejection(market_id, "Insufficient data")
            return None
        df = compute_enhanced_indicators(df)
        if df.iloc[-2].isna().any():
            analyzer.add_rejection(market_id, "NaN in indicators")
            return None

        base_signal = None
        market_regime_hurst = df['market_regime_hurst'].iloc[-2]
        if config.dynamic_strategy_selection:
            if market_regime_hurst == 'trending':
                base_signal = await generate_trend_momentum_signal(market_id, df, analyzer)
            elif market_regime_hurst == 'mean_reverting':
                base_signal = await generate_mean_reversion_signal(market_id, df, analyzer)
            else: # Efficient
                base_signal = await generate_volatility_breakout_signal(market_id, df, analyzer)
        else:
            base_signal = await generate_trend_momentum_signal(market_id, df, analyzer)

        if not base_signal:
            analyzer.add_rejection(market_id, "No strategy entry condition met")
            return None

        # --- ### ML CONFIDENCE LOGIC ### ---
        conf = 0.5 # Default confidence
        xai_explanation = "N/A"
        if config.use_ml and hybrid_strategy.ensemble_model.is_trained:
            try:
                exchange = await exchange_factory.get_exchange()
                ml_features = await hybrid_strategy.prepare_ml_features(df.copy(), market_id, exchange)
                
                if ml_features is not None and not ml_features.empty:
                    # Ensure we have the latest data point
                    recent_features = ml_features.iloc[-1:]
                    
                    # Check if scaler is available
                    if hybrid_strategy.ensemble_model.scaler is not None:
                        conf = hybrid_strategy.calculate_ml_confidence(recent_features)
                        
                        # Apply confidence boosting for strong technical signals
                        if base_signal and base_signal.get('strategy_name') == 'TrendMomentum_2/3':
                            # Boost confidence for trend momentum signals
                            conf = min(1.0, conf * 1.2)
                            
                        if config.use_xai_explanations and xai_analyzer:
                            xai_explanation = xai_analyzer.get_top_contributors(recent_features)
                    else:
                        logger.warning("Scaler not available, using default confidence", symbol=market_id)
                else:
                    logger.warning("ML feature preparation returned empty data", symbol=market_id)
                    
            except Exception as e:
                logger.warning("ML confidence calculation failed", symbol=market_id, error=str(e))

        mtf = await advanced_mtf_analyzer.analyze(market_id, 'binance')
        if mtf['mtf_score'] < config.base_require_mtf_score or conf < config.base_confidence_floor:
            analyzer.add_rejection(market_id, f"Quality fail (Conf:{conf:.2f} MTF:{mtf['mtf_score']:.2f})")
            return None

        can_trade, reason = await risk_manager.can_open_trade(market_id)
        if not can_trade:
            analyzer.add_rejection(market_id, f"Risk Manager: {reason}")
            return None

        pos_value = (await advanced_risk_manager.calculate_dynamic_size(conf, mtf['mtf_score']))['position_size_usd']
        if pos_value < 10:
            analyzer.add_rejection(market_id, f"Position size too small (${pos_value:.2f})")
            return None
        pos_coin = pos_value / base_signal['entry']
        
        final_signal = {
            **base_signal, "symbol": market_id, "market_id": market_id,
            "confidence": conf, "position_size_coin": pos_coin, "position_value": pos_value,
            "mtf_score": mtf['mtf_score'], "scan_time": datetime.now(timezone.utc),
            "market_regime": market_regime_hurst, "xai_explanation": xai_explanation
        }
        return final_signal

    except Exception as e:
        logger.error("dispatch_strategy_scan_failed", symbol=market_id, error=str(e), exc_info=True)
        analyzer.add_rejection(market_id, f"Strategy Dispatch Exception: {e}")
        return None
    finally:
        performance_monitor.record_metric('dispatch_strategy_duration_seconds', time.time() - start_time)

# ------------------------------------------------------------------------------
# Scan Analyzer & Main Loops
# ------------------------------------------------------------------------------
class ScanAnalyzer:
    def __init__(self):
        self.rejections = {}
        self.scan_results = []

    def add_rejection(self, market_id: str, reason: str):
        self.rejections[market_id] = reason
        logger.debug("market_rejected", market=market_id, reason=reason)

    def add_signal(self, signal_data: Dict):
        self.scan_results.append(signal_data)

    def get_summary(self) -> Dict:
        return {
            'total_scanned': len(self.rejections) + len(self.scan_results),
            'signals_found': len(self.scan_results),
            'rejections': len(self.rejections),
            'rejection_reasons': self.rejections
        }

@track_performance
async def scan_markets(context: Optional[ContextTypes.DEFAULT_TYPE] = None, exchange_name: str = 'binance'):
    if IS_PAUSED:
        logger.info("Scanning paused, skipping market scan")
        return

    start_time = time.time()
    try:
        logger.info("market_scan_started")
        analyzer = ScanAnalyzer()

        ex = await exchange_factory.get_exchange(exchange_name, use_futures=True)
        markets = await ex.fetch_tickers()

        usdt_pairs = {k: v for k, v in markets.items() if '/USDT' in k and v.get('quoteVolume', 0) > 0 and v.get('active', True)}
        sorted_pairs = sorted(usdt_pairs.items(), key=lambda x: x[1].get('quoteVolume', 0), reverse=True)[:config.top_n_markets]

        logger.info(f"Scanning top {len(sorted_pairs)} USDT pairs")

        for symbol, ticker in sorted_pairs:
            if len(risk_manager.open_positions) >= config.max_concurrent_trades:
                logger.info("Max concurrent trades reached, ending scan early.")
                break

            if any(skip in symbol for skip in config.skip_symbols):
                analyzer.add_rejection(symbol, "Skipped symbol")
                continue

            try:
                signal = await dispatch_strategy_scan(symbol, analyzer)

                if signal:
                    entry_time = datetime.now(timezone.utc)
                    signal['scan_time'] = entry_time

                    async with engine.begin() as conn:
                        result = await conn.execute(
                            signals_table.insert().values(
                                market_id=symbol, symbol=symbol, direction=signal['side'].lower(),
                                entry_price=signal['entry'], entry_time=entry_time,
                                tp1=signal['tps'][0], tp2=signal['tps'][1], tp3=signal['tps'][2],
                                sl=signal['sl'], position_size=signal['position_size_coin'],
                                confidence=signal['confidence'], market_regime=signal.get('market_regime', 'unknown'),
                                strategy_name=signal.get('strategy_name', 'unknown'), status='open'
                            )
                        )
                        signal_id = result.lastrowid

                    risk_manager.add_position(symbol, signal['position_value'])
                    advanced_risk_manager.add_position(symbol, signal['position_value'])
                    analyzer.add_signal(signal)

                    if context:
                        df_for_chart = await fetch_ohlcv_cached(symbol, config.timeframe, 200, exchange_name)
                        chart_path = ""
                        if df_for_chart is not None:
                            df_for_chart = compute_enhanced_indicators(df_for_chart)
                            chart_path = await plotter.plot_annotated_chart(df_for_chart, symbol, signal['entry'], signal['sl'], signal['tps'])
                        
                        alert_text = await format_crypto_quant_alert(signal, ex)
                        await send_alert(context, {'text': alert_text, 'chart': chart_path, 'signal_id': signal_id})

                await asyncio.sleep(0.2)

            except Exception as e:
                analyzer.add_rejection(symbol, f"Scan error: {str(e)}")
                logger.error("market_scan_error", symbol=symbol, error=str(e), exc_info=True)
                health_monitor.record_error()

        summary = analyzer.get_summary()
        logger.info("market_scan_completed", **summary)
        memory_manager.guard()
        
        # Record successful scan
        health_monitor.record_scan_result(True)

    except Exception as e:
        logger.error("scan_markets_failed", error=str(e), exc_info=True)
        health_monitor.record_scan_result(False)
        health_monitor.record_error()
        await error_recovery.handle_error(e, context="scan_markets")
    finally:
        performance_monitor.record_metric('scan_duration_seconds', time.time() - start_time)

@track_performance
async def monitor_positions(context: Optional[ContextTypes.DEFAULT_TYPE] = None, exchange_name: str = 'binance'):
    start_time = time.time()
    try:
        async with engine.begin() as conn:
            rows = (await conn.execute(sql_text("SELECT * FROM signals WHERE status='open'"))).fetchall()

        if not rows: return

        ex = await exchange_factory.get_exchange(exchange_name, use_futures=True)
        symbols = [r.symbol for r in rows]
        if not symbols: return
        tickers = await ex.fetch_tickers(symbols)

        for r in rows:
            try:
                ticker = tickers.get(r.symbol)
                if not ticker or not ticker.get('last'): continue
                price = float(ticker['last'])

                position_closed, exit_reason = False, ""

                if r.direction == 'long':
                    if price <= r.sl: position_closed, exit_reason = True, "Stop Loss"
                    elif not r.tp3_hit and price >= r.tp3: position_closed, exit_reason = True, "TP3"
                    elif not r.tp2_hit and price >= r.tp2:
                        async with engine.begin() as conn:
                            await conn.execute(sql_text("UPDATE signals SET tp2_hit=1 WHERE id=:id"), {"id": r.id})
                        if context: await send_notification(context, f"🎯 TP2 Hit for {r.symbol}!")
                    elif not r.tp1_hit and price >= r.tp1:
                        async with engine.begin() as conn:
                            await conn.execute(sql_text("UPDATE signals SET tp1_hit=1, sl=:new_sl WHERE id=:id"), {"id": r.id, "new_sl": r.entry_price})
                        if context: await send_notification(context, f"✅ TP1 Hit for {r.symbol}! SL moved to Breakeven.")
                else: # Short
                    if price >= r.sl: position_closed, exit_reason = True, "Stop Loss"
                    elif not r.tp3_hit and price <= r.tp3: position_closed, exit_reason = True, "TP3"
                    elif not r.tp2_hit and price <= r.tp2:
                        async with engine.begin() as conn:
                            await conn.execute(sql_text("UPDATE signals SET tp2_hit=1 WHERE id=:id"), {"id": r.id})
                        if context: await send_notification(context, f"🎯 TP2 Hit for {r.symbol}!")
                    elif not r.tp1_hit and price <= r.tp1:
                        async with engine.begin() as conn:
                            await conn.execute(sql_text("UPDATE signals SET tp1_hit=1, sl=:new_sl WHERE id=:id"), {"id": r.id, "new_sl": r.entry_price})
                        if context: await send_notification(context, f"✅ TP1 Hit for {r.symbol}! SL moved to Breakeven.")

                if not position_closed and config.trailing_stop_loss_enabled and r.tp1_hit == 1:
                    new_sl = 0.0
                    if r.direction == 'long':
                        potential_sl = price * (1 - config.trailing_stop_loss_percent)
                        if potential_sl > r.sl: new_sl = potential_sl
                    else:
                        potential_sl = price * (1 + config.trailing_stop_loss_percent)
                        if potential_sl < r.sl: new_sl = potential_sl
                    if new_sl > 0:
                        async with engine.begin() as conn:
                            await conn.execute(sql_text("UPDATE signals SET sl=:new_sl WHERE id=:id"), {"id": r.id, "new_sl": new_sl})
                        logger.info("trailing_stop_loss_updated", symbol=r.symbol, old_sl=r.sl, new_sl=new_sl)

                if position_closed:
                    pnl = (price - r.entry_price) * r.position_size if r.direction == 'long' else (r.entry_price - price) * r.position_size
                    exit_time = datetime.now(timezone.utc)
                    async with engine.begin() as conn:
                        await conn.execute(sql_text("UPDATE signals SET status='closed', exit_price=:p, exit_time=:t, pnl=:pl, exit_reason=:reason WHERE id=:id"),
                                           {"p": price, "t": exit_time, "pl": pnl, "reason": exit_reason, "id": r.id})
                    risk_manager.remove_position(r.symbol)
                    advanced_risk_manager.remove_position(r.symbol)
                    risk_manager.update_daily_pnl(pnl)
                    pnl_pct = (pnl / (r.entry_price * r.position_size)) * 100 if r.entry_price > 0 and r.position_size > 0 else 0.0
                    entry_dt = r.entry_time
                    if isinstance(entry_dt, str): entry_dt = parser.parse(entry_dt)
                    if entry_dt.tzinfo is None: entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                    duration = (exit_time - entry_dt).total_seconds() / 60
                    performance_tracker.record_trade({'timestamp': exit_time.isoformat(), 'symbol': r.symbol, 'direction': r.direction, 'entry_price': r.entry_price,
                                                      'exit_price': price, 'pnl_percent': pnl_pct, 'confidence': r.confidence, 'duration_minutes': duration, 'market_regime': r.market_regime})
                    
                    # Record successful trade if profitable
                    if pnl > 0:
                        health_monitor.record_successful_trade()
                        
                    if context:
                        await send_notification(context, f"🔴 Closed {r.symbol} ({exit_reason}). PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            except Exception as e:
                logger.error("monitor_error_single_pos", signal_id=r.id, error=str(e))
                health_monitor.record_error()
    except Exception as e:
        logger.error("monitor_loop_error", error=str(e))
        health_monitor.record_error()
        await error_recovery.handle_error(e, context="monitor_positions")
    finally:
        performance_monitor.record_metric('monitor_duration_seconds', time.time() - start_time)

# ------------------------------------------------------------------------------
# Telegram Bot & Main Application
# ------------------------------------------------------------------------------
application: Optional[Application] = None

# --- FIXED: THIS IS THE CORRECTED FUNCTION ---
async def format_crypto_quant_alert(signal: Dict, exchange) -> str:
    """Formats the alert in the CRYPTO-QUANT ADVISORY style with proper HTML escaping."""

    # Generate timestamp in IST
    ist_time = signal['scan_time'].astimezone(config.report_timezone)
    timestamp_str = ist_time.strftime("%d-%m-%Y %H:%M:%S IST")

    # Calculate position value
    position_value = signal.get('position_value', signal['position_size_coin'] * signal['entry'])

    # Direction display
    direction_display = "LONG (Bullish)" if signal['side'].lower() == 'long' else "SHORT (Bearish)"

    # Get ticker info for volume and 24h change
    try:
        ticker_info = await exchange.fetch_ticker(signal['symbol'])
    except Exception as e:
        logger.warning(f"Could not fetch ticker info for {signal['symbol']}: {e}")
        ticker_info = {}

    volume_24h = ticker_info.get('quoteVolume', 0)
    change_24h = ticker_info.get('percentage', 0)

    # Calculate percentages
    sl_percentage_raw = abs(signal['entry'] - signal['sl']) / signal['entry'] if signal['entry'] > 0 else 0
    sl_percentage_display = sl_percentage_raw * 100

    tp2_percentage = abs(signal['tps'][1] - signal['entry']) / signal['entry'] if signal['entry'] > 0 else 0
    risk_reward_ratio = tp2_percentage / sl_percentage_raw if sl_percentage_raw > 0 else 0

    # Portfolio risk calculation
    portfolio_risk = (position_value * sl_percentage_raw / config.portfolio_value * 100) if config.portfolio_value > 0 else 0

    # Calculate suggested leverage
    leverage = calculate_leverage(signal['entry'], signal['sl'])

    # Generate TP percentages for display
    tp1_pct = (abs(signal['tps'][0] - signal['entry']) / signal['entry'] * 100) if signal['entry'] > 0 else 0
    tp2_pct = (abs(signal['tps'][1] - signal['entry']) / signal['entry'] * 100) if signal['entry'] > 0 else 0
    tp3_pct = (abs(signal['tps'][2] - signal['entry']) / signal['entry'] * 100) if signal['entry'] > 0 else 0

    # XAI Explanation
    xai_explanation = signal.get('xai_explanation', 'N/A')

    # Build the clean alert message (exactly matching the requested format)
    alert = f"""🎯 🚨 <b>CRYPTO-QUANT ALERT</b> 🚨 🎯 

⚡ <b>TRADE SIGNAL</b> ⚡
┌ <b>Asset:</b> <code>{signal['symbol']}</code>
├ <b>Direction:</b> {direction_display} {'📈' if direction_display == 'LONG' else '📉'}
├ <b>Strategy:</b> <code>{signal.get('strategy_name', 'Unknown')}</code>
└ <b>Timeframe:</b> <code>{config.timeframe}</code>

📊 <b>MARKET ANALYSIS</b>
┌ <b>Market Regime:</b> <code>{signal.get('market_regime', 'N/A')}</code>
├ <b>24h Volume:</b> <code>${volume_24h:,.0f}</code> 💰
├ <b>24h Change:</b> <code>{change_24h:+.2f}%</code> {'🟢' if change_24h > 0 else '🔴'}
└ <b>MTF Score:</b> <code>{signal.get('mtf_score', 0):.2f}</code> ⭐

🤖 <b>AI INSIGHTS</b>
┌ <b>Prediction Factors:</b> <code>{xai_explanation}</code>
└ <b>Confidence Score:</b> <code>{signal.get('confidence', 0.5)*100:.1f}%</code> 🎯

💵 <b>ORDER DETAILS</b>
┌ <b>Entry Price:</b> <code>{signal['entry']:.5f}</code>
├ <b>Stop Loss:</b> <code>{signal['sl']:.5f}</code> (<code>🚫 -{sl_percentage_display:.2f}%</code>)
├ <b>Risk/Reward:</b> <code>1:{risk_reward_ratio:.2f}</code> ⚖️
└ <b>Leverage:</b> <code>{leverage}x</code> 🚀

🎯 <b>TARGET LEVELS</b>
┌ <b>TP1:</b> <code>{signal['tps'][0]:.5f}</code> (<code>🟢 +{tp1_pct:.2f}%</code>)
├ <b>TP2:</b> <code>{signal['tps'][1]:.5f}</code> (<code>🟡 +{tp2_pct:.2f}%</code>)
└ <b>TP3:</b> <code>{signal['tps'][2]:.5f}</code> (<code>🔴 +{tp3_pct:.2f}%</code>)

💰 <b>POSITION MANAGEMENT</b>
┌ <b>Size ({signal['symbol'].split('/')[0]}):</b> <code>{signal['position_size_coin']:.4f}</code>
├ <b>Value (USDT):</b> <code>${position_value:,.2f}</code> 💎
└ <b>Portfolio Risk:</b> <code>{portfolio_risk:.2f}%</code> ⚠️

⏰ <i>Generated: {timestamp_str}</i>
📝 <i>Disclaimer: For informational purposes only.</i>"""

    return alert

def is_authorized(update: Update) -> bool:
    user_id = str(update.effective_user.id)
    return user_id == config.owner_id

# ------------------------------------------------------------------------------
# FIXED: send_alert FUNCTION WITH PROPER GROUP CHAT HANDLING
# ------------------------------------------------------------------------------
async def send_alert(context: ContextTypes.DEFAULT_TYPE, alert_data: dict):
    """Send alert with chart and interactive buttons, using HTML parse mode."""
    kb = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("Move SL to BE", callback_data=f"be:{alert_data['signal_id']}"),
            InlineKeyboardButton("Close Manually", callback_data=f"close:{alert_data['signal_id']}")
        ]
    ])
    chart_path = alert_data.get('chart')
    alert_text = alert_data['text']

    # Use HTML parse mode
    parse_mode = ParseMode.HTML

    try:
        # Send to owner first
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, 'rb') as photo:
                await context.bot.send_photo(
                    chat_id=config.owner_id_int,
                    photo=photo,
                    caption=alert_text,
                    reply_markup=kb,
                    parse_mode=parse_mode
                )
        else:
            await context.bot.send_message(
                chat_id=config.owner_id_int,
                text=alert_text,
                reply_markup=kb,
                parse_mode=parse_mode
            )

        # Enhanced group chat sending with better error handling
        if config.group_chat_id_int:
            try:
                # Small delay to ensure owner message sends first
                await asyncio.sleep(0.5)
                
                if chart_path and os.path.exists(chart_path):
                    with open(chart_path, 'rb') as photo:
                        await context.bot.send_photo(
                            chat_id=config.group_chat_id_int,
                            photo=photo,
                            caption=alert_text,  # No buttons for group
                            parse_mode=parse_mode
                        )
                else:
                    await context.bot.send_message(
                        chat_id=config.group_chat_id_int,
                        text=alert_text,
                        parse_mode=parse_mode
                    )
                logger.info(f"Alert successfully sent to group chat: {config.group_chat_id_int}")
                
            except Exception as group_error:
                logger.error(f"Failed to send alert to group chat {config.group_chat_id_int}: {str(group_error)}")
                # Don't re-raise, just log the error

    except Exception as e:
        logger.error("send_alert_failed", error=str(e))
    finally:
        # Clean up chart file
        if chart_path and os.path.exists(chart_path):
            try:
                os.remove(chart_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup chart file: {cleanup_error}")

# ------------------------------------------------------------------------------
# FIXED: send_notification FUNCTION WITH PROPER GROUP CHAT HANDLING
# ------------------------------------------------------------------------------
async def send_notification(context: ContextTypes.DEFAULT_TYPE, message: str):
    """Send notification to owner and group with better error handling."""
    try:
        # Send to owner
        await context.bot.send_message(
            chat_id=config.owner_id_int, 
            text=message, 
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Send to group chat if configured
        if config.group_chat_id_int:
            try:
                # Small delay
                await asyncio.sleep(0.3)
                await context.bot.send_message(
                    chat_id=config.group_chat_id_int,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN
                )
                logger.info(f"Notification sent to group: {config.group_chat_id_int}")
            except Exception as group_error:
                logger.error(f"Failed to send notification to group {config.group_chat_id_int}: {str(group_error)}")
                
    except Exception as e:
        logger.error("send_notification_failed", error=str(e))

async def send_critical_error_notification(context: ContextTypes.DEFAULT_TYPE, error_message: str):
    """Sends a critical error alert to the bot owner."""
    try:
        message = (f"🚨 <b>CRITICAL BOT ERROR</b> 🚨\n\n"
                   f"The bot has encountered a fatal error and may have stopped working.\n"
                   f"Please check the logs immediately.\n\n"
                   f"<b>Error:</b>\n<code>{error_message}</code>")
        await context.bot.send_message(chat_id=config.owner_id_int, text=message, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error("failed_to_send_critical_error_notification", error=str(e))

# ------------------------------------------------------------------------------
# NEW: TEST GROUP COMMAND FOR DEBUGGING
# ------------------------------------------------------------------------------
async def test_group_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Test command to verify group chat functionality."""
    if not is_authorized(update):
        return
        
    try:
        # Test message to owner
        await update.message.reply_text("Testing group chat functionality...")
        
        # Test message to group
        if config.group_chat_id_int:
            await context.bot.send_message(
                chat_id=config.group_chat_id_int,
                text="🤖 *Bot Group Chat Test* 🤖\n\nThis is a test message to verify group chat functionality.",
                parse_mode=ParseMode.MARKDOWN
            )
            await update.message.reply_text(f"✅ Test message sent to group chat: {config.group_chat_id_int}")
        else:
            await update.message.reply_text("❌ Group chat ID not configured")
            
    except Exception as e:
        error_msg = f"❌ Group chat test failed: {str(e)}"
        await update.message.reply_text(error_msg)
        logger.error("group_chat_test_failed", error=str(e))

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_authorized(update):
        await update.message.reply_text(
            "✅ <b>Adaptive Bot Online.</b>\n\n"
            f"Dynamic Strategies: <code>{config.dynamic_strategy_selection}</code>\n"
            f"XAI Explanations: <code>{config.use_xai_explanations}</code>\n"
            f"RL Agent Active: <code>{config.use_rl_agent}</code>\n\n"
            "Use /status to see current state.",
            parse_mode=ParseMode.HTML
        )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    heat = advanced_risk_manager.open_positions_usd
    heat_pct = sum(heat.values()) / config.portfolio_value if config.portfolio_value > 0 else 0
    msg = (f"<b>📊 Bot Status</b>\n\n"
           f"<b>Mode</b>: {'⏸️ Paused' if IS_PAUSED else '▶️ Running'}\n"
           f"<b>Open Trades</b>: {len(risk_manager.open_positions)} / {config.max_concurrent_trades}\n"
           f"<b>Daily PnL</b>: <code>${risk_manager.daily_pnl:,.2f}</code>\n"
           f"<b>Portfolio Heat</b>: <code>${sum(heat.values()):,.2f}</code> (<code>{heat_pct:.1%}</code> of max <code>{config.max_portfolio_heat*100:.0f}%</code>)\n"
           f"<b>Consecutive Errors</b>: <code>{error_recovery.consecutive_errors}</code>")
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def pause_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    global IS_PAUSED
    IS_PAUSED = True
    await update.message.reply_text("⏸️ Scanning paused.")

async def resume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    global IS_PAUSED
    IS_PAUSED = False
    await update.message.reply_text("▶️ Scanning resumed.")

async def positions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    async with engine.begin() as conn:
        rows = (await conn.execute(sql_text("SELECT symbol, direction, entry_price, sl, strategy_name FROM signals WHERE status='open'"))).fetchall()
    if not rows:
        await update.message.reply_text("No open positions.")
    else:
        lines = ["<b>Open Positions:</b>\n"]
        for r in rows:
            lines.append(f"- <code>{r.symbol}</code> ({r.direction.upper()}) | <code>{r.strategy_name}</code>\n  E: <code>{r.entry_price:.4f}</code> | SL: <code>{r.sl:.4f}</code>")
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)

async def scan_now_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return    
    await update.message.reply_text("⏳ Manual scan initiated with adaptive strategies...")
    await scan_markets(context)
    await update.message.reply_text("✅ Manual scan complete.")

async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    try:
        ex = await exchange_factory.get_exchange()
        await ex.fetch_time()
        ex_status = "Connected"
    except:
        ex_status = "Disconnected"
        health_monitor.record_error()
        
    redis_status = "N/A"
    if redis_client:
        try:
            await redis_client.ping()
            redis_status = "Connected"
        except:
            redis_status = "Disconnected"
            health_monitor.record_error()
            
    # Get health status
    health_status = health_monitor.get_health_status()
    status_emoji = {
        'HEALTHY': '✅',
        'WARNING': '⚠️', 
        'CRITICAL': '🔴'
    }
    
    mem = psutil.virtual_memory().percent
    msg = (f"{status_emoji[health_status['status']]} <b>Health Status:</b> {health_status['status']}\n\n"
           f"Memory Usage: <code>{health_status['memory_usage']:.1f}%</code>\n"
           f"CPU Usage: <code>{health_status['cpu_usage']:.1f}%</code>\n"
           f"Errors (1h): <code>{health_status['errors_last_hour']}</code>\n"
           f"Scan Success Rate: <code>{health_status['scan_success_rate']:.1%}</code>\n"
           f"Last Successful Scan: <code>{health_status['last_successful_scan'] or 'Never'}</code>\n"
           f"Last Successful Trade: <code>{health_status['last_successful_trade'] or 'Never'}</code>\n\n"
           f"<b>Exchange:</b> <code>{ex_status}</code>\n"
           f"<b>Database:</b> <code>{engine.url.drivername}</code>\n"
           f"<b>Redis:</b> <code>{redis_status}</code>\n"
           f"<b>Bot Status:</b> <code>{'Paused' if IS_PAUSED else 'Running'}</code>")

    if health_status.get('warning'):
        msg += f"\n⚠️ Warning: {health_status['warning']}"
    if health_status.get('critical'):
        msg += f"\n🔴 Critical: {health_status['critical']}"
            
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def performance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    summary = performance_monitor.get_summary()
    msg = f"<b>🚀 Performance Metrics</b>\n\nUptime: <code>{summary.get('uptime_hours', 0):.2f} hours</code>\n\n"
    for name, data in summary.items():
        if name == 'uptime_hours': continue
        msg += (f"<b>{name.replace('_', ' ').title()}:</b>\n"
                f"  - Avg: <code>{data['avg']:.4f}</code>\n  - Min: <code>{data['min']:.4f}</code>\n"
                f"  - Max: <code>{data['max']:.4f}</code>\n  - Last: <code>{data['last']:.4f}</code>\n"
                f"  - Count: <code>{data['count']}</code>\n")
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    if not query or not is_authorized(update): return
    await query.answer()
    try:
        action, signal_id_str = query.data.split(':')
        signal_id = int(signal_id_str)
        async with engine.begin() as conn:
            if action == 'be':
                res = await conn.execute(sql_text("UPDATE signals SET sl = entry_price WHERE id = :id AND status = 'open'"), {"id": signal_id})
                msg = "SL moved to Breakeven!" if res.rowcount else "Signal not found or already closed."
            elif action == 'close':
                signal_row = (await conn.execute(sql_text("SELECT symbol FROM signals WHERE id = :id AND status = 'open'"), {"id": signal_id})).first()
                if signal_row:
                    exit_time = datetime.now(timezone.utc)
                    res = await conn.execute(sql_text("UPDATE signals SET status = 'closed', exit_reason = 'Manual Close', exit_time = :exit_time WHERE id = :id AND status = 'open'"), {"id": signal_id, "exit_time": exit_time})
                    if res.rowcount:
                        risk_manager.remove_position(signal_row.symbol)
                        advanced_risk_manager.remove_position(signal_row.symbol)
                        msg = "Position marked for manual closing."
                    else:
                        msg = "Failed to mark position for closing."
                else:
                    msg = "Signal not found or already closed."
            else:
                msg = "Unknown action."
        await query.edit_message_caption(caption=(query.message.caption or "") + f"\n\n✅ <b>Action:</b> {msg}", parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error("callback_handler_error", error=str(e))
        await query.edit_message_caption(caption=(query.message.caption or "") + f"\n\n❌ <b>Error:</b> {str(e)}", parse_mode=ParseMode.HTML)

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update): return
    chat_id = update.effective_chat.id
    try:
        await context.bot.send_message(chat_id=chat_id, text="⏳ Generating report for the last 24 hours...")
        start_time = datetime.now(timezone.utc) - timedelta(hours=24)
        async with engine.connect() as conn:
            query = sql_text("SELECT * FROM signals WHERE entry_time >= :start_time ORDER BY entry_time DESC")
            result = await conn.execute(query, {"start_time": start_time})
            trades = result.fetchall()
        if not trades:
            await context.bot.send_message(chat_id=chat_id, text="No trades found in the last 24 hours.")
            return
        df = pd.DataFrame(trades, columns=signals_table.columns.keys())
        position_value = df['entry_price'] * df['position_size']
        df['pnl_percent'] = np.where(position_value > 0, (df['pnl'] / position_value) * 100, 0)
        report_df = df[['entry_time', 'symbol', 'strategy_name', 'direction', 'entry_price', 'exit_price', 'sl', 'pnl', 'pnl_percent', 'status', 'exit_reason', 'exit_time']].copy()
        report_df.rename(columns={'entry_time': 'Entry Time (UTC)', 'strategy_name': 'Strategy', 'direction': 'Direction', 'entry_price': 'Entry Price', 'exit_price': 'Exit Price', 'sl': 'Stop Loss', 'pnl': 'PnL ($)', 'pnl_percent': 'PnL (%)', 'status': 'Status', 'exit_reason': 'Exit Reason', 'exit_time': 'Exit Time (UTC)'}, inplace=True)
        report_df['PnL ($)'] = report_df['PnL ($)'].round(2)
        report_df['PnL (%)'] = report_df['PnL (%)'].round(2)
        filename = f"trade_report_{int(time.time())}.xlsx"
        report_df.to_excel(filename, index=False, sheet_name="Trade_Report")
        with open(filename, 'rb') as doc:
            await context.bot.send_document(chat_id=chat_id, document=doc, caption="Here is your 24-hour trade report.")
        os.remove(filename)
    except Exception as e:
        logger.error("Failed to generate report", error=str(e))
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Failed to generate report. Error: {e}")

# ------------------------------------------------------------------------------
# MODEL TRAINING AND MAIN APPLICATION
# ------------------------------------------------------------------------------
async def train_ml_models():
    try:
        logger.info("ml_model_training_started")
        # Placeholder for model training logic
        logger.info("ml_model_training_completed")
    except Exception as e:
        logger.error("ml_training_failed", error=str(e))

async def scheduled_runner(app: Application):
    scan_timer = 0
    while not shutdown_manager.is_shutdown_requested():
        try:
            if shutdown_manager.is_shutdown_requested(): break
            now = datetime.now(config.report_timezone)
            if now.hour == 0 and now.minute < 2 and scan_timer % 3600 < config.monitor_interval:
                risk_manager.reset_daily_pnl()
                logger.info("Daily PnL reset.")
                await asyncio.sleep(60)
            if scan_timer <= 0:
                await scan_markets(app)
                scan_timer = config.scan_interval
            await monitor_positions(app)
            error_recovery.reset_error_count()
            await asyncio.sleep(config.monitor_interval)
            scan_timer -= config.monitor_interval
        except Exception as e:
            await error_recovery.handle_error(e, "scheduled_runner")
            await asyncio.sleep(config.monitor_interval * 2)

async def load_open_positions_from_db():
    logger.info("Attempting to reload open positions from database...")
    open_trade_count = 0
    try:
        async with engine.connect() as conn:
            result = await conn.execute(sql_text("SELECT symbol, entry_price, position_size FROM signals WHERE status='open'"))
            for trade in result.fetchall():
                position_value = trade.entry_price * trade.position_size
                risk_manager.add_position(trade.symbol, position_value)
                advanced_risk_manager.add_position(trade.symbol, position_value)
                open_trade_count += 1
        if open_trade_count > 0:
            logger.info(f"Successfully reloaded {open_trade_count} open positions into memory.")
        else:
            logger.info("No open positions found in the database to reload.")
    except Exception as e:
        logger.error("Failed to load open positions from database", error=str(e))

async def enhanced_main():
    global hybrid_model, application, xai_analyzer, rl_agent
    
    xai_analyzer = None
    
    try:
        with DelayedKeyboardInterrupt():
            print("🚀 Starting Enhanced Crypto Trading Bot...")
            print(f"📊 Timeframe: {config.timeframe} | Portfolio: ${config.portfolio_value:,.2f}")
            await wait_for_network()
            await init_database()
            await load_open_positions_from_db()
            
            try:
                await exchange_factory.get_exchange('binance', use_futures=True)
                logger.info("Exchange factory initialized")
            except Exception as e:
                logger.critical("FATAL: Exchange initialization failed", error=str(e))
                if not SKIP_NETWORK_CHECK: 
                    return
            
            if config.ml_enabled:
                hybrid_strategy.ensemble_model.initialize_models()
                
                if os.path.exists(config.ensemble_model_path):
                    if hybrid_strategy.ensemble_model.load_model(config.ensemble_model_path):
                        logger.info("Pre-trained ensemble model confirmed loaded, initializing XAI.")
                        xai_analyzer = initialize_xai_analyzer(hybrid_strategy.ensemble_model)
                    else:
                        logger.warning("Failed to load or confirm pre-trained ensemble model.")
                else:
                    logger.warning(f"Ensemble model file not found: {config.ensemble_model_path}")
            
            application = Application.builder().token(config.bot_token).build()
            application.add_handler(CommandHandler("start", start_command))
            application.add_handler(CommandHandler("status", status_command))
            application.add_handler(CommandHandler("pause", pause_command))
            application.add_handler(CommandHandler("resume", resume_command))
            application.add_handler(CommandHandler("positions", positions_command))
            application.add_handler(CommandHandler("health", health_command))
            application.add_handler(CommandHandler("scan_now", scan_now_command))
            application.add_handler(CommandHandler("performance", performance_command))
            application.add_handler(CommandHandler("report", report_command))
            application.add_handler(CommandHandler("test_group", test_group_command))  # NEW: Add test group command
            application.add_handler(CallbackQueryHandler(callback_handler))

            print("✅ Adaptive bot initialization completed successfully")
            print(f"- 1H Timeframe Optimized: ✅")
            print(f"- Enhanced ML Features: {config.enhanced_ml_features}")
            print(f"- Dynamic Strategy Selection: {config.dynamic_strategy_selection}")
            print(f"- XAI Explanations: {config.use_xai_explanations and xai_analyzer is not None}")
            print(f"- RL Agent Active: {config.use_rl_agent and rl_agent is not None and rl_agent.model is not None}")
            print(f"- Group Chat Enabled: {config.group_chat_id_int is not None}")

        # Start health monitoring
        monitor_task = asyncio.create_task(
            health_monitor.monitor_loop(
                recovery_callback=perform_cleanup
            )
        )
        
        scheduler = asyncio.create_task(scheduled_runner(application))
        logger.info("adaptive_bot_startup_complete", environment=config.environment, timeframe=config.timeframe)
        async with application:
            await application.start()
            await application.updater.start_polling()
            logger.info("Telegram polling started.")
            await shutdown_manager.shutdown_event.wait()
            print("🛑 Shutdown signal received, stopping application...")
    except KeyboardInterrupt:
        print("🛑 Keyboard interrupt received")
        shutdown_manager.exit_gracefully(signal.SIGINT, None)
    except Exception as e:
        logger.critical("enhanced_main_app_crashed", error=str(e), exc_info=True)
        if application:
            await send_critical_error_notification(application, f"Enhanced main application failed: {e}")
    finally:
        await perform_cleanup()
        print("Enhanced Crypto Trading Bot stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(enhanced_main())
    except KeyboardInterrupt:
        logger.info("enhanced_application_shutdown_by_user")
    except Exception as e:
        logger.critical("enhanced_application_failed_to_run", error=str(e), exc_info=True)
