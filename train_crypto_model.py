import asyncio
import time
import nest_asyncio
nest_asyncio.apply()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import ccxt
import ta
import aiohttp
from typing import List, Dict, Tuple, Optional, Union
import pickle
from tqdm import tqdm
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from pathlib import Path
import seaborn as sns
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import glob
from torch.cuda.amp import autocast, GradScaler
import traceback
import yfinance as yf
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CUDA for performance
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# ---------------- Configuration ---------------- #
class Config:
    # Model architecture
    SEQ_LEN = 168  # 1 week of hourly data (7*24)
    PREDICTION_LENGTH = 24  # Predict 24 hours ahead
    BATCH_SIZE = 128
    HIDDEN_SIZE = 512
    NHEAD = 8
    DROPOUT = 0.2
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-6
    EPOCHS = 100  # 100 epochs as requested
    EARLY_STOPPING_PATIENCE = 10
    
    # Data parameters - 5 years of hourly data
    TOP_SYMBOLS = 10  # Focus on 10 cryptocurrencies
    TIMEFRAME = '1h'  # Hourly data
    DATA_LIMIT = 24 * 365 * 5  # 5 years of hourly data
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Trading parameters
    ENTRY_THRESHOLD = 0.65
    STOP_LOSS_PCT = 0.02
    TAKE_PROFIT_PCT = 0.04
    RISK_FREE_RATE = 0.02
    
    # Optimization
    OPTUNA_TRIALS = 20
    OPTUNA_EPOCHS = 25
    
    # Paths
    CACHE_DIR = "crypto_data"
    MODEL_DIR = "models"
    RESULTS_DIR = "results"
    STUDY_NAME = "crypto_model_optuna_v3"
    
    # Advanced settings
    GRADIENT_ACCUMULATION_STEPS = 4
    USE_AMP = True
    MAX_GRAD_NORM = 1.0
    
    # Data fetching settings
    MAX_RETRIES = 5
    RETRY_DELAY = 2  # seconds
    ALTERNATIVE_API_TIMEOUT = 30  # seconds
    CACHE_EXPIRY_DAYS = 7  # Refresh cache after 7 days

config = Config()

# Create directories
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# ---------------- Enhanced Data Fetching with Retry Logic ---------------- #
class DataFetcher:
    """Handles data fetching from multiple sources with robust retry logic"""
    
    def __init__(self):
        self.session = None
        self.exchanges = ['binance', 'kucoin', 'okx']
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.cryptocompare_base = "https://min-api.cryptocompare.com/data/v2"
        
        # Setup requests session with retry logic
        self.setup_session()
    
    def setup_session(self):
        """Setup requests session with retry logic"""
        retry_strategy = Retry(
            total=config.MAX_RETRIES,
            backoff_factor=config.RETRY_DELAY,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
    
    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, asyncio.TimeoutError))
    )
    async def fetch_top_perpetual_coins(self, limit: int = config.TOP_SYMBOLS) -> List[str]:
        """Return top cryptocurrency symbols with retry logic"""
        try:
            # Try CoinGecko first
            url = f"{self.coingecko_base}/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': 'false'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=config.ALTERNATIVE_API_TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [f"{coin['symbol'].upper()}-USD" for coin in data]
        
        except Exception as e:
            logger.warning(f"CoinGecko API failed: {e}. Using fallback list.")
        
        # Fallback to predefined list
        return [
            'BTC-USD', 'ETH-USD', 'BNB-USD', 
            'SOL-USD', 'XRP-USD', 'ADA-USD', 
            'AVAX-USD', 'DOT-USD', 'DOGE-USD', 
            'MATIC-USD'
        ][:limit]
    
    def is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache is still valid based on expiry time"""
        if not os.path.exists(cache_path):
            return False
        
        cache_time = os.path.getmtime(cache_path)
        cache_age = (time.time() - cache_time) / (3600 * 24)  # Age in days
        
        return cache_age < config.CACHE_EXPIRY_DAYS
    
    @retry(
        stop=stop_after_attempt(config.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, asyncio.TimeoutError))
    )
    async def fetch_ohlcv(self, symbol: str, timeframe: str = config.TIMEFRAME, 
                         limit: int = config.DATA_LIMIT) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with robust error handling and retry logic"""
        # Check cache first
        symbol_clean = symbol.replace('/', '-').replace('-USD', '')
        cache_file = os.path.join(config.CACHE_DIR, f"{symbol_clean}_{timeframe}.pkl")
        
        if os.path.exists(cache_file) and self.is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                logger.info(f"Loaded cached data for {symbol}")
                return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")
        
        # If no valid cache, fetch new data
        df = await self.fetch_ohlcv_from_sources(symbol, timeframe, limit)
        
        if df is not None and not df.empty:
            # Cache the data
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                logger.info(f"Cached data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to cache data for {symbol}: {e}")
        
        return df
    
    async def fetch_ohlcv_from_sources(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Try multiple data sources with fallback logic"""
        sources = [
            self.fetch_from_binance,
            self.fetch_from_yahoo,
            self.fetch_from_coingecko,
            self.fetch_from_cryptocompare
        ]
        
        for source in sources:
            try:
                df = await source(symbol, timeframe, limit)
                if df is not None and not df.empty:
                    logger.info(f"Successfully fetched {symbol} from {source.__name__}")
                    return df
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from {source.__name__}: {e}")
                continue
        
        logger.error(f"All data sources failed for {symbol}")
        return None
    
    async def fetch_from_binance(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from Binance API"""
        try:
            # Convert symbol format (BTC-USD -> BTC/USDT)
            binance_symbol = symbol.replace('-USD', '/USDT')
            
            # Initialize Binance exchange
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': config.ALTERNATIVE_API_TIMEOUT * 1000,  # ms
            })
            
            # Calculate since parameter for 5 years of data
            since = exchange.parse8601((datetime.now() - timedelta(days=365*5)).isoformat())
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(binance_symbol, timeframe, since, limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"Binance fetch failed for {symbol}: {e}")
            return None
    
  
    
    async def fetch_from_coingecko(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from CoinGecko API"""
        try:
            # Convert symbol format (BTC-USD -> bitcoin)
            coin_id = symbol.lower().replace('-usd', '')
            
            # Map common symbols to CoinGecko IDs
            coin_map = {
                'btc': 'bitcoin',
                'eth': 'ethereum',
                'bnb': 'binancecoin',
                'sol': 'solana',
                'xrp': 'ripple',
                'ada': 'cardano',
                'avax': 'avalanche-2',
                'dot': 'polkadot',
                'doge': 'dogecoin',
                'matic': 'matic-network'
            }
            
            coin_id = coin_map.get(coin_id, coin_id)
            
            # Calculate days based on timeframe
            if timeframe == '1h':
                days = 90  # Max 90 days for hourly data
            else:
                days = 365 * 5  # 5 years for daily data
            
            url = f"{self.coingecko_base}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': 'usd',
                'days': days,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=config.ALTERNATIVE_API_TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process data
                        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                        # Add volume (not available in OHLC endpoint)
                        df['volume'] = 0
                        
                        return df
            
            return None
            
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed for {symbol}: {e}")
            return None

    # Fix the Yahoo Finance data fetching method

# Fix the Yahoo Finance data fetching method
async def fetch_from_yahoo(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch data from Yahoo Finance with improved error handling"""
    try:
        # Convert symbol format if needed
        yf_symbol = symbol.replace('/', '-')
        
        # For hourly data, use a shorter period due to Yahoo's limitations
        if timeframe == '1h':
            # Yahoo limits hourly data to 730 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            # Download data using the correct parameter format
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1h')
        else:
            # For daily data, we can get more history
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*5)
            
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=timeframe)
        
        if df.empty:
            return None
            
        # Rename columns and process data
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low', 
            'Close': 'close', 'Volume': 'volume'
        })
        
        df.index.name = 'timestamp'
        
        # Data quality checks
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)
        
        return df
        
    except Exception as e:
        logger.warning(f"Yahoo Finance fetch failed for {symbol}: {e}")
        return None

# Fix the data type conversion issue in the dataset creation
class CryptoDataset(Dataset):
    """Fixed dataset class with proper data type handling"""
    
    def __init__(self, symbol: str, df: pd.DataFrame, seq_len: int = config.SEQ_LEN,
                 feature_columns: Optional[List[str]] = None, 
                 scaler: Optional[StandardScaler] = None,
                 mode: str = 'train'):
        
        self.symbol = symbol
        self.df = df
        self.seq_len = seq_len
        self.mode = mode
        
        # Define feature columns
        if feature_columns is None:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume', 
                'rsi_7', 'rsi_14', 'rsi_21', 'rsi_30',
                'macd', 'macd_signal', 'macd_diff',
                'ma_7', 'ma_14', 'ma_21', 'ma_50', 'ma_100', 'ma_200',
                'ema_7', 'ema_14', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
                'bb_high_1', 'bb_low_1', 'bb_width_1',
                'bb_high_2', 'bb_low_2', 'bb_width_2',
                'atr', 'obv', 'cci', 'adx',
                'price_change', 'high_low_ratio', 'close_open_ratio', 'volume_change'
            ]
        else:
            self.feature_columns = feature_columns
        
        # Ensure label column is integer type
        if 'label' in df.columns:
            self.df['label'] = self.df['label'].astype(int)
        
        # Create index mapping
        self.index_mapping = []
        n_samples = len(df) - seq_len - config.PREDICTION_LENGTH
        
        for i in range(n_samples):
            self.index_mapping.append(i)
        
        # FIX: Add indices property to prevent AttributeError
        self.indices = self.index_mapping
        
        # Handle scaler
        if scaler is None:
            self.scaler = StandardScaler()
            train_size = int(n_samples * config.TRAIN_RATIO)
            train_features = df[self.feature_columns].iloc[:train_size + seq_len].values
            self.scaler.fit(train_features)
        else:
            self.scaler = scaler
        
        print(f"Created {mode} dataset for {symbol} with {len(self.index_mapping)} samples")
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        start_idx = self.index_mapping[idx]
        
        # Get features and apply scaling
        features = self.df[self.feature_columns].iloc[start_idx:start_idx + self.seq_len].values
        features = self.scaler.transform(features)
        
        # Get label - ensure it's an integer
        label = int(self.df['label'].iloc[start_idx + self.seq_len])
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label])
        
        return features_tensor, label_tensor
    
    def get_scaler(self):
        return self.scaler

# Update the data fetcher to prioritize Yahoo Finance and skip Binance in restricted regions
async def fetch_ohlcv_from_sources(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Try multiple data sources with fallback logic, prioritizing Yahoo"""
    sources = [
        self.fetch_from_yahoo,  # Prioritize Yahoo
        self.fetch_from_coingecko,
        self.fetch_from_cryptocompare,
        # Skip Binance for restricted regions
    ]
    
    for source in sources:
        try:
            df = await source(symbol, timeframe, limit)
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched {symbol} from {source.__name__}")
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol} from {source.__name__}: {e}")
            continue
    
    logger.error(f"All data sources failed for {symbol}")
    return None

# Fix the data type conversion issue in the dataset creation
class CryptoDataset(Dataset):
    """Fixed dataset class with proper data type handling"""
    
    def __init__(self, symbol: str, df: pd.DataFrame, seq_len: int = config.SEQ_LEN,
                 feature_columns: Optional[List[str]] = None, 
                 scaler: Optional[StandardScaler] = None,
                 mode: str = 'train'):
        
        self.symbol = symbol
        self.df = df
        self.seq_len = seq_len
        self.mode = mode
        
        # Define feature columns
        if feature_columns is None:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume', 
                'rsi_7', 'rsi_14', 'rsi_21', 'rsi_30',
                'macd', 'macd_signal', 'macd_diff',
                'ma_7', 'ma_14', 'ma_21', 'ma_50', 'ma_100', 'ma_200',
                'ema_7', 'ema_14', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
                'bb_high_1', 'bb_low_1', 'bb_width_1',
                'bb_high_2', 'bb_low_2', 'bb_width_2',
                'atr', 'obv', 'cci', 'adx',
                'price_change', 'high_low_ratio', 'close_open_ratio', 'volume_change'
            ]
        else:
            self.feature_columns = feature_columns
        
        # Ensure label column is integer type
        if 'label' in df.columns:
            self.df['label'] = self.df['label'].astype(int)
        
        # Create index mapping
        self.index_mapping = []
        n_samples = len(df) - seq_len - config.PREDICTION_LENGTH
        
        for i in range(n_samples):
            self.index_mapping.append(i)
        
        # FIX: Add indices property to prevent AttributeError
        self.indices = self.index_mapping
        
        # Handle scaler
        if scaler is None:
            self.scaler = StandardScaler()
            train_size = int(n_samples * config.TRAIN_RATIO)
            train_features = df[self.feature_columns].iloc[:train_size + seq_len].values
            self.scaler.fit(train_features)
        else:
            self.scaler = scaler
        
        print(f"Created {mode} dataset for {symbol} with {len(self.index_mapping)} samples")
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        start_idx = self.index_mapping[idx]
        
        # Get features and apply scaling
        features = self.df[self.feature_columns].iloc[start_idx:start_idx + self.seq_len].values
        features = self.scaler.transform(features)
        
        # Get label - ensure it's an integer
        label = int(self.df['label'].iloc[start_idx + self.seq_len])
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label])
        
        return features_tensor, label_tensor
    
    def get_scaler(self):
        return self.scaler

# Update the data fetcher to prioritize Yahoo Finance and skip Binance in restricted regions
async def fetch_ohlcv_from_sources(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Try multiple data sources with fallback logic, prioritizing Yahoo"""
    sources = [
        self.fetch_from_yahoo,  # Prioritize Yahoo
        self.fetch_from_coingecko,
        self.fetch_from_cryptocompare,
        # Skip Binance for restricted regions
    ]
    
    for source in sources:
        try:
            df = await source(symbol, timeframe, limit)
            if df is not None and not df.empty:
                logger.info(f"Successfully fetched {symbol} from {source.__name__}")
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch {symbol} from {source.__name__}: {e}")
            continue
    
    logger.error(f"All data sources failed for {symbol}")
    return None

    async def fetch_from_cryptocompare(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Fetch data from CryptoCompare API"""
        try:
            # Convert symbol format (BTC-USD -> BTC)
            cryptocompare_symbol = symbol.replace('-USD', '')
            
            # Map timeframe
            tf_map = {
                '1h': 'hour',
                '1d': 'day'
            }
            period = tf_map.get(timeframe, 'hour')
            
            # Calculate limit based on timeframe
            if timeframe == '1h':
                limit = min(2000, limit)  # Max 2000 for hourly
            else:
                limit = min(2000, limit)  # Max 2000 for daily
            
            url = f"{self.cryptocompare_base}/histo{period}"
            params = {
                'fsym': cryptocompare_symbol,
                'tsym': 'USD',
                'limit': limit,
                'aggregate': 1,
                'e': 'CCCAGG'  # Exchange
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=config.ALTERNATIVE_API_TIMEOUT) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data['Response'] == 'Success':
                            df = pd.DataFrame(data['Data']['Data'])
                            df['time'] = pd.to_datetime(df['time'], unit='s')
                            df.set_index('time', inplace=True)
                            df.rename(columns={
                                'open': 'open',
                                'high': 'high',
                                'low': 'low',
                                'close': 'close',
                                'volumefrom': 'volume'
                            }, inplace=True)
                            
                            return df[['open', 'high', 'low', 'close', 'volume']]
            
            return None
            
        except Exception as e:
            logger.warning(f"CryptoCompare fetch failed for {symbol}: {e}")
            return None
    
    async def fetch_all_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Fetch data for all symbols with parallel processing"""
        tasks = [self.fetch_ohlcv(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
                continue
            if result is not None and not result.empty:
                valid_data[symbol] = result
        
        logger.info(f"Successfully fetched data for {len(valid_data)} symbols")
        return valid_data

# ---------------- Technical Indicators ---------------- #
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    df = df.copy()
    
    # Price-based features
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['high_low_ratio'] = (df['high'] / df['low']).replace([np.inf, -np.inf], 1).fillna(1)
    df['close_open_ratio'] = (df['close'] / df['open']).replace([np.inf, -np.inf], 1).fillna(1)
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    
    # Multiple RSI periods
    for window in [7, 14, 21, 30]:
        try:
            df[f'rsi_{window}'] = ta.momentum.RSIIndicator(df['close'], window=window).rsi().fillna(50)
        except:
            df[f'rsi_{window}'] = 50
    
    # MACD with different configurations
    try:
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd().fillna(0)
        df['macd_signal'] = macd.macd_signal().fillna(0)
        df['macd_diff'] = macd.macd_diff().fillna(0)
    except:
        df['macd'] = df['macd_signal'] = df['macd_diff'] = 0
    
    # Multiple moving averages
    for window in [7, 14, 21, 50, 100, 200]:
        try:
            df[f'ma_{window}'] = df['close'].rolling(window).mean().fillna(df['close'])
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean().fillna(df['close'])
        except:
            df[f'ma_{window}'] = df[f'ema_{window}'] = df['close']
    
    # Bollinger Bands with different deviations
    for dev in [1, 2]:
        try:
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=dev)
            df[f'bb_high_{dev}'] = bb.bollinger_hband().fillna(df['close'])
            df[f'bb_low_{dev}'] = bb.bollinger_lband().fillna(df['close'])
            df[f'bb_width_{dev}'] = ((df[f'bb_high_{dev}'] - df[f'bb_low_{dev}']) / 
                                   df['close']).fillna(0)
        except:
            df[f'bb_high_{dev}'] = df[f'bb_low_{dev}'] = df['close']
            df[f'bb_width_{dev}'] = 0
    
    # Additional indicators
    indicators = {
        'atr': ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']),
        'obv': ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']),
        'cci': ta.trend.CCIIndicator(df['high'], df['low'], df['close']),
        'adx': ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
    }
    
    for name, indicator in indicators.items():
        try:
            df[name] = getattr(indicator, name)() if hasattr(indicator, name) else indicator
            df[name] = df[name].fillna(0)
        except:
            df[name] = 0
    
    # Final cleanup
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.fillna(0, inplace=True)
    
    return df

# ---------------- Enhanced Dataset Class ---------------- #
class CryptoDataset(Dataset):
    """Fixed dataset class with indices attribute"""
    
    def __init__(self, symbol: str, df: pd.DataFrame, seq_len: int = config.SEQ_LEN,
                 feature_columns: Optional[List[str]] = None, 
                 scaler: Optional[StandardScaler] = None,
                 mode: str = 'train'):
        
        self.symbol = symbol
        self.df = df
        self.seq_len = seq_len
        self.mode = mode
        
        # Define feature columns
        if feature_columns is None:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume', 
                'rsi_7', 'rsi_14', 'rsi_21', 'rsi_30',
                'macd', 'macd_signal', 'macd_diff',
                'ma_7', 'ma_14', 'ma_21', 'ma_50', 'ma_100', 'ma_200',
                'ema_7', 'ema_14', 'ema_21', 'ema_50', 'ema_100', 'ema_200',
                'bb_high_1', 'bb_low_1', 'bb_width_1',
                'bb_high_2', 'bb_low_2', 'bb_width_2',
                'atr', 'obv', 'cci', 'adx',
                'price_change', 'high_low_ratio', 'close_open_ratio', 'volume_change'
            ]
        else:
            self.feature_columns = feature_columns
        
        # Create index mapping
        self.index_mapping = []
        n_samples = len(df) - seq_len - config.PREDICTION_LENGTH
        
        for i in range(n_samples):
            self.index_mapping.append(i)
        
        # FIX: Add indices property to prevent AttributeError
        self.indices = self.index_mapping
        
        # Handle scaler
        if scaler is None:
            self.scaler = StandardScaler()
            train_size = int(n_samples * config.TRAIN_RATIO)
            train_features = df[self.feature_columns].iloc[:train_size + seq_len].values
            self.scaler.fit(train_features)
        else:
            self.scaler = scaler
        
        print(f"Created {mode} dataset for {symbol} with {len(self.index_mapping)} samples")
    
    def __len__(self):
        return len(self.index_mapping)
    
    def __getitem__(self, idx):
        start_idx = self.index_mapping[idx]
        
        # Get features and apply scaling
        features = self.df[self.feature_columns].iloc[start_idx:start_idx + self.seq_len].values
        features = self.scaler.transform(features)
        
        # Get label
        label = self.df['label'].iloc[start_idx + self.seq_len]
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label])
        
        return features_tensor, label_tensor
    
    def get_scaler(self):
        return self.scaler

# ---------------- Enhanced Hybrid Model ---------------- #
class EnhancedHybridModel(nn.Module):
    """Advanced hybrid model with GRU, Transformer, and residual connections"""
    
    def __init__(self, input_size: int, hidden_size: int = config.HIDDEN_SIZE, 
                 nhead: int = config.NHEAD, dropout: float = config.DROPOUT, 
                 num_layers: int = 3, use_attention: bool = True, 
                 use_residual: bool = True):
        super().__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Feature attention
        if use_attention:
            self.feature_attention = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, input_size),
                nn.Sigmoid()
            )
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(input_size)
        
        # GRU layer with more capacity
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Transformer with enhanced configuration
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * 2,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        if use_residual:
            self.residual_blocks = nn.Sequential(
                *[ResidualBlock(hidden_size // 2, dropout) for _ in range(2)]
            )
        else:
            self.residual_blocks = nn.Identity()
        
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 3),
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        # Feature attention
        if self.use_attention:
            attn_weights = self.feature_attention(x.mean(dim=1, keepdim=True))
            x = x * attn_weights
        
        # Batch norm
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        
        # GRU
        gru_out, _ = self.gru(x)
        
        # Transformer
        transformer_out = self.transformer(gru_out)
        
        # Attention pooling
        attn_weights = self.attention_pool(transformer_out)
        context = torch.sum(transformer_out * attn_weights, dim=1)
        
        # Classification
        features = self.classifier(context)
        
        # Residual connection
        if self.use_residual:
            features = features + self.residual_blocks(features)
        
        return self.final_classifier(features)

# ---------------- Missing Functions ---------------- #
def create_triple_barrier_labels(df: pd.DataFrame, prediction_length: int = config.PREDICTION_LENGTH) -> np.ndarray:
    """Create trading labels using triple barrier method"""
    labels = np.zeros(len(df))  # 0: hold, 1: buy, 2: sell
    
    for i in range(len(df) - prediction_length):
        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+prediction_length+1]
        
        # Calculate returns
        max_return = (future_prices.max() - current_price) / current_price
        min_return = (future_prices.min() - current_price) / current_price
        
        # Apply barriers
        if max_return >= config.TAKE_PROFIT_PCT:
            labels[i] = 1  # Buy signal
        elif min_return <= -config.STOP_LOSS_PCT:
            labels[i] = 2  # Sell signal
        else:
            labels[i] = 0  # Hold
    
    return labels

def time_based_split(dataset: CryptoDataset) -> Tuple[Subset, Subset, Subset]:
    """Split dataset in time-based manner"""
    n = len(dataset)
    train_size = int(n * config.TRAIN_RATIO)
    val_size = int(n * config.VAL_RATIO)
    test_size = n - train_size - val_size
    
    # Create indices for splitting
    indices = list(range(n))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices)
    )

class ResidualBlock(nn.Module):
    """Residual block for enhanced model"""
    def __init__(self, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return x + self.block(x)

# ---------------- Training Utilities ---------------- #
def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device, 
                  criterion: nn.Module, return_predictions: bool = False) -> Dict:
    """Comprehensive model evaluation with proper class handling"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device).squeeze()
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.exp(outputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Handle missing classes in evaluation
    unique_labels = np.unique(all_labels)
    if len(unique_labels) < 3:
        # Add dummy samples for missing classes
        for missing_class in set([0, 1, 2]) - set(unique_labels):
            all_labels.append(missing_class)
            all_preds.append(missing_class)
    
    # Calculate metrics
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Confusion matrix with explicit labels
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'class_distribution': np.bincount(all_labels, minlength=3)
    }
    
    if return_predictions:
        results.update({
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        })
    
    return results

def train_epoch(model, train_loader, optimizer, criterion, device, scaler):
    """Single training epoch with gradient accumulation"""
    model.train()
    running_loss = 0.0
    
    for i, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device).squeeze()
        
        with autocast(enabled=config.USE_AMP):
            outputs = model(data)
            loss = criterion(outputs, labels) / config.GRADIENT_ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (i + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        running_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
    
    return running_loss / len(train_loader)

# ---------------- Main Training Function ---------------- #
async def main():
    """Complete training pipeline with 100 epochs"""
    print("🚀 Starting enhanced crypto model training...")
    print(f"📊 Target: {config.TOP_SYMBOLS} cryptocurrencies")
    print(f"⏰ Timeframe: {config.TIMEFRAME}, Data: 5 years")
    print(f"📈 Epochs: {config.EPOCHS}")
    print(f"💻 Device: {'GPU' if torch.cuda.is_available() else 'CPU'} available")
    
    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Fetch data
    async with DataFetcher() as fetcher:
        symbols = await fetcher.fetch_top_perpetual_coins(config.TOP_SYMBOLS)
        
        # Check for cached data
        cache_file = os.path.join(config.CACHE_DIR, "crypto_5year_hourly_data.pkl")
        cached_data = None
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"📦 Loaded {len(cached_data)} symbols from cache")
            except Exception as e:
                print(f"❌ Error loading cached data: {e}")
                cached_data = None
        
        if cached_data:
            data_dict = cached_data
        else:
            print("🌐 Fetching fresh data...")
            data_dict = await fetcher.fetch_all_data(symbols)
            
            if data_dict:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(data_dict, f)
                    print(f"💾 Data cached to {cache_file}")
                except Exception as e:
                    print(f"❌ Error caching data: {e}")
            else:
                print("❌ No data fetched. Exiting.")
                return
    
    # Process each symbol
    results = {}
    best_models = {}
    
    for symbol, df in data_dict.items():
        try:
            print(f"\n🔧 Processing {symbol}...")
            
            # Calculate technical indicators
            df_with_indicators = calculate_technical_indicators(df)
            
            # Create labels using triple barrier method
            labels = create_triple_barrier_labels(df_with_indicators)
            df_with_indicators['label'] = labels
            
            # Check class distribution
            class_counts = np.bincount(labels, minlength=3)
            print(f"📊 Class distribution: {class_counts}")
            
            if np.any(class_counts < 50):
                print(f"⏩ Skipping {symbol} - insufficient samples")
                continue
            
            # Create dataset
            dataset = CryptoDataset(symbol, df_with_indicators)
            
            if len(dataset) < 1000:
                print(f"⏩ Skipping {symbol} - insufficient sequences")
                continue
            
            # Split dataset
            train_dataset, val_dataset, test_dataset = time_based_split(dataset)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
            
            # Setup device and model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model = EnhancedHybridModel(
                input_size=len(dataset.feature_columns),
                hidden_size=config.HIDDEN_SIZE,
                nhead=config.NHEAD,
                dropout=config.DROPOUT,
                num_layers=3,
                use_attention=True,
                use_residual=True
            ).to(device)
            
            # Loss function and optimizer
            criterion = nn.NLLLoss()
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=config.LEARNING_RATE, 
                weight_decay=config.WEIGHT_DECAY
            )
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            )
            scaler = GradScaler(enabled=config.USE_AMP)
            
            # Training variables
            best_val_loss = float('inf')
            best_model_state = None
            patience_counter = 0
            
            print(f"🏋️ Training {symbol} for {config.EPOCHS} epochs...")
            
            # Training loop
            for epoch in range(config.EPOCHS):
                # Train epoch
                train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
                
                # Validation
                val_metrics = evaluate_model(model, val_loader, device, criterion)
                
                # Update learning rate
                scheduler.step(val_metrics['loss'])
                
                # Check for improvement
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Print progress
                if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == config.EPOCHS - 1:
                    print(f"Epoch {epoch+1}/{config.EPOCHS} - "
                          f"Train Loss: {train_loss:.4f} | "
                          f"Val Loss: {val_metrics['loss']:.4f} | "
                          f"Val Acc: {val_metrics['accuracy']*100:.2f}%")
                
                # Early stopping
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"⏹️ Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model and evaluate
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            test_metrics = evaluate_model(model, test_loader, device, criterion, True)
            
            # Save results and model
            results[symbol] = test_metrics
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"{symbol}_model.pth"))
            
            print(f"✅ Completed {symbol} | Test Acc: {test_metrics['accuracy']*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")
            continue
    
    # Save final results
    with open(os.path.join(config.RESULTS_DIR, "final_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎯 Training completed! Results saved.")

# ---------------- Execution ---------------- #
if __name__ == "__main__":
    start_time = time.time()
    
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
    finally:
        elapsed = time.time() - start_time
        print(f"\n⏱️ Total execution time: {elapsed/3600:.2f} hours")