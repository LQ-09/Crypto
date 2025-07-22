import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')

BINANCE_URL = 'https://data-api.binance.vision/api/v3/klines'
COLUMNS = [
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

def fetch_binance_klines(symbol, interval, start_time, end_time, limit=1000):
    all_data = []
    start = int(start_time.timestamp() * 1000)
    end = int(end_time.timestamp() * 1000) - 1
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start,
            'endTime': end,
            'limit': limit
        }
        response = requests.get(BINANCE_URL, params=params)
        data = response.json()
        if not data or not isinstance(data, list):
            break
        all_data.extend(data)
        if len(data) < limit:
            break
        # 下一个start为最后一根K线的open_time+1ms
        last_open_time = data[-1][0] + 1
        start = last_open_time
        if start > end:
            break
    if not all_data:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(all_data, columns=COLUMNS)
    # 将open_time转换为datetime对象，并设为'timestamp'列
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)

    # 精简和重命名字段
    df = df.rename(columns={
        'number_of_trades': 'trades',
        'taker_buy_base_asset_volume': 'bid_volume',
        'taker_buy_quote_asset_volume': 'bid_quote_volume'
    })

    # 计算ask_volume = volume - bid_volume
    df['bid_volume'] = df['bid_volume'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['ask_volume'] = df['volume'] - df['bid_volume']

    # 选择并重排我们需要的列，确保'timestamp'是第一列
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades', 'bid_volume', 'ask_volume']]
    return df

def fetch_binance_klines_min(symbol, start_time, end_time):
    return fetch_binance_klines(symbol, '1m', start_time, end_time)

def fetch_binance_klines_hour(symbol, start_time, end_time):
    return fetch_binance_klines(symbol, '1h', start_time, end_time)

def fetch_binance_klines_day(symbol, start_time, end_time):
    return fetch_binance_klines(symbol, '1d', start_time, end_time)
