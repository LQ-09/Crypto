import pandas as pd
import numpy as np

def factor_001(df):
    return - df['close'].rolling(window=15).mean()

def factor_002(df):
    log_vol = np.log(df['volume'])
    gradient = log_vol.pct_change()
    kurt = log_vol.rolling(30).kurt()
    return kurt * gradient

def factor_003(df):
    low_diff = np.log(df['close']) - np.log(df['low']).rolling(120).min()
    high_diff = np.log(df['high']).rolling(120).max() - np.log(df['close'])
    z_low = (low_diff - low_diff.rolling(240).mean()) / low_diff.rolling(240).std()
    z_high = (high_diff - high_diff.rolling(240).mean()) / high_diff.rolling(240).std()
    return - z_low / z_high


def factor_004(df):
    vwap = (df['close'] * df['volume']).rolling(240).sum() / df['volume'].rolling(240).sum()
    f = (vwap - df['close']) / df['close'].rolling(5).std()
    return - f.rolling(48).mean()

def factor_005(df):
    f = df['volume'] / df['trades']
    return f.pct_change().rolling(12 * 60).sum()

def factro_006(df):
    std_close = df['close'].rolling(120).std()
    condition = std_close.where(df['close'] > df['close'].shift(1), 0)
    return condition.ewm(span=12 * 60, adjust=False).mean()
