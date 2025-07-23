import pandas as pd
import numpy as np

def factor_001(df):
    return - df['close'].rolling(window=15).mean()

def factor_002(df):
    vwap = (df['close'] * df['volume']).rolling(1440).sum() / df['volume'].rolling(1440).sum()
    f = (vwap - df['close']) / df['close'].rolling(12 * 60).std()
    return - f.rolling(1440).mean()

def factor_003(df):
    f = df['volume'] / df['trades']
    return f.pct_change().rolling(12 * 60).sum()

def factor_004(df):
    n = 60

    adv_s = df['volume'].rolling(n).mean()
    adv_l = df['volume'].rolling(5*n).mean()
    A = 1 + (adv_s/adv_l)
    B = 1 -  (adv_s/adv_l)
    f = df['bid_volume'] / df['volume'].rolling(30).mean() - 1
    f = (f*A).where(f >= 0, f*B)
    return f

def factor_005(df):
    n = 60

    diff = df['high'] - df['low']
    return diff.pct_change().rolling(2*n).sum()