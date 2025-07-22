import pandas as pd
import numpy as np

# 3. 选择因子函数
def factor_ma15(df):
    return df['close'].rolling(window=15).mean()

def factor_ma30(df):
    return df['close'].rolling(window=30).mean()
