import pandas as pd
import numpy as np

def factor_ma15(df):
    return - df['close'].rolling(window=15).mean()
