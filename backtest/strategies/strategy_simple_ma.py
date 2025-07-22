import pandas as pd
from backtest.factors import factor_ma15, factor_ma30

def generate_signals(df):
    """
    一个简单的双均线交叉策略。
    当短期均线（MA15）上穿长期均线（MA30）时，产生买入信号。
    当短期均线（MA15）下穿长期均线（MA30）时，产生卖出信号。
    
    :param df: 包含 'close' 价格的 DataFrame
    :return: 包含 'signal' 列的 DataFrame
    """
    df['ma15'] = factor_ma15(df)
    df['ma30'] = factor_ma30(df)
    
    # 当ma15 > ma30时，我们希望持有，信号为1；否则为-1（代表做空或空仓）
    df['signal'] = np.sign(df['ma15'] - df['ma30'])
    
    # 将NaN信号填充为0（无操作），并处理信号延迟（当前bar的信号在下一bar生效）
    df['signal'] = df['signal'].fillna(0).shift(1).fillna(0)
    
    return df