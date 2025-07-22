import pandas as pd
import numpy as np

def run_backtest(df_with_signals, initial_capital, fee_bps, period_min):
    """
    一个简单的向量化回测引擎，适配指定换仓周期。
    :param df_with_signals: 必须包含 'close' 和 'signal' 列的DataFrame。
    :param initial_capital: 初始资金。
    :param fee_bps: 单边交易手续费（基点）。
    :param period_min: 换仓周期（分钟）。
    :return: 包含回测结果的字典。
    """
    df = df_with_signals.copy()
    fee_rate = fee_bps / 10000.0
    
    # 1. 按换仓周期进行采样，只在决策点进行交易
    # 注意：这里的signal已经是shift过的，所以我们直接取每个周期的第一个信号即可
    backtest_df = df.resample(f'{period_min}T').agg({
        'close': 'first',
        'signal': 'first'
    }).dropna()
    
    # 强制每个周期开始时平仓：将上一个周期的信号置为0
    backtest_df['signal'] = backtest_df['signal'].shift(1).fillna(0)
    
    # 2. 计算每个持仓周期（小时级）的收益率
    backtest_df['ret'] = backtest_df['close'].pct_change()
    # 3. 计算策略收益率 (用上一个周期的信号，乘以当前周期的收益)
    backtest_df['strategy_ret'] = backtest_df['signal'].shift(1) * backtest_df['ret']
    
    # 4. 计算交易成本 (仅在信号变化时发生)
    trades = backtest_df['signal'].diff().abs()
    backtest_df['costs'] = trades * fee_rate
    
    # 5. 计算净收益
    backtest_df['net_strategy_ret'] = backtest_df['strategy_ret'] - backtest_df['costs']
    
    # 6. 计算累计收益曲线
    backtest_df['cumulative_ret'] = (1 + backtest_df['net_strategy_ret'].fillna(0)).cumprod()
    
    # 7. 计算最终结果
    # 7.1 总回报
    total_return = backtest_df['cumulative_ret'].iloc[-1] - 1
    
    # 7.2 年化夏普率
    periods_per_year = (365 * 24 * 60) / period_min
    sharpe_ratio = 0
    if backtest_df['net_strategy_ret'].std() != 0:
        sharpe_ratio = (backtest_df['net_strategy_ret'].mean() / backtest_df['net_strategy_ret'].std()) * np.sqrt(periods_per_year)

    # 7.3 最大回撤
    running_max = backtest_df['cumulative_ret'].cummax()
    drawdown = (running_max - backtest_df['cumulative_ret']) / running_max
    max_drawdown = drawdown.max()
    
    results = {
        'total_return': f"{total_return:.2%}",
        'annualized_sharpe_ratio': f"{sharpe_ratio:.2f}",
        'max_drawdown': f"{max_drawdown:.2%}",
        'cumulative_returns': backtest_df['cumulative_ret'], # 返回序列用于画图
        'backtest_details': backtest_df # 返回详细数据用于进一步分析
    }
    
    return results
