import pandas as pd
import numpy as np

def run_backtest(df_with_signals, initial_capital, fee_bps, period_min, leverage=1.0):
    """
    一个简单的向量化回测引擎，适配指定换仓周期。
    :param df_with_signals: 必须包含 'close' 和 'signal' 列的DataFrame。
    :param initial_capital: 初始资金。
    :param fee_bps: 单边交易手续费（基点）。
    :param period_min: 换仓周期（分钟）。
    :param leverage: 杠杆/仓位乘数。
    :return: 包含回测结果的字典。
    """
    df = df_with_signals.copy()
    fee_rate = fee_bps / 10000.0
    
    # 1. 按换仓周期进行采样
    # 策略文件生成的signal已经是t-1时刻的信号，决定t时刻的持仓
    backtest_df = df.resample(f'{period_min}min').agg({
        'close': 'first',
        'signal': 'first'
    }).dropna()
    
    # 2. 计算每个持仓周期的收益率
    backtest_df['ret'] = backtest_df['close'].pct_change()
    
    # 3. 计算策略毛收益 (信号在策略文件中已经shift(1)，所以这里直接使用)
    # 策略收益 = 信号强度 * 资产收益 * 杠杆
    backtest_df['strategy_ret'] = backtest_df['signal'] * backtest_df['ret'] * leverage
    
    # 4. 计算交易成本 (核心修正)
    # 逻辑：每个周期都必须先平掉上期仓位，再开立本期新仓位。
    # 上一期的仓位信号
    prev_signal = backtest_df['signal'].shift(1).fillna(0)
    # 总交易成本 = 平掉上期仓位的成本 + 开立本期仓位的成本
    total_traded_ratio = (prev_signal.abs() + backtest_df['signal'].abs()) * leverage
    backtest_df['costs'] = total_traded_ratio * fee_rate
    
    # 5. 计算净收益
    backtest_df['net_strategy_ret'] = backtest_df['strategy_ret'] - backtest_df['costs']
    
    # 6. 计算累计收益曲线
    backtest_df['cumulative_ret'] = (1 + backtest_df['net_strategy_ret'].fillna(0)).cumprod()
    
    # 新增：计算胜率 (Win Rate)
    # 定义：做多(signal=1)且收益为正，或做空(signal=-1)且收益为负，即为一次获胜的交易。
    # backtest_df['ret'] 是当前周期的收益率
    # backtest_df['signal'] 是在当前周期持有的仓位
    winning_trades = (
        ((backtest_df['signal'] > 0) & (backtest_df['ret'] > 0)) |
        ((backtest_df['signal'] < 0) & (backtest_df['ret'] < 0))
    ).sum()
    total_trades = (backtest_df['signal'] != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 7. 计算性能指标
    total_return = backtest_df['cumulative_ret'].iloc[-1] - 1
    
    # 年化夏普率
    periods_per_year = (365 * 24 * 60) / period_min
    sharpe_ratio = 0
    if backtest_df['net_strategy_ret'].std() != 0:
        sharpe_ratio = (backtest_df['net_strategy_ret'].mean() / backtest_df['net_strategy_ret'].std()) * np.sqrt(periods_per_year)

    # 最大回撤
    running_max = backtest_df['cumulative_ret'].cummax()
    drawdown = (running_max - backtest_df['cumulative_ret']) / running_max
    max_drawdown = drawdown.max()
    
    results = {
        'total_return': total_return,
        'annualized_sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'cumulative_returns': backtest_df['cumulative_ret'], # 返回序列用于画图
        'backtest_details': backtest_df # 返回详细数据用于进一步分析
    }
    
    return results
