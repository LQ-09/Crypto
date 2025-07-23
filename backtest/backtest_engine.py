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
    # 重命名为 period_return 以便更清晰地区分，并与 PnL 归因脚本对齐
    backtest_df['period_return'] = backtest_df['close'].pct_change()
    
    # 3. 计算策略毛收益 (信号在策略文件中已经shift(1)，所以这里直接使用)
    # 策略收益 = 信号强度 * 资产收益 * 杠杆
    backtest_df['strategy_gross_ret'] = backtest_df['signal'] * backtest_df['period_return'] * leverage
    
    # 4. 计算交易成本 (核心逻辑修正)
    # 正确的逻辑：成本只在仓位发生变化时产生，与仓位变化的绝对值成正比。
    # 上一期的仓位信号
    prev_signal = backtest_df['signal'].shift(1).fillna(0)
    # 交易量 = (当前信号 - 上一期信号) 的绝对值
    turnover = (backtest_df['signal'] - prev_signal).abs() * leverage
    backtest_df['costs'] = turnover * fee_rate
    
    # 5. 计算净收益
    backtest_df['net_strategy_ret'] = backtest_df['strategy_gross_ret'] - backtest_df['costs']
    
    # 6. 计算累计收益曲线
    backtest_df['cumulative_return_index'] = (1 + backtest_df['net_strategy_ret'].fillna(0)).cumprod()
    backtest_df['equity_curve'] = initial_capital * backtest_df['cumulative_return_index']
    
    # 新增：计算胜率 (Win Rate)
    # 定义：做多(signal=1)且收益为正，或做空(signal=-1)且收益为负，即为一次获胜的交易。
    # backtest_df['period_return'] 是当前周期的收益率
    # backtest_df['signal'] 是在当前周期持有的仓位
    winning_trades = (
        ((backtest_df['signal'] > 0) & (backtest_df['period_return'] > 0)) |
        ((backtest_df['signal'] < 0) & (backtest_df['period_return'] < 0))
    ).sum()
    total_trades = (backtest_df['signal'] != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # 7. 计算性能指标
    total_return = backtest_df['cumulative_return_index'].iloc[-1] - 1
    
    # 年化夏普率
    periods_per_year = (365 * 24 * 60) / period_min
    sharpe_ratio = 0
    # 增加一个检查，确保有足够的数据点来计算标准差
    if len(backtest_df) > 1 and backtest_df['net_strategy_ret'].std() != 0:
        sharpe_ratio = (backtest_df['net_strategy_ret'].mean() / backtest_df['net_strategy_ret'].std()) * np.sqrt(periods_per_year)

    # 最大回撤
    running_max = backtest_df['cumulative_return_index'].cummax()
    drawdown = (running_max - backtest_df['cumulative_return_index']) / running_max
    max_drawdown = drawdown.max()
    
    results = {
        'total_return': total_return,
        'annualized_sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'cumulative_returns': backtest_df['cumulative_return_index'], # 返回序列用于画图
        'equity_curve': backtest_df['equity_curve'],
    }
    
    # 返回结果字典和包含详细计算过程的DataFrame，以供PnL归因分析使用
    return results, backtest_df
