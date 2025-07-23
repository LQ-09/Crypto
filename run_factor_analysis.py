import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# 设置绘图风格
sns.set_style('whitegrid')

# --- 导入项目模块 ---
from backtest.utils import load_data_in_period, load_factors_from_config, ensure_data_is_available


def run_single_factor_analysis(df, factor_func, period_min, transaction_fee_bps=0, quantile_window=1440, long_quantile=0.10, short_quantile=0.90):
    """
    对单个因子函数运行深度分析，并返回结果和用于绘图的数据。
    :param df: 1分钟原始数据
    :param factor_func: 单个因子函数
    :param period_min: 换仓周期（分钟）
    :param transaction_fee_bps: 手续费（基点）
    :return: (结果DataFrame, 详细分析DataFrame)
    """
    # 1. 计算未来N分钟的收益率，N为换仓周期
    df['future_ret'] = df['close'].shift(-period_min) / df['close'] - 1
    
    # 2. 计算年化因子
    periods_per_year = (365 * 24 * 60) / period_min
    annualization_factor = np.sqrt(periods_per_year)
    fee_rate = transaction_fee_bps / 10000.0
    
    factor_name = factor_func.__name__
    # 3. 在完整的1分钟数据上计算因子值
    df['factor'] = factor_func(df)
    
    # 更新：基于滚动分位数的动态信号生成 (与策略回测逻辑对齐)
    # 3.1 计算交易信号的动态边界
    df['q_low'] = df['factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(long_quantile)
    df['q_high'] = df['factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(short_quantile)
    df['q_min'] = df['factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(0.01)
    df['q_max'] = df['factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(0.99)
    
    df['signal'] = 0.0

    # 3.2 计算动态仓位
    long_cond = df['factor'] <= df['q_low']
    long_numerator = df['q_low'] - df['factor']
    long_denominator = df['q_low'] - df['q_min']
    long_denominator.replace(0, np.nan, inplace=True)
    long_strength = (long_numerator / long_denominator).clip(0, 1)
    df.loc[long_cond, 'signal'] = long_strength

    short_cond = df['factor'] >= df['q_high']
    short_numerator = df['factor'] - df['q_high']
    short_denominator = df['q_max'] - df['q_high']
    short_denominator.replace(0, np.nan, inplace=True)
    short_strength = (short_numerator / short_denominator).clip(0, 1)
    df.loc[short_cond, 'signal'] = -short_strength

    # 4. 按换仓周期进行采样，只在换仓点进行分析
    # 使用resample确保我们只在每个周期的开始点（如整点）进行分析。
    analysis_df = df.resample(f'{period_min}min').agg({
        'factor': 'first',      # 取10:00的因子值
        'signal': 'first',      # 取10:00的交易信号
        'future_ret': 'first'   # 取10:00时计算的未来60分钟收益
    }).fillna({'signal': 0}).dropna(subset=['factor', 'future_ret']) # 填充信号中的NaN并保留有效数据

    # 5. 计算IC、Rank IC 和 因子换手率 (Turnover)
    ic = analysis_df[['factor', 'future_ret']].corr().iloc[0, 1]
    rank_ic = analysis_df[['factor', 'future_ret']].rank().corr().iloc[0, 1]
    # Turnover是因子在每个换仓点的自相关性，衡量因子的稳定性
    turnover = analysis_df['factor'].corr(analysis_df['factor'].shift(1))

    # 5.5 计算胜率 (Win Rate)
    # 胜率定义为：做多（因子值为正）时未来收益为正的比例，加上做空（因子值为负）时未来收益为负的比例。
    # 使用上一周期的信号作为本周期的持仓
    trade_signal = analysis_df['signal'].shift(1)
    
    # 识别出做多和做空的周期
    long_trades = trade_signal > 0
    short_trades = trade_signal < 0
    
    # 计算获胜的交易数量
    wins = (long_trades & (analysis_df['future_ret'] > 0)).sum() + (short_trades & (analysis_df['future_ret'] < 0)).sum()
    total_trades = (long_trades | short_trades).sum()
    win_rate = wins / total_trades if total_trades > 0 else 0

    # 6. 计算因子夏普率 (Factor Sharpe) - 基于新的交易规则
    # 6.1 因子收益率（毛利），持仓为-1, 0, 1
    analysis_df['factor_return'] = analysis_df['signal'].shift(1) * analysis_df['future_ret']
    
    # 6.2 计算交易成本 (与持仓变化成正比)
    # 更新：与策略回测引擎的成本计算逻辑对齐 (每期都重新开仓)
    prev_signal = analysis_df['signal'].shift(1).fillna(0)
    transaction_costs = (prev_signal.abs() + analysis_df['signal'].abs()) * fee_rate
    analysis_df['net_factor_return'] = analysis_df['factor_return'] - transaction_costs

    # 6.4 计算年化夏普率
    gross_sharpe = (analysis_df['factor_return'].mean() / analysis_df['factor_return'].std()) * annualization_factor if analysis_df['factor_return'].std() != 0 else 0
    net_sharpe = (analysis_df['net_factor_return'].mean() / analysis_df['net_factor_return'].std()) * annualization_factor if analysis_df['net_factor_return'].std() != 0 else 0

    # 7. 计算最大回撤 (Max Drawdown)
    # 7.1 计算累计净收益曲线 (也用于后续绘图)
    analysis_df['cumulative_gross_pnl'] = (1 + analysis_df['factor_return'].fillna(0)).cumprod()
    analysis_df['cumulative_pnl'] = (1 + analysis_df['net_factor_return'].fillna(0)).cumprod()
    # 7.2 计算滚动最大值
    running_max = analysis_df['cumulative_pnl'].cummax()
    # 7.3 计算回撤百分比
    drawdown = (running_max - analysis_df['cumulative_pnl']) / running_max
    # 7.4 获取最大回撤
    max_drawdown = drawdown.max()

    # 8. 计算年化收益率 (Annual Return)
    num_periods = len(analysis_df)
    if num_periods > 0 and periods_per_year > 0:
        # (1 + total_ret) ** (1/num_years) - 1
        annual_return = (analysis_df['cumulative_pnl'].iloc[-1]) ** (periods_per_year / num_periods) - 1
    else:
        annual_return = 0

    results = {
        'Factor': factor_name,
        'Period (min)': period_min,
        'Annual Return': annual_return,
        'IC': ic, 'Rank IC': rank_ic, 'Turnover': turnover, 'Win Rate': win_rate,
        'Sharpe (Gross)': gross_sharpe, 'Sharpe (Net)': net_sharpe,
        'MDD': max_drawdown
    }

    return pd.DataFrame([results]), analysis_df

def plot_correlation_chart(correlation_series, factor_name, ax):
    """
    在指定的Axes上绘制新因子与已归档因子相关性的柱状图。
    :param correlation_series: 包含相关性数据的Pandas Series
    :param factor_name: 新因子的名称
    :param ax: Matplotlib Axes object to draw on
    """
    if correlation_series is None or correlation_series.empty:
        ax.text(0.5, 0.5, 'No archived factors to compare', ha='center', va='center', fontsize=12, color='grey')
        ax.set_title(f'Correlation of "{factor_name}"')
        return

    # 为正负相关性设置不同颜色，使其更直观
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in correlation_series.values]
    bars = ax.bar(correlation_series.index, correlation_series.values, color=colors)

    # 在柱状图上显示数值
    ax.bar_label(bars, fmt='%.3f', padding=3)

    ax.set_title(f'Correlation of "{factor_name}" with Archived Factors')
    ax.set_ylabel('Correlation Coefficient')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(min(correlation_series.min() * 1.2, -1.05), max(correlation_series.max() * 1.2, 1.05))
    ax.axhline(0, color='grey', linewidth=0.8)

def plot_gross_vs_net_pnl(df_period, factor_name, period_min, ax):
    """
    在指定的Axes上绘制毛收益与净收益的对比曲线。
    """
    if df_period is None or df_period.empty:
        ax.text(0.5, 0.5, 'No data for this period', ha='center', va='center', fontsize=12, color='grey')
        ax.set_title(f'Gross vs Net PnL (Period: {period_min} min)')
        return

    df_period['cumulative_gross_pnl'].plot(label='Gross PnL (w/o fee)', ax=ax, color='skyblue')
    df_period['cumulative_pnl'].plot(label='Net PnL (w/ fee)', ax=ax, color='orangered')
    ax.set_title(f'Gross vs Net PnL (Period: {period_min} min)')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)

def plot_multi_period_pnl(pnl_data_by_period, factor_name, ax):
    """
    在指定的Axes上绘制多个周期下的累计PnL曲线。
    :param pnl_data_by_period: 字典，key为周期(分钟), value为包含'cumulative_pnl'的DataFrame
    :param factor_name: 因子名称
    :param ax: Matplotlib Axes object to draw on
    """
    # 使用colormap来自动分配不同颜色，视觉效果更好
    colors = plt.cm.viridis(np.linspace(0, 1, len(pnl_data_by_period)))

    for (period, df), color in zip(pnl_data_by_period.items(), colors):
        df['cumulative_pnl'].plot(label=f'Period: {period} min', ax=ax, color=color)

    ax.set_title(f'Multi-Period Net PnL Curve for Factor: {factor_name}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)

# --- 因子原型开发区 (Factor Prototyping Area) ---
# 在这里定义和测试您当前正在研究的新因子。
def factor_to_test(df):
    volumes = df["volume"]
    closes = df["close"]
    opens = df['open']
    lows = df["low"]
    highs = df["high"]
    trades = df["trades"]
    n = 12 * 60
 
    ret = closes/closes.shift(1) - 1
    weighted_ret = ret.ewm(span=n, adjust=False).mean()
    f = weighted_ret.pct_change()
    return  - f * volumes.pct_change()

if __name__ == '__main__':
    # ==============================================================================
    # --- 1. 配置区 ---
    # ==============================================================================
    # 指定在原型区中要测试的因子函数
    FACTOR_TO_TEST = factor_to_test
    QUANTILE_WINDOW = 24 * 60    # 用于计算分位数信号的滚动窗口 (1440分钟 = 1天)
    LONG_ENTRY_QUANTILE = 0.10   # 做多信号的因子值分位数阈值 (e.g., 0.10 for bottom 10%)
    SHORT_ENTRY_QUANTILE = 0.90  # 做空信号的因子值分位数阈值 (e.g., 0.90 for top 10%)

    SYMBOL = 'BTCUSDT'
    TRANSACTION_FEE_BPS = 10     # 用于计算净值夏普率的交易手续费（基点），10个基点即0.1%
    REBALANCE_PERIODS_MIN = [60, 120, 180, 240] # 测试多个换仓周期
    START_DATE = '2025-01-01'    # 回测开始日期
    END_DATE = '2025-06-01'      # 回测结束日期 (包含当天)

    # ==============================================================================
    # --- 2. 数据加载 ---
    # ==============================================================================
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, 'data')
    config_path = os.path.join(project_root, 'backtest', 'factor_config.json') # 因子配置文件路径
    result_dir = os.path.join(project_root, 'backtest', 'factor_result') # 存放因子图表和结果的文件夹
    os.makedirs(result_dir, exist_ok=True) # 确保结果目录存在
    symbol_dir = SYMBOL[:3]
    data_dir = os.path.join(data_root, symbol_dir)

    # 在加载数据前，确保所有需要的文件都已存在
    ensure_data_is_available(SYMBOL, START_DATE, END_DATE, data_dir)

    df = load_data_in_period(data_dir, SYMBOL, START_DATE, END_DATE)

    # ==============================================================================
    # --- 3. 计算与已归档因子的相关性 ---
    # ==============================================================================
    if not df.empty:
        print("\n--- 计算与已归档因子的相关性 ---")
        # 从配置文件加载已归档的因子
        archived_factors = load_factors_from_config(config_path)
        
        # 创建一个DataFrame来存储所有因子的值
        factor_values_df = pd.DataFrame(index=df.index)
        
        # 计算当前测试因子的值
        factor_values_df[FACTOR_TO_TEST.__name__] = FACTOR_TO_TEST(df)

        # 计算所有已归档因子的值
        for name, func in archived_factors.items():
            factor_values_df[name] = func(df)
            
        # 计算相关性矩阵，并提取新因子与其他因子的相关性序列
        if not factor_values_df.empty and len(factor_values_df.columns) > 1:
            correlation_matrix = factor_values_df.corr()
            correlation_series = correlation_matrix[FACTOR_TO_TEST.__name__].drop(FACTOR_TO_TEST.__name__)
        else:
            correlation_series = None # 如果无法计算，则设为None
            print("因子数量不足（少于2个），无法计算相关性。")

    # ==============================================================================
    # --- 4. 运行新因子的性能分析与绘图 ---
    # ==============================================================================
    if not df.empty:
        all_results = []
        pnl_data_for_plotting = {} # 存储不同周期的PnL数据用于绘图

        print(f"\n--- 正在对新因子进行性能分析: {FACTOR_TO_TEST.__name__} ---")
        # 针对每个周期进行回测
        for period in tqdm(REBALANCE_PERIODS_MIN, desc=f"Analyzing {FACTOR_TO_TEST.__name__}"):
            # 传入df.copy()确保每次分析都基于干净的原始数据
            analysis_results, detailed_df = run_single_factor_analysis(
                df.copy(), FACTOR_TO_TEST, period, TRANSACTION_FEE_BPS, QUANTILE_WINDOW,
                LONG_ENTRY_QUANTILE, SHORT_ENTRY_QUANTILE
            )
            all_results.append(analysis_results)
            pnl_data_for_plotting[period] = detailed_df

        # 合并结果并筛选出60分钟周期的结果进行展示
        if all_results:
            final_results_df = pd.concat(all_results, ignore_index=True)

            # 安全地筛选60分钟周期的结果
            result_60min = final_results_df[final_results_df['Period (min)'] == 60].copy()

            print(f"\n--- {FACTOR_TO_TEST.__name__} 在 60min 周期下的核心指标 ---")
            print(f"交易手续费: {TRANSACTION_FEE_BPS} bps ({TRANSACTION_FEE_BPS/10000.0:.4f})")

            if not result_60min.empty:
                # 在打印前移除 Period (min) 列以保持输出简洁
                result_to_print = result_60min.drop(columns=['Period (min)'])
                print(result_to_print.to_string(index=False))
            else:
                print("未能找到 60min 周期的分析结果。请确保 60 在 REBALANCE_PERIODS_MIN 配置中。")

            # --- 整合绘图 ---
            print("\n--- 正在生成整合分析图表 ---")
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(2, 2, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, :])

            # 1. 绘制相关性图
            plot_correlation_chart(correlation_series, FACTOR_TO_TEST.__name__, ax1)

            # 2. 绘制毛利vs净利对比图 (使用60min周期数据)
            df_60min_pnl = pnl_data_for_plotting.get(60)
            plot_gross_vs_net_pnl(df_60min_pnl, FACTOR_TO_TEST.__name__, 60, ax2)

            # 3. 绘制多周期对比图
            plot_multi_period_pnl(pnl_data_for_plotting, FACTOR_TO_TEST.__name__, ax3)

            fig.suptitle(f'Comprehensive Analysis for Factor: {FACTOR_TO_TEST.__name__}', fontsize=20)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # 调整布局以适应主标题
            
            save_path = os.path.join(result_dir, f'{FACTOR_TO_TEST.__name__}_comprehensive_analysis.png')
            plt.savefig(save_path)
            print(f"整合分析图表已保存至: {save_path}")
        else:
            print("错误：未能生成任何分析结果。")

    # 在脚本末尾一次性显示所有已创建但未关闭的图表
    print("\n分析完成。显示所有图表...")
    plt.show()
