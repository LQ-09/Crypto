import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_style('whitegrid')

# --- 导入项目模块 ---
from backtest.utils import load_data_in_period, load_factors_from_config
from backtest.data_download import fetch_binance_klines_min

def run_single_factor_analysis(df, factor_func, period_min, transaction_fee_bps=0, quantile_window=1440):
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
    
    # 新增：基于滚动分位数的信号生成
    # 当因子值低于10%分位数时做多(信号为1)，高于90%分位数时做空(信号为-1)
    df['q10'] = df['factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(0.10)
    df['q90'] = df['factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(0.90)
    
    df['signal'] = 0.0
    df.loc[df['factor'] <= df['q10'], 'signal'] = 1.0
    df.loc[df['factor'] >= df['q90'], 'signal'] = -1.0

    # 4. 按换仓周期进行采样，只在换仓点进行分析
    # 使用resample确保我们只在每个周期的开始点（如整点）进行分析。
    analysis_df = df.resample(f'{period_min}min').agg({
        'factor': 'first',      # 取10:00的因子值
        'signal': 'first',      # 取10:00的交易信号
        'future_ret': 'first'   # 取10:00时计算的未来60分钟收益
    }).dropna()

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
    position_change = analysis_df['signal'].diff().abs()
    transaction_costs = position_change * fee_rate
    analysis_df['net_factor_return'] = analysis_df['factor_return'] - transaction_costs

    # 6.4 计算年化夏普率
    gross_sharpe = (analysis_df['factor_return'].mean() / analysis_df['factor_return'].std()) * annualization_factor if analysis_df['factor_return'].std() != 0 else 0
    net_sharpe = (analysis_df['net_factor_return'].mean() / analysis_df['net_factor_return'].std()) * annualization_factor if analysis_df['net_factor_return'].std() != 0 else 0

    # 7. 计算最大回撤 (Max Drawdown)
    # 7.1 计算累计净收益曲线 (也用于后续绘图)
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

def plot_correlation_chart(correlation_series, factor_name, result_dir):
    """
    绘制并保存新因子与已归档因子相关性的柱状图。
    :param correlation_series: 包含相关性数据的Pandas Series
    :param factor_name: 新因子的名称
    :param result_dir: 保存图片的目录
    """
    if correlation_series.empty:
        print("没有可用于绘制相关性图表的已归档因子。")
        return

    plt.figure(figsize=(10, 6))
    # 为正负相关性设置不同颜色，使其更直观
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in correlation_series.values]
    bars = plt.bar(correlation_series.index, correlation_series.values, color=colors)

    # 在柱状图上显示数值
    plt.bar_label(bars, fmt='%.3f', padding=3)

    plt.title(f'Correlation of "{factor_name}" with Archived Factors', fontsize=16)
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha="right")
    plt.ylim(min(correlation_series.min() * 1.2, -1.05), max(correlation_series.max() * 1.2, 1.05)) # 动态调整Y轴范围
    plt.axhline(0, color='grey', linewidth=0.8) # 在0处画一条线
    plt.tight_layout() # 调整布局防止标签重叠

    save_path = os.path.join(result_dir, f'{factor_name}_correlation.png')
    plt.savefig(save_path)
    print(f"因子相关性图表已保存至: {save_path}")
    plt.show()
    plt.close()

def plot_multi_period_pnl(pnl_data_by_period, factor_name, result_dir):
    """
    绘制单个因子在多个周期下的累计PnL曲线，并保存。
    :param pnl_data_by_period: 字典，key为周期(分钟), value为包含'cumulative_pnl'的DataFrame
    :param factor_name: 因子名称
    :param result_dir: 保存图片的目录
    """
    # 确保结果目录存在
    plt.figure(figsize=(15, 8))

    # 使用colormap来自动分配不同颜色，视觉效果更好
    colors = plt.cm.viridis(np.linspace(0, 1, len(pnl_data_by_period)))

    for (period, df), color in zip(pnl_data_by_period.items(), colors):
        df['cumulative_pnl'].plot(label=f'Period: {period} min', color=color)

    plt.title(f'Multi-Period PnL Curve for Factor: {factor_name}', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)

    # 保存图像
    save_path = os.path.join(result_dir, f'{factor_name}_multi_period_pnl.png')
    plt.savefig(save_path)
    print(f"多周期 PnL 曲线图已保存至: {save_path}")

    plt.show()  # 仍然在屏幕上显示图像
    plt.close() # 关闭图像以释放内存，防止在循环中打开过多窗口

def ensure_data_is_available(symbol, start_date_str, end_date_str, data_dir):
    """
    检查所需的数据文件是否存在，如果不存在则自动下载。
    数据按周（周一为一周的开始）进行切分和存储。
    """
    print("\n--- 正在检查并确保数据可用性 ---")
    start_dt = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)

    # 找到包含开始日期的那一周的周一
    current_week_start = start_dt - timedelta(days=start_dt.weekday())

    while current_week_start <= end_dt:
        current_week_end = current_week_start + timedelta(days=7)
        
        # 文件名格式: SYMBOL_1min_YYYYMMDD_YYYYMMDD.csv (结束日期为开区间)
        fname = f"{symbol}_1min_{current_week_start.strftime('%Y%m%d')}_{current_week_end.strftime('%Y%m%d')}.csv"
        file_path = os.path.join(data_dir, fname)

        if os.path.exists(file_path):
            pass
        else:
            print(f"数据文件不存在，正在下载: {fname}...")
            try:
                df_week = fetch_binance_klines_min(symbol, current_week_start, current_week_end)
                if not df_week.empty:
                    df_week.to_csv(file_path, index=False)
                    print(f"已保存: {file_path} ({len(df_week)} 行)")
                else:
                    print(f"警告: 未能从币安获取 {current_week_start.date()} 到 {current_week_end.date()} 的数据。将创建一个空文件以避免重复下载。")
                    # 创建一个空文件，防止下次运行时再次尝试下载无效数据
                    open(file_path, 'a').close()
            except Exception as e:
                print(f"下载数据时发生错误: {e}")

        # 移至下一周
        current_week_start = current_week_end

# --- 因子原型开发区 (Factor Prototyping Area) ---
# 在这里定义和测试您当前正在研究的新因子。
def factor_to_test(df):
    """一个示例因子，用于原型测试。该因子结合了价格方向和交易量。"""
    direction = np.sign(df['close'].diff())
    # 过去24分钟的方向性总和，乘以当前分钟的交易量
    f = (direction).rolling(24).sum() * df['volume']
    # 返回原始值，正值可能预示上涨，负值可能预示下跌
    return f


if __name__ == '__main__':
    # ==============================================================================
    # --- 1. 配置区 ---
    # ==============================================================================
    # 指定在原型区中要测试的因子函数
    FACTOR_TO_TEST = factor_to_test
    QUANTILE_WINDOW = 24 * 60    # 用于计算分位数信号的滚动窗口 (1440分钟 = 1天)

    SYMBOL = 'BTCUSDT'
    TRANSACTION_FEE_BPS = 2      # 用于计算净值夏普率的交易手续费（基点），设为0可以看得更清楚
    REBALANCE_PERIODS_MIN = [60, 120, 180, 240] # 测试多个换仓周期
    START_DATE = '2024-01-01'    # 回测开始日期
    END_DATE = '2025-01-01'      # 回测结束日期 (包含当天)

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
            plot_correlation_chart(correlation_series, FACTOR_TO_TEST.__name__, result_dir)
        else:
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
                df.copy(), FACTOR_TO_TEST, period, TRANSACTION_FEE_BPS, QUANTILE_WINDOW
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

            # 为当前因子绘制多周期对比图
            plot_multi_period_pnl(pnl_data_for_plotting, FACTOR_TO_TEST.__name__, result_dir)
        else:
            print("错误：未能生成任何分析结果。")
