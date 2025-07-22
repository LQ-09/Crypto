import pandas as pd
import numpy as np
import os
import glob
from factors import factor_ma15, factor_ma30
from tqdm import tqdm

# --- 配置区 ---
SYMBOL = 'BTCUSDT'
TRANSACTION_FEE_BPS = 2  # 交易手续费（基点），例如 2 bps = 0.02%
# --- 回测周期配置 ---
START_DATE = '2024-01-08' # 回测开始日期
END_DATE = '2024-01-15'   # 回测结束日期 (包含当天)

# --- 路径设置 (与 base.py 保持一致) ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_root = os.path.join(project_root, 'data')
symbol_dir = SYMBOL[:3]
data_dir = os.path.join(data_root, symbol_dir)

def load_data_in_period(data_directory, symbol, start_date, end_date):
    """加载并合并指定币种在特定时间段内的数据文件"""
    search_path = os.path.join(data_directory, f"{symbol}_1min_*.csv")
    all_files = sorted(glob.glob(search_path))
    if not all_files:
        print(f"在 {data_directory} 中未找到 {symbol} 的数据文件")
        return pd.DataFrame()
    
    print(f"发现 {len(all_files)} 个数据文件，正在加载...")
    # 注意：这里我们使用了新的数据格式，时间列为 'timestamp'
    df_list = [pd.read_csv(f, parse_dates=['timestamp']) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)  # 确保数据按时间排序
    # 如果数据下载时有重叠，可能会有重复索引，这里去重
    df = df[~df.index.duplicated(keep='first')]
    
    # 根据指定的开始和结束日期筛选数据
    df = df.loc[start_date:end_date]

    print(f"已加载并筛选数据, 共 {len(df)} 条")
    print(f"回测时间范围: {df.index.min()} to {df.index.max()}")
    return df

def run_factor_backtest(df, factor_funcs, transaction_fee_bps=0):
    """对一组因子函数运行回测"""
    # 1. 只计算一次未来收益率
    df['ret'] = df['close'].shift(-1) / df['close'] - 1
    
    # 2. 使用正确的加密货币市场年化因子 (365天)
    minutes_in_year = 365 * 24 * 60
    annualization_factor = np.sqrt(minutes_in_year)
    
    results = []

    for func in tqdm(factor_funcs, desc='因子回测进度'):
        factor_name = func.__name__
        df['factor'] = func(df)
        
        # 清理因子和收益计算产生的NaN值，保证统计的准确性
        clean_df = df[['factor', 'ret', 'close']].dropna()

        # 3. 计算IC和Rank IC
        ic = clean_df[['factor', 'ret']].corr().iloc[0, 1]
        rank_ic = clean_df[['factor', 'ret']].rank().corr().iloc[0, 1]
        
        # 4. 计算策略表现 (包含交易成本)
        clean_df['signal'] = np.sign(clean_df['factor'] - clean_df['close'])
        
        # 策略的原始收益
        clean_df['strategy_return'] = clean_df['signal'].shift(1) * clean_df['ret']
        
        # 计算交易成本：当信号变化时，产生交易
        trades = clean_df['signal'].diff().abs().shift(1).fillna(0)
        transaction_costs = trades * (transaction_fee_bps / 10000.0)
        clean_df['net_strategy_return'] = clean_df['strategy_return'] - transaction_costs

        # 5. 计算夏普率 (包含扣费前和扣费后)
        gross_sharpe = (clean_df['strategy_return'].mean() / clean_df['strategy_return'].std()) * annualization_factor if clean_df['strategy_return'].std() != 0 else 0
        net_sharpe = (clean_df['net_strategy_return'].mean() / clean_df['net_strategy_return'].std()) * annualization_factor if clean_df['net_strategy_return'].std() != 0 else 0

        results.append({
            'Factor': factor_name,
            'IC': ic,
            'Rank IC': rank_ic,
            'Gross Sharpe': gross_sharpe,
            'Net Sharpe': net_sharpe
        })

    return pd.DataFrame(results)

if __name__ == '__main__':
    # 1. 根据配置的周期加载1分钟数据
    df = load_data_in_period(data_dir, SYMBOL, START_DATE, END_DATE)

    if not df.empty:
        # 2. 定义要测试的因子函数列表
        factor_funcs_to_test = [factor_ma15, factor_ma30] 
        
        # 3. 运行回测
        backtest_results = run_factor_backtest(df, factor_funcs_to_test, TRANSACTION_FEE_BPS)
        
        # 4. 打印格式化的结果
        print("\n--- 因子回测结果 ---")
        print(f"交易手续费: {TRANSACTION_FEE_BPS} bps ({TRANSACTION_FEE_BPS/100}%)")
        print(backtest_results.to_string(index=False))
