import pandas as pd
import os
import glob

# --- 导入你的策略和回测引擎 ---
from backtest.strategies import strategy_simple_ma
from backtest.backtest_engine import run_backtest
from backtest.utils import load_data_in_period

# --- 配置区 ---
SYMBOL = 'BTCUSDT'
START_DATE = '2024-01-08'
END_DATE = '2024-01-15'
INITIAL_CAPITAL = 100000  # 初始资金
TRANSACTION_FEE_BPS = 2   # 手续费 (bps)
REBALANCE_PERIOD_MIN = 60 # 换仓周期（分钟）

# --- 数据加载 (可以从 run_factor_analysis.py 复制或重构为一个公共模块) ---
project_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(project_root, 'data')
symbol_dir = SYMBOL[:3]
data_dir = os.path.join(data_root, symbol_dir)

if __name__ == '__main__':
    # 1. 加载数据
    df = load_data_in_period(data_dir, SYMBOL, START_DATE, END_DATE)
    
    # 2. 运行策略生成信号
    df_with_signals = strategy_simple_ma.generate_signals(df)
    
    # 3. 将带信号的数据送入回测引擎
    results = run_backtest(df_with_signals, INITIAL_CAPITAL, TRANSACTION_FEE_BPS, REBALANCE_PERIOD_MIN)
    
    # 4. 打印结果 (这里的结果会是更详细的策略表现)
    print(results)