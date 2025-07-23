import pandas as pd
import os
import matplotlib.pyplot as plt

# --- 导入你的策略、回测引擎和工具函数 ---
from backtest.strategies import strategy_combined_factors
from backtest.backtest_engine import run_backtest
from backtest.utils import load_data_in_period, ensure_data_is_available # 统一使用工具函数

# --- 配置区 (与因子分析脚本保持相似结构) ---
SYMBOL = 'ETHUSDT'
START_DATE = '2025-01-01'
END_DATE = '2025-06-01'
INITIAL_CAPITAL = 2000
TRANSACTION_FEE_BPS = 0   # 手续费 (bps), e.g., 2 bps = 0.02%
REBALANCE_PERIOD_MIN = 60 # 换仓周期（分钟）

# 新增：交易信号生成阈值 (与因子分析脚本对齐)
QUANTILE_WINDOW = 24 * 60    # 用于计算分位数信号的滚动窗口 (1440分钟 = 1天)
LONG_ENTRY_QUANTILE = 0.10   # 做多信号的因子值分位数阈值 (e.g., 0.10 for bottom 10%)
SHORT_ENTRY_QUANTILE = 0.90  # 做空信号的因子值分位数阈值 (e.g., 0.90 for top 10%)

# 新增：最大杠杆/仓位。策略的实际仓位 = 信号强度 * LEVERAGE
LEVERAGE = 1.0 # 1.0 表示最强信号时开满仓，0.5表示最强信号时开半仓

# --- 路径配置 (与因子分析脚本保持一致) ---
project_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(project_root, 'data')
# 策略回测也需要因子配置文件
config_path = os.path.join(project_root, 'backtest', 'factor_config.json')
result_dir = os.path.join(project_root, 'backtest', 'strategy_result') # 为策略结果创建单独目录
os.makedirs(result_dir, exist_ok=True)
symbol_dir = SYMBOL[:3]
data_dir = os.path.join(data_root, symbol_dir)

if __name__ == '__main__':
    # 1. 确保数据可用 (与因子分析脚本逻辑统一)
    ensure_data_is_available(SYMBOL, START_DATE, END_DATE, data_dir)

    # 2. 加载数据
    print(f"\n加载数据: {SYMBOL} from {START_DATE} to {END_DATE}...")
    df = load_data_in_period(data_dir, SYMBOL, START_DATE, END_DATE)

    if df.empty:
        print("数据为空，程序退出。请检查数据路径和日期范围。")
    else:
        # 3. 运行策略生成信号
        print("\n正在生成多因子组合策略信号...")
        # 使用我们定义的多因子组合策略
        df_with_signals = strategy_combined_factors.generate_signals(
            df,
            config_path,
            QUANTILE_WINDOW,
            LONG_ENTRY_QUANTILE,
            SHORT_ENTRY_QUANTILE
        )

        # 4. 将带信号的数据送入回测引擎
        print("\n开始回测...")
        results = run_backtest(
            df_with_signals,
            INITIAL_CAPITAL,
            TRANSACTION_FEE_BPS,
            REBALANCE_PERIOD_MIN,
            leverage=LEVERAGE
        )

        # 5. 打印和可视化结果 (与因子分析脚本的输出风格统一)
        print("\n--- 策略回测结果 ---")
        print(f"总回报: {results['total_return']:.2%}")
        print(f"年化夏普率: {results['annualized_sharpe_ratio']:.2f}")
        print(f"最大回撤: {results['max_drawdown']:.2%}")
        print(f"胜率: {results['win_rate']:.2%}")

        # 6. 绘制并保存累计收益曲线图
        print("\n正在生成累计收益曲线图...")
        plt.figure(figsize=(12, 6))
        results['cumulative_returns'].plot(title=f'Strategy Cumulative Returns - {SYMBOL}', grid=True)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        
        save_path = os.path.join(result_dir, f'strategy_pnl_{SYMBOL}_{START_DATE}_to_{END_DATE}.png')
        plt.savefig(save_path)
        print(f"累计收益曲线图已保存至: {save_path}")
        plt.show()