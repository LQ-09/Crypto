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
TRANSACTION_FEE_BPS = 10   # 手续费 (bps), e.g., 10 bps = 0.1%
REBALANCE_PERIOD_MIN = 1440 # 换仓周期（分钟）, 1440分钟 = 24小时

# 新增：交易信号生成阈值 (与因子分析脚本对齐)
QUANTILE_WINDOW = 24 * 60    # 用于计算分位数信号的滚动窗口 (1440分钟 = 1天)
LONG_ENTRY_QUANTILE = 0.10   # 做多信号的因子值分位数阈值 (e.g., 0.10 for bottom 10%)
SHORT_ENTRY_QUANTILE = 0.90  # 做空信号的因子值分位数阈值 (e.g., 0.90 for top 10%)

# 新增：因子标准化窗口
FACTOR_STD_WINDOW = 24 * 60  # 用于计算因子Z-Score的滚动窗口 (1440分钟 = 1天)

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
            SHORT_ENTRY_QUANTILE,
            FACTOR_STD_WINDOW
        )

        # 4. 将带信号的数据送入回测引擎
        print("\n开始回测...")
        # 注意：假设 run_backtest 已修改为返回 (results, backtest_df)
        results, backtest_df = run_backtest(
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

        # 6. 绘制并保存累计收益曲线图 (简化版)
        print("\n正在生成累计收益曲线图...")
        plt.figure(figsize=(14, 8))
        results['cumulative_returns'].plot(title=f'Strategy Cumulative Returns - {SYMBOL}', grid=True)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.tight_layout()
        
        save_path = os.path.join(result_dir, f'strategy_pnl_{SYMBOL}_{START_DATE}_to_{END_DATE}.png')
        plt.savefig(save_path)
        print(f"累计收益曲线图已保存至: {save_path}")
        plt.show()

        # 7. 新增：因子贡献度归因分析 (PnL Attribution)
        print("\n--- 正在生成因子贡献度归因分析图 ---")
        
        # 7.1 识别出所有标准化后的独立因子列
        factor_std_cols = [col for col in df_with_signals.columns if col.startswith('factor_std_')]
        
        if not factor_std_cols:
            print("未找到标准化后的独立因子列 (以 'factor_std_' 开头)，跳过归因分析。")
        else:
            # 7.2 计算每个因子的独立 PnL 贡献
            # 核心思想: 每个因子的(毛利)贡献 ≈ 它的标准化因子值 * 周期回报
            # 这是一个行业内常用的近似归因方法，能有效反映各因子贡献方向和大小。
            
            pnl_attribution = pd.DataFrame(index=backtest_df.index)
            
            # 将 backtest_df 中的 period_return 合并过来，确保时间对齐
            aligned_df = df_with_signals.join(backtest_df['period_return'], how='right')

            for col in factor_std_cols:
                factor_name = col.replace('factor_std_', '')
                # 因子贡献 = 上一期的因子值 * 本期的回报率
                # 这里我们不考虑杠杆和手续费，只看毛利贡献，以评估因子原始能力
                pnl_attribution[f'{factor_name}_contrib'] = aligned_df[col].shift(1) * aligned_df['period_return']

            # 7.3 计算累计贡献
            # 我们这里计算的是每个因子贡献的累加值，而不是复利曲线，这样更容易看出绝对贡献大小
            cumulative_attribution = pnl_attribution.cumsum()

            # 7.4 绘图
            plt.figure(figsize=(14, 8))
            
            # 绘制每个因子的累计贡献曲线
            cumulative_attribution.plot(ax=plt.gca(), colormap='viridis')
            
            plt.title(f'PnL Attribution by Factor - {SYMBOL}')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Contribution (Gross PnL)')
            plt.legend(title='Factor Contribution')
            plt.grid(True)
            plt.tight_layout()
            
            save_path = os.path.join(result_dir, f'pnl_attribution_{SYMBOL}_{START_DATE}_to_{END_DATE}.png')
            plt.savefig(save_path)
            print(f"因子贡献度归因图已保存至: {save_path}")
            plt.show()