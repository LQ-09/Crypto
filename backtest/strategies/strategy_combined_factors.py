import pandas as pd
import numpy as np
from backtest.utils import load_factors_from_config

def generate_signals(df, factor_config_path, quantile_window, long_quantile, short_quantile):
    """
    根据配置文件中的因子，等权相加后，基于滚动分位数生成交易信号。
    
    策略逻辑:
    1. 从 JSON 配置文件加载所有因子函数。
    2. 计算所有因子的值，并进行滚动标准化，然后等权相加得到一个综合因子。
    3. 计算综合因子的滚动分位数，作为动态的交易阈值。
    4. 生成交易信号 (与因子分析脚本逻辑对齐):
       - combined_factor <= q_low: 做多 (signal = 1)
       - combined_factor >= q_high: 做空 (signal = -1)
       - 否则: 空仓 (signal = 0)
    5. 信号延迟一期生效，以避免使用未来数据。

    :param df: 包含OHLCV数据的DataFrame。
    :param factor_config_path: 因子配置文件的路径。
    :param quantile_window: 计算分位数的滚动窗口。
    :param long_quantile: 做多信号的分位数阈值。
    :param short_quantile: 做空信号的分位数阈值。
    :return: 带有 'signal' 列的DataFrame。
    """
    df_factors = pd.DataFrame(index=df.index)
    
    # 1. 加载并计算所有因子
    factor_functions = load_factors_from_config(factor_config_path)
    if not factor_functions:
        print("未能加载任何因子，无法生成信号。")
        df['signal'] = 0
        return df

    for name, func in factor_functions.items():
        try:
            factor_series = func(df)
            # --- 标准化 (关键步骤) ---
            # 为了让不同量纲的因子能被“等权”相加，标准化是必不可少的。
            # 它将每个因子的值都转换为相似的尺度，确保每个因子都有机会影响最终结果。
            # 这里使用滚动标准化(rolling z-score)来避免使用未来数据。窗口期可以根据因子特性调整。
            rolling_window = 1440 # 使用过去24小时(1440分钟)的数据进行标准化
            mean = factor_series.rolling(window=rolling_window, min_periods=60).mean()
            std = factor_series.rolling(window=rolling_window, min_periods=60).std()
            factor_series = (factor_series - mean) / std
            
            df_factors[name] = factor_series
        except Exception as e:
            print(f"计算因子 {name} 时出错: {e}")

    # 2. 等权相加 (fillna(0)可以处理计算初期的NaN，以及某些因子在某些点上无值的情况)
    df['combined_factor'] = df_factors.sum(axis=1, skipna=True).fillna(0)
    
    # 3. 根据滚动分位数生成交易信号 (新逻辑，与因子分析对齐)
    # 仅当组合因子值足够极端时才产生信号，可以有效过滤噪音，减少不必要的交易。
    df['q_low'] = df['combined_factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(long_quantile)
    df['q_high'] = df['combined_factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(short_quantile)
    
    df['signal'] = 0.0
    # 当组合因子值低于下分位数时，我们认为这是强烈的买入信号
    df.loc[df['combined_factor'] <= df['q_low'], 'signal'] = 1.0
    # 当组合因子值高于上分位数时，我们认为这是强烈的卖出信号
    df.loc[df['combined_factor'] >= df['q_high'], 'signal'] = -1.0
    
    # 4. 信号延迟一期生效，避免使用未来数据
    df['signal'] = df['signal'].shift(1).fillna(0)
    
    print(f"多因子组合策略信号已生成。信号分布:\n{df['signal'].value_counts(normalize=True).round(2)}")
    
    return df