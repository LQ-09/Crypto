import pandas as pd
import numpy as np
from backtest.utils import load_factors_from_config

def generate_signals(df, factor_config_path, quantile_window, long_quantile, short_quantile, std_window):
    """
    根据配置文件中的因子，等权相加后，基于滚动分位数生成交易信号。
    
    策略逻辑:
    1. 从 JSON 配置文件加载所有因子函数。
    2. 计算所有因子的值，并进行滚动标准化，然后等权相加得到一个综合因子。
    3. 计算综合因子的滚动分位数，定义交易区间。
    4. **新增：根据因子值在交易区间内的相对强度，计算动态仓位。**
       - 信号强度与因子值超过开仓阈值的程度成正比。
       - 例如，因子值越接近分布的底部，做多仓位越大（趋近于1）。
       - 因子值越接近分布的顶部，做空仓位越大（趋近于-1）。
    5. 信号延迟一期生效，以避免使用未来数据。

    :param df: 包含OHLCV数据的DataFrame。
    :param factor_config_path: 因子配置文件的路径。
    :param quantile_window: 计算分位数的滚动窗口。
    :param long_quantile: 做多信号的分位数阈值。
    :param short_quantile: 做空信号的分位数阈值。
    :param std_window: 因子标准化（Z-Score）的滚动窗口。
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
            # 使用更稳健的min_periods，例如窗口的1/24，而不是硬编码的60
            mean = factor_series.rolling(window=std_window, min_periods=max(1, std_window // 24)).mean()
            std = factor_series.rolling(window=std_window, min_periods=max(1, std_window // 24)).std()
            factor_series = (factor_series - mean) / std
            
            df_factors[name] = factor_series
        except Exception as e:
            print(f"计算因子 {name} 时出错: {e}")

    # 2. 等权相加 (fillna(0)可以处理计算初期的NaN，以及某些因子在某些点上无值的情况)
    df['combined_factor'] = df_factors.sum(axis=1, skipna=True).fillna(0)
    
    # 3. 计算交易信号的动态边界
    # q_low/q_high 是开仓的“门槛”
    df['q_low'] = df['combined_factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(long_quantile)
    df['q_high'] = df['combined_factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(short_quantile)
    # q_min/q_max 定义了仓位从0到100%的完整区间，使用1%和99%分位数比直接用min/max更稳健，可以避免极端异常值的干扰。
    df['q_min'] = df['combined_factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(0.01)
    df['q_max'] = df['combined_factor'].rolling(window=quantile_window, min_periods=quantile_window // 2).quantile(0.99)
    
    df['signal'] = 0.0

    # 4. 计算动态仓位
    # 做多仓位：当因子值低于开仓阈值(q_low)时
    long_cond = df['combined_factor'] <= df['q_low']
    long_numerator = df['q_low'] - df['combined_factor']
    long_denominator = df['q_low'] - df['q_min']
    long_denominator.replace(0, np.nan, inplace=True) # 避免除以零
    long_strength = (long_numerator / long_denominator).clip(0, 1) # 归一化并限制在[0, 1]
    df.loc[long_cond, 'signal'] = long_strength

    # 做空仓位：当因子值高于开仓阈值(q_high)时
    short_cond = df['combined_factor'] >= df['q_high']
    short_numerator = df['combined_factor'] - df['q_high']
    short_denominator = df['q_max'] - df['q_high']
    short_denominator.replace(0, np.nan, inplace=True) # 避免除以零
    short_strength = (short_numerator / short_denominator).clip(0, 1) # 归一化并限制在[0, 1]
    df.loc[short_cond, 'signal'] = -short_strength
    
    df['signal'] = df['signal'].fillna(0) # 将计算中可能出现的NaN填充为0
    
    # 5. 信号延迟一期生效，避免使用未来数据
    df['signal'] = df['signal'].shift(1).fillna(0)
    
    print(f"多因子组合策略信号已生成。信号分布:\n{df['signal'].value_counts(normalize=True).round(2)}")
    
    return df