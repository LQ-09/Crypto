import pandas as pd
import os
import glob
import json
import importlib

def load_data_in_period(data_directory, symbol, start_date, end_date):
    """加载并合并指定币种在特定时间段内的数据文件"""
    search_path = os.path.join(data_directory, f"{symbol}_1min_*.csv")
    all_files = sorted(glob.glob(search_path))
    if not all_files:
        print(f"在 {data_directory} 中未找到 {symbol} 的数据文件")
        return pd.DataFrame()

    df_list = [pd.read_csv(f, parse_dates=['timestamp']) for f in all_files]
    df = pd.concat(df_list, ignore_index=True).set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    # 根据指定的开始和结束日期筛选数据
    return df.loc[start_date:end_date]

def load_factors_from_config(config_path):
    """
    从JSON配置文件中动态加载因子函数。
    :param config_path: factor_config.json 文件的路径。
    :return: 一个字典，键是因子名，值是因子函数对象。
    """
    loaded_factors = {}
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        for factor_name, info in config.items():
            module = importlib.import_module(info['module'])
            func = getattr(module, info['function'])
            loaded_factors[factor_name] = func
        print(f"成功从 {config_path} 加载 {len(loaded_factors)} 个因子。")
    except FileNotFoundError:
        print(f"错误: 因子配置文件未找到于 {config_path}")
    except Exception as e:
        print(f"加载因子时出错: {e}")
    return loaded_factors