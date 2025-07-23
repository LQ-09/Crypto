import pandas as pd
import os
import glob
import json
import importlib
from datetime import datetime, timedelta, timezone
# 使用相对导入，因为data_download和utils在同一个backtest包下
from .data_download import fetch_binance_klines_min

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

        if not os.path.exists(file_path):
            print(f"数据文件不存在，正在下载: {fname}...")
            try:
                df_week = fetch_binance_klines_min(symbol, current_week_start, current_week_end)
                if not df_week.empty:
                    df_week.to_csv(file_path, index=False)
                    print(f"已保存: {file_path} ({len(df_week)} 行)")
                else:
                    # 创建一个空文件，防止下次运行时再次尝试下载无效数据
                    open(file_path, 'a').close()
            except Exception as e:
                print(f"下载数据时发生错误: {e}")

        # 移至下一周
        current_week_start = current_week_end