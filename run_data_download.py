import os
from datetime import datetime, timezone, timedelta
from backtest.data_download import fetch_binance_klines_min
# from backtest import run_backtest
import pandas as pd
from tqdm import tqdm

project_root = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(project_root, 'data')

if __name__ == '__main__':
    # 下载2024年第一周到第四周
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2024, 1, 29, tzinfo=timezone.utc)  # 第五周的起点，不包含
    symbol = 'BTCUSDT'
    symbol_dir = symbol[:3]  # 取前3位作为币种目录（如BTC、ETH）
    data_dir = os.path.join(data_root, symbol_dir)
    os.makedirs(data_dir, exist_ok=True)

    cur_start = start_time
    weeks = []
    while cur_start < end_time:
        cur_end = min(cur_start + timedelta(days=7), end_time)
        weeks.append((cur_start, cur_end))
        cur_start = cur_end

    for cur_start, cur_end in tqdm(weeks, desc='下载进度'):
        fname = f"{symbol}_1min_{cur_start.strftime('%Y%m%d')}_{cur_end.strftime('%Y%m%d')}.csv"
        csv_path = os.path.join(data_dir, fname)
        if os.path.exists(csv_path):
            print(f"已存在，跳过: {csv_path}")
            continue
        df_1min = fetch_binance_klines_min(symbol, cur_start, cur_end)
        df_1min.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path} ({len(df_1min)} rows)")

    # 3. 读取数据并回测
    # df = pd.read_csv(csv_path)
    # result = run_backtest(df)
    # print(result)
