# Crypto Factor & Position-Sizing Research

Maintained by [Ling Qi](https://github.com/LQ-09).

A Python research prototype for studying minute-bar crypto factors, combining rolling standardized signals, and testing how position sizing and trading costs change simulated outcomes.

## Method and data

The downloader requests public Binance candles and writes weekly CSV files. Factor scripts calculate diagnostics against future returns. The combined strategy standardizes each selected factor over a rolling window, adds them with equal weights, and maps rolling quantiles into long/short position strength. Signals are shifted by one input row before the engine resamples prices and positions.

The engine charges configured basis-point costs on absolute position changes, applies a leverage multiplier, and reports simulated return, annualized Sharpe, drawdown and active-period win rate. The factor-analysis path uses a different cost calculation; its curves are not directly interchangeable with the combined-strategy curves.

## Source entry points

```sh
python run_data_download.py
python run_factor_analysis.py
python run_strategy_backtest.py
```

Run from the repository root after reviewing symbols, dates, windows, fees and leverage in the scripts. Imports include pandas, NumPy, requests, tqdm, matplotlib and seaborn. No locked environment is supplied. Downloads access the network and write local files.

## Recorded iteration

| Date | Change | Evidence |
| --- | --- | --- |
| 2025-07-22 | Extended single-factor diagnostics, metrics and charts | [change](https://github.com/LQ-09/Crypto/commit/f986ced488abdd82461a8e71fa5374abccbfe441) |
| 2025-07-23 | Compared runs with and without fees | [change](https://github.com/LQ-09/Crypto/commit/847f3c6b2dbfed844f3921fd5266401937a53a70) |
| 2025-07-23 | Added strategy framework and dynamic position experiments | [framework](https://github.com/LQ-09/Crypto/commit/1c2cc93c5ee0a05dad429dba8f35c84103539043), [sizing](https://github.com/LQ-09/Crypto/commit/cf4523e0384b2cdf82d5d519593d41ee112c917b) |
| 2025-07-23 | Changed the prediction period and removed selected saved outputs | [period](https://github.com/LQ-09/Crypto/commit/93b44c3ae0859a22bcbc7a31ff8dbfd4b2ce3c55), [output cleanup](https://github.com/LQ-09/Crypto/commit/8ed2a1722d6c96b6daf5c577231d8ed5219652d9) |

The inspected repository history is a short research iteration in July 2025. It does not demonstrate continuous subsequent maintenance.

## Results and limits

The project produces exploratory charts and metrics; no validated profitability or live execution is claimed. Historical charts include both favorable and unfavorable outcomes and were not recomputed for this presentation.

A one-row signal delay does not by itself validate timing after resampling: the position/return alignment needs explicit boundary tests. Funding, slippage, liquidation, market impact, survivorship and out-of-sample robustness are not established. The “win rate” is an active-period statistic, not a closed-trade win rate.

## Next steps

Test position/return alignment on tiny deterministic inputs; reconcile fee conventions; add walk-forward evaluation and a frozen dependency environment; save input hashes and parameter manifests; report gross/net returns and cost sensitivity together. These are proposed validations, not demonstrated performance gains.
