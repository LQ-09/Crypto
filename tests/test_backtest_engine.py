"""Synthetic, offline regression tests for initial transaction-cost accounting."""
import unittest

import pandas as pd

from backtest.backtest_engine import run_backtest


class InitialTransactionCostTests(unittest.TestCase):
    def run_case(self, signals, fee_bps=10, leverage=1.0):
        frame = pd.DataFrame(
            {"close": [100.0] * len(signals), "signal": signals},
            index=pd.date_range("2025-01-01", periods=len(signals), freq="min"),
        )
        original = frame.copy(deep=True)
        result, detail = run_backtest(frame, 1000, fee_bps, 1, leverage)
        pd.testing.assert_frame_equal(frame, original)
        return result, detail

    def test_initial_long_and_short_entry_are_charged(self):
        for signal in (1.0, -1.0):
            with self.subTest(signal=signal):
                result, detail = self.run_case([signal, signal])
                self.assertAlmostEqual(detail["net_strategy_ret"].iloc[0], -0.001)
                self.assertAlmostEqual(result["equity_curve"].iloc[-1], 999.0)
                self.assertAlmostEqual(result["total_return"], -0.001)

    def test_initial_cost_counts_as_drawdown_from_initial_capital(self):
        result, _ = self.run_case([1.0, 1.0])
        self.assertAlmostEqual(result["max_drawdown"], 0.001)

    def test_initial_turnover_uses_position_size_and_leverage(self):
        result, _ = self.run_case([0.5, 0.5], leverage=3.0)
        self.assertAlmostEqual(result["equity_curve"].iloc[-1], 998.5)

    def test_reversal_charges_both_sides_without_double_entry(self):
        result, detail = self.run_case([1.0, -1.0, -1.0])
        self.assertEqual(detail["costs"].tolist(), [0.001, 0.002, 0.0])
        self.assertAlmostEqual(result["equity_curve"].iloc[-1], 1000 * 0.999 * 0.998)

    def test_flat_and_fee_free_paths_stay_unchanged(self):
        for signals, fee in (([0.0, 0.0], 10), ([1.0, 1.0], 0)):
            with self.subTest(signals=signals, fee=fee):
                result, _ = self.run_case(signals, fee_bps=fee)
                self.assertEqual(result["equity_curve"].iloc[-1], 1000)
                self.assertEqual(result["max_drawdown"], 0)

    def test_one_observation_still_charges_entry(self):
        result, _ = self.run_case([1.0])
        self.assertAlmostEqual(result["equity_curve"].iloc[0], 999.0)
        self.assertEqual(result["annualized_sharpe_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
