import pandas as pd
import numpy as np
from tqdm import tqdm

class Backtester:
    def __init__(self, data_engine, forecaster, strategy):
        self.data_engine = data_engine
        self.forecaster = forecaster
        self.strategy = strategy

    def run_walk_forward(self,
                         ticker: str,
                         df: pd.DataFrame,
                         start_idx: int,
                         horizon: int,
                         covariate_cols: list,
                         progress_bar=None):
        """
        Run a walk-forward backtest on the provided data.
        """
        results = []

        # We start from start_idx and move one step at a time
        # At each step, we use data up to i to forecast i+horizon
        # And generate a signal

        total_steps = len(df) - 1 - start_idx
        for i in range(start_idx, len(df) - 1):
            if progress_bar:
                progress_bar.progress((i - start_idx) / total_steps)

            current_df = df.iloc[:i+1]

            # 3. Forecast
            try:
                point, quant = self.forecaster.forecast(current_df, horizon=horizon, covariate_cols=covariate_cols)
                forecast_df = self.forecaster.get_forecast_df(current_df, point, quant, horizon=horizon)

                # 4. Generate Signal
                sig_data = self.strategy.generate_signal(current_df, forecast_df, covariate_cols)

                # Record result
                actual_next_price = df['close'].iloc[i+1]
                actual_change = (actual_next_price - sig_data['latest_price']) / sig_data['latest_price']

                res = sig_data.copy()
                res['date'] = df.index[i]
                res['actual_next_price'] = round(float(actual_next_price), 2)
                res['actual_change_pct'] = round(float(actual_change * 100), 2)

                # Simple PnL calculation (assume entry at latest_price, exit at actual_next_price)
                if res['signal'] == "BUY":
                    res['pnl_pct'] = res['actual_change_pct']
                elif res['signal'] == "SELL":
                    res['pnl_pct'] = -res['actual_change_pct']
                else:
                    res['pnl_pct'] = 0

                results.append(res)
            except Exception as e:
                print(f"Error at index {i}: {e}")
                continue

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.set_index('date', inplace=True)

        return results_df

    def calculate_metrics(self, results_df: pd.DataFrame):
        """
        Calculate summary performance metrics from backtest results.
        """
        if results_df.empty:
            return {}

        total_trades = len(results_df[results_df['signal'] != "HOLD"])
        if total_trades == 0:
            return {"total_trades": 0}

        winning_trades = len(results_df[results_df['pnl_pct'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = results_df['pnl_pct'].sum()
        avg_pnl = results_df['pnl_pct'].mean()

        # Simple Sharpe-ish ratio (assuming daily if interval is 1d)
        std_pnl = results_df['pnl_pct'].std()
        sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl != 0 else 0

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate * 100, 2),
            "total_pnl_pct": round(total_pnl, 2),
            "avg_pnl_pct": round(avg_pnl, 2),
            "sharpe_ratio": round(sharpe, 2)
        }
