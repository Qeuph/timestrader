import pandas as pd
import numpy as np
import logging

class Backtester:
    def __init__(self, data_engine, forecaster, strategy):
        self.data_engine = data_engine
        self.forecaster = forecaster
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        self.last_errors = []

    def run_walk_forward(self,
                         ticker: str,
                         df: pd.DataFrame,
                         start_idx: int,
                         horizon: int,
                         covariate_cols: list,
                         entry_delay_bars: int = 1,
                         holding_period_bars: int = None,
                         fee_bps: float = 5.0,
                         slippage_bps: float = 5.0,
                         progress_bar=None):
        """
        Run a walk-forward backtest on the provided data.
        """
        results = []

        # We start from start_idx and move one step at a time
        # At each step, we use data up to i to forecast i+horizon
        # And generate a signal

        self.last_errors = []
        holding_period = holding_period_bars or max(horizon, 1)
        total_steps = len(df) - start_idx
        for i in range(start_idx, len(df)):
            if progress_bar:
                progress_bar.progress((i - start_idx) / total_steps)

            current_df = df.iloc[:i+1]
            entry_idx = i + entry_delay_bars
            exit_idx = entry_idx + holding_period
            if exit_idx >= len(df):
                continue

            try:
                point, quant = self.forecaster.forecast(current_df, horizon=horizon, covariate_cols=covariate_cols)
                forecast_df = self.forecaster.get_forecast_df(
                    current_df, point, quant, horizon=horizon, interval=df.attrs.get("interval")
                )

                sig_data = self.strategy.generate_signal(current_df, forecast_df, covariate_cols)

                entry_price = self._get_execution_price(df, entry_idx)
                exit_price = self._get_execution_price(df, exit_idx)
                gross_return = 0.0
                if sig_data['signal'] == "BUY":
                    gross_return = (exit_price - entry_price) / entry_price
                elif sig_data['signal'] == "SELL":
                    gross_return = (entry_price - exit_price) / entry_price

                cost_decimal = 2 * ((fee_bps + slippage_bps) / 10000.0) if sig_data['signal'] != "HOLD" else 0.0
                net_return = gross_return - cost_decimal

                res = sig_data.copy()
                res['date'] = df.index[i]
                res['entry_date'] = df.index[entry_idx]
                res['exit_date'] = df.index[exit_idx]
                res['entry_price'] = round(float(entry_price), 4)
                res['exit_price'] = round(float(exit_price), 4)
                res['gross_pnl_pct'] = round(float(gross_return * 100), 4)
                res['fees_pct'] = round(float((2 * fee_bps) / 100), 4) if sig_data['signal'] != "HOLD" else 0.0
                res['slippage_pct'] = round(float((2 * slippage_bps) / 100), 4) if sig_data['signal'] != "HOLD" else 0.0
                res['net_pnl_pct'] = round(float(net_return * 100), 4)
                res['pnl_pct'] = res['net_pnl_pct']

                results.append(res)
            except Exception as e:
                error_msg = f"Error at index {i}: {e}"
                self.last_errors.append(error_msg)
                self.logger.exception(error_msg)
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

        trade_results = results_df[results_df['signal'] != "HOLD"].copy()
        winning_trades = len(trade_results[trade_results['net_pnl_pct'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = trade_results['net_pnl_pct'].sum()
        avg_pnl = trade_results['net_pnl_pct'].mean()
        gross_profit = trade_results.loc[trade_results['net_pnl_pct'] > 0, 'net_pnl_pct'].sum()
        gross_loss = trade_results.loc[trade_results['net_pnl_pct'] < 0, 'net_pnl_pct'].sum()
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.inf

        trade_returns = trade_results['net_pnl_pct'] / 100.0
        avg_return = trade_returns.mean()
        std_pnl = trade_returns.std()
        sharpe = (avg_return / std_pnl) * np.sqrt(252) if std_pnl != 0 else 0
        cum = trade_results['net_pnl_pct'].cumsum()
        drawdown = cum - cum.cummax()
        max_drawdown = drawdown.min() if not drawdown.empty else 0

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate * 100, 2),
            "total_pnl_pct": round(total_pnl, 2),
            "avg_pnl_pct": round(avg_pnl, 2),
            "sharpe_ratio": round(sharpe, 2),
            "profit_factor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else "inf",
            "max_drawdown_pct": round(float(max_drawdown), 2)
        }

    def _get_execution_price(self, df: pd.DataFrame, idx: int):
        if 'open' in df.columns:
            return float(df['open'].iloc[idx])
        return float(df['close'].iloc[idx])
