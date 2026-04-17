import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class BacktestResult:
    results_df: pd.DataFrame
    metrics: Dict[str, Any]
    benchmark_metrics: Dict[str, Any]
    monte_carlo_results: Optional[Dict] = None
    out_of_sample_metrics: Optional[Dict] = None


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
                         progress_bar=None,
                         warmup_periods: int = 0,
                         apply_position_sizing: bool = True):
        """
        Run a walk-forward backtest on the provided data with position sizing support.
        """
        results = []

        self.last_errors = []
        holding_period = holding_period_bars or max(horizon, 1)
        total_steps = len(df) - start_idx
        
        # Track cumulative PnL for position sizing
        cumulative_pnl = 0.0
        trade_history = []
        
        # Cache for forecasts to speed up backtest
        forecast_cache = {}

        for i in range(start_idx + warmup_periods, len(df)):
            if progress_bar:
                progress_bar.progress((i - start_idx) / total_steps)

            current_df = df.iloc[:i+1]
            entry_idx = i + entry_delay_bars
            exit_idx = entry_idx + holding_period
            if exit_idx >= len(df):
                continue

            try:
                # Use cache key based on last index and horizon
                cache_key = (current_df.index[-1], horizon, tuple(covariate_cols or []))
                if cache_key in forecast_cache:
                    point, quant = forecast_cache[cache_key]
                else:
                    point, quant = self.forecaster.forecast(current_df, horizon=horizon, covariate_cols=covariate_cols)
                    forecast_cache[cache_key] = (point, quant)

                forecast_df = self.forecaster.get_forecast_df(
                    current_df, point, quant, horizon=horizon, interval=df.attrs.get("interval")
                )

                # Pass historical trades for Kelly criterion calculation
                sig_result = self.strategy.generate_signal(
                    current_df, 
                    forecast_df, 
                    covariate_cols,
                    historical_trades=trade_history[-20:] if trade_history else None,
                    data_engine=self.data_engine,
                    ticker=ticker,
                    current_time=current_df.index[-1]
                )
                sig_data = asdict(sig_result)

                entry_price = self._get_execution_price(df, entry_idx)

                # Real-time simulation of exit (check SL/TP before holding period ends)
                exit_price, exit_reason = self._simulate_exit(
                    df, entry_idx, exit_idx, entry_price,
                    sig_result.stop_loss, sig_result.take_profit,
                    is_long=(sig_result.signal == "BUY")
                )
                
                # Calculate gross return
                gross_return = 0.0
                if sig_result.signal == "BUY":
                    gross_return = (exit_price - entry_price) / entry_price
                elif sig_result.signal == "SELL":
                    gross_return = (entry_price - exit_price) / entry_price

                # Apply costs
                cost_decimal = 2 * ((fee_bps + slippage_bps) / 10000.0) if sig_result.signal != "HOLD" else 0.0
                net_return = gross_return - cost_decimal
                
                # Apply position sizing to returns
                if apply_position_sizing and sig_result.signal != "HOLD":
                    position_size_pct = sig_result.suggested_size_pct / 100.0
                    net_return = net_return * position_size_pct

                res = {
                    'signal': sig_result.signal,
                    'confidence': sig_result.confidence,
                    'timesfm_change_pct': sig_result.timesfm_change_pct,
                    'tech_score': sig_result.tech_score,
                    'latest_price': sig_result.latest_price,
                    'predicted_next': sig_result.predicted_next,
                    'expected_edge': sig_result.expected_edge,
                    'uncertainty_width': sig_result.uncertainty_width,
                    'risk_per_unit': sig_result.risk_per_unit,
                    'stop_loss': sig_result.stop_loss,
                    'take_profit': sig_result.take_profit,
                    'suggested_size_pct': sig_result.suggested_size_pct,
                    'trailing_stop': sig_result.trailing_stop,
                    'market_regime': sig_result.market_regime,
                    'date': df.index[i],
                    'entry_date': df.index[entry_idx],
                    'exit_date': df.index[exit_idx],
                    'entry_price': round(float(entry_price), 4),
                    'exit_price': round(float(exit_price), 4),
                    'gross_pnl_pct': round(float(gross_return * 100), 4),
                    'fees_pct': round(float((2 * fee_bps) / 100), 4) if sig_data.signal != "HOLD" else 0.0,
                    'slippage_pct': round(float((2 * slippage_bps) / 100), 4) if sig_data.signal != "HOLD" else 0.0,
                    'net_pnl_pct': round(float(net_return * 100), 4),
                    'pnl_pct': round(float(net_return * 100), 4),
                    'position_size_pct': sig_data.suggested_size_pct if apply_position_sizing else 100.0
                }

                results.append(res)
                
                # Update cumulative PnL and trade history
                if sig_result.signal != "HOLD":
                    cumulative_pnl += net_return * 100
                    trade_history.append(res)

            except Exception as e:
                error_msg = f"Error at index {i}: {e}"
                self.last_errors.append(error_msg)
                self.logger.exception(error_msg)
                continue

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df.set_index('date', inplace=True)

        return results_df

    def calculate_benchmark_returns(self, df: pd.DataFrame, results_df: pd.DataFrame):
        """Calculate buy-and-hold benchmark returns for comparison."""
        if results_df.empty or len(df) < 2:
            return {"total_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0}
        
        # Get the date range of our backtest
        start_date = results_df.index[0]
        end_date = results_df.index[-1]
        
        # Filter df to same period
        benchmark_df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if len(benchmark_df) < 2:
            return {"total_return_pct": 0.0, "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0}
        
        # Calculate buy-and-hold return
        start_price = benchmark_df['close'].iloc[0]
        end_price = benchmark_df['close'].iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
        
        # Calculate daily returns for Sharpe
        daily_returns = benchmark_df['close'].pct_change().dropna()
        avg_return = daily_returns.mean()
        std_return = daily_returns.std()
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        # Calculate max drawdown
        cum_returns = (1 + daily_returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min() * 100
        
        return {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_drawdown, 2)
        }

    def run_monte_carlo(self, results_df: pd.DataFrame, iterations: int = 1000):
        """Run Monte Carlo simulation on backtest results."""
        if results_df.empty or len(results_df) < 10:
            return None
        
        trade_results = results_df[results_df['signal'] != "HOLD"].copy()
        if len(trade_results) < 5:
            return None
        
        pnl_series = trade_results['net_pnl_pct']
        final_pnls = []
        
        for _ in range(iterations):
            # Bootstrap resampling
            sampled_pnls = pnl_series.sample(n=len(pnl_series), replace=True).sum()
            final_pnls.append(sampled_pnls)
        
        final_pnls = np.array(final_pnls)
        
        return {
            "mean_final_pnl": round(np.mean(final_pnls), 2),
            "std_final_pnl": round(np.std(final_pnls), 2),
            "percentile_5": round(np.percentile(final_pnls, 5), 2),
            "percentile_25": round(np.percentile(final_pnls, 25), 2),
            "percentile_75": round(np.percentile(final_pnls, 75), 2),
            "percentile_95": round(np.percentile(final_pnls, 95), 2),
            "probability_positive": round(np.mean(final_pnls > 0) * 100, 1),
            "probability_negative": round(np.mean(final_pnls < 0) * 100, 1)
        }

    def split_out_of_sample(self, df: pd.DataFrame, out_of_sample_ratio: float = 0.3):
        """Split data into in-sample and out-of-sample periods."""
        split_idx = int(len(df) * (1 - out_of_sample_ratio))
        in_sample = df.iloc[:split_idx].copy()
        out_of_sample = df.iloc[split_idx:].copy()
        return in_sample, out_of_sample, split_idx

    def calculate_metrics(self, results_df: pd.DataFrame):
        """
        Calculate summary performance metrics from backtest results.
        """
        if results_df.empty:
            return {}

        trade_results = results_df[results_df['signal'] != "HOLD"].copy()
        total_trades = len(trade_results)
        if total_trades == 0:
            return {"total_trades": 0}

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

        # Sharpe Ratio
        sharpe = (avg_return / std_pnl) * np.sqrt(252) if std_pnl != 0 else 0
        
        # Sortino Ratio
        downside_returns = trade_returns[trade_returns < 0]
        downside_std = downside_returns.std()
        sortino = (avg_return / downside_std) * np.sqrt(252) if downside_std > 0 else (sharpe if avg_return > 0 else 0)

        # Calculate cumulative returns and drawdown
        cum_pnl = trade_results['net_pnl_pct'].cumsum()
        rolling_max = cum_pnl.cummax()
        drawdown = cum_pnl - rolling_max
        max_drawdown = drawdown.min() if not drawdown.empty else 0
        
        # Additional metrics
        avg_win = trade_results.loc[trade_results['net_pnl_pct'] > 0, 'net_pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trade_results.loc[trade_results['net_pnl_pct'] < 0, 'net_pnl_pct'].mean() if len(trade_results) - winning_trades > 0 else 0
        largest_win = trade_results['net_pnl_pct'].max()
        largest_loss = trade_results['net_pnl_pct'].min()
        
        # Profit factor with safety
        if not np.isfinite(profit_factor):
            profit_factor = 0 if gross_loss == 0 else np.inf

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate * 100, 2),
            "total_pnl_pct": round(total_pnl, 2),
            "avg_pnl_pct": round(avg_pnl, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "profit_factor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else "inf",
            "max_drawdown_pct": round(float(max_drawdown), 2),
            "avg_win_pct": round(avg_win, 2) if avg_win else 0,
            "avg_loss_pct": round(avg_loss, 2) if avg_loss else 0,
            "largest_win_pct": round(largest_win, 2),
            "largest_loss_pct": round(largest_loss, 2),
            "expectancy": round(avg_pnl * win_rate + avg_loss * (1 - win_rate), 2) if avg_loss else round(avg_pnl * win_rate, 2)
        }

    def _simulate_exit(self, df, entry_idx, target_exit_idx, entry_price, sl, tp, is_long):
        """Simulates price movement between entry and exit to check for SL/TP hits."""
        for j in range(entry_idx + 1, target_exit_idx + 1):
            if j >= len(df):
                break

            low = df['low'].iloc[j]
            high = df['high'].iloc[j]

            if is_long:
                if sl and low <= sl:
                    return sl, "SL"
                if tp and high >= tp:
                    return tp, "TP"
            else:
                if sl and high >= sl:
                    return sl, "SL"
                if tp and low <= tp:
                    return tp, "TP"

        return self._get_execution_price(df, target_exit_idx), "Target"

    def _get_execution_price(self, df: pd.DataFrame, idx: int):
        if 'open' in df.columns:
            return float(df['open'].iloc[idx])
        return float(df['close'].iloc[idx])
