import torch
import numpy as np
import timesfm
import pandas as pd
import logging
from typing import List, Dict, Optional

class TimesFMForecast:
    def __init__(self, repo_id: str = "google/timesfm-2.5-200m-pytorch"):
        self.repo_id = repo_id
        self.model = None
        self.forecast_config = None
        self.logger = logging.getLogger(__name__)
        self.last_warnings = []

    def load_model(self):
        """
        Load and compile the TimesFM model.
        """
        if self.model is not None:
            return

        torch.set_float32_matmul_precision("high")
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo_id)

        # Default forecast configuration
        self.forecast_config = timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=128,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            return_backcast=True  # Required for XReg
        )
        self.model.compile(self.forecast_config)

    def forecast(self,
                 df: pd.DataFrame,
                 horizon: int = 12,
                 covariate_cols: Optional[List[str]] = None,
                 strict_mode: bool = False):
        """
        Perform forecasting with TimesFM.
        If covariate_cols are provided, uses forecast_with_covariates (XReg).
        """
        if self.model is None:
            self.load_model()

        # Ensure horizon is within max_horizon
        horizon = min(horizon, self.forecast_config.max_horizon)

        # Prepare inputs
        close_prices = df['close'].values.astype(np.float32)

        self.last_warnings = []
        if not covariate_cols:
            # Simple forecast
            point_forecast, quantile_forecast = self.model.forecast(
                horizon=horizon,
                inputs=[close_prices]
            )
        else:
            # Forecast with covariates (XReg)
            # We need to provide covariates for both context AND horizon.
            # Since we dont know future indicator values, we will use the last known value
            # (effectively a persistent covariate) or a simple trend for the horizon.
            # In a real scenario, some covariates (like time-based) are known.
            # For technical indicators, they are NOT known in the future.
            # However, TimesFM XReg requires them for the horizon.

            dynamic_numerical_covariates = {}
            interval = df.attrs.get("interval")
            freq = self._resolve_forecast_frequency(df, interval)
            known_future_covs = self._build_known_future_covariates(df.index, horizon, freq)
            dynamic_numerical_covariates.update(known_future_covs)

            for col in covariate_cols:
                # Try exact match first, then lowercase
                target_col = col
                if target_col not in df.columns:
                    target_col = col.lower()

                if target_col in df.columns:
                    context_values = df[target_col].values.astype(np.float32)
                    # Known-future covariates can be carried as-is for horizon.
                    # Unknown-future technical indicators are excluded by default.
                    if self._is_known_future_covariate(target_col):
                        horizon_values = np.repeat(context_values[-1], horizon).astype(np.float32)
                        full_values = np.concatenate([context_values, horizon_values])
                        dynamic_numerical_covariates[target_col] = [full_values]
                    else:
                        warning_msg = (
                            f"Skipped unknown-future covariate '{col}'. "
                            "Technical indicators are path-dependent and not known for future horizon."
                        )
                        self.last_warnings.append(warning_msg)
                        self.logger.warning(warning_msg)
                else:
                    warning_msg = f"Covariate column '{col}' not found in DataFrame."
                    self.last_warnings.append(warning_msg)
                    self.logger.warning(warning_msg)

            if strict_mode and self.last_warnings:
                raise ValueError("Strict mode enabled and one or more covariates were invalid/unknown.")

            if not dynamic_numerical_covariates:
                # Fallback to simple forecast if no covariates were found
                warning_msg = "No valid covariates found. Falling back to simple forecast."
                self.last_warnings.append(warning_msg)
                self.logger.warning(warning_msg)
                point_forecast, quantile_forecast = self.model.forecast(
                    horizon=horizon,
                    inputs=[close_prices]
                )
            else:
                point_forecast, quantile_forecast = self.model.forecast_with_covariates(
                inputs=[close_prices],
                dynamic_numerical_covariates=dynamic_numerical_covariates,
                xreg_mode="xreg + timesfm"
            )
            # forecast_with_covariates returns a list of arrays
            point_forecast = np.array(point_forecast)
            quantile_forecast = np.array(quantile_forecast)

        return point_forecast, quantile_forecast

    def get_forecast_df(self,
                        df: pd.DataFrame,
                        point_forecast: np.ndarray,
                        quantile_forecast: np.ndarray,
                        horizon: int,
                        interval: Optional[str] = None):
        """
        Combine historical data and forecast into a single DataFrame for visualization.
        """
        last_date = df.index[-1]

        freq = self._resolve_forecast_frequency(df, interval or df.attrs.get("interval"))

        forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        forecast_df = pd.DataFrame(index=forecast_dates)
        forecast_df['point_forecast'] = point_forecast[0][:horizon]

        # Quantiles: 0=mean, 1=q10, 5=median, 9=q90
        forecast_df['lower_80'] = quantile_forecast[0, :horizon, 1]
        forecast_df['upper_80'] = quantile_forecast[0, :horizon, 9]

        return forecast_df

    def _resolve_forecast_frequency(self, df: pd.DataFrame, interval: Optional[str]):
        interval_map = {
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
            "1wk": "1W",
            "1mo": "1M",
        }
        if interval in interval_map:
            return interval_map[interval]
        if hasattr(df.index, "freq") and df.index.freq is not None:
            return df.index.freq
        inferred = pd.infer_freq(df.index)
        if inferred is not None:
            return inferred
        return df.index[-1] - df.index[-2]

    def _build_known_future_covariates(self, index: pd.Index, horizon: int, freq):
        last_date = index[-1]
        future_index = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        full_index = index.append(future_index)
        return {
            "day_of_week": [full_index.dayofweek.astype(np.float32).values],
            "month": [full_index.month.astype(np.float32).values],
            "is_month_end": [full_index.is_month_end.astype(np.float32).values],
        }

    def _is_known_future_covariate(self, col_name: str):
        known_prefixes = ("day_", "month", "is_", "holiday", "weekday", "hour")
        return col_name.lower().startswith(known_prefixes)
