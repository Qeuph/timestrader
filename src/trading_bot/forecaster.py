import torch
import numpy as np
import timesfm
import pandas as pd
from typing import List, Dict, Optional

class TimesFMForecast:
    def __init__(self, repo_id: str = "google/timesfm-2.5-200m-pytorch"):
        self.repo_id = repo_id
        self.model = None
        self.forecast_config = None

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
                 covariate_cols: Optional[List[str]] = None):
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
            for col in covariate_cols:
                # Try exact match first, then lowercase
                target_col = col
                if target_col not in df.columns:
                    target_col = col.lower()

                if target_col in df.columns:
                    context_values = df[target_col].values.astype(np.float32)
                    # Extend with last value for the horizon
                    horizon_values = np.full(horizon, context_values[-1], dtype=np.float32)
                    full_values = np.concatenate([context_values, horizon_values])
                    dynamic_numerical_covariates[target_col] = [full_values]
                else:
                    print(f"Warning: Covariate column {col} not found in DataFrame.")

            if not dynamic_numerical_covariates:
                # Fallback to simple forecast if no covariates were found
                print("No valid covariates found. Falling back to simple forecast.")
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
                        horizon: int):
        """
        Combine historical data and forecast into a single DataFrame for visualization.
        """
        last_date = df.index[-1]

        # Robustly determine frequency
        if hasattr(df.index, 'freq') and df.index.freq is not None:
            freq = df.index.freq
        else:
            freq = pd.infer_freq(df.index)

        if freq is None:
            # Fallback to calculating the difference between last two timestamps
            freq = df.index[-1] - df.index[-2]

        forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]

        forecast_df = pd.DataFrame(index=forecast_dates)
        forecast_df['point_forecast'] = point_forecast[0][:horizon]

        # Quantiles: 0=mean, 1=q10, 5=median, 9=q90
        forecast_df['lower_80'] = quantile_forecast[0, :horizon, 1]
        forecast_df['upper_80'] = quantile_forecast[0, :horizon, 9]

        return forecast_df
