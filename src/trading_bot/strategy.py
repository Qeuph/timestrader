import pandas as pd
import numpy as np
from typing import Dict, Any

class SignalStrategy:
    def __init__(self,
                 forecast_threshold_pct: float = 1.0,
                 max_risk_per_trade_pct: float = 1.0,
                 max_position_size_pct: float = 20.0):
        self.forecast_threshold_pct = forecast_threshold_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_position_size_pct = max_position_size_pct

    def update_config(self,
                      forecast_threshold_pct: float,
                      max_risk_per_trade_pct: float,
                      max_position_size_pct: float):
        self.forecast_threshold_pct = forecast_threshold_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_position_size_pct = max_position_size_pct

    def generate_signal(self,
                        df: pd.DataFrame,
                        forecast_df: pd.DataFrame,
                        selected_indicators: list):
        """
        Generate a Buy/Sell/Hold signal based on TimesFM forecast and technical indicators.
        """
        latest_price = float(df['close'].iloc[-1])

        # Multi-horizon Analysis: Look at the mean of the forecast horizon
        predicted_price_next = float(forecast_df['point_forecast'].iloc[0])
        avg_predicted_price = float(forecast_df['point_forecast'].mean())

        # Calculate changes for both next-step and full-horizon
        predicted_change_next = (predicted_price_next - latest_price) / latest_price
        predicted_change_horizon = (avg_predicted_price - latest_price) / latest_price

        # Combined TimesFM Signal (Weighted average of next-step and horizon trend)
        # We give 40% weight to immediate next step and 60% to the overall horizon trend
        combined_forecast_change = (0.4 * predicted_change_next) + (0.6 * predicted_change_horizon)

        threshold = self.forecast_threshold_pct / 100.0
        timesfm_signal = 0
        if combined_forecast_change > threshold:
            timesfm_signal = 1
        elif combined_forecast_change < -threshold:
            timesfm_signal = -1

        # Technical Indicators Signal
        tech_signal = 0
        active_indicators = 0

        # Trend filters
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            active_indicators += 1
            if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
                tech_signal += 1
            else:
                tech_signal -= 1

        # Momentum filters
        if 'RSI_14' in df.columns:
            active_indicators += 1
            rsi = df['RSI_14'].iloc[-1]
            if rsi < 30: # Oversold
                tech_signal += 1
            elif rsi > 70: # Overbought
                tech_signal -= 1
            else:
                # Neutral RSI but direction matters
                if df['RSI_14'].iloc[-1] > df['RSI_14'].iloc[-2]:
                    tech_signal += 0.5
                else:
                    tech_signal -= 0.5

        if 'MACDh_12_26_9' in df.columns:
            active_indicators += 1
            if df['MACDh_12_26_9'].iloc[-1] > 0:
                tech_signal += 1
            else:
                tech_signal -= 1

        # Final Synthesis
        # We give weight to both TimesFM and Tech Indicators
        # Normalized Tech Signal
        norm_tech = tech_signal / active_indicators if active_indicators > 0 else 0

        combined_score = (0.6 * timesfm_signal) + (0.4 * norm_tech)

        signal = "HOLD"
        interval_width = float(
            max(
                forecast_df['upper_80'].iloc[0] - forecast_df['lower_80'].iloc[0],
                1e-8,
            )
        )
        expected_edge = abs(predicted_price_next - latest_price)
        uncertainty_score = min(expected_edge / interval_width, 3.0) / 3.0
        confidence = min(1.0, 0.5 * abs(combined_score) + 0.5 * uncertainty_score)

        if combined_score > 0.3:
            signal = "BUY"
        elif combined_score < -0.3:
            signal = "SELL"

        # Risk Management: SL/TP based on ATR
        atr_col = 'ATR_14' if 'ATR_14' in df.columns else 'ATRr_14'
        if atr_col in df.columns:
            atr = float(df[atr_col].iloc[-1])
            if signal == "BUY":
                stop_loss = latest_price - (2 * atr)
                take_profit = latest_price + (4 * atr)
            elif signal == "SELL":
                stop_loss = latest_price + (2 * atr)
                take_profit = latest_price - (4 * atr)
            else:
                stop_loss = None
                take_profit = None
        else:
            # Fallback if ATR is missing
            if signal == "BUY":
                stop_loss = latest_price * 0.95
                take_profit = latest_price * 1.10
            elif signal == "SELL":
                stop_loss = latest_price * 1.05
                take_profit = latest_price * 0.90
            else:
                stop_loss = None
                take_profit = None

        # Position Sizing: risk-based sizing using stop distance and risk budget
        risk_budget_pct = self.max_risk_per_trade_pct
        if signal != "HOLD" and stop_loss is not None:
            stop_distance_pct = abs((latest_price - stop_loss) / latest_price) * 100.0
            risk_per_unit = max(stop_distance_pct, 1e-6)
            raw_size = (risk_budget_pct / risk_per_unit) * 100.0
            suggested_size_pct = raw_size * confidence
            suggested_size_pct = min(max(suggested_size_pct, 0.0), self.max_position_size_pct)
        else:
            suggested_size_pct = 0
            risk_per_unit = 0.0

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "timesfm_change_pct": round(combined_forecast_change * 100, 2),
            "tech_score": round(norm_tech, 2),
            "latest_price": round(latest_price, 2),
            "predicted_next": round(predicted_price_next, 2),
            "expected_edge": round(expected_edge, 4),
            "uncertainty_width": round(interval_width, 4),
            "risk_per_unit": round(risk_per_unit, 4),
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "take_profit": round(take_profit, 2) if take_profit else None,
            "suggested_size_pct": round(suggested_size_pct, 1)
        }
