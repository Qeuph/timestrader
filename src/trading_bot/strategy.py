import pandas as pd
import numpy as np
from typing import Dict, Any

class SignalStrategy:
    def __init__(self):
        pass

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

        timesfm_signal = 0
        if combined_forecast_change > 0.01: # 1% expected gain
            timesfm_signal = 1
        elif combined_forecast_change < -0.01: # 1% expected loss
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
        confidence = abs(combined_score)

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

        # Position Sizing: Confidence + Volatility (Inverse of ATR)
        # Higher confidence -> larger size
        # Higher volatility (ATR) -> smaller size
        if signal != "HOLD" and atr_col in df.columns:
            # Normalize ATR relative to price
            volatility_ratio = (df[atr_col].iloc[-1] / latest_price)
            # Simple heuristic: base size 10%, adjusted by confidence, penalized by volatility
            suggested_size_pct = (confidence * 20) / (volatility_ratio * 100)
            suggested_size_pct = min(max(suggested_size_pct, 1.0), 100.0) # Clamp between 1% and 100%
        elif signal != "HOLD":
            suggested_size_pct = confidence * 50
        else:
            suggested_size_pct = 0

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "timesfm_change_pct": round(combined_forecast_change * 100, 2),
            "tech_score": round(norm_tech, 2),
            "latest_price": round(latest_price, 2),
            "predicted_next": round(predicted_price_next, 2),
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "take_profit": round(take_profit, 2) if take_profit else None,
            "suggested_size_pct": round(suggested_size_pct, 1)
        }
