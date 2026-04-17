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
        latest_price = df['close'].iloc[-1]
        predicted_price = forecast_df['point_forecast'].iloc[0] # Next step prediction
        predicted_change = (predicted_price - latest_price) / latest_price

        # TimesFM Signal
        timesfm_signal = 0
        if predicted_change > 0.01: # 1% expected gain
            timesfm_signal = 1
        elif predicted_change < -0.01: # 1% expected loss
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

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "timesfm_change_pct": round(predicted_change * 100, 2),
            "tech_score": round(norm_tech, 2),
            "latest_price": round(float(latest_price), 2),
            "predicted_next": round(float(predicted_price), 2)
        }
