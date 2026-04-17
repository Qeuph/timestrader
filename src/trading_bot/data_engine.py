import yfinance as yf
import pandas as pd
import pandas_ta as ta

class DataEngine:
    def __init__(self):
        pass

    def fetch_data(self, ticker: str, period: str = "1y", interval: str = "1d"):
        """
        Fetch historical data from Yahoo Finance.
        """
        df = yf.download(ticker, period=period, interval=interval)
        if df.empty:
            return None
        return df

    def add_indicators(self, df: pd.DataFrame):
        """
        Add a comprehensive set of indicators using pandas-ta.
        """
        # Handle MultiIndex if present (yfinance v1.3.0+ behavior)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure column names are lowercase for pandas-ta
        df.columns = [str(col).lower() for col in df.columns]

        # Trend Indicators
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.macd(append=True)
        df.ta.adx(append=True)

        # Momentum Indicators
        df.ta.rsi(length=14, append=True)
        df.ta.stoch(append=True)
        df.ta.cci(append=True)

        # Volatility Indicators
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.atr(append=True)

        # Volume Indicators
        df.ta.obv(append=True)
        df.ta.ad(append=True)

        # Fill NaNs using linear interpolation instead of dropping everything
        df = df.interpolate(method='linear').ffill().bfill()
        return df

    def get_indicator_list(self):
        """
        Returns a list of available indicator categories and their names.
        Note: pandas-ta might use ATRr_14 depending on data. We will check both.
        """
        return {
            "Trend": ["SMA_20", "SMA_50", "EMA_20", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9", "ADX_14"],
            "Momentum": ["RSI_14", "STOCHk_14_3_3", "STOCHd_14_3_3", "CCI_14_0.015"],
            "Volatility": ["BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "ATR_14", "ATRr_14"],
            "Volume": ["OBV", "AD"]
        }
