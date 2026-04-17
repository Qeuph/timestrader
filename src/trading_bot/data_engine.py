import yfinance as yf
import pandas as pd
import pandas_ta as ta
import logging

class DataEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def fetch_data(self, ticker: str, period: str = None, interval: str = "1d", start=None, end=None):
        """
        Fetch historical data from Yahoo Finance.
        """
        try:
            if start or end:
                df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            else:
                df = yf.download(ticker, period=period or "1y", interval=interval, progress=False)
            if df.empty:
                self.logger.warning(f"No data found for {ticker}")
                return None

            # Handle newer yfinance versions which might return MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure column names are lowercase
            df.columns = [str(col).lower() for col in df.columns]

            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Expected DatetimeIndex from market data source.")

            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            df.attrs["interval"] = interval
            df.attrs["ticker"] = ticker

            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def add_indicators(self, df: pd.DataFrame):
        """
        Add a comprehensive set of indicators using pandas-ta.
        """
        # Handle MultiIndex if present (yfinance v1.3.0+ behavior)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure column names are lowercase for pandas-ta
        df.columns = [str(col).lower() for col in df.columns]

        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if df.index.duplicated().any():
            self.logger.warning("Found duplicated timestamps; keeping latest entries.")
            df = df[~df.index.duplicated(keep="last")]
        if len(df) < 60:
            raise ValueError("Insufficient history for indicator computation (need at least 60 rows).")

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

        # Validation: ensure we have essential columns
        required = ['close', 'high', 'low', 'open', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            self.logger.error(f"Missing essential columns after indicator computation: {missing}")

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
