import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_engine import DataEngine
from forecaster import TimesFMForecast
from strategy import SignalStrategy
from backtester import Backtester
from portfolio import PortfolioManager

st.set_page_config(page_title="TimesFM Pro Trading Bot", layout="wide")

@st.cache_resource
def get_engines():
    return DataEngine(), TimesFMForecast(), SignalStrategy(), PortfolioManager()

data_engine, forecaster, strategy, portfolio = get_engines()
backtester = Backtester(data_engine, forecaster, strategy)

st.title("🤖 TimesFM Pro: Advanced Trading Bot")
st.markdown("""
Using **Google Research's TimesFM 2.5** foundation model with technical covariates (XReg).
Enhanced with multi-horizon analysis, risk management, and walk-forward backtesting.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker (e.g., AAPL, BTC-USD, EURUSD=X)", "BTC-USD")
period = st.sidebar.selectbox("History Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
interval = st.sidebar.selectbox("Interval", ["1h", "4h", "1d", "1wk"], index=2)
horizon = st.sidebar.slider("Forecast Horizon", 5, 50, 12)

indicator_list = data_engine.get_indicator_list()
selected_indicators = []
st.sidebar.subheader("Indicators for XReg")
for cat, inds in indicator_list.items():
    with st.sidebar.expander(cat):
        for ind in inds:
            if st.checkbox(ind, value=(ind in ["RSI_14", "SMA_20", "EMA_20", "ATR_14", "ATRr_14"])):
                selected_indicators.append(ind)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Live Forecast", "🧪 Backtesting", "💼 Portfolio"])

with tab1:
    if st.sidebar.button("Generate Signal & Forecast"):
        with st.spinner(f"Fetching data and generating forecast for {ticker}..."):
            df = data_engine.fetch_data(ticker, period=period, interval=interval)
            if df is None or df.empty:
                st.error(f"Failed to fetch data for {ticker}.")
            else:
                df = data_engine.add_indicators(df)
                point, quant = forecaster.forecast(df, horizon=horizon, covariate_cols=selected_indicators)
                forecast_df = forecaster.get_forecast_df(df, point, quant, horizon=horizon)
                sig_data = strategy.generate_signal(df, forecast_df, selected_indicators)

                # Store in session state for persistence across button clicks
                st.session_state.current_df = df
                st.session_state.current_forecast_df = forecast_df
                st.session_state.current_sig_data = sig_data
                st.session_state.current_ticker = ticker

    if "current_sig_data" in st.session_state:
        df = st.session_state.current_df
        forecast_df = st.session_state.current_forecast_df
        sig_data = st.session_state.current_sig_data
        ticker = st.session_state.current_ticker

        # Metrics Row 1
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Latest Price", f"{sig_data['latest_price']:,}")
        with col2:
            st.metric("Predicted Next", f"{sig_data['predicted_next']:,}", f"{sig_data['timesfm_change_pct']}%")
        with col3:
            color = "green" if sig_data['signal'] == "BUY" else "red" if sig_data['signal'] == "SELL" else "white"
            st.markdown(f"### Signal: <span style='color:{color}'>{sig_data['signal']}</span>", unsafe_allow_html=True)
        with col4:
            st.metric("Confidence", f"{int(sig_data['confidence']*100)}%")

        # Metrics Row 2 (Risk Management)
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.info(f"**Stop Loss:** {sig_data['stop_loss']}")
        with rcol2:
            st.success(f"**Take Profit:** {sig_data['take_profit']}")
        with rcol3:
            st.warning(f"**Suggested Size:** {sig_data['suggested_size_pct']}%")

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-100:], y=df['close'][-100:], name="Historical Price", line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['point_forecast'], name="TimesFM Forecast", line=dict(color='firebrick', dash='dash')))
        fig.add_trace(go.Scatter(
            x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
            y=forecast_df['upper_80'].tolist() + forecast_df['lower_80'].tolist()[::-1],
            fill='toself', fillcolor='rgba(178, 34, 34, 0.2)', line=dict(color='rgba(255,255,255,0)'),
            name="80% Interval"
        ))

        # Add SL/TP lines if signal
        if sig_data['signal'] != "HOLD":
            fig.add_hline(y=sig_data['stop_loss'], line_dash="dash", line_color="red", annotation_text="Stop Loss")
            fig.add_hline(y=sig_data['take_profit'], line_dash="dash", line_color="green", annotation_text="Take Profit")

        fig.update_layout(template="plotly_dark", height=600, title=f"{ticker} Forecast & Risk Levels")
        st.plotly_chart(fig, use_container_width=True)

        if st.button("Save Trade to Portfolio"):
            portfolio.save_trade({
                "ticker": ticker,
                "date": df.index[-1],
                **sig_data
            })
            st.success("Trade saved!")

with tab2:
    st.subheader("Walk-Forward Backtesting")
    st.markdown("This will simulate the bot's performance over the historical period selected in the sidebar.")

    backtest_days = st.slider("Backtest window (steps)", 10, 100, 30)

    if st.button("Run Walk-Forward Backtest"):
        progress_bar = st.progress(0)
        with st.spinner("Running simulation... this may take a minute as TimesFM re-forecasts at every step."):
            df = data_engine.fetch_data(ticker, period=period, interval=interval)
            df = data_engine.add_indicators(df)

            start_idx = len(df) - backtest_days
            results_df = backtester.run_walk_forward(ticker, df, start_idx, horizon, selected_indicators, progress_bar=progress_bar)

            progress_bar.empty()
            if not results_df.empty:
                metrics = backtester.calculate_metrics(results_df)

                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Win Rate", f"{metrics['win_rate']}%")
                mcol2.metric("Total PnL", f"{metrics['total_pnl_pct']}%")
                mcol3.metric("Avg PnL/Trade", f"{metrics['avg_pnl_pct']}%")
                mcol4.metric("Trades", metrics['total_trades'])

                st.subheader("Backtest Equity Curve (PnL %)")
                results_df['cumulative_pnl'] = results_df['pnl_pct'].cumsum()
                st.line_chart(results_df['cumulative_pnl'])

                with st.expander("Detailed Backtest Logs"):
                    st.dataframe(results_df)

with tab3:
    st.subheader("Saved Trade History")
    trades_df = portfolio.get_trades_df()
    if trades_df.empty:
        st.info("No trades saved yet.")
    else:
        st.dataframe(trades_df)
        if st.button("Clear Portfolio"):
            portfolio.clear_portfolio()
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Google TimesFM 2.5")
