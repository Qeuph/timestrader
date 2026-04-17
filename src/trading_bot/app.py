import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_engine import DataEngine
from forecaster import TimesFMForecast
from strategy import SignalStrategy

st.set_page_config(page_title="TimesFM Trading Bot", layout="wide")

@st.cache_resource
def get_engines():
    return DataEngine(), TimesFMForecast(), SignalStrategy()

data_engine, forecaster, strategy = get_engines()

st.title("🤖 TimesFM Automatic Trading Bot")
st.markdown("""
This bot uses **Google Research's TimesFM 2.5** foundation model to forecast prices
using technical indicators as covariates.
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
            if st.checkbox(ind, value=(ind in ["RSI_14", "SMA_20", "EMA_20"])):
                selected_indicators.append(ind)

if st.sidebar.button("Generate Signal & Forecast"):
    with st.spinner(f"Fetching data and generating forecast for {ticker}..."):
        # 1. Fetch Data
        df = data_engine.fetch_data(ticker, period=period, interval=interval)

        if df is None or df.empty:
            st.error(f"Failed to fetch data for {ticker}. Please check the ticker symbol.")
        else:
            # 2. Add Indicators
            df = data_engine.add_indicators(df)

            # 3. Forecast
            point, quant = forecaster.forecast(df, horizon=horizon, covariate_cols=selected_indicators)
            forecast_df = forecaster.get_forecast_df(df, point, quant, horizon=horizon)

            # 4. Generate Signal
            sig_data = strategy.generate_signal(df, forecast_df, selected_indicators)

            # UI Layout
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Latest Price", f"{sig_data['latest_price']:,}")
                st.metric("Predicted Next", f"{sig_data['predicted_next']:,}", f"{sig_data['timesfm_change_pct']}%")

            with col2:
                color = "green" if sig_data['signal'] == "BUY" else "red" if sig_data['signal'] == "SELL" else "white"
                st.markdown(f"### Signal: <span style='color:{color}'>{sig_data['signal']}</span>", unsafe_allow_html=True)
                st.progress(sig_data['confidence'])
                st.caption(f"Confidence: {sig_data['confidence']}")

            with col3:
                st.markdown(f"**Tech Score:** {sig_data['tech_score']}")
                st.markdown(f"**XReg Indicators:** {len(selected_indicators)}")

            # Visualization
            st.subheader("Price Forecast & Indicators")

            fig = go.Figure()

            # Historical Price
            fig.add_trace(go.Scatter(
                x=df.index[-100:],
                y=df['close'][-100:],
                name="Historical Price",
                line=dict(color='royalblue')
            ))

            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df.index,
                y=forecast_df['point_forecast'],
                name="TimesFM Forecast",
                line=dict(color='firebrick', dash='dash')
            ))

            # Confidence Interval
            fig.add_trace(go.Scatter(
                x=forecast_df.index.tolist() + forecast_df.index.tolist()[::-1],
                y=forecast_df['upper_80'].tolist() + forecast_df['lower_80'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(178, 34, 34, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="80% Prediction Interval"
            ))

            # Indicators (Overlay)
            if "SMA_20" in df.columns:
                fig.add_trace(go.Scatter(x=df.index[-100:], y=df['SMA_20'][-100:], name="SMA 20", line=dict(width=1), opacity=0.5))
            if "SMA_50" in df.columns:
                fig.add_trace(go.Scatter(x=df.index[-100:], y=df['SMA_50'][-100:], name="SMA 50", line=dict(width=1), opacity=0.5))

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode="x unified",
                height=600
            )

            st.plotly_chart(fig, use_container_width=True)

            # Secondary Charts for Indicators
            st.subheader("Key Technical Indicators")
            ind_col1, ind_col2 = st.columns(2)

            with ind_col1:
                if "RSI_14" in df.columns:
                    rsi_fig = go.Figure()
                    rsi_fig.add_trace(go.Scatter(x=df.index[-100:], y=df['RSI_14'][-100:], name="RSI"))
                    rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
                    rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
                    rsi_fig.update_layout(title="RSI (14)", template="plotly_dark", height=300)
                    st.plotly_chart(rsi_fig, use_container_width=True)

            with ind_col2:
                if "MACDh_12_26_9" in df.columns:
                    macd_fig = go.Figure()
                    macd_fig.add_trace(go.Bar(x=df.index[-100:], y=df['MACDh_12_26_9'][-100:], name="MACD Histogram"))
                    macd_fig.update_layout(title="MACD Histogram", template="plotly_dark", height=300)
                    st.plotly_chart(macd_fig, use_container_width=True)

            # Data Table
            with st.expander("View Raw Data"):
                st.write("Latest Historical Data")
                st.dataframe(df.tail(10))
                st.write("Forecast Data")
                st.dataframe(forecast_df)

else:
    st.info("👈 Configure the bot in the sidebar and click **Generate Signal & Forecast** to begin.")

    # Show dummy data or explanation
    st.image("https://research.google/wp-content/uploads/2024/05/TimesFM_fig1.png", caption="TimesFM Architecture")
