import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
from data_engine import DataEngine
from forecaster import TimesFMForecast
from strategy import SignalStrategy
from backtester import Backtester
from portfolio import PortfolioManager

st.set_page_config(page_title="TimesFM Pro Trading Bot", layout="wide")
logging.basicConfig(level=logging.INFO)

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
forecast_threshold_pct = st.sidebar.slider("Forecast Threshold %", 0.1, 5.0, 1.0, 0.1)
max_risk_per_trade_pct = st.sidebar.slider("Risk Budget %", 0.1, 5.0, 1.0, 0.1)
max_position_size_pct = st.sidebar.slider("Max Position Size %", 1.0, 50.0, 20.0, 1.0)
strategy.update_config(
    forecast_threshold_pct=forecast_threshold_pct,
    max_risk_per_trade_pct=max_risk_per_trade_pct,
    max_position_size_pct=max_position_size_pct,
)

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
                point, quant = forecaster.forecast(
                    df, horizon=horizon, covariate_cols=selected_indicators, strict_mode=False
                )
                forecast_df = forecaster.get_forecast_df(df, point, quant, horizon=horizon, interval=interval)
                sig_data = strategy.generate_signal(df, forecast_df, selected_indicators)
                if forecaster.last_warnings:
                    st.warning(" | ".join(forecaster.last_warnings))

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
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.caption(f"Expected Edge: {sig_data['expected_edge']}")
        with dcol2:
            st.caption(f"Uncertainty Width (80%): {sig_data['uncertainty_width']}")

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
                "model_repo_id": forecaster.repo_id,
                "period": period,
                "interval": interval,
                "horizon": horizon,
                "selected_indicators": selected_indicators,
                "strategy_config": {
                    "forecast_threshold_pct": forecast_threshold_pct,
                    "max_risk_per_trade_pct": max_risk_per_trade_pct,
                    "max_position_size_pct": max_position_size_pct,
                },
                **sig_data
            })
            st.success("Trade saved!")

with tab2:
    st.subheader("Walk-Forward Backtesting")
    st.markdown("This will simulate the bot's performance over the historical period selected in the sidebar.")

    backtest_days = st.slider("Backtest window (steps)", 10, 100, 30)
    cost_col1, cost_col2, cost_col3, cost_col4 = st.columns(4)
    with cost_col1:
        entry_delay_bars = st.number_input("Entry Delay Bars", min_value=1, max_value=10, value=1, step=1)
    with cost_col2:
        holding_period_bars = st.number_input("Holding Period Bars", min_value=1, max_value=50, value=horizon, step=1)
    with cost_col3:
        fee_bps = st.number_input("Fee (bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    with cost_col4:
        slippage_bps = st.number_input("Slippage (bps)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

    # Advanced backtest options
    with st.expander("Advanced Backtest Options"):
        col_adv1, col_adv2 = st.columns(2)
        with col_adv1:
            warmup_periods = st.number_input("Warmup Periods", min_value=0, max_value=50, value=10, step=1,
                                            help="Number of periods to skip at start for indicator stabilization")
            apply_position_sizing = st.checkbox("Apply Position Sizing", value=True,
                                               help="Use suggested position sizes instead of 100% allocation")
        with col_adv2:
            run_monte_carlo = st.checkbox("Run Monte Carlo Simulation", value=False,
                                         help="Run bootstrap simulation to test strategy robustness")
            show_benchmark = st.checkbox("Show Buy & Hold Benchmark", value=True,
                                        help="Compare strategy returns to buy-and-hold")

    if st.button("Run Walk-Forward Backtest"):
        progress_bar = st.progress(0)
        with st.spinner("Running simulation... this may take a minute as TimesFM re-forecasts at every step."):
            df = data_engine.fetch_data(ticker, period=period, interval=interval)
            df = data_engine.add_indicators(df)

            start_idx = len(df) - backtest_days
            results_df = backtester.run_walk_forward(
                ticker=ticker,
                df=df,
                start_idx=start_idx,
                horizon=horizon,
                covariate_cols=selected_indicators,
                entry_delay_bars=int(entry_delay_bars),
                holding_period_bars=int(holding_period_bars),
                fee_bps=float(fee_bps),
                slippage_bps=float(slippage_bps),
                progress_bar=progress_bar,
                warmup_periods=int(warmup_periods),
                apply_position_sizing=apply_position_sizing
            )

            progress_bar.empty()
            if not results_df.empty:
                metrics = backtester.calculate_metrics(results_df)
                
                # Calculate benchmark comparison
                benchmark_metrics = None
                if show_benchmark:
                    benchmark_metrics = backtester.calculate_benchmark_returns(df, results_df)

                # Run Monte Carlo if requested
                mc_results = None
                if run_monte_carlo:
                    mc_results = backtester.run_monte_carlo(results_df, iterations=500)

                # Display metrics
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                mcol1.metric("Win Rate", f"{metrics['win_rate']}%")
                mcol2.metric("Total PnL", f"{metrics['total_pnl_pct']}%")
                mcol3.metric("Avg PnL/Trade", f"{metrics['avg_pnl_pct']}%")
                mcol4.metric("Trades", metrics['total_trades'])
                
                mx1, mx2, mx3 = st.columns(3)
                mx1.metric("Sharpe", metrics["sharpe_ratio"])
                mx2.metric("Profit Factor", metrics["profit_factor"])
                mx3.metric("Max Drawdown", f"{metrics['max_drawdown_pct']}%")
                
                # Show additional metrics
                with st.expander("Additional Performance Metrics"):
                    am1, am2, am3, am4 = st.columns(4)
                    am1.metric("Avg Win", f"{metrics.get('avg_win_pct', 0)}%")
                    am2.metric("Avg Loss", f"{metrics.get('avg_loss_pct', 0)}%")
                    am3.metric("Largest Win", f"{metrics.get('largest_win_pct', 0)}%")
                    am4.metric("Largest Loss", f"{metrics.get('largest_loss_pct', 0)}%")
                    exp1, exp2 = st.columns(2)
                    exp1.metric("Expectancy", f"{metrics.get('expectancy', 0)}%")
                    exp2.metric("Position Size Avg", f"{results_df['position_size_pct'].mean():.1f}%" if 'position_size_pct' in results_df.columns else "N/A")

                # Benchmark comparison
                if benchmark_metrics:
                    st.subheader("Benchmark Comparison")
                    bm1, bm2, bm3 = st.columns(3)
                    bm1.metric("Strategy Total Return", f"{metrics['total_pnl_pct']}%")
                    bm2.metric("Buy & Hold Return", f"{benchmark_metrics['total_return_pct']}%")
                    excess_return = metrics['total_pnl_pct'] - benchmark_metrics['total_return_pct']
                    bm3.metric("Excess Return (Alpha)", f"{excess_return}%")
                    
                    # Visual comparison
                    comp_df = pd.DataFrame({
                        'Strategy': results_df['pnl_pct'].cumsum(),
                        'Buy & Hold': df.loc[results_df.index[0]:results_df.index[-1], 'close'].pct_change().fillna(0).cumsum() * 100
                    })
                    st.line_chart(comp_df)

                # Monte Carlo results
                if mc_results:
                    st.subheader("Monte Carlo Simulation (500 iterations)")
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Mean Final PnL", f"{mc_results['mean_final_pnl']}%")
                    mc2.metric("Std Dev", f"{mc_results['std_final_pnl']}%")
                    mc3.metric("Positive Probability", f"{mc_results['probability_positive']}%")
                    mc4.metric("Negative Probability", f"{mc_results['probability_negative']}%")
                    
                    # Percentile range
                    mc_range1, mc_range2 = st.columns(2)
                    mc_range1.metric("5th Percentile (Worst Case)", f"{mc_results['percentile_5']}%")
                    mc_range2.metric("95th Percentile (Best Case)", f"{mc_results['percentile_95']}%")

                st.subheader("Backtest Equity Curve (PnL %)")
                results_df['cumulative_pnl'] = results_df['pnl_pct'].cumsum()
                st.line_chart(results_df['cumulative_pnl'])

                with st.expander("Detailed Backtest Logs"):
                    st.dataframe(results_df)
                if backtester.last_errors:
                    st.warning(f"Backtest completed with {len(backtester.last_errors)} step errors.")

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
