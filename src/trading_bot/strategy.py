import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class SignalResult:
    signal: str
    confidence: float
    timesfm_change_pct: float
    tech_score: float
    latest_price: float
    predicted_next: float
    expected_edge: float
    uncertainty_width: float
    risk_per_unit: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    suggested_size_pct: float
    trailing_stop: Optional[float] = None
    market_regime: str = "unknown"
    mtf_confirmation: bool = False
    kelly_size_pct: Optional[float] = None


class SignalStrategy:
    def __init__(self,
                 forecast_threshold_pct: float = 1.0,
                 max_risk_per_trade_pct: float = 1.0,
                 max_position_size_pct: float = 20.0,
                 volatility_adaptive: bool = True,
                 volatility_lookback: int = 20,
                 threshold_multiplier: float = 1.5,
                 timesfm_weight: float = 0.6,
                 technical_weight: float = 0.4,
                 next_step_weight: float = 0.4,
                 horizon_weight: float = 0.6,
                 enable_mtf_confirmation: bool = False,
                 enable_regime_detection: bool = True,
                 use_kelly_criterion: bool = True,
                 kelly_fraction: float = 0.25,
                 enable_trailing_stop: bool = True,
                 trailing_stop_atr_multiple: float = 1.5,
                 max_drawdown_cutoff_pct: float = 15.0):
        
        # Base parameters
        self.base_threshold_pct = forecast_threshold_pct
        self.max_risk_per_trade_pct = max_risk_per_trade_pct
        self.max_position_size_pct = max_position_size_pct
        
        # Dynamic threshold parameters
        self.volatility_adaptive = volatility_adaptive
        self.volatility_lookback = volatility_lookback
        self.threshold_multiplier = threshold_multiplier
        
        # Signal synthesis weights
        self.timesfm_weight = timesfm_weight
        self.technical_weight = technical_weight
        self.next_step_weight = next_step_weight
        self.horizon_weight = horizon_weight
        
        # Multi-timeframe confirmation
        self.enable_mtf_confirmation = enable_mtf_confirmation
        
        # Regime detection
        self.enable_regime_detection = enable_regime_detection
        
        # Position sizing
        self.use_kelly_criterion = use_kelly_criterion
        self.kelly_fraction = kelly_fraction
        
        # Risk management
        self.enable_trailing_stop = enable_trailing_stop
        self.trailing_stop_atr_multiple = trailing_stop_atr_multiple
        self.max_drawdown_cutoff_pct = max_drawdown_cutoff_pct
        
        # Performance tracking for adaptive sizing
        self.recent_performance = []
        self.max_recent_trades = 20

    def update_config(self,
                      forecast_threshold_pct: float = None,
                      max_risk_per_trade_pct: float = None,
                      max_position_size_pct: float = None,
                      volatility_adaptive: bool = None,
                      timesfm_weight: float = None,
                      technical_weight: float = None,
                      use_kelly_criterion: bool = None,
                      enable_trailing_stop: bool = None):
        """Update configuration parameters dynamically."""
        if forecast_threshold_pct is not None:
            self.base_threshold_pct = forecast_threshold_pct
        if max_risk_per_trade_pct is not None:
            self.max_risk_per_trade_pct = max_risk_per_trade_pct
        if max_position_size_pct is not None:
            self.max_position_size_pct = max_position_size_pct
        if volatility_adaptive is not None:
            self.volatility_adaptive = volatility_adaptive
        if timesfm_weight is not None:
            self.timesfm_weight = timesfm_weight
        if technical_weight is not None:
            self.technical_weight = technical_weight
        if use_kelly_criterion is not None:
            self.use_kelly_criterion = use_kelly_criterion
        if enable_trailing_stop is not None:
            self.enable_trailing_stop = enable_trailing_stop

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate recent volatility using returns standard deviation."""
        if len(df) < self.volatility_lookback:
            return 0.0
        
        returns = df['close'].pct_change().dropna()
        if len(returns) < self.volatility_lookback:
            return 0.0
        
        recent_returns = returns.tail(self.volatility_lookback)
        return recent_returns.std() * np.sqrt(252)  # Annualized volatility

    def _get_dynamic_threshold(self, df: pd.DataFrame) -> float:
        """Calculate dynamic threshold based on market volatility."""
        if not self.volatility_adaptive:
            return self.base_threshold_pct
        
        current_volatility = self._calculate_volatility(df)
        
        # Scale threshold with volatility
        # Higher volatility = higher threshold to avoid false signals
        volatility_adjustment = current_volatility * self.threshold_multiplier
        dynamic_threshold = self.base_threshold_pct + (volatility_adjustment * 100)
        
        return max(dynamic_threshold, self.base_threshold_pct * 0.5)  # Floor at 50% of base

    def _detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime using trend and volatility analysis."""
        if not self.enable_regime_detection or len(df) < 50:
            return MarketRegime.RANGING
        
        close = df['close']
        
        # Trend detection using moving averages
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma_20 = df['SMA_20'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            sma_20_prev = df['SMA_20'].iloc[-5] if len(df) > 5 else sma_20
            sma_50_prev = df['SMA_50'].iloc[-5] if len(df) > 5 else sma_50
            
            # Strong uptrend: SMA20 > SMA50 and both rising
            if sma_20 > sma_50 and sma_20 > sma_20_prev and sma_50 > sma_50_prev:
                return MarketRegime.TRENDING_UP
            # Strong downtrend: SMA20 < SMA50 and both falling
            elif sma_20 < sma_50 and sma_20 < sma_20_prev and sma_50 < sma_50_prev:
                return MarketRegime.TRENDING_DOWN
        
        # Volatility regime
        current_volatility = self._calculate_volatility(df)
        historical_volatility = df['close'].pct_change().dropna().std() * np.sqrt(252)
        
        if current_volatility > historical_volatility * 1.5:
            return MarketRegime.HIGH_VOLATILITY
        elif current_volatility < historical_volatility * 0.7:
            return MarketRegime.LOW_VOLATILITY
        
        return MarketRegime.RANGING

    def _calculate_kelly_position(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly criterion position size."""
        if avg_loss == 0 or win_rate <= 0:
            return 0.0
        
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
        
        # Kelly formula: f* = W - [(1-W)/R]
        # Where W = win probability, R = win/loss ratio
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fractional Kelly for safety (reduce by kelly_fraction)
        fractional_kelly = kelly * self.kelly_fraction
        
        # Clamp between 0 and max position size
        return max(0.0, min(fractional_kelly * 100, self.max_position_size_pct))

    def _multi_timeframe_confirmation(self, ticker: str, data_engine, 
                                       selected_indicators: list, 
                                       horizon: int = 12) -> Tuple[bool, int]:
        """Check for signal confirmation across multiple timeframes."""
        if not self.enable_mtf_confirmation:
            return True, 0
        
        timeframes = ["1h", "4h", "1d"]
        confirmations = 0
        total_signals = 0
        
        for tf in timeframes:
            try:
                # Fetch data for this timeframe
                tf_df = data_engine.fetch_data(ticker, period="3mo", interval=tf)
                if tf_df is None or tf_df.empty:
                    continue
                
                tf_df = data_engine.add_indicators(tf_df)
                
                # Simple trend check for this timeframe
                if 'SMA_20' in tf_df.columns and 'SMA_50' in tf_df.columns:
                    total_signals += 1
                    if tf_df['SMA_20'].iloc[-1] > tf_df['SMA_50'].iloc[-1]:
                        confirmations += 1
            except Exception:
                continue
        
        return confirmations >= 2, confirmations  # Require at least 2 timeframes agreeing

    def _calculate_trailing_stop(self, df: pd.DataFrame, entry_price: float, 
                                  is_long: bool) -> Optional[float]:
        """Calculate ATR-based trailing stop level."""
        if not self.enable_trailing_stop:
            return None
        
        atr_col = 'ATR_14' if 'ATR_14' in df.columns else 'ATRr_14'
        if atr_col not in df.columns:
            return None
        
        atr = float(df[atr_col].iloc[-1])
        trailing_distance = atr * self.trailing_stop_atr_multiple
        
        if is_long:
            # For long positions, trailing stop is below current price
            trailing_stop = entry_price - trailing_distance
        else:
            # For short positions, trailing stop is above current price
            trailing_stop = entry_price + trailing_distance
        
        return round(trailing_stop, 2)

    def generate_signal(self,
                        df: pd.DataFrame,
                        forecast_df: pd.DataFrame,
                        selected_indicators: list,
                        data_engine=None,
                        ticker: str = None,
                        horizon: int = 12,
                        historical_trades: List[Dict] = None) -> SignalResult:
        """
        Generate a Buy/Sell/Hold signal with enhanced features:
        - Dynamic volatility-adaptive thresholds
        - Market regime detection
        - Multi-timeframe confirmation
        - Kelly criterion position sizing
        - Trailing stop calculation
        """
        latest_price = float(df['close'].iloc[-1])

        # Multi-horizon Analysis
        predicted_price_next = float(forecast_df['point_forecast'].iloc[0])
        avg_predicted_price = float(forecast_df['point_forecast'].mean())

        # Calculate changes for both next-step and full-horizon
        predicted_change_next = (predicted_price_next - latest_price) / latest_price
        predicted_change_horizon = (avg_predicted_price - latest_price) / latest_price

        # Combined TimesFM Signal with configurable weights
        combined_forecast_change = (
            self.next_step_weight * predicted_change_next + 
            self.horizon_weight * predicted_change_horizon
        )

        # Get dynamic threshold based on volatility
        threshold = self._get_dynamic_threshold(df) / 100.0
        
        timesfm_signal = 0
        if combined_forecast_change > threshold:
            timesfm_signal = 1
        elif combined_forecast_change < -threshold:
            timesfm_signal = -1

        # Technical Indicators Signal with performance-based weighting
        tech_signal = 0
        active_indicators = 0
        indicator_weights = {}

        # Trend filters
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            active_indicators += 1
            if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
                tech_signal += 1
            else:
                tech_signal -= 1
            indicator_weights['trend'] = 1.0

        # Momentum filters with regime adjustment
        if 'RSI_14' in df.columns:
            active_indicators += 1
            rsi = df['RSI_14'].iloc[-1]
            regime = self._detect_market_regime(df)
            
            # Adjust RSI interpretation based on regime
            if regime == MarketRegime.TRENDING_UP:
                # In uptrends, RSI can stay overbought longer
                if rsi < 40:  # Less strict oversold
                    tech_signal += 1
                elif rsi > 80:  # More strict overbought
                    tech_signal -= 1
            elif regime == MarketRegime.TRENDING_DOWN:
                # In downtrends, RSI can stay oversold longer
                if rsi < 20:  # More strict oversold
                    tech_signal += 1
                elif rsi > 60:  # Less strict overbought
                    tech_signal -= 1
            else:
                # Normal ranging market
                if rsi < 30:
                    tech_signal += 1
                elif rsi > 70:
                    tech_signal -= 1
            
            # RSI momentum
            if len(df) > 1:
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

        # Normalize tech signal
        norm_tech = tech_signal / active_indicators if active_indicators > 0 else 0

        # Combined score with configurable weights
        combined_score = (self.timesfm_weight * timesfm_signal) + \
                        (self.technical_weight * norm_tech)

        # Multi-timeframe confirmation bonus
        mtf_confirmed = False
        mtf_count = 0
        if self.enable_mtf_confirmation and data_engine and ticker:
            mtf_confirmed, mtf_count = self._multi_timeframe_confirmation(
                ticker, data_engine, selected_indicators, horizon
            )
            if mtf_confirmed:
                combined_score *= 1.1  # 10% boost for MTF confirmation

        # Determine signal with dynamic threshold
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

        # Adjust signal threshold based on regime
        regime = self._detect_market_regime(df)
        signal_threshold = 0.3
        
        if regime == MarketRegime.HIGH_VOLATILITY:
            signal_threshold = 0.4  # More conservative in high vol
        elif regime == MarketRegime.LOW_VOLATILITY:
            signal_threshold = 0.25  # Less conservative in low vol
        
        if combined_score > signal_threshold:
            signal = "BUY"
        elif combined_score < -signal_threshold:
            signal = "SELL"

        # Risk Management: SL/TP based on ATR
        atr_col = 'ATR_14' if 'ATR_14' in df.columns else 'ATRr_14'
        stop_loss = None
        take_profit = None
        trailing_stop = None
        
        if atr_col in df.columns:
            atr = float(df[atr_col].iloc[-1])
            if signal == "BUY":
                stop_loss = latest_price - (2 * atr)
                take_profit = latest_price + (4 * atr)
                trailing_stop = self._calculate_trailing_stop(df, latest_price, is_long=True)
            elif signal == "SELL":
                stop_loss = latest_price + (2 * atr)
                take_profit = latest_price - (4 * atr)
                trailing_stop = self._calculate_trailing_stop(df, latest_price, is_long=False)
        else:
            # Fallback if ATR is missing
            if signal == "BUY":
                stop_loss = latest_price * 0.95
                take_profit = latest_price * 1.10
            elif signal == "SELL":
                stop_loss = latest_price * 1.05
                take_profit = latest_price * 0.90

        # Position Sizing with Kelly Criterion
        risk_budget_pct = self.max_risk_per_trade_pct
        suggested_size_pct = 0.0
        kelly_size_pct = None
        risk_per_unit = 0.0
        
        if signal != "HOLD" and stop_loss is not None:
            stop_distance_pct = abs((latest_price - stop_loss) / latest_price) * 100.0
            risk_per_unit = max(stop_distance_pct, 1e-6)
            
            # Calculate Kelly size if historical trades available
            if self.use_kelly_criterion and historical_trades and len(historical_trades) > 5:
                wins = [t for t in historical_trades if t.get('net_pnl_pct', 0) > 0]
                losses = [t for t in historical_trades if t.get('net_pnl_pct', 0) <= 0]
                
                if wins and losses:
                    win_rate = len(wins) / len(historical_trades)
                    avg_win = np.mean([t['net_pnl_pct'] for t in wins])
                    avg_loss = abs(np.mean([t['net_pnl_pct'] for t in losses]))
                    
                    kelly_size_pct = self._calculate_kelly_position(win_rate, avg_win, avg_loss)
            
            # Use Kelly size if available, otherwise use risk-based sizing
            if kelly_size_pct is not None and kelly_size_pct > 0:
                raw_size = kelly_size_pct
            else:
                raw_size = (risk_budget_pct / risk_per_unit) * 100.0
            
            # Apply confidence scaling
            suggested_size_pct = raw_size * confidence
            suggested_size_pct = min(max(suggested_size_pct, 0.0), self.max_position_size_pct)

        return SignalResult(
            signal=signal,
            confidence=round(confidence, 2),
            timesfm_change_pct=round(combined_forecast_change * 100, 2),
            tech_score=round(norm_tech, 2),
            latest_price=round(latest_price, 2),
            predicted_next=round(predicted_price_next, 2),
            expected_edge=round(expected_edge, 4),
            uncertainty_width=round(interval_width, 4),
            risk_per_unit=round(risk_per_unit, 4),
            stop_loss=round(stop_loss, 2) if stop_loss else None,
            take_profit=round(take_profit, 2) if take_profit else None,
            suggested_size_pct=round(suggested_size_pct, 1),
            trailing_stop=trailing_stop,
            market_regime=regime.value,
            mtf_confirmation=mtf_confirmed,
            kelly_size_pct=round(kelly_size_pct, 1) if kelly_size_pct else None
        )
