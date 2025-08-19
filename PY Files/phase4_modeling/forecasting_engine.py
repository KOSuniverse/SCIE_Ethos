# PY Files/phase4_modeling/forecasting_engine.py
# Phase 4A: Forecasting sub-skills (par_policy, safety_stock, demand_projection)

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

class ForecastingEngine:
    """
    Phase 4A: Comprehensive forecasting engine with backtests and policy math.
    Implements par_policy, safety_stock, and demand_projection sub-skills.
    """
    
    def __init__(self, service_level: float = 0.95):
        self.service_level = service_level
        self.z_score_map = {
            0.90: 1.645,
            0.95: 1.960,
            0.975: 2.241,
            0.99: 2.576
        }
        
    def get_service_level_zscore(self, service_level: float = None) -> float:
        """Get z-score for given service level."""
        sl = service_level or self.service_level
        return self.z_score_map.get(sl, 1.960)  # Default to 95%
    
    def demand_projection(
        self, 
        historical_data: pd.DataFrame,
        forecast_periods: int = 12,
        method: str = "exponential_smoothing"
    ) -> Dict[str, Any]:
        """
        Phase 4A: Demand projection with multiple forecasting methods.
        
        Args:
            historical_data: DataFrame with columns ['period', 'demand', 'part_number']
            forecast_periods: Number of periods to forecast
            method: 'exponential_smoothing', 'moving_average', 'linear_trend'
        
        Returns:
            Dict with forecast, confidence intervals, and model metadata
        """
        
        if historical_data is None or len(historical_data) == 0:
            return {
                "error": "No historical data provided",
                "forecast": [],
                "method": method,
                "periods": forecast_periods
            }
        
        # Ensure demand column exists and is numeric
        demand_col = None
        for col in ['demand', 'quantity', 'usage', 'consumption']:
            if col in historical_data.columns:
                demand_col = col
                break
        
        if not demand_col:
            return {
                "error": "No demand/quantity column found in historical data",
                "available_columns": list(historical_data.columns)
            }
        
        # Clean and prepare data
        df = historical_data.copy()
        df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce').fillna(0)
        
        # Group by part if part_number column exists
        if 'part_number' in df.columns:
            forecasts = {}
            for part in df['part_number'].unique():
                part_data = df[df['part_number'] == part][demand_col].values
                forecasts[part] = self._forecast_single_series(
                    part_data, forecast_periods, method
                )
            
            return {
                "forecasts_by_part": forecasts,
                "method": method,
                "service_level": self.service_level,
                "total_parts": len(forecasts)
            }
        else:
            # Single series forecast
            demand_series = df[demand_col].values
            forecast_result = self._forecast_single_series(
                demand_series, forecast_periods, method
            )
            
            return {
                "forecast": forecast_result["forecast"],
                "confidence_intervals": forecast_result["confidence_intervals"],
                "method": method,
                "service_level": self.service_level,
                "model_metadata": forecast_result.get("metadata", {})
            }
    
    def _forecast_single_series(
        self, 
        demand_series: np.ndarray, 
        periods: int, 
        method: str
    ) -> Dict[str, Any]:
        """Internal method to forecast a single demand series."""
        
        if len(demand_series) < 2:
            return {
                "forecast": [0] * periods,
                "confidence_intervals": [(0, 0)] * periods,
                "error": "Insufficient historical data"
            }
        
        if method == "exponential_smoothing":
            return self._exponential_smoothing_forecast(demand_series, periods)
        elif method == "moving_average":
            return self._moving_average_forecast(demand_series, periods)
        elif method == "linear_trend":
            return self._linear_trend_forecast(demand_series, periods)
        else:
            # Default to exponential smoothing
            return self._exponential_smoothing_forecast(demand_series, periods)
    
    def _exponential_smoothing_forecast(
        self, 
        demand_series: np.ndarray, 
        periods: int,
        alpha: float = 0.3
    ) -> Dict[str, Any]:
        """Exponential smoothing forecast with confidence intervals."""
        
        # Simple exponential smoothing
        smoothed = np.zeros(len(demand_series))
        smoothed[0] = demand_series[0]
        
        for i in range(1, len(demand_series)):
            smoothed[i] = alpha * demand_series[i] + (1 - alpha) * smoothed[i-1]
        
        # Forecast future periods
        last_smoothed = smoothed[-1]
        forecast = [last_smoothed] * periods
        
        # Calculate forecast error (MAD - Mean Absolute Deviation)
        errors = np.abs(demand_series[1:] - smoothed[:-1])
        mad = np.mean(errors) if len(errors) > 0 else np.std(demand_series)
        
        # Confidence intervals using MAD and z-score
        z_score = self.get_service_level_zscore()
        confidence_intervals = []
        
        for i in range(periods):
            # Error grows with forecast horizon
            forecast_error = mad * math.sqrt(i + 1)
            lower_bound = max(0, last_smoothed - z_score * forecast_error)
            upper_bound = last_smoothed + z_score * forecast_error
            confidence_intervals.append((lower_bound, upper_bound))
        
        return {
            "forecast": forecast,
            "confidence_intervals": confidence_intervals,
            "metadata": {
                "method": "exponential_smoothing",
                "alpha": alpha,
                "mad": mad,
                "last_actual": float(demand_series[-1]),
                "last_smoothed": float(last_smoothed)
            }
        }
    
    def _moving_average_forecast(
        self, 
        demand_series: np.ndarray, 
        periods: int,
        window: int = 3
    ) -> Dict[str, Any]:
        """Moving average forecast."""
        
        window = min(window, len(demand_series))
        last_values = demand_series[-window:]
        avg_demand = np.mean(last_values)
        
        forecast = [avg_demand] * periods
        
        # Simple confidence interval based on historical variance
        historical_std = np.std(demand_series)
        z_score = self.get_service_level_zscore()
        
        confidence_intervals = []
        for i in range(periods):
            lower_bound = max(0, avg_demand - z_score * historical_std)
            upper_bound = avg_demand + z_score * historical_std
            confidence_intervals.append((lower_bound, upper_bound))
        
        return {
            "forecast": forecast,
            "confidence_intervals": confidence_intervals,
            "metadata": {
                "method": "moving_average",
                "window": window,
                "avg_demand": float(avg_demand),
                "historical_std": float(historical_std)
            }
        }
    
    def _linear_trend_forecast(
        self, 
        demand_series: np.ndarray, 
        periods: int
    ) -> Dict[str, Any]:
        """Linear trend forecast using least squares."""
        
        x = np.arange(len(demand_series))
        
        # Calculate linear trend
        slope, intercept = np.polyfit(x, demand_series, 1)
        
        # Forecast future periods
        future_x = np.arange(len(demand_series), len(demand_series) + periods)
        forecast = slope * future_x + intercept
        forecast = np.maximum(forecast, 0)  # Ensure non-negative
        
        # Calculate prediction intervals
        residuals = demand_series - (slope * x + intercept)
        mse = np.mean(residuals ** 2)
        
        z_score = self.get_service_level_zscore()
        confidence_intervals = []
        
        for i, pred in enumerate(forecast):
            # Standard error grows with distance from data
            se = math.sqrt(mse * (1 + 1/len(demand_series) + (future_x[i] - np.mean(x))**2 / np.sum((x - np.mean(x))**2)))
            lower_bound = max(0, pred - z_score * se)
            upper_bound = pred + z_score * se
            confidence_intervals.append((lower_bound, upper_bound))
        
        return {
            "forecast": forecast.tolist(),
            "confidence_intervals": confidence_intervals,
            "metadata": {
                "method": "linear_trend",
                "slope": float(slope),
                "intercept": float(intercept),
                "mse": float(mse),
                "r_squared": float(np.corrcoef(x, demand_series)[0, 1] ** 2) if len(x) > 1 else 0
            }
        }
    
    def safety_stock_calculation(
        self,
        avg_demand: float,
        demand_std: float,
        lead_time_days: float,
        lead_time_std: float = 0,
        service_level: float = None
    ) -> Dict[str, Any]:
        """
        Phase 4A: Safety stock calculation with multiple methods.
        
        Formula: SS = Z * sqrt(LT * Var(D) + D² * Var(LT))
        Where:
        - Z = service level z-score
        - LT = lead time
        - Var(D) = demand variance
        - D = average demand
        - Var(LT) = lead time variance
        """
        
        sl = service_level or self.service_level
        z_score = self.get_service_level_zscore(sl)
        
        # Convert to variance
        demand_var = demand_std ** 2
        lead_time_var = lead_time_std ** 2
        
        # Safety stock formula
        safety_stock = z_score * math.sqrt(
            lead_time_days * demand_var + 
            (avg_demand ** 2) * lead_time_var
        )
        
        # Alternative simplified method (demand variability only)
        simple_safety_stock = z_score * demand_std * math.sqrt(lead_time_days)
        
        return {
            "safety_stock": round(safety_stock, 2),
            "simple_safety_stock": round(simple_safety_stock, 2),
            "parameters": {
                "service_level": sl,
                "z_score": z_score,
                "avg_demand": avg_demand,
                "demand_std": demand_std,
                "lead_time_days": lead_time_days,
                "lead_time_std": lead_time_std
            },
            "formula": "Z * sqrt(LT * Var(D) + D² * Var(LT))"
        }
    
    def reorder_point_calculation(
        self,
        avg_demand: float,
        lead_time_days: float,
        safety_stock: float
    ) -> Dict[str, Any]:
        """
        Phase 4A: Reorder Point (ROP) calculation.
        
        Formula: ROP = (Average Demand × Lead Time) + Safety Stock
        """
        
        lead_time_demand = avg_demand * lead_time_days
        reorder_point = lead_time_demand + safety_stock
        
        return {
            "reorder_point": round(reorder_point, 2),
            "lead_time_demand": round(lead_time_demand, 2),
            "safety_stock": safety_stock,
            "parameters": {
                "avg_demand": avg_demand,
                "lead_time_days": lead_time_days
            },
            "formula": "(Avg Demand × Lead Time) + Safety Stock"
        }
    
    def par_level_policy(
        self,
        avg_demand: float,
        demand_std: float,
        lead_time_days: float,
        review_period_days: float = 30,
        service_level: float = None
    ) -> Dict[str, Any]:
        """
        Phase 4A: Par level calculation for periodic review systems.
        
        Formula: Par Level = (Avg Demand × (Lead Time + Review Period)) + Safety Stock
        """
        
        sl = service_level or self.service_level
        
        # Calculate safety stock for the combined period
        combined_period = lead_time_days + review_period_days
        safety_stock_result = self.safety_stock_calculation(
            avg_demand=avg_demand,
            demand_std=demand_std,
            lead_time_days=combined_period,
            service_level=sl
        )
        
        expected_demand = avg_demand * combined_period
        par_level = expected_demand + safety_stock_result["safety_stock"]
        
        return {
            "par_level": round(par_level, 2),
            "expected_demand": round(expected_demand, 2),
            "safety_stock": safety_stock_result["safety_stock"],
            "parameters": {
                "avg_demand": avg_demand,
                "demand_std": demand_std,
                "lead_time_days": lead_time_days,
                "review_period_days": review_period_days,
                "service_level": sl,
                "combined_period_days": combined_period
            },
            "formula": "(Avg Demand × (Lead Time + Review Period)) + Safety Stock"
        }
    
    def backtest_forecast(
        self,
        historical_data: pd.DataFrame,
        test_periods: int = 6,
        method: str = "exponential_smoothing"
    ) -> Dict[str, Any]:
        """
        Phase 4A: Backtest forecasting model performance.
        
        Uses walk-forward validation to test forecast accuracy.
        """
        
        if len(historical_data) < test_periods + 3:
            return {
                "error": "Insufficient data for backtesting",
                "required_periods": test_periods + 3,
                "available_periods": len(historical_data)
            }
        
        # Identify demand column
        demand_col = None
        for col in ['demand', 'quantity', 'usage', 'consumption']:
            if col in historical_data.columns:
                demand_col = col
                break
        
        if not demand_col:
            return {"error": "No demand column found"}
        
        demand_series = pd.to_numeric(historical_data[demand_col], errors='coerce').fillna(0).values
        
        # Walk-forward validation
        actuals = []
        forecasts = []
        errors = []
        
        for i in range(len(demand_series) - test_periods, len(demand_series)):
            # Use data up to period i for training
            train_data = demand_series[:i]
            actual = demand_series[i]
            
            # Generate 1-period forecast
            forecast_result = self._forecast_single_series(train_data, 1, method)
            forecast = forecast_result["forecast"][0]
            
            actuals.append(actual)
            forecasts.append(forecast)
            errors.append(abs(actual - forecast))
        
        # Calculate accuracy metrics
        mae = np.mean(errors)  # Mean Absolute Error
        mape = np.mean([abs(a - f) / max(a, 1) for a, f in zip(actuals, forecasts)]) * 100  # MAPE
        rmse = math.sqrt(np.mean([(a - f) ** 2 for a, f in zip(actuals, forecasts)]))  # RMSE
        
        return {
            "backtest_results": {
                "actuals": actuals,
                "forecasts": forecasts,
                "errors": errors
            },
            "accuracy_metrics": {
                "mae": round(mae, 2),
                "mape": round(mape, 2),
                "rmse": round(rmse, 2)
            },
            "method": method,
            "test_periods": test_periods,
            "avg_error": round(mae, 2)
        }
    
    def comprehensive_policy_analysis(
        self,
        part_data: Dict[str, Any],
        forecast_periods: int = 12
    ) -> Dict[str, Any]:
        """
        Phase 4A: Comprehensive analysis combining all forecasting sub-skills.
        
        Args:
            part_data: Dict with keys like 'historical_demand', 'lead_time', 'current_stock'
            forecast_periods: Number of periods to forecast
        
        Returns:
            Complete analysis with demand forecast, safety stock, ROP, and par levels
        """
        
        # Extract parameters
        historical_demand = part_data.get('historical_demand', [])
        lead_time = part_data.get('lead_time_days', 7)
        current_stock = part_data.get('current_stock', 0)
        part_number = part_data.get('part_number', 'Unknown')
        
        if not historical_demand or len(historical_demand) == 0:
            return {"error": "No historical demand data provided"}
        
        # Convert to DataFrame if needed
        if isinstance(historical_demand, list):
            df = pd.DataFrame({'demand': historical_demand})
        else:
            df = historical_demand
        
        # 1. Demand Projection
        forecast_result = self.demand_projection(df, forecast_periods)
        
        # 2. Calculate demand statistics
        demand_values = pd.to_numeric(df['demand'], errors='coerce').fillna(0)
        avg_demand = demand_values.mean()
        demand_std = demand_values.std()
        
        # 3. Safety Stock Calculation
        safety_stock_result = self.safety_stock_calculation(
            avg_demand=avg_demand,
            demand_std=demand_std,
            lead_time_days=lead_time
        )
        
        # 4. Reorder Point Calculation
        rop_result = self.reorder_point_calculation(
            avg_demand=avg_demand,
            lead_time_days=lead_time,
            safety_stock=safety_stock_result["safety_stock"]
        )
        
        # 5. Par Level Policy
        par_result = self.par_level_policy(
            avg_demand=avg_demand,
            demand_std=demand_std,
            lead_time_days=lead_time
        )
        
        # 6. Backtest Performance
        try:
            backtest_result = self.backtest_forecast(df)
        except Exception as e:
            backtest_result = {"error": f"Backtest failed: {e}", "accuracy_metrics": {}}
        
        # 7. Current Status Analysis
        days_of_supply = current_stock / avg_demand if avg_demand > 0 else float('inf')
        stock_status = "Excess" if current_stock > par_result["par_level"] else \
                     "Adequate" if current_stock > rop_result["reorder_point"] else \
                     "Reorder Needed"
        
        return {
            "part_number": part_number,
            "current_analysis": {
                "current_stock": current_stock,
                "days_of_supply": round(days_of_supply, 1),
                "stock_status": stock_status,
                "avg_demand": round(avg_demand, 2),
                "demand_std": round(demand_std, 2)
            },
            "demand_forecast": forecast_result,
            "safety_stock": safety_stock_result,
            "reorder_point": rop_result,
            "par_level": par_result,
            "backtest_performance": backtest_result,
            "recommendations": self._generate_policy_recommendations(
                current_stock, rop_result["reorder_point"], 
                par_result["par_level"], stock_status
            ),
            "timestamp": datetime.now().isoformat(),
            "service_level": self.service_level
        }
    
    def _generate_policy_recommendations(
        self, 
        current_stock: float, 
        rop: float, 
        par_level: float, 
        status: str
    ) -> List[str]:
        """Generate actionable policy recommendations."""
        
        recommendations = []
        
        if status == "Reorder Needed":
            qty_needed = par_level - current_stock
            recommendations.append(f"🔴 URGENT: Order {qty_needed:.0f} units immediately")
            recommendations.append(f"Current stock ({current_stock}) below ROP ({rop:.0f})")
        
        elif status == "Adequate":
            recommendations.append("🟡 Stock levels adequate, monitor for next review period")
            
        elif status == "Excess":
            excess_qty = current_stock - par_level
            recommendations.append(f"🟠 Excess stock: {excess_qty:.0f} units above par level")
            recommendations.append("Consider reducing next order quantity or extending review period")
        
        # General recommendations
        recommendations.append(f"Target par level: {par_level:.0f} units")
        recommendations.append(f"Reorder when stock hits: {rop:.0f} units")
        
        return recommendations
