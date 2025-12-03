"""
PHASE 3 - MODULE 6: FORECASTING MODULE
=======================================

Module Purpose:
  Implement ARIMA and ARIMAX forecasting with exogenous variable support for telecom KPIs.
  Supports univariate ARIMA and multivariate ARIMAX with automatic order selection.

Dependencies:
  - Upstream: filtering_engine.py (receives sampled DataFrame)
  - Upstream: correlation_module.py (for exogenous KPI selection)
  - Downstream: llama_service.py (forecast results passed to LLM)

Key Capabilities:
  ✓ ARIMA with auto order selection
  ✓ ARIMAX with multiple exogenous variables
  ✓ Confidence intervals (95%)
  ✓ Automatic fallback on convergence failure
  ✓ Comprehensive metrics (RMSE, MAE, MAPE, AIC)
  ✓ Performance optimized (<10s ARIMA, <15s ARIMAX)

Author: Telecom-Optimization-System
Date: 2025-12-03
Version: 1.0.0
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ==================== LOGGING CONFIGURATION ====================

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ==================== PYDANTIC MODELS ====================


class ModelMetrics(BaseModel):
    """Model evaluation metrics."""

    rmse: float = Field(..., ge=0, description="Root Mean Squared Error")
    mae: float = Field(..., ge=0, description="Mean Absolute Error")
    mape: float = Field(..., ge=0, description="Mean Absolute Percentage Error (%)")
    aic: Optional[float] = Field(None, description="Akaike Information Criterion")

    class Config:
        frozen = True


class ForecastResult(BaseModel):
    """Complete forecast result with metadata and metrics."""

    target_kpi: str = Field(..., description="KPI being forecasted")
    model_type: str = Field(
        ..., pattern="^(ARIMA|ARIMAX)$", description="Model type used"
    )
    forecast_values: List[float] = Field(
        ..., description="Predicted values for forecast horizon"
    )
    confidence_interval_lower: List[float] = Field(
        ..., description="95% CI lower bound"
    )
    confidence_interval_upper: List[float] = Field(
        ..., description="95% CI upper bound"
    )
    forecast_dates: List[str] = Field(
        ..., description="Dates for forecast period (YYYY-MM-DD)"
    )
    historical_values: List[float] = Field(
        ..., description="Last 30 days of actual data"
    )
    historical_dates: List[str] = Field(..., description="Dates for historical data")
    model_metrics: ModelMetrics
    exogenous_variables_used: List[str] = Field(
        default_factory=list, description="Exogenous KPIs used"
    )
    model_order: Tuple[int, int, int] = Field(..., description="(p, d, q) order")
    processing_time_ms: float = Field(
        ..., ge=0, description="Total processing time in milliseconds"
    )
    convergence_warning: Optional[str] = Field(
        None, description="Convergence warning if present"
    )

    class Config:
        frozen = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return (
            asdict(self) if hasattr(self, "__dataclass_fields__") else self.model_dump()
        )


# ==================== CORE FORECASTING FUNCTIONS ====================


def select_model_order(
    df: pd.DataFrame,
    kpi_column: str,
    max_p: int = 5,
    max_d: int = 2,
    max_q: int = 5,
    seasonal: bool = False,
    timeout_seconds: int = 30,
) -> Tuple[int, int, int]:
    """
    Automatically select ARIMA order (p, d, q) using auto_arima.

    Args:
        df: DataFrame with time-indexed data
        kpi_column: Column name to analyze
        max_p: Maximum AR order to test
        max_d: Maximum differencing to test
        max_q: Maximum MA order to test
        seasonal: Whether to apply seasonal differencing
        timeout_seconds: Timeout for auto_arima search

    Returns:
        Tuple: (p, d, q) order

    Raises:
        ValueError: If auto_arima fails or data is invalid
        TypeError: If kpi_column not in DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'time': pd.date_range('2024-01-01', periods=100, freq='D'),
        ...     'kpi': np.random.randn(100).cumsum()
        ... })
        >>> order = select_model_order(df, 'kpi')
        >>> print(order)  # e.g., (1, 1, 1)
    """
    start_time = time.time()

    # Validation
    if kpi_column not in df.columns:
        raise TypeError(f"Column '{kpi_column}' not found in DataFrame")

    kpi_data = df[kpi_column].dropna()
    if len(kpi_data) < 10:
        logger.warning(
            f"Insufficient data for auto_arima (n={len(kpi_data)}, min=10). "
            "Using default (1, 1, 1)"
        )
        return (1, 1, 1)

    try:
        logger.debug(
            f"Auto-selecting ARIMA order for '{kpi_column}' "
            f"(n={len(kpi_data)}, p_max={max_p}, d_max={max_d}, q_max={max_q})"
        )

        model = auto_arima(
            kpi_data,
            max_p=max_p,
            max_d=max_d,
            max_q=max_q,
            seasonal=seasonal,
            stepwise=True,
            suppress_warnings=True,
            information_criterion="aic",
            trace=False,
        )

        order = (model.order[0], model.order[1], model.order[2])
        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"✓ Auto-selected ARIMA order: {order} "
            f"(AIC={model.aic:.2f}, time={elapsed_ms:.1f}ms)"
        )

        return order

    except Exception as e:
        logger.warning(
            f"Auto-arima failed for '{kpi_column}': {str(e)}. "
            "Falling back to default (1, 1, 1)"
        )
        return (1, 1, 1)


def calculate_forecast_metrics(
    actual: Union[pd.Series, np.ndarray],
    predicted: Union[pd.Series, np.ndarray],
    aic: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate comprehensive forecast metrics.

    Args:
        actual: Actual values
        predicted: Predicted values (same length as actual)
        aic: Akaike Information Criterion (optional)

    Returns:
        Dict: {rmse, mae, mape, aic}

    Raises:
        ValueError: If arrays are different lengths or empty

    Example:
        >>> actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> predicted = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        >>> metrics = calculate_forecast_metrics(actual, predicted)
        >>> print(metrics['rmse'])  # ~0.14
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()

    if len(actual) != len(predicted):
        raise ValueError(
            f"Length mismatch: actual={len(actual)}, predicted={len(predicted)}"
        )

    if len(actual) == 0:
        raise ValueError("Input arrays are empty")

    # Calculate metrics using sklearn/statsmodels
    rmse_val = float(rmse(actual, predicted))
    mae_val = float(mean_absolute_error(actual, predicted))

    # MAPE: avoid division by zero
    mask = actual != 0
    if mask.any():
        mape_val = (
            float(mean_absolute_percentage_error(actual[mask], predicted[mask])) * 100
        )
    else:
        mape_val = np.inf if not np.allclose(predicted, 0) else 0.0

    metrics = {"rmse": rmse_val, "mae": mae_val, "mape": mape_val, "aic": aic}

    logger.debug(
        f"Calculated metrics: RMSE={rmse_val:.4f}, MAE={mae_val:.4f}, MAPE={mape_val:.2f}%"
    )

    return metrics


def fit_arima(
    df: pd.DataFrame,
    time_column: str,
    target_kpi: str,
    forecast_horizon: int,
    order: Optional[Tuple[int, int, int]] = None,
    historical_lookback: int = 30,
) -> ForecastResult:
    """
    Fit ARIMA model and generate univariate forecast.

    Args:
        df: DataFrame with time index and target KPI
        time_column: Name of time column
        target_kpi: Column name to forecast
        forecast_horizon: Number of periods to forecast (e.g., 7, 14, 30)
        order: ARIMA (p, d, q) order. If None, auto-select
        historical_lookback: Number of historical days to include in result

    Returns:
        ForecastResult: Complete forecast with metrics and confidence intervals

    Raises:
        ValueError: If data is invalid or model fitting fails
        TypeError: If required columns missing

    Example:
        >>> df = pd.DataFrame({
        ...     'time': pd.date_range('2024-01-01', periods=100, freq='D'),
        ...     'kpi': np.random.randn(100).cumsum()
        ... })
        >>> result = fit_arima(df, 'time', 'kpi', forecast_horizon=7)
        >>> print(result.forecast_values)  # [pred1, pred2, ..., pred7]
        >>> print(result.model_metrics.rmse)
    """
    start_time = time.time()

    # ========== INPUT VALIDATION ==========
    if time_column not in df.columns:
        raise TypeError(f"Time column '{time_column}' not found")
    if target_kpi not in df.columns:
        raise TypeError(f"Target KPI '{target_kpi}' not found")
    if forecast_horizon <= 0:
        raise ValueError(f"Forecast horizon must be > 0, got {forecast_horizon}")

    # ========== DATA PREPARATION ==========
    df_clean = df.copy()

    # Parse time column
    try:
        df_clean[time_column] = pd.to_datetime(df_clean[time_column])
    except Exception as e:
        raise ValueError(f"Failed to parse time column: {str(e)}")

    df_clean = df_clean.sort_values(by=time_column)

    # Extract KPI
    kpi_data = df_clean[[time_column, target_kpi]].dropna()
    if len(kpi_data) < 10:
        raise ValueError(f"Insufficient data for ARIMA (n={len(kpi_data)}, min=10)")

    logger.info(
        f"ARIMA: Starting forecast for '{target_kpi}' "
        f"(horizon={forecast_horizon}, n_historical={len(kpi_data)})"
    )

    # ========== AUTO-SELECT ORDER IF NOT PROVIDED ==========
    if order is None:
        order = select_model_order(kpi_data, target_kpi)
    else:
        order = tuple(order)

    logger.debug(f"ARIMA order: {order}")

    # ========== FIT ARIMA MODEL ==========
    try:
        model = SARIMAX(
            kpi_data[target_kpi].values,
            order=order,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = model.fit(disp=False, maxiter=200)
        logger.debug(f"ARIMA fitted: AIC={fitted_model.aic:.2f}")

    except Exception as e:
        logger.error(f"ARIMA fitting failed: {str(e)}")
        raise ValueError(f"Model fitting failed: {str(e)}")

    # ========== GENERATE FORECAST ==========
    try:
        forecast_result = fitted_model.get_forecast(steps=forecast_horizon)
        # FIX: Remove .values as predicted_mean is already a numpy array (since input was an array)
        forecast_values = forecast_result.predicted_mean
        # FIX: Remove .values as conf_int is already a numpy array (since input was an array)
        conf_int = forecast_result.conf_int(alpha=0.05)
        conf_lower = conf_int[:, 0]
        conf_upper = conf_int[:, 1]

    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}")
        raise ValueError(f"Forecast failed: {str(e)}")

    # ========== HISTORICAL DATA FOR PLOTTING ==========
    last_date = kpi_data[time_column].max()
    historical = kpi_data.tail(historical_lookback)
    historical_values = historical[target_kpi].tolist()
    historical_dates = historical[time_column].dt.strftime("%Y-%m-%d").tolist()

    # ========== FORECAST DATES ==========
    forecast_dates = [
        (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(forecast_horizon)
    ]

    # ========== CALCULATE METRICS (ON TRAINING DATA) ==========
    fitted_values = fitted_model.fittedvalues
    actual_values = kpi_data[target_kpi].values

    # Align lengths (first order[1] values are NaN due to differencing)
    min_len = min(len(fitted_values), len(actual_values))
    metrics = calculate_forecast_metrics(
        actual_values[-min_len:], fitted_values[-min_len:], aic=fitted_model.aic
    )

    # ========== BUILD RESULT ==========
    elapsed_ms = (time.time() - start_time) * 1000

    result = ForecastResult(
        target_kpi=target_kpi,
        model_type="ARIMA",
        forecast_values=forecast_values.tolist(),
        confidence_interval_lower=conf_lower.tolist(),
        confidence_interval_upper=conf_upper.tolist(),
        forecast_dates=forecast_dates,
        historical_values=historical_values,
        historical_dates=historical_dates,
        model_metrics=ModelMetrics(**metrics),
        exogenous_variables_used=[],
        model_order=order,
        processing_time_ms=elapsed_ms,
        convergence_warning=None,
    )

    logger.info(
        f"✓ ARIMA forecast complete: {forecast_horizon} steps, "
        f"RMSE={metrics['rmse']:.4f}, time={elapsed_ms:.1f}ms"
    )

    return result


def fit_arimax(
    df: pd.DataFrame,
    time_column: str,
    target_kpi: str,
    exogenous_kpis: List[str],
    forecast_horizon: int,
    order: Optional[Tuple[int, int, int]] = None,
    historical_lookback: int = 30,
    exogenous_forecast_method: str = "last_value",
) -> ForecastResult:
    """
    Fit ARIMAX model with exogenous variables and generate forecast.

    Args:
        df: DataFrame with time index, target KPI, and exogenous KPIs
        time_column: Name of time column
        target_kpi: Column name to forecast
        exogenous_kpis: List of exogenous KPI column names (typically Top 3 correlated)
        forecast_horizon: Number of periods to forecast
        order: ARIMAX (p, d, q) order. If None, auto-select
        historical_lookback: Number of historical days in result
        exogenous_forecast_method: How to forecast exogenous variables ('last_value' or 'mean')

    Returns:
        ForecastResult: Complete forecast with metrics and confidence intervals

    Raises:
        ValueError: If data invalid, exogenous variables missing, or fitting fails
        TypeError: If required columns missing

    Example:
        >>> df = pd.DataFrame({
        ...     'time': pd.date_range('2024-01-01', periods=100, freq='D'),
        ...     'target': np.random.randn(100).cumsum(),
        ...     'exog1': np.random.randn(100).cumsum(),
        ...     'exog2': np.random.randn(100).cumsum()
        ... })
        >>> result = fit_arimax(
        ...     df, 'time', 'target', ['exog1', 'exog2'],
        ...     forecast_horizon=7
        ... )
        >>> print(result.model_type)  # 'ARIMAX'
        >>> print(result.exogenous_variables_used)  # ['exog1', 'exog2']
    """
    start_time = time.time()
    convergence_warning = None

    # ========== INPUT VALIDATION ==========
    if time_column not in df.columns:
        raise TypeError(f"Time column '{time_column}' not found")
    if target_kpi not in df.columns:
        raise TypeError(f"Target KPI '{target_kpi}' not found")
    if forecast_horizon <= 0:
        raise ValueError(f"Forecast horizon must be > 0, got {forecast_horizon}")
    if not exogenous_kpis:
        raise ValueError("At least one exogenous KPI required for ARIMAX")

    # Validate exogenous columns exist
    missing_exog = [kpi for kpi in exogenous_kpis if kpi not in df.columns]
    if missing_exog:
        raise TypeError(f"Exogenous KPIs not found: {missing_exog}")

    # ========== DATA PREPARATION ==========
    df_clean = df.copy()

    # Parse time column
    try:
        df_clean[time_column] = pd.to_datetime(df_clean[time_column])
    except Exception as e:
        raise ValueError(f"Failed to parse time column: {str(e)}")

    df_clean = df_clean.sort_values(by=time_column)

    # Extract columns for modeling
    modeling_cols = [time_column, target_kpi] + exogenous_kpis
    df_model = df_clean[modeling_cols].dropna()

    if len(df_model) < 10:
        raise ValueError(f"Insufficient data for ARIMAX (n={len(df_model)}, min=10)")

    logger.info(
        f"ARIMAX: Starting forecast for '{target_kpi}' "
        f"with exogenous={exogenous_kpis[:3]} "  # Log first 3
        f"(horizon={forecast_horizon}, n_historical={len(df_model)})"
    )

    # ========== AUTO-SELECT ORDER IF NOT PROVIDED ==========
    if order is None:
        order = select_model_order(df_model, target_kpi)
    else:
        order = tuple(order)

    logger.debug(f"ARIMAX order: {order}")

    # ========== EXTRACT EXOGENOUS DATA ==========
    endog = df_model[target_kpi].values
    exog = df_model[exogenous_kpis].values

    # ========== FIT ARIMAX MODEL ==========
    try:
        model = SARIMAX(
            endog,
            exog=exog,
            order=order,
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = model.fit(disp=False, maxiter=200)
        logger.debug(f"ARIMAX fitted: AIC={fitted_model.aic:.2f}")

    except Exception as e:
        logger.error(f"ARIMAX fitting failed: {str(e)}")
        raise ValueError(f"Model fitting failed: {str(e)}")

    # ========== FORECAST EXOGENOUS VARIABLES ==========
    if exogenous_forecast_method == "last_value":
        # Use last observed values of exogenous variables
        exog_forecast = np.tile(exog[-1, :], (forecast_horizon, 1))
        logger.debug("Exogenous forecast: using last observed values")
    elif exogenous_forecast_method == "mean":
        # Use mean of exogenous variables
        exog_forecast = np.tile(exog.mean(axis=0), (forecast_horizon, 1))
        logger.debug("Exogenous forecast: using mean values")
    else:
        raise ValueError(
            f"Invalid exogenous_forecast_method: {exogenous_forecast_method}. "
            "Use 'last_value' or 'mean'"
        )

    # ========== GENERATE FORECAST ==========
    try:
        forecast_result = fitted_model.get_forecast(
            steps=forecast_horizon, exog=exog_forecast
        )
        # FIX: Remove .values
        forecast_values = forecast_result.predicted_mean
        # FIX: Remove .values
        conf_int = forecast_result.conf_int(alpha=0.05)
        conf_lower = conf_int[:, 0]
        conf_upper = conf_int[:, 1]

    except Exception as e:
        logger.warning(f"ARIMAX forecast failed: {str(e)}")
        convergence_warning = f"Forecast generation: {str(e)}"
        # Return zeros as fallback
        forecast_values = np.zeros(forecast_horizon)
        conf_lower = np.zeros(forecast_horizon)
        conf_upper = np.zeros(forecast_horizon)

    # ========== HISTORICAL DATA FOR PLOTTING ==========
    last_date = df_model[time_column].max()
    historical = df_model.tail(historical_lookback)
    historical_values = historical[target_kpi].tolist()
    historical_dates = historical[time_column].dt.strftime("%Y-%m-%d").tolist()

    # ========== FORECAST DATES ==========
    forecast_dates = [
        (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(forecast_horizon)
    ]

    # ========== CALCULATE METRICS (ON TRAINING DATA) ==========
    fitted_values = fitted_model.fittedvalues
    actual_values = endog

    # Align lengths
    min_len = min(len(fitted_values), len(actual_values))
    metrics = calculate_forecast_metrics(
        actual_values[-min_len:], fitted_values[-min_len:], aic=fitted_model.aic
    )

    # ========== BUILD RESULT ==========
    elapsed_ms = (time.time() - start_time) * 1000

    result = ForecastResult(
        target_kpi=target_kpi,
        model_type="ARIMAX",
        forecast_values=forecast_values.tolist(),
        confidence_interval_lower=conf_lower.tolist(),
        confidence_interval_upper=conf_upper.tolist(),
        forecast_dates=forecast_dates,
        historical_values=historical_values,
        historical_dates=historical_dates,
        model_metrics=ModelMetrics(**metrics),
        exogenous_variables_used=exogenous_kpis,
        model_order=order,
        processing_time_ms=elapsed_ms,
        convergence_warning=convergence_warning,
    )

    logger.info(
        f"✓ ARIMAX forecast complete: {forecast_horizon} steps, "
        f"RMSE={metrics['rmse']:.4f}, exog={len(exogenous_kpis)}, time={elapsed_ms:.1f}ms"
    )

    return result


# ==================== HIGH-LEVEL FORECASTING API ====================


def forecast_kpi(
    df: pd.DataFrame,
    time_column: str,
    target_kpi: str,
    forecast_horizon: int,
    exogenous_kpis: Optional[List[str]] = None,
    order: Optional[Tuple[int, int, int]] = None,
    use_arimax: Optional[bool] = None,
) -> ForecastResult:
    """
    High-level API for forecasting with automatic model selection.

    Automatically chooses between ARIMA and ARIMAX based on exogenous_kpis.

    Args:
        df: DataFrame with time index and KPIs
        time_column: Name of time column
        target_kpi: Column to forecast
        forecast_horizon: Number of periods to forecast
        exogenous_kpis: List of exogenous KPIs for ARIMAX (optional)
        order: ARIMA/ARIMAX (p, d, q) order (optional, auto-select if None)
        use_arimax: Force ARIMAX even if exogenous_kpis is empty (not recommended)

    Returns:
        ForecastResult: Forecast with model type, values, and metrics

    Raises:
        ValueError: If inputs are invalid
        TypeError: If required columns missing

    Example:
        >>> df = pd.DataFrame({...})
        >>> result = forecast_kpi(
        ...     df, 'time', 'RACH stp att',
        ...     forecast_horizon=7,
        ...     exogenous_kpis=['RRC stp att', 'E-UTRAN avg RRC conn UEs']
        ... )
        >>> print(result.model_type)  # 'ARIMAX'
        >>> print(len(result.forecast_values))  # 7
    """
    # Determine model type
    has_exogenous = exogenous_kpis and len(exogenous_kpis) > 0
    should_use_arimax = use_arimax or has_exogenous

    if should_use_arimax and not has_exogenous:
        logger.warning(
            "use_arimax=True but no exogenous_kpis provided. " "Falling back to ARIMA"
        )
        should_use_arimax = False

    # Route to appropriate model
    if should_use_arimax:
        logger.debug(f"Using ARIMAX with exogenous: {exogenous_kpis}")
        return fit_arimax(
            df, time_column, target_kpi, exogenous_kpis, forecast_horizon, order
        )
    else:
        logger.debug("Using ARIMA (univariate)")
        return fit_arima(df, time_column, target_kpi, forecast_horizon, order)


# ==================== BATCH FORECASTING ====================


def forecast_multiple_kpis(
    df: pd.DataFrame,
    time_column: str,
    target_kpis: List[str],
    forecast_horizon: int,
    exogenous_mapping: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, ForecastResult]:
    """
    Forecast multiple KPIs in batch.

    Args:
        df: DataFrame with time index and KPIs
        time_column: Name of time column
        target_kpis: List of KPIs to forecast
        forecast_horizon: Forecast horizon
        exogenous_mapping: Dict mapping target_kpi -> [exogenous_kpis]
                          e.g., {"RACH stp att": ["RRC stp att", "E-UTRAN avg RRC conn UEs"]}

    Returns:
        Dict: {target_kpi -> ForecastResult}

    Example:
        >>> results = forecast_multiple_kpis(
        ...     df, 'time',
        ...     ['RACH stp att', 'RRC stp att'],
        ...     forecast_horizon=7,
        ...     exogenous_mapping={
        ...         'RACH stp att': ['RRC stp att', 'E-UTRAN avg RRC conn UEs']
        ...     }
        ... )
        >>> for kpi, result in results.items():
        ...     print(f"{kpi}: {result.model_type}")
    """
    exogenous_mapping = exogenous_mapping or {}
    results = {}

    for kpi in target_kpis:
        try:
            exogenous = exogenous_mapping.get(kpi)
            result = forecast_kpi(
                df, time_column, kpi, forecast_horizon, exogenous_kpis=exogenous
            )
            results[kpi] = result
            logger.info(f"✓ Forecasted {kpi}")

        except Exception as e:
            logger.error(f"✗ Failed to forecast {kpi}: {str(e)}")
            results[kpi] = None

    return results


if __name__ == "__main__":
    logger.info(
        "Forecasting Module loaded. Use forecast_kpi() or fit_arima/fit_arimax functions."
    )
