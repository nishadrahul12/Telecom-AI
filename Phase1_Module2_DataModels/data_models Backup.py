# =============================================================================
# PROJECT: Intelligent Telecom Optimization System
# MODULE: Phase 1, Module 2 - Data Models (Pydantic Schemas)
# PURPOSE: Type-safe schemas for data validation, serialization, and API contracts
# AUTHOR: Telecom Optimization Team
# VERSION: 1.0.0
# =============================================================================

"""
Data Models Module - Pydantic Schemas for Type Safety

This module defines all Pydantic models used throughout the system for:
1. Type validation and documentation
2. JSON serialization for APIs
3. Data contract enforcement
4. Error handling and validation rules

Models are organized in logical groups:
- Core Data Models (DataFrameMetadata, ColumnClassification)
- Analytics Results (AnomalyResult, CorrelationResult, ForecastResult)
- Request/Response Models (FilterRequest, DetectionRequest)
- LLM Integration Schemas (Causal, Scenario, Interpretation)

Example:
    >>> from data_models import DataFrameMetadata, AnomalyResult
    >>> metadata = DataFrameMetadata(
    ...     file_path="/data/sample.csv",
    ...     total_rows=10000,
    ...     time_format="Daily"
    ... )
    >>> anomaly = AnomalyResult(
    ...     timestamp="2024-01-01",
    ...     kpi_name="DL_Throughput",
    ...     observed_value=2.5,
    ...     z_score=3.2,
    ...     severity="High"
    ... )
"""

from typing import List, Dict, Optional, Literal, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json


# =============================================================================
# ENUMS - Controlled Vocabularies
# =============================================================================

class SeverityLevel(str, Enum):
    """Severity levels for anomalies and alerts."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class ColumnType(str, Enum):
    """Classification of data columns."""
    DIMENSION_TEXT = "Dimension_Text"
    DIMENSION_ID = "Dimension_ID"
    KPI = "KPI"
    TIME = "Time"
    UNKNOWN = "Unknown"


class TimeFormat(str, Enum):
    """Supported time formats in data."""
    DAILY = "Daily"  # MM/DD/YYYY
    HOURLY = "Hourly"  # YYYY-MM-DD HH:MM:SS
    MONTHLY = "Monthly"  # YYYY-MM
    WEEKLY = "Weekly"  # YYYY-W##


class AggregationLevel(str, Enum):
    """Telecom data aggregation hierarchy."""
    PLMN = "PLMN"  # Highest level
    REGION = "Region"
    CARRIER = "Carrier"  # Frequency band
    CELL = "Cell"  # Lowest level


class AnomalyMethod(str, Enum):
    """Anomaly detection methods."""
    Z_SCORE = "Z-Score"
    IQR = "IQR"
    ISOLATION_FOREST = "Isolation_Forest"


# =============================================================================
# CORE DATA MODELS
# =============================================================================

class ColumnClassification(BaseModel):
    """
    Metadata about a single column's classification.
    
    Attributes:
        column_name: Name of the column in source data
        column_type: Classification (Dimension_Text, Dimension_ID, KPI, Time)
        data_type: Python data type (str, int, float, datetime)
        non_null_count: Number of non-null values
        unique_count: Number of unique values
        sample_values: First 3 unique values as examples
        is_numeric: Whether column contains numeric data
        
    Example:
        >>> col = ColumnClassification(
        ...     column_name="DL_Throughput",
        ...     column_type=ColumnType.KPI,
        ...     data_type="float",
        ...     non_null_count=9950,
        ...     unique_count=8234,
        ...     sample_values=[2.45, 2.50, 2.48],
        ...     is_numeric=True
        ... )
    """
    
    column_name: str = Field(..., description="Column name from source data")
    column_type: ColumnType = Field(..., description="Type classification")
    data_type: str = Field(..., description="Python data type (str, int, float, datetime)")
    non_null_count: int = Field(..., ge=0, description="Count of non-null values")
    unique_count: int = Field(..., ge=0, description="Count of unique values")
    sample_values: List[Any] = Field(..., description="First 3 unique values")
    is_numeric: bool = Field(..., description="Whether column is numeric")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "column_name": "DL_Throughput",
            "column_type": "KPI",
            "data_type": "float",
            "non_null_count": 9950,
            "unique_count": 8234,
            "sample_values": [2.45, 2.50, 2.48],
            "is_numeric": True
        }
    })


class DataFrameMetadata(BaseModel):
    """
    Metadata about ingested DataFrame from Module 1.
    
    This is the OUTPUT contract from data_ingestion.py and serves as the
    INPUT to all downstream analytics modules.
    
    Attributes:
        file_path: Full path to source CSV file
        total_rows: Total number of rows after ingestion
        total_columns: Total number of columns
        time_format: Detected time format (Daily, Hourly, etc.)
        aggregation_level: PLMN, Region, Carrier, or Cell level
        ingestion_timestamp: When data was loaded
        columns: List of ColumnClassification objects
        time_column: Name of detected time column
        dimension_columns: List of dimension column names
        kpi_columns: List of KPI column names
        date_range_start: Earliest timestamp in data
        date_range_end: Latest timestamp in data
        encoding: File encoding (e.g., utf-8)
        has_missing_values: Whether NaN values exist
        sampling_applied: Whether smart sampling was used
        original_row_count: Pre-sampling row count
        
    Example:
        >>> metadata = DataFrameMetadata(
        ...     file_path="/data/cell_level.csv",
        ...     total_rows=50000,
        ...     total_columns=25,
        ...     time_format=TimeFormat.HOURLY,
        ...     aggregation_level=AggregationLevel.CELL,
        ...     columns=[...],
        ...     time_column="Timestamp",
        ...     dimension_columns=["Region", "Carrier", "Cell_ID"],
        ...     kpi_columns=["DL_Throughput", "UL_Throughput", ...],
        ...     date_range_start="2024-01-01",
        ...     date_range_end="2024-01-31"
        ... )
    """
    
    file_path: str = Field(..., description="Full path to source CSV file")
    total_rows: int = Field(..., ge=1, description="Total rows in ingested data")
    total_columns: int = Field(..., ge=1, description="Total columns")
    time_format: TimeFormat = Field(..., description="Detected time format")
    aggregation_level: AggregationLevel = Field(..., description="Data aggregation level")
    ingestion_timestamp: datetime = Field(default_factory=datetime.utcnow, 
                                         description="When data was ingested")
    columns: List[ColumnClassification] = Field(..., description="Column metadata list")
    time_column: str = Field(..., description="Name of time column")
    dimension_columns: List[str] = Field(default_factory=list, 
                                        description="Dimension column names")
    kpi_columns: List[str] = Field(default_factory=list, description="KPI column names")
    date_range_start: Optional[str] = Field(None, description="Earliest timestamp")
    date_range_end: Optional[str] = Field(None, description="Latest timestamp")
    encoding: str = Field(default="utf-8", description="File encoding")
    has_missing_values: bool = Field(default=False, description="Whether NaN exists")
    sampling_applied: bool = Field(default=False, 
                                   description="Whether smart sampling was used")
    original_row_count: Optional[int] = Field(None, 
                                             description="Row count before sampling")
    
    @field_validator('total_rows', 'total_columns')
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be positive integer")
        return v
    
    @field_validator('date_range_start', 'date_range_end', mode='before')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v and not isinstance(v, str):
            return str(v)
        return v
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "file_path": "/data/cell_level.csv",
            "total_rows": 50000,
            "total_columns": 25,
            "time_format": "Hourly",
            "aggregation_level": "Cell",
            "columns": [],
            "time_column": "Timestamp",
            "dimension_columns": ["Region", "Carrier"],
            "kpi_columns": ["DL_Throughput", "UL_Throughput"],
            "date_range_start": "2024-01-01",
            "date_range_end": "2024-01-31"
        }
    })


# =============================================================================
# ANALYTICS RESULT MODELS
# =============================================================================

class AnomalyResult(BaseModel):
    """
    Single anomaly detected by anomaly detection module.
    
    Represents one anomalous data point with metadata about why it's anomalous.
    
    Attributes:
        timestamp: Time when anomaly occurred
        kpi_name: Name of the KPI that was anomalous
        observed_value: Actual value from data
        expected_value: Mean or predicted value
        z_score: Z-score (for Z-Score method)
        deviation_percent: Percentage deviation from expected
        severity: Severity level (Low, Medium, High, Critical)
        method: Detection method used (Z-Score, IQR)
        lower_bound: Lower acceptable range bound
        upper_bound: Upper acceptable range bound
        dimension_filters: Dimension values (Region, Carrier, Cell_ID)
        
    Example:
        >>> anomaly = AnomalyResult(
        ...     timestamp="2024-01-15 14:30:00",
        ...     kpi_name="DL_Throughput",
        ...     observed_value=0.5,
        ...     expected_value=2.4,
        ...     z_score=-3.8,
        ...     deviation_percent=-79.2,
        ...     severity=SeverityLevel.CRITICAL,
        ...     method=AnomalyMethod.Z_SCORE,
        ...     lower_bound=1.8,
        ...     upper_bound=3.2,
        ...     dimension_filters={"Region": "North", "Carrier": "L2100", "Cell_ID": "C123"}
        ... )
    """
    
    timestamp: str = Field(..., description="ISO format timestamp when anomaly occurred")
    kpi_name: str = Field(..., description="Name of anomalous KPI")
    observed_value: float = Field(..., description="Actual observed value")
    expected_value: Optional[float] = Field(None, description="Mean or predicted value")
    z_score: Optional[float] = Field(None, description="Z-score (σ units from mean)")
    deviation_percent: Optional[float] = Field(None, description="Percentage deviation from expected")
    severity: SeverityLevel = Field(..., description="Anomaly severity level")
    method: AnomalyMethod = Field(..., description="Detection method used")
    lower_bound: Optional[float] = Field(None, description="Lower acceptable bound")
    upper_bound: Optional[float] = Field(None, description="Upper acceptable bound")
    dimension_filters: Dict[str, str] = Field(default_factory=dict, 
                                            description="Dimension values (Region, Carrier, etc.)")
    
    @field_validator('z_score', 'observed_value', 'expected_value', mode='before')
    @classmethod
    def validate_numeric(cls, v):
        if v is None:
            return v
        try:
            return float(v)
        except (ValueError, TypeError):
            raise ValueError("Must be convertible to float")
    
    @field_validator('deviation_percent', mode='before')
    @classmethod
    def validate_deviation_percent(cls, v):
        if v is None:
            return v
        v = float(v)
        if v < -100:
            raise ValueError("Deviation cannot be less than -100%")
        return v
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "timestamp": "2024-01-15 14:30:00",
            "kpi_name": "DL_Throughput",
            "observed_value": 0.5,
            "expected_value": 2.4,
            "z_score": -3.8,
            "deviation_percent": -79.2,
            "severity": "Critical",
            "method": "Z-Score",
            "lower_bound": 1.8,
            "upper_bound": 3.2,
            "dimension_filters": {"Region": "North", "Carrier": "L2100"}
        }
    })


class CorrelationPair(BaseModel):
    """
    Correlation between two KPIs.
    
    Represents a single correlation result with statistical information.
    
    Attributes:
        kpi_x: First KPI name
        kpi_y: Second KPI name
        correlation_score: Pearson correlation coefficient (-1 to 1)
        p_value: Statistical significance p-value (0 to 1)
        is_significant: Whether correlation is statistically significant (p < 0.05)
        data_points_used: Number of data points in calculation
        interpretation: Human-readable interpretation
        
    Example:
        >>> corr = CorrelationPair(
        ...     kpi_x="DL_Throughput",
        ...     kpi_y="Signal_Strength",
        ...     correlation_score=0.82,
        ...     p_value=0.0001,
        ...     is_significant=True,
        ...     data_points_used=9850,
        ...     interpretation="Strong positive correlation"
        ... )
    """
    
    kpi_x: str = Field(..., description="First KPI name")
    kpi_y: str = Field(..., description="Second KPI name")
    correlation_score: float = Field(..., ge=-1.0, le=1.0, 
                                    description="Pearson correlation [-1, 1]")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Statistical p-value [0, 1]")
    is_significant: bool = Field(..., description="Is p-value < 0.05?")
    data_points_used: int = Field(..., ge=2, description="Sample size used")
    interpretation: Optional[str] = Field(None, description="Human-readable interpretation")
    
    @field_validator('correlation_score')
    @classmethod
    def validate_correlation(cls, v: float) -> float:
        if not -1.0 <= v <= 1.0:
            raise ValueError("Correlation must be between -1.0 and 1.0")
        return v
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "kpi_x": "DL_Throughput",
            "kpi_y": "Signal_Strength",
            "correlation_score": 0.82,
            "p_value": 0.0001,
            "is_significant": True,
            "data_points_used": 9850,
            "interpretation": "Strong positive correlation: Signal improvement increases throughput"
        }
    })


class CorrelationResult(BaseModel):
    """
    Top-3 correlations for a single KPI.
    
    Output from correlation analysis: one KPI and its 3 strongest correlations.
    
    Attributes:
        kpi_name: Name of the reference KPI
        top_3_correlations: List of top 3 correlation pairs (by absolute value)
        analysis_timestamp: When analysis was performed
        
    Example:
        >>> result = CorrelationResult(
        ...     kpi_name="DL_Throughput",
        ...     top_3_correlations=[corr1, corr2, corr3],
        ...     analysis_timestamp="2024-01-15T14:30:00"
        ... )
    """
    
    kpi_name: str = Field(..., description="Reference KPI name")
    top_3_correlations: List[CorrelationPair] = Field(min_length=0, max_length=3,
                                                      description="Top 3 correlations by absolute value")
    analysis_timestamp: datetime = Field(default_factory=datetime.utcnow,
                                        description="When analysis was performed")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "kpi_name": "DL_Throughput",
            "top_3_correlations": [],
            "analysis_timestamp": "2024-01-15T14:30:00"
        }
    })


class ForecastValue(BaseModel):
    """
    Single forecast prediction with confidence interval.
    
    One time-step forecast from ARIMA/ARIMAX model.
    
    Attributes:
        timestamp: Forecasted time point
        predicted_value: Point estimate (mean forecast)
        lower_ci: Lower confidence interval bound (95%)
        upper_ci: Upper confidence interval bound (95%)
        confidence_level: CI confidence level (e.g., 0.95)
        
    Example:
        >>> fv = ForecastValue(
        ...     timestamp="2024-02-01",
        ...     predicted_value=2.45,
        ...     lower_ci=2.10,
        ...     upper_ci=2.80,
        ...     confidence_level=0.95
        ... )
    """
    
    timestamp: str = Field(..., description="Forecasted time point")
    predicted_value: float = Field(..., description="Point estimate (mean forecast)")
    lower_ci: float = Field(..., description="Lower confidence interval bound")
    upper_ci: float = Field(..., description="Upper confidence interval bound")
    confidence_level: float = Field(default=0.95, ge=0.80, le=0.99,
                                   description="Confidence level (0.80-0.99)")
    
    @field_validator('lower_ci', 'upper_ci')
    @classmethod
    def validate_ci_bounds(cls, v: float) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError("CI bounds must be numeric")
        return v
    
    @field_validator('upper_ci')
    @classmethod
    def validate_upper_greater_than_lower(cls, v: float, info) -> float:
        lower = info.data.get('lower_ci')
        if lower is not None and v <= lower:
            raise ValueError("upper_ci must be > lower_ci")
        return v
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "timestamp": "2024-02-01",
            "predicted_value": 2.45,
            "lower_ci": 2.10,
            "upper_ci": 2.80,
            "confidence_level": 0.95
        }
    })


class ForecastResult(BaseModel):
    """
    Complete forecast for a KPI over multiple time periods.
    
    Output from ARIMA/ARIMAX forecasting module.
    
    Attributes:
        kpi_name: Name of forecasted KPI
        forecast_period: Forecast horizon (e.g., "7 days", "30 days")
        forecast_values: List of ForecastValue predictions
        model_type: Model used (ARIMA, ARIMAX)
        rmse: Root Mean Square Error on validation set (if available)
        fit_timestamp: When model was fitted
        exogenous_variables: List of exogenous KPI names (if ARIMAX)
        
    Example:
        >>> forecast = ForecastResult(
        ...     kpi_name="DL_Throughput",
        ...     forecast_period="30 days",
        ...     forecast_values=[fv1, fv2, ...],
        ...     model_type="ARIMAX",
        ...     exogenous_variables=["Signal_Strength", "Traffic_Volume"]
        ... )
    """
    
    kpi_name: str = Field(..., description="Forecasted KPI name")
    forecast_period: str = Field(..., description="Forecast horizon (e.g., '7 days')")
    forecast_values: List[ForecastValue] = Field(..., min_length=1,
                                                 description="List of forecast points")
    model_type: Literal["ARIMA", "ARIMAX"] = Field(..., description="Model type used")
    rmse: Optional[float] = Field(None, ge=0, description="RMSE on validation set")
    fit_timestamp: datetime = Field(default_factory=datetime.utcnow,
                                   description="When model was fitted")
    exogenous_variables: List[str] = Field(default_factory=list,
                                          description="Exogenous KPI names (ARIMAX only)")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "kpi_name": "DL_Throughput",
            "forecast_period": "30 days",
            "forecast_values": [],
            "model_type": "ARIMAX",
            "exogenous_variables": ["Signal_Strength"]
        }
    })


class FilteredDataFrameResult(BaseModel):
    """
    Result from filtered data extraction (Module 3).
    
    This is the OUTPUT from filtering module and INPUT to analysis modules.
    
    Attributes:
        original_metadata: DataFrameMetadata from ingestion
        filter_selections: User's filter selections
        filtered_row_count: Number of rows after filtering
        dimension_values_applied: Actual dimension values used
        
    Example:
        >>> result = FilteredDataFrameResult(
        ...     original_metadata=metadata,
        ...     filter_selections={"Region": "North", "Carrier": "L2100"},
        ...     filtered_row_count=5000,
        ...     dimension_values_applied={"Region": "North", "Carrier": "L2100"}
        ... )
    """
    
    original_metadata: DataFrameMetadata = Field(..., 
                                                 description="Original ingested metadata")
    filter_selections: Dict[str, str] = Field(..., description="User filter selections")
    filtered_row_count: int = Field(..., ge=0, description="Rows after filtering")
    dimension_values_applied: Dict[str, str] = Field(...,
                                                     description="Actual dimension values used")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "filter_selections": {"Region": "North"},
            "filtered_row_count": 5000,
            "dimension_values_applied": {"Region": "North"}
        }
    })


# =============================================================================
# REQUEST/RESPONSE MODELS (API Contracts)
# =============================================================================

class FilterRequest(BaseModel):
    """
    User's filter request for data extraction.
    
    Used by frontend to request filtered data from backend.
    
    Attributes:
        region: Filter by region (optional)
        carrier: Filter by carrier/band (optional)
        cell_id: Filter by cell ID (optional)
        start_date: Start date for time range
        end_date: End date for time range
        kpi_names: List of KPIs to include (empty = all)
        
    Example:
        >>> req = FilterRequest(
        ...     region="North",
        ...     carrier="L2100",
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     kpi_names=["DL_Throughput", "Signal_Strength"]
        ... )
    """
    
    region: Optional[str] = Field(None, description="Filter by region")
    carrier: Optional[str] = Field(None, description="Filter by carrier/band")
    cell_id: Optional[str] = Field(None, description="Filter by cell ID")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    kpi_names: List[str] = Field(default_factory=list, description="KPI names (empty=all)")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "region": "North",
            "carrier": "L2100",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "kpi_names": ["DL_Throughput", "Signal_Strength"]
        }
    })


class AnomalyDetectionRequest(BaseModel):
    """
    Request for anomaly detection analysis.
    
    Parameters for anomaly detection module.
    
    Attributes:
        filtered_data_result: Input filtered data
        method: Detection method (Z-Score or IQR)
        z_score_threshold: Z-score threshold (default 3.0 for 3σ)
        kpi_names: Which KPIs to analyze (empty = all)
        
    Example:
        >>> req = AnomalyDetectionRequest(
        ...     filtered_data_result=filtered_result,
        ...     method=AnomalyMethod.Z_SCORE,
        ...     z_score_threshold=3.0,
        ...     kpi_names=["DL_Throughput", "Signal_Strength"]
        ... )
    """
    
    filtered_data_result: FilteredDataFrameResult = Field(..., description="Input filtered data")
    method: AnomalyMethod = Field(default=AnomalyMethod.Z_SCORE, description="Detection method")
    z_score_threshold: float = Field(default=3.0, gt=0, description="Z-score threshold (σ units)")
    kpi_names: List[str] = Field(default_factory=list, description="KPI names (empty=all)")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "method": "Z-Score",
            "z_score_threshold": 3.0,
            "kpi_names": ["DL_Throughput"]
        }
    })


class ForecastRequest(BaseModel):
    """
    Request for forecasting analysis.
    
    Parameters for ARIMA/ARIMAX forecasting.
    
    Attributes:
        filtered_data_result: Input filtered data
        kpi_name: Single KPI to forecast
        forecast_periods: Number of periods to forecast
        model_type: ARIMA or ARIMAX
        exogenous_kpi_names: For ARIMAX, list of exogenous KPIs
        
    Example:
        >>> req = ForecastRequest(
        ...     filtered_data_result=filtered_result,
        ...     kpi_name="DL_Throughput",
        ...     forecast_periods=30,
        ...     model_type="ARIMAX",
        ...     exogenous_kpi_names=["Signal_Strength"]
        ... )
    """
    
    filtered_data_result: FilteredDataFrameResult = Field(..., description="Input filtered data")
    kpi_name: str = Field(..., description="Single KPI to forecast")
    forecast_periods: int = Field(default=30, gt=0, le=365, description="Periods to forecast (1-365)")
    model_type: Literal["ARIMA", "ARIMAX"] = Field(default="ARIMA", description="Model type")
    exogenous_kpi_names: List[str] = Field(default_factory=list, 
                                          description="Exogenous KPIs (for ARIMAX)")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "kpi_name": "DL_Throughput",
            "forecast_periods": 30,
            "model_type": "ARIMAX",
            "exogenous_kpi_names": ["Signal_Strength"]
        }
    })


# =============================================================================
# LLM REQUEST SCHEMAS (Structured JSON for Ollama)
# =============================================================================

class LLMCausalAnalysisRequest(BaseModel):
    """
    Schema for "Why did this anomaly occur?" analysis.
    
    Sent to Llama 70B to analyze root causes of detected anomalies.
    
    Attributes:
        anomaly: The detected anomaly
        historical_context: Recent KPI trends for context
        correlated_kpis: Correlated KPIs that may explain the anomaly
        domain_context: Telecom domain context (cell info, region, carrier)
        
    Example:
        >>> req = LLMCausalAnalysisRequest(
        ...     anomaly=anomaly_result,
        ...     historical_context={
        ...         "mean_recent": 2.4,
        ...         "trend": "stable",
        ...         "recent_events": []
        ...     },
        ...     correlated_kpis=[corr1, corr2],
        ...     domain_context={
        ...         "region": "North",
        ...         "carrier": "L2100",
        ...         "cell_type": "Macro"
        ...     }
        ... )
    """
    
    anomaly: AnomalyResult = Field(..., description="The detected anomaly")
    historical_context: Dict[str, Any] = Field(..., 
                                              description="Recent trends and context")
    correlated_kpis: List[CorrelationPair] = Field(default_factory=list,
                                                  description="Related KPIs and correlations")
    domain_context: Dict[str, str] = Field(default_factory=dict,
                                          description="Telecom domain context")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "anomaly": {},
            "historical_context": {"mean_recent": 2.4, "trend": "stable"},
            "correlated_kpis": [],
            "domain_context": {"region": "North", "carrier": "L2100"}
        }
    })


class LLMScenarioPlanningRequest(BaseModel):
    """
    Schema for "What if?" scenario planning analysis.
    
    Sent to Llama 70B for scenario analysis and planning.
    
    Attributes:
        scenario_description: Hypothetical scenario (e.g., "carrier added 20% traffic")
        affected_kpis: KPIs affected by scenario
        baseline_values: Current baseline values
        change_magnitude: Magnitude of change (percentage or absolute)
        
    Example:
        >>> req = LLMScenarioPlanningRequest(
        ...     scenario_description="Carrier L2100 receives 20% additional traffic",
        ...     affected_kpis=["DL_Throughput", "UL_Throughput", "Latency"],
        ...     baseline_values={"DL_Throughput": 2.4, "Latency": 45},
        ...     change_magnitude=20
        ... )
    """
    
    scenario_description: str = Field(..., description="Hypothetical scenario text")
    affected_kpis: List[str] = Field(..., min_length=1, description="KPIs affected")
    baseline_values: Dict[str, float] = Field(..., description="Current baseline values")
    change_magnitude: float = Field(..., description="Change magnitude (%)")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "scenario_description": "Carrier L2100 receives 20% additional traffic",
            "affected_kpis": ["DL_Throughput", "Latency"],
            "baseline_values": {"DL_Throughput": 2.4, "Latency": 45},
            "change_magnitude": 20
        }
    })


class LLMCorrelationInterpretationRequest(BaseModel):
    """
    Schema for "So what?" interpretation analysis.
    
    Sent to Llama 70B to interpret correlation findings.
    
    Attributes:
        correlation_result: Correlation analysis result
        domain_knowledge: Domain context and business impact
        
    Example:
        >>> req = LLMCorrelationInterpretationRequest(
        ...     correlation_result=corr_result,
        ...     domain_knowledge={
        ...         "business_impact": "Signal strength directly affects user experience",
        ...         "optimization_opportunity": "Improve signal via antenna tilt"
        ...     }
        ... )
    """
    
    correlation_result: CorrelationResult = Field(..., description="Correlation analysis")
    domain_knowledge: Dict[str, str] = Field(default_factory=dict,
                                            description="Domain context and business impact")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "correlation_result": {},
            "domain_knowledge": {"business_impact": "Signal affects throughput"}
        }
    })


class LLMAnalysisResponse(BaseModel):
    """
    Response structure from LLM analysis.
    
    Standardized response from Llama 70B for all analysis types.
    
    Attributes:
        analysis_type: Type of analysis (Causal, Scenario, Interpretation)
        reasoning: Detailed reasoning from LLM
        recommendations: List of actionable recommendations
        confidence_level: LLM's confidence in recommendations (0.0-1.0)
        model_used: Model identifier (e.g., "llama2:70b")
        inference_timestamp: When inference was performed
        
    Example:
        >>> response = LLMAnalysisResponse(
        ...     analysis_type="Causal",
        ...     reasoning="The anomaly coincides with increased traffic...",
        ...     recommendations=[
        ...         "Check traffic volume for the same period",
        ...         "Verify carrier load balancing settings"
        ...     ],
        ...     confidence_level=0.85,
        ...     model_used="llama2:70b"
        ... )
    """
    
    analysis_type: Literal["Causal", "Scenario", "Interpretation"] = Field(...,
                                                                           description="Analysis type")
    reasoning: str = Field(..., description="Detailed reasoning from LLM")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence [0, 1]")
    model_used: str = Field(default="llama2:70b", description="LLM model identifier")
    inference_timestamp: datetime = Field(default_factory=datetime.utcnow,
                                         description="When inference was performed")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "analysis_type": "Causal",
            "reasoning": "The anomaly coincides with increased traffic patterns.",
            "recommendations": ["Check carrier load", "Verify settings"],
            "confidence_level": 0.85,
            "model_used": "llama2:70b"
        }
    })


# =============================================================================
# SUMMARY & DOCUMENTATION
# =============================================================================

"""
SCHEMA ORGANIZATION SUMMARY:

1. ENUMS (Controlled Vocabularies):
   - SeverityLevel: Low, Medium, High, Critical
   - ColumnType: Dimension_Text, Dimension_ID, KPI, Time
   - TimeFormat: Daily, Hourly, Monthly, Weekly
   - AggregationLevel: PLMN, Region, Carrier, Cell
   - AnomalyMethod: Z-Score, IQR, Isolation_Forest

2. CORE DATA MODELS:
   - ColumnClassification: Metadata for single column
   - DataFrameMetadata: Complete data ingestion output (→ All modules)

3. ANALYTICS RESULTS:
   - AnomalyResult: Single anomaly detection
   - CorrelationPair: Two-KPI correlation
   - CorrelationResult: Top-3 correlations for single KPI
   - ForecastValue: Single forecast point with CI
   - ForecastResult: Complete forecast horizon
   - FilteredDataFrameResult: Filtered data output

4. REQUEST/RESPONSE (API):
   - FilterRequest: User filter selections
   - AnomalyDetectionRequest: Anomaly detection parameters
   - ForecastRequest: Forecasting parameters

5. LLM SCHEMAS:
   - LLMCausalAnalysisRequest: "Why?" analysis
   - LLMScenarioPlanningRequest: "What if?" analysis
   - LLMCorrelationInterpretationRequest: "So what?" analysis
   - LLMAnalysisResponse: Standardized LLM response

USAGE FLOW:
   Data Ingestion
        ↓
   DataFrameMetadata (+ ColumnClassification list)
        ↓
   FilterRequest (user selection)
        ↓
   FilteredDataFrameResult
        ↓
   Analytics Modules:
   - AnomalyDetectionRequest → AnomalyResult[]
   - Correlation Analysis → CorrelationResult[]
   - ForecastRequest → ForecastResult
        ↓
   LLM Schemas: Feed results to Llama for reasoning
        ↓
   LLMAnalysisResponse (actionable insights)

VALIDATION RULES:
   - Correlation scores: -1.0 ≤ score ≤ 1.0
   - P-values: 0.0 ≤ p ≤ 1.0
   - Z-scores: unbounded (can be -10 to +10)
   - Confidence intervals: lower_ci < upper_ci
   - Row counts: must be > 0
   - Confidence level: 0.80 ≤ confidence ≤ 0.99
   - Forecast periods: 1 ≤ periods ≤ 365

SERIALIZATION:
   All models support:
   - .model_dump() → Python dict
   - .model_dump_json() → JSON string
   - .model_validate_json() → Parse JSON

EXAMPLE SERIALIZATION:
   >>> result = AnomalyResult(...)
   >>> json_str = result.model_dump_json()
   >>> result_restored = AnomalyResult.model_validate_json(json_str)
"""

if __name__ == "__main__":
    # Example usage and validation
    print("Data Models Module - Pydantic Schemas Initialized")
    print(f"Total Models: 20+")
    print(f"Enums: 5")
    print(f"Core Models: 2")
    print(f"Result Models: 6")
    print(f"Request Models: 3")
    print(f"LLM Schemas: 4")
