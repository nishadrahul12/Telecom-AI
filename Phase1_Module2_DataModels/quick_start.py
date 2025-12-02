#!/usr/bin/env python3
"""Quick Start: Data Models Usage Example"""

from data_models import (
    ColumnClassification, DataFrameMetadata, AnomalyResult,
    CorrelationPair, CorrelationResult, ForecastValue, ForecastResult,
    SeverityLevel, ColumnType, TimeFormat, AggregationLevel,
    AnomalyMethod
)
from datetime import datetime

# =========================================================================
# STEP 1: Create Column Metadata
# =========================================================================

time_col = ColumnClassification(
    column_name="Timestamp",
    column_type=ColumnType.TIME,
    data_type="datetime",
    non_null_count=10000,
    unique_count=10000,
    sample_values=["2024-01-01", "2024-01-02", "2024-01-03"],
    is_numeric=False
)

throughput_kpi = ColumnClassification(
    column_name="DL_Throughput",
    column_type=ColumnType.KPI,
    data_type="float",
    non_null_count=9950,
    unique_count=8234,
    sample_values=[2.45, 2.50, 2.48],
    is_numeric=True
)

signal_kpi = ColumnClassification(
    column_name="Signal_Strength",
    column_type=ColumnType.KPI,
    data_type="float",
    non_null_count=9950,
    unique_count=5000,
    sample_values=[85.5, 86.0, 84.5],
    is_numeric=True
)

region_dim = ColumnClassification(
    column_name="Region",
    column_type=ColumnType.DIMENSION_TEXT,
    data_type="str",
    non_null_count=10000,
    unique_count=5,
    sample_values=["North", "South", "East"],
    is_numeric=False
)

# =========================================================================
# STEP 2: Create DataFrame Metadata
# =========================================================================

metadata = DataFrameMetadata(
    file_path="/data/cell_level_data.csv",
    total_rows=10000,
    total_columns=4,
    time_format=TimeFormat.DAILY,
    aggregation_level=AggregationLevel.CELL,
    columns=[time_col, throughput_kpi, signal_kpi, region_dim],
    time_column="Timestamp",
    dimension_columns=["Region"],
    kpi_columns=["DL_Throughput", "Signal_Strength"],
    date_range_start="2024-01-01",
    date_range_end="2024-12-31",
    has_missing_values=False,
    sampling_applied=False,
    encoding="utf-8"
)

print(f"✓ Ingested {metadata.total_rows} rows from {metadata.file_path}")
print(f"✓ Time Format: {metadata.time_format.value}")
print(f"✓ Aggregation Level: {metadata.aggregation_level.value}")
print(f"✓ KPIs: {', '.join(metadata.kpi_columns)}")

# =========================================================================
# STEP 3: Create Anomaly Detection Result
# =========================================================================

anomaly = AnomalyResult(
    timestamp="2024-01-15 14:30:00",
    kpi_name="DL_Throughput",
    observed_value=0.5,
    expected_value=2.4,
    z_score=-3.8,
    deviation_percent=-79.2,
    severity=SeverityLevel.CRITICAL,
    method=AnomalyMethod.Z_SCORE,
    lower_bound=1.8,
    upper_bound=3.2,
    dimension_filters={"Region": "North"}
)

print(f"\n✓ Detected Anomaly:")
print(f"  - KPI: {anomaly.kpi_name}")
print(f"  - Timestamp: {anomaly.timestamp}")
print(f"  - Z-Score: {anomaly.z_score:.2f}σ")
print(f"  - Severity: {anomaly.severity.value}")

# =========================================================================
# STEP 4: Create Correlation Analysis Result
# =========================================================================

corr_pair = CorrelationPair(
    kpi_x="DL_Throughput",
    kpi_y="Signal_Strength",
    correlation_score=0.82,
    p_value=0.0001,
    is_significant=True,
    data_points_used=9850,
    interpretation="Strong positive correlation: Signal improves throughput"
)

corr_result = CorrelationResult(
    kpi_name="DL_Throughput",
    top_3_correlations=[corr_pair]
)

print(f"\n✓ Correlation Analysis:")
print(f"  - Reference KPI: {corr_result.kpi_name}")
print(f"  - Top Correlation: {corr_pair.kpi_y} (r={corr_pair.correlation_score:.3f})")
print(f"  - Significant: {'Yes' if corr_pair.is_significant else 'No'}")

# =========================================================================
# STEP 5: Create Forecast Result
# =========================================================================

forecast_values = [
    ForecastValue(
        timestamp="2024-02-01",
        predicted_value=2.45,
        lower_ci=2.10,
        upper_ci=2.80,
        confidence_level=0.95
    ),
    ForecastValue(
        timestamp="2024-02-02",
        predicted_value=2.48,
        lower_ci=2.08,
        upper_ci=2.88,
        confidence_level=0.95
    ),
    ForecastValue(
        timestamp="2024-02-03",
        predicted_value=2.50,
        lower_ci=2.05,
        upper_ci=2.95,
        confidence_level=0.95
    ),
]

forecast = ForecastResult(
    kpi_name="DL_Throughput",
    forecast_period="3 days",
    forecast_values=forecast_values,
    model_type="ARIMAX",
    rmse=0.12,
    exogenous_variables=["Signal_Strength"]
)

print(f"\n✓ Forecast Result:")
print(f"  - KPI: {forecast.kpi_name}")
print(f"  - Period: {forecast.forecast_period}")
print(f"  - Model: {forecast.model_type}")
print(f"  - Exogenous: {', '.join(forecast.exogenous_variables)}")
print(f"  - Forecast Points: {len(forecast.forecast_values)}")

# =========================================================================
# STEP 6: JSON Serialization (for API/LLM Integration)
# =========================================================================

print(f"\n✓ JSON Serialization:")

# Convert to JSON
anomaly_json = anomaly.model_dump_json(indent=2)
print(f"  Anomaly JSON: {len(anomaly_json)} chars")

# Parse back from JSON
anomaly_restored = AnomalyResult.model_validate_json(anomaly_json)
print(f"  ✓ Anomaly restored from JSON: z_score={anomaly_restored.z_score}")

forecast_json = forecast.model_dump_json(indent=2)
print(f"  Forecast JSON: {len(forecast_json)} chars")

print("\n" + "="*60)
print("✓ All models created and serialized successfully!")
print("="*60)
