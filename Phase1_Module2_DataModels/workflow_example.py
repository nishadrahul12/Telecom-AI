#!/usr/bin/env python3
"""
Real-World Workflow: From CSV to LLM Insights

This shows the complete flow:
CSV → Metadata → Filtering → Analytics → LLM → Insights
"""

from data_models import *

print("\n" + "="*70)
print("REAL-WORLD WORKFLOW: CSV to LLM Insights")
print("="*70)

# STEP 1: Data Ingestion (from CSV file)
print("\n[STEP 1] Data Ingestion: Load CSV file")
print("         Module 1 reads CSV and produces metadata")

col_time = ColumnClassification(
    column_name="Timestamp",
    column_type=ColumnType.TIME,
    data_type="datetime",
    non_null_count=8760,
    unique_count=8760,
    sample_values=["2024-01-01", "2024-01-02", "2024-01-03"],
    is_numeric=False
)

col_dl = ColumnClassification(
    column_name="DL_Throughput",
    column_type=ColumnType.KPI,
    data_type="float",
    non_null_count=8700,
    unique_count=7500,
    sample_values=[2.45, 2.50, 2.48],
    is_numeric=True
)

col_signal = ColumnClassification(
    column_name="Signal_Strength",
    column_type=ColumnType.KPI,
    data_type="float",
    non_null_count=8700,
    unique_count=6000,
    sample_values=[85.5, 86.0, 84.5],
    is_numeric=True
)

col_region = ColumnClassification(
    column_name="Region",
    column_type=ColumnType.DIMENSION_TEXT,
    data_type="str",
    non_null_count=8760,
    unique_count=5,
    sample_values=["North", "South", "East"],
    is_numeric=False
)

col_cell = ColumnClassification(
    column_name="Cell_ID",
    column_type=ColumnType.DIMENSION_ID,
    data_type="int",
    non_null_count=8760,
    unique_count=100,
    sample_values=[101, 102, 103],
    is_numeric=True
)

# Module 1 output: DataFrameMetadata
raw_metadata = DataFrameMetadata(
    file_path="/data/raw/cell_level_hourly.csv",
    total_rows=8760,
    total_columns=5,
    time_format=TimeFormat.HOURLY,
    aggregation_level=AggregationLevel.CELL,
    columns=[col_time, col_dl, col_signal, col_region, col_cell],
    time_column="Timestamp",
    dimension_columns=["Region", "Cell_ID"],
    kpi_columns=["DL_Throughput", "Signal_Strength"],
    date_range_start="2024-01-01 00:00:00",
    date_range_end="2024-12-31 23:00:00",
    has_missing_values=False,
    sampling_applied=False,
    encoding="utf-8"
)

print(f"✓ Ingested: {raw_metadata.file_path}")
print(f"  - {raw_metadata.total_rows} hourly records")
print(f"  - {raw_metadata.total_columns} columns ({len(raw_metadata.kpi_columns)} KPIs)")
print(f"  - Period: {raw_metadata.date_range_start} to {raw_metadata.date_range_end}")

# STEP 2: User Requests Filtered Data
print("\n[STEP 2] User Request: Filter for specific region")
print("         Frontend sends FilterRequest to backend")

user_filter = FilterRequest(
    region="North",
    start_date="2024-01-15",
    end_date="2024-01-20",
    kpi_names=["DL_Throughput", "Signal_Strength"]
)

print(f"✓ User filter:")
print(f"  - Region: {user_filter.region}")
print(f"  - Dates: {user_filter.start_date} to {user_filter.end_date}")
print(f"  - KPIs: {', '.join(user_filter.kpi_names)}")

# STEP 3: Module 3 Filters Data
print("\n[STEP 3] Data Filtering: Apply filters")
print("         Module 3 applies filters")

filtered_metadata = FilteredDataFrameResult(
    original_metadata=raw_metadata,
    filter_selections={
        "Region": user_filter.region,
        "StartDate": user_filter.start_date,
        "EndDate": user_filter.end_date
    },
    filtered_row_count=144,  # 6 days * 24 hours
    dimension_values_applied={"Region": "North"}
)

print(f"✓ Filtered data:")
print(f"  - Original: {filtered_metadata.original_metadata.total_rows} rows")
print(f"  - Filtered: {filtered_metadata.filtered_row_count} rows")
print(f"  - Applied: {filtered_metadata.dimension_values_applied}")

# STEP 4: Anomaly Detection
print("\n[STEP 4] Anomaly Detection: Find unusual values")
print("         Module 2 detects anomalies")

anomaly_request = AnomalyDetectionRequest(
    filtered_data_result=filtered_metadata,
    method=AnomalyMethod.Z_SCORE,
    z_score_threshold=3.0,
    kpi_names=["DL_Throughput"]
)

print(f"✓ Anomaly detection request:")
print(f"  - Method: {anomaly_request.method.value}")
print(f"  - Threshold: {anomaly_request.z_score_threshold}σ")
print(f"  - KPIs: {', '.join(anomaly_request.kpi_names)}")

# Simulated anomaly result
detected_anomaly = AnomalyResult(
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
    dimension_filters={"Region": "North", "Cell_ID": "101"}
)

print(f"\n✗ ALERT: Anomaly detected!")
print(f"  - {detected_anomaly.kpi_name} dropped to {detected_anomaly.observed_value}")
print(f"  - Expected: {detected_anomaly.expected_value}")
print(f"  - Z-Score: {detected_anomaly.z_score:.1f}σ")
print(f"  - Severity: {detected_anomaly.severity.value}")

# STEP 5: Correlation Analysis
print("\n[STEP 5] Correlation Analysis: Find related KPIs")
print("         Module 2 analyzes correlations")

signal_corr = CorrelationPair(
    kpi_x="DL_Throughput",
    kpi_y="Signal_Strength",
    correlation_score=0.82,
    p_value=0.0001,
    is_significant=True,
    data_points_used=140,
    interpretation="Strong positive: Signal is key factor"
)

corr_result = CorrelationResult(
    kpi_name="DL_Throughput",
    top_3_correlations=[signal_corr]
)

print(f"✓ Correlation analysis:")
print(f"  - KPI: {corr_result.kpi_name}")
print(f"  - Top correlation: {signal_corr.kpi_y} (r={signal_corr.correlation_score:.2f})")

# STEP 6: Send to LLM for Analysis
print("\n[STEP 6] LLM Analysis: Get domain insights")
print("         Send to Llama 70B via Ollama")

llm_request = LLMCausalAnalysisRequest(
    anomaly=detected_anomaly,
    historical_context={
        "mean_recent": 2.4,
        "std_dev": 0.3,
        "trend": "stable before anomaly",
        "recent_events": ["Maintenance window Jan 14 18:00-20:00"]
    },
    correlated_kpis=[signal_corr],
    domain_context={
        "region": "North",
        "cell_id": "101",
        "cell_type": "Macro Cell",
        "sector_type": "3-sector",
        "recent_weather": "Clear",
        "recent_traffic": "Normal"
    }
)

print(f"✓ LLM request sent:")
print(f"  - Anomaly: {detected_anomaly.kpi_name}")
print(f"  - Context: {llm_request.domain_context['region']}, {llm_request.domain_context['cell_id']}")
print(f"  - Recent event: {llm_request.historical_context['recent_events']}")

# STEP 7: LLM Response
print("\n[STEP 7] LLM Response: Get actionable insights")
print("         Llama 70B provides analysis")

llm_response = LLMAnalysisResponse(
    analysis_type="Causal",
    reasoning="""The throughput drop coincides with maintenance window on Jan 14.
    Signal strength also dropped from 86dBm to 75dBm in the same period.
    This suggests antenna misconfiguration during maintenance or power control issue.
    The strong correlation (r=0.82) confirms signal is the limiting factor.""",
    recommendations=[
        "Check antenna tilt settings on Cell 101 sector 1",
        "Verify power control parameters are within expected range",
        "Compare pre/post-maintenance RAN configuration",
        "Review O&M logs for maintenance activities",
        "Consider rollback if recent changes caused degradation"
    ],
    confidence_level=0.88,
    model_used="llama2:70b"
)

print(f"✓ LLM Analysis Complete:")
print(f"  - Type: {llm_response.analysis_type}")
print(f"  - Confidence: {llm_response.confidence_level:.0%}")
print(f"  - Recommendations: {len(llm_response.recommendations)}")
for i, rec in enumerate(llm_response.recommendations, 1):
    print(f"    {i}. {rec}")

# STEP 8: Forecasting
print("\n[STEP 8] Forecasting: Predict future values")
print("         Module 2 forecasts next 7 days")

forecast_request = ForecastRequest(
    filtered_data_result=filtered_metadata,
    kpi_name="DL_Throughput",
    forecast_periods=7,
    model_type="ARIMAX",
    exogenous_kpi_names=["Signal_Strength"]
)

# Simulated forecast
fv1 = ForecastValue(timestamp="2024-01-21", predicted_value=2.45, lower_ci=2.10, upper_ci=2.80)
fv2 = ForecastValue(timestamp="2024-01-22", predicted_value=2.48, lower_ci=2.08, upper_ci=2.88)
fv3 = ForecastValue(timestamp="2024-01-23", predicted_value=2.50, lower_ci=2.05, upper_ci=2.95)

forecast = ForecastResult(
    kpi_name="DL_Throughput",
    forecast_period="7 days",
    forecast_values=[fv1, fv2, fv3],
    model_type="ARIMAX",
    rmse=0.12,
    exogenous_variables=["Signal_Strength"]
)

print(f"✓ Forecast generated:")
print(f"  - KPI: {forecast.kpi_name}")
print(f"  - Model: {forecast.model_type}")
print(f"  - Forecast points: {len(forecast.forecast_values)}")
print(f"  - Accuracy (RMSE): {forecast.rmse:.2f}")

print("\n" + "="*70)
print("✓ COMPLETE WORKFLOW DEMONSTRATED")
print("="*70)
print("\nFlow Summary:")
print("1. CSV → DataFrameMetadata (Module 1)")
print("2. FilterRequest → FilteredDataFrameResult (Module 3)")
print("3. Analytics → AnomalyResult, CorrelationResult, ForecastResult")
print("4. Results → LLMCausalAnalysisRequest → LLMAnalysisResponse")
print("5. All results → Streamlit Dashboard (Module 4)")
