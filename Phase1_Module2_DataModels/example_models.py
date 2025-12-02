#!/usr/bin/env python3
"""
Practical Examples: Understanding Each Model Type

This file demonstrates how to create and use each model type
in real-world scenarios.
"""

from data_models import (
    # Enums
    SeverityLevel, ColumnType, TimeFormat, AggregationLevel, AnomalyMethod,
    # Core Models
    ColumnClassification, DataFrameMetadata,
    # Analytics Results
    AnomalyResult, CorrelationPair, CorrelationResult,
    ForecastValue, ForecastResult, FilteredDataFrameResult,
    # Request/Response
    FilterRequest, AnomalyDetectionRequest, ForecastRequest,
    # LLM
    LLMCausalAnalysisRequest, LLMAnalysisResponse
)
from datetime import datetime

print("="*70)
print("PART 1: ENUMS - Controlled Vocabularies")
print("="*70)

# Enums ensure you use only valid values
print("\n1. SeverityLevel (Anomaly importance)")
print("   Low, Medium, High, Critical")
severity = SeverityLevel.CRITICAL
print(f"   Example: {severity.value}")

print("\n2. ColumnType (What each column represents)")
print("   DIMENSION_TEXT (Region), DIMENSION_ID (Cell_ID), KPI (Throughput), TIME")
col_type = ColumnType.KPI
print(f"   Example: {col_type.value}")

print("\n3. TimeFormat (How dates are stored)")
print("   Daily (MM/DD/YYYY), Hourly (YYYY-MM-DD HH), Monthly, Weekly")
time_fmt = TimeFormat.DAILY
print(f"   Example: {time_fmt.value}")

print("\n4. AggregationLevel (Data hierarchy)")
print("   PLMN (Network) > Region > Carrier (Band) > Cell (Site)")
agg_level = AggregationLevel.CELL
print(f"   Example: {agg_level.value}")

print("\n5. AnomalyMethod (How anomalies are detected)")
print("   Z-Score (statistical), IQR (range), Isolation_Forest (ML)")
method = AnomalyMethod.Z_SCORE
print(f"   Example: {method.value}")

print("\n" + "="*70)
print("PART 2: CORE MODELS - Data Metadata")
print("="*70)

# Every CSV file becomes metadata
print("\n1. ColumnClassification - Describe ONE column")
print("   Purpose: Tell system about each column in CSV")

col1 = ColumnClassification(
    column_name="Timestamp",
    column_type=ColumnType.TIME,
    data_type="datetime",
    non_null_count=10000,
    unique_count=10000,
    sample_values=["2024-01-01", "2024-01-02", "2024-01-03"],
    is_numeric=False
)

print(f"   Column: {col1.column_name}")
print(f"   Type: {col1.column_type.value}")
print(f"   Data Type: {col1.data_type}")
print(f"   Non-null: {col1.non_null_count}/{col1.non_null_count + 50} = 95%")

# Another column
col2 = ColumnClassification(
    column_name="DL_Throughput",
    column_type=ColumnType.KPI,
    data_type="float",
    non_null_count=9950,
    unique_count=8234,
    sample_values=[2.45, 2.50, 2.48],
    is_numeric=True
)

col3 = ColumnClassification(
    column_name="Region",
    column_type=ColumnType.DIMENSION_TEXT,
    data_type="str",
    non_null_count=10000,
    unique_count=5,
    sample_values=["North", "South", "East"],
    is_numeric=False
)

print(f"\n   Column: {col2.column_name} (KPI)")
print(f"   Column: {col3.column_name} (Dimension)")

# Now describe the entire file
print("\n2. DataFrameMetadata - Describe ENTIRE file")
print("   Purpose: Complete metadata about ingested CSV file")

metadata = DataFrameMetadata(
    file_path="/data/telecom_cell_level.csv",
    total_rows=10000,
    total_columns=3,
    time_format=TimeFormat.DAILY,
    aggregation_level=AggregationLevel.CELL,
    columns=[col1, col2, col3],
    time_column="Timestamp",
    dimension_columns=["Region"],
    kpi_columns=["DL_Throughput"],
    date_range_start="2024-01-01",
    date_range_end="2024-01-31",
    has_missing_values=False,
    sampling_applied=False
)

print(f"   File: {metadata.file_path}")
print(f"   Rows: {metadata.total_rows}")
print(f"   Time: {metadata.time_format.value} format")
print(f"   Level: {metadata.aggregation_level.value}")
print(f"   Columns: {metadata.total_columns} ({', '.join(metadata.kpi_columns)} KPIs)")

print("\n" + "="*70)
print("PART 3: ANALYTICS RESULTS - What modules produce")
print("="*70)

# Anomaly Detection produces AnomalyResult
print("\n1. AnomalyResult - Single detected anomaly")
print("   From: Anomaly Detection Module")
print("   Trigger: Z-score > 3σ (3 standard deviations)")

anomaly = AnomalyResult(
    timestamp="2024-01-15 14:30:00",
    kpi_name="DL_Throughput",
    observed_value=0.5,           # What we actually saw
    expected_value=2.4,           # What we expected
    z_score=-3.8,                 # How many σ from mean
    deviation_percent=-79.2,      # Percentage drop
    severity=SeverityLevel.CRITICAL,
    method=AnomalyMethod.Z_SCORE,
    lower_bound=1.8,              # Normal range
    upper_bound=3.2,
    dimension_filters={"Region": "North", "Carrier": "L2100"}
)

print(f"   ✗ ALERT: {anomaly.kpi_name}")
print(f"   Observed: {anomaly.observed_value} (Expected: {anomaly.expected_value})")
print(f"   Z-Score: {anomaly.z_score:.1f}σ (CRITICAL!)")
print(f"   Drop: {anomaly.deviation_percent:.1f}%")
print(f"   Location: {anomaly.dimension_filters}")

# Correlation Analysis produces CorrelationPair and CorrelationResult
print("\n2. CorrelationPair - Correlation between 2 KPIs")
print("   From: Correlation Analysis Module")
print("   Meaning: How much do two KPIs change together?")

corr_pair = CorrelationPair(
    kpi_x="DL_Throughput",
    kpi_y="Signal_Strength",
    correlation_score=0.82,           # Range: -1 to +1
    p_value=0.0001,                   # Statistical significance
    is_significant=True,              # p < 0.05?
    data_points_used=9850,
    interpretation="Strong positive: Better signal → Higher throughput"
)

print(f"   {corr_pair.kpi_x} ↔ {corr_pair.kpi_y}")
print(f"   Correlation: r = {corr_pair.correlation_score:.2f} (Strong)")
print(f"   P-value: {corr_pair.p_value:.4f} (Significant: {corr_pair.is_significant})")
print(f"   Meaning: {corr_pair.interpretation}")

# But we only show Top-3 per KPI
print("\n3. CorrelationResult - Top 3 correlations for ONE KPI")
print("   From: Correlation Analysis Module")

corr_result = CorrelationResult(
    kpi_name="DL_Throughput",
    top_3_correlations=[corr_pair]  # Usually 3, but showing 1 here
)

print(f"   KPI: {corr_result.kpi_name}")
print(f"   Top correlations: {len(corr_result.top_3_correlations)}")
for i, corr in enumerate(corr_result.top_3_correlations, 1):
    print(f"      {i}. {corr.kpi_y}: r={corr.correlation_score:.3f}")

# Forecasting produces ForecastValue and ForecastResult
print("\n4. ForecastValue - ONE forecast point")
print("   From: Forecasting Module (ARIMA/ARIMAX)")

fv = ForecastValue(
    timestamp="2024-02-01",
    predicted_value=2.45,         # Point estimate
    lower_ci=2.10,                # 95% confidence interval
    upper_ci=2.80,
    confidence_level=0.95
)

print(f"   Date: {fv.timestamp}")
print(f"   Forecast: {fv.predicted_value:.2f}")
print(f"   95% Range: [{fv.lower_ci:.2f}, {fv.upper_ci:.2f}]")
print(f"   Uncertainty: ±{(fv.upper_ci - fv.lower_ci) / 2:.2f}")

print("\n5. ForecastResult - MULTIPLE forecast points")
print("   From: Forecasting Module")

fv2 = ForecastValue(timestamp="2024-02-02", predicted_value=2.48, lower_ci=2.08, upper_ci=2.88)
fv3 = ForecastValue(timestamp="2024-02-03", predicted_value=2.50, lower_ci=2.05, upper_ci=2.95)

forecast = ForecastResult(
    kpi_name="DL_Throughput",
    forecast_period="3 days",
    forecast_values=[fv, fv2, fv3],
    model_type="ARIMAX",
    rmse=0.12,
    exogenous_variables=["Signal_Strength"]
)

print(f"   KPI: {forecast.kpi_name}")
print(f"   Period: {forecast.forecast_period}")
print(f"   Model: {forecast.model_type} (with {len(forecast.exogenous_variables)} exogenous variables)")
print(f"   Accuracy (RMSE): {forecast.rmse:.2f}")
for fv_point in forecast.forecast_values:
    print(f"      {fv_point.timestamp}: {fv_point.predicted_value:.2f}")

# Filtering produces FilteredDataFrameResult
print("\n6. FilteredDataFrameResult - Filtered subset of data")
print("   From: Data Filtering Module")
print("   Purpose: Pass filtered data to analytics modules")

filtered = FilteredDataFrameResult(
    original_metadata=metadata,
    filter_selections={"Region": "North"},
    filtered_row_count=2000,
    dimension_values_applied={"Region": "North"}
)

print(f"   Original: {filtered.original_metadata.total_rows} rows")
print(f"   Filters: {filtered.filter_selections}")
print(f"   Result: {filtered.filtered_row_count} rows")
print(f"   Reduction: {100 * (1 - filtered.filtered_row_count / filtered.original_metadata.total_rows):.1f}%")

print("\n" + "="*70)
print("PART 4: REQUEST/RESPONSE MODELS - API Contracts")
print("="*70)

print("\n1. FilterRequest - User tells system what data they want")
print("   From: Streamlit Frontend")
print("   To: Backend Filtering Module")

filter_req = FilterRequest(
    region="North",
    carrier="L2100",
    start_date="2024-01-01",
    end_date="2024-01-31",
    kpi_names=["DL_Throughput", "Signal_Strength"]
)

print(f"   Region: {filter_req.region}")
print(f"   Carrier: {filter_req.carrier}")
print(f"   Date Range: {filter_req.start_date} to {filter_req.end_date}")
print(f"   KPIs: {', '.join(filter_req.kpi_names)}")

print("\n2. AnomalyDetectionRequest - Frontend requests anomalies")
print("   From: Streamlit Frontend")
print("   To: Anomaly Detection Module")

anomaly_req = AnomalyDetectionRequest(
    filtered_data_result=filtered,
    method=AnomalyMethod.Z_SCORE,
    z_score_threshold=3.0,
    kpi_names=["DL_Throughput"]
)

print(f"   Data: {anomaly_req.filtered_data_result.filtered_row_count} rows")
print(f"   Method: {anomaly_req.method.value}")
print(f"   Threshold: {anomaly_req.z_score_threshold}σ")
print(f"   KPIs: {', '.join(anomaly_req.kpi_names)}")

print("\n3. ForecastRequest - Frontend requests forecast")
print("   From: Streamlit Frontend")
print("   To: Forecasting Module")

forecast_req = ForecastRequest(
    filtered_data_result=filtered,
    kpi_name="DL_Throughput",
    forecast_periods=30,
    model_type="ARIMAX",
    exogenous_kpi_names=["Signal_Strength"]
)

print(f"   KPI: {forecast_req.kpi_name}")
print(f"   Forecast Days: {forecast_req.forecast_periods}")
print(f"   Model: {forecast_req.model_type}")
print(f"   Exogenous: {', '.join(forecast_req.exogenous_kpi_names)}")

print("\n" + "="*70)
print("PART 5: LLM SCHEMAS - Domain Reasoning")
print("="*70)

print("\n1. LLMCausalAnalysisRequest - 'Why did this happen?'")
print("   From: Analytics Module → LLM")
print("   Question: What caused this anomaly?")

llm_causal_req = LLMCausalAnalysisRequest(
    anomaly=anomaly,
    historical_context={
        "mean_recent": 2.4,
        "std_dev": 0.3,
        "trend": "stable",
        "recent_events": ["Network upgrade on Jan 14"]
    },
    correlated_kpis=[corr_pair],
    domain_context={
        "region": "North",
        "carrier": "L2100",
        "cell_type": "Macro Cell",
        "weather": "Clear"
    }
)

print(f"   Anomaly: {llm_causal_req.anomaly.kpi_name} dropped to {llm_causal_req.anomaly.observed_value}")
print(f"   Historical mean: {llm_causal_req.historical_context['mean_recent']}")
print(f"   Recent events: {llm_causal_req.historical_context['recent_events']}")
print(f"   Context: {llm_causal_req.domain_context['region']}, {llm_causal_req.domain_context['carrier']}")

print("\n2. LLMAnalysisResponse - LLM's answer")
print("   From: LLM (Llama 70B)")
print("   To: Streamlit Dashboard")

llm_response = LLMAnalysisResponse(
    analysis_type="Causal",
    reasoning="The throughput drop coincides with network upgrade on Jan 14. "
              "Signal strength also dropped, suggesting antenna misconfiguration during upgrade.",
    recommendations=[
        "Check antenna tilt settings on macro cell",
        "Verify RAN parameter settings post-upgrade",
        "Compare signal maps with baseline"
    ],
    confidence_level=0.87,
    model_used="llama2:70b"
)

print(f"   Analysis: {llm_response.analysis_type}")
print(f"   Confidence: {llm_response.confidence_level:.0%}")
print(f"   Reasoning: {llm_response.reasoning[:80]}...")
print(f"   Recommendations:")
for i, rec in enumerate(llm_response.recommendations, 1):
    print(f"      {i}. {rec}")

print("\n" + "="*70)
print("PART 6: JSON SERIALIZATION - For APIs")
print("="*70)

print("\n1. Convert Model → JSON (for API response)")
json_str = anomaly.model_dump_json(indent=2)
print(f"   Anomaly as JSON ({len(json_str)} chars):")
print(json_str[:200] + "...")

print("\n2. Convert JSON → Model (for API request)")
restored = AnomalyResult.model_validate_json(json_str)
print(f"   ✓ Restored from JSON: {restored.kpi_name} with z_score={restored.z_score}")

print("\n" + "="*70)
print("✓ ALL MODEL TYPES DEMONSTRATED")
print("="*70)
