#!/usr/bin/env python3
"""
Test: Verify each model type works correctly
"""

from data_models import *
from pydantic import ValidationError

print("\n" + "="*70)
print("TESTING EACH MODEL TYPE")
print("="*70)

# Test 1: Enums
print("\n[TEST 1] Enums - Valid values only")
try:
    severity = SeverityLevel.CRITICAL
    print(f"✓ Enum validation passed: {severity.value}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 2: ColumnClassification
print("\n[TEST 2] ColumnClassification - Column metadata")
try:
    col = ColumnClassification(
        column_name="DL_Throughput",
        column_type=ColumnType.KPI,
        data_type="float",
        non_null_count=9950,
        unique_count=8234,
        sample_values=[2.45, 2.50, 2.48],
        is_numeric=True
    )
    print(f"✓ ColumnClassification created: {col.column_name}")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 3: DataFrameMetadata
print("\n[TEST 3] DataFrameMetadata - Complete file metadata")
try:
    metadata = DataFrameMetadata(
        file_path="/data/test.csv",
        total_rows=10000,
        total_columns=1,
        time_format=TimeFormat.DAILY,
        aggregation_level=AggregationLevel.CELL,
        columns=[col],
        time_column="Timestamp",
        kpi_columns=["DL_Throughput"]
    )
    print(f"✓ DataFrameMetadata created: {metadata.total_rows} rows")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 4: AnomalyResult
print("\n[TEST 4] AnomalyResult - Detected anomaly")
try:
    anomaly = AnomalyResult(
        timestamp="2024-01-15 14:30:00",
        kpi_name="DL_Throughput",
        observed_value=0.5,
        z_score=-3.8,
        severity=SeverityLevel.CRITICAL,
        method=AnomalyMethod.Z_SCORE
    )
    print(f"✓ AnomalyResult created: {anomaly.kpi_name} (z={anomaly.z_score})")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 5: CorrelationPair and CorrelationResult
print("\n[TEST 5] CorrelationPair - Two-KPI correlation")
try:
    corr = CorrelationPair(
        kpi_x="DL_Throughput",
        kpi_y="Signal_Strength",
        correlation_score=0.82,
        p_value=0.0001,
        is_significant=True,
        data_points_used=9850
    )
    print(f"✓ CorrelationPair created: r={corr.correlation_score}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n[TEST 6] CorrelationResult - Top 3 per KPI")
try:
    corr_result = CorrelationResult(
        kpi_name="DL_Throughput",
        top_3_correlations=[corr]
    )
    print(f"✓ CorrelationResult created: {len(corr_result.top_3_correlations)} correlations")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 6: ForecastValue and ForecastResult
print("\n[TEST 7] ForecastValue - Single forecast point")
try:
    fv = ForecastValue(
        timestamp="2024-02-01",
        predicted_value=2.45,
        lower_ci=2.10,
        upper_ci=2.80
    )
    print(f"✓ ForecastValue created: {fv.predicted_value}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n[TEST 8] ForecastResult - Multiple forecast points")
try:
    forecast = ForecastResult(
        kpi_name="DL_Throughput",
        forecast_period="30 days",
        forecast_values=[fv],
        model_type="ARIMA"
    )
    print(f"✓ ForecastResult created: {len(forecast.forecast_values)} points")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 7: Request Models
print("\n[TEST 9] FilterRequest - User filter selections")
try:
    filter_req = FilterRequest(
        region="North",
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    print(f"✓ FilterRequest created: {filter_req.region}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n[TEST 10] AnomalyDetectionRequest")
try:
    filtered = FilteredDataFrameResult(
        original_metadata=metadata,
        filter_selections={},
        filtered_row_count=10000,
        dimension_values_applied={}
    )
    anomaly_req = AnomalyDetectionRequest(
        filtered_data_result=filtered,
        method=AnomalyMethod.Z_SCORE
    )
    print(f"✓ AnomalyDetectionRequest created")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 8: Validation
print("\n[TEST 11] Validation - Reject invalid data")
try:
    bad_corr = CorrelationPair(
        kpi_x="A",
        kpi_y="B",
        correlation_score=1.5,  # INVALID!
        p_value=0.05,
        is_significant=False,
        data_points_used=100
    )
    print(f"✗ Validation FAILED - should have rejected correlation_score=1.5")
except ValidationError as e:
    print(f"✓ Validation correctly rejected: correlation_score must be -1 to 1")

# Test 9: JSON Serialization
print("\n[TEST 12] JSON Serialization - Convert to/from JSON")
try:
    json_str = anomaly.model_dump_json()
    restored = AnomalyResult.model_validate_json(json_str)
    assert restored.z_score == anomaly.z_score
    print(f"✓ JSON serialization works: {len(json_str)} chars")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test 10: LLM Models
print("\n[TEST 13] LLMAnalysisResponse - LLM output")
try:
    llm_response = LLMAnalysisResponse(
        analysis_type="Causal",
        reasoning="Test reasoning",
        recommendations=["Rec1", "Rec2"],
        confidence_level=0.85
    )
    print(f"✓ LLMAnalysisResponse created: {llm_response.analysis_type}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "="*70)
print("✓ ALL MODEL TESTS COMPLETED")
print("="*70)
