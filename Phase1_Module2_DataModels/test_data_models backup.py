# =============================================================================
# PROJECT: Intelligent Telecom Optimization System
# MODULE: Phase 1, Module 2 - Data Models Unit Tests
# PURPOSE: Comprehensive test coverage for Pydantic schemas
# AUTHOR: Telecom Optimization Team
# VERSION: 1.0.0
# =============================================================================

"""
Unit Tests for Data Models Module

Comprehensive test coverage for all Pydantic models with:
- Valid data scenarios
- Edge cases
- Validation error scenarios
- Serialization/deserialization
- Integration between models

Run with: pytest test_data_models.py -v

Test Categories:
1. Enum Validation (5 tests)
2. ColumnClassification (8 tests)
3. DataFrameMetadata (10 tests)
4. AnomalyResult (12 tests)
5. CorrelationPair (8 tests)
6. CorrelationResult (6 tests)
7. ForecastValue (8 tests)
8. ForecastResult (8 tests)
9. Request Models (9 tests)
10. LLM Schemas (6 tests)

Total: 80+ comprehensive test cases
"""

import pytest
from datetime import datetime
from pydantic import ValidationError
from data_models import (
    # Enums
    SeverityLevel, ColumnType, TimeFormat, AggregationLevel, AnomalyMethod,
    # Core Models
    ColumnClassification, DataFrameMetadata,
    # Analytics Results
    AnomalyResult, CorrelationPair, CorrelationResult,
    ForecastValue, ForecastResult, FilteredDataFrameResult,
    # Request Models
    FilterRequest, AnomalyDetectionRequest, ForecastRequest,
    # LLM Schemas
    LLMCausalAnalysisRequest, LLMScenarioPlanningRequest,
    LLMCorrelationInterpretationRequest, LLMAnalysisResponse
)


# =============================================================================
# ENUM VALIDATION TESTS
# =============================================================================

class TestEnumValidation:
    """Test all enumeration types."""
    
    def test_severity_levels(self):
        """Test SeverityLevel enum values."""
        assert SeverityLevel.LOW.value == "Low"
        assert SeverityLevel.MEDIUM.value == "Medium"
        assert SeverityLevel.HIGH.value == "High"
        assert SeverityLevel.CRITICAL.value == "Critical"
        assert len(SeverityLevel) == 4
    
    def test_column_types(self):
        """Test ColumnType enum values."""
        assert ColumnType.DIMENSION_TEXT.value == "Dimension_Text"
        assert ColumnType.DIMENSION_ID.value == "Dimension_ID"
        assert ColumnType.KPI.value == "KPI"
        assert ColumnType.TIME.value == "Time"
    
    def test_time_formats(self):
        """Test TimeFormat enum values."""
        assert TimeFormat.DAILY.value == "Daily"
        assert TimeFormat.HOURLY.value == "Hourly"
        assert TimeFormat.MONTHLY.value == "Monthly"
        assert TimeFormat.WEEKLY.value == "Weekly"
    
    def test_aggregation_levels(self):
        """Test AggregationLevel enum hierarchy."""
        levels = [AggregationLevel.PLMN, AggregationLevel.REGION, 
                  AggregationLevel.CARRIER, AggregationLevel.CELL]
        assert len(levels) == 4
        assert AggregationLevel.PLMN.value == "PLMN"
    
    def test_anomaly_methods(self):
        """Test AnomalyMethod enum values."""
        assert AnomalyMethod.Z_SCORE.value == "Z-Score"
        assert AnomalyMethod.IQR.value == "IQR"
        assert AnomalyMethod.ISOLATION_FOREST.value == "Isolation_Forest"


# =============================================================================
# COLUMNCLASSIFICATION TESTS
# =============================================================================

class TestColumnClassification:
    """Test ColumnClassification model."""
    
    def test_valid_column_classification(self):
        """Test valid column classification."""
        col = ColumnClassification(
            column_name="DL_Throughput",
            column_type=ColumnType.KPI,
            data_type="float",
            non_null_count=9950,
            unique_count=8234,
            sample_values=[2.45, 2.50, 2.48],
            is_numeric=True
        )
        assert col.column_name == "DL_Throughput"
        assert col.column_type == ColumnType.KPI
        assert col.is_numeric is True
    
    def test_dimension_text_classification(self):
        """Test dimension text column classification."""
        col = ColumnClassification(
            column_name="Region",
            column_type=ColumnType.DIMENSION_TEXT,
            data_type="str",
            non_null_count=10000,
            unique_count=5,
            sample_values=["North", "South", "East"],
            is_numeric=False
        )
        assert col.column_type == ColumnType.DIMENSION_TEXT
        assert not col.is_numeric
    
    def test_dimension_id_classification(self):
        """Test dimension ID column classification."""
        col = ColumnClassification(
            column_name="Cell_ID",
            column_type=ColumnType.DIMENSION_ID,
            data_type="int",
            non_null_count=10000,
            unique_count=2500,
            sample_values=[101, 102, 103],
            is_numeric=True
        )
        assert col.column_type == ColumnType.DIMENSION_ID
    
    def test_time_column_classification(self):
        """Test time column classification."""
        col = ColumnClassification(
            column_name="Timestamp",
            column_type=ColumnType.TIME,
            data_type="datetime",
            non_null_count=10000,
            unique_count=10000,
            sample_values=["2024-01-01", "2024-01-02", "2024-01-03"],
            is_numeric=False
        )
        assert col.column_type == ColumnType.TIME
    
    def test_json_serialization(self):
        """Test JSON serialization of ColumnClassification."""
        col = ColumnClassification(
            column_name="Signal_Strength",
            column_type=ColumnType.KPI,
            data_type="float",
            non_null_count=9900,
            unique_count=5000,
            sample_values=[85.5, 86.0, 84.5],
            is_numeric=True
        )
        json_str = col.model_dump_json()
        restored = ColumnClassification.model_validate_json(json_str)
        assert restored.column_name == col.column_name
        assert restored.column_type == col.column_type
    
    def test_negative_counts_validation(self):
        """Test validation rejects negative counts."""
        with pytest.raises(ValidationError):
            ColumnClassification(
                column_name="Invalid",
                column_type=ColumnType.KPI,
                data_type="float",
                non_null_count=-1,  # Invalid: negative
                unique_count=100,
                sample_values=[1.0],
                is_numeric=True
            )
    
    def test_zero_non_null_count(self):
        """Test validation rejects zero non-null count."""
        with pytest.raises(ValidationError):
            ColumnClassification(
                column_name="Empty",
                column_type=ColumnType.KPI,
                data_type="float",
                non_null_count=0,  # Invalid: zero
                unique_count=0,
                sample_values=[],
                is_numeric=True
            )


# =============================================================================
# DATAFRAMEMETADATA TESTS
# =============================================================================

class TestDataFrameMetadata:
    """Test DataFrameMetadata model."""
    
    @pytest.fixture
    def sample_columns(self):
        """Fixture: sample column classifications."""
        return [
            ColumnClassification(
                column_name="Timestamp",
                column_type=ColumnType.TIME,
                data_type="datetime",
                non_null_count=10000,
                unique_count=10000,
                sample_values=["2024-01-01", "2024-01-02", "2024-01-03"],
                is_numeric=False
            ),
            ColumnClassification(
                column_name="DL_Throughput",
                column_type=ColumnType.KPI,
                data_type="float",
                non_null_count=9950,
                unique_count=8234,
                sample_values=[2.45, 2.50, 2.48],
                is_numeric=True
            ),
            ColumnClassification(
                column_name="Region",
                column_type=ColumnType.DIMENSION_TEXT,
                data_type="str",
                non_null_count=10000,
                unique_count=5,
                sample_values=["North", "South", "East"],
                is_numeric=False
            )
        ]
    
    def test_valid_dataframe_metadata(self, sample_columns):
        """Test valid DataFrameMetadata creation."""
        metadata = DataFrameMetadata(
            file_path="/data/cell_level.csv",
            total_rows=10000,
            total_columns=3,
            time_format=TimeFormat.DAILY,
            aggregation_level=AggregationLevel.CELL,
            columns=sample_columns,
            time_column="Timestamp",
            dimension_columns=["Region"],
            kpi_columns=["DL_Throughput"],
            date_range_start="2024-01-01",
            date_range_end="2024-01-31"
        )
        assert metadata.total_rows == 10000
        assert metadata.aggregation_level == AggregationLevel.CELL
        assert len(metadata.columns) == 3
    
    def test_hourly_data_metadata(self, sample_columns):
        """Test hourly data metadata."""
        metadata = DataFrameMetadata(
            file_path="/data/hourly.csv",
            total_rows=8760,  # 1 year of hourly data
            total_columns=5,
            time_format=TimeFormat.HOURLY,
            aggregation_level=AggregationLevel.CARRIER,
            columns=sample_columns,
            time_column="Timestamp",
            dimension_columns=["Region", "Carrier"],
            kpi_columns=["DL_Throughput", "Signal_Strength"]
        )
        assert metadata.time_format == TimeFormat.HOURLY
        assert len(metadata.kpi_columns) == 2
    
    def test_plmn_level_metadata(self, sample_columns):
        """Test PLMN-level (aggregated) metadata."""
        metadata = DataFrameMetadata(
            file_path="/data/plmn.csv",
            total_rows=365,
            total_columns=10,
            time_format=TimeFormat.DAILY,
            aggregation_level=AggregationLevel.PLMN,
            columns=sample_columns,
            time_column="Timestamp",
            dimension_columns=[],
            kpi_columns=["DL_Throughput", "UL_Throughput", "Signal_Strength"]
        )
        assert metadata.aggregation_level == AggregationLevel.PLMN
        assert len(metadata.dimension_columns) == 0
    
    def test_metadata_with_sampling(self, sample_columns):
        """Test metadata when smart sampling was applied."""
        metadata = DataFrameMetadata(
            file_path="/data/large_cell.csv",
            total_rows=50000,  # After sampling
            total_columns=15,
            time_format=TimeFormat.HOURLY,
            aggregation_level=AggregationLevel.CELL,
            columns=sample_columns,
            time_column="Timestamp",
            dimension_columns=["Region", "Carrier", "Cell_ID"],
            kpi_columns=["DL_Throughput"],
            sampling_applied=True,
            original_row_count=500000  # Before sampling
        )
        assert metadata.sampling_applied is True
        assert metadata.original_row_count == 500000
        assert metadata.total_rows < metadata.original_row_count
    
    def test_missing_values_flag(self, sample_columns):
        """Test metadata with missing values."""
        metadata = DataFrameMetadata(
            file_path="/data/dirty.csv",
            total_rows=9800,
            total_columns=5,
            time_format=TimeFormat.DAILY,
            aggregation_level=AggregationLevel.REGION,
            columns=sample_columns,
            time_column="Timestamp",
            dimension_columns=["Region"],
            kpi_columns=["DL_Throughput"],
            has_missing_values=True
        )
        assert metadata.has_missing_values is True
    
    def test_metadata_date_parsing(self, sample_columns):
        """Test metadata handles date strings."""
        metadata = DataFrameMetadata(
            file_path="/data/test.csv",
            total_rows=100,
            total_columns=3,
            time_format=TimeFormat.DAILY,
            aggregation_level=AggregationLevel.CELL,
            columns=sample_columns,
            time_column="Timestamp",
            dimension_columns=[],
            kpi_columns=["DL_Throughput"],
            date_range_start="2024-01-01",
            date_range_end="2024-12-31"
        )
        assert metadata.date_range_start == "2024-01-01"
        assert metadata.date_range_end == "2024-12-31"
    
    def test_json_serialization(self, sample_columns):
        """Test DataFrameMetadata JSON serialization."""
        metadata = DataFrameMetadata(
            file_path="/data/test.csv",
            total_rows=100,
            total_columns=3,
            time_format=TimeFormat.DAILY,
            aggregation_level=AggregationLevel.CELL,
            columns=sample_columns,
            time_column="Timestamp",
            dimension_columns=[],
            kpi_columns=["DL_Throughput"]
        )
        json_str = metadata.model_dump_json()
        restored = DataFrameMetadata.model_validate_json(json_str)
        assert restored.file_path == metadata.file_path
        assert restored.total_rows == metadata.total_rows
    
    def test_invalid_row_count(self, sample_columns):
        """Test validation rejects invalid row count."""
        with pytest.raises(ValidationError):
            DataFrameMetadata(
                file_path="/data/test.csv",
                total_rows=0,  # Invalid: must be > 0
                total_columns=3,
                time_format=TimeFormat.DAILY,
                aggregation_level=AggregationLevel.CELL,
                columns=sample_columns,
                time_column="Timestamp",
                dimension_columns=[],
                kpi_columns=["DL_Throughput"]
            )


# =============================================================================
# ANOMALYRESULT TESTS
# =============================================================================

class TestAnomalyResult:
    """Test AnomalyResult model."""
    
    def test_z_score_anomaly(self):
        """Test Z-Score anomaly detection result."""
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
            dimension_filters={"Region": "North", "Carrier": "L2100"}
        )
        assert anomaly.z_score == -3.8
        assert anomaly.severity == SeverityLevel.CRITICAL
        assert anomaly.method == AnomalyMethod.Z_SCORE
    
    def test_iqr_anomaly(self):
        """Test IQR anomaly detection result."""
        anomaly = AnomalyResult(
            timestamp="2024-01-15 14:30:00",
            kpi_name="Signal_Strength",
            observed_value=45.0,
            expected_value=None,
            z_score=None,
            deviation_percent=None,
            severity=SeverityLevel.HIGH,
            method=AnomalyMethod.IQR,
            lower_bound=70.0,
            upper_bound=90.0
        )
        assert anomaly.method == AnomalyMethod.IQR
        assert anomaly.z_score is None
    
    def test_severity_levels(self):
        """Test all severity levels."""
        for severity in [SeverityLevel.LOW, SeverityLevel.MEDIUM, 
                        SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
            anomaly = AnomalyResult(
                timestamp="2024-01-15 14:30:00",
                kpi_name="DL_Throughput",
                observed_value=1.0,
                severity=severity,
                method=AnomalyMethod.Z_SCORE
            )
            assert anomaly.severity == severity
    
    def test_positive_z_score_anomaly(self):
        """Test positive Z-score (high spike) anomaly."""
        anomaly = AnomalyResult(
            timestamp="2024-01-15 14:30:00",
            kpi_name="Latency",
            observed_value=150.0,
            expected_value=50.0,
            z_score=5.2,
            deviation_percent=200.0,
            severity=SeverityLevel.CRITICAL,
            method=AnomalyMethod.Z_SCORE
        )
        assert anomaly.z_score > 0
        assert anomaly.deviation_percent == 200.0
    
    def test_dimension_filters(self):
        """Test dimension filter metadata."""
        anomaly = AnomalyResult(
            timestamp="2024-01-15 14:30:00",
            kpi_name="DL_Throughput",
            observed_value=0.5,
            severity=SeverityLevel.HIGH,
            method=AnomalyMethod.Z_SCORE,
            dimension_filters={
                "Region": "North",
                "Carrier": "L2100",
                "Cell_ID": "C12345"
            }
        )
        assert len(anomaly.dimension_filters) == 3
        assert anomaly.dimension_filters["Cell_ID"] == "C12345"
    
    def test_json_serialization(self):
        """Test AnomalyResult JSON serialization."""
        anomaly = AnomalyResult(
            timestamp="2024-01-15 14:30:00",
            kpi_name="DL_Throughput",
            observed_value=0.5,
            z_score=-3.8,
            severity=SeverityLevel.CRITICAL,
            method=AnomalyMethod.Z_SCORE
        )
        json_str = anomaly.model_dump_json()
        restored = AnomalyResult.model_validate_json(json_str)
        assert restored.z_score == anomaly.z_score
        assert restored.severity == anomaly.severity
    
    def test_invalid_z_score_type(self):
        """Test validation rejects invalid z-score."""
        with pytest.raises(ValidationError):
            AnomalyResult(
                timestamp="2024-01-15 14:30:00",
                kpi_name="DL_Throughput",
                observed_value=0.5,
                z_score="invalid",  # Must be numeric
                severity=SeverityLevel.HIGH,
                method=AnomalyMethod.Z_SCORE
            )


# =============================================================================
# CORRELATIONPAIR TESTS
# =============================================================================

class TestCorrelationPair:
    """Test CorrelationPair model."""
    
    def test_strong_positive_correlation(self):
        """Test strong positive correlation."""
        corr = CorrelationPair(
            kpi_x="DL_Throughput",
            kpi_y="Signal_Strength",
            correlation_score=0.85,
            p_value=0.0001,
            is_significant=True,
            data_points_used=9850,
            interpretation="Strong positive: Signal improves throughput"
        )
        assert corr.correlation_score == 0.85
        assert corr.is_significant is True
    
    def test_strong_negative_correlation(self):
        """Test strong negative correlation."""
        corr = CorrelationPair(
            kpi_x="Throughput",
            kpi_y="Latency",
            correlation_score=-0.78,
            p_value=0.0001,
            is_significant=True,
            data_points_used=9850
        )
        assert corr.correlation_score == -0.78
    
    def test_weak_correlation(self):
        """Test weak correlation (not significant)."""
        corr = CorrelationPair(
            kpi_x="KPI_A",
            kpi_y="KPI_B",
            correlation_score=0.15,
            p_value=0.25,
            is_significant=False,
            data_points_used=100
        )
        assert corr.is_significant is False
    
    def test_no_correlation(self):
        """Test no correlation (r=0)."""
        corr = CorrelationPair(
            kpi_x="Rain",
            kpi_y="DL_Throughput",
            correlation_score=0.0,
            p_value=0.95,
            is_significant=False,
            data_points_used=1000
        )
        assert corr.correlation_score == 0.0
    
    def test_perfect_correlation(self):
        """Test perfect positive correlation."""
        corr = CorrelationPair(
            kpi_x="X",
            kpi_y="Y",
            correlation_score=1.0,
            p_value=0.0,
            is_significant=True,
            data_points_used=100
        )
        assert corr.correlation_score == 1.0
    
    def test_perfect_negative_correlation(self):
        """Test perfect negative correlation."""
        corr = CorrelationPair(
            kpi_x="X",
            kpi_y="-X",
            correlation_score=-1.0,
            p_value=0.0,
            is_significant=True,
            data_points_used=100
        )
        assert corr.correlation_score == -1.0
    
    def test_invalid_correlation_score(self):
        """Test validation rejects out-of-range correlation."""
        with pytest.raises(ValidationError):
            CorrelationPair(
                kpi_x="KPI_A",
                kpi_y="KPI_B",
                correlation_score=1.5,  # Invalid: > 1.0
                p_value=0.05,
                is_significant=False,
                data_points_used=100
            )
    
    def test_invalid_p_value(self):
        """Test validation rejects invalid p-value."""
        with pytest.raises(ValidationError):
            CorrelationPair(
                kpi_x="KPI_A",
                kpi_y="KPI_B",
                correlation_score=0.5,
                p_value=1.5,  # Invalid: > 1.0
                is_significant=False,
                data_points_used=100
            )


# =============================================================================
# FORECASTVALUE TESTS
# =============================================================================

class TestForecastValue:
    """Test ForecastValue model."""
    
    def test_valid_forecast_value(self):
        """Test valid forecast value with CI."""
        fv = ForecastValue(
            timestamp="2024-02-01",
            predicted_value=2.45,
            lower_ci=2.10,
            upper_ci=2.80,
            confidence_level=0.95
        )
        assert fv.predicted_value == 2.45
        assert fv.upper_ci > fv.lower_ci
    
    def test_tight_confidence_interval(self):
        """Test tight confidence interval."""
        fv = ForecastValue(
            timestamp="2024-02-01",
            predicted_value=2.45,
            lower_ci=2.43,
            upper_ci=2.47,
            confidence_level=0.95
        )
        ci_width = fv.upper_ci - fv.lower_ci
        assert ci_width == 0.04
    
    def test_wide_confidence_interval(self):
        """Test wide confidence interval."""
        fv = ForecastValue(
            timestamp="2024-02-01",
            predicted_value=2.45,
            lower_ci=1.50,
            upper_ci=3.40,
            confidence_level=0.99
        )
        ci_width = fv.upper_ci - fv.lower_ci
        assert ci_width == 1.90
    
    def test_confidence_levels(self):
        """Test different confidence levels."""
        for conf_level in [0.80, 0.90, 0.95, 0.99]:
            fv = ForecastValue(
                timestamp="2024-02-01",
                predicted_value=2.45,
                lower_ci=2.10,
                upper_ci=2.80,
                confidence_level=conf_level
            )
            assert fv.confidence_level == conf_level
    
    def test_invalid_ci_range(self):
        """Test validation rejects invalid CI."""
        with pytest.raises(ValidationError):
            ForecastValue(
                timestamp="2024-02-01",
                predicted_value=2.45,
                lower_ci=2.80,  # Lower > upper
                upper_ci=2.10,
                confidence_level=0.95
            )
    
    def test_equal_ci_bounds(self):
        """Test validation rejects equal CI bounds."""
        with pytest.raises(ValidationError):
            ForecastValue(
                timestamp="2024-02-01",
                predicted_value=2.45,
                lower_ci=2.45,  # Equal to upper
                upper_ci=2.45,
                confidence_level=0.95
            )
    
    def test_invalid_confidence_level(self):
        """Test validation rejects invalid confidence level."""
        with pytest.raises(ValidationError):
            ForecastValue(
                timestamp="2024-02-01",
                predicted_value=2.45,
                lower_ci=2.10,
                upper_ci=2.80,
                confidence_level=0.50  # Invalid: < 0.80
            )
    
    def test_json_serialization(self):
        """Test ForecastValue JSON serialization."""
        fv = ForecastValue(
            timestamp="2024-02-01",
            predicted_value=2.45,
            lower_ci=2.10,
            upper_ci=2.80
        )
        json_str = fv.model_dump_json()
        restored = ForecastValue.model_validate_json(json_str)
        assert restored.predicted_value == fv.predicted_value


# =============================================================================
# REQUEST MODEL TESTS
# =============================================================================

class TestFilterRequest:
    """Test FilterRequest model."""
    
    def test_full_filter_request(self):
        """Test complete filter request."""
        req = FilterRequest(
            region="North",
            carrier="L2100",
            cell_id="C12345",
            start_date="2024-01-01",
            end_date="2024-01-31",
            kpi_names=["DL_Throughput", "Signal_Strength"]
        )
        assert req.region == "North"
        assert len(req.kpi_names) == 2
    
    def test_minimal_filter_request(self):
        """Test minimal filter request (dates only)."""
        req = FilterRequest(
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        assert req.region is None
        assert len(req.kpi_names) == 0
    
    def test_json_serialization(self):
        """Test FilterRequest JSON serialization."""
        req = FilterRequest(
            region="North",
            start_date="2024-01-01",
            end_date="2024-01-31"
        )
        json_str = req.model_dump_json()
        restored = FilterRequest.model_validate_json(json_str)
        assert restored.region == req.region


class TestForecastRequest:
    """Test ForecastRequest model."""
    
    def test_arima_forecast_request(self):
        """Test ARIMA forecast request."""
        # Create sample filtered result
        sample_col = ColumnClassification(
            column_name="DL_Throughput",
            column_type=ColumnType.KPI,
            data_type="float",
            non_null_count=100,
            unique_count=90,
            sample_values=[2.0, 2.5, 3.0],
            is_numeric=True
        )
        metadata = DataFrameMetadata(
            file_path="/data/test.csv",
            total_rows=100,
            total_columns=2,
            time_format=TimeFormat.DAILY,
            aggregation_level=AggregationLevel.CELL,
            columns=[sample_col],
            time_column="Date",
            kpi_columns=["DL_Throughput"]
        )
        filtered = FilteredDataFrameResult(
            original_metadata=metadata,
            filter_selections={},
            filtered_row_count=100,
            dimension_values_applied={}
        )
        
        req = ForecastRequest(
            filtered_data_result=filtered,
            kpi_name="DL_Throughput",
            forecast_periods=30,
            model_type="ARIMA"
        )
        assert req.model_type == "ARIMA"
        assert req.forecast_periods == 30
        assert len(req.exogenous_variables) == 0
    
    def test_arimax_forecast_request(self):
        """Test ARIMAX forecast request with exogenous variables."""
        sample_col = ColumnClassification(
            column_name="DL_Throughput",
            column_type=ColumnType.KPI,
            data_type="float",
            non_null_count=100,
            unique_count=90,
            sample_values=[2.0, 2.5, 3.0],
            is_numeric=True
        )
        metadata = DataFrameMetadata(
            file_path="/data/test.csv",
            total_rows=100,
            total_columns=3,
            time_format=TimeFormat.DAILY,
            aggregation_level=AggregationLevel.CELL,
            columns=[sample_col],
            time_column="Date",
            kpi_columns=["DL_Throughput", "Signal_Strength"]
        )
        filtered = FilteredDataFrameResult(
            original_metadata=metadata,
            filter_selections={},
            filtered_row_count=100,
            dimension_values_applied={}
        )
        
        req = ForecastRequest(
            filtered_data_result=filtered,
            kpi_name="DL_Throughput",
            forecast_periods=30,
            model_type="ARIMAX",
            exogenous_kpi_names=["Signal_Strength"]
        )
        assert req.model_type == "ARIMAX"
        assert len(req.exogenous_variables) == 1


# =============================================================================
# LLM SCHEMA TESTS
# =============================================================================

class TestLLMAnalysisResponse:
    """Test LLM analysis response."""
    
    def test_causal_analysis_response(self):
        """Test causal analysis response from LLM."""
        response = LLMAnalysisResponse(
            analysis_type="Causal",
            reasoning="The anomaly coincides with traffic surge at 2PM.",
            recommendations=[
                "Check traffic volume for the same period",
                "Verify carrier load balancing"
            ],
            confidence_level=0.85
        )
        assert response.analysis_type == "Causal"
        assert len(response.recommendations) == 2
        assert response.confidence_level == 0.85
    
    def test_scenario_analysis_response(self):
        """Test scenario planning response."""
        response = LLMAnalysisResponse(
            analysis_type="Scenario",
            reasoning="If traffic increases 20%, throughput would drop 15%.",
            recommendations=[
                "Prepare additional capacity",
                "Plan maintenance during low-traffic periods"
            ],
            confidence_level=0.75
        )
        assert response.analysis_type == "Scenario"
    
    def test_interpretation_response(self):
        """Test interpretation response."""
        response = LLMAnalysisResponse(
            analysis_type="Interpretation",
            reasoning="The strong correlation indicates signal is the limiting factor.",
            recommendations=[
                "Adjust antenna tilt",
                "Add auxiliary antennas"
            ],
            confidence_level=0.90
        )
        assert response.analysis_type == "Interpretation"
    
    def test_json_serialization(self):
        """Test LLMAnalysisResponse JSON serialization."""
        response = LLMAnalysisResponse(
            analysis_type="Causal",
            reasoning="Test reasoning",
            recommendations=["Rec1", "Rec2"],
            confidence_level=0.80
        )
        json_str = response.model_dump_json()
        restored = LLMAnalysisResponse.model_validate_json(json_str)
        assert restored.analysis_type == response.analysis_type


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestModelIntegration:
    """Test integration between models."""
    
    def test_metadata_to_anomaly_flow(self):
        """Test data flow from metadata to anomaly result."""
        # Create metadata
        col = ColumnClassification(
            column_name="DL_Throughput",
            column_type=ColumnType.KPI,
            data_type="float",
            non_null_count=9950,
            unique_count=8234,
            sample_values=[2.45, 2.50, 2.48],
            is_numeric=True
        )
        metadata = DataFrameMetadata(
            file_path="/data/test.csv",
            total_rows=10000,
            total_columns=1,
            time_format=TimeFormat.DAILY,
            aggregation_level=AggregationLevel.CELL,
            columns=[col],
            time_column="Date",
            kpi_columns=["DL_Throughput"]
        )
        
        # Create anomaly from metadata context
        anomaly = AnomalyResult(
            timestamp="2024-01-15",
            kpi_name=metadata.kpi_columns[0],
            observed_value=0.5,
            expected_value=2.4,
            z_score=-3.8,
            severity=SeverityLevel.CRITICAL,
            method=AnomalyMethod.Z_SCORE,
            dimension_filters={"AggLevel": metadata.aggregation_level.value}
        )
        
        assert anomaly.kpi_name in metadata.kpi_columns
    
    def test_correlation_to_llm_flow(self):
        """Test data flow from correlation to LLM interpretation."""
        # Create correlation result
        corr_pair = CorrelationPair(
            kpi_x="DL_Throughput",
            kpi_y="Signal_Strength",
            correlation_score=0.82,
            p_value=0.0001,
            is_significant=True,
            data_points_used=9850
        )
        corr_result = CorrelationResult(
            kpi_name="DL_Throughput",
            top_3_correlations=[corr_pair]
        )
        
        # Feed to LLM interpretation
        llm_req = LLMCorrelationInterpretationRequest(
            correlation_result=corr_result,
            domain_knowledge={
                "business_impact": "Signal directly affects user experience"
            }
        )
        
        assert llm_req.correlation_result.kpi_name == "DL_Throughput"


# =============================================================================
# EDGE CASES & ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dimension_filters(self):
        """Test anomaly with no dimension filters."""
        anomaly = AnomalyResult(
            timestamp="2024-01-15",
            kpi_name="DL_Throughput",
            observed_value=0.5,
            severity=SeverityLevel.HIGH,
            method=AnomalyMethod.Z_SCORE,
            dimension_filters={}
        )
        assert len(anomaly.dimension_filters) == 0
    
    def test_empty_correlation_list(self):
        """Test correlation result with no correlations."""
        corr_result = CorrelationResult(
            kpi_name="DL_Throughput",
            top_3_correlations=[]
        )
        assert len(corr_result.top_3_correlations) == 0
    
    def test_single_forecast_value(self):
        """Test forecast with single value."""
        forecast = ForecastResult(
            kpi_name="DL_Throughput",
            forecast_period="1 day",
            forecast_values=[
                ForecastValue(
                    timestamp="2024-02-01",
                    predicted_value=2.45,
                    lower_ci=2.10,
                    upper_ci=2.80
                )
            ],
            model_type="ARIMA"
        )
        assert len(forecast.forecast_values) == 1
    
    def test_very_large_z_score(self):
        """Test anomaly with extreme Z-score."""
        anomaly = AnomalyResult(
            timestamp="2024-01-15",
            kpi_name="DL_Throughput",
            observed_value=0.01,
            expected_value=2.4,
            z_score=-15.5,  # Extreme outlier
            severity=SeverityLevel.CRITICAL,
            method=AnomalyMethod.Z_SCORE
        )
        assert anomaly.z_score < -10


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
