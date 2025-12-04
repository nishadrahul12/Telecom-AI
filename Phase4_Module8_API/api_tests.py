# PHASE 4 MODULE 8 - api_tests.py (Unit & Integration Tests)
# Comprehensive test coverage for all 11 API endpoints

"""
Unit tests for FastAPI Telecom Optimization API.
Tests all endpoints with valid/invalid inputs, error cases, and integration scenarios.

Run with: pytest api_tests.py -v
Coverage: pytest api_tests.py --cov=api --cov-report=html
"""

import pytest
import pandas as pd
import numpy as np
import json
from fastapi.testclient import TestClient
from io import BytesIO

# Import API application
from api import app, session_state, SessionState

# ============================================================================
# TEST CLIENT SETUP
# ============================================================================

client = TestClient(app)


@pytest.fixture
def reset_session():
    """Reset session state before each test."""
    session_state.reset(hard_reset=True)
    yield
    session_state.reset(hard_reset=True)


@pytest.fixture
def sample_dataframe():
    """Create sample telecom data for testing."""
    data = {
        'TIME': ['3/1/2024', '3/2/2024', '3/3/2024', '3/4/2024', '3/5/2024'],
        'REGION': ['N1', 'N1', 'N1', 'N2', 'N2'],
        'CARRIER_NAME': ['L2100', 'L2100', 'L1800', 'L2100', 'L1800'],
        'MRBTS_ID': [100001, 100001, 100002, 100003, 100003],
        'LNCEL_ID': [111, 112, 111, 111, 112],
        'RACH stp att': [50000, 52000, 48000, 55000, 51000],
        'RRC stp att': [30000, 31000, 29000, 32000, 30000],
        'E-UTRAN avg RRC conn UEs': [15.5, 16.2, 14.8, 17.1, 15.9],
        'RACH Stp Completion SR': [99.5, 99.4, 99.6, 99.3, 99.5],
        'RRC conn stp SR': [99.8, 99.7, 99.9, 99.6, 99.8],
    }
    return pd.DataFrame(data)


@pytest.fixture
def upload_sample_file(sample_dataframe, reset_session):
    """Upload sample CSV file for testing."""
    # Convert DataFrame to CSV bytes
    csv_buffer = BytesIO()
    sample_dataframe.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    # Upload via API
    response = client.post(
        "/upload",
        files={"file": ("test.csv", csv_buffer, "text/csv")},
    )
    
    assert response.status_code == 200
    yield response.json()


# ============================================================================
# TESTS: POST /upload
# ============================================================================

class TestUploadEndpoint:
    """Tests for POST /upload endpoint."""

    def test_upload_valid_csv(self, reset_session, sample_dataframe):
        """Test uploading a valid CSV file."""
        csv_buffer = BytesIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/upload",
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["file_info"]["filename"] == "test.csv"
        assert data["dataframe_metadata"]["row_count"] == 5
        assert "TIME" in data["dataframe_metadata"]["time_column"]

    def test_upload_no_file(self, reset_session):
        """Test uploading without file (should fail)."""
        response = client.post("/upload", files={})
        assert response.status_code in [400, 422]

    def test_upload_invalid_format(self, reset_session):
        """Test uploading non-CSV file."""
        response = client.post(
            "/upload",
            files={"file": ("test.txt", b"invalid", "text/plain")},
        )
        assert response.status_code == 422

    def test_upload_stores_in_session(self, reset_session, sample_dataframe):
        """Test that upload stores DataFrame in session."""
        csv_buffer = BytesIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/upload",
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )

        assert response.status_code == 200
        assert session_state.dataframe is not None
        assert len(session_state.dataframe) == 5

    def test_upload_auto_classifies_columns(self, reset_session, sample_dataframe):
        """Test column classification."""
        csv_buffer = BytesIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/upload",
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )

        data = response.json()
        assert len(data["dataframe_metadata"]["dimensions_text"]) > 0
        assert len(data["dataframe_metadata"]["kpis"]) > 0


# ============================================================================
# TESTS: GET /levels
# ============================================================================

class TestLevelsEndpoint:
    """Tests for GET /levels endpoint."""

    def test_get_levels(self):
        """Test retrieving aggregation levels."""
        response = client.get("/levels")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert set(data["levels"]) == {"PLMN", "Region", "Carrier", "Cell"}

    def test_levels_static_response(self):
        """Test that levels response is consistent."""
        response1 = client.get("/levels")
        response2 = client.get("/levels")

        assert response1.json() == response2.json()


# ============================================================================
# TESTS: GET /filters/{level}
# ============================================================================

class TestFiltersEndpoint:
    """Tests for GET /filters/{level} endpoint."""

    def test_get_filters_no_data(self, reset_session):
        """Test getting filters without uploaded data."""
        response = client.get("/filters/Region")
        assert response.status_code == 404

    def test_get_filters_with_data(self, upload_sample_file):
        """Test getting filters after uploading data."""
        response = client.get("/filters/Region")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["data_level"] == "Region"
        assert "unique_values" in data
        assert "value_counts" in data

    def test_get_filters_invalid_level(self, upload_sample_file):
        """Test with invalid data level."""
        response = client.get("/filters/InvalidLevel")
        assert response.status_code in [400, 422]

    def test_get_filters_returns_dimensions(self, upload_sample_file):
        """Test that filters include text and ID dimensions."""
        response = client.get("/filters/Region")

        data = response.json()
        assert len(data["text_dimensions"]) > 0
        assert len(data["id_dimensions"]) > 0


# ============================================================================
# TESTS: POST /apply-filters
# ============================================================================

class TestApplyFiltersEndpoint:
    """Tests for POST /apply-filters endpoint."""

    def test_apply_filters_no_data(self, reset_session):
        """Test applying filters without data."""
        response = client.post(
            "/apply-filters",
            json={
                "data_level": "Region",
                "filters": {"REGION": ["N1"]},
                "sampling_strategy": "smart",
            },
        )
        assert response.status_code == 404

    def test_apply_filters_valid(self, upload_sample_file):
        """Test applying valid filters."""
        response = client.post(
            "/apply-filters",
            json={
                "data_level": "Region",
                "filters": {"REGION": ["N1"]},
                "sampling_strategy": "smart",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["filtered_dataframe_summary"]["row_count_original"] == 5
        assert data["filtered_dataframe_summary"]["row_count_after_filtering"] > 0

    def test_apply_filters_multiple_criteria(self, upload_sample_file):
        """Test applying multiple filter criteria."""
        response = client.post(
            "/apply-filters",
            json={
                "data_level": "Region",
                "filters": {
                    "REGION": ["N1"],
                    "CARRIER_NAME": ["L2100"],
                },
                "sampling_strategy": "smart",
            },
        )

        assert response.status_code == 200

    def test_apply_filters_nonexistent_column(self, upload_sample_file):
        """Test filtering on non-existent column."""
        response = client.post(
            "/apply-filters",
            json={
                "data_level": "Region",
                "filters": {"INVALID_COL": ["value"]},
                "sampling_strategy": "smart",
            },
        )
        assert response.status_code == 404

    def test_apply_filters_updates_session(self, upload_sample_file):
        """Test that filters update session state."""
        response = client.post(
            "/apply-filters",
            json={
                "data_level": "Region",
                "filters": {"REGION": ["N1"]},
                "sampling_strategy": "smart",
            },
        )

        assert response.status_code == 200
        assert session_state.applied_filters == {"REGION": ["N1"]}


# ============================================================================
# TESTS: GET /anomalies
# ============================================================================

class TestAnomaliesEndpoint:
    """Tests for GET /anomalies endpoint."""

    def test_get_anomalies_no_data(self, reset_session):
        """Test getting anomalies without data."""
        response = client.get("/anomalies")
        assert response.status_code == 404

    def test_get_anomalies_with_data(self, upload_sample_file):
        """Test getting anomalies after uploading data."""
        response = client.get("/anomalies")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "time_series_anomalies" in data
        assert "distributional_outliers" in data
        assert "total_anomalies" in data
        assert "anomaly_percentage" in data

    def test_get_anomalies_response_structure(self, upload_sample_file):
        """Test anomalies response structure."""
        response = client.get("/anomalies")

        data = response.json()
        assert isinstance(data["time_series_anomalies"], list)
        assert isinstance(data["distributional_outliers"], dict)
        assert data["anomaly_percentage"] >= 0
        assert data["anomaly_percentage"] <= 1


# ============================================================================
# TESTS: GET /correlation
# ============================================================================

class TestCorrelationEndpoint:
    """Tests for GET /correlation endpoint."""

    def test_get_correlation_no_data(self, reset_session):
        """Test getting correlation without data."""
        response = client.get("/correlation")
        assert response.status_code == 404

    def test_get_correlation_with_data(self, upload_sample_file):
        """Test getting correlation after uploading data."""
        response = client.get("/correlation")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "correlation_matrix" in data
        assert "kpi_names" in data
        assert "top_3_per_kpi" in data

    def test_get_correlation_matrix_format(self, upload_sample_file):
        """Test correlation matrix format."""
        response = client.get("/correlation")

        data = response.json()
        assert isinstance(data["correlation_matrix"], list)
        assert all(isinstance(row, list) for row in data["correlation_matrix"])
        assert len(data["correlation_matrix"]) > 0

    def test_get_correlation_top_3(self, upload_sample_file):
        """Test Top-3 correlation rankings."""
        response = client.get("/correlation")

        data = response.json()
        for kpi, correlations in data["top_3_per_kpi"].items():
            assert len(correlations) <= 3
            for corr in correlations:
                assert "target_kpi" in corr
                assert "correlation_score" in corr


# ============================================================================
# TESTS: POST /forecast
# ============================================================================

class TestForecastEndpoint:
    """Tests for POST /forecast endpoint."""

    def test_forecast_no_data(self, reset_session):
        """Test forecast without data."""
        response = client.post(
            "/forecast",
            json={
                "target_kpi": "RACH stp att",
                "forecast_horizon": 7,
                "mode": "univariate",
            },
        )
        assert response.status_code == 404

    def test_forecast_univariate(self, upload_sample_file):
        """Test univariate forecast."""
        response = client.post(
            "/forecast",
            json={
                "target_kpi": "RACH stp att",
                "forecast_horizon": 7,
                "mode": "univariate",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["model_type"] == "ARIMA"
        assert len(data["forecast_values"]) == 7
        assert len(data["forecast_dates"]) == 7

    def test_forecast_multivariate(self, upload_sample_file):
        """Test multivariate forecast with exogenous variables."""
        response = client.post(
            "/forecast",
            json={
                "target_kpi": "RACH stp att",
                "forecast_horizon": 7,
                "mode": "multivariate",
                "exogenous_kpis": ["E-UTRAN avg RRC conn UEs"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "ARIMAX"

    def test_forecast_invalid_horizon(self, upload_sample_file):
        """Test forecast with invalid horizon."""
        response = client.post(
            "/forecast",
            json={
                "target_kpi": "RACH stp att",
                "forecast_horizon": 100,  # > 30
                "mode": "univariate",
            },
        )
        assert response.status_code == 422

    def test_forecast_kpi_not_found(self, upload_sample_file):
        """Test forecast with non-existent KPI."""
        response = client.post(
            "/forecast",
            json={
                "target_kpi": "INVALID_KPI",
                "forecast_horizon": 7,
                "mode": "univariate",
            },
        )
        assert response.status_code == 422

    def test_forecast_response_structure(self, upload_sample_file):
        """Test forecast response contains all required fields."""
        response = client.post(
            "/forecast",
            json={
                "target_kpi": "RACH stp att",
                "forecast_horizon": 7,
                "mode": "univariate",
            },
        )

        data = response.json()
        assert "forecast_values" in data
        assert "confidence_interval_lower" in data
        assert "confidence_interval_upper" in data
        assert "historical_values" in data
        assert "model_metrics" in data


# ============================================================================
# TESTS: POST /llama-analyze
# ============================================================================

class TestLlamaAnalyzeEndpoint:
    """Tests for POST /llama-analyze endpoint."""

    def test_llama_causal_analysis(self):
        """Test Llama causal analysis request."""
        response = client.post(
            "/llama-analyze",
            json={
                "request_type": "Causal_Anomaly_Analysis",
                "target_anomaly": {
                    "kpi_name": "RACH stp att",
                    "actual_value": 89000,
                },
                "contextual_data": [],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "llm_response" in data

    def test_llama_scenario_planning(self):
        """Test Llama scenario planning request."""
        response = client.post(
            "/llama-analyze",
            json={
                "request_type": "Scenario_Planning_Forecast",
                "forecast_target": "RACH stp att",
                "forecast_horizon_days": 7,
                "current_value": 50000,
                "predicted_value": 55000,
                "critical_threshold": 60000,
                "model_parameters": [],
            },
        )

        assert response.status_code == 200

    def test_llama_correlation_interpretation(self):
        """Test Llama correlation interpretation request."""
        response = client.post(
            "/llama-analyze",
            json={
                "request_type": "Correlation_Interpretation",
                "source_kpi": "RACH stp att",
                "target_kpi": "RRC stp att",
                "correlation_score": 0.85,
            },
        )

        assert response.status_code == 200

    def test_llama_response_has_fallback(self):
        """Test that Llama returns fallback when unavailable."""
        response = client.post(
            "/llama-analyze",
            json={
                "request_type": "Causal_Anomaly_Analysis",
                "target_anomaly": {"kpi_name": "RACH stp att"},
                "contextual_data": [],
            },
        )

        data = response.json()
        # Should have fallback template since Ollama not running
        assert "model_used" in data["llm_response"]


# ============================================================================
# TESTS: GET /health
# ============================================================================

class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "components" in data

    def test_health_components_present(self):
        """Test that all components listed in health check."""
        response = client.get("/health")

        data = response.json()
        required_components = [
            "fastapi",
            "data_ingestion",
            "correlation_module",
            "forecasting_module",
        ]
        for component in required_components:
            assert component in data["components"]


# ============================================================================
# TESTS: GET /current-state
# ============================================================================

class TestCurrentStateEndpoint:
    """Tests for GET /current-state endpoint."""

    def test_current_state_no_data(self, reset_session):
        """Test getting state without data."""
        response = client.get("/current-state")
        assert response.status_code == 404

    def test_current_state_with_data(self, upload_sample_file):
        """Test getting current state after upload."""
        response = client.get("/current-state")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "session_state" in data

    def test_current_state_structure(self, upload_sample_file):
        """Test current state structure."""
        response = client.get("/current-state")

        data = response.json()["session_state"]
        assert "dataframe_shape" in data
        assert "kpi_count" in data
        assert "dimension_count" in data
        assert "sampling_applied" in data


# ============================================================================
# TESTS: POST /reset-state
# ============================================================================

class TestResetStateEndpoint:
    """Tests for POST /reset-state endpoint."""

    def test_reset_state_no_data(self, reset_session):
        """Test reset without data."""
        response = client.post("/reset-state")
        assert response.status_code == 404

    def test_reset_state_soft_reset(self, upload_sample_file):
        """Test soft reset (clears filters only)."""
        # Apply filters first
        client.post(
            "/apply-filters",
            json={
                "data_level": "Region",
                "filters": {"REGION": ["N1"]},
            },
        )

        # Reset
        response = client.post("/reset-state?hard_reset=false")

        assert response.status_code == 200
        assert session_state.applied_filters == {}

    def test_reset_state_hard_reset(self, upload_sample_file):
        """Test hard reset (clears everything)."""
        response = client.post("/reset-state?hard_reset=true")

        assert response.status_code == 200
        assert session_state.dataframe is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, sample_dataframe, reset_session):
        """Test complete analysis workflow."""
        # 1. Upload
        csv_buffer = BytesIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        response = client.post(
            "/upload",
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )
        assert response.status_code == 200

        # 2. Get levels
        response = client.get("/levels")
        assert response.status_code == 200

        # 3. Get filters
        response = client.get("/filters/Region")
        assert response.status_code == 200

        # 4. Apply filters
        response = client.post(
            "/apply-filters",
            json={
                "data_level": "Region",
                "filters": {"REGION": ["N1"]},
                "sampling_strategy": "smart",
            },
        )
        assert response.status_code == 200

        # 5. Get anomalies
        response = client.get("/anomalies")
        assert response.status_code == 200

        # 6. Get correlation
        response = client.get("/correlation")
        assert response.status_code == 200

        # 7. Forecast
        response = client.post(
            "/forecast",
            json={
                "target_kpi": "RACH stp att",
                "forecast_horizon": 3,
                "mode": "univariate",
            },
        )
        assert response.status_code == 200

        # 8. Health check
        response = client.get("/health")
        assert response.status_code == 200

    def test_session_state_persistence(self, sample_dataframe, reset_session):
        """Test that session state persists across requests."""
        # Upload
        csv_buffer = BytesIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        client.post(
            "/upload",
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )

        # Verify state persists
        state1 = client.get("/current-state").json()
        state2 = client.get("/current-state").json()

        assert state1["session_state"]["dataframe_shape"] == state2["session_state"]["dataframe_shape"]


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Performance tests to verify response time targets."""

    def test_upload_performance(self, sample_dataframe, reset_session):
        """Test upload meets <3s target."""
        csv_buffer = BytesIO()
        sample_dataframe.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        import time
        start = time.time()
        response = client.post(
            "/upload",
            files={"file": ("test.csv", csv_buffer, "text/csv")},
        )
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 3.0  # 3 second target

    def test_levels_performance(self):
        """Test /levels meets <100ms target."""
        import time
        start = time.time()
        response = client.get("/levels")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 0.1  # 100ms target

    def test_health_performance(self):
        """Test /health meets <200ms target."""
        import time
        start = time.time()
        response = client.get("/health")
        elapsed = time.time() - start

        assert response.status_code == 200
        assert elapsed < 0.2  # 200ms target


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_malformed_json(self, reset_session):
        """Test handling of malformed JSON."""
        response = client.post(
            "/apply-filters",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in [400, 422]

    def test_missing_required_field(self, reset_session, upload_sample_file):
        """Test POST with missing required field."""
        response = client.post(
            "/forecast",
            json={"forecast_horizon": 7},  # Missing target_kpi
        )
        assert response.status_code == 422

    def test_invalid_enum_value(self, reset_session, upload_sample_file):
        """Test POST with invalid enum value."""
        response = client.post(
            "/apply-filters",
            json={
                "data_level": "INVALID",  # Invalid enum
                "filters": {},
            },
        )
        assert response.status_code == 422


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])