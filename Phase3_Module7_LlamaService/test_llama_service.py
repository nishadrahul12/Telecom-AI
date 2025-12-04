# Phase3_Module7_LlamaService/test_llama_service.py

import pytest
import json
import requests
from unittest.mock import Mock, patch

from llama_service import (  # Added '.'
    LlamaService,
    OllamaConnectionError,
    OllamaTimeoutError,
)

from models import (         # Added '.'
    LLMResponse,
    CausalAnalysisRequest,
    AnomalyDetails,
    ContextualKPI,
    ConfidenceLevel,
)




# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def service():
    """Create LlamaService instance for testing"""
    return LlamaService(
        host="localhost",
        port=11434,
        timeout_seconds=10
    )


@pytest.fixture
def anomaly_data():
    """Sample anomaly data for testing"""
    return {
        "request_type": "Causal_Anomaly_Analysis",
        "target_anomaly": {
            "kpi_name": "RACH_Stp_att",
            "date_time": "2024-03-15",
            "actual_value": 227799,
            "expected_range": "50000 - 100000",
            "severity": "High",
            "zscore": 3.8
        },
        "contextual_data": [
            {
                "kpi_name": "E-UTRAN_avg_RRC_conn_UEs",
                "value_on_anomaly_date": 25.73,
                "correlation_score": 0.92,
                "historical_state": "Elevated"
            }
        ]
    }


@pytest.fixture
def scenario_data():
    """Sample scenario planning data for testing"""
    return {
        "request_type": "Scenario_Planning_Forecast",
        "forecast_target": "E-RAB_Setup_SR",
        "forecast_horizon_days": 7,
        "current_value": 98.5,
        "predicted_value": 90.1,
        "critical_threshold": 95.0,
        "model_parameters": [
            {
                "variable_name": "Traffic_Volume_DL",
                "projected_change": "Increased by 15%",
                "influence_score": 0.78,
                "influence_description": "Directly drives the predicted decline"
            }
        ]
    }


# ============================================================================
# CONNECTION TESTS
# ============================================================================

def test_connect_to_ollama_success(service):
    """Test successful connection to Ollama"""
    with patch('requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama2:70b"}]}
        mock_get.return_value = mock_response
        
        assert service.connect_to_ollama() is True


def test_connect_to_ollama_connection_error(service):
    """Test connection refused error"""
    with patch('requests.get') as mock_get:
        # Must mock the specific connection error requests raises
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        with pytest.raises(OllamaConnectionError):
            service.connect_to_ollama()


def test_connect_to_ollama_timeout(service):
    """Test connection timeout"""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(OllamaConnectionError):
            service.connect_to_ollama()


# ============================================================================
# CAUSAL ANALYSIS TESTS
# ============================================================================

def test_generate_causal_analysis_success(service, anomaly_data):
    """Test successful causal analysis generation"""
    with patch.object(service, '_call_ollama') as mock_ollama:
        mock_ollama.return_value = """
The RACH setup attempts spike to 227,799 is primarily driven by elevated RRC connections 
(25.73 UEs). This suggests a traffic surge event.

Recommendations:
1. Verify load balancing across carriers
2. Check handover success rates
3. Investigate interference patterns
"""
        
        response = service.generate_causal_analysis(anomaly_data)
        
        assert response.request_type == "Causal_Anomaly_Analysis"
        assert len(response.analysis) > 0
        assert len(response.recommendations) >= 1
        assert response.error is None


def test_generate_causal_analysis_ollama_unavailable(service, anomaly_data):
    """Test causal analysis fallback when Ollama unavailable"""
    with patch.object(service, '_call_ollama') as mock_ollama:
        mock_ollama.side_effect = OllamaConnectionError("Connection failed")
        
        response = service.generate_causal_analysis(anomaly_data)
        
        assert response.request_type == "Causal_Anomaly_Analysis"
        assert response.model_used == "Fallback-Text-Template"
        assert response.error is not None


# ============================================================================
# SCENARIO PLANNING TESTS
# ============================================================================

def test_generate_scenario_planning_success(service, scenario_data):
    """Test successful scenario planning generation"""
    with patch.object(service, '_call_ollama') as mock_ollama:
        mock_ollama.return_value = """
E-RAB Setup Success Rate is forecast to decline from 98.5% to 90.1%, crossing 
the critical threshold of 95%. This represents significant service degradation.

Mitigation recommendations:
1. Increase carrier capacity by activating additional sectors
2. Rebalance traffic load across available carriers
3. Optimize RACH parameters for improved efficiency
"""
        
        response = service.generate_scenario_planning(scenario_data)
        
        assert response.request_type == "Scenario_Planning_Forecast"
        assert len(response.recommendations) >= 1


def test_generate_scenario_planning_confidence_assessment(service, scenario_data):
    """Test confidence level assessment in scenario planning"""
    with patch.object(service, '_call_ollama') as mock_ollama:
        # Test HIGH confidence
        mock_ollama.return_value = "This clearly indicates a significant decline..."
        response = service.generate_scenario_planning(scenario_data)
        assert response.confidence_level == ConfidenceLevel.HIGH
        
        # Test LOW confidence
        mock_ollama.return_value = "The model might suggest a possible decline..."
        response = service.generate_scenario_planning(scenario_data)
        assert response.confidence_level == ConfidenceLevel.LOW


# ============================================================================
# CORRELATION INTERPRETATION TESTS
# ============================================================================

def test_generate_correlation_interpretation_success(service):
    """Test correlation interpretation"""
    correlation_data = {
        "request_type": "Correlation_Interpretation",
        "source_kpi": "RRC_conn_stp_SR",
        "target_kpi": "Comp_Cont_based_RACH_stp_SR",
        "correlation_score": -0.95,
        "correlation_method": "Pearson"
    }
    
    with patch.object(service, '_call_ollama') as mock_ollama:
        mock_ollama.return_value = """
The strong negative correlation (-0.95) indicates an inverse relationship between 
RRC connection success and RACH setup success rates. When one improves, the other 
tends to degrade, typical of capacity-constrained scenarios.
"""
        
        response = service.generate_correlation_interpretation(correlation_data)
        
        assert response.request_type == "Correlation_Interpretation"
        assert len(response.analysis) > 0


# ============================================================================
# STREAMING TESTS
# ============================================================================

def test_stream_llama_response_success(service):
    """Test token streaming"""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = [
            b'{"response": "The "}',
            b'{"response": "anomaly "}',
            b'{"response": "indicates..."}',
            b'{"done": true}'
        ]
        mock_post.return_value = mock_response
        
        tokens = list(service.stream_llama_response("Test prompt"))
        
        assert len(tokens) == 3
        # Correctly assert the first element, not the whole list
        assert tokens[0] == "The "
        assert "anomaly" in tokens[1]


def test_stream_llama_response_error(service):
    """Test streaming error handling"""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = Exception("Connection error")
        
        tokens = list(service.stream_llama_response("Test prompt"))
        
        assert len(tokens) == 1
        # Check if the string "Error" is INSIDE the first token
        assert "Error" in tokens[0]


# ============================================================================
# RECOMMENDATION EXTRACTION TESTS
# ============================================================================

def test_extract_recommendations_numbered(service):
    """Test extraction of numbered recommendations"""
    text = """
Analysis text here.

1. First recommendation
2. Second recommendation
3. Third recommendation
"""
    
    recommendations = service._extract_recommendations(text)
    
    assert len(recommendations) == 3
    # Look for partial match in the first recommendation
    assert "First" in recommendations[0]
    assert "Second" in recommendations[1]
    assert "Third" in recommendations[2]


def test_extract_recommendations_sentence_fallback(service):
    """Test fallback to sentences when no numbered items"""
    text = "First sentence. Second sentence. Third sentence."
    
    recommendations = service._extract_recommendations(text)
    
    assert len(recommendations) >= 2


# ============================================================================
# PROMPT FORMATTING TESTS
# ============================================================================

def test_format_causal_prompt(service, anomaly_data):
    """Test causal prompt formatting"""
    prompt = service._format_causal_prompt(anomaly_data)
    
    assert "RACH_Stp_att" in prompt
    # Expected formatting includes the comma
    assert "227,799" in prompt
    assert "2024-03-15" in prompt
    assert "0.92" in prompt  # correlation score


def test_format_scenario_prompt(service, scenario_data):
    """Test scenario prompt formatting"""
    prompt = service._format_scenario_prompt(scenario_data)
    
    assert "E-RAB_Setup_SR" in prompt
    assert "98.5" in prompt
    assert "90.1" in prompt
    assert "95.0" in prompt


def test_format_correlation_prompt(service):
    """Test correlation prompt formatting"""
    corr_data = {
        "source_kpi": "KPI_A",
        "target_kpi": "KPI_B",
        "correlation_score": 0.75,
        "correlation_method": "Pearson"
    }
    
    prompt = service._format_correlation_prompt(corr_data)
    
    assert "KPI_A" in prompt
    assert "KPI_B" in prompt
    assert "0.750" in prompt


# ============================================================================
# RESPONSE TIME TESTS
# ============================================================================

def test_processing_time_tracked(service, anomaly_data):
    """Test that processing time is accurately tracked"""
    with patch.object(service, '_call_ollama') as mock_ollama:
        # Mock time.time with extra values to cover internal logging calls
        # 1. start_time = 100.0
        # 2. end_time calculation = 100.5
        # 3. logger timestamp = 100.6 (and extra buffer just in case)
        with patch('time.time', side_effect=[100.0, 100.5, 100.6, 100.7]):
            mock_ollama.return_value = "Analysis text"
            
            response = service.generate_causal_analysis(anomaly_data)
            
            # 100.5 - 100.0 = 0.5s = 500ms
            assert response.processing_time_ms == 500.0


# ============================================================================
# TIMEOUT TESTS
# ============================================================================

def test_timeout_handling(service, anomaly_data):
    """Test timeout error handling"""
    with patch.object(service, '_call_ollama') as mock_ollama:
        mock_ollama.side_effect = OllamaTimeoutError("Timeout")
        
        # Pass valid data so the pre-processing doesn't fail
        response = service.generate_causal_analysis(anomaly_data)
        
        assert response.error is not None
        assert "Timeout" in str(response.error)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])