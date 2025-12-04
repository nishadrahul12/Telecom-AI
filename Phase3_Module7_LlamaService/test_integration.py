# Create test script: test_integration.py

from llama_service import LlamaService

# Initialize
service = LlamaService(host="localhost", port=11434)

# Test 1: Connection
print("Test 1: Checking Ollama connection...")
if service.connect_to_ollama():
    print("✓ Connected to Ollama")
else:
    print("✗ Failed to connect")
    exit(1)

# Test 2: Causal Analysis
print("\nTest 2: Causal Analysis...")
anomaly_data = {
    "request_type": "Causal_Anomaly_Analysis",
    "target_anomaly": {
        "kpi_name": "RACH_Stp_att",
        "date_time": "2024-03-15",
        "actual_value": 89000,
        "expected_range": "45000 - 55000",
        "severity": "High",
        "zscore": 3.2
    },
    "contextual_data": [
        {
            "kpi_name": "E-UTRAN_avg_RRC_conn_UEs",
            "value_on_anomaly_date": 25.1,
            "correlation_score": 0.90,
            "historical_state": "Too High"
        }
    ]
}

response = service.generate_causal_analysis(anomaly_data)
print(f"Analysis: {response.analysis[:100]}...")
print(f"Confidence: {response.confidence_level}")
print(f"Processing Time: {response.processing_time_ms}ms")
