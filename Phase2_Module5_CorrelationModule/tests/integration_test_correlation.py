"""
Integration Test: Correlation Module with Real Data

Tests the complete pipeline using actual telecom KPI CSV data.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src/ to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from correlation_module import CorrelationAnalyzer


def find_sample_csv():
    """
    Find Sample_KPI_Data.csv by searching upward from test directory.
    
    File structure:
    Telecom-AI/
    ├── Sample_KPI_Data.csv
    └── Phase2_Module5_CorrelationModule/
        ├── tests/
        │   └── integration_test_correlation.py  ← We are here
    """
    current = os.path.dirname(os.path.abspath(__file__))
    
    # Search up to 5 levels up
    for _ in range(5):
        csv_path = os.path.join(current, 'Sample_KPI_Data.csv')
        if os.path.exists(csv_path):
            return csv_path
        current = os.path.dirname(current)
    
    # If not found, raise error with paths searched
    raise FileNotFoundError(
        f"Sample_KPI_Data.csv not found. "
        f"Started from: {os.path.dirname(os.path.abspath(__file__))}"
    )


def test_correlation_with_real_data():
    """Integration test with real telecom CSV data"""
    
    # Find CSV path (searches upward)
    csv_path = find_sample_csv()
    print(f"📁 Using CSV: {csv_path}")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} rows")
    
    # Extract KPI columns (numeric only)
    kpi_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"📈 Found {len(kpi_columns)} KPI columns")
    
    assert len(kpi_columns) >= 2, "Need at least 2 numeric columns"
    
    # Run analysis
    print("🔄 Running correlation analysis...")
    analyzer = CorrelationAnalyzer()
    result = analyzer.analyze(df=df, kpi_columns=kpi_columns)
    
    # Assertions
    assert len(result.top_3_per_kpi) > 0, "No correlations found"
    assert result.processing_time_ms < 5000, "Performance target exceeded"
    assert result.heatmap_data is not None, "Heatmap data missing"
    
    # Print results
    print(f"\n✅ Integration test PASSED!")
    print(f"   - KPIs analyzed: {len(kpi_columns)}")
    print(f"   - Rows processed: {len(df)}")
    print(f"   - Processing time: {result.processing_time_ms:.2f}ms")
    print(f"   - Correlations found: {sum(len(v) for v in result.top_3_per_kpi.values())}")
    
    return result


if __name__ == '__main__':
    test_correlation_with_real_data()
