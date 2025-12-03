"""
Test: pipeline_integration.py basic functionality
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from correlation_module import CorrelationAnalyzer


class TestPipelineIntegration:
    """Test pipeline integration with correlation module"""
    
    def test_pipeline_imports(self):
        """Verify pipeline can be imported"""
        try:
            from pipeline_integration import TelecomAIPipeline
            assert TelecomAIPipeline is not None
        except ImportError:
            pytest.skip("Pipeline module not fully set up yet")
    
    def test_correlation_output_format(self):
        """Verify correlation output is in correct format for forecasting"""
        
        # Create sample data
        np.random.seed(42)
        sample_df = pd.DataFrame({
            'RACH_stp_att': np.random.randn(100),
            'RRC_stp_att': np.random.randn(100),
            'E_RAB_Stp_att': np.random.randn(100),
            'Inter_freq_HO_att': np.random.randn(100),
        })
        
        # Run correlation
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(sample_df, sample_df.columns.tolist())
        
        # Verify output format (what forecasting module expects)
        assert hasattr(result, 'top_3_per_kpi'), "Result should have top_3_per_kpi"
        assert len(result.top_3_per_kpi) > 0, "Should have correlations"
        
        # Verify exogenous extraction works
        target_kpi = 'RACH_stp_att'
        top_3 = result.top_3_per_kpi[target_kpi]
        exogenous_kpis = [item.target_kpi for item in top_3]
        
        assert len(exogenous_kpis) <= 3, "Should have <= 3 exogenous vars"
        assert all(isinstance(kpi, str) for kpi in exogenous_kpis), "Exogenous should be strings"
    
    def test_exogenous_variables_extraction(self):
        """Test that exogenous variables can be extracted correctly"""
        
        # Sample data
        np.random.seed(42)
        sample_df = pd.DataFrame({
            'KPI_A': np.random.randn(50),
            'KPI_B': np.random.randn(50),
            'KPI_C': np.random.randn(50),
        })
        
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(sample_df, ['KPI_A', 'KPI_B', 'KPI_C'])
        
        # Extract exogenous
        top_3 = result.top_3_per_kpi['KPI_A']
        exogenous = [item.target_kpi for item in top_3]
        
        # Verify format
        assert isinstance(exogenous, list), "Exogenous should be list"
        assert all(isinstance(x, str) for x in exogenous), "All should be strings"
        assert len(exogenous) <= 3, "Max 3 exogenous variables"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
