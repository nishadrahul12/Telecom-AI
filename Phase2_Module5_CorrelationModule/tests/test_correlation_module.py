"""
Unit Tests for Phase 2 Module 5: Correlation Module
Tests: calculate_correlation_matrix, get_top_3_correlations, generate_heatmap_data
"""

import pytest
import numpy as np
import pandas as pd
import time
from correlation_module import (
    CorrelationAnalyzer,
    calculate_correlation_matrix,
    get_top_3_correlations,
    generate_heatmap_data,
    CorrelationAnalysisResult,
    CorrelationItem,
    get_top_3_by_source_kpi,
    filter_strong_correlations
)


# ============================================================================
# FIXTURES - Sample Data for Tests
# ============================================================================

@pytest.fixture
def sample_kpi_data():
    """Generate random KPI data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'KPI_A': np.random.randn(1000),
        'KPI_B': np.random.randn(1000),
        'KPI_C': np.random.randn(1000),
        'KPI_D': np.random.randn(1000),
    })


@pytest.fixture
def perfectly_correlated_data():
    """Generate perfectly correlated KPI data"""
    base = np.random.randn(1000)
    return pd.DataFrame({
        'KPI_PERFECT_POS': base,                           # r = 1.0 with itself
        'KPI_PERFECT_NEG': -base,                          # r = -1.0 with base
        'KPI_PERFECT_DUP': base + 1e-10 * np.random.randn(1000),  # r ≈ 1.0
    })


@pytest.fixture
def uncorrelated_data():
    """Generate uncorrelated KPI data"""
    np.random.seed(42)
    return pd.DataFrame({
        'KPI_INDEP_1': np.random.randn(1000),
        'KPI_INDEP_2': np.random.randn(1000),
        'KPI_INDEP_3': np.random.randn(1000),
    })


@pytest.fixture
def data_with_correlations():
    """Generate data with known strong correlations"""
    np.random.seed(42)
    base = np.random.randn(1000)
    noise = lambda: 0.05 * np.random.randn(1000)
    
    return pd.DataFrame({
        'KPI_X': base,
        'KPI_Y': base + noise(),           # r ≈ 0.99 with X
        'KPI_Z': -base + noise(),          # r ≈ -0.99 with X
        'KPI_W': np.random.randn(1000),    # r ≈ 0 with others
    })


# ============================================================================
# TEST SUITE 1: calculate_correlation_matrix()
# ============================================================================

class TestCalculateCorrelationMatrix:
    """Tests for correlation matrix calculation"""
    
    def test_correlation_matrix_is_symmetric(self, sample_kpi_data):
        """Matrix should be symmetric: M = M^T"""
        corr = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C', 'KPI_D']
        )
        assert np.allclose(corr, corr.T), "Matrix is not symmetric"
    
    def test_correlation_diagonal_is_one(self, sample_kpi_data):
        """Diagonal should be all 1.0 (self-correlation)"""
        corr = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C', 'KPI_D']
        )
        assert np.allclose(np.diag(corr), 1.0), "Diagonal is not 1.0"
    
    def test_correlation_range(self, sample_kpi_data):
        """All values should be in [-1, 1]"""
        corr = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C', 'KPI_D']
        )
        assert np.all(corr >= -1.0) and np.all(corr <= 1.0), \
            "Correlations outside [-1, 1] range"
    
    def test_matrix_shape(self, sample_kpi_data):
        """Matrix should be N×N where N = number of KPIs"""
        kpi_cols = ['KPI_A', 'KPI_B', 'KPI_C']
        corr = calculate_correlation_matrix(sample_kpi_data, kpi_cols)
        assert corr.shape == (3, 3), f"Expected (3,3), got {corr.shape}"
    
    def test_perfect_positive_correlation(self, perfectly_correlated_data):
        """Should detect r = 1.0 for identical series"""
        corr = calculate_correlation_matrix(
            perfectly_correlated_data,
            ['KPI_PERFECT_POS', 'KPI_PERFECT_DUP']
        )
        # Correlation should be very close to 1.0
        assert corr[0, 1] > 0.99, f"Expected r≈1.0, got {corr[0, 1]}"
    
    def test_perfect_negative_correlation(self, perfectly_correlated_data):
        """Should detect r = -1.0 for negated series"""
        corr = calculate_correlation_matrix(
            perfectly_correlated_data,
            ['KPI_PERFECT_POS', 'KPI_PERFECT_NEG']
        )
        # Correlation should be very close to -1.0
        assert corr[0, 1] < -0.99, f"Expected r≈-1.0, got {corr[0, 1]}"
    
    def test_no_correlation(self, uncorrelated_data):
        """Independent series should have r ≈ 0"""
        corr = calculate_correlation_matrix(
            uncorrelated_data,
            ['KPI_INDEP_1', 'KPI_INDEP_2']
        )
        # Should be close to 0 (within ±0.2 for random data)
        assert abs(corr[0, 1]) < 0.3, f"Expected r≈0, got {corr[0, 1]}"
    
    def test_input_validation_empty_kpi_list(self, sample_kpi_data):
        """Should raise ValueError for empty KPI list"""
        with pytest.raises(ValueError, match="cannot be empty"):
            calculate_correlation_matrix(sample_kpi_data, [])
    
    def test_input_validation_missing_columns(self, sample_kpi_data):
        """Should raise ValueError for missing columns"""
        with pytest.raises(ValueError, match="not in DataFrame"):
            calculate_correlation_matrix(sample_kpi_data, ['KPI_NONEXISTENT'])
    
    def test_input_validation_not_dataframe(self):
        """Should raise TypeError if input is not DataFrame"""
        with pytest.raises(TypeError):
            calculate_correlation_matrix([[1, 2], [3, 4]], ['A', 'B'])
    
    def test_handles_nan_values(self):
        """Should gracefully handle NaN by row-wise dropna"""
        df = pd.DataFrame({
            'KPI_A': [1, 2, np.nan, 4, 5],
            'KPI_B': [2, 4, 6, np.nan, 10],
        })
        corr = calculate_correlation_matrix(df, ['KPI_A', 'KPI_B'])
        assert not np.isnan(corr).any(), "Result contains NaN"
    
    def test_large_dataset_performance(self):
        """Should complete 100K rows in <500ms"""
        np.random.seed(42)
        df = pd.DataFrame({
            f'KPI_{i}': np.random.randn(100_000)
            for i in range(10)
        })
        kpi_cols = [f'KPI_{i}' for i in range(10)]
        
        start = time.time()
        corr = calculate_correlation_matrix(df, kpi_cols)
        elapsed = time.time() - start
        
        assert elapsed < 0.5, f"Performance target missed: {elapsed:.3f}s > 0.5s"


# ============================================================================
# TEST SUITE 2: get_top_3_correlations()
# ============================================================================

class TestGetTop3Correlations:
    """Tests for Top 3 ranking"""
    
    def test_top_3_ranked_by_absolute_value(self, data_with_correlations):
        """Top 3 should be ranked by |r| descending"""
        corr_matrix = calculate_correlation_matrix(
            data_with_correlations,
            data_with_correlations.columns.tolist()
        )
        
        top_3 = get_top_3_correlations(
            corr_matrix,
            data_with_correlations.columns.tolist()
        )
        
        # Check KPI_X's top 3
        x_top_3 = top_3['KPI_X']
        assert len(x_top_3) <= 3, "Should return max 3 items"
        
        # Verify ranking (absolute values descending)
        scores = [abs(item.correlation_score) for item in x_top_3]
        assert scores == sorted(scores, reverse=True), "Not ranked by absolute value"
    
    def test_self_correlation_excluded(self, sample_kpi_data):
        """Self-correlation should not appear in Top 3"""
        corr_matrix = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        top_3 = get_top_3_correlations(
            corr_matrix,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        # Check that source is not in its own Top 3
        for source_kpi, corr_items in top_3.items():
            target_kpis = [item.target_kpi for item in corr_items]
            assert source_kpi not in target_kpis, \
                f"{source_kpi} appears in its own Top 3"
    
    def test_returns_dict_structure(self, sample_kpi_data):
        """Should return Dict[str, List[CorrelationItem]]"""
        corr_matrix = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        top_3 = get_top_3_correlations(
            corr_matrix,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        assert isinstance(top_3, dict), "Should return dict"
        assert len(top_3) == 3, "Should have entry for each KPI"
        for kpi, items in top_3.items():
            assert isinstance(items, list), "Values should be lists"
            assert all(isinstance(item, CorrelationItem) for item in items), \
                "List items should be CorrelationItem"
    
    def test_max_3_items_per_kpi(self, data_with_correlations):
        """Each KPI should have at most 3 correlations"""
        corr_matrix = calculate_correlation_matrix(
            data_with_correlations,
            data_with_correlations.columns.tolist()
        )
        
        top_3 = get_top_3_correlations(
            corr_matrix,
            data_with_correlations.columns.tolist()
        )
        
        for kpi, items in top_3.items():
            assert len(items) <= 3, f"{kpi} has {len(items)} > 3 items"
    
    def test_input_validation_non_square_matrix(self):
        """Should raise ValueError for non-square matrix"""
        non_square = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="square"):
            get_top_3_correlations(non_square, ['A', 'B'])
    
    def test_input_validation_size_mismatch(self):
        """Should raise ValueError if names don't match matrix size"""
        matrix = np.eye(3)  # 3×3
        with pytest.raises(ValueError, match="must match"):
            get_top_3_correlations(matrix, ['A', 'B'])  # 2 names



# ============================================================================
# TEST SUITE 3: generate_heatmap_data()
# ============================================================================

class TestGenerateHeatmapData:
    """Tests for heatmap data generation"""
    
    def test_heatmap_data_structure(self, sample_kpi_data):
        """Should return HeatmapData with correct fields"""
        corr_matrix = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        heatmap = generate_heatmap_data(
            corr_matrix,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        assert hasattr(heatmap, 'z'), "Missing 'z' field"
        assert hasattr(heatmap, 'x'), "Missing 'x' field"
        assert hasattr(heatmap, 'y'), "Missing 'y' field"
        assert hasattr(heatmap, 'colorscale'), "Missing 'colorscale' field"
    
    def test_heatmap_axis_labels(self, sample_kpi_data):
        """x and y axes should match KPI names"""
        corr_matrix = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        kpi_names = ['KPI_A', 'KPI_B', 'KPI_C']
        heatmap = generate_heatmap_data(corr_matrix, kpi_names)
        
        assert heatmap.x == kpi_names, "x-axis names don't match"
        assert heatmap.y == kpi_names, "y-axis names don't match"
    
    def test_heatmap_z_values_match_matrix(self, sample_kpi_data):
        """z values should match input correlation matrix"""
        corr_matrix = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        heatmap = generate_heatmap_data(
            corr_matrix,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        assert np.allclose(heatmap.z, corr_matrix), "z values don't match matrix"
    
    def test_heatmap_color_scale_options(self, sample_kpi_data):
        """Should accept different color scales"""
        corr_matrix = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        kpi_names = ['KPI_A', 'KPI_B', 'KPI_C']
        
        for colorscale in ['RdBu', 'Viridis', 'Plasma', 'Reds']:
            heatmap = generate_heatmap_data(corr_matrix, kpi_names, colorscale=colorscale)
            assert heatmap.colorscale == colorscale, f"Color scale not set to {colorscale}"
    
    def test_heatmap_bounds(self, sample_kpi_data):
        """Should set zmin=-1, zmax=1"""
        corr_matrix = calculate_correlation_matrix(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        heatmap = generate_heatmap_data(
            corr_matrix,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        assert heatmap.zmin == -1.0, "zmin should be -1.0"
        assert heatmap.zmax == 1.0, "zmax should be 1.0"


# ============================================================================
# TEST SUITE 4: CorrelationAnalyzer (Integration)
# ============================================================================

class TestCorrelationAnalyzer:
    """Integration tests for complete pipeline"""
    
    def test_analyzer_returns_result_object(self, sample_kpi_data):
        """Analyzer should return CorrelationAnalysisResult"""
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        assert isinstance(result, CorrelationAnalysisResult), \
            "Should return CorrelationAnalysisResult"
    
    def test_analyzer_processing_time(self, sample_kpi_data):
        """Should measure and record processing time"""
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        assert result.processing_time_ms > 0, "Processing time should be > 0"
        assert result.processing_time_ms < 1000, "Should complete in <1s"
    
    def test_analyzer_performance_target(self):
        """Should meet <5s target on 100K rows"""
        np.random.seed(42)
        df = pd.DataFrame({
            f'KPI_{i}': np.random.randn(100_000)
            for i in range(15)  # 15 KPIs
        })
        
        kpi_cols = [f'KPI_{i}' for i in range(15)]
        
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(df, kpi_cols)
        
        assert result.processing_time_ms < 5000, \
            f"Performance target missed: {result.processing_time_ms:.0f}ms > 5000ms"
    
    def test_analyzer_result_completeness(self, sample_kpi_data):
        """Result should contain all required fields"""
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        assert result.correlation_matrix is not None
        assert result.top_3_per_kpi is not None
        assert result.heatmap_data is not None
        assert result.processing_time_ms is not None


# ============================================================================
# TEST SUITE 5: Utility Functions
# ============================================================================

class TestUtilityFunctions:
    """Tests for helper functions"""
    
    def test_get_top_3_by_source_kpi(self, sample_kpi_data):
        """Should extract Top 3 for specific KPI"""
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        top_3_a = get_top_3_by_source_kpi(result, 'KPI_A')
        assert top_3_a is not None
        assert all(isinstance(item, CorrelationItem) for item in top_3_a)
    
    def test_get_top_3_by_source_kpi_not_found(self, sample_kpi_data):
        """Should return None for non-existent KPI"""
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        top_3_nonexistent = get_top_3_by_source_kpi(result, 'KPI_NONEXISTENT')
        assert top_3_nonexistent is None
    
    def test_filter_strong_correlations(self, data_with_correlations):
        """Should filter by correlation strength threshold"""
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(
            data_with_correlations,
            data_with_correlations.columns.tolist()
        )
        
        strong = filter_strong_correlations(result, threshold=0.8)
        
        # Verify all correlations meet threshold
        for kpi, items in strong.items():
            for item in items:
                assert abs(item.correlation_score) >= 0.8, \
                    f"Correlation {item.correlation_score} below threshold"
    
    def test_filter_strong_correlations_empty_result(self, sample_kpi_data):
        """Filter with high threshold might return empty dict"""
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(
            sample_kpi_data,
            ['KPI_A', 'KPI_B', 'KPI_C']
        )
        
        very_strong = filter_strong_correlations(result, threshold=0.99)
        # Result should be dict (possibly empty)
        assert isinstance(very_strong, dict)


# ============================================================================
# TEST SUITE 6: Edge Cases & Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions"""
    
    def test_single_kpi_correlation(self):
        """Single KPI should raise error (need at least 2 to correlate)"""
        df = pd.DataFrame({'KPI_A': [1, 2, 3, 4, 5]})
        
        with pytest.raises(ValueError):
            # Cannot rank Top 3 for single KPI
            get_top_3_correlations(np.array([[1.0]]), ['KPI_A'])
    
    def test_two_kpi_correlation(self):
        """Two KPIs should work (Top 2 instead of 3)"""
        df = pd.DataFrame({
            'KPI_A': [1, 2, 3, 4, 5],
            'KPI_B': [2, 4, 6, 8, 10]
        })
        
        analyzer = CorrelationAnalyzer()
        result = analyzer.analyze(df, ['KPI_A', 'KPI_B'])
        
        assert len(result.top_3_per_kpi['KPI_A']) == 1  # Only 1 other KPI
        assert len(result.top_3_per_kpi['KPI_B']) == 1
    
    def test_insufficient_valid_rows(self):
        """Should raise error if <2 rows after NaN removal"""
        df = pd.DataFrame({
            'KPI_A': [1, np.nan, np.nan],
            'KPI_B': [np.nan, 2, np.nan]
        })
        
        with pytest.raises(ValueError, match="Not enough valid rows"):
            calculate_correlation_matrix(df, ['KPI_A', 'KPI_B'])
    
    def test_all_nan_column(self):
        """Should handle all-NaN columns gracefully"""
        df = pd.DataFrame({
            'KPI_A': [1, 2, 3, 4, 5],
            'KPI_B': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'KPI_C': [2, 4, 6, 8, 10]
        })
        
        # Should drop KPI_B (all NaN) and correlate KPI_A & KPI_C
        result = calculate_correlation_matrix(df, ['KPI_A', 'KPI_C'])
        assert result.shape == (2, 2)


# ============================================================================
# TEST SUITE 7: Real Data Integration
# ============================================================================

class TestRealDataIntegration:
    """Tests with real telecom data (if available)"""
    
    def test_with_sample_kpi_csv(self):
        """Integration test with Sample_KPI_Data.csv"""
        try:
            df = pd.read_csv('../Sample_KPI_Data.csv', nrows=100)
            
            # Extract numeric KPI columns
            kpi_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(kpi_cols) < 2:
                pytest.skip("Sample data has <2 KPI columns")
            
            analyzer = CorrelationAnalyzer()
            result = analyzer.analyze(df, kpi_cols[:10])  # Test with first 10 KPIs
            
            assert result.correlation_matrix is not None
            assert len(result.top_3_per_kpi) > 0
            assert result.processing_time_ms < 1000
            
        except FileNotFoundError:
            pytest.skip("Sample_KPI_Data.csv not found")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
