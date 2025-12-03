"""
EXAMPLE USAGE: ANOMALY DETECTION ENGINE
========================================

This file demonstrates how to use the AnomalyDetectionEngine with
real telecom data (Sample_KPI_Data.csv).

Usage:
    python example_usage.py
"""

import pandas as pd
import numpy as np
import json
from anomaly_detection import AnomalyDetectionEngine, detect_anomalies


# ============================================================================
# EXAMPLE 1: Basic Usage with Sample Data
# ============================================================================

def example_basic_usage():
    """Example 1: Load sample data and detect anomalies."""
    
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage with Sample Data")
    print("="*70)
    
    # Load sample data
    df = pd.read_csv('Sample_KPI_Data.csv', encoding='utf-8')
    print(f"✓ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  Columns: {df.columns.tolist()[:5]}... (truncated)")
    
    # Initialize engine
    engine = AnomalyDetectionEngine(window=7, zscore_threshold=3.0)
    print(f"✓ Engine initialized: window=7, threshold=3.0")
    
    # Generate report
    kpi_columns = ['RACH stp att', 'RRC conn stp SR', 'E-UTRAN E-RAB stp SR']
    report = engine.generate_report(
        df=df,
        time_column='TIME',
        kpi_columns=kpi_columns
    )
    
    print(f"\n✓ Analysis complete:")
    print(f"  - Total anomalies found: {report['total_anomalies']}")
    print(f"  - Processing time: {report['processing_time_ms']:.2f}ms")
    print(f"  - KPIs analyzed: {len(kpi_columns)}")
    
    # Display top anomalies
    if report['time_series_anomalies']:
        print(f"\n✓ Top 3 Anomalies:")
        for i, anomaly in enumerate(report['time_series_anomalies'][:3]):
            print(f"  {i+1}. {anomaly['kpi_name']} @ {anomaly['date_time']}")
            print(f"     Severity: {anomaly['severity']}, Z-Score: {anomaly['zscore']:.2f}")
            print(f"     Actual: {anomaly['actual_value']:.2f}, Range: {anomaly['expected_range']}")


# ============================================================================
# EXAMPLE 2: Time-Series Anomaly Detection Only
# ============================================================================

def example_timeseries_detection():
    """Example 2: Detect only time-series anomalies."""
    
    print("\n" + "="*70)
    print("EXAMPLE 2: Time-Series Anomaly Detection Only")
    print("="*70)
    
    # Create sample data with anomaly
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    kpi_values = np.random.normal(loc=100, scale=10, size=100)
    kpi_values[50] = 200  # Inject anomaly
    
    df = pd.DataFrame({
        'TIME': dates.strftime('%Y-%m-%d'),
        'KPI_TEST': kpi_values
    })
    print(f"✓ Created synthetic data with anomaly at index 50")
    
    # Detect anomalies
    engine = AnomalyDetectionEngine(window=7, zscore_threshold=3.0)
    anomalies = engine.detect_timeseries_anomalies(
        df=df,
        time_column='TIME',
        kpi_columns=['KPI_TEST']
    )
    
    print(f"✓ Detected {len(anomalies)} anomalies")
    for anomaly in anomalies:
        print(f"  - Date: {anomaly['date_time']}, Value: {anomaly['actual_value']:.2f}, "
              f"Z-Score: {anomaly['zscore']:.2f}, Severity: {anomaly['severity']}")


# ============================================================================
# EXAMPLE 3: IQR Outlier Detection
# ============================================================================

def example_iqr_detection():
    """Example 3: Detect distributional outliers using IQR."""
    
    print("\n" + "="*70)
    print("EXAMPLE 3: IQR Outlier Detection")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    kpi_values = np.random.normal(loc=1000, scale=100, size=200)
    kpi_values[10] = 3000  # Extreme outlier
    kpi_values[50] = 100   # Extreme outlier
    
    df = pd.DataFrame({
        'KPI_METRIC': kpi_values
    })
    
    # Detect outliers
    engine = AnomalyDetectionEngine()
    outliers = engine.detect_distributional_outliers(
        df=df,
        kpi_columns=['KPI_METRIC']
    )
    
    stats = outliers['KPI_METRIC']
    print(f"✓ Outlier statistics for KPI_METRIC:")
    print(f"  - Q1: {stats['q1']:.2f}")
    print(f"  - Q3: {stats['q3']:.2f}")
    print(f"  - IQR: {stats['iqr']:.2f}")
    print(f"  - Bounds: [{stats['lower_bound']:.2f}, {stats['upper_bound']:.2f}]")
    print(f"  - Outlier count: {stats['outlier_count']}")
    print(f"  - Outlier indices: {stats['outlier_indices'][:5]}... (first 5)")


# ============================================================================
# EXAMPLE 4: Box Plot Data Generation
# ============================================================================

def example_boxplot_generation():
    """Example 4: Generate box plot data for visualization."""
    
    print("\n" + "="*70)
    print("EXAMPLE 4: Box Plot Data Generation (Plotly Compatible)")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'KPI_A': np.random.normal(100, 10, 200),
        'KPI_B': np.random.normal(500, 50, 200)
    })
    
    engine = AnomalyDetectionEngine()
    
    # Generate box plot data
    for kpi_col in ['KPI_A', 'KPI_B']:
        boxplot_data = engine.generate_boxplot_data(df, kpi_col)
        print(f"\n✓ Box plot data for {kpi_col}:")
        print(f"  - Type: {boxplot_data['type']}")
        print(f"  - Data points: {len(boxplot_data['y'])}")
        print(f"  - Range: [{min(boxplot_data['y']):.2f}, {max(boxplot_data['y']):.2f}]")
        
        # Show how to use with Plotly
        print(f"  - Plotly usage:")
        print(f"    import plotly.graph_objects as go")
        print(f"    fig = go.Figure(data=[boxplot_data])")
        print(f"    fig.show()")


# ============================================================================
# EXAMPLE 5: Convenience Function (Stateless)
# ============================================================================

def example_convenience_function():
    """Example 5: Using convenience function for one-off analysis."""
    
    print("\n" + "="*70)
    print("EXAMPLE 5: Convenience Function (Stateless)")
    print("="*70)
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'TIME': pd.date_range('2024-01-01', periods=50, freq='D').strftime('%Y-%m-%d'),
        'KPI': np.random.normal(100, 10, 50)
    })
    df.loc[25, 'KPI'] = 200  # Inject anomaly
    
    # Use convenience function (creates engine internally)
    report = detect_anomalies(
        df=df,
        time_column='TIME',
        kpi_columns=['KPI'],
        window=7,
        zscore_threshold=3.0
    )
    
    print(f"✓ Analysis complete:")
    print(f"  - Anomalies: {report['total_anomalies']}")
    print(f"  - Time: {report['processing_time_ms']:.2f}ms")


# ============================================================================
# EXAMPLE 6: Multiple KPI Analysis
# ============================================================================

def example_multiple_kpis():
    """Example 6: Analyze multiple KPIs simultaneously."""
    
    print("\n" + "="*70)
    print("EXAMPLE 6: Multiple KPI Analysis")
    print("="*70)
    
    # Create realistic telecom data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    df = pd.DataFrame({
        'TIME': dates.strftime('%Y-%m-%d'),
        'REGION': ['N1'] * 100,
        'RACH_ATTEMPTS': np.random.normal(50000, 5000, 100),
        'RRC_SUCCESS_RATE': np.random.normal(99.5, 0.5, 100),
        'E_RAB_SUCCESS_RATE': np.random.normal(99.0, 1.0, 100),
        'PRACH_COMPLETIONS': np.random.normal(1000, 100, 100)
    })
    
    # Inject anomalies
    df.loc[30, 'RACH_ATTEMPTS'] = 150000
    df.loc[60, 'RRC_SUCCESS_RATE'] = 85.0
    
    # Analyze all KPIs
    engine = AnomalyDetectionEngine()
    kpi_list = ['RACH_ATTEMPTS', 'RRC_SUCCESS_RATE', 'E_RAB_SUCCESS_RATE', 'PRACH_COMPLETIONS']
    
    report = engine.generate_report(
        df=df,
        time_column='TIME',
        kpi_columns=kpi_list
    )
    
    print(f"✓ Multi-KPI Analysis Result:")
    print(f"  - KPIs analyzed: {len(kpi_list)}")
    print(f"  - Total anomalies: {report['total_anomalies']}")
    print(f"  - Processing time: {report['processing_time_ms']:.2f}ms")
    
    # Anomalies by KPI
    anomaly_counts = {}
    for anomaly in report['time_series_anomalies']:
        kpi = anomaly['kpi_name']
        anomaly_counts[kpi] = anomaly_counts.get(kpi, 0) + 1
    
    print(f"\n  Anomalies by KPI:")
    for kpi, count in anomaly_counts.items():
        print(f"    - {kpi}: {count}")


# ============================================================================
# EXAMPLE 7: Error Handling
# ============================================================================

def example_error_handling():
    """Example 7: Demonstrate error handling."""
    
    print("\n" + "="*70)
    print("EXAMPLE 7: Error Handling")
    print("="*70)
    
    engine = AnomalyDetectionEngine()
    
    # Create test data
    df = pd.DataFrame({
        'TIME': ['2024-01-01', '2024-01-02'],
        'KPI': [100, 101]
    })
    
    # Error 1: Missing time column
    print(f"\n✓ Error case 1: Missing time column")
    try:
        engine.detect_timeseries_anomalies(
            df=df,
            time_column='NONEXISTENT',
            kpi_columns=['KPI']
        )
    except ValueError as e:
        print(f"  Caught: {e}")
    
    # Error 2: Missing KPI column
    print(f"\n✓ Error case 2: Missing KPI column")
    try:
        engine.detect_timeseries_anomalies(
            df=df,
            time_column='TIME',
            kpi_columns=['NONEXISTENT_KPI']
        )
    except ValueError as e:
        print(f"  Caught: {e}")
    
    # Error 3: Invalid window size
    print(f"\n✓ Error case 3: Invalid window size")
    try:
        AnomalyDetectionEngine(window=0)
    except ValueError as e:
        print(f"  Caught: {e}")


# ============================================================================
# EXAMPLE 8: Output Serialization for LLM
# ============================================================================

def example_llm_output():
    """Example 8: Prepare output for LLM service."""
    
    print("\n" + "="*70)
    print("EXAMPLE 8: Output Serialization for LLM Service")
    print("="*70)
    
    # Generate report
    np.random.seed(42)
    df = pd.DataFrame({
        'TIME': pd.date_range('2024-01-01', periods=50, freq='D').strftime('%Y-%m-%d'),
        'KPI': np.random.normal(100, 10, 50)
    })
    
    engine = AnomalyDetectionEngine()
    report = engine.generate_report(
        df=df,
        time_column='TIME',
        kpi_columns=['KPI']
    )
    
    # Convert to JSON (for LLM service)
    report_json = json.dumps(report, indent=2)
    print(f"✓ Report serialized to JSON ({len(report_json)} characters)")
    
    # Show structure
    print(f"\n✓ Report structure:")
    print(f"  - time_series_anomalies: {len(report['time_series_anomalies'])} items")
    print(f"  - distributional_outliers: {len(report['distributional_outliers'])} KPIs")
    print(f"  - total_anomalies: {report['total_anomalies']}")
    print(f"  - processing_time_ms: {report['processing_time_ms']:.2f}ms")
    
    # Show first few items
    if report['time_series_anomalies']:
        print(f"\n✓ Sample anomaly (for LLM input):")
        print(json.dumps(report['time_series_anomalies'][0], indent=2))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("ANOMALY DETECTION ENGINE - USAGE EXAMPLES")
    print("="*70)
    
    try:
        # Run all examples
        example_basic_usage()
        example_timeseries_detection()
        example_iqr_detection()
        example_boxplot_generation()
        example_convenience_function()
        example_multiple_kpis()
        example_error_handling()
        example_llm_output()
        
        print("\n" + "="*70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
        
    except FileNotFoundError as e:
        print(f"\n⚠ Note: Sample_KPI_Data.csv not found for Example 1")
        print(f"  Other examples use synthetic data and work without it.\n")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
