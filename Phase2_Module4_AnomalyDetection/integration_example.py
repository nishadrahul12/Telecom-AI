"""
Integration with FilteringEngine (Phase 2, Module 3)
======================================================

This script demonstrates how the Anomaly Detection Engine
integrates with the output from the Filtering Engine and
prepares data for the LLM Service (Phase 3).
"""

import pandas as pd
import os
from pathlib import Path
from anomaly_detection import AnomalyDetectionEngine

# ============================================================================
# Helper Function: Locate data file
# ============================================================================

def get_data_file_path(filename='Sample_KPI_Data.csv'):
    """
    Locate the data file with fallback paths.
    
    Searches in:
    1. Current working directory
    2. Same directory as this script
    3. Parent directory
    """
    # Try current working directory first
    if os.path.exists(filename):
        return filename
    
    # Try same directory as script
    script_dir = Path(__file__).parent
    script_path = script_dir / filename
    if script_path.exists():
        return str(script_path)
    
    # Try parent directory
    parent_path = script_dir.parent / filename
    if parent_path.exists():
        return str(parent_path)
    
    # Try Phase2_Module4_AnomalyDetection folder
    module_path = script_dir / 'Phase2_Module4_AnomalyDetection' / filename
    if module_path.exists():
        return str(module_path)
    
    raise FileNotFoundError(
        f"\n{filename} not found.\n"
        f"Searched in:\n"
        f"  1. {os.path.abspath('.')}\n"
        f"  2. {script_dir}\n"
        f"  3. {parent_path}\n"
        f"\nPlease ensure {filename} is in one of these locations."
    )

# ============================================================================
# Get Filtered Data
# ============================================================================

def get_filtered_data():
    """
    Get filtered data from previous phase.
    
    In production, this would call:
        filtered_df = FilteringEngine.get_filtered_data()
    
    For now, we simulate by loading the CSV file.
    """
    data_file = get_data_file_path('Sample_KPI_Data.csv')
    print(f"Loading data from: {data_file}")
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df

# ============================================================================
# Main Integration Flow
# ============================================================================

def main():
    """Main integration example."""
    print("\n" + "="*70)
    print("PHASE 2: MODULE 4 - ANOMALY DETECTION ENGINE")
    print("Integration with Filtering Engine → LLM Service")
    print("="*70 + "\n")
    
    # Step 1: Initialize the anomaly detection engine
    print("Step 1: Initializing Anomaly Detection Engine...")
    engine = AnomalyDetectionEngine(window=7, zscore_threshold=3.0)
    print(f"  ✓ Engine initialized (window=7, threshold=3.0)\n")
    
    # Step 2: Get filtered data from previous module
    print("Step 2: Loading filtered data from previous module...")
    try:
        filtered_df = get_filtered_data()
    except FileNotFoundError as e:
        print(f"  ✗ Error: {e}\n")
        return
    
    print(f"  ✓ Data loaded: {len(filtered_df)} rows\n")
    
    # Step 3: Select KPI columns for analysis
    print("Step 3: Selecting KPI columns for analysis...")
    kpi_columns = [
        'RACH stp att',           # RACH Setup Attempts
        'RRC conn stp SR',        # RRC Connection Setup Success Rate
        'E-UTRAN E-RAB stp SR'    # E-RAB Setup Success Rate
    ]
    
    # Verify columns exist
    available_columns = [col for col in kpi_columns if col in filtered_df.columns]
    if not available_columns:
        print(f"  ⚠ Warning: None of the selected columns found in data")
        print(f"  Available columns: {filtered_df.columns.tolist()[:10]}...")
        kpi_columns = filtered_df.columns[7:]  # Use available numeric columns
    else:
        print(f"  ✓ Selected {len(available_columns)} KPI columns")
    
    print(f"  Using columns: {kpi_columns}\n")
    
    # Step 4: Generate comprehensive anomaly report
    print("Step 4: Generating anomaly report...")
    try:
        report = engine.generate_report(
            df=filtered_df,
            time_column='TIME',
            kpi_columns=kpi_columns
        )
    except Exception as e:
        print(f"  ✗ Error generating report: {e}\n")
        return
    
    print(f"  ✓ Report generated\n")
    
    # Step 5: Display results
    print("Step 5: Results Summary")
    print("-" * 70)
    print(f"  Total Anomalies Detected:    {report['total_anomalies']}")
    print(f"  Time-Series Anomalies:       {len(report['time_series_anomalies'])}")
    print(f"  Distributional Outliers:     {len(report['distributional_outliers'])}")
    print(f"  Processing Time:             {report['processing_time_ms']:.2f}ms")
    print("-" * 70 + "\n")
    
    # Step 6: Show sample anomalies
    if report['time_series_anomalies']:
        print("Sample Time-Series Anomalies (first 3):")
        for i, anomaly in enumerate(report['time_series_anomalies'][:3], 1):
            print(f"\n  Anomaly {i}:")
            print(f"    KPI:            {anomaly['kpi_name']}")
            print(f"    Date/Time:      {anomaly['date_time']}")
            print(f"    Actual Value:   {anomaly['actual_value']:.2f}")
            print(f"    Expected Range: {anomaly['expected_range']}")
            print(f"    Z-Score:        {anomaly['zscore']:.2f}")
            print(f"    Severity:       {anomaly['severity']}")
    else:
        print("No time-series anomalies detected in the data.\n")
    
    # Step 7: Show outlier statistics
    if report['distributional_outliers']:
        print("\nDistributional Outlier Statistics (first 2 KPIs):")
        for i, (kpi_name, stats) in enumerate(list(report['distributional_outliers'].items())[:2], 1):
            print(f"\n  KPI {i}: {kpi_name}")
            print(f"    Q1:             {stats['q1']:.2f}")
            print(f"    Q3:             {stats['q3']:.2f}")
            print(f"    IQR:            {stats['iqr']:.2f}")
            print(f"    Outlier Count:  {stats['outlier_count']}")
    
    # Step 8: Prepare for Phase 3 (LLM Service)
    print("\n" + "="*70)
    print("Step 6: Data Ready for Phase 3 (LLM Service)")
    print("="*70)
    print(f"\nReport structure (ready for LLM analysis):")
    print(f"  Keys: {list(report.keys())}")
    print(f"\nThis report can now be passed to the LLM Service for:")
    print(f"  - Root cause analysis")
    print(f"  - Anomaly correlation detection")
    print(f"  - Automated remediation suggestions")
    print(f"  - Network optimization recommendations\n")
    
    # Example of how to pass to LLM service (Phase 3)
    print("Phase 3 Integration Example:")
    print("""
    # from llm_service import analyze_telecom_anomalies
    # analysis = analyze_telecom_anomalies(report)
    # print(analysis['root_cause'])
    # print(analysis['recommendations'])
    """)
    
    print("="*70)
    print("✓ Integration example completed successfully!\n")

# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    main()
