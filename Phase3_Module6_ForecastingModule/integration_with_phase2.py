#!/usr/bin/env python
"""
Quick Integration Test: Phase 2 → Phase 3 (FIXED VERSION)

Simple script to verify the forecasting pipeline works
Fixes: Date parsing issue by dropping non-numeric columns before correlation
"""

import pandas as pd
import numpy as np
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*60)
    print("QUICK INTEGRATION TEST: PHASE 2 → PHASE 3")
    print("="*60)
    
    try:
        # Step 1: Load data
        print("\n📊 Loading data...")
        df = pd.read_csv('Sample_KPI_Data.csv')
        print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Step 2: Import forecasting
        print("\n📦 Importing forecasting module...")
        from forecasting_module import forecast_kpi
        print("✅ Imported successfully")
        
        # Step 3: Prepare data (FIX: Drop string columns first!)
        print("\n🧹 Preparing data...")
        df['time'] = pd.to_datetime(df['TIME'])
        
        # CRITICAL FIX: Drop non-numeric columns BEFORE correlation
        # Keep only numeric columns + the 'time' column
        numeric_df = df.select_dtypes(include=[np.number]).copy()
        numeric_df['time'] = df['time']  # Add time back
        
        # Also drop NaN values
        df_clean = numeric_df.dropna()
        print(f"✅ Data prepared: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        
        # Step 4: Find correlations manually (NOW WITH ONLY NUMERIC DATA)
        print("\n🔗 Finding correlated KPIs...")
        target_kpi = 'RACH stp att'
        
        # Use only numeric columns for correlation
        numeric_for_corr = df_clean.drop('time', axis=1)
        correlations = numeric_for_corr.corr()[target_kpi].abs()
        
        # Remove target itself
        correlations = correlations.drop(target_kpi, errors='ignore')
        
        # Get top N
        top_kpis = correlations.nlargest(3).index.tolist()
        print(f"✅ Top 3 correlated KPIs: {top_kpis}")
        
        # Step 5: Run forecast (ARIMA - univariate)
        print("\n📈 TEST 1: ARIMA Forecast (univariate)...")
        result_arima = forecast_kpi(
            df_clean, 'time', target_kpi, 
            forecast_horizon=7,
            exogenous_kpis=None  # Univariate
        )
        print(f"✅ ARIMA Complete:")
        print(f"   Model: {result_arima.model_type}")
        print(f"   Forecast: {len(result_arima.forecast_values)} days")
        print(f"   RMSE: {result_arima.model_metrics.rmse:.4f}")
        print(f"   Time: {result_arima.processing_time_ms:.1f}ms")
        
        # Step 6: Run forecast (ARIMAX - with exogenous)
        print("\n📈 TEST 2: ARIMAX Forecast (with 3 exogenous KPIs)...")
        result_arimax = forecast_kpi(
            df_clean, 'time', target_kpi,
            forecast_horizon=7,
            exogenous_kpis=top_kpis  # With exogenous
        )
        print(f"✅ ARIMAX Complete:")
        print(f"   Model: {result_arimax.model_type}")
        print(f"   Forecast: {len(result_arimax.forecast_values)} days")
        print(f"   RMSE: {result_arimax.model_metrics.rmse:.4f}")
        print(f"   Exogenous: {result_arimax.exogenous_variables_used}")
        print(f"   Time: {result_arimax.processing_time_ms:.1f}ms")
        
        # Step 7: Compare
        print("\n📊 Comparison:")
        print(f"   ARIMA RMSE:  {result_arima.model_metrics.rmse:.4f}")
        print(f"   ARIMAX RMSE: {result_arimax.model_metrics.rmse:.4f}")
        
        if result_arima.model_metrics.rmse > 0:
            improvement = ((result_arima.model_metrics.rmse - result_arimax.model_metrics.rmse) / result_arima.model_metrics.rmse * 100)
            print(f"   Improvement: {improvement:.1f}%")
        
        print("\n" + "="*60)
        print("✅ INTEGRATION TEST PASSED!")
        print("="*60)
        print("\nAll components working:")
        print("  ✓ Data loading")
        print("  ✓ Data preparation (numeric only)")
        print("  ✓ Forecasting module import")
        print("  ✓ ARIMA forecasting")
        print("  ✓ ARIMAX forecasting")
        print("  ✓ Metric calculation")
        print("\n🚀 Ready for Phase 4!\n")
        
        return 0
        
    except FileNotFoundError:
        print("\n❌ ERROR: Sample_KPI_Data.csv not found!")
        print("\nSolution:")
        print("  1. Find where your CSV is located")
        print("  2. Copy it to this folder:")
        print(f"     Copy-Item 'C:\\path\\to\\Sample_KPI_Data.csv' -Destination '.'")
        return 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
