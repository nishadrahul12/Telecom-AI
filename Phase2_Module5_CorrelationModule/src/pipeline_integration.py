"""
Phase 2 Module 5 ↔ Phase 3 Integration Pipeline

This module demonstrates:
1. Consuming correlation module output
2. Extracting exogenous variables
3. Preparing data for forecasting module
"""

import pandas as pd
import numpy as np
import os
import sys
from typing import List, Dict, Any

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import correlation module
from correlation_module import CorrelationAnalyzer, CorrelationAnalysisResult


class TelecomAIPipeline:
    """
    Pipeline orchestrating correlation analysis and forecasting preparation.
    """
    
    def __init__(self):
        self.corr_analyzer = CorrelationAnalyzer()
        self.last_result = None
    
    def run_pipeline(
        self,
        df: pd.DataFrame = None,
        filepath: str = None,
        target_kpi: str = 'RACH_stp_att'
    ) -> Dict[str, Any]:
        """
        Execute pipeline: Load → Analyze → Extract Exogenous Variables
        
        Args:
            df: DataFrame to use (if provided, filepath ignored)
            filepath: Path to CSV file (alternative to df)
            target_kpi: KPI to prepare forecasting for
        
        Returns:
            Dictionary with analysis results and exogenous variables
        """
        
        print("\n" + "=" * 70)
        print("TELECOM-AI: CORRELATION → FORECASTING PREPARATION")
        print("=" * 70)
        
        # Step 1: Load data
        print("\n📊 Step 1: Loading Data")
        print("-" * 70)
        data = self._load_data(df, filepath)
        
        # Step 2: Extract KPIs
        print("\n📊 Step 2: Extracting KPI Columns")
        print("-" * 70)
        kpi_columns = self._extract_kpi_columns(data)
        
        # Step 3: Run correlation analysis
        print("\n📊 Step 3: Running Correlation Analysis")
        print("-" * 70)
        correlation_result = self._run_correlation(data, kpi_columns)
        
        # Step 4: Extract exogenous variables
        print("\n📊 Step 4: Extracting Exogenous Variables")
        print("-" * 70)
        exogenous_kpis = self._extract_exogenous(correlation_result, target_kpi)
        
        # Compile results
        results = {
            'data': data,
            'kpi_columns': kpi_columns,
            'correlation_result': correlation_result,
            'target_kpi': target_kpi,
            'exogenous_kpis': exogenous_kpis,
            'ready_for_forecasting': True
        }
        
        self.last_result = results
        
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE - READY FOR FORECASTING")
        print("=" * 70)
        print(f"\nOutput Summary:")
        print(f"  ├─ Rows processed: {len(data)}")
        print(f"  ├─ KPIs analyzed: {len(kpi_columns)}")
        print(f"  ├─ Target KPI: {target_kpi}")
        print(f"  ├─ Exogenous variables: {exogenous_kpis}")
        print(f"  └─ Status: READY FOR PHASE 3 ✅")
        
        return results
    
    def _load_data(self, df: pd.DataFrame = None, filepath: str = None) -> pd.DataFrame:
        """Load data from DataFrame or CSV file"""
        
        if df is not None:
            print(f"✅ Using provided DataFrame: {df.shape}")
            return df.copy()
        
        if filepath:
            if os.path.exists(filepath):
                data = pd.read_csv(filepath)
                print(f"✅ Loaded CSV: {filepath}")
                print(f"   Shape: {data.shape}")
                return data
            else:
                raise FileNotFoundError(f"CSV not found: {filepath}")
        
        sample_path = self._find_sample_csv()
        if sample_path:
            data = pd.read_csv(sample_path)
            print(f"✅ Loaded sample data: {sample_path}")
            print(f"   Shape: {data.shape}")
            return data
        
        raise ValueError("No data provided and sample CSV not found")
    
    def _find_sample_csv(self) -> str:
        """Find Sample_KPI_Data.csv by searching upward"""
        
        current = os.path.dirname(os.path.abspath(__file__))
        
        for _ in range(5):
            sample_path = os.path.join(current, 'Sample_KPI_Data.csv')
            if os.path.exists(sample_path):
                return sample_path
            current = os.path.dirname(current)
        
        return None
    
    def _extract_kpi_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract numeric KPI columns and normalize names"""
        
        # Normalize column names: replace spaces and special chars with underscores
        df.columns = df.columns.str.replace(' ', '_', regex=False)
        df.columns = df.columns.str.replace(';', '', regex=False)
        df.columns = df.columns.str.replace('(', '', regex=False)
        df.columns = df.columns.str.replace(')', '', regex=False)
        df.columns = df.columns.str.replace('-', '_', regex=False)
        
        # Get numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove dimension columns (not KPIs)
        dimension_columns = ['POSTCODE', 'BAND_NUMBER', 'MRBTS_ID', 'LNBTS_ID', 'LNCEL_ID']
        kpi_columns = [col for col in numeric_columns if col not in dimension_columns]
        
        print(f"✅ Extracted {len(kpi_columns)} KPI columns (after normalization)")
        print(f"   Sample: {kpi_columns[:3]}")
        print(f"   Note: Column names normalized (spaces → underscores)")
        
        return kpi_columns
    
    def _run_correlation(
        self, 
        df: pd.DataFrame, 
        kpi_columns: List[str]
    ) -> CorrelationAnalysisResult:
        """Run correlation analysis"""
        
        result = self.corr_analyzer.analyze(df, kpi_columns)
        
        print(f"✅ Correlation analysis complete")
        print(f"   Processing time: {result.processing_time_ms:.2f}ms")
        print(f"   KPIs analyzed: {len(result.top_3_per_kpi)}")
        print(f"   Correlations found: {sum(len(v) for v in result.top_3_per_kpi.values())}")
        
        return result
    
    def _extract_exogenous(
        self, 
        correlation_result: CorrelationAnalysisResult, 
        target_kpi: str
    ) -> List[str]:
        """Extract exogenous variables for target KPI"""
        
        if target_kpi not in correlation_result.top_3_per_kpi:
            available = list(correlation_result.top_3_per_kpi.keys())[:5]
            raise ValueError(
                f"Target KPI '{target_kpi}' not in results.\n"
                f"Available KPIs: {available}..."
            )
        
        top_3 = correlation_result.top_3_per_kpi[target_kpi]
        exogenous_kpis = [item.target_kpi for item in top_3]
        
        print(f"✅ Exogenous variables for '{target_kpi}':")
        for i, item in enumerate(top_3, 1):
            print(f"   {i}. {item.target_kpi}: r = {item.correlation_score:.3f}")
        
        return exogenous_kpis


def demo_pipeline():
    """Demo: Run pipeline with sample data"""
    
    pipeline = TelecomAIPipeline()
    
    # Run with default sample data
    # Note: Column names are normalized automatically
    results = pipeline.run_pipeline(
        target_kpi='RACH_stp_att'  # Will match "RACH stp att" after normalization
    )
    
    return results


if __name__ == '__main__':
    
    try:
        results = demo_pipeline()
        
        print("\n" + "=" * 70)
        print("🚀 PIPELINE EXECUTION SUCCESSFUL")
        print("=" * 70)
        print(f"\nReady to pass to Phase 3 Forecasting Module:")
        print(f"  - Target: {results['target_kpi']}")
        print(f"  - Exogenous: {results['exogenous_kpis']}")
        print(f"  - Data shape: {results['data'].shape}")
        
    except Exception as e:
        print(f"\n❌ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()
