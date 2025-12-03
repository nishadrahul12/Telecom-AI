import time
import pandas as pd
import numpy as np
from src.correlation_module import CorrelationAnalyzer

def benchmark():
    """Verify <5s performance on 100K rows"""
    
    # Test Case 1: 10 KPIs, 100K rows
    print("Benchmarking: 10 KPIs × 100K rows")
    df = pd.DataFrame(np.random.randn(100_000, 10), 
                     columns=[f'KPI_{i}' for i in range(10)])
    
    start = time.time()
    analyzer = CorrelationAnalyzer()
    result = analyzer.analyze(df, kpi_columns=df.columns.tolist())
    elapsed = time.time() - start
    
    print(f"✅ Processed {len(df)} rows in {elapsed*1000:.0f}ms")
    assert elapsed < 5.0, f"Performance target missed: {elapsed}s > 5s"
    
    # Test Case 2: 50 KPIs, 100K rows (larger matrix)
    print("\nBenchmarking: 50 KPIs × 100K rows")
    df = pd.DataFrame(np.random.randn(100_000, 50), 
                     columns=[f'KPI_{i}' for i in range(50)])
    
    start = time.time()
    result = analyzer.analyze(df, kpi_columns=df.columns.tolist())
    elapsed = time.time() - start
    
    print(f"✅ Processed {len(df)} rows, {len(df.columns)} columns in {elapsed*1000:.0f}ms")
    assert elapsed < 5.0, f"Performance target missed: {elapsed}s > 5s"

if __name__ == '__main__':
    benchmark()
