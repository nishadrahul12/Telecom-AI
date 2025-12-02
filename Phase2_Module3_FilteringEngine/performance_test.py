"""
Performance test: Verify performance targets on various dataset sizes
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import time
from filtering_engine import apply_filters_and_sample, DataFrameMetadata, DataLevel

print("=" * 80)
print("PERFORMANCE TEST: Various Dataset Sizes")
print("=" * 80)

# Create synthetic datasets
def create_test_df(size):
    return pd.DataFrame({
        'TIME': pd.date_range('2024-01-01', periods=size, freq='H'),
        'REGION': np.random.choice(['N1', 'N2', 'N3'], size),
        'CARRIER': np.random.choice(['L700', 'L1800', 'L2100'], size),
        'METRIC1': np.random.random(size) * 100,
        'METRIC2': np.random.random(size) * 100,
        'METRIC3': np.random.random(size) * 100,
    })

metadata = DataFrameMetadata(
    text_dimensions=['REGION', 'CARRIER'],
    numeric_dimensions=[],
    kpi_columns=['METRIC1', 'METRIC2', 'METRIC3'],
    time_column='TIME',
    data_level='Cell'
)

test_cases = [
    ('Small (1k)', 1000),
    ('Medium (10k)', 10000),
    ('Large (100k)', 100000),
    ('XL (500k)', 500000),
]

print("\nPerformance Results:")
print("-" * 80)

for name, size in test_cases:
    df = create_test_df(size)
    
    start = time.time()
    result = apply_filters_and_sample(
        df, metadata, 'Cell',
        {'REGION': ['N1', 'N2']}
    )
    elapsed = time.time() - start
    
    target = 100 if size >= 500000 else 50
    status = "✅" if elapsed < target/1000 else "❌"
    
    print(f"{status} {name:20} | {elapsed*1000:7.2f}ms | Target: {target}ms")

print("-" * 80)
print("✅ PERFORMANCE TEST COMPLETE")
print("=" * 80)
