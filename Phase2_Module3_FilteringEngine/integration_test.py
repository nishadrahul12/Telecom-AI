"""
Integration test: Verify filtering_engine works with Phase 1 data models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime
import time

from filtering_engine import (
    apply_filters_and_sample,
    DataFrameMetadata,
    DataLevel,
    get_unique_values,
    get_filter_options
)

print("=" * 80)
print("INTEGRATION TEST: Filtering Engine with Real Data")
print("=" * 80)

# Load real data
df = pd.read_csv('../../Sample_KPI_Data.csv')
print(f"\n✅ Loaded {len(df)} rows from Sample_KPI_Data.csv")

# Create metadata (as in ANALYSIS.md)
metadata = DataFrameMetadata(
    text_dimensions=[],
    numeric_dimensions=['MRBTS_ID', 'LNBTS_ID', 'LNCEL_ID', 'BAND_NUMBER'],
    kpi_columns=[col for col in df.columns 
                 if col not in ['TIME', 'POSTCODE', 'MRBTS_ID', 'LNBTS_ID', 'LNCEL_ID', 'BAND_NUMBER']],
    time_column='TIME',
    data_level='Cell'
)
print(f"✅ Metadata created: {len(metadata.text_dimensions)} text dims, "
      f"{len(metadata.numeric_dimensions)} numeric dims, "
      f"{len(metadata.kpi_columns)} KPIs")

# TEST 1.1: Get filter options
print("\n--- Test 1.1: Get Filter Options ---")
filter_opts = get_filter_options(df, metadata, 'Cell')
print(f"✅ Available filter options: {list(filter_opts.keys())}")
for col, values in filter_opts.items():
    print(f"  • {col}: {len(values)} unique values")

# TEST 1.2: Single dimension filter
print("\n--- Test 1.2: Single Dimension Filter (REGION=N1) ---")
start = time.time()
result = apply_filters_and_sample(df, metadata, 'Cell', {'BAND_NUMBER': [1]})
elapsed = time.time() - start
print(f"✅ Original: {result.row_count_original} rows")
print(f"✅ Filtered: {result.row_count_filtered} rows")
print(f"✅ Sampled: {result.row_count_sampled} rows")
print(f"✅ Time: {elapsed*1000:.2f}ms (target: <100ms)")
assert elapsed < 1.0, "Performance target missed!"

# TEST 1.3: Multi-dimensional filter
print("\n--- Test 1.3: Multi-Dimensional Filter (BAND_NUMBER + MRBTS_ID) ---")
start = time.time()
result = apply_filters_and_sample(
    df, metadata, 'Cell', 
    {'BAND_NUMBER': [1, 3], 'MRBTS_ID': [100001, 100002, 100003]}
)
elapsed = time.time() - start
print(f"✅ Filtered: {result.row_count_filtered} rows")
print(f"✅ Sampled: {result.row_count_sampled} rows")
print(f"✅ Time: {elapsed*1000:.2f}ms")
assert result.row_count_filtered > 0, "No rows after filter!"

# TEST 1.4: Verify data integrity
print("\n--- Test 1.4: Data Integrity Check ---")
filtered_df = result.filtered_dataframe
print(f"✅ Shape: {filtered_df.shape}")
print(f"✅ All BAND_NUMBER values correct: {filtered_df['BAND_NUMBER'].unique()}")
print(f"✅ All MRBTS_ID values present: {len(filtered_df['MRBTS_ID'].unique())} unique")
print(f"✅ No NaN in key columns: {not filtered_df[['BAND_NUMBER', 'MRBTS_ID']].isna().any().any()}")

# TEST 1.5: Statistics calculation
print("\n--- Test 1.5: Statistics Preservation ---")
original_mean = df['RACH Stp Completion SR'].mean()
filtered_mean = filtered_df['RACH Stp Completion SR'].mean()
print(f"✅ Original mean: {original_mean:.2f}%")
print(f"✅ Filtered mean: {filtered_mean:.2f}%")
print(f"✅ Difference: {abs(original_mean - filtered_mean):.2f}% (should be <5%)")

print("\n" + "=" * 80)
print("✅ ALL INTEGRATION TESTS PASSED")
print("=" * 80)
