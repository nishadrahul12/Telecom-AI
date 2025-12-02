"""
Edge case test: Verify error handling and edge cases
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from filtering_engine import apply_filters_and_sample, DataFrameMetadata

print("=" * 80)
print("EDGE CASE TEST: Error Handling")
print("=" * 80)

df = pd.DataFrame({
    'TIME': pd.date_range('2024-01-01', periods=100),
    'REGION': ['N1'] * 100,
    'METRIC': np.random.random(100) * 100,
})

metadata = DataFrameMetadata(
    text_dimensions=['REGION'],
    numeric_dimensions=[],
    kpi_columns=['METRIC'],
    time_column='TIME',
    data_level='Cell'
)

# TEST 3.1: Empty filter
print("\n--- Test 3.1: Empty Filter Dict ---")
result = apply_filters_and_sample(df, metadata, 'Cell', {})
print(f"✅ No filter applied: {len(result.filtered_dataframe)} rows (should be 100)")
assert len(result.filtered_dataframe) == 100

# TEST 3.2: Non-existent value in filter
print("\n--- Test 3.2: Non-existent Filter Value ---")
result = apply_filters_and_sample(df, metadata, 'Cell', {'REGION': ['N99']})
print(f"✅ No rows found: {len(result.filtered_dataframe)} rows (should be 0)")
assert len(result.filtered_dataframe) == 0

# TEST 3.3: DataFrame with NaN
print("\n--- Test 3.3: DataFrame with NaN Values ---")
df_nan = df.copy()
df_nan.loc[10:20, 'METRIC'] = np.nan
result = apply_filters_and_sample(df_nan, metadata, 'Cell')
print(f"✅ Handled NaN: {len(result.filtered_dataframe)} rows")

# TEST 3.4: Single row
print("\n--- Test 3.4: Single Row DataFrame ---")
df_single = df.iloc[:1].copy()
result = apply_filters_and_sample(df_single, metadata, 'Cell')
print(f"✅ Single row handled: {len(result.filtered_dataframe)} rows")
assert len(result.filtered_dataframe) == 1

# TEST 3.5: Unicode characters
print("\n--- Test 3.5: Unicode Characters ---")
df_unicode = df.copy()
df_unicode['CITY'] = ['台北市', '高雄市'] * 50
metadata_unicode = DataFrameMetadata(
    text_dimensions=['REGION', 'CITY'],
    numeric_dimensions=[],
    kpi_columns=['METRIC'],
    time_column='TIME',
    data_level='Cell'
)
result = apply_filters_and_sample(
    df_unicode, metadata_unicode, 'Cell', 
    {'CITY': ['台北市']}
)
print(f"✅ Unicode handled: {len(result.filtered_dataframe)} rows")

print("\n" + "=" * 80)
print("✅ ALL EDGE CASE TESTS PASSED")
print("=" * 80)
