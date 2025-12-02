"""Quick validation with Sample_KPI_Data.csv"""
import pandas as pd
from filtering_engine import (
    get_unique_values,
    apply_filters_and_sample,
    DataFrameMetadata,
    get_filter_options
)

# Load your actual sample data
df = pd.read_csv('../Sample_KPI_Data.csv')

print(f"Loaded {len(df):,} rows")
print(f"Columns: {df.columns.tolist()}")

# Setup metadata (CUSTOMIZE FOR YOUR DATA)
metadata = DataFrameMetadata(
    text_dimensions=['REGION', 'CITY', 'DISTRICT', 'SITENAME', 'CARRIER_NAME'],
    numeric_dimensions=['MRBTS_ID', 'LNBTS_ID', 'LNCEL_ID'],
    kpi_columns=['RACH stp att', 'RRC conn stp SR', 'E-RAB SAtt'],
    time_column='TIME',
    data_level='Cell'
)

# Test 1: Get available filters
print("\n=== Test 1: Available Filters ===")
filter_opts = get_filter_options(df, metadata, 'Cell')
print(f"Filterable columns: {filter_opts['filterable_columns']}")
print(f"Regions: {filter_opts['all_options']['REGION'][:5]}")  # First 5

# Test 2: Apply filters
print("\n=== Test 2: Apply Filters ===")
filters = {'REGION': ['N1']}
result = apply_filters_and_sample(df, metadata, 'Cell', filters)
print(f"Original: {result.row_count_original:,} rows")
print(f"Filtered: {result.row_count_filtered:,} rows")
print(f"Sampled: {result.row_count_sampled:,} rows")
print(f"Sampling factor: {result.sampling_factor}")
print(f"Time: {result.processing_time_ms:.2f}ms")

# Test 3: Verify DataFrame
print("\n=== Test 3: Verify Output DataFrame ===")
output_df = result.filtered_dataframe
print(f"Output shape: {output_df.shape}")
print(f"Columns match: {set(output_df.columns) == set(df.columns)}")
print(f"All REGION='N1': {all(output_df['REGION'] == 'N1')}")

print("\n✓ All validations passed!")
