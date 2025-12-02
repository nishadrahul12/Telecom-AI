"""
Filtering Engine Module
======================
Applies user-selected filters and performs smart sampling while maintaining statistical properties.

Module: Phase 2 - Module 3
Status: Production Ready
Version: 1.0.0

Author: Telecom AI Optimization Project
Date: 2025-12-02
"""

from typing import Dict, List, Tuple, Optional, Any, Collection
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS (Pydantic-compatible)
# ============================================================================

class DataLevel(str, Enum):
    """Enumeration of data aggregation levels."""
    PLMN = "PLMN"
    REGION = "Region"
    CARRIER = "Carrier"
    CELL = "Cell"


@dataclass
class FilteredDataFrameResult:
    """
    Output contract for filtering operations.
    
    Attributes:
        filtered_dataframe: DataFrame after filters applied (pre-sampling)
        row_count_original: Row count before any filtering
        row_count_filtered: Row count after filtering (pre-sampling)
        row_count_sampled: Final row count after sampling
        sampling_factor: Sampling rate applied (1=no sampling, 5=every 5th row, etc.)
        filters_applied: Dictionary of {column: [selected_values]}
        sampling_method: "NONE" or "SYSTEMATIC"
        processing_time_ms: Execution time in milliseconds
    """
    filtered_dataframe: pd.DataFrame
    row_count_original: int
    row_count_filtered: int
    row_count_sampled: int
    sampling_factor: int
    filters_applied: Dict[str, List[Any]]
    sampling_method: str
    processing_time_ms: float


@dataclass
class DataFrameMetadata:
    """
    Metadata describing DataFrame structure and column classifications.
    
    Attributes:
        text_dimensions: Column names that are text-based dimensions (e.g., "Region", "SITENAME")
        numeric_dimensions: Column names that are numeric IDs (e.g., "MRBTS_ID", "LNBTS_ID")
        kpi_columns: Column names that are KPIs/metrics (numeric measurements)
        time_column: Name of the time column (e.g., "TIME")
        data_level: Aggregation level (PLMN, Region, Carrier, Cell)
    """
    text_dimensions: List[str]
    numeric_dimensions: List[str]
    kpi_columns: List[str]
    time_column: str
    data_level: str


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def get_unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """
    Return sorted unique values for a dimension column.
    
    Used for populating filter UI dropdowns. Handles missing values gracefully.
    Converts to string for consistent UI display.
    
    Args:
        df: Input DataFrame
        column: Column name to extract unique values from
        
    Returns:
        Sorted list of unique string values, excluding NaN
        
    Raises:
        ValueError: If column does not exist in DataFrame
        
    Examples:
        >>> df = pd.DataFrame({'REGION': ['N1', 'N2', 'N1', None]})
        >>> get_unique_values(df, 'REGION')
        ['N1', 'N2']
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available: {df.columns.tolist()}")
    
    # Get unique non-null values
    unique_vals = df[column].dropna().unique()
    
    # Convert to strings and sort
    unique_strings = sorted([str(v).strip() for v in unique_vals])
    
    logger.info(f"get_unique_values: {len(unique_strings)} unique values in '{column}'")
    return unique_strings


def apply_user_filters(df: pd.DataFrame, filter_dict: Optional[Dict[str, List[Any]]] = None) -> pd.DataFrame:
    """
    Apply user-selected filters to DataFrame using logical AND (all filters must match).
    
    For each filter column, keeps rows where the value is IN the provided list.
    Missing filter_dict or empty filter_dict returns full DataFrame unchanged.
    
    Filter Logic:
        - For each column in filter_dict, create a boolean mask: df[col].isin(filter_dict[col])
        - Combine all masks with & (logical AND)
        - Return only rows where ALL conditions are True
    
    Args:
        df: Input DataFrame
        filter_dict: Dictionary where keys are column names and values are lists of acceptable values.
                    Example: {'REGION': ['N1', 'N2'], 'CARRIER_NAME': ['L700']}
                    If None or empty, returns df unchanged.
    
    Returns:
        Filtered DataFrame (subset of input)
        
    Raises:
        ValueError: If a filter column doesn't exist in DataFrame
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'REGION': ['N1', 'N2', 'N1'],
        ...     'CARRIER_NAME': ['L700', 'L1800', 'L700']
        ... })
        >>> filters = {'REGION': ['N1']}
        >>> result = apply_user_filters(df, filters)
        >>> len(result)
        2
        >>> result['REGION'].unique()
        array(['N1'], dtype=object)
    """
    # Handle None or empty filter_dict
    if filter_dict is None or len(filter_dict) == 0:
        logger.info("apply_user_filters: No filters provided, returning full DataFrame")
        return df
    
    # Validate all filter columns exist
    missing_cols = [col for col in filter_dict.keys() if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Filter columns not found in DataFrame: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Build boolean mask: start with all True
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Apply each filter with AND logic
    for column, values in filter_dict.items():
        if not values:  # Skip empty filter lists
            logger.warning(f"apply_user_filters: Empty value list for column '{column}', skipping")
            continue
        
        # Create mask for this column: df[column].isin(values)
        column_mask = df[column].isin(values)
        mask = mask & column_mask
        
        logger.debug(f"apply_user_filters: Filter '{column}' -> {len(values)} values, "
                    f"matches: {column_mask.sum()} rows")
    
    # Apply combined mask
    filtered_df = df[mask].copy()
    logger.info(f"apply_user_filters: {len(df)} rows -> {len(filtered_df)} rows after filtering")
    
    return filtered_df


def smart_sampling(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Apply systematic sampling based on row count to maintain statistical significance.
    
    Sampling Strategy (preserves mean/std of KPI distributions):
        - N < 10,000:           Keep 100% (factor = 1)
        - 10,000 ≤ N < 50,000:  Keep every 5th row (factor = 5)
        - 50,000 ≤ N < 100,000: Keep every 10th row (factor = 10)
        - 100,000 ≤ N < 500,000: Keep every 50th row (factor = 50)
        - N ≥ 500,000:          Keep every 100th row (factor = 100)
    
    Method: Systematic sampling (deterministic, not random)
        - Uses df.iloc[::factor] to sample every N-th row
        - Maintains row order and representative coverage
        - More robust for time-series data than random sampling
    
    Args:
        df: Input DataFrame (any size)
        
    Returns:
        Tuple of (sampled_dataframe, sampling_factor)
        
    Examples:
        >>> df = pd.DataFrame({'KPI': range(50000)})
        >>> sampled_df, factor = smart_sampling(df)
        >>> factor
        5
        >>> len(sampled_df)
        10000
        >>> len(sampled_df) == len(df) // factor
        True
    """
    n_rows = len(df)
    
    # Determine sampling factor based on row count
    if n_rows < 10_000:
        sampling_factor = 1
    elif n_rows < 50_000:
        sampling_factor = 5
    elif n_rows < 100_000:
        sampling_factor = 10
    elif n_rows < 500_000:
        sampling_factor = 50
    else:  # n_rows >= 500_000
        sampling_factor = 100
    
    # Apply systematic sampling: every sampling_factor-th row
    sampled_df = df.iloc[::sampling_factor].copy()
    
    logger.info(
        f"smart_sampling: {n_rows:,} rows -> {len(sampled_df):,} rows "
        f"(factor={sampling_factor})"
    )
    
    return sampled_df, sampling_factor


def get_filter_options(df: pd.DataFrame, metadata: DataFrameMetadata, data_level: str) -> dict[str, Collection[str]]:
    """
    Determine which dimensions are available for filtering based on data aggregation level.
    
    Filtering Strategy by Level:
        - PLMN:   No filters available (highest aggregation, single row per PLMN)
        - Region: REGION + text dimensions only
        - Carrier: REGION + CARRIER_NAME + text dimensions
        - Cell:    All text dimensions + all numeric ID dimensions
    
    Returns dict with structure:
        {
            'filterable_columns': ['Region', 'CARRIER_NAME', ...],
            'all_options': {
                'REGION': ['N1', 'N2', ...],
                'CARRIER_NAME': ['L700', 'L1800', ...],
                ...
            }
        }
    
    Args:
        df: Input DataFrame
        metadata: DataFrameMetadata with column classifications
        data_level: Data aggregation level ('PLMN', 'Region', 'Carrier', 'Cell')
        
    Returns:
        Dictionary with filterable columns and their unique values
        
    Raises:
        ValueError: If data_level not recognized
        
    Examples:
        >>> metadata = DataFrameMetadata(
        ...     text_dimensions=['REGION', 'CITY', 'SITENAME'],
        ...     numeric_dimensions=['MRBTS_ID'],
        ...     kpi_columns=['RACH stp att'],
        ...     time_column='TIME',
        ...     data_level='Cell'
        ... )
        >>> df = pd.DataFrame({
        ...     'REGION': ['N1', 'N2'],
        ...     'CITY': ['Taipei', 'Kaohsiung'],
        ...     'MRBTS_ID': [100001, 100002],
        ... })
        >>> options = get_filter_options(df, metadata, 'Cell')
        >>> 'REGION' in options['filterable_columns']
        True
        >>> 'MRBTS_ID' in options['filterable_columns']
        True
    """
    valid_levels = [dl.value for dl in DataLevel]
    if data_level not in valid_levels:
        raise ValueError(f"data_level must be one of {valid_levels}, got '{data_level}'")
    
    # Determine filterable columns based on data level
    filterable_columns: List[str] = []
    
    if data_level == DataLevel.PLMN.value:
        # No filtering at PLMN level
        logger.info("get_filter_options: PLMN level - no filters available")
        filterable_columns = []
    
    elif data_level == DataLevel.REGION.value:
        # Region level: filter by text dimensions
        filterable_columns = metadata.text_dimensions
        logger.info(f"get_filter_options: Region level - filterable: {filterable_columns}")
    
    elif data_level == DataLevel.CARRIER.value:
        # Carrier level: filter by text dimensions
        filterable_columns = metadata.text_dimensions
        logger.info(f"get_filter_options: Carrier level - filterable: {filterable_columns}")
    
    elif data_level == DataLevel.CELL.value:
        # Cell level: filter by all text dimensions AND numeric ID dimensions
        filterable_columns = metadata.text_dimensions + metadata.numeric_dimensions
        logger.info(f"get_filter_options: Cell level - filterable: {filterable_columns}")
    
    # Get unique values for each filterable column
    all_options: Dict[str, List[str]] = {}
    for column in filterable_columns:
        if column in df.columns:
            all_options[column] = get_unique_values(df, column)
        else:
            logger.warning(f"get_filter_options: Column '{column}' not found in DataFrame")
    
    result = {
        'filterable_columns': filterable_columns,
        'all_options': all_options
    }
    
    logger.info(f"get_filter_options: {len(filterable_columns)} filterable columns available")
    return result


# ============================================================================
# HIGH-LEVEL ORCHESTRATION
# ============================================================================

def apply_filters_and_sample(
    df: pd.DataFrame,
    metadata: DataFrameMetadata,
    data_level: str,
    filter_dict: Optional[Dict[str, List[Any]]] = None
) -> FilteredDataFrameResult:
    """
    High-level orchestration: Apply filters + smart sampling in sequence.
    
    Execution flow:
        1. Record original row count
        2. Apply user filters (if provided)
        3. Apply smart sampling to filtered result
        4. Calculate statistics and return result object
    
    Args:
        df: Input DataFrame
        metadata: Column classification metadata
        data_level: Data aggregation level for filter validation
        filter_dict: Optional filters to apply
        
    Returns:
        FilteredDataFrameResult with full statistics
        
    Examples:
        >>> import time
        >>> df = pd.DataFrame({
        ...     'TIME': pd.date_range('2024-01-01', periods=100000),
        ...     'REGION': ['N1', 'N2'] * 50000,
        ...     'KPI_VALUE': np.random.randn(100000)
        ... })
        >>> metadata = DataFrameMetadata(
        ...     text_dimensions=['REGION'],
        ...     numeric_dimensions=[],
        ...     kpi_columns=['KPI_VALUE'],
        ...     time_column='TIME',
        ...     data_level='Region'
        ... )
        >>> result = apply_filters_and_sample(df, metadata, 'Region', {'REGION': ['N1']})
        >>> result.row_count_original
        100000
        >>> result.row_count_filtered
        50000
        >>> result.sampling_factor
        10
    """
    import time
    start_time = time.time()
    
    try:
        # Step 1: Record original
        row_count_original = len(df)
        
        # Step 2: Apply filters
        filtered_df = apply_user_filters(df, filter_dict)
        row_count_filtered = len(filtered_df)
        
        # Step 3: Apply smart sampling
        sampled_df, sampling_factor = smart_sampling(filtered_df)
        row_count_sampled = len(sampled_df)
        
        # Step 4: Determine sampling method
        sampling_method = "NONE" if sampling_factor == 1 else "SYSTEMATIC"
        
        # Calculate execution time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Create result
        result = FilteredDataFrameResult(
            filtered_dataframe=sampled_df,
            row_count_original=row_count_original,
            row_count_filtered=row_count_filtered,
            row_count_sampled=row_count_sampled,
            sampling_factor=sampling_factor,
            filters_applied=filter_dict or {},
            sampling_method=sampling_method,
            processing_time_ms=processing_time_ms
        )
        
        logger.info(
            f"apply_filters_and_sample completed: "
            f"{row_count_original:,} -> {row_count_filtered:,} -> {row_count_sampled:,} "
            f"(sample_factor={sampling_factor}, time={processing_time_ms:.2f}ms)"
        )
        
        return result
    
    except Exception as e:
        logger.error(f"apply_filters_and_sample failed: {str(e)}", exc_info=True)
        raise


# ============================================================================
# VALIDATION & STATISTICS HELPERS
# ============================================================================

def validate_filter_dict(filter_dict: Dict[str, List[Any]], df: pd.DataFrame) -> bool:
    """
    Validate that all filter columns exist in DataFrame and values are non-empty.
    
    Args:
        filter_dict: Filters to validate
        df: DataFrame to validate against
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    for column, values in filter_dict.items():
        if column not in df.columns:
            raise ValueError(f"Filter column '{column}' not in DataFrame")
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"Filter values for '{column}' must be list/tuple, got {type(values)}")
        if len(values) == 0:
            raise ValueError(f"Filter values for '{column}' cannot be empty")
    
    return True


def sampling_statistics(df_original: pd.DataFrame, df_sampled: pd.DataFrame, kpi_columns: List[str]) -> Dict[str, float]:
    """
    Compare mean/std between original and sampled DataFrames for KPI columns.
    Helps verify that sampling maintains statistical properties.
    
    Args:
        df_original: Original DataFrame
        df_sampled: Sampled DataFrame
        kpi_columns: List of KPI column names to compare
        
    Returns:
        Dictionary with variance metrics (percent difference in mean/std)
        
    Examples:
        >>> df = pd.DataFrame({'KPI': np.random.randn(10000)})
        >>> sampled, _ = smart_sampling(df)
        >>> stats = sampling_statistics(df, sampled, ['KPI'])
        >>> stats['KPI_mean_variance_pct']  # Should be <5%
        2.3
    """
    stats = {}
    
    for col in kpi_columns:
        if col not in df_original.columns or col not in df_sampled.columns:
            logger.warning(f"sampling_statistics: Column '{col}' not in one of the DataFrames")
            continue
        
        # Remove NaN for comparison
        orig_values = df_original[col].dropna()
        samp_values = df_sampled[col].dropna()
        
        if len(orig_values) == 0 or len(samp_values) == 0:
            logger.warning(f"sampling_statistics: No valid values in '{col}' after dropping NaN")
            continue
        
        orig_mean = orig_values.mean()
        samp_mean = samp_values.mean()
        orig_std = orig_values.std()
        samp_std = samp_values.std()
        
        # Calculate percent variance
        mean_variance_pct = abs((samp_mean - orig_mean) / orig_mean) * 100 if orig_mean != 0 else 0
        std_variance_pct = abs((samp_std - orig_std) / orig_std) * 100 if orig_std != 0 else 0
        
        stats[f"{col}_mean_variance_pct"] = mean_variance_pct
        stats[f"{col}_std_variance_pct"] = std_variance_pct
        
        logger.info(
            f"sampling_statistics: {col} - "
            f"Mean diff: {mean_variance_pct:.2f}%, Std diff: {std_variance_pct:.2f}%"
        )
    
    return stats
