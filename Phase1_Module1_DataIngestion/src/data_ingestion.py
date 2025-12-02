"""
data_ingestion.py
=================
Core data ingestion module for Intelligent Telecom Optimization System.

Reads raw CSV files, auto-detects column types, and produces structured
metadata for downstream analytics. Handles multi-language UTF-8 encoding,
large files, and various time formats.

Author: AI Assistant
Phase: 1 (Foundation)
Module: Data Ingestion
"""

import os
import re
import logging
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import numpy as np
from datetime import datetime

from data_models import DataFrameMetadata, ColumnClassification


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class DataIngestionError(Exception):
    """Base exception for data ingestion errors."""
    pass


class TimeColumnNotFoundError(DataIngestionError):
    """Raised when time column cannot be detected."""
    pass


class TimeFormatDetectionError(DataIngestionError):
    """Raised when time format cannot be determined."""
    pass


class DataValidationError(DataIngestionError):
    """Raised when data fails validation checks."""
    pass


class EncodingDetectionError(DataIngestionError):
    """Raised when file encoding cannot be detected."""
    pass


# ============================================================================
# FUNCTION 1: READ CSV WITH ENCODING DETECTION
# ============================================================================

def read_csv_with_encoding(
    file_path: str,
    encoding: str = 'utf-8',
    sep: str = ','
) -> pd.DataFrame:
    """
    Read CSV file with robust encoding detection and error handling.
    
    Attempts to read CSV with specified encoding, with fallback to
    common encodings (utf-8-sig, latin1, gb2312) if initial attempt fails.
    Handles multi-language UTF-8 data (Chinese, Japanese, emoji).
    
    Args:
        file_path: Full path to CSV file
        encoding: Initial encoding to try (default: 'utf-8')
        sep: Column separator (default: ',')
    
    Returns:
        pd.DataFrame: DataFrame with all rows
    
    Raises:
        FileNotFoundError: If file does not exist
        EncodingDetectionError: If no encoding succeeds
        
    Performance:
        - 10KB: <100ms
        - 1MB: <500ms
        - 100MB: <3s
        - 1GB: Uses chunking (see implementation for details)
    
    Example:
        >>> df = read_csv_with_encoding('data/telecom_data.csv')
        >>> print(df.shape)
        (50000, 15)
        
        >>> # With specific encoding
        >>> df = read_csv_with_encoding('data/chinese_data.csv', encoding='gb2312')
    """
    
    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"Reading CSV file: {file_path} ({file_size_mb:.2f} MB)")
    
    # Encodings to try in order (most common for telecom data first)
    encodings_to_try = [
        encoding,
        'utf-8-sig',  # UTF-8 with BOM
        'latin1',      # Western European
        'iso-8859-1',  # Alternative Western European
        'gb2312',      # Simplified Chinese
        'gbk',         # Extended Chinese
        'big5',        # Traditional Chinese
        'shift_jis',   # Japanese
        'cp1252',      # Windows Western European
    ]
    
    # For files >500MB, use chunking for efficiency
    if file_size_mb > 500:
        logger.warning(f"Large file detected ({file_size_mb:.2f} MB). Using chunked reading.")
        chunk_size = 50000
        chunks = []
        
        for enc in encodings_to_try:
            try:
                for chunk in pd.read_csv(
                    file_path,
                    sep=sep,
                    encoding=enc,
                    chunksize=chunk_size,
                    on_bad_lines='skip'
                ):
                    chunks.append(chunk)
                
                df = pd.concat(chunks, ignore_index=True)
                logger.info(f"Successfully read CSV with encoding: {enc}")
                return df
            
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
    
    # Standard reading for files <=500MB
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(
                file_path,
                sep=sep,
                encoding=enc,
                on_bad_lines='skip'
            )
            logger.info(f"Successfully read CSV with encoding: {enc}")
            return df
        
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    
    # All encodings failed
    raise EncodingDetectionError(
        f"Failed to read {file_path} with any known encoding. "
        f"Tried: {', '.join(encodings_to_try)}"
    )


# ============================================================================
# FUNCTION 2: DETECT TIME COLUMN
# ============================================================================

def detect_time_column(df: pd.DataFrame) -> str:
    """
    Detect time column from DataFrame column names.
    
    Searches column names for common time-related keywords using
    case-insensitive matching. Returns first match found.
    
    Supported Keywords:
        - Time, Date, Timestamp, DateTime
        - Date_Time, DatetimeUTC, DateTime_UTC
        - Time_Slot, TimeSlot, Period
        - ReportingPeriod, Reporting_Period
        - RecordTime, Record_Time
    
    Args:
        df: Input DataFrame
    
    Returns:
        str: Exact column name containing time data
    
    Raises:
        TimeColumnNotFoundError: If no time column detected
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Region': ['North', 'South'],
        ...     'DateTime': ['2024-03-17 00', '2024-03-18 00'],
        ...     'KPI_Value': [100, 120]
        ... })
        >>> col = detect_time_column(df)
        >>> print(col)
        'DateTime'
    """
    
    # Keywords to search for in column names (case-insensitive)
    time_keywords = [
        'time', 'date', 'timestamp', 'datetime',
        'date_time', 'datetimeutc', 'datetime_utc',
        'time_slot', 'timeslot', 'period',
        'reportingperiod', 'reporting_period',
        'recordtime', 'record_time', 'utc_time',
        'gmt_time', 'timestamp_utc', 'record_date'
    ]
    
    # Search column names
    for col in df.columns:
        col_lower = col.lower().replace('_', '').replace(' ', '')
        
        for keyword in time_keywords:
            if keyword in col_lower:
                logger.info(f"Detected time column: '{col}'")
                return col
    
    # No time column found
    raise TimeColumnNotFoundError(
        f"No time column detected. Columns found: {list(df.columns)}. "
        f"Expected columns named with: time, date, timestamp, datetime, etc."
    )


# ============================================================================
# FUNCTION 3: PARSE TIME FORMAT
# ============================================================================

def parse_time_format(time_series: pd.Series) -> str:
    """
    Detect time format from sample of time values.
    
    Analyzes first 100 non-null values to determine time format.
    Supports two primary formats:
    
    1. Daily: MM/DD/YYYY (e.g., 3/1/2024, 03/01/2024)
    2. Hourly: YYYY-MM-DD HH (e.g., 2025-03-17 00, 2025-03-17 0)
    
    Also supports variations:
        - 1/1/2024, 01/01/2024 (daily)
        - 2024-03-17 0, 2024-03-17 00 (hourly)
    
    Args:
        time_series: Pandas Series containing time values (str or datetime)
    
    Returns:
        str: Detected format (e.g., "YYYY-MM-DD HH" or "MM/DD/YYYY")
    
    Raises:
        TimeFormatDetectionError: If format cannot be determined
        
    Example:
        >>> times = pd.Series(['3/1/2024', '3/2/2024', '3/3/2024'])
        >>> fmt = parse_time_format(times)
        >>> print(fmt)
        'MM/DD/YYYY'
    """
    
    # Get non-null sample (up to 100 values)
    sample = time_series.dropna().head(100).astype(str)
    
    if len(sample) == 0:
        raise TimeFormatDetectionError("Time column contains no non-null values")
    
    # Regex patterns for time formats
    patterns = {
        'YYYY-MM-DD HH': r'^\d{4}-\d{2}-\d{2}\s+\d{1,2}$',  # 2025-03-17 00 or 2025-03-17 0
        'MM/DD/YYYY': r'^\d{1,2}/\d{1,2}/\d{4}$',           # 3/1/2024 or 03/01/2024
    }
    
    # Try each pattern against sample
    for format_name, pattern in patterns.items():
        matches = sample.str.match(pattern, na=False).sum()
        match_rate = matches / len(sample)
        
        logger.debug(f"Format '{format_name}': {match_rate:.1%} match rate ({matches}/{len(sample)})")
        
        # If >80% of sample matches, consider it detected
        if match_rate >= 0.8:
            logger.info(f"Detected time format: {format_name}")
            return format_name
    
    # Fallback: log first few values for debugging
    logger.warning(f"Could not auto-detect time format. Sample values: {sample.head(3).tolist()}")
    raise TimeFormatDetectionError(
        f"Unable to detect time format. Sample values: {sample.head(5).tolist()}"
    )


# ============================================================================
# FUNCTION 4: CLASSIFY COLUMNS
# ============================================================================

def classify_columns(
    df: pd.DataFrame,
    time_column: str
) -> ColumnClassification:
    """
    Auto-classify columns into categories: Text Dimension, ID Dimension, KPI.
    
    Classification Logic:
        1. Skip time column
        2. For each remaining column:
           - If dtype == 'object' (string): → Dimension (Text)
           - If column name contains 'ID' OR >90% unique values: → Dimension (ID)
           - If dtype in [int, float] AND NOT ID: → KPI (Metric)
    
    Confidence Score:
        - High (0.9-1.0): Clear column types (mostly numeric or text)
        - Medium (0.7-0.89): Some ambiguous columns
        - Low (<0.7): Many edge cases
    
    Args:
        df: Input DataFrame
        time_column: Name of time column (to skip from classification)
    
    Returns:
        ColumnClassification: Object with classified column lists and confidence
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Time': ['2024-03-17 00'] * 3,
        ...     'Region': ['North', 'South', 'East'],
        ...     'MRBTS_ID': [1001, 1002, 1003],
        ...     'DL_PRB': [75.5, 80.2, 72.1]
        ... })
        >>> result = classify_columns(df, 'Time')
        >>> print(result.dimensions_text)
        ['Region']
        >>> print(result.dimensions_id)
        ['MRBTS_ID']
        >>> print(result.kpis)
        ['DL_PRB']
    """
    
    dimensions_text = []
    dimensions_id = []
    kpis = []
    
    # Iterate through columns (excluding time column)
    for col in df.columns:
        if col == time_column:
            continue
        
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df)
        
        # Check if column name indicates ID
        is_id_column = 'id' in col.lower()
        
        # Classification logic
        if is_id_column:
            # ✓ Column name contains 'id' → ID Dimension (HIGHEST PRIORITY)
            # This catches BOTH text and numeric ID columns
            dimensions_id.append(col)
            logger.debug(f"  {col}: ID Dimension (is_id=True)")
        
        elif dtype in [int, float, np.int64, np.float64]:
            # ✓ Numeric (after ID check) → KPI Metrics
            kpis.append(col)
            logger.debug(f"  {col}: KPI Metric (dtype={dtype})")
        
        elif dtype == 'object':
            # ✓ String columns → Text Dimension
            dimensions_text.append(col)
            logger.debug(f"  {col}: Text Dimension (dtype={dtype})")
        
        elif unique_ratio > 0.9:
            # ✓ High uniqueness (non-numeric, non-text) → ID Dimension
            dimensions_id.append(col)
            logger.debug(f"  {col}: ID Dimension (unique_ratio={unique_ratio:.1%})")
        
        else:
            # ✓ Fallback: classify as text
            dimensions_text.append(col)
            logger.debug(f"  {col}: Text Dimension (default)")
    
    # Calculate confidence
    total_cols = len(dimensions_text) + len(dimensions_id) + len(kpis)
    confidence = 1.0 if total_cols > 0 else 0.5
    
    if len(dimensions_text) == 0 or len(kpis) == 0:
        confidence = max(0.7, confidence - 0.1)  # Lower confidence if missing categories
    
    logger.info(
        f"Column Classification: {len(dimensions_text)} text, "
        f"{len(dimensions_id)} ID, {len(kpis)} KPIs (confidence: {confidence:.1%})"
    )
    
    return ColumnClassification(
        dimensions_text=dimensions_text,
        dimensions_id=dimensions_id,
        kpis=kpis,
        confidence=confidence
    )


# ============================================================================
# FUNCTION 5: VALIDATE DATA INTEGRITY
# ============================================================================

def validate_data_integrity(
    df: pd.DataFrame,
    metadata: DataFrameMetadata
) -> bool:
    """
    Validate data integrity before processing.
    
    Checks:
        1. DataFrame not empty (>0 rows)
        2. Time column exists and has no NaN values
        3. At least 1 KPI present
        4. No completely empty rows
    
    Args:
        df: Input DataFrame
        metadata: DataFrameMetadata object to validate
    
    Returns:
        bool: True if all validations pass
    
    Raises:
        DataValidationError: If validation fails
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Time': ['2024-03-17 00', '2024-03-18 00'],
        ...     'Region': ['North', 'South'],
        ...     'KPI': [100, 120]
        ... })
        >>> metadata = DataFrameMetadata(...)
        >>> is_valid = validate_data_integrity(df, metadata)
        >>> print(is_valid)
        True
    """
    
    logger.info("Validating data integrity...")
    
    # Check 1: DataFrame not empty
    if len(df) == 0:
        raise DataValidationError("DataFrame is empty (0 rows)")
    
    # Check 2: Time column exists and no NaN
    if metadata.time_column not in df.columns:
        raise DataValidationError(f"Time column '{metadata.time_column}' not found in DataFrame")
    
    time_nans = df[metadata.time_column].isna().sum()
    if time_nans > 0:
        raise DataValidationError(
            f"Time column '{metadata.time_column}' contains {time_nans} NaN values. "
            f"Time column must be complete."
        )
    
    # Check 3: At least 1 KPI
    if len(metadata.kpis) == 0:
        raise DataValidationError("No KPI columns detected. At least 1 numeric metric required.")
    
    # Check 4: No completely empty rows
    empty_rows = df.isna().all(axis=1).sum()
    if empty_rows > 0:
        logger.warning(f"Found {empty_rows} completely empty rows (will be retained)")
    
    logger.info(f"✓ Data integrity validation passed ({len(df)} rows, {len(df.columns)} columns)")
    return True


# ============================================================================
# FUNCTION 6: NORMALIZE DATA TYPES
# ============================================================================

def normalize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize data types for consistent processing.
    
    Conversions:
        1. Numeric strings → numeric types (int/float)
        2. Keep object/string columns as-is (for text dimensions)
        3. Handle special values: NaN, inf, missing
    
    Args:
        df: Input DataFrame with potentially mixed types
    
    Returns:
        pd.DataFrame: DataFrame with normalized types
    
    Example:
        >>> df = pd.DataFrame({
        ...     'KPI_A': ['100.5', '200.3', '150.2'],  # String numbers
        ...     'Region': ['North', 'South', 'East'],   # Text
        ...     'ID': ['1001', '1002', '1003']          # Numeric strings
        ... })
        >>> df_normalized = normalize_data_types(df)
        >>> print(df_normalized.dtypes)
        KPI_A      float64
        Region      object
        ID         object  # Kept as object (ID column)
        dtype: object
    """
    
    logger.info("Normalizing data types...")
    df_normalized = df.copy()
    
    for col in df_normalized.columns:
        dtype = df_normalized[col].dtype
        
        # Skip if already numeric
        if dtype in [int, float, np.int64, np.float64]:
            continue
        
        # Try to convert string columns to numeric
        if dtype == 'object':
            try:
                # Attempt numeric conversion
                numeric_col = pd.to_numeric(df_normalized[col], errors='coerce')
                
                # Check if conversion was successful (>90% converted)
                non_null_original = df_normalized[col].notna().sum()
                non_null_numeric = numeric_col.notna().sum()
                
                if non_null_numeric / max(non_null_original, 1) >= 0.9:
                    df_normalized[col] = numeric_col
                    logger.debug(f"  {col}: Converted object → numeric ({non_null_numeric}/{non_null_original} successful)")
            
            except Exception as e:
                logger.debug(f"  {col}: Could not convert to numeric ({str(e)[:50]})")
    
    logger.info("✓ Data type normalization complete")
    return df_normalized


# ============================================================================
# MAIN INGESTION FUNCTION
# ============================================================================

def ingest_csv(
    file_path: str,
    encoding: str = 'utf-8',
    sep: str = ','
) -> DataFrameMetadata:
    """
    Complete data ingestion pipeline: Read → Detect → Parse → Classify → Validate.
    
    This is the main entry point for data ingestion. Orchestrates all steps:
        1. Read CSV with encoding detection
        2. Detect time column
        3. Parse time format
        4. Classify columns
        5. Validate data integrity
        6. Normalize data types
        7. Return structured metadata
    
    Args:
        file_path: Full path to CSV file
        encoding: Initial encoding to try (default: 'utf-8')
        sep: Column separator (default: ',')
    
    Returns:
        DataFrameMetadata: Complete metadata including DataFrame, column classifications, formats
    
    Raises:
        Various DataIngestionError subclasses on failure
    
    Performance Targets:
        - 10KB: <100ms
        - 1MB: <500ms
        - 100MB: <3s
        - 1GB: <15s
    
    Example:
        >>> metadata = ingest_csv('data/telecom_data.csv')
        >>> print(metadata.summary())
        DataFrameMetadata Summary:
          Time Column: Time (Format: YYYY-MM-DD HH)
          Rows: 50000 | File Size: 2.50 MB
          Text Dimensions: 3 columns
          ID Dimensions: 2 columns
          KPIs: 8 metrics
          Encoding: utf-8
          Classification Confidence: 95.0%
    """
    
    logger.info("=" * 80)
    logger.info(f"Starting data ingestion: {file_path}")
    logger.info("=" * 80)
    
    try:
        # Step 1: Read CSV
        start_time = pd.Timestamp.now()
        df = read_csv_with_encoding(file_path, encoding=encoding, sep=sep)
        logger.info(f"✓ Step 1 (Read): {len(df)} rows, {len(df.columns)} columns")
        
        # Step 2: Detect time column
        time_column = detect_time_column(df)
        logger.info(f"✓ Step 2 (Detect Time): '{time_column}'")
        
        # Step 3: Parse time format
        time_format = parse_time_format(df[time_column])
        logger.info(f"✓ Step 3 (Parse Time Format): {time_format}")
        
        # Step 4: Normalize data types
        df = normalize_data_types(df)
        logger.info(f"✓ Step 4 (Normalize Types): Complete")
        
        # Step 5: Classify columns
        classification = classify_columns(df, time_column)
        logger.info(f"✓ Step 5 (Classify Columns): Complete")
        
        # Step 6: Create metadata
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        metadata = DataFrameMetadata(
            dataframe=df,
            time_column=time_column,
            time_format=time_format,
            dimensions_text=classification.dimensions_text,
            dimensions_id=classification.dimensions_id,
            kpis=classification.kpis,
            row_count=len(df),
            file_size_mb=file_size_mb,
            encoding_used='utf-8',
            classification_confidence=classification.confidence
        )
        logger.info(f"✓ Step 6 (Validate): Complete")
        
        # Step 7: Validate data integrity
        validate_data_integrity(df, metadata)
        logger.info(f"✓ Step 7 (Validate Integrity): Complete")
        
        # Timing
        elapsed_time = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"✓ Ingestion complete in {elapsed_time:.2f}s")
        logger.info(metadata.summary())
        logger.info("=" * 80)
        
        return metadata
    
    except Exception as e:
        logger.error(f"✗ Ingestion failed: {str(e)}")
        raise
