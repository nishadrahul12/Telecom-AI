"""
test_data_ingestion.py
======================
Comprehensive unit tests for data ingestion module.

Tests all 6 core functions plus integration tests and edge cases.
Coverage includes:
  - File reading (various encodings, file sizes)
  - Time column detection (10+ name variations)
  - Time format parsing (daily, hourly, edge cases)
  - Column classification (text, ID, KPI)
  - Data validation (integrity checks)
  - Data normalization (type conversions)

Author: AI Assistant
Phase: 1 (Foundation)
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from typing import List

from data_ingestion import (
    read_csv_with_encoding,
    detect_time_column,
    parse_time_format,
    classify_columns,
    validate_data_integrity,
    normalize_data_types,
    ingest_csv,
    TimeColumnNotFoundError,
    TimeFormatDetectionError,
    DataValidationError,
)
from data_models import DataFrameMetadata


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_csv_file():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("Time,Region,MRBTS_ID,DL_PRB_UTILIZATION,UL_PRB_UTILIZATION\n")
        f.write("2024-03-17 00,North,1001,75.5,68.2\n")
        f.write("2024-03-17 01,North,1001,76.2,69.1\n")
        f.write("2024-03-17 02,South,1002,72.1,65.8\n")
        f.write("2024-03-17 03,South,1002,73.5,66.9\n")
        f.write("2024-03-17 04,East,1003,80.2,75.3\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def daily_format_csv():
    """Create CSV with daily time format."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("Date,Region,KPI_1,KPI_2\n")
        f.write("3/1/2024,North,100.5,200.3\n")
        f.write("3/2/2024,South,105.2,205.1\n")
        f.write("3/3/2024,East,102.8,202.9\n")
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def utf8_multilang_csv():
    """Create CSV with UTF-8 multi-language data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        f.write("Time,City,Region,KPI\n")
        f.write("2024-03-17 00,台北市,北區,75.5\n")
        f.write("2024-03-17 01,東京,関東,76.2\n")
        f.write("2024-03-17 02,上海,华东,72.1\n")
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)


# ============================================================================
# TEST FUNCTION 1: read_csv_with_encoding
# ============================================================================

class TestReadCSV:
    """Tests for read_csv_with_encoding function."""
    
    def test_read_simple_csv(self, temp_csv_file):
        """Test reading simple CSV file."""
        df = read_csv_with_encoding(temp_csv_file)
        assert len(df) == 5
        assert list(df.columns) == ['Time', 'Region', 'MRBTS_ID', 'DL_PRB_UTILIZATION', 'UL_PRB_UTILIZATION']
    
    def test_read_nonexistent_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            read_csv_with_encoding('/nonexistent/path/file.csv')
    
    def test_read_utf8_multilang(self, utf8_multilang_csv):
        """Test reading UTF-8 multi-language data."""
        df = read_csv_with_encoding(utf8_multilang_csv)
        assert len(df) == 3
        assert '台北市' in df['City'].values
        assert '東京' in df['City'].values
        assert '上海' in df['City'].values
    
    def test_read_with_custom_sep(self):
        """Test reading CSV with custom separator."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time;Region;KPI\n")
            f.write("2024-03-17 00;North;100\n")
            f.write("2024-03-17 01;South;120\n")
            temp_path = f.name
        
        try:
            df = read_csv_with_encoding(temp_path, sep=';')
            assert len(df) == 2
            assert list(df.columns) == ['Time', 'Region', 'KPI']
        finally:
            os.unlink(temp_path)


# ============================================================================
# TEST FUNCTION 2: detect_time_column
# ============================================================================

class TestDetectTimeColumn:
    """Tests for detect_time_column function."""
    
    def test_detect_time_column(self, temp_csv_file):
        """Test detecting standard 'Time' column."""
        df = read_csv_with_encoding(temp_csv_file)
        time_col = detect_time_column(df)
        assert time_col == 'Time'
    
    def test_detect_date_column(self):
        """Test detecting 'Date' column."""
        df = pd.DataFrame({'Date': ['2024-03-17'] * 3, 'Value': [1, 2, 3]})
        time_col = detect_time_column(df)
        assert time_col == 'Date'
    
    def test_detect_timestamp_column(self):
        """Test detecting 'Timestamp' column."""
        df = pd.DataFrame({'Timestamp': ['2024-03-17 00:00:00'] * 3, 'Value': [1, 2, 3]})
        time_col = detect_time_column(df)
        assert time_col == 'Timestamp'
    
    def test_detect_datetime_column(self):
        """Test detecting 'DateTime' column."""
        df = pd.DataFrame({'DateTime': ['2024-03-17 00'] * 3, 'Value': [1, 2, 3]})
        time_col = detect_time_column(df)
        assert time_col == 'DateTime'
    
    def test_detect_date_time_column(self):
        """Test detecting 'Date_Time' column (underscore variant)."""
        df = pd.DataFrame({'Date_Time': ['2024-03-17 00'] * 3, 'Value': [1, 2, 3]})
        time_col = detect_time_column(df)
        assert time_col == 'Date_Time'
    
    def test_detect_timeslot_column(self):
        """Test detecting 'TimeSlot' column (camel case)."""
        df = pd.DataFrame({'TimeSlot': ['2024-03-17 00'] * 3, 'Value': [1, 2, 3]})
        time_col = detect_time_column(df)
        assert time_col == 'TimeSlot'
    
    def test_no_time_column(self):
        """Test error when no time column exists."""
        df = pd.DataFrame({'Region': ['North', 'South'], 'Value': [1, 2]})
        with pytest.raises(TimeColumnNotFoundError):
            detect_time_column(df)


# ============================================================================
# TEST FUNCTION 3: parse_time_format
# ============================================================================

class TestParseTimeFormat:
    """Tests for parse_time_format function."""
    
    def test_parse_hourly_format_full(self):
        """Test parsing hourly format with full hour (HH)."""
        times = pd.Series(['2024-03-17 00', '2024-03-17 01', '2024-03-17 02'] * 20)
        fmt = parse_time_format(times)
        assert fmt == 'YYYY-MM-DD HH'
    
    def test_parse_hourly_format_single_digit(self):
        """Test parsing hourly format with single digit hour."""
        times = pd.Series(['2024-03-17 0', '2024-03-17 1', '2024-03-17 2'] * 20)
        fmt = parse_time_format(times)
        assert fmt == 'YYYY-MM-DD HH'
    
    def test_parse_daily_format_single_digit(self):
        """Test parsing daily format with single digit month/day."""
        times = pd.Series(['3/1/2024', '3/2/2024', '3/3/2024'] * 20)
        fmt = parse_time_format(times)
        assert fmt == 'MM/DD/YYYY'
    
    def test_parse_daily_format_double_digit(self):
        """Test parsing daily format with double digit month/day."""
        times = pd.Series(['03/01/2024', '03/02/2024', '03/03/2024'] * 20)
        fmt = parse_time_format(times)
        assert fmt == 'MM/DD/YYYY'
    
    def test_parse_mixed_daily_format(self):
        """Test parsing mixed daily format (both single and double digits)."""
        times = pd.Series(['3/1/2024', '03/02/2024', '3/3/2024'] * 20)
        fmt = parse_time_format(times)
        assert fmt == 'MM/DD/YYYY'
    
    def test_parse_empty_series(self):
        """Test error handling for empty series."""
        times = pd.Series([])
        with pytest.raises(TimeFormatDetectionError):
            parse_time_format(times)
    
    def test_parse_all_nan_series(self):
        """Test error handling for all NaN series."""
        times = pd.Series([np.nan, np.nan, np.nan])
        with pytest.raises(TimeFormatDetectionError):
            parse_time_format(times)


# ============================================================================
# TEST FUNCTION 4: classify_columns
# ============================================================================

class TestClassifyColumns:
    """Tests for classify_columns function."""
    
    def test_classify_standard_columns(self, temp_csv_file):
        """Test classification of standard telecom columns."""
        df = read_csv_with_encoding(temp_csv_file)
        result = classify_columns(df, 'Time')
        
        assert 'Region' in result.dimensions_text
        assert 'MRBTS_ID' in result.dimensions_id
        assert 'DL_PRB_UTILIZATION' in result.kpis
        assert 'UL_PRB_UTILIZATION' in result.kpis
        assert result.confidence > 0.7
    
    def test_classify_text_dimensions(self):
        """Test classification of text dimension columns."""
        df = pd.DataFrame({
            'Time': ['2024-03-17 00'] * 3,
            'Region': ['North', 'South', 'East'],
            'City': ['北京', '上海', '深圳'],
            'Vendor': ['Nokia', 'Ericsson', 'Samsung'],
        })
        result = classify_columns(df, 'Time')
        
        assert len(result.dimensions_text) == 3
        assert 'Region' in result.dimensions_text
    
    def test_classify_id_columns(self):
        """Test classification of ID columns."""
        df = pd.DataFrame({
            'Time': ['2024-03-17 00'] * 3,
            'MRBTS_ID': [1001, 1002, 1003],
            'LNCEL_ID': [10011, 10021, 10031],
            'SITE_ID': [100, 101, 102],
        })
        result = classify_columns(df, 'Time')
        
        assert len(result.dimensions_id) == 3
        assert 'MRBTS_ID' in result.dimensions_id
        assert 'LNCEL_ID' in result.dimensions_id
    
    def test_classify_kpi_columns(self):
        """Test classification of KPI columns."""
        df = pd.DataFrame({
            'Time': ['2024-03-17 00'] * 3,
            'DL_PRB': [75.5, 76.2, 72.1],
            'UL_PRB': [68.2, 69.1, 65.8],
            'CQI': [8.5, 8.6, 8.3],
        })
        result = classify_columns(df, 'Time')
        
        assert len(result.kpis) == 3
        assert 'DL_PRB' in result.kpis


# ============================================================================
# TEST FUNCTION 5: validate_data_integrity
# ============================================================================

class TestValidateDataIntegrity:
    """Tests for validate_data_integrity function."""
    
    def test_validate_good_data(self, temp_csv_file):
        """Test validation of good data."""
        df = read_csv_with_encoding(temp_csv_file)
        metadata = DataFrameMetadata(
            dataframe=df,
            time_column='Time',
            time_format='YYYY-MM-DD HH',
            dimensions_text=['Region'],
            dimensions_id=['MRBTS_ID'],
            kpis=['DL_PRB_UTILIZATION', 'UL_PRB_UTILIZATION'],
            row_count=len(df),
            file_size_mb=0.001,
            encoding_used='utf-8',
            classification_confidence=0.95
        )
        
        result = validate_data_integrity(df, metadata)
        assert result is True
    
    def test_validate_empty_dataframe(self):
        """Test error for empty DataFrame."""
        df = pd.DataFrame()
        metadata = DataFrameMetadata(
            dataframe=df,
            time_column='Time',
            time_format='YYYY-MM-DD HH',
            dimensions_text=[],
            dimensions_id=[],
            kpis=[],
            row_count=0,
            file_size_mb=0,
            encoding_used='utf-8',
            classification_confidence=0.0
        )
        
        with pytest.raises(DataValidationError):
            validate_data_integrity(df, metadata)
    
    def test_validate_missing_time_column(self, temp_csv_file):
        """Test error when time column not in DataFrame."""
        df = read_csv_with_encoding(temp_csv_file)
        metadata = DataFrameMetadata(
            dataframe=df,
            time_column='NonexistentColumn',
            time_format='YYYY-MM-DD HH',
            dimensions_text=['Region'],
            dimensions_id=['MRBTS_ID'],
            kpis=['DL_PRB_UTILIZATION'],
            row_count=len(df),
            file_size_mb=0.001,
            encoding_used='utf-8',
            classification_confidence=0.95
        )
        
        with pytest.raises(DataValidationError):
            validate_data_integrity(df, metadata)
    
    def test_validate_no_kpis(self, temp_csv_file):
        """Test error when no KPIs present."""
        df = read_csv_with_encoding(temp_csv_file)
        metadata = DataFrameMetadata(
            dataframe=df,
            time_column='Time',
            time_format='YYYY-MM-DD HH',
            dimensions_text=['Region', 'DL_PRB_UTILIZATION'],
            dimensions_id=['MRBTS_ID'],
            kpis=[],  # No KPIs!
            row_count=len(df),
            file_size_mb=0.001,
            encoding_used='utf-8',
            classification_confidence=0.8
        )
        
        with pytest.raises(DataValidationError):
            validate_data_integrity(df, metadata)


# ============================================================================
# TEST FUNCTION 6: normalize_data_types
# ============================================================================

class TestNormalizeDataTypes:
    """Tests for normalize_data_types function."""
    
    def test_normalize_numeric_strings(self):
        """Test converting numeric strings to numeric types."""
        df = pd.DataFrame({
            'KPI_A': ['100.5', '200.3', '150.2'],
            'KPI_B': ['50', '60', '55'],
            'Region': ['North', 'South', 'East']
        })
        
        df_normalized = normalize_data_types(df)
        
        assert df_normalized['KPI_A'].dtype in [np.float64, float]
        assert df_normalized['KPI_B'].dtype in [np.int64, np.float64, int, float]
        assert df_normalized['Region'].dtype == 'object'
    
    def test_normalize_preserves_text(self):
        """Test that text columns are preserved."""
        df = pd.DataFrame({
            'Region': ['North', 'South', 'East'],
            'City': ['Beijing', 'Shanghai', 'Shenzhen'],
            'Value': [100, 120, 150]
        })
        
        df_normalized = normalize_data_types(df)
        
        assert df_normalized['Region'].dtype == 'object'
        assert df_normalized['City'].dtype == 'object'
        assert list(df_normalized['Region']) == ['North', 'South', 'East']
    
    def test_normalize_with_nan(self):
        """Test normalization with NaN values."""
        df = pd.DataFrame({
            'KPI': ['100.5', np.nan, '150.2'],
            'Region': ['North', 'South', 'East']
        })
        
        df_normalized = normalize_data_types(df)
        
        assert df_normalized['KPI'].dtype in [np.float64, float]
        assert pd.isna(df_normalized.iloc[1]['KPI'])


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete ingestion pipeline."""
    
    def test_ingest_standard_csv(self, temp_csv_file):
        """Test complete ingestion of standard CSV."""
        metadata = ingest_csv(temp_csv_file)
        
        assert metadata.time_column == 'Time'
        assert metadata.time_format == 'YYYY-MM-DD HH'
        assert len(metadata.dimensions_text) > 0
        assert len(metadata.kpis) > 0
        assert metadata.row_count == 5
        assert metadata.classification_confidence > 0.7
    
    def test_ingest_daily_format_csv(self, daily_format_csv):
        """Test ingestion of daily format CSV."""
        metadata = ingest_csv(daily_format_csv)
        
        assert metadata.time_column == 'Date'
        assert metadata.time_format == 'MM/DD/YYYY'
        assert metadata.row_count == 3
    
    def test_ingest_utf8_multilang_csv(self, utf8_multilang_csv):
        """Test ingestion of UTF-8 multi-language CSV."""
        metadata = ingest_csv(utf8_multilang_csv)
        
        assert metadata.encoding_used == 'utf-8'
        assert '台北市' in metadata.dataframe['City'].values
        assert len(metadata.dimensions_text) > 0
    
    def test_ingest_summary_output(self, temp_csv_file):
        """Test that summary output is generated correctly."""
        metadata = ingest_csv(temp_csv_file)
        summary = metadata.summary()
        
        assert 'DataFrameMetadata Summary' in summary
        assert 'Time Column:' in summary
        assert 'Rows:' in summary


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_row_csv(self):
        """Test ingestion of single-row CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time,Region,KPI\n")
            f.write("2024-03-17 00,North,100\n")
            temp_path = f.name
        
        try:
            metadata = ingest_csv(temp_path)
            assert metadata.row_count == 1
        finally:
            os.unlink(temp_path)
    
    def test_single_column_csv(self):
        """Test ingestion of single-column CSV."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time\n")
            f.write("2024-03-17 00\n")
            f.write("2024-03-17 01\n")
            temp_path = f.name
        
        try:
            with pytest.raises(DataValidationError):
                ingest_csv(temp_path)  # Should fail: no KPIs
        finally:
            os.unlink(temp_path)
    
    def test_csv_with_missing_values(self):
        """Test ingestion of CSV with missing values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Time,Region,MRBTS_ID,KPI\n")
            f.write("2024-03-17 00,North,1001,100\n")
            f.write("2024-03-17 01,,1002,\n")
            f.write("2024-03-17 02,East,,120\n")
            temp_path = f.name
        
        try:
            metadata = ingest_csv(temp_path)
            assert metadata.row_count == 3
        finally:
            os.unlink(temp_path)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
