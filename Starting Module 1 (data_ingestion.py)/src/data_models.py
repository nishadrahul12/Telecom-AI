"""
data_models.py
==============
Pydantic models for data validation and schema definition.

This module defines the core data structures used throughout the
Intelligent Telecom Optimization System. It provides type-safe,
validated data containers with clear contracts.

Author: AI Assistant
Phase: 1 (Foundation)
"""

from typing import List
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, ConfigDict
import pandas as pd


class DataFrameMetadata(BaseModel):
    """
    Output contract for data ingestion module.
    
    Contains metadata about ingested CSV file including detected columns,
    formats, and file statistics.
    
    Attributes:
        dataframe: Raw Pandas DataFrame with all rows
        time_column: Detected time column name (e.g., "Time", "Date")
        time_format: Detected format (e.g., "YYYY-MM-DD HH", "MM/DD/YYYY")
        dimensions_text: List of text columns (Region, Carrier, City, etc.)
        dimensions_id: List of ID columns (MRBTS_ID, LNCEL_ID, SITE_ID)
        kpis: List of numeric KPI columns (metrics)
        row_count: Total number of rows in DataFrame
        file_size_mb: File size in megabytes
        encoding_used: Actual encoding used to read file
        classification_confidence: Confidence score (0.0-1.0) for column classification
    
    Example:
        >>> metadata = DataFrameMetadata(
        ...     dataframe=df,
        ...     time_column="Time",
        ...     time_format="YYYY-MM-DD HH",
        ...     dimensions_text=["Region", "Carrier"],
        ...     dimensions_id=["MRBTS_ID"],
        ...     kpis=["DL_PRB_UTILIZATION", "UL_PRB_UTILIZATION"],
        ...     row_count=50000,
        ...     file_size_mb=2.5,
        ...     encoding_used="utf-8",
        ...     classification_confidence=0.95
        ... )
    """
    
    dataframe: pd.DataFrame
    time_column: str
    time_format: str
    dimensions_text: List[str] = Field(default_factory=list)
    dimensions_id: List[str] = Field(default_factory=list)
    kpis: List[str] = Field(default_factory=list)
    row_count: int
    file_size_mb: float
    encoding_used: str
    classification_confidence: float = Field(ge=0.0, le=1.0)
    
    # NEW CONFIG (replace old Config class):
    model_config = ConfigDict(arbitrary_types_allowed=True)
        
    @field_validator('time_format')
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate time format is not empty."""
        if not v or len(v) < 3:
            raise ValueError("time_format must be a non-empty string (e.g., 'YYYY-MM-DD HH')")
        return v
    
    @field_validator('row_count')
    @classmethod
    def validate_row_count(cls, v: int) -> int:
        """Validate row count is non-negative."""
        if v < 0:
            raise ValueError("row_count must be non-negative")
        return v
    
    @field_validator('file_size_mb')
    @classmethod
    def validate_file_size(cls, v: float) -> float:
        """Validate file size is non-negative."""
        if v < 0:
            raise ValueError("file_size_mb must be non-negative")
        return v
    
    def to_dict(self) -> dict:
        """Convert metadata to dictionary (excluding DataFrame)."""
        return {
            "time_column": self.time_column,
            "time_format": self.time_format,
            "dimensions_text": self.dimensions_text,
            "dimensions_id": self.dimensions_id,
            "kpis": self.kpis,
            "row_count": self.row_count,
            "file_size_mb": self.file_size_mb,
            "encoding_used": self.encoding_used,
            "classification_confidence": self.classification_confidence,
        }
    
    def summary(self) -> str:
        """Return human-readable summary of metadata."""
        return f"""
DataFrameMetadata Summary:
  Time Column: {self.time_column} (Format: {self.time_format})
  Rows: {self.row_count} | File Size: {self.file_size_mb:.2f} MB
  Text Dimensions: {len(self.dimensions_text)} columns
  ID Dimensions: {len(self.dimensions_id)} columns
  KPIs: {len(self.kpis)} metrics
  Encoding: {self.encoding_used}
  Classification Confidence: {self.classification_confidence:.1%}
        """.strip()


class ColumnClassification(BaseModel):
    """
    Internal model for column classification results.
    
    Attributes:
        dimensions_text: Text columns
        dimensions_id: ID columns
        kpis: Numeric KPI columns
        confidence: Classification confidence (0.0-1.0)
    """
    dimensions_text: List[str] = Field(default_factory=list)
    dimensions_id: List[str] = Field(default_factory=list)
    kpis: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
