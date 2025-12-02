"""Data Models Module with Filtering Engine Integration"""

# Existing imports
from dataclasses import dataclass
from typing import List, Dict

# NEW: Import filtering engine classes
from ..Phase2_Module3_FilteringEngine.filtering_engine import (
    FilteredDataFrameResult,
    DataFrameMetadata,
    DataLevel
)

# Your existing data models...
# (Keep everything as is)

__all__ = [
    'DataFrameMetadata',      # NEW
    'FilteredDataFrameResult', # NEW
    'DataLevel',              # NEW
    # ... other exports
]
