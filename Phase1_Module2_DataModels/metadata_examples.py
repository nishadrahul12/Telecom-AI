"""Example metadata configurations for different data levels"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Phase2_Module3_FilteringEngine'))
from filtering_engine import DataFrameMetadata

# Cell-level telecom data
CELL_LEVEL_METADATA = DataFrameMetadata(
    text_dimensions=[
        'REGION', 'CITY', 'DISTRICT', 'SITENAME', 
        'CARRIER_NAME', 'MRBTS_NAME', 'LNBTS_NAME'
    ],
    numeric_dimensions=['MRBTS_ID', 'LNBTS_ID', 'LNCEL_ID', 'BAND_NUMBER'],
    kpi_columns=[
        'RACH stp att', 'RACH Stp Completion SR',
        'RRC conn stp SR', 'E-RAB SAtt',
        'E-UTRAN Inter-Freq HO SR', 'E-UTRAN Intra-Freq HO SR',
        # ... add all your KPIs
    ],
    time_column='TIME',
    data_level='Cell'
)

# Region-level (aggregated)
REGION_LEVEL_METADATA = DataFrameMetadata(
    text_dimensions=['REGION', 'CITY'],
    numeric_dimensions=[],
    kpi_columns=['RACH stp att', 'RRC conn stp SR', 'E-RAB SAtt'],
    time_column='TIME',
    data_level='Region'
)

# PLMN-level (highest aggregation)
PLMN_LEVEL_METADATA = DataFrameMetadata(
    text_dimensions=[],
    numeric_dimensions=[],
    kpi_columns=['RACH stp att', 'RRC conn stp SR'],
    time_column='TIME',
    data_level='PLMN'
)
print("✅ All metadata templates created successfully!")