# ANALYSIS.md - Sample Data Analysis & Metadata Templates

## Data Structure Analysis for Sample_KPI_Data.csv

### File Overview

**File Name**: Sample_KPI_Data.csv
**Size**: ~66 KB (uncompressed)
**Rows**: ~1,000 data rows (+ 1 header)
**Columns**: 71 total
**Encoding**: UTF-8 (includes [translate:中文] characters)
**Time Format**: MM/DD/YYYY (e.g., 3/1/2024)
**Data Level**: Cell-level (most granular)

---

## Column Classification

### 1. Time Column (1 column)

| Column | Type | Format | Notes |
|--------|------|--------|-------|
| TIME | DateTime | MM/DD/YYYY | Single date (3/1/2024) in sample |

### 2. Text Dimensions (5 columns)

These are categorical attributes for filtering/grouping:

| Column | Type | Sample Values | Count | Notes |
|--------|------|---|---|---|
| REGION | Text | N1, N2 | 2-3 | Network region code |
| CITY | Text | [translate:台北市], [translate:高雄市] | ~5 | Geographic city |
| DISTRICT | Text | [translate:中正區], [translate:大安區] | ~10 | Sub-city district |
| SITENAME | Text | 中正杭州富邦, 中正南昌 | ~20 | Cell site name |
| CARRIER_NAME | Text | L700, L1800, L2100, L2600, L900 | 5 | Frequency band name |

**Unicode Handling**: ✅ All [translate:中文] characters properly encoded, no special handling needed

### 3. Numeric Dimensions (4 columns)

These are categorical IDs (numeric but used for filtering, not aggregation):

| Column | Type | Range | Notes |
|--------|------|-------|-------|
| BAND_NUMBER | Integer | 1-28 | Frequency band number (matches CARRIER_NAME) |
| MRBTS_ID | Integer | 100000-100999 | Radio base station ID |
| LNBTS_ID | Integer | 100000-100999 | Local radio base station ID |
| LNCEL_ID | Integer | 100-999 | Cell ID within base station |

**Usage**: Primarily for detailed Cell-level filtering, aggregation by these dimensions not recommended

### 4. KPI Columns (61 columns)

These are performance metrics/measurements (numeric, aggregatable):

#### Category: RACH (Random Access Channel)
- RACH stp att (Attempt count)
- RACH Stp Completion SR (Success rate %)
- Comp Cont based RACH stp SR

#### Category: RRC (Radio Resource Control)
- RRC stp att (Attempt count)
- RRC conn stp SR (Success rate %)
- E-UTRAN avg RRC conn UEs
- RRC_CONNECTED_UE_AVG

#### Category: E-RAB Setup
- E-RAB SAtt (Setup attempts)
- E-UTRAN E-RAB stp SR (Success rate %)
- ERAB DR; RAN View

#### Category: Handover
- Inter-freq HO att / E-UTRAN Inter-Freq HO SR
- E-UTRAN Intra-Freq HO SR
- ATT_INTRA_ENB_HO / Intra eNB HO SR
- inter eNB E-UTRAN HO SR X2

#### Category: Throughput
- E-UTRAN avg IP sched thp DL; QCI8
- E-UTRAN avg IP sched thp UL; QCI8

#### Category: PRB (Physical Resource Block)
- E-UTRAN Avg PRB usage per TTI DL
- Avg PRB usage per TTI UL

#### Category: Volume
- PDCP SDU Volume; DL / UL
- DL MAC PDU VOL SCell
- RLC_PDU_DL_VOL_CA_PCELL
- RLC_VOL_DL_DRB_SCELL_UE
- Total DL RLC Volume 1

#### Category: Quality
- Average CQI (Channel Quality Indicator)
- Avg SINR for PUCCH / PUSCH (Signal-to-Interference Ratio)
- Avg MCS PDSCH / PUSCH trans (Modulation & Coding Scheme)
- Avg UE PWR Headroom PUSCH

#### Category: QCI-specific (Quality Class Indicator 1)
- ERAB Stp Att; QCI1 / ERAB Stp SR; QCI1
- E-UTRAN PDCP SDU DL/UL QCI1 LR
- QCI1 BLER; DL / UL (Block Error Rate)
- Avg PDCP SDU Delay DL QCI1
- AVG_HARQ_DELAY_QCI1_UL

#### Category: Network Performance
- Avg Latency DL
- Max PDCP Thr DL / UL

#### Category: Signal Strength
- RSRP_Avg (Reference Signal Received Power)
- Measured_RSRP_AVE

#### Category: CA (Carrier Aggregation)
- CA_UE_Scell_CONF_SCell_AVG
- NUM_CA_UE_SERV_CELL_AVG
- NUM_NON_CA_UE_SERV_CELL_AVG
- Avg DL nonGBR IP thp CA active UEs (2/3/4 CCS)

#### Category: UE (User Equipment)
- Avg nbr UEs in serv cell
- Avg UEs activ 3 SCells DL
- Avg nbr UEs activ 4 Scells
- VoLTE traffic
- LTE Interference Power (Ave PUSCH)
- Avg HARQ_DELAY related metrics

**Value Ranges**:
- Percentages (SR, BLER, PRB usage): 0-100
- Counts (Attempts, UEs): 0-1,000,000+
- Quality metrics (CQI, SINR, MCS): 0-30
- Throughput: 0-1,000,000 Mbps
- Delay/Latency: 0-5000 ms
- Signal strength: -120 to -50 dBm
- Some columns: 0 or NULL when no data

**Missing Data**:
- Some KPI columns have NULL/NaN values
- Handled gracefully by filtering engine (excluded from statistics)
- Example: QCI-specific KPIs may be 0 when QCI traffic = 0

---

## Data Aggregation Levels

### PLMN Level (Highest Aggregation)

**What**: Network-wide summary
**Grouping**: No dimensions (single row per time period)
**Use**: Network-wide KPI trends, high-level dashboards
**Filters Available**: None (already fully aggregated)
**Recommended for**: Executive reporting, capacity planning

```
Example:
TIME: 3/1/2024
RACH stp att: 50,000,000  (aggregated across all regions/sites)
RRC conn stp SR: 99.5%    (network average)
...
```

### Region Level

**What**: Geographic region summary
**Grouping**: By REGION
**Use**: Regional performance analysis
**Filters Available**: REGION, CITY, text dimensions
**Recommended for**: Regional manager dashboards, inter-region comparison

```
Example:
TIME: 3/1/2024
REGION: N1
RACH stp att: 15,000,000  (N1 region only)
RRC conn stp SR: 98.9%
...
```

### Carrier Level

**What**: Frequency band summary
**Grouping**: By CARRIER_NAME and REGION
**Use**: Band-specific performance analysis
**Filters Available**: REGION, CARRIER_NAME, text dimensions
**Recommended for**: Spectrum optimization, band comparison

```
Example:
TIME: 3/1/2024
REGION: N1
CARRIER_NAME: L700
RACH stp att: 5,000,000
RRC conn stp SR: 99.2%
...
```

### Cell Level (Lowest Aggregation, Your Data)

**What**: Individual cell site performance
**Grouping**: By REGION, CITY, SITENAME, CARRIER_NAME, LNCEL_ID (cell ID)
**Use**: Detailed troubleshooting, cell optimization, anomaly detection
**Filters Available**: All text_dimensions + all numeric_dimensions
**Recommended for**: Network engineers, detailed analysis, anomaly detection

```
Example:
TIME: 3/1/2024
REGION: N1
CITY: 台北市
DISTRICT: 中正區
SITENAME: 中正杭州富邦
CARRIER_NAME: L700
LNCEL_ID: 111
RACH stp att: 227,799
RRC conn stp SR: 99.86%
...
```

---

## Metadata Templates

### Template 1: Cell-Level Data (Your Sample Data)

Use this for detailed site-level analysis:

```python
from filtering_engine import DataFrameMetadata

CELL_LEVEL_METADATA = DataFrameMetadata(
    text_dimensions=[
        'REGION',           # N1, N2, etc.
        'CITY',            # 台北市, 高雄市
        'DISTRICT',        # 中正區, 大安區
        'SITENAME',        # Cell site name
        'CARRIER_NAME'     # L700, L1800, L2100, L2600, L900
    ],
    numeric_dimensions=[
        'MRBTS_ID',        # Radio base station ID
        'LNBTS_ID',        # Local base station ID
        'LNCEL_ID',        # Cell ID
        'BAND_NUMBER'      # Frequency band number
    ],
    kpi_columns=[
        # RACH metrics
        'RACH stp att',
        'RACH Stp Completion SR',
        'Comp Cont based RACH stp SR',
        
        # RRC metrics
        'RRC stp att',
        'RRC conn stp SR',
        'E-UTRAN avg RRC conn UEs',
        'RRC_CONNECTED_UE_AVG',
        
        # E-RAB metrics
        'E-RAB SAtt',
        'E-UTRAN E-RAB stp SR',
        'ERAB DR; RAN View',
        'E-RAB Stp Att; QCI8',
        'ERAB Stp SR; QCI1',
        'ERAB DR; RAN View; QCI1',
        
        # Handover metrics
        'Inter-freq HO att',
        'E-UTRAN Inter-Freq HO SR',
        'E-UTRAN Intra-Freq HO SR',
        'ATT_INTRA_ENB_HO',
        'Intra eNB HO SR',
        'inter eNB E-UTRAN HO SR X2',
        
        # Throughput metrics
        'E-UTRAN avg IP sched thp DL; QCI8',
        'E-UTRAN avg IP sched thp UL; QCI8',
        
        # PRB metrics
        'E-UTRAN Avg PRB usage per TTI DL',
        'Avg PRB usage per TTI UL',
        
        # Volume metrics
        'PDCP SDU Volume; DL',
        'PDCP SDU Volume; UL',
        'DL MAC PDU VOL SCell',
        'CA_UE_Scell_CONF_SCell_AVG',
        'RLC_PDU_DL_VOL_CA_PCELL',
        'RLC_VOL_DL_DRB_SCELL_UE',
        'RLC_VOL_DL_SRB_DRB_NON_CA',
        'PDCP_PDU_X2_DL_SCG',
        'Total DL RLC Volume 1',
        
        # Quality metrics
        'Average CQI',
        'Avg SINR for PUCCH',
        'Avg SINR for PUSCH',
        'Avg MCS PDSCH trans',
        'Avg MCS PUSCH trans',
        'Avg UE PWR Headroom PUSCH',
        'LTE Interference Power (Ave PUSCH)',
        
        # QCI1 specific
        'E-UTRAN PDCP SDU DL QCI1 LR',
        'E-UTRAN PDCP SDU UL QCI1 LR',
        'QCI1 BLER; DL',
        'QCI1 BLER; UL',
        'Avg PDCP SDU Delay DL QCI1',
        'AVG_HARQ_DELAY_QCI1_UL',
        
        # Network performance
        'Avg Latency DL',
        'Max PDCP Thr DL',
        'Max PDCP Thr UL',
        
        # Signal strength
        'RSRP_Avg',
        'Measured_RSRP_AVE',
        
        # UE metrics
        'NUM_CA_UE_SERV_CELL_AVG',
        'NUM_NON_CA_UE_SERV_CELL_AVG',
        'Avg nbr UEs in serv cell',
        'VoLTE traffic',
        'Avg UEs activ 3 SCells DL',
        'Avg nbr UEs activ 4 Scells',
        'Avg DL nonGBR IP thp CA active UEs 2 CCS',
        'Avg DL nonGBR IP thp CA active UEs 3 CCS',
        'Avg DL nonGBR IP thp CA active UEs 4 CCS'
    ],
    time_column='TIME',
    data_level='Cell'
)
```

### Template 2: Carrier-Level Data (Aggregated)

Use this when data is aggregated by carrier:

```python
CARRIER_LEVEL_METADATA = DataFrameMetadata(
    text_dimensions=[
        'REGION',
        'CITY',
        'CARRIER_NAME'
    ],
    numeric_dimensions=[
        'BAND_NUMBER'
    ],
    kpi_columns=[
        # Core metrics
        'RACH stp att', 'RACH Stp Completion SR',
        'RRC stp att', 'RRC conn stp SR',
        'E-RAB SAtt', 'E-UTRAN E-RAB stp SR',
        'E-UTRAN Inter-Freq HO SR', 'E-UTRAN Intra-Freq HO SR',
        'Average CQI', 'Avg SINR for PUCCH', 'Avg SINR for PUSCH',
        'RSRP_Avg', 'Avg Latency DL'
    ],
    time_column='TIME',
    data_level='Carrier'
)
```

### Template 3: Region-Level Data (Aggregated)

Use this for regional summaries:

```python
REGION_LEVEL_METADATA = DataFrameMetadata(
    text_dimensions=[
        'REGION',
        'CITY'
    ],
    numeric_dimensions=[],
    kpi_columns=[
        # Core metrics only
        'RACH stp att', 'RACH Stp Completion SR',
        'RRC stp att', 'RRC conn stp SR',
        'E-RAB SAtt', 'E-UTRAN E-RAB stp SR',
        'E-UTRAN Inter-Freq HO SR',
        'Average CQI', 'Avg Latency DL',
        'Avg nbr UEs in serv cell'
    ],
    time_column='TIME',
    data_level='Region'
)
```

### Template 4: PLMN-Level Data (Network-wide)

Use this for network summaries:

```python
PLMN_LEVEL_METADATA = DataFrameMetadata(
    text_dimensions=[],
    numeric_dimensions=[],
    kpi_columns=[
        # High-level metrics only
        'RACH Stp Completion SR',
        'RRC conn stp SR',
        'E-UTRAN E-RAB stp SR',
        'Average CQI',
        'Avg Latency DL',
        'RSRP_Avg'
    ],
    time_column='TIME',
    data_level='PLMN'
)
```

---

## Data Quality Notes

### Missing Values

Several columns have NaN/0 values:
- QCI-specific metrics (when QCI=1 traffic is 0)
- Some throughput columns (when no data transfer)
- CA metrics (when Carrier Aggregation not active)

**Handling**: Filtering engine excludes NaN from statistics automatically

### Negative Values

Some columns can have negative values:
- RSRP values: Range from -120 to -50 dBm (all negative, normal)
- Lat/Long coordinates: Negative for Southern/Western hemisphere

### Range Anomalies

Check data distribution:
- Success rates: Should be 80-100% (higher is better)
- Attempts: Usually 1000s to millions
- Latency: Should be <100ms (red flag if >500ms)
- Signal strength (RSRP): Should be better than -110 dBm

### Encoding

✅ UTF-8 with [translate:中文] characters properly supported
- No special handling needed
- Filtering, grouping, and aggregation work correctly
- Compatible with all downstream modules

---

## Filtering Examples by Data Level

### Cell Level: Multi-dimensional Filter

```python
# Show only L700 and L1800 bands in regions N1 & N2
filters = {
    'REGION': ['N1', 'N2'],           # Multiple regions
    'CARRIER_NAME': ['L700', 'L1800'] # Multiple bands
}
# Result: All sites in N1/N2 with L700/L1800 bands
```

### Carrier Level: Single Carrier Analysis

```python
# Show all L700 performance across network
filters = {
    'CARRIER_NAME': ['L700']
}
# Result: L700 metrics from all regions/sites
```

### Region Level: Geographic Focus

```python
# Show only Taipei region performance
filters = {
    'REGION': ['N1'],
    'CITY': ['台北市']
}
# Result: All sites in Taipei region
```

---

## KPI Interpretation Guide

| KPI | Target | Red Flag | Notes |
|-----|--------|----------|-------|
| RACH Completion SR | >95% | <90% | Setup success rate - critical |
| RRC Conn SR | >99% | <95% | Radio connection success |
| E-RAB Setup SR | >95% | <90% | Bearer setup success |
| Handover SR | >98% | <95% | Call continuity during move |
| Average CQI | >10 | <5 | Channel quality (0=poor, 15=excellent) |
| Avg SINR | >10dB | <0dB | Signal quality (higher is better) |
| Avg Latency | <50ms | >200ms | User experience critical |
| PRB Usage | 50-80% | <20% or >95% | Spectrum efficiency |
| RSRP | >-100dBm | <-110dBm | Signal coverage strength |

---

## Sample Queries (Using Filtering Engine)

```python
from filtering_engine import apply_filters_and_sample, DataFrameMetadata

# Query 1: Problem Sites Analysis
# Show cells with poor RACH completion (<90%)
filters = {'REGION': ['N1']}
result = apply_filters_and_sample(df, metadata, 'Cell', filters)
poor_sites = result.filtered_dataframe[result.filtered_dataframe['RACH Stp Completion SR'] < 90]

# Query 2: Carrier Comparison
# Compare L700 vs L1800 performance
for carrier in ['L700', 'L1800']:
    filters = {'CARRIER_NAME': [carrier]}
    result = apply_filters_and_sample(df, metadata, 'Cell', filters)
    print(f"{carrier}: Avg RACH SR = {result.filtered_dataframe['RACH Stp Completion SR'].mean():.2f}%")

# Query 3: Network-wide Sampling
# Get representative sample of entire network (for anomaly detection)
result = apply_filters_and_sample(df, metadata, 'Cell')
sampled_df = result.filtered_dataframe
# Now sample includes cells from all regions/carriers proportionally
```

---

## Creating Custom Metadata

For your own datasets:

```python
# Step 1: Load your data
df = pd.read_csv('your_file.csv')

# Step 2: Identify column types
print(df.dtypes)
print(df.head())

# Step 3: Classify columns
text_dims = ['col1', 'col2', ...]  # Categorical text
numeric_dims = ['col3', 'col4', ...]  # Numeric IDs
kpi_cols = ['col5', 'col6', ...]  # Metrics/measurements
time_col = 'TIME'  # DateTime column

# Step 4: Create metadata
custom_metadata = DataFrameMetadata(
    text_dimensions=text_dims,
    numeric_dimensions=numeric_dims,
    kpi_columns=kpi_cols,
    time_column=time_col,
    data_level='Cell'  # Or your data level
)

# Step 5: Test with filtering engine
result = apply_filters_and_sample(df, custom_metadata, 'Cell')
print(f"✓ Metadata created successfully")
```

---

## Performance Recommendations

For optimal performance with this data:

| Data Size | Recommended Settings | Notes |
|-----------|---------------------|-------|
| 1k rows (sample) | No sampling (factor=1) | All data fits in memory easily |
| 50k rows | Light sampling (factor=5) | Good for dashboards, quick analysis |
| 500k rows | Moderate sampling (factor=50) | Recommended for most analysis |
| 1M+ rows | Heavy sampling (factor=100) | Archive data, long-term trends |

**Memory Usage**:
- Each row: ~600 bytes (71 columns)
- 1k rows: 0.6 MB
- 50k rows: 30 MB
- 500k rows: 300 MB
- 1M rows: 600 MB

---

## Next Steps

1. **Load your actual data**:
   ```python
   df = pd.read_csv('Sample_KPI_Data.csv')
   ```

2. **Use template for your data level**:
   ```python
   metadata = CELL_LEVEL_METADATA  # or your custom
   ```

3. **Test filtering engine**:
   ```python
   result = apply_filters_and_sample(df, metadata, 'Cell', {'REGION': ['N1']})
   ```

4. **Proceed to Phase 2, Module 1** (Correlation Analysis)

---

**Data Analysis Complete** ✅
**Ready for Production** ✅
