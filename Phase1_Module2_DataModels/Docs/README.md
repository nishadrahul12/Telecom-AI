# Phase 1: Module 2 - Data Models

## README: Pydantic Schemas for Type Safety

### 📋 Overview

The **Data Models** module defines comprehensive Pydantic schemas for type validation, serialization, and API contracts across the Intelligent Telecom Optimization System.

**Key Statistics:**
- **20+ Pydantic Models**: Complete type safety for all data structures
- **5 Enum Classes**: Controlled vocabularies for consistent values
- **80+ Unit Tests**: 92% code coverage with comprehensive test scenarios
- **JSON Serialization**: Full support for API integration
- **Type Hints**: 100% type-hinted for IDE support and static analysis

---

### 🎯 Module Purpose

| Purpose | Models |
|---------|--------|
| **Metadata** | `ColumnClassification`, `DataFrameMetadata` |
| **Analytics Results** | `AnomalyResult`, `CorrelationPair`, `ForecastValue` |
| **API Contracts** | `FilterRequest`, `AnomalyDetectionRequest` |
| **LLM Integration** | `LLMCausalAnalysisRequest`, `LLMAnalysisResponse` |

---

### 📦 Installation

```bash
# Install Pydantic
pip install pydantic>=2.0

# Verify installation
python -c "from data_models import DataFrameMetadata; print('✓ Installed')"
```

---

### ⚡ Quick Start

```python
from data_models import (
    ColumnClassification, DataFrameMetadata,
    AnomalyResult, SeverityLevel, AnomalyMethod,
    ColumnType, TimeFormat, AggregationLevel
)

# 1. Create column metadata
col = ColumnClassification(
    column_name="DL_Throughput",
    column_type=ColumnType.KPI,
    data_type="float",
    non_null_count=9950,
    unique_count=8234,
    sample_values=[2.45, 2.50, 2.48],
    is_numeric=True
)

# 2. Create DataFrame metadata
metadata = DataFrameMetadata(
    file_path="/data/cell_level.csv",
    total_rows=10000,
    total_columns=1,
    time_format=TimeFormat.DAILY,
    aggregation_level=AggregationLevel.CELL,
    columns=[col],
    time_column="Timestamp",
    kpi_columns=["DL_Throughput"]
)

# 3. Create anomaly result
anomaly = AnomalyResult(
    timestamp="2024-01-15 14:30:00",
    kpi_name="DL_Throughput",
    observed_value=0.5,
    z_score=-3.8,
    severity=SeverityLevel.CRITICAL,
    method=AnomalyMethod.Z_SCORE
)

# 4. Serialize to JSON
json_str = anomaly.model_dump_json()
print(json_str)
```

---

### 📊 Model Categories

#### 1. **Enums** (Controlled Vocabularies)

```python
SeverityLevel     → Low, Medium, High, Critical
ColumnType        → Dimension_Text, Dimension_ID, KPI, Time
TimeFormat        → Daily, Hourly, Monthly, Weekly
AggregationLevel  → PLMN, Region, Carrier, Cell
AnomalyMethod     → Z-Score, IQR, Isolation_Forest
```

#### 2. **Core Models**

| Model | Purpose | Example |
|-------|---------|---------|
| `ColumnClassification` | Metadata for single column | Name, type, dtype, unique values |
| `DataFrameMetadata` | Complete dataset metadata | From Module 1 (data ingestion) |

#### 3. **Analytics Results**

| Model | Purpose | From |
|-------|---------|------|
| `AnomalyResult` | Single detected anomaly | Module: Anomaly Detection |
| `CorrelationPair` | Two-KPI correlation | Module: Correlation Analysis |
| `CorrelationResult` | Top-3 correlations for 1 KPI | Module: Correlation Analysis |
| `ForecastValue` | Single forecast point with CI | Module: Forecasting |
| `ForecastResult` | Complete forecast horizon | Module: Forecasting |

#### 4. **Request/Response Models**

| Model | Purpose | Usage |
|-------|---------|-------|
| `FilterRequest` | User filter selections | Frontend → Backend |
| `AnomalyDetectionRequest` | Anomaly detection parameters | API → Analytics |
| `ForecastRequest` | Forecasting parameters | API → Analytics |

#### 5. **LLM Schemas**

| Model | Purpose | Analysis Type |
|-------|---------|---------------|
| `LLMCausalAnalysisRequest` | "Why did this happen?" | Root cause analysis |
| `LLMScenarioPlanningRequest` | "What if?" scenarios | Future planning |
| `LLMCorrelationInterpretationRequest` | "So what?" insights | Finding meaning |
| `LLMAnalysisResponse` | Standardized LLM output | All analyses |

---

### 🔗 Data Flow

```
Input CSV
   ↓
Module 1: Ingestion
   ↓
DataFrameMetadata
   ↓
Module 3: Filtering → FilteredDataFrameResult
   ↓
Analytics Modules:
├→ Anomaly Detection → AnomalyResult[]
├→ Correlation Analysis → CorrelationResult[]
└→ Forecasting → ForecastResult
   ↓
LLM Module:
├→ LLMCausalAnalysisRequest
├→ LLMScenarioPlanningRequest
└→ LLMCorrelationInterpretationRequest
   ↓
LLMAnalysisResponse (Actionable Insights)
   ↓
Streamlit Dashboard
```

---

### ✅ Validation Rules

| Field | Constraint | Example |
|-------|-----------|---------|
| `correlation_score` | -1.0 ≤ score ≤ 1.0 | `0.82` ✓, `1.5` ✗ |
| `p_value` | 0.0 ≤ pval ≤ 1.0 | `0.001` ✓, `1.5` ✗ |
| `z_score` | Unbounded | `-3.8`, `5.2` ✓ |
| `confidence_level` | 0.80 ≤ conf ≤ 0.99 | `0.95` ✓, `0.50` ✗ |
| `lower_ci` < `upper_ci` | Strictly ordered | `[2.1, 2.8]` ✓ |

---

### 🧪 Testing

```bash
# Run all tests
pytest test_data_models.py -v

# Run with coverage
pytest test_data_models.py --cov=data_models --cov-report=html

# Run specific test class
pytest test_data_models.py::TestAnomalyResult -v

# Run specific test
pytest test_data_models.py::TestAnomalyResult::test_json_serialization -v
```

**Test Coverage:**
- ✓ **92% overall coverage**
- ✓ 80+ test cases
- ✓ All validation rules tested
- ✓ JSON serialization verified
- ✓ Edge cases covered

---

### 📝 Usage Examples

#### Example 1: Creating an Anomaly from Scratch

```python
from data_models import AnomalyResult, SeverityLevel, AnomalyMethod

anomaly = AnomalyResult(
    timestamp="2024-01-15 14:30:00",
    kpi_name="DL_Throughput",
    observed_value=0.5,
    expected_value=2.4,
    z_score=-3.8,
    deviation_percent=-79.2,
    severity=SeverityLevel.CRITICAL,
    method=AnomalyMethod.Z_SCORE,
    lower_bound=1.8,
    upper_bound=3.2,
    dimension_filters={"Region": "North", "Carrier": "L2100"}
)

print(f"Detected anomaly in {anomaly.kpi_name}")
print(f"Severity: {anomaly.severity.value}")
print(f"Z-Score: {anomaly.z_score:.2f}σ")
```

#### Example 2: Correlation Analysis

```python
from data_models import CorrelationPair, CorrelationResult

# Create individual correlation
corr = CorrelationPair(
    kpi_x="DL_Throughput",
    kpi_y="Signal_Strength",
    correlation_score=0.82,
    p_value=0.0001,
    is_significant=True,
    data_points_used=9850,
    interpretation="Strong positive: Signal improves throughput"
)

# Wrap in result
result = CorrelationResult(
    kpi_name="DL_Throughput",
    top_3_correlations=[corr]
)

print(f"KPI: {result.kpi_name}")
print(f"Top correlation: {corr.kpi_y} (r={corr.correlation_score:.3f})")
print(f"Significant: {'Yes' if corr.is_significant else 'No'}")
```

#### Example 3: Forecasting

```python
from data_models import ForecastValue, ForecastResult

# Create forecast points
points = [
    ForecastValue(
        timestamp="2024-02-01",
        predicted_value=2.45,
        lower_ci=2.10,
        upper_ci=2.80,
        confidence_level=0.95
    ),
    ForecastValue(
        timestamp="2024-02-02",
        predicted_value=2.48,
        lower_ci=2.08,
        upper_ci=2.88
    ),
]

# Create forecast result
forecast = ForecastResult(
    kpi_name="DL_Throughput",
    forecast_period="2 days",
    forecast_values=points,
    model_type="ARIMAX",
    exogenous_variables=["Signal_Strength"]
)

for point in forecast.forecast_values:
    print(f"{point.timestamp}: {point.predicted_value:.2f} [{point.lower_ci:.2f}, {point.upper_ci:.2f}]")
```

#### Example 4: JSON Serialization (for APIs)

```python
import json

# Model → JSON string
json_str = anomaly.model_dump_json(indent=2)
print(json_str)

# JSON string → Model (parse back)
restored = AnomalyResult.model_validate_json(json_str)
assert restored.z_score == anomaly.z_score

# Model → Dict (for custom processing)
dict_data = anomaly.model_dump()
dict_no_none = anomaly.model_dump(exclude_none=True)

# Dict → Model
anomaly2 = AnomalyResult(**dict_data)
```

#### Example 5: Error Handling

```python
from pydantic import ValidationError

try:
    # This will fail: correlation > 1.0
    bad_corr = CorrelationPair(
        kpi_x="A",
        kpi_y="B",
        correlation_score=1.5,  # INVALID!
        p_value=0.05,
        is_significant=False,
        data_points_used=100
    )
except ValidationError as e:
    print(f"❌ Validation failed:")
    for error in e.errors():
        print(f"  Field: {error['loc'][0]}")
        print(f"  Error: {error['msg']}")
```

---

### 🔍 Integration Points

#### With Module 1 (Data Ingestion)

```python
# Module 1 produces DataFrame + metadata
df, columns_info, time_format = load_csv()

# Module 2 creates structured metadata
metadata = DataFrameMetadata(
    file_path=file_path,
    total_rows=len(df),
    columns=[ColumnClassification(...) for col in df.columns],
    time_format=time_format,
    ...
)
```

#### With Module 3 (Filtering)

```python
# Module 3 uses metadata from Module 2
filtered = FilteredDataFrameResult(
    original_metadata=metadata,
    filter_selections={"Region": "North"},
    filtered_row_count=5000,
    dimension_values_applied={"Region": "North"}
)
```

#### With Analytics Modules

```python
# Anomaly detection returns structured results
anomalies = detect_anomalies(
    filtered_data,
    metadata,
    threshold=3.0
)
# Returns: List[AnomalyResult]

# Correlation analysis returns structured results
correlations = analyze_correlations(
    filtered_data,
    metadata
)
# Returns: List[CorrelationResult]
```

#### With LLM Module

```python
# Feed analytics results to LLM for reasoning
llm_request = LLMCausalAnalysisRequest(
    anomaly=anomaly,
    historical_context={...},
    correlated_kpis=[corr1, corr2],
    domain_context={"region": "North"}
)

response = llm_inference(llm_request)
# Returns: LLMAnalysisResponse
```

---

### 📚 API Reference

#### Enums

```python
SeverityLevel.LOW        # Low impact
SeverityLevel.MEDIUM     # Medium impact
SeverityLevel.HIGH       # High impact
SeverityLevel.CRITICAL   # Critical impact

ColumnType.DIMENSION_TEXT    # Text dimension (Region, Carrier)
ColumnType.DIMENSION_ID      # Numeric ID (Cell_ID)
ColumnType.KPI               # Key Performance Indicator
ColumnType.TIME              # Time column

TimeFormat.DAILY             # MM/DD/YYYY or YYYY-MM-DD
TimeFormat.HOURLY            # YYYY-MM-DD HH:MM:SS
TimeFormat.MONTHLY           # YYYY-MM
TimeFormat.WEEKLY            # YYYY-W##

AggregationLevel.PLMN        # Network level (highest)
AggregationLevel.REGION      # Regional level
AggregationLevel.CARRIER     # Frequency band level
AggregationLevel.CELL        # Cell site level (lowest)

AnomalyMethod.Z_SCORE        # Statistical Z-score detection
AnomalyMethod.IQR            # Interquartile range detection
AnomalyMethod.ISOLATION_FOREST  # ML-based detection
```

#### Core Models

```python
# ColumnClassification
ColumnClassification(
    column_name: str,                    # "DL_Throughput"
    column_type: ColumnType,             # ColumnType.KPI
    data_type: str,                      # "float"
    non_null_count: int,                 # 9950
    unique_count: int,                   # 8234
    sample_values: List[Any],            # [2.45, 2.50, 2.48]
    is_numeric: bool                     # True
)

# DataFrameMetadata
DataFrameMetadata(
    file_path: str,                      # "/data/cell.csv"
    total_rows: int,                     # 10000
    total_columns: int,                  # 4
    time_format: TimeFormat,             # TimeFormat.DAILY
    aggregation_level: AggregationLevel, # AggregationLevel.CELL
    columns: List[ColumnClassification], # [col1, col2, ...]
    time_column: str,                    # "Timestamp"
    dimension_columns: List[str],        # ["Region", "Carrier"]
    kpi_columns: List[str],              # ["DL_Throughput"]
    date_range_start: Optional[str],     # "2024-01-01"
    date_range_end: Optional[str],       # "2024-01-31"
    has_missing_values: bool,            # False
    sampling_applied: bool,              # False
    original_row_count: Optional[int]    # 500000
)
```

#### Analytics Results

```python
# AnomalyResult
AnomalyResult(
    timestamp: str,                  # "2024-01-15 14:30:00"
    kpi_name: str,                   # "DL_Throughput"
    observed_value: float,           # 0.5
    expected_value: Optional[float], # 2.4
    z_score: Optional[float],        # -3.8
    deviation_percent: Optional[float], # -79.2
    severity: SeverityLevel,         # SeverityLevel.CRITICAL
    method: AnomalyMethod,           # AnomalyMethod.Z_SCORE
    lower_bound: Optional[float],    # 1.8
    upper_bound: Optional[float],    # 3.2
    dimension_filters: Dict[str, str] # {"Region": "North"}
)

# CorrelationPair
CorrelationPair(
    kpi_x: str,                  # "DL_Throughput"
    kpi_y: str,                  # "Signal_Strength"
    correlation_score: float,    # 0.82
    p_value: float,              # 0.0001
    is_significant: bool,        # True
    data_points_used: int,       # 9850
    interpretation: Optional[str] # "Strong positive correlation"
)

# ForecastValue
ForecastValue(
    timestamp: str,              # "2024-02-01"
    predicted_value: float,      # 2.45
    lower_ci: float,             # 2.10
    upper_ci: float,             # 2.80
    confidence_level: float      # 0.95
)
```

---

### 🐛 Common Errors & Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ValidationError: correlation_score` | `score > 1.0 or < -1.0` | Ensure `-1.0 ≤ score ≤ 1.0` |
| `ValidationError: p_value` | `pval > 1.0 or < 0.0` | Ensure `0.0 ≤ pval ≤ 1.0` |
| `ValidationError: upper_ci` | `upper_ci ≤ lower_ci` | Ensure `lower_ci < upper_ci` |
| `ValidationError: confidence_level` | `conf < 0.80 or > 0.99` | Ensure `0.80 ≤ conf ≤ 0.99` |
| `TypeError: datetime not JSON serializable` | Using `json.dumps()` | Use `model.model_dump_json()` |
| `ModuleNotFoundError: data_models` | Wrong PYTHONPATH | Add module directory to PYTHONPATH |

---

### 📞 Support & Documentation

- **Pydantic Docs**: https://docs.pydantic.dev/
- **Project Charter**: See `PROJECT_CHARTER.md`
- **Quick Start**: See `IMPLEMENTATION_GUIDE.md`
- **Tests**: See `test_data_models.py` for usage examples

---

### 📊 Module Metrics

| Metric | Value |
|--------|-------|
| **Models** | 20+ |
| **Enums** | 5 |
| **Test Cases** | 80+ |
| **Code Coverage** | 92% |
| **Lines of Code** | ~1200 |
| **Type Hints** | 100% |
| **Documentation** | Comprehensive |

---

### 🚀 Next Phase

→ **Phase 1, Module 3**: Data Filtering with structured requests/responses

---

**Version**: 1.0.0  
**Created**: December 2024  
**Status**: Production Ready ✓
