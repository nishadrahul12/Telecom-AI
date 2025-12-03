# Phase 2 Module 5: Correlation Module - Step-by-Step Implementation Guide

## 📋 Overview

This guide walks you through implementing **Phase2_Module5_CorrelationModule** - a production-ready correlation analysis engine that:
- Calculates Pearson correlation matrices between all KPI pairs
- Ranks Top 3 correlations per KPI (by absolute value)
- Generates interactive heatmap data for visualization
- Meets <5 second performance target on 100K rows

---

## 🎯 Quick Start Checklist

```
Phase 2 Module 5 Deliverables:
├── ✅ correlation_module.py (Core implementation)
├── ✅ test_correlation_module.py (Comprehensive unit tests)
├── ✅ IMPLEMENTATION_STEPS.md (This file)
├── ✅ README.md (User documentation)
├── ✅ API_CONTRACT.md (Technical specification)
└── ✅ requirements.txt (Dependencies)
```

---

## 📁 Directory Structure

```
Phase2_Module5_CorrelationModule/
├── src/
│   └── correlation_module.py          # Main implementation
├── tests/
│   ├── test_correlation_module.py      # Unit tests
│   └── conftest.py                     # Pytest fixtures
├── README.md                           # User guide
├── API_CONTRACT.md                     # Technical specification
├── requirements.txt                    # Dependencies
└── IMPLEMENTATION_STEPS.md             # This guide
```

---

## 🚀 Step 1: Setup & File Creation

### 1.1 Create Directory Structure

```bash
# Navigate to your Telecom-AI project
cd C:\Users\Rahul\Desktop\Projects\Telecom-AI

# Create Phase 2 Module 5
mkdir -p Phase2_Module5_CorrelationModule\src
mkdir -p Phase2_Module5_CorrelationModule\tests
```

### 1.2 Create requirements.txt

**File**: `Phase2_Module5_CorrelationModule/requirements.txt`

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
pydantic>=1.10.0
plotly>=5.10.0
pytest>=7.0.0
pytest-cov>=4.0.0
```

### 1.3 Verify Dependencies

```powershell
# Install dependencies
pip install -r Phase2_Module5_CorrelationModule/requirements.txt

# Verify imports
python -c "import pandas, numpy, scipy, pydantic, plotly; print('✅ All dependencies installed')"
```

---

## 📝 Step 2: Implement Core Module (correlation_module.py)

### 2.1 Copy the Complete Implementation

**File**: `Phase2_Module5_CorrelationModule/src/correlation_module.py`

See: **correlation_module.py** (attached)

### 2.2 Key Implementation Details

#### Pydantic Models (Input/Output Validation)

```python
class CorrelationAnalysisResult(BaseModel):
    """Validated output structure"""
    correlation_matrix: List[List[float]]      # N×N matrix
    top_3_per_kpi: Dict[str, List[Dict]]       # Top 3 per KPI
    heatmap_data: Dict[str, Any]               # Plotly-ready data
    processing_time_ms: float                   # Execution time
```

#### Core Functions

1. **`calculate_correlation_matrix()`**
   - Uses pandas `.corr()` method (vectorized Pearson)
   - Handles NaN values with row-wise dropna
   - Returns symmetric N×N matrix

2. **`get_top_3_correlations()`**
   - Vectorized ranking using NumPy argsort
   - Excludes self-correlation (diagonal)
   - Sorts by ABSOLUTE value (captures both positive/negative)

3. **`generate_heatmap_data()`**
   - Creates Plotly-compatible heatmap JSON
   - Color coding: Green (r>0.7), Red (r<-0.7), Gray (neutral)
   - Returns {"z": values, "x": names, "y": names, ...}

---

## 🧪 Step 3: Implement Unit Tests

### 3.1 Copy Test Suite

**File**: `Phase2_Module5_CorrelationModule/tests/test_correlation_module.py`

See: **test_correlation_module.py** (attached)

### 3.2 Test Coverage

```
✅ 10+ Test Cases:
├── test_correlation_matrix_is_symmetric
├── test_correlation_diagonal_is_one
├── test_correlation_range
├── test_top_3_ranked_by_absolute_value
├── test_self_correlation_excluded
├── test_perfect_positive_correlation
├── test_perfect_negative_correlation
├── test_no_correlation
├── test_performance_target
└── test_heatmap_data_structure
```

### 3.3 Pytest Fixtures

**File**: `Phase2_Module5_CorrelationModule/tests/conftest.py`

```python
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    """Generate random sample KPI data"""
    np.random.seed(42)
    return pd.DataFrame({
        'KPI_A': np.random.randn(1000),
        'KPI_B': np.random.randn(1000),
        'KPI_C': np.random.randn(1000),
    })

@pytest.fixture
def correlated_data():
    """Generate highly correlated KPI data"""
    np.random.seed(42)
    base = np.random.randn(1000)
    return pd.DataFrame({
        'KPI_X': base,
        'KPI_Y': base + 0.1 * np.random.randn(1000),  # 0.99 correlation
        'KPI_Z': -base,  # -1.0 correlation
    })
```

---

## ▶️ Step 4: Run Tests & Verify

### 4.1 Run Unit Tests

```powershell
# Navigate to module directory
cd Phase2_Module5_CorrelationModule

# Run all tests
pytest tests/test_correlation_module.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test
pytest tests/test_correlation_module.py::test_correlation_matrix_is_symmetric -v
```

### 4.2 Expected Output

```
======================= test session starts =======================
collected 10 items

test_correlation_module.py::test_correlation_matrix_is_symmetric PASSED
test_correlation_module.py::test_correlation_diagonal_is_one PASSED
test_correlation_module.py::test_correlation_range PASSED
test_correlation_module.py::test_top_3_ranked_by_absolute_value PASSED
test_correlation_module.py::test_self_correlation_excluded PASSED
test_correlation_module.py::test_perfect_positive_correlation PASSED
test_correlation_module.py::test_perfect_negative_correlation PASSED
test_correlation_module.py::test_no_correlation PASSED
test_correlation_module.py::test_performance_target PASSED
test_correlation_module.py::test_heatmap_data_structure PASSED

======================= 10 passed in 0.45s =======================
```

### 4.3 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import Error | Ensure `src/` is in PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:./src"` |
| Test Timeout | Reduce data size in fixture or increase timeout: `pytest --timeout=10` |
| NaN in Results | Ensure input DataFrame has no all-NaN columns |

---

## 📊 Step 5: Integration with Upstream Modules

### 5.1 Consume Filtering Engine Output

```python
from filtering_engine import FilteringEngine
from correlation_module import CorrelationAnalyzer

# Step 1: Load and filter data
filter_engine = FilteringEngine()
sampled_df = filter_engine.filter(
    filepath='data.csv',
    region='N1',
    carrier='L700'
)

# Step 2: Extract KPI columns
kpi_columns = sampled_df.select_dtypes(include=[np.number]).columns.tolist()

# Step 3: Run correlation analysis
analyzer = CorrelationAnalyzer()
result = analyzer.analyze(
    df=sampled_df,
    kpi_columns=kpi_columns
)

# Step 4: Access results
print(result.top_3_per_kpi['RACH_stp_att'])
# Output: [
#   {"target_kpi": "RRC_stp_att", "correlation_score": 0.89, ...},
#   {"target_kpi": "E-RAB_SAtt", "correlation_score": 0.76, ...},
#   {"target_kpi": "Inter-freq_HO_att", "correlation_score": 0.65, ...}
# ]
```

### 5.2 Pass Results to Phase 3 (Forecasting)

```python
# Use Top 3 as exogenous variables for ARIMAX forecasting
top_3_correlated = result.top_3_per_kpi['RACH_stp_att']
exogenous_kpis = [item['target_kpi'] for item in top_3_correlated]

# Pass to forecasting module
forecast_result = forecasting_module.forecast(
    df=sampled_df,
    target_kpi='RACH_stp_att',
    exogenous_variables=exogenous_kpis
)
```

---

## 📈 Step 6: Integration Tests

### 6.1 Create Integration Test

**File**: `tests/integration_test_correlation.py`

```python
import pytest
import pandas as pd
import numpy as np
from src.correlation_module import CorrelationAnalyzer

def test_correlation_with_real_data():
    """Integration test with real telecom CSV"""
    # Load sample data
    df = pd.read_csv('../../../Sample_KPI_Data.csv')
    
    # Extract KPI columns (numeric only)
    kpi_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Run analysis
    analyzer = CorrelationAnalyzer()
    result = analyzer.analyze(df=df, kpi_columns=kpi_columns)
    
    # Assertions
    assert len(result.top_3_per_kpi) > 0
    assert result.processing_time_ms < 5000  # <5 seconds
    assert result.heatmap_data is not None
    print(f"✅ Integration test passed - {len(kpi_columns)} KPIs analyzed in {result.processing_time_ms:.2f}ms")

if __name__ == '__main__':
    test_correlation_with_real_data()
```

### 6.2 Run Integration Test

```powershell
python tests/integration_test_correlation.py
```

---

## 📚 Step 7: Documentation

### 7.1 Create README.md

**File**: `README.md` (See attached README.md)

### 7.2 Create API_CONTRACT.md

**File**: `API_CONTRACT.md` (See attached API_CONTRACT.md)

### 7.3 Generate Code Documentation

```powershell
# Generate HTML documentation (optional)
pip install pdoc
pdoc src/correlation_module.py -o docs/

# View documentation
start docs/correlation_module.html
```

---

## 🔄 Step 8: Performance Verification

### 8.1 Benchmark Against Success Criteria

```python
import time
import pandas as pd
import numpy as np
from src.correlation_module import CorrelationAnalyzer

def benchmark():
    """Verify <5s performance on 100K rows"""
    
    # Test Case 1: 10 KPIs, 100K rows
    print("Benchmarking: 10 KPIs × 100K rows")
    df = pd.DataFrame(np.random.randn(100_000, 10), 
                     columns=[f'KPI_{i}' for i in range(10)])
    
    start = time.time()
    analyzer = CorrelationAnalyzer()
    result = analyzer.analyze(df, kpi_columns=df.columns.tolist())
    elapsed = time.time() - start
    
    print(f"✅ Processed {len(df)} rows in {elapsed*1000:.0f}ms")
    assert elapsed < 5.0, f"Performance target missed: {elapsed}s > 5s"
    
    # Test Case 2: 50 KPIs, 100K rows (larger matrix)
    print("\nBenchmarking: 50 KPIs × 100K rows")
    df = pd.DataFrame(np.random.randn(100_000, 50), 
                     columns=[f'KPI_{i}' for i in range(50)])
    
    start = time.time()
    result = analyzer.analyze(df, kpi_columns=df.columns.tolist())
    elapsed = time.time() - start
    
    print(f"✅ Processed {len(df)} rows, {len(df.columns)} columns in {elapsed*1000:.0f}ms")
    assert elapsed < 5.0, f"Performance target missed: {elapsed}s > 5s"

if __name__ == '__main__':
    benchmark()
```

### 8.2 Run Benchmark

```powershell
python benchmark_correlation.py
```

---

## ✅ Step 9: Pre-GitHub Checklist

Before pushing to GitHub, execute this final verification:

```powershell
# 1. Run all tests
pytest tests/ -v --cov=src --cov-report=term-missing

# 2. Check code quality
flake8 src/ --max-line-length=100
black src/ --check

# 3. Type checking
mypy src/ --ignore-missing-imports

# 4. Run integration tests
python tests/integration_test_correlation.py

# 5. Run performance benchmark
python benchmark_correlation.py

# 6. Verify file structure
tree . /I "__pycache__"
# Expected:
# Phase2_Module5_CorrelationModule/
# ├── src/
# │   └── correlation_module.py
# ├── tests/
# │   ├── test_correlation_module.py
# │   ├── conftest.py
# │   └── integration_test_correlation.py
# ├── README.md
# ├── API_CONTRACT.md
# ├── requirements.txt
# └── IMPLEMENTATION_STEPS.md
```

---

## 🔐 Step 10: GitHub Push Process

### 10.1 Stage Changes

```powershell
cd C:\Users\Rahul\Desktop\Projects\Telecom-AI

# Check status
git status

# Stage all module files
git add Phase2_Module5_CorrelationModule/

# Verify staging
git status  # Should show "Changes to be committed"
```

### 10.2 Commit with Clear Message

```powershell
git commit -m "Phase 2 Module 5: Correlation Analysis - Production Ready

Features:
- Pearson correlation matrix calculation (vectorized)
- Top 3 correlation ranking per KPI
- Plotly heatmap data generation
- <5s performance on 100K rows

Testing:
- 10 comprehensive unit tests
- Integration tests with real data
- Performance benchmarks

Documentation:
- Complete API contract
- User guide with examples
- Type hints on all functions

Closes #[issue-number]"
```

### 10.3 Push to GitHub

```powershell
# Push to your branch
git push origin main

# Or create a feature branch (if team repo)
git checkout -b phase2-module5-correlation
git push -u origin phase2-module5-correlation

# Then create Pull Request on GitHub UI
```

### 10.4 Verify Push Success

```powershell
# Check remote
git log --oneline -3 --graph

# Expected output:
# * abc1234 Phase 2 Module 5: Correlation Analysis - Production Ready
# * def5678 Phase 2 Module 4: Anomaly Detection - Production Ready
# * ghi9012 Phase 2 Module 3: Filtering Engine - Production Ready
```

---

## 🎓 Common Questions & Troubleshooting

### Q1: How do I test with my own CSV?

```python
import pandas as pd
from src.correlation_module import CorrelationAnalyzer

df = pd.read_csv('my_data.csv')
kpi_cols = df.select_dtypes(include=[np.number]).columns.tolist()

analyzer = CorrelationAnalyzer()
result = analyzer.analyze(df, kpi_columns=kpi_cols)
print(result.top_3_per_kpi)
```

### Q2: What if correlation takes >5 seconds?

Check your data size:
```python
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"Matrix size: {len(df.columns)}×{len(df.columns)}")
```

If >100K rows and >50 KPIs, consider:
1. Increasing smart sampling from Phase 2 Module 3
2. Selecting subset of KPIs
3. Computing on larger machine

### Q3: How do I integrate with forecasting?

```python
# Extract Top 3 correlations
for kpi, top_3_list in result.top_3_per_kpi.items():
    exogenous_vars = [item['target_kpi'] for item in top_3_list]
    # Pass to forecasting module
```

### Q4: Can I customize the heatmap colors?

Yes! Modify `generate_heatmap_data()`:
```python
colorscale = 'RdBu'  # Red-Blue diverging
# or
colorscale = 'Viridis'  # Sequential
```

---

## 📞 Support & Next Steps

✅ **Module 5 Complete!**

**Next Phase**: Phase 3 Module 1 - Time Series Forecasting
- Will consume Top 3 correlations as exogenous variables
- Build ARIMA/ARIMAX models
- Generate forecasts with confidence intervals

**Questions?** Refer to:
- `README.md` - User documentation
- `API_CONTRACT.md` - Technical specification
- Code docstrings - Inline documentation

---

## 📋 Checklist: Module Closure

- [ ] All tests passing (10/10)
- [ ] Performance <5s verified
- [ ] README completed
- [ ] API contract documented
- [ ] Type hints on all functions
- [ ] Docstrings on all functions
- [ ] Integration test passing
- [ ] Code formatted (Black)
- [ ] Linting clean (Flake8)
- [ ] GitHub push successful

---

**Status**: ✅ Ready for Production  
**Created**: Dec 2025  
**Author**: Rahul / AI Assistant  
**Version**: 1.0.0
