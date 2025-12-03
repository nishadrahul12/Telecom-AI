"""Profile anomaly detection performance"""

import pandas as pd
import numpy as np
import time
from anomaly_detection import AnomalyDetectionEngine

# Create test data (100k rows)
np.random.seed(42)
n_rows = 100000
data = {
    'TIME': pd.date_range('2020-01-01', periods=n_rows, freq='h').strftime('%Y-%m-%d %H:00'),
    'KPI_A': np.random.normal(1000, 100, n_rows),
    'KPI_B': np.random.normal(500, 50, n_rows),
}
df = pd.DataFrame(data)

# Benchmark
engine = AnomalyDetectionEngine()
start = time.time()
report = engine.generate_report(
    df=df,
    time_column='TIME',
    kpi_columns=['KPI_A', 'KPI_B']
)
elapsed = time.time() - start

print(f"Performance Report:")
print(f"  Rows processed: {n_rows:,}")
print(f"  KPIs analyzed: 2")
print(f"  Time elapsed: {elapsed:.3f}s")
print(f"  Rows/second: {n_rows/elapsed:,.0f}")
print(f"  Anomalies found: {report['total_anomalies']}")
