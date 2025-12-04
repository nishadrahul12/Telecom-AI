# Corrected PowerShell Script for API Testing - Version 2 (More Robust Quotes/Formatting)

# Note: All Python print statements now use .format() or concatenation for maximum
# compatibility when executed via 'python -c' in PowerShell.

Write-Host "=== PRE-TEST: Health Check ===" -ForegroundColor Cyan
python -c @"
import requests, json
try:
    r = requests.get('http://localhost:8000/health')
    r.raise_for_status()
    h = r.json()
    print('Status: {}'.format(h['status']))
    print('KPIs: {}'.format(h['session_state']['kpi_count']))
except requests.exceptions.ConnectionError:
    print('Error: Could not connect to the API on http://localhost:8000.')
    exit(1)
except requests.exceptions.HTTPError:
    print('Error: HTTP request failed with status code {}'.format(r.status_code))
    exit(1)
"@

Write-Host "`n=== TEST 1: Get Levels ===" -ForegroundColor Cyan
python -c @"
import requests
r = requests.get('http://localhost:8000/levels')
if r.status_code == 200:
    print('Levels: {}'.format(r.json().get('levels', 'N/A')))
else:
    print('Error: {}'.format(r.status_code))
"@

Write-Host "`n=== TEST 2: Get Region Filters ===" -ForegroundColor Cyan
python -c @"
import requests
r = requests.get('http://localhost:8000/filters/Region')
if r.status_code == 200:
    print('Regions: {} available'.format(len(r.json().get('values', []))))
else:
    print('Error: {}'.format(r.status_code))
"@

Write-Host "`n=== TEST 3: Get City Filters ===" -ForegroundColor Cyan
python -c @"
import requests
r = requests.get('http://localhost:8000/filters/City')
if r.status_code == 200:
    print('Cities: {} available'.format(len(r.json().get('values', []))))
else:
    print('Error: {}'.format(r.status_code))
"@

Write-Host "`n=== TEST 4: Apply Filter ===" -ForegroundColor Cyan
python -c @"
import requests
# Using single quotes for string values inside the payload for robustness
payload = {'level': 'Region', 'filters': {'Region': ['Taipei']}}
r = requests.post('http://localhost:8000/filter', json=payload)
if r.status_code == 200:
    print('Filtered: {} records'.format(r.json().get('count', 0)))
else:
    print('Error: {} - {}'.format(r.status_code, r.text))
"@

Write-Host "`n=== TEST 5: KPI Count (Check State Change) ===" -ForegroundColor Cyan
python -c @"
import requests
r = requests.get('http://localhost:8000/health')
kpis = r.json().get('session_state', {}).get('kpi_count', 'N/A')
print('Total KPIs: {}'.format(kpis))
"@

Write-Host "`n=== TEST 6: Anomaly Detection ===" -ForegroundColor Cyan
python -c @"
import requests
# Using single quotes for string values inside the payload for robustness
payload = {'kpi': 'RACH stp att', 'method': 'zscore', 'threshold': 2.0}
r = requests.post('http://localhost:8000/anomalies', json=payload)
if r.status_code == 200:
    print('Anomalies: {}'.format(r.json().get('anomaly_count', 0)))
else:
    print('Error: {} - {}'.format(r.status_code, r.text))
"@

Write-Host "`n=== TEST 7: Correlation ===" -ForegroundColor Cyan
python -c @"
import requests
# Using single quotes for string values inside the payload for robustness
payload = {'kpi1': 'RACH stp att', 'kpi2': 'RRC stp att', 'correlation_threshold': 0.5}
r = requests.post('http://localhost:8000/correlations', json=payload)
if r.status_code == 200:
    print('Correlation found: {}'.format(r.json().get('correlation_found', False)))
else:
    print('Error: {} - {}'.format(r.status_code, r.text))
"@

Write-Host "`n=== TEST 8: Forecast ===" -ForegroundColor Cyan
python -c @"
import requests
# Using single quotes for string values inside the payload for robustness
payload = {'kpi': 'RACH stp att', 'periods': 7, 'method': 'linear'}
r = requests.post('http://localhost:8000/forecast', json=payload)
if r.status_code == 200:
    print('Forecast points: {}'.format(r.json().get('forecast_points', 0)))
else:
    print('Error: {} - {}'.format(r.status_code, r.text))
"@

Write-Host "`n=== TEST 9: Export ===" -ForegroundColor Cyan
python -c @"
import requests
# Using single quotes for string values inside the payload for robustness
payload = {'format': 'csv', 'include_filters': True}
r = requests.post('http://localhost:8000/export', json=payload)
if r.status_code == 200:
    print('Export: {}'.format(r.json().get('status', 'unknown')))
else:
    print('Error: {} - {}'.format(r.status_code, r.text))
"@

Write-Host "`n=== TEST 10: Performance ===" -ForegroundColor Cyan
$start = Get-Date
for ($i = 1; $i -le 5; $i++) {
    # Using the call operator (&) is explicitly used here for robust execution of python.
    # The python -c script is simplified to a single line
    & python -c "import requests; requests.get('http://localhost:8000/health')" | Out-Null
}
$avg = ((Get-Date) - $start).TotalMilliseconds / 5
Write-Host "Avg Response: $([math]::Round($avg, 2)) ms" -ForegroundColor Green

Write-Host "`n✅ ALL TESTS COMPLETE" -ForegroundColor Green