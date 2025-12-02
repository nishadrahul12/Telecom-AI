# Data Ingestion Module - Simple User Guide

## What is This?

Imagine you have an **Excel file with telecom data** (like call logs, network usage, customer info). This tool **automatically reads and organizes** that data for analysis. It's like having a smart assistant who:

- Opens your files
- Understands dates and numbers
- Separates important info
- Fixes common mistakes

---

## Simple 3-Step Process

### Step 1: Prepare Your Data File

You need a **CSV file** (Excel saved as CSV - comma-separated values).

Example:
```
Date,Time,Network,Calls,Duration
2024-01-15,09:00,4G,145,2340
2024-01-15,10:00,4G,168,2451
2024-01-15,11:00,5G,201,2890
```

**Important:** Make sure your file has:
- A header row (column names at the top)
- At least one time column (date/time)
- Data values (numbers or text)

### Step 2: Run the Tool

```python
from src.data_ingestion import ingest_csv

# Point to your file
result = ingest_csv('path/to/your/file.csv')
```

That's it! The tool does the rest.

### Step 3: Get Your Results

```python
# See what the tool found
print(result.summary())

# Get the organized data
data = result.dataframe

# See the columns
print("Time column:", result.time_column)
print("Measurement columns:", result.kpis)
print("Info columns:", result.dimensions_text)
```

---

## What Does It Find?

The tool automatically **labels each column**:

| Label | Example | Used For |
|-------|---------|----------|
| **Time Column** | "2024-01-15 09:00" | When something happened |
| **Measurement (KPI)** | Calls, Duration, Revenue | Numbers to track |
| **Info (Dimension)** | Network, Location, Customer | Additional details |
| **ID** | Customer_ID, Cell_Tower_ID | Unique identifiers |

---

## Example: Real Telecom Data

### Your File:
```
timestamp,cell_tower,region,calls_processed,avg_duration,revenue_usd
2024-01-15T09:00,T001,North,145,234,1240.50
2024-01-15T10:00,T001,North,168,241,1450.80
2024-01-15T11:00,T001,North,201,289,1890.25
```

### What The Tool Does:

1. Finds "timestamp" as the **time column**
2. Understands date format: **2024-01-15T09:00**
3. Labels columns:
   - **Measurements:** calls_processed, avg_duration, revenue_usd
   - **Info:** cell_tower, region
4. Converts numbers properly (so you can do math later)

### What You Get:
```python
result.summary()

TIME COLUMN: timestamp
TIME RANGE: 2024-01-15 09:00 to 2024-01-15 11:00

MEASUREMENTS (3 columns):
  - calls_processed
  - avg_duration  
  - revenue_usd

INFORMATION (2 columns):
  - cell_tower
  - region

Ready for analysis!
```

---

## Common File Formats Supported

The tool works with these types of files:

**Saved as CSV with:**
- English text
- Chinese/Japanese characters
- Numbers (with decimals or commas)
- Dates in any common format
- Time in 12-hour or 24-hour format

**File sizes:** Small (KB) to Large (100MB+)

---

## Common Mistakes & Fixes

### Problem: File Not Found
**Solution:** Make sure the file path is correct
```python
# WRONG
ingest_csv('my_file.csv')

# RIGHT (full path)
ingest_csv('C:/Users/YourName/Documents/my_file.csv')
```

### Problem: Time Column Not Found
**Solution:** Make sure you have a column with dates/times named something like:
- "date", "time", "timestamp", "Date_Time"
- Not "T001", "value123" (too random)

### Problem: Numbers treated as text
**Solution:** No decimals or special characters in numbers
- `234` ✓ Good
- `234.50` ✓ Good
- `$234.50` ✗ Remove $
- `234,567` ✓ Commas OK

---

## What Happens Behind The Scenes?

```
YOUR FILE
    |
    v
Check encoding (handles 9+ languages)
    |
    v
Find date/time column
    |
    v
Understand date format (daily/hourly)
    |
    v
Classify each column (measurement/info/ID)
    |
    v
Convert numbers to numbers (not text)
    |
    v
Check for problems/mistakes
    |
    v
ORGANIZED DATA READY TO USE
```

---

## What's Next?

Now that your data is organized, you can:

- **Filter** specific dates or regions
- **Count** total calls, revenue, etc.
- **Compare** different time periods
- **Find patterns** and problems
- **Export** clean data to Excel/PDF

---

## Quick Reference

| Action | Code |
|--------|------|
| **Load file** | `result = ingest_csv('file.csv')` |
| **See summary** | `print(result.summary())` |
| **Get data** | `df = result.dataframe` |
| **Get time column** | `result.time_column` |
| **Get measurements** | `result.kpis` |
| **Get info columns** | `result.dimensions_text` |
| **Get IDs** | `result.dimensions_id` |

---

## Need Help?

If something doesn't work:

1. Check file is saved as **CSV format**
2. Make sure file has **column headers**
3. Check file path is **correct**
4. File should have **date/time column**

---

**Version:** 1.0  
**Last Updated:** December 2, 2024  
**Status:** Production Ready
