# Phase 4 Module 8 - API Gateway: User Guide & README

**Version**: 1.0.0  
**Status**: Production Ready  
**Last Updated**: 2025-12-04

---

## 📖 What is This Module?

Think of **Phase 4 Module 8** as the **"Control Center"** of your telecom analysis system.

Imagine you have a team of specialists:
- One analyzes anomalies (weird things that happen in your network)
- One studies relationships between different metrics
- One predicts the future based on patterns
- One gives expert analysis using AI

**Module 8's job**: Connect everyone together so they work as ONE unified system.

---

## 🎯 What Can You Do With It?

### 1. **Upload Your Data** 📤
Upload a CSV file with your telecom data
```
Example: Your_Telecom_Data.csv
```
- System automatically understands your data structure
- Works with ANY column names (Region, City, Carrier, etc.)
- Handles files from a few KB to 500+ MB

### 2. **Browse Available Data** 🔍
See what dimensions you can filter by
```
Example: View all Regions, Cities, Carriers available in your data
```

### 3. **Filter Your Data** 🎯
Select specific slices of your data
```
Example: "Show me only North Region data for Carrier A"
```

### 4. **Find Unusual Patterns** 🚨
Automatically detect anomalies (problems/outliers)
```
Example: "Which KPIs have unusual values?"
```

### 5. **Understand Relationships** 🔗
See which metrics influence each other
```
Example: "Does signal strength affect call quality?"
```

### 6. **Predict Future Values** 📈
Forecast what will happen next week/month
```
Example: "What will bandwidth usage be in 7 days?"
```

### 7. **Get Expert Analysis** 🤖
Get AI-powered insights about your data
```
Example: "Why did this anomaly occur? What should we do?"
```

---

## 🚀 How to Use It (Step-by-Step)

### **Step 1: Start the API**

Open your terminal/PowerShell:
```bash
cd C:\Users\Rahul\Desktop\Projects\Telecom-AI\Phase4_Module8_API
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

You'll see:
```
✓ Application startup complete
✓ Uvicorn running on http://0.0.0.0:8000
```

### **Step 2: Upload Your CSV File**

Use any tool that can send HTTP requests (Python, Postman, curl):

```bash
curl -X POST \
  -F "file=@Your_Telecom_Data.csv" \
  http://localhost:8000/upload
```

**Response**:
```json
{
  "status": "success",
  "file_info": {
    "filename": "Your_Telecom_Data.csv",
    "size_bytes": 1500000
  },
  "dataframe_metadata": {
    "row_count": 50000,
    "dimensions_text": ["REGION", "CITY", "CARRIER"],
    "kpis": ["SIGNAL_STRENGTH", "BANDWIDTH", "LATENCY"]
  }
}
```

### **Step 3: Check System Health**

Make sure everything is working:

```bash
curl http://localhost:8000/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-12-04T16:00:00",
  "components": {
    "fastapi": "ready",
    "data_ingestion": "ready",
    "filtering_engine": "ready",
    "anomaly_detection": "ready",
    "forecasting_module": "ready"
  }
}
```

### **Step 4: View Available Filters**

See what dimensions you can filter by:

```bash
curl http://localhost:8000/filters/Region
```

**Response**:
```json
{
  "status": "success",
  "data_level": "Region",
  "text_dimensions": ["REGION", "CITY", "DISTRICT"],
  "unique_values": {
    "REGION": ["North", "South", "East", "West"],
    "CITY": ["New York", "Los Angeles", "Chicago"],
    "CARRIER": ["Carrier_A", "Carrier_B"]
  }
}
```

### **Step 5: Apply Filters** (Optional)

Select specific data to analyze:

```bash
curl -X POST http://localhost:8000/apply-filters \
  -H "Content-Type: application/json" \
  -d '{
    "data_level": "Region",
    "filters": {
      "REGION": ["North"],
      "CARRIER": ["Carrier_A"]
    },
    "sampling_strategy": "smart"
  }'
```

### **Step 6: Run Analysis**

#### Get Anomalies:
```bash
curl http://localhost:8000/anomalies
```

#### Get Correlations:
```bash
curl http://localhost:8000/correlation
```

#### Get Forecast:
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "target_kpi": "SIGNAL_STRENGTH",
    "forecast_horizon": 7
  }'
```

---

## 📁 File Structure

```
Phase4_Module8_API/
├── api.py                 ← Main API code
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
├── .gitignore             ← Files to ignore on GitHub
└── Sample_KPI_Data.csv    ← Example data
```

---

## 🔧 Requirements

**Python 3.8+** with these libraries:
```
fastapi==0.104.0
uvicorn==0.24.0
pandas==2.0.0
numpy==1.24.0
pydantic==2.5.0
```

Install them:
```bash
pip install -r requirements.txt
```

---

## 📊 Understanding the API Endpoints

| Endpoint | Method | Purpose | Example |
|----------|--------|---------|---------|
| `/health` | GET | Check system status | Health check |
| `/upload` | POST | Upload CSV file | Load your data |
| `/levels` | GET | Get data levels available | PLMN, Region, Carrier, Cell |
| `/filters/{level}` | GET | See what you can filter by | Show all Regions, Cities |
| `/apply-filters` | POST | Filter data | Select North Region only |
| `/anomalies` | GET | Find unusual patterns | What's behaving oddly? |
| `/correlation` | GET | See relationships | Which metrics influence each other? |
| `/forecast` | POST | Predict future values | What will happen next week? |
| `/llama-analyze` | POST | Get AI insights | Why did this happen? |
| `/current-state` | GET | See what's loaded | What data am I working with? |
| `/reset-state` | POST | Start fresh | Clear all filters and data |

---

## 💡 Common Use Cases

### **Use Case 1: Anomaly Detection**
```
1. Upload CSV
2. Call /anomalies
3. System finds unusual values
4. Use /llama-analyze to understand WHY
```

### **Use Case 2: Forecasting**
```
1. Upload CSV
2. Filter to specific region
3. Call /forecast for target KPI
4. Get predictions for next 7-30 days
```

### **Use Case 3: Root Cause Analysis**
```
1. Upload CSV
2. Call /anomalies to find problems
3. Call /correlation to see relationships
4. Call /llama-analyze to understand causes
```

---

## ⚠️ Troubleshooting

### **Problem: API won't start**
```
Error: "Port 8000 already in use"
Solution: Change port in command: --port 8001
```

### **Problem: Upload fails**
```
Error: "File must be CSV format"
Solution: Make sure file ends with .csv
```

### **Problem: No data when calling /filters**
```
Error: "No DataFrame loaded"
Solution: Make sure you called /upload first
```

### **Problem: Large file takes too long**
```
Solution: System automatically samples large files (>50K rows)
This makes analysis faster without losing important patterns
```

---

## 🔐 Important Notes

1. **Data is stored in memory** - If API restarts, data is cleared
2. **Single user per session** - API designed for single developer (not production multi-user yet)
3. **Sample data included** - Test with `Sample_KPI_Data.csv` first
4. **No authentication yet** - All endpoints are open (add security before production deployment)

---

## 🔌 Connecting to Other Modules

This API automatically connects to:
- **Phase 2, Module 4**: Anomaly Detection
- **Phase 2, Module 5**: Correlation Analysis
- **Phase 3, Module 6**: Forecasting Engine
- **Phase 3, Module 7**: Llama AI Service

No manual configuration needed - it's all built-in!

---

## 📚 Further Reading

- **Detailed Architecture**: See `ARCHITECTURE_COMPATIBILITY_ANALYSIS.md`
- **Full API Documentation**: Visit `http://localhost:8000/docs` (when running)
- **Code Structure**: See `api.py` for implementation details

---

## ✅ Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-12-04 | Initial release - All endpoints working |

---

## 👤 Support

For issues or questions:
1. Check troubleshooting section above
2. Review API logs when running
3. Test with provided `Sample_KPI_Data.csv` first
4. Check GitHub issues: `nishadalab/Telecom-AI/issues`

---

## 📋 Quick Reference

**Start API**:
```bash
cd Phase4_Module8_API
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Test Upload**:
```bash
curl -F "file=@Sample_KPI_Data.csv" http://localhost:8000/upload
```

**View API Docs**:
```
Visit: http://localhost:8000/docs
```

**Test Filters**:
```bash
curl http://localhost:8000/filters/Region
```

---

**🎉 You're ready to start analyzing telecom data!**

For questions, check the detailed architecture document or code comments.
