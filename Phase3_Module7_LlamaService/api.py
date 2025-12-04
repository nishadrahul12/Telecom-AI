# FastAPI integration with Phase 3 Module 7 (Llama Service)
# This API integrates LLM-powered analysis for telecom anomaly detection

import json
import logging
import traceback
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from llama_service import LlamaService
from models import (
    CausalAnalysisRequest,
    ScenarioPlanningRequest,
    CorrelationRequest,
    LLMResponse
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Telecom-AI Phase 3 - LLM Service",
    description="LLM-powered analysis for telecom network anomalies",
    version="1.0.0"
)

# Global variable for LLM service (will be initialized on startup)
llama_service = None

# ============================================================================
# STARTUP / SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize LLM service on application startup"""
    global llama_service
    try:
        logger.info("Starting LLM Service initialization...")
        llama_service = LlamaService(
            host="localhost",
            port=11434,
            model="llama2:70b",
            timeout_seconds=30
        )
        
        # Test connection to Ollama
        try:
            is_connected = llama_service.connect_to_ollama()
            if is_connected:
                logger.info("✓ LLM Service initialized and connected to Ollama")
                print("✓ LLM Service initialized and connected to Ollama")
            else:
                logger.warning("⚠ Ollama unavailable - will use fallback templates")
                print("⚠ LLM Service initialized (Ollama unavailable - will use fallback templates)")
        except Exception as e:
            logger.warning(f"⚠ LLM Service startup warning: {str(e)}")
            print(f"⚠ LLM Service startup warning: {str(e)}")
    except Exception as e:
        logger.error(f"STARTUP ERROR: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global llama_service
    llama_service = None
    logger.info("✓ LLM Service shutdown complete")
    print("✓ LLM Service shutdown complete")

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/api/v1/health")
async def health_check():
    """Simple API status check."""
    return {
        "status": "API Operational",
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/v1/health/llm")
async def llm_health_check():
    """Detailed LLM service connection and model check."""
    global llama_service
    
    if llama_service is None:
        return {
            "status": "Critical",
            "message": "LLM Service not initialized. Check server logs.",
            "ollama_status": "Disconnected"
        }

    try:
        # Check connection again, returns True/False
        is_connected = await run_in_threadpool(llama_service.connect_to_ollama)
        
        return {
            "status": "Operational" if is_connected else "Degraded",
            "message": "Ollama connection successful." if is_connected else "Ollama connection failed or model is missing.",
            "ollama_status": "Connected" if is_connected else "Disconnected",
            "model": llama_service.ollama_model
        }
    except Exception as e:
        # Use the logger defined at the top of api.py
        logger.error(f"LLM Health Check failed: {e}")
        return {
            "status": "Critical",
            "message": f"Exception during Ollama connection check: {str(e)}",
            "ollama_status": "Disconnected"
        }

# ============================================================================
# CAUSAL ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/v1/llm/causal-analysis", response_model=LLMResponse)
async def causal_analysis(request: CausalAnalysisRequest):
    """
    Generate causal analysis for detected anomaly
    """
    try:
        logger.info(f"Received causal analysis request: {request}")
        # Convert Pydantic model to pure dict
        request_dict = json.loads(request.json())
        logger.debug(f"Request dict: {request_dict}")
        
        response = llama_service.generate_causal_analysis(request_dict)
        logger.info(f"Causal analysis completed successfully")
        return response
    except Exception as e:
        logger.error(f"❌ CAUSAL ANALYSIS ERROR: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        try:
            return LLMResponse(
                request_type="Causal_Anomaly_Analysis",
                analysis="Analysis failed due to internal error.",
                recommendations=["Check system logs for details."],
                confidence_level="Low",
                model_used="Llama-70B",
                processing_time_ms=0,
                error=str(e)
            )
        except Exception as response_error:
            logger.error(f"❌ RESPONSE CREATION ERROR: {str(response_error)}")
            logger.error(traceback.format_exc())
            raise

# ============================================================================
# SCENARIO PLANNING ENDPOINTS
# ============================================================================

@app.post("/api/v1/llm/scenario-planning", response_model=LLMResponse)
async def scenario_planning(request: ScenarioPlanningRequest):
    """
    Generate scenario planning analysis for forecasted changes
    """
    try:
        logger.info(f"Received scenario planning request: {request}")
        # Convert Pydantic model to pure dict
        request_dict = json.loads(request.json())
        logger.debug(f"Request dict: {request_dict}")
        
        response = llama_service.generate_scenario_planning(request_dict)
        logger.info(f"Scenario planning completed successfully")
        return response
    except Exception as e:
        logger.error(f"❌ SCENARIO PLANNING ERROR: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        try:
            return LLMResponse(
                request_type="Scenario_Planning_Forecast",
                analysis="Analysis failed due to internal error.",
                recommendations=["Check system logs for details."],
                confidence_level="Low",
                model_used="Llama-70B",
                processing_time_ms=0,
                error=str(e)
            )
        except Exception as response_error:
            logger.error(f"❌ RESPONSE CREATION ERROR: {str(response_error)}")
            logger.error(traceback.format_exc())
            raise

# ============================================================================
# CORRELATION INTERPRETATION ENDPOINTS
# ============================================================================

@app.post("/api/v1/llm/correlation", response_model=LLMResponse)
async def correlation_interpretation(request: CorrelationRequest):
    """
    Generate interpretation of KPI correlation
    """
    try:
        logger.info(f"Received correlation request: {request}")
        # Convert Pydantic model to pure dict
        request_dict = json.loads(request.json())
        logger.debug(f"Request dict: {request_dict}")
        
        response = llama_service.generate_correlation_interpretation(request_dict)
        logger.info(f"Correlation interpretation completed successfully")
        return response
    except Exception as e:
        logger.error(f"❌ CORRELATION ERROR: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        try:
            return LLMResponse(
                request_type="Correlation_Interpretation",
                analysis="Analysis failed due to internal error.",
                recommendations=["Check system logs for details."],
                confidence_level="Low",
                model_used="Llama-70B",
                processing_time_ms=0,
                error=str(e)
            )
        except Exception as response_error:
            logger.error(f"❌ RESPONSE CREATION ERROR: {str(response_error)}")
            logger.error(traceback.format_exc())
            raise

# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """API root endpoint with HTML UI"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Telecom-AI Phase 3 - LLM Service</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1000px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            .container {
                background: white;
                border-radius: 10px;
                padding: 40px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                margin: 0 0 10px 0;
            }
            .subtitle {
                color: #666;
                margin: 0 0 30px 0;
            }
            .section {
                margin: 30px 0;
            }
            .section h2 {
                color: #764ba2;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }
            .endpoint {
                background: #f5f5f5;
                padding: 15px;
                margin: 10px 0;
                border-radius: 5px;
                border-left: 4px solid #667eea;
                font-family: monospace;
            }
            .method {
                color: #667eea;
                font-weight: bold;
                margin-right: 10px;
            }
            .status {
                background: #d4edda;
                color: #155724;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .links {
                text-align: center;
                margin-top: 30px;
            }
            a {
                display: inline-block;
                margin: 0 10px;
                padding: 10px 20px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: all 0.3s;
            }
            a:hover {
                background: #764ba2;
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Telecom-AI Phase 3 - LLM Service API</h1>
            <p class="subtitle">LLM-powered analysis for telecom network anomalies</p>
            
            <div class="status">
                ✅ API Status: <strong>Operational</strong><br>
                Version: <strong>1.0.0</strong>
            </div>
            
            <div class="section">
                <h2>📊 Available Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method">GET</span> /api/v1/health
                    <br><small>Simple API health check</small>
                </div>
                
                <div class="endpoint">
                    <span class="method">GET</span> /api/v1/health/llm
                    <br><small>Detailed LLM service and Ollama connection status</small>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> /api/v1/llm/causal-analysis
                    <br><small>Generate root cause analysis for anomalies</small>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> /api/v1/llm/scenario-planning
                    <br><small>Analyze forecast scenarios and impacts</small>
                </div>
                
                <div class="endpoint">
                    <span class="method">POST</span> /api/v1/llm/correlation
                    <br><small>Interpret KPI correlations</small>
                </div>
            </div>
            
            <div class="section">
                <h2>🔗 Quick Links</h2>
                <div class="links">
                    <a href="/docs">📚 Interactive API Docs (Swagger UI)</a>
                    <a href="/api/v1/health">✅ Health Check</a>
                    <a href="/api/v1/health/llm">🔌 LLM Status</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ GLOBAL EXCEPTION HANDLER TRIGGERED")
    logger.error(f"Exception type: {type(exc).__name__}")
    logger.error(f"Exception message: {str(exc)}")
    logger.error(f"Full traceback: {traceback.format_exc()}")
    
    return {
        "error": "Internal server error",
        "message": str(exc),
        "type": type(exc).__name__
    }

# ============================================================================
# STARTUP INSTRUCTIONS
# ============================================================================

r"""
STARTUP INSTRUCTIONS:

1. Make sure you're in the ROOT project folder:
   cd C:\Users\Rahul\Desktop\Projects\Telecom-AI

2. Start Ollama (in separate terminal):
   ollama serve

3. Pull the model (if not already pulled):
   ollama pull llama2:70b

4. Start FastAPI server (this terminal):
   uvicorn api:app --host 127.0.0.1 --port 8000 --reload

5. Access API documentation:
   http://127.0.0.1:8000/docs

6. Test health endpoint:
   curl http://127.0.0.1:8000/api/v1/health/llm

Expected output:
   ✓ LLM Service initialized and connected to Ollama
   Uvicorn running on http://127.0.0.1:8000

7. When running tests, WATCH THE CONSOLE for error messages!

KEY: This version logs ALL errors to console so you can see what's failing!
"""
