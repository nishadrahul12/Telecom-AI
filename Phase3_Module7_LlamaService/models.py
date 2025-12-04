# Phase3_Module7_LlamaService/models.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum


class RequestType(str, Enum):
    """Enumeration of valid LLM request types"""
    CAUSAL_ANOMALY_ANALYSIS = "Causal_Anomaly_Analysis"
    SCENARIO_PLANNING = "Scenario_Planning_Forecast"
    CORRELATION_INTERPRETATION = "Correlation_Interpretation"


class ConfidenceLevel(str, Enum):
    """LLM confidence level assessment"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class AnomalyDetails(BaseModel):
    """Schema for target anomaly in causal analysis"""
    kpi_name: str = Field(..., description="KPI name (e.g., RACH_Stp_att)")
    date_time: str = Field(..., description="Anomaly date/time (YYYY-MM-DD)")
    actual_value: float = Field(..., description="Observed value")
    expected_range: str = Field(..., description="Expected range (e.g., '45000 - 55000')")
    severity: str = Field(..., description="Severity level (Low/Medium/High)")
    zscore: float = Field(..., description="Z-score deviation (>3 = significant)")

    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = {"Low", "Medium", "High"}
        if v not in valid_severities:
            raise ValueError(f"Severity must be one of {valid_severities}")
        return v


class ContextualKPI(BaseModel):
    """Schema for contextual correlated KPI in anomaly analysis"""
    kpi_name: str = Field(..., description="Correlated KPI name")
    value_on_anomaly_date: float = Field(..., description="Value on anomaly date")
    correlation_score: float = Field(..., ge=-1.0, le=1.0, description="Pearson correlation (-1 to 1)")
    historical_state: str = Field(..., description="State assessment (e.g., 'Too High', 'Too Low')")


class CausalAnalysisRequest(BaseModel):
    """Input schema for causal anomaly analysis"""
    request_type: RequestType = Field(RequestType.CAUSAL_ANOMALY_ANALYSIS, description="Request type")
    target_anomaly: AnomalyDetails = Field(..., description="Anomaly being analyzed")
    contextual_data: List[ContextualKPI] = Field(..., min_items=1, max_items=10, description="Correlated KPIs")

    class Config:
        use_enum_values = True


class ModelParameter(BaseModel):
    """Schema for model parameter in scenario planning"""
    variable_name: str = Field(..., description="Variable name (e.g., Traffic_Volume_DL)")
    projected_change: float = Field(..., description="Projected change as numeric percentage (e.g., 15 for 15%)")
    influence_score: float = Field(..., ge=0.0, le=1.0, description="Influence on forecast (0-1)")
    influence_description: str = Field(..., description="How this variable influences the forecast")


class ScenarioPlanningRequest(BaseModel):
    """Input schema for scenario planning forecast analysis"""
    request_type: RequestType = Field(RequestType.SCENARIO_PLANNING, description="Request type")
    forecast_target: str = Field(..., description="Target KPI being forecast")
    forecast_horizon_days: int = Field(..., ge=1, le=30, description="Forecast horizon (days)")
    current_value: float = Field(..., description="Current KPI value")
    predicted_value: float = Field(..., description="Predicted KPI value at horizon")
    critical_threshold: float = Field(..., description="Critical threshold (alert if below)")
    model_parameters: List[ModelParameter] = Field(..., min_items=1, max_items=5, description="Driving variables")

    class Config:
        use_enum_values = True


class CorrelationRequest(BaseModel):
    """Input schema for correlation interpretation"""
    request_type: RequestType = Field(RequestType.CORRELATION_INTERPRETATION, description="Request type")
    source_kpi: str = Field(..., description="Source KPI name")
    target_kpi: str = Field(..., description="Target KPI name")
    correlation_score: float = Field(..., ge=-1.0, le=1.0, description="Pearson correlation coefficient")
    correlation_method: str = Field(default="Pearson", description="Correlation method used")

    class Config:
        use_enum_values = True


# ============================================================================
# RESPONSE SCHEMA
# ============================================================================

class LLMResponse(BaseModel):
    """Output schema for all LLM operations"""
    request_type: str = Field(..., description="Echo of request type")
    analysis: str = Field(..., description="Main reasoning and explanation (100-300 words)")
    recommendations: List[str] = Field(..., min_items=1, max_items=5, description="Actionable steps (1-5 items)")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence in analysis (Low/Medium/High)")
    model_used: str = Field(default="Llama-70B", description="Model name")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    error: Optional[str] = Field(default=None, description="Error message if operation failed")

    class Config:
        use_enum_values = True


# ============================================================================
# FALLBACK RESPONSE SCHEMA
# ============================================================================

class FallbackAnalysis(BaseModel):
    """Fallback response when Ollama is unavailable"""
    request_type: str
    analysis: str
    recommendations: List[str]
    confidence_level: str = "Medium"
    model_used: str = "Fallback-Text-Template"
    processing_time_ms: float = 0.0
    error: Optional[str] = None

    def to_llm_response(self) -> LLMResponse:
        """Convert to LLMResponse format"""
        return LLMResponse(
            request_type=self.request_type,
            analysis=self.analysis,
            recommendations=self.recommendations,
            confidence_level=self.confidence_level,
            model_used=self.model_used,
            processing_time_ms=self.processing_time_ms,
            error=self.error
        )
