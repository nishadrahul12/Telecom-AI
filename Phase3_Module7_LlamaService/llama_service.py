# Phase3_Module7_LlamaService/llama_service.py

import requests
import json
import logging
import time
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime
from functools import lru_cache

# Import schemas (from models.py)
from models import LLMResponse, FallbackAnalysis, ConfidenceLevel
from prompts import (
    SYSTEM_PROMPT_CAUSAL,
    SYSTEM_PROMPT_SCENARIO,
    SYSTEM_PROMPT_CORRELATION,
    FALLBACK_TEMPLATES
)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class OllamaConnectionError(Exception):
    """Raised when connection to Ollama fails"""
    pass


class OllamaTimeoutError(Exception):
    """Raised when LLM inference times out"""
    pass


# ============================================================================
# LLAMA SERVICE CLASS
# ============================================================================

class LlamaService:
    """
    Local Llama 70B integration service for telecom anomaly analysis.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 11434,
        model: str = "llama2:70b",
        timeout_seconds: int = 30,
        log_level: str = "INFO"
    ):
        self.host = host
        self.port = port
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.base_url = f"http://{host}:{port}"
        self.ollama_model = model
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    
    # ========================================================================
    # CONNECTION MANAGEMENT
    # ========================================================================
    
    def connect_to_ollama(self) -> bool:
        """Test connection to Ollama API."""
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                self.logger.info(
                    f"✓ Connected to Ollama at {self.base_url} "
                    f"(models: {len(data.get('models', []))})"
                )
                return True
            else:
                self.logger.warning(
                    f"Ollama returned status {response.status_code}"
                )
                return False
                
        except requests.exceptions.ConnectionError as e:
            self.logger.error(
                f"✗ Connection refused to {self.base_url}: {str(e)}"
            )
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? (ollama serve)"
            )
        except requests.exceptions.Timeout:
            self.logger.error(f"Connection timeout to {self.base_url}")
            raise OllamaConnectionError("Connection timeout to Ollama")
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            return False
    
    
    # ========================================================================
    # CORE LLM GENERATION METHODS
    # ========================================================================
    
    def generate_causal_analysis(
        self,
        anomaly_json: Dict[str, Any]
    ) -> LLMResponse:
        start_time = time.time()
        
        try:
            # Format prompt with data
            prompt = self._format_causal_prompt(anomaly_json)
            
            # Generate response from LLM
            analysis_text = self._call_ollama(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT_CAUSAL
            )
            
            # Parse response structure
            recommendations = self._extract_recommendations(analysis_text)
            confidence = self._assess_confidence(analysis_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Causal analysis completed in {processing_time:.1f}ms "
                f"(confidence: {confidence})"
            )
            
            return LLMResponse(
                request_type="Causal_Anomaly_Analysis",
                analysis=analysis_text,
                recommendations=recommendations,
                confidence_level=confidence,
                model_used="Llama-70B",
                processing_time_ms=processing_time,
                error=None
            )
            
        except OllamaConnectionError:
            self.logger.warning("Ollama unavailable, using fallback template")
            return self._get_fallback_causal(anomaly_json, start_time)
        except Exception as e:
            self.logger.error(f"Causal analysis error: {str(e)}")
            return LLMResponse(
                request_type="Causal_Anomaly_Analysis",
                analysis="Analysis failed due to internal error.",
                recommendations=["Check system logs for details."],
                confidence_level=ConfidenceLevel.LOW,
                model_used="Llama-70B",
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    
    def generate_scenario_planning(
        self,
        forecast_json: Dict[str, Any]
    ) -> LLMResponse:
        start_time = time.time()
        
        try:
            prompt = self._format_scenario_prompt(forecast_json)
            analysis_text = self._call_ollama(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT_SCENARIO
            )
            
            recommendations = self._extract_recommendations(analysis_text)
            confidence = self._assess_confidence(analysis_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.info(
                f"Scenario planning completed in {processing_time:.1f}ms"
            )
            
            return LLMResponse(
                request_type="Scenario_Planning_Forecast",
                analysis=analysis_text,
                recommendations=recommendations,
                confidence_level=confidence,
                model_used="Llama-70B",
                processing_time_ms=processing_time,
                error=None
            )
            
        except OllamaConnectionError:
            self.logger.warning("Ollama unavailable, using fallback")
            return self._get_fallback_scenario(forecast_json, start_time)
        except Exception as e:
            import traceback
            self.logger.error(f"Scenario planning error: {str(e)}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return LLMResponse(
                request_type="Scenario_Planning_Forecast",
                analysis="Analysis failed due to internal error.",
                recommendations=["Check system logs for details."],
                confidence_level=ConfidenceLevel.LOW,
                model_used="Llama-70B",
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    
    def generate_correlation_interpretation(
        self,
        correlation_json: Dict[str, Any]
    ) -> LLMResponse:
        start_time = time.time()
        
        try:
            prompt = self._format_correlation_prompt(correlation_json)
            analysis_text = self._call_ollama(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT_CORRELATION
            )
            
            recommendations = self._extract_recommendations(analysis_text)
            confidence = self._assess_confidence(analysis_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            return LLMResponse(
                request_type="Correlation_Interpretation",
                analysis=analysis_text,
                recommendations=recommendations,
                confidence_level=confidence,
                model_used="Llama-70B",
                processing_time_ms=processing_time,
                error=None
            )
            
        except OllamaConnectionError:
            return self._get_fallback_correlation(correlation_json, start_time)
        except Exception as e:
            self.logger.error(f"Correlation interpretation error: {str(e)}")
            return LLMResponse(
                request_type="Correlation_Interpretation",
                analysis="Analysis failed due to internal error.",
                recommendations=["Check system logs for details."],
                confidence_level=ConfidenceLevel.LOW,
                model_used="Llama-70B",
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    
    # ========================================================================
    # STREAMING SUPPORT
    # ========================================================================
    
    def stream_llama_response(
        self,
        prompt: str,
        system_prompt: str = ""
    ) -> Generator[str, None, None]:
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": True,
                "temperature": 0.7
            }
            
            response = requests.post(
                url,
                json=payload,
                stream=True,
                timeout=self.timeout_seconds
            )
            
            if response.status_code != 200:
                raise OllamaConnectionError(
                    f"Ollama returned status {response.status_code}"
                )
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'response' in data:
                        yield data['response']
                        
        except Exception as e:
            self.logger.error(f"Streaming error: {str(e)}")
            yield f"[Error: {str(e)}]"
    
    
    # ========================================================================
    # INTERNAL HELPER METHODS
    # ========================================================================
    
    def _call_ollama(
        self,
        prompt: str,
        system_prompt: str = ""
    ) -> str:
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout_seconds
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', '').strip()
            else:
                raise OllamaConnectionError(
                    f"Ollama returned status {response.status_code}"
                )
                
        except requests.exceptions.Timeout:
            raise OllamaTimeoutError(
                f"LLM inference timeout after {self.timeout_seconds}s"
            )
        except requests.exceptions.ConnectionError:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
            )
        
    def _safe_float(self, val: Any, default: float = 0.0) -> float:
        """Safely convert any value to float, returns default on failure"""
        try:
            if val is None:
                return default
            return float(val)
        except (ValueError, TypeError):
            return default
        
    def _safe_str(self, val: Any, default: str = "N/A") -> str:
        """Safely convert any value to string representation"""
        try:
            if val is None:
                return default
            # Explicitly convert to string and strip whitespace
            return str(val).strip()
        except Exception:
            return default
    
    
    @lru_cache(maxsize=128)
    def _get_system_prompt(self, prompt_type: str) -> str:
        prompts = {
            "causal": SYSTEM_PROMPT_CAUSAL,
            "scenario": SYSTEM_PROMPT_SCENARIO,
            "correlation": SYSTEM_PROMPT_CORRELATION
        }
        return prompts.get(prompt_type, "")
    
    
    def _format_causal_prompt(self, anomaly_json: Dict[str, Any]) -> str:
        target = anomaly_json.get('target_anomaly', {})
        contextual = anomaly_json.get('contextual_data', [])
        
        # Use safe access for values that might need formatting
        actual_val = target.get('actual_value')
        formatted_actual = f"{actual_val:,.0f}" if actual_val is not None else "N/A"
        
        zscore = target.get('zscore')
        formatted_zscore = f"{zscore:.2f}" if zscore is not None else "N/A"

        prompt = f"""
Analyze the following KPI anomaly:

**Target Anomaly:**
- KPI: {target.get('kpi_name')}
- Date: {target.get('date_time')}
- Actual Value: {formatted_actual}
- Expected Range: {target.get('expected_range')}
- Severity: {target.get('severity')}
- Z-Score: {formatted_zscore}σ

**Correlated KPIs on Anomaly Date:**
"""
        for kpi in contextual:
            val = kpi.get('value_on_anomaly_date')
            corr = kpi.get('correlation_score')
            prompt += f"""
- {kpi.get('kpi_name')}: {val}
  Correlation: {corr:.3f}
  State: {kpi.get('historical_state')}
"""
        
        prompt += "\nProvide root cause analysis and 2-3 specific investigation steps."
        return prompt
    
    
    def _format_scenario_prompt(self, forecast_json: Dict[str, Any]) -> str:
        """Format scenario planning prompt with comprehensive safe type conversion"""
        
        # NOTE: safe_float and safe_str helpers have been moved to private methods
        
        # Convert all numeric values safely using the new methods
        current_val = self._safe_float(forecast_json.get('current_value'), 0)
        predicted_val = self._safe_float(forecast_json.get('predicted_value'), 0)
        threshold_val = self._safe_float(forecast_json.get('critical_threshold'), 0)
        
        forecast_days_raw = forecast_json.get('forecast_horizon_days')
        # FIX: Ensure 'days' is explicitly added to the string representation
        forecast_days_str = self._safe_str(forecast_days_raw, 'Unknown') 
        
        # Build prompt with safely converted values
        prompt = f"""
    Analyze the following forecast scenario:

    **Forecast Details:**
    - Target KPI: {self._safe_str(forecast_json.get('forecast_target'), 'Unknown')}
    - Forecast Horizon: {forecast_days_str} days 
    - Current Value: {current_val:.2f}
    - Predicted Value: {predicted_val:.2f}
    - Critical Threshold: {threshold_val:.2f}

    **Driving Variables:**
    """
        
        # Process model parameters with comprehensive error handling
        model_params = forecast_json.get('model_parameters', [])
        
        if not model_params:
            prompt += "\n- No driving variables specified\n"
        else:
            for param in model_params:
                # Safely extract all parameter values
                var_name = self._safe_str(param.get('variable_name'), 'Unknown')
                description = self._safe_str(param.get('influence_description'), 'No description')
                
                # Handle 'change' value - convert to safe string representation
                change_val = param.get('projected_change', 0)
                if change_val is None:
                    change_str = "N/A"
                elif isinstance(change_val, str):
                    # If already a string, check if it's numeric
                    try:
                        float(change_val)
                        change_str = f"{change_val}%"
                    except ValueError:
                        # Non-numeric string, keep as-is
                        change_str = change_val
                elif isinstance(change_val, (int, float)):
                    # Numeric value, format with %
                    change_str = f"{change_val}%"
                else:
                    # Unknown type, convert to string
                    change_str = str(change_val)
                
                # Safely convert influence score to float for formatting
                influence_val = self._safe_float(param.get('influence_score'), 0)
                
                # Build parameter line
                prompt += f"""
    - {var_name}: {change_str}
    Influence Score: {influence_val:.2f}
    Impact: {description}
    """
        
        prompt += "\nProvide impact assessment and 2-3 mitigation strategies."
        return prompt

    
    def _format_correlation_prompt(self, corr_json: Dict[str, Any]) -> str:
        score = corr_json.get('correlation_score', 0)
        
        return f"""
Interpret the following KPI correlation:

- Source KPI: {corr_json.get('source_kpi')}
- Target KPI: {corr_json.get('target_kpi')}
- Correlation Coefficient: {score:.3f}
- Method: {corr_json.get('correlation_method')}

Explain the operational meaning and what network behavior this reflects.
Provide 1-2 actionable insights.
"""
    
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract numbered recommendations from response text"""
        recommendations = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered items (1., 2., 3., etc.)
            parts = line.split('.', 1)
            if len(parts) > 1 and parts[0].strip().isdigit():
                recommendation = parts[1].strip()
                if recommendation:
                    recommendations.append(recommendation)
        
        # If no numbered items found, extract first 2-3 sentences
        if not recommendations:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            recommendations = sentences[:3]
        
        return recommendations[:5]  # Max 5 recommendations
    
    
    def _assess_confidence(self, text: str) -> ConfidenceLevel:
        text_lower = text.lower()
        
        if any(phrase in text_lower for phrase in [
            "clearly", "definitely", "strong", "significant", "conclusive"
        ]):
            return ConfidenceLevel.HIGH
        
        if any(phrase in text_lower for phrase in [
            "unclear", "possible", "might", "uncertain", "cannot determine"
        ]):
            return ConfidenceLevel.LOW
        
        return ConfidenceLevel.MEDIUM
    
    
    def _get_fallback_causal(
        self,
        anomaly_json: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        kpi_name = anomaly_json.get('target_anomaly', {}).get('kpi_name', 'KPI')
        
        return LLMResponse(
            request_type="Causal_Anomaly_Analysis",
            analysis=FALLBACK_TEMPLATES['causal'].format(kpi=kpi_name),
            recommendations=[
                "Verify load balancing parameters across all carriers",
                "Check handover success rates and optimize if needed",
                "Measure signal strength and investigate interference"
            ],
            confidence_level=ConfidenceLevel.MEDIUM,
            model_used="Fallback-Text-Template",
            processing_time_ms=(time.time() - start_time) * 1000,
            error="Ollama unavailable - using template"
        )
    
    
    def _get_fallback_scenario(
        self,
        forecast_json: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        target = forecast_json.get('forecast_target', 'KPI')
        
        return LLMResponse(
            request_type="Scenario_Planning_Forecast",
            analysis=FALLBACK_TEMPLATES['scenario'].format(kpi=target),
            recommendations=[
                "Monitor the predicted decline closely over the forecast period",
                "Prepare mitigation actions if threshold is approached",
                "Review model parameters and driving variables"
            ],
            confidence_level=ConfidenceLevel.MEDIUM,
            model_used="Fallback-Text-Template",
            processing_time_ms=(time.time() - start_time) * 1000,
            error="Ollama unavailable - using template"
        )
    
    
    def _get_fallback_correlation(
        self,
        correlation_json: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        score = correlation_json.get('correlation_score', 0)
        source = correlation_json.get('source_kpi', 'Source')
        target = correlation_json.get('target_kpi', 'Target')
        
        return LLMResponse(
            request_type="Correlation_Interpretation",
            analysis=FALLBACK_TEMPLATES['correlation'].format(
                source=source,
                target=target,
                score=score
            ),
            recommendations=[
                "Investigate the causal relationship between these KPIs",
                "Monitor both metrics for co-movement patterns",
                "Apply correlation insights to network optimization"
            ],
            confidence_level=ConfidenceLevel.MEDIUM,
            model_used="Fallback-Text-Template",
            processing_time_ms=(time.time() - start_time) * 1000,
            error="Ollama unavailable - using template"
        )