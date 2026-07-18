"""
SupplyShield — LLM Guidance Agent

Integrates with an LLM (Gemini via google-genai) to provide rich, narrative
mitigation strategies when high-risk delays are detected.

Falls back to the rule-based engine if the API key is missing or an error occurs.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import USE_LLM, GEMINI_API_KEY
from utils import logger
from guidance.rules import generate_rule_based_guidance

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


class LLMGuide:
    def __init__(self):
        self.use_llm = USE_LLM and bool(GEMINI_API_KEY) and genai is not None
        
        if self.use_llm:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info("Initialized LLM Guide with Gemini")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.use_llm = False
        else:
            logger.info("LLM Guide disabled or API key missing. Using rule-based fallback.")

    def generate_guidance(self, explanation: dict) -> list[str]:
        """
        Generate guidance using LLM, or fallback to rule-based engine.
        """
        # Always use rules for non-delayed predictions to save API calls
        if not explanation.get("prediction", {}).get("is_delayed", False):
            return generate_rule_based_guidance(explanation)
            
        if not self.use_llm:
            return generate_rule_based_guidance(explanation)
            
        try:
            return self._call_llm(explanation)
        except Exception as e:
            logger.error(f"LLM API Error: {e}. Falling back to rule-based guidance.")
            return generate_rule_based_guidance(explanation)

    def _call_llm(self, explanation: dict) -> list[str]:
        """Format the explanation and call Gemini to get a mitigation plan."""
        prompt = f"""
        You are SupplyShield, an AI logistics expert.
        A shipment has a high risk of delay ({explanation['prediction'].get('delay_probability', 0)*100:.1f}%).
        
        The SHAP explainability model identified the following top drivers for the delay:
        {json.dumps(explanation['explainability'].get('top_drivers', []), indent=2)}
        
        Please provide 3 specific, actionable mitigation strategies for the supply chain manager.
        Format your response as a JSON array of 3 strings. Example: ["Mitigation 1", "Mitigation 2", "Mitigation 3"]
        """
        
        response = self.client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            )
        )
        
        # Parse the JSON array
        try:
            recommendations = json.loads(response.text)
            if isinstance(recommendations, list):
                return recommendations
            else:
                return [str(recommendations)]
        except json.JSONDecodeError:
            # If parsing fails, just split by newlines as a fallback
            return [line.strip("- ") for line in response.text.split("\n") if line.strip()]

# Lazy singleton instance for the API
_guide = None

def get_guidance(explanation: dict) -> list[str]:
    """Public interface to get guidance."""
    global _guide
    if _guide is None:
        _guide = LLMGuide()
    return _guide.generate_guidance(explanation)
