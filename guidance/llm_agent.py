"""
SupplyShield — LLM Guidance Agent

Integrates with Gemini to provide rich, structured mitigation strategies when
high-risk delays are detected. Falls back to the rule-based engine gracefully.

Both Gemini and rule-based paths return the same format:
  list of {"title": str, "description": str}
  + guidance_source: "gemini" | "rule_based"
"""

import sys
import json
from pathlib import Path

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
            logger.info("LLM Guide disabled or API key missing — using rule-based fallback.")

    def generate_guidance(self, explanation: dict) -> dict:
        """
        Generate guidance using LLM, or fallback to rule-based engine.

        Returns:
            {
                "recommendations": [{"title": str, "description": str}, ...],
                "guidance_source": "gemini" | "rule_based"
            }
        """
        if not explanation.get("prediction", {}).get("is_delayed", False):
            recs = generate_rule_based_guidance(explanation)
            return {"recommendations": recs, "guidance_source": "rule_based"}

        if not self.use_llm:
            recs = generate_rule_based_guidance(explanation)
            return {"recommendations": recs, "guidance_source": "rule_based"}

        try:
            recs = self._call_llm(explanation)
            return {"recommendations": recs, "guidance_source": "gemini"}
        except Exception as e:
            logger.error(f"LLM API Error: {e}. Falling back to rule-based guidance.")
            recs = generate_rule_based_guidance(explanation)
            return {"recommendations": recs, "guidance_source": "rule_based"}

    def _call_llm(self, explanation: dict) -> list[dict]:
        """Format the explanation and call Gemini to get structured mitigation strategies."""
        prediction    = explanation.get("prediction", {})
        explainability = explanation.get("explainability", {})
        context       = explanation.get("context", {})
        business_rules = explanation.get("business_rules", [])

        context_str = " | ".join([f"{k}: {v}" for k, v in context.items()]) if context else "Not provided"
        rules_str   = "\n".join([f"- {r}" for r in business_rules]) if business_rules else "None triggered"

        drivers_str = ""
        for i, d in enumerate(explainability.get("top_drivers", []), 1):
            drivers_str += (
                f"  {i}. {d['feature']} = {d['value']}  "
                f"→ impact: {d['impact']:+.2f} ({d['direction']})\n"
            )

        prompt = f"""
You are SupplyShield, an AI logistics risk analyst.

SHIPMENT CONTEXT:
  {context_str}

PREDICTION:
  Delay Probability: {prediction.get('delay_probability', 0) * 100:.1f}%  →  HIGH RISK

TOP SHAP DRIVERS (why the model flagged this shipment):
{drivers_str}
BUSINESS RULES TRIGGERED:
{rules_str}

Generate exactly 3 specific, actionable mitigation strategies for the supply chain manager.
DO NOT invent model results. The figures above come from the ML model output directly.

Return ONLY a JSON array with exactly 3 objects, each with "title" (short, ≤5 words) and "description" (1–2 sentences).
Example format:
[
  {{"title": "Reroute Shipment", "description": "Reroute via Mumbai hub to avoid North region congestion."}},
  {{"title": "Upgrade Transport", "description": "Switch from Truck to Air freight for this 847km route."}},
  {{"title": "Alert Supplier", "description": "Notify supplier 48 hours early given adverse weather conditions."}}
]
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        raw = json.loads(response.text)
        if isinstance(raw, list):
            # Validate structure
            result = []
            for item in raw:
                if isinstance(item, dict) and "title" in item and "description" in item:
                    result.append({"title": str(item["title"]), "description": str(item["description"])})
            return result[:3] if result else generate_rule_based_guidance(explanation)

        return generate_rule_based_guidance(explanation)


# Lazy singleton
_guide = None

def get_guidance(explanation: dict) -> dict:
    """Public interface. Returns {"recommendations": [...], "guidance_source": "gemini"|"rule_based"}"""
    global _guide
    if _guide is None:
        _guide = LLMGuide()
    return _guide.generate_guidance(explanation)
