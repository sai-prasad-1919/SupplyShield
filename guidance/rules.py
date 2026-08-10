"""
SupplyShield — Rule-Based Guidance Engine (top-level, used by LLM agent)

Returns structured recommendations: list of {title, description} dicts.
Aligned to the real 9-feature set (no traffic/weather legacy names).
"""

def generate_rule_based_guidance(explanation: dict) -> list[dict]:
    """
    Generate actionable recommendations based on the prediction and SHAP explanation.
    Returns list of {title, description} dicts — same format as the LLM output.
    """
    prediction    = explanation.get("prediction", {})
    explainability = explanation.get("explainability", {})
    top_drivers   = explainability.get("top_drivers", []) if explainability else []

    is_delayed = prediction.get("is_delayed", False)
    prob       = prediction.get("delay_probability", 0.0)

    if not is_delayed:
        return [{"title": "Route Clear", "description": "Route appears clear. Proceed with standard logistics plan."}]

    recommendations: list[dict] = []

    # Generate recommendations from top SHAP drivers
    for driver in top_drivers:
        feature = driver.get("feature", "")
        impact  = driver.get("impact", 0)

        if impact <= 0:
            continue  # only act on delay-increasing features

        if feature == "weather_condition":
            recommendations.append({
                "title": "Weather Contingency",
                "description": "Severe weather conditions detected. Consider rerouting or switching to a weather-resistant transport mode. Pre-notify consignee of potential delay.",
            })
        elif feature == "vehicle_type":
            recommendations.append({
                "title": "Upgrade Vehicle",
                "description": "Vehicle type correlates with delays on this route. Consider upgrading to a faster or more reliable vehicle type for this distance.",
            })
        elif feature == "distance_km":
            recommendations.append({
                "title": "Route Optimisation",
                "description": "Long distance is a key delay driver. Consider cross-docking at an intermediate hub or expedited staging to reduce end-to-end transit time.",
            })
        elif feature == "delivery_mode":
            recommendations.append({
                "title": "Upgrade Delivery Mode",
                "description": "Current delivery mode is flagged as high-risk for this shipment profile. Upgrading to express or priority shipping may mitigate delay.",
            })
        elif feature == "region":
            recommendations.append({
                "title": "Regional Alert",
                "description": "Destination region is experiencing logistical delays. Check local congestion alerts and validate route with the carrier.",
            })
        elif feature == "package_weight_kg":
            recommendations.append({
                "title": "Weight Review",
                "description": "Package weight is a contributing delay factor. Verify vehicle load capacity and consider splitting the shipment if feasible.",
            })

        if len(recommendations) >= 3:
            break

    # Fallback if no specific driver-based recommendation was generated
    if not recommendations:
        recommendations.append({
            "title": "Standard Monitoring",
            "description": f"High delay risk detected ({prob * 100:.1f}%). Apply standard delay protocols and escalate to the delivery partner if the shipment does not move within 4 hours.",
        })

    return recommendations[:3]
