RULES = [
    {
        "id": "critical_probability",
        "condition": lambda s: s.get("delay_probability", 0) > 0.95,
        "message": "Critical delay risk detected. Open an emergency logistics ticket and contact the carrier directly.",
        "reason": "rule: delay_probability > 95%",
    },
    {
        "id": "weather_storm_long_distance",
        "condition": lambda s: s.get("weather_condition") in ["stormy", "foggy"] and s.get("distance_km", 0) > 500,
        "message": "Stormy/foggy weather combined with long-distance route increases delay risk significantly. Pre-notify consignee and consider route diversion.",
        "reason": "rule: stormy/foggy + distance > 500km",
    },
    {
        "id": "high_time_diff",
        "condition": lambda s: s.get("time_diff_hours", 0) > 24,
        "message": f"Shipment is already running heavily over expected time. Escalate to logistics partner immediately.",
        "reason": "rule: actual_time - expected_time > 24h",
    },
    {
        "id": "med_time_diff",
        "condition": lambda s: s.get("time_diff_hours", 0) > 8,
        "message": "Shipment is running behind schedule. Send delay alert to consignee to manage expectations.",
        "reason": "rule: actual_time - expected_time > 8h",
    },
    {
        "id": "fragile_weather",
        "condition": lambda s: s.get("package_type") in ["fragile items", "electronics"] and s.get("weather_condition") in ["stormy", "rainy"],
        "message": "Sensitive cargo exposed to adverse weather. Humidity-controlled or secure storage required at transit hub.",
        "reason": "rule: fragile/electronics + rainy/stormy",
    },
    {
        "id": "vehicle_mismatch",
        "condition": lambda s: s.get("vehicle_type") in ["bike", "scooter"] and s.get("distance_km", 0) > 200,
        "message": "Vehicle type mismatch for distance. Reassign to truck or van to prevent breakdowns or delays.",
        "reason": "rule: bike/scooter + distance > 200km",
    },
    {
        "id": "express_sla_breach",
        "condition": lambda s: s.get("delivery_mode") == "express" and s.get("time_diff_hours", 0) > 2,
        "message": "Express SLA breach likely. Initiate SLA claim and expedite final mile delivery.",
        "reason": "rule: express + time_diff > 2h",
    },
    {
        "id": "region_flood",
        "condition": lambda s: s.get("region") == "central" and s.get("weather_condition") == "rainy",
        "message": "Regional flooding risk in Central zone. Validate route with driver and check for road closures.",
        "reason": "rule: region=central + weather=rainy",
    },
    {
        "id": "overweight_vehicle",
        "condition": lambda s: s.get("package_weight_kg", 0) > 200 and s.get("vehicle_type") in ["van", "scooter", "bike"],
        "message": "Overweight for vehicle class. Reassign to truck immediately to avoid compliance/safety delays.",
        "reason": "rule: weight > 200kg + van/scooter/bike",
    },
    {
        "id": "standard_delay",
        "condition": lambda s: s.get("risk_level") in ["High", "Medium"],
        "message": "Standard delay risk detected. Monitor closely over the next 48 hours.",
        "reason": "rule: standard model prediction",
    }
]

def evaluate_rules(shipment: dict):
    """
    Evaluates the ordered rules against shipment features.
    Returns the first matching rule, or a fallback.
    """
    for rule in RULES:
        try:
            if rule["condition"](shipment):
                return {
                    "message": rule["message"],
                    "reason": rule["reason"],
                    "rule_id": rule["id"]
                }
        except Exception:
            continue
            
    return {
        "message": "Shipment on track. No anomaly patterns detected. Standard monitoring applies.",
        "reason": "fallback: low risk, standard ops",
        "rule_id": "fallback"
    }
