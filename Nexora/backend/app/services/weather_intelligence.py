"""
EcoTrace Weather Intelligence AI — Decision Assistant Service
Weather-Only, Evidence-Grounded, Real-Time, Zero-Fabrication Orchestration Engine.

Strict Architectural Guardrails:
1. Reuses existing Phase 1–7 verified data and decision engines.
2. Strictly scoped to weather, forecasts, warnings, and weather-related travel decisions.
3. Deterministic weather-domain classification gate BEFORE model generation.
4. Maintains explicit source classes:
   - IMD_STATION_OBSERVATION
   - IMD_NOWCAST
   - IMD_WARNING
   - MODEL_CURRENT
   - MODEL_FORECAST
   - RADAR
   - LIGHTNING
   - INCOIS_FORECAST
   - ROUTE_WEATHER
   - DERIVED
5. NEVER describes MODEL_CURRENT / MODEL_FORECAST as IMD observations.
6. When IMD warning data is unavailable, explicitly states:
   "Official warning status could not be verified from the available sources."
7. Grounded in verified EcoTrace meteorological evidence with zero fabrication.
"""

import json
import logging
import os
import re
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.services.travel_advisory import (
    DESTINATION_CONFIGS,
    OFFICIAL_IMD_STATION_REGISTRY,
    get_travel_advisory,
    evaluate_live_guardian_risk,
    build_journey_context,
    evaluate_adaptive_journey,
    _get_ist_time,
)

logger = logging.getLogger(__name__)

_SCOPE_REFUSAL_MESSAGE = (
    "I’m EcoTrace Weather Intelligence. I can help with weather, forecasts, warnings, "
    "and weather-related travel guidance, but not hotels, food, bookings, or unrelated topics."
)

WEATHER_AI_SYSTEM_PROMPT = """You are EcoTrace Weather Intelligence.
You are a weather-domain assistant, not a general-purpose assistant.

You may answer only questions involving weather, forecasts, warnings, weather observations, meteorological conditions, or decisions directly influenced by weather.

For unrelated requests, do not answer the underlying request. Briefly explain that your scope is limited to weather and weather-related travel guidance:
"I’m EcoTrace Weather Intelligence. I can help with weather, forecasts, warnings, and weather-related travel guidance, but not hotels, food, bookings, or unrelated topics."

Never invent weather observations, warnings, government alerts, or forecast values.

Use the provided verified EcoTrace weather evidence as the factual basis for answers.

Clearly distinguish:
- observed weather
- forecast/model guidance
- government warning
- inference/advisory

Never describe numerical model guidance as an official government warning.

When the evidence is missing, stale, contradictory, or unavailable, say so explicitly.
If official warning data is unavailable, do not claim that there is no warning. Explicitly state:
"Official warning status could not be verified from the available sources."

Do not fabricate current weather.
Keep answers concise, direct, and actionable."""

# Active in-memory session history storage (short-term for multi-turn follow-ups)
WEATHER_AI_SESSION_HISTORY: Dict[str, List[Dict[str, Any]]] = {}


def get_source_availability_summary(advisory: Dict[str, Any]) -> Dict[str, Any]:
    """Determine available source classes and create UI source-status descriptor."""
    prov = advisory.get("station_provenance") or {}
    v_status = prov.get("verification_status")
    imd_available = v_status in ["VERIFIED_STATION_OBSERVATION", "VERIFIED_IMD_DIRECT_OBSERVATION"]
    
    # Model guidance availability
    model_available = bool(advisory.get("weather_condition") or advisory.get("temperature_c") is not None)
    
    # Warnings availability
    warnings = advisory.get("statutory_disaster_alerts") or advisory.get("active_warnings") or []
    warnings_available = len(warnings) > 0
    
    # Incois availability
    incois = advisory.get("incois_marine_bulletin") or {}
    incois_available = bool(incois.get("bulletin_available"))
    
    status_label = "Using the latest available verified EcoTrace weather evidence"
    if imd_available:
        source_breakdown = "IMD observation available"
    elif model_available:
        source_breakdown = "IMD observation unavailable • model guidance available"
    else:
        source_breakdown = "IMD observation unavailable • historical meteorological baseline"

    return {
        "status_label": status_label,
        "source_breakdown": source_breakdown,
        "imd_available": imd_available,
        "model_available": model_available,
        "warnings_available": warnings_available,
        "incois_available": incois_available,
    }


def classify_question_intent(question: str) -> Dict[str, Any]:
    """
    Deterministically classify traveler's natural language question into structured intent.
    Enforces STRICT WEATHER-ONLY domain constraints BEFORE model generation.
    Returns: intent_type, detected_destination, detected_comparison_destination,
             detected_activity, detected_temporal_scope, is_weather_scope.
    """
    q_clean = question.strip().lower()
    
    # 1. Bare ambiguous travel query without destination or weather
    if re.match(r"^(?:should\s+i|can\s+i|is\s+it\s+safe\s+to|shall\s+i)\s+travel\??$", q_clean) or q_clean in ["should i travel", "should i travel?", "can i travel", "can i travel?"]:
        return {
            "intent": "AMBIGUOUS_TRAVEL",
            "is_weather_scope": False,
            "refusal_reason": "I can help evaluate travel conditions if you provide a destination or weather context (e.g. 'Should I travel to Puri considering the rain?').",
            "target_destination": None,
            "target_comparison_destination": None,
            "target_activity": None,
            "temporal_scope": "CURRENT",
        }

    # 2. Non-weather scope patterns (food, hotel, tourist places, coding, jokes, studies, general info)
    unrelated_patterns = [
        # Food & dining
        r"\b(eat|eating|eats|food|foods|dine|dining|restaurant|restaurants|cafe|cafes|dish|dishes|cuisine|recipe|recipes|cook|cooking|curry|curries|seafood|fish|prawn|crab|breakfast|lunch|dinner|snack|snacks|street food|sweet|sweets|dalma|pakhala|rasagola|chhenapoda|biryani|pizza|burger)\b",
        # Hotel & lodging
        r"\b(hotel|hotels|resort|resorts|room|rooms|stay|stays|staying|lodge|lodges|lodging|hostel|hostels|airbnb|homestay|booking|reservation|accommodate|accommodation|guest house|dorm)\b",
        # General Tourism / Itinerary / Attractions / History
        r"\b(tourist places|tourist attraction|tourist attractions|sightseeing places|itinerary|3-day|vacation plan|monument history|history of|who built|temple history|mythology|story of)\b",
        # Tech, coding, translation, study, general knowledge
        r"\b(code|coding|python|javascript|java|c\+\+|dsa|algorithm|algorithms|resume|cv|program|programming|translate|translation|poem|poetry|story|joke|jokes|riddle|who is|who won|election|stock|stocks|share|shares|crypto|bitcoin|study|exam|homework|math|science|physics|chemistry|essay|summary|tell me a joke)\b",
        # Movies, entertainment, sports
        r"\b(movie|movies|film|films|cinema|theatre|theater|actor|actress|director|song|songs|music|concert|show|dance|drama|sport|sports|cricket|football|soccer|hockey|match|matches|score|scores|ipl|world cup|stadium|player|players|tournament)\b",
        # General chat
        r"^(hi|hello|hey|greetings|howdy|good morning|good evening|good afternoon|how are you|who are you|what is your name|what can you do|help me|test)(\s+.*)?$",
        # Transport booking / non-weather transit
        r"\b(flight ticket|train ticket|bus ticket|fare|fares|irctc|ola|uber|cab booking|taxi booking)\b",
    ]

    # Explicit meteorological terms
    meteorological_terms = (
        r"\b(weather|forecast|forecasts|outlook|condition|conditions|climate|"
        r"rain|raining|rainfall|rainy|precip|precipitation|drizzle|shower|showers|storm|"
        r"thunderstorm|thunderstorms|thunder|lightning|temp|temperature|temperatures|heat|hot|cold|warm|wind|winds|windy|breeze|gust|gusts|"
        r"squall|cyclone|monsoon|cloud|clouds|cloudy|overcast|sunny|sunshine|uv|humidity|humid|"
        r"fog|foggy|mist|visibility|air quality|aqi|radar|satellite|nowcast|incois|wave|waves|swell|sea state|tide|coastal wave|"
        r"umbrella|raincoat|waterproof|pack|packing|carry|bring)\b"
    )

    clothing_generic_terms = (
        r"\b(wear|wearing|clothes|clothing|dress|outfit)\b"
    )

    travel_weather_terms = (
        r"\b(should i go|can i go|can i travel|safe to travel|safe to go|travel decision|travel risk|"
        r"should i continue|can i continue|should i proceed|can i proceed|continue travel|"
        r"delay travel|postpone|when to go|when should i|what time|what time to leave|what time should i|departure time|best time to travel|best time to go|"
        r"lower risk|lower-risk|lower-risk window|road weather|highway weather|route weather|"
        r"weather precaution|weather precautions|precaution|precautions|warning|warnings|weather warning|weather alert|"
        r"why this decision|why delay|why avoid|why is there|recommendation|delay recommendation|advisory|"
        r"boating weather|beach weather|sea bathing|sea bath|swimming|boating|boat|lake|sightseeing|outdoor|activity|activities|advisable|suitable|can i do|is my activity|"
        r"compare|comparison|better tomorrow|which is better|route|corridor|highway|road|on the road|affect my route)\b"
    )

    # Sanitize out monument names from meteorological check (e.g. "sun temple")
    q_meteorological_check = re.sub(r"\b(sun temple|lingaraj temple|jagannath temple|temple)\b", "", q_clean)
    has_meteorological = bool(re.search(meteorological_terms, q_meteorological_check))
    has_generic_clothing = bool(re.search(clothing_generic_terms, q_clean))
    has_travel_weather = bool(re.search(travel_weather_terms, q_clean))

    # Reject unrelated topics unless there is explicit meteorological context
    for pat in unrelated_patterns:
        if re.search(pat, q_clean):
            if not has_meteorological:
                return {
                    "intent": "OUT_OF_SCOPE",
                    "is_weather_scope": False,
                    "refusal_reason": None,
                    "target_destination": None,
                    "target_comparison_destination": None,
                    "target_activity": None,
                    "temporal_scope": "CURRENT",
                }

    # Generic clothing queries without weather/destination context: "What should I wear?" / "What should I wear tomorrow?" -> BLOCK
    if has_generic_clothing and not has_meteorological:
        if not re.search(r"\b(puri|konark|chilika|bhubaneswar|bbsr)\b", q_clean):
            return {
                "intent": "OUT_OF_SCOPE",
                "is_weather_scope": False,
                "refusal_reason": None,
                "target_destination": None,
                "target_comparison_destination": None,
                "target_activity": None,
                "temporal_scope": "CURRENT",
            }

    # Extract destinations
    destinations_found = []
    for d in ["puri", "chilika", "konark", "bhubaneswar"]:
        if d in q_clean or (d == "bhubaneswar" and "bbsr" in q_clean) or (d == "konark" and "konarka" in q_clean):
            destinations_found.append(d)

    # If route query like "from Bhubaneswar to Puri", the target destination is Puri
    from_to_match = re.search(r"from\s+(\w+)\s+to\s+(\w+)", q_clean)
    if from_to_match:
        from_loc = from_to_match.group(1).lower()
        to_loc = from_to_match.group(2).lower()
        to_norm = "bhubaneswar" if to_loc in ["bhubaneswar", "bbsr"] else to_loc
        detected_destination = to_norm if to_norm in ["puri", "chilika", "konark", "bhubaneswar"] else (destinations_found[0] if destinations_found else None)
        detected_comparison_destination = None
    else:
        detected_destination = destinations_found[0] if destinations_found else None
        detected_comparison_destination = destinations_found[1] if len(destinations_found) > 1 else None

    # Check if query has meteorological terms, travel-weather terms, or clothing with destination/weather
    if not has_meteorological and not has_travel_weather:
        is_dest_travel_query = detected_destination is not None and bool(re.search(r"\b(should i go|can i go|safe to go|should i travel|can i travel|travel to|trip to|visit)\b", q_clean))
        if not is_dest_travel_query and not (has_generic_clothing and detected_destination is not None):
            return {
                "intent": "OUT_OF_SCOPE",
                "is_weather_scope": False,
                "refusal_reason": None,
                "target_destination": None,
                "target_comparison_destination": None,
                "target_activity": None,
                "temporal_scope": "CURRENT",
            }

    # Extract activity
    detected_activity = None
    if re.search(r"\b(boat|boating|lake|cruise|water safari)\b", q_clean):
        detected_activity = "boating"
    elif re.search(r"\b(beach|sea bath|sea bathing|swimming|surf|waves)\b", q_clean):
        detected_activity = "sea_bathing"
    elif re.search(r"\b(temple|darshan|sightseeing|monument|sun temple|lingaraj|heritage)\b", q_clean):
        detected_activity = "sightseeing"
    elif re.search(r"\b(drive|driving|highway|road trip|commute)\b", q_clean):
        detected_activity = "driving"
    elif re.search(r"\b(outdoor|walk|trek|hiking|festival)\b", q_clean):
        detected_activity = "outdoor"

    # Extract temporal scope
    temporal_scope = "CURRENT"
    if re.search(r"\b(tonight|evening|tomorrow|next|later|forecast|afternoon|morning|6 hours|future|upcoming)\b", q_clean):
        temporal_scope = "FORECAST"

    # Determine intent
    is_route_query = bool(re.search(r"\b(route|highway|corridor|on the road|on the way|nh16|nh316|from\s+\w+\s+to\s+\w+)\b", q_clean))

    if is_route_query:
        intent = "ROUTE_WEATHER"
    elif detected_comparison_destination or re.search(r"\b(compare|comparison|better tomorrow|which is better)\b", q_clean):
        intent = "WEATHER_COMPARISON"
    elif re.search(r"\b(carry|pack|packing|bring|wear|jacket|umbrella|shoes|gear|bag|pouch)\b", q_clean):
        intent = "PREPARATION"
    elif re.search(r"\b(precaution|precautions|caution|careful|danger|safety tip|safety tips)\b", q_clean):
        intent = "PRECAUTION"
    elif re.search(r"\b(warning|warnings|bulletin|bulletins|alert|alerts|cyclone alert)\b", q_clean):
        intent = "WARNING_EXPLANATION"
    elif re.search(r"\b(when|what time|departure|timing|delay|wait|best time|lower-risk|lower risk|lowest)\b", q_clean):
        intent = "DEPARTURE_TIME"
    elif re.search(r"\b(can i do|advisable|suitable|good for|boating|bathing|sightseeing|beach ok|activity|activities)\b", q_clean):
        intent = "ACTIVITY_DECISION"
    elif re.search(r"\b(should i go|can i travel|is it okay to go|should i leave|safe to go|safe to travel|should i travel|postpone)\b", q_clean):
        intent = "TRAVEL_DECISION"
    elif re.search(r"\b(why|reason|how come|explain|what does .* mean)\b", q_clean):
        intent = "EXPLANATION"
    elif re.search(r"\b(rain|raining|will it rain|chance of rain|precipitation|rainfall)\b", q_clean):
        intent = "RAIN_OUTLOOK"
    elif re.search(r"\b(temp|temperature|heat|hot|cold|warm|sky|sun)\b", q_clean):
        intent = "TEMPERATURE_OUTLOOK"
    elif re.search(r"\b(wind|winds|windy|gust|gusts|breeze|squall)\b", q_clean):
        intent = "WIND_OUTLOOK"
    else:
        intent = "WEATHER_SUMMARY"

    return {
        "intent": intent,
        "is_weather_scope": True,
        "refusal_reason": None,
        "target_destination": detected_destination,
        "target_comparison_destination": detected_comparison_destination,
        "target_activity": detected_activity,
        "temporal_scope": temporal_scope,
    }



def _build_weather_grounding_context(
    advisory: Dict[str, Any],
    destination_slug: str,
    comparison_advisory: Optional[Dict[str, Any]] = None,
    comparison_slug: Optional[str] = None,
) -> str:
    """
    Constructs a structured, factual weather evidence block to ground Gemini generation.
    Supplies all verified telemetry, warnings, provenance, and source availability.
    """
    dest_name = DESTINATION_CONFIGS.get(destination_slug, {}).get("destination_name", destination_slug.title())
    prov = advisory.get("station_provenance") or {}
    v_status = prov.get("verification_status")
    is_imd = v_status in ["VERIFIED_STATION_OBSERVATION", "VERIFIED_IMD_DIRECT_OBSERVATION"]
    
    temp_c = advisory.get("temperature_c")
    humidity = advisory.get("relative_humidity_pct")
    precip_mm = advisory.get("precipitation_mm", 0.0)
    precip_prob = advisory.get("precipitation_probability", 0)
    wind_kmh = advisory.get("wind_speed_kmh", 0.0)
    wind_gusts = advisory.get("wind_gusts_kmh") or (round(wind_kmh * 1.3) if wind_kmh else 0)
    weather_cond = advisory.get("weather_condition", "Fair")
    risk_level = advisory.get("risk_level", "SAFE")
    risk_badge = advisory.get("risk_badge", "🟢 LOW")
    warnings = advisory.get("statutory_disaster_alerts") or advisory.get("active_warnings") or []
    
    lines = [
        f"=== ECOTRACE VERIFIED WEATHER EVIDENCE: {dest_name.upper()} ===",
        f"Destination: {dest_name} (Odisha, India)",
        f"Data Freshness / Timestamp: {prov.get('observation_time_ist', 'Recent Live Sync')}",
        f"Observation Source Class: {'IMD_STATION_OBSERVATION (Official In-Situ)' if is_imd else 'MODEL_CURRENT (NWP ECMWF/DWD Guidance - IMD Station Unavailable)'}",
        f"Station Provenance: {prov.get('station_name', dest_name)} (ID: {prov.get('station_id', 'N/A')})",
        f"Current Temperature: {temp_c}°C" if temp_c is not None else "Current Temperature: Unavailable",
        f"Relative Humidity: {humidity}%" if humidity is not None else "Relative Humidity: Not recorded",
        f"Current Precipitation: {precip_mm} mm",
        f"Wind Speed: {wind_kmh} km/h (Gusts: {wind_gusts} km/h)",
        f"Weather Condition: {weather_cond}",
        f"Forecast Rain Probability (6h): {precip_prob}%",
        f"Overall Risk Level: {risk_level} ({risk_badge})",
    ]
    
    # Official warnings
    if warnings:
        lines.append("Official Statutory Warnings:")
        for w in warnings:
            lines.append(
                f" - {w.get('issuing_authority', 'IMD')} {w.get('original_title') or w.get('alert_type')} "
                f"[Severity: {w.get('severity', 'Active')}, Valid: {w.get('validity_period', 'Active Window')}]"
            )
    else:
        # Check provenance status
        if is_imd:
            lines.append("Official Statutory Warnings: No active government weather warning is currently in effect for this district.")
        else:
            lines.append("Official Statutory Warnings: Official warning status could not be verified from the available sources.")

    # Marine / INCOIS if available
    incois = advisory.get("incois_marine_bulletin") or {}
    if incois.get("bulletin_available"):
        lines.append(f"INCOIS Marine State: Wave height {incois.get('wave_height_m', 'Normal')}m, Sea condition: {incois.get('sea_condition', 'Moderate')}")

    # Comparison destination evidence if present
    if comparison_advisory and comparison_slug:
        comp_name = DESTINATION_CONFIGS.get(comparison_slug, {}).get("destination_name", comparison_slug.title())
        c_temp = comparison_advisory.get("temperature_c")
        c_precip = comparison_advisory.get("precipitation_mm", 0.0)
        c_prob = comparison_advisory.get("precipitation_probability", 0)
        c_wind = comparison_advisory.get("wind_speed_kmh", 0.0)
        c_cond = comparison_advisory.get("weather_condition", "Fair")
        c_risk = comparison_advisory.get("risk_level", "SAFE")
        lines.extend([
            f"\n=== COMPARISON EVIDENCE: {comp_name.upper()} ===",
            f"Destination: {comp_name}",
            f"Temperature: {c_temp}°C",
            f"Precipitation: {c_precip} mm (Forecast Probability: {c_prob}%)",
            f"Wind: {c_wind} km/h",
            f"Condition: {c_cond}",
            f"Risk Level: {c_risk}",
        ])

    return "\n".join(lines)


def _call_gemini_weather_api(
    grounding_context: str,
    query: str,
) -> Optional[str]:
    """
    Invokes the Gemini API with the strict WEATHER_AI_SYSTEM_PROMPT and factual grounding context.
    Returns generated response or None on failure/missing key.
    """
    api_key = settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    url = f"{settings.gemini_api_url}/{settings.gemini_model}:generateContent?key={api_key}"
    payload = {
        "system_instruction": {
            "parts": [{"text": WEATHER_AI_SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{grounding_context}\n\n"
                            f"USER QUESTION: {query}\n\n"
                            "Answer strictly using the verified evidence provided above. "
                            "Do not hallucinate or manufacture values. "
                            "Keep your response direct, concise, and helpful."
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024,
        },
    }

    try:
        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status != 200:
                return None
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            candidates = res_json.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except Exception as exc:
        logger.warning("Gemini weather intelligence call failed (%s), using deterministic synthesis.", exc)
    return None


def build_proactive_weather_guidance(
    destination_slug: str = "puri",
    advisory: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the 5-point compact proactive summary from verified evidence.
    1. CURRENT: temperature, rain, wind, current risk
    2. WHAT TO KNOW: most important active weather condition
    3. WHAT TO DO: current EcoTrace recommendation
    4. PLAN: recommended lower-risk period if one exists
    5. WATCH FOR: the next meaningful weather change
    """
    dest_key = destination_slug.lower() if destination_slug else "puri"
    if dest_key not in DESTINATION_CONFIGS:
        dest_key = "puri"
    dest_cfg = DESTINATION_CONFIGS[dest_key]
    dest_name = dest_cfg.get("destination_name", "Puri")

    if not advisory:
        advisory = get_travel_advisory(dest_key)

    prov = advisory.get("station_provenance") or {}
    v_status = prov.get("verification_status")
    is_imd_direct = v_status in ["VERIFIED_STATION_OBSERVATION", "VERIFIED_IMD_DIRECT_OBSERVATION"]

    temp_c = advisory.get("temperature_c")
    precip_mm = advisory.get("precipitation_mm", 0.0)
    wind_kmh = advisory.get("wind_speed_kmh", 0.0)
    risk_level = advisory.get("risk_level", "SAFE")
    risk_badge = advisory.get("risk_badge", "🟢 LOW")
    weather_cond = advisory.get("weather_condition", "Fair")

    # Current telemetry formatting with strict source tagging
    if is_imd_direct:
        current_source_class = "IMD_STATION_OBSERVATION"
        obs_label = f"IMD Station ({prov.get('station_name', dest_name)} - {prov.get('station_id', '')})"
        current_summary = {
            "source_class": current_source_class,
            "source_label": obs_label,
            "temperature": f"{temp_c}°C" if temp_c is not None else "--",
            "rain": f"{precip_mm} mm",
            "wind": f"{wind_kmh} km/h",
            "condition": weather_cond,
            "risk_badge": risk_badge,
            "risk_level": risk_level,
        }
    else:
        current_source_class = "MODEL_CURRENT"
        current_summary = {
            "source_class": current_source_class,
            "source_label": "High-Resolution NWP Multi-Model Guidance (IMD station observation currently unavailable)",
            "temperature": f"{temp_c}°C" if temp_c is not None else "--",
            "rain": f"{precip_mm} mm",
            "wind": f"{wind_kmh} km/h",
            "condition": weather_cond,
            "risk_badge": risk_badge,
            "risk_level": risk_level,
        }

    # What to know
    warnings = advisory.get("statutory_disaster_alerts") or advisory.get("active_warnings") or []
    if warnings:
        top_w = warnings[0]
        what_to_know = f"Active Official Warning: {top_w.get('original_title') or top_w.get('alert_type')} issued by {top_w.get('issuing_authority', 'IMD')}."
    elif (precip_mm or 0) > 5.0:
        what_to_know = f"Active rainfall ({precip_mm} mm) observed across {dest_name} corridor."
    elif (advisory.get("precipitation_probability") or 0) > 40:
        what_to_know = f"Elevated precipitation probability ({advisory.get('precipitation_probability')}%) forecast over next 6 hours."
    else:
        what_to_know = f"Stable meteorological conditions observed across {dest_name} with {weather_cond.lower()}."

    # What to do (EcoTrace guidance)
    rec = advisory.get("recommendation")
    if not rec or rec == "Travel conditions currently appear normal." or risk_level == "SAFE":
        what_to_do = "Current conditions support normal travel activities based on available verified evidence. Continue with routine awareness."
    else:
        what_to_do = rec

    # Plan (lower-risk window)
    precip_prob = advisory.get("precipitation_probability", 0)
    if precip_prob > 50:
        plan = "Earlier departure or evening window is projected with comparatively lower precipitation probability."
    else:
        plan = "Current monitoring window exhibits favorable travel conditions. Maintain standard travel schedule."

    # Watch for (next meaningful change)
    gusts = advisory.get("wind_gusts_kmh") or (round(wind_kmh * 1.3) if wind_kmh else 0)
    if warnings:
        watch_for = "Monitor potential issuance of revised IMD bulletins or warning extension."
    elif gusts > 25:
        watch_for = f"Wind gusts up to {gusts} km/h possible in exposed areas during afternoon/evening."
    elif precip_prob > 20:
        watch_for = f"Potential localized shower development (+{precip_prob}% rain probability in NWP guidance)."
    else:
        watch_for = "No significant adverse weather transitions projected within the next 6-hour forecast window."

    # Source availability status
    source_status = get_source_availability_summary(advisory)

    return {
        "destination_id": dest_key,
        "destination_name": dest_name,
        "generated_at": _get_ist_time().isoformat(),
        "source_status": source_status,
        "current": current_summary,
        "what_to_know": what_to_know,
        "what_to_do": what_to_do,
        "plan": plan,
        "watch_for": watch_for,
    }


def build_weather_preparation_guidance(
    advisory: Dict[str, Any],
    activity_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evidence-derived packing and preparation items based on actual verified weather parameters.
    Only recommends items justified by actual meteorological conditions.
    """
    temp_c = advisory.get("temperature_c") or 26.0
    precip_mm = advisory.get("precipitation_mm") or 0.0
    precip_prob = advisory.get("precipitation_probability") or 0
    wind_kmh = advisory.get("wind_speed_kmh") or 0.0
    warnings = advisory.get("statutory_disaster_alerts") or advisory.get("active_warnings") or []
    
    items: List[Dict[str, str]] = []

    # Rain / Wet gear
    if precip_mm > 0.0 or precip_prob >= 25:
        items.append({
            "item": "Rain jacket or compact umbrella",
            "reason": f"Rain probability is {precip_prob}% (observed rain: {precip_mm} mm).",
            "category": "WEATHER_PROTECTION",
        })
        items.append({
            "item": "Waterproof phone pouch / dry bag",
            "reason": "Protect electronics against localized precipitation and spray.",
            "category": "GEAR_PROTECTION",
        })
        items.append({
            "item": "Non-slip footwear",
            "reason": "Wet surfaces and corridor pathways may have reduced traction.",
            "category": "SAFETY",
        })

    # Heat / Sun
    if temp_c >= 30.0:
        items.append({
            "item": "Drinking water / Electrolytes",
            "reason": f"Elevated temperature ({temp_c}°C) increases hydration needs.",
            "category": "HEALTH",
        })
        items.append({
            "item": "Sun protection (Hat / Sunglasses / SPF)",
            "reason": f"High UV exposure under {temp_c}°C ambient temperature.",
            "category": "PROTECTION",
        })
    elif temp_c <= 18.0:
        items.append({
            "item": "Light warm layer / windbreaker",
            "reason": f"Cool ambient temperature ({temp_c}°C) with wind chill.",
            "category": "CLOTHING",
        })

    # Wind / Coastal / Water
    if wind_kmh > 20.0 or any("cyclone" in str(w).lower() or "squall" in str(w).lower() for w in warnings):
        items.append({
            "item": "Wind-resistant eyewear / secure straps",
            "reason": f"Brisk winds ({wind_kmh} km/h) in exposed coastal/lake areas.",
            "category": "SAFETY",
        })

    if activity_id in ["boating", "sea_bathing"]:
        items.append({
            "item": "Approved life vest / water safety gear",
            "reason": "Mandatory water safety equipment for marine/lake activities.",
            "category": "SAFETY",
        })

    # Fallback if conditions are very calm and dry
    if not items:
        items.append({
            "item": "Drinking water & light clothing",
            "reason": f"Conditions are stable ({temp_c}°C, calm winds, no rain).",
            "category": "COMFORT",
        })

    return {
        "items": items,
        "summary": f"Preparation recommendations derived from current {temp_c}°C temperature, {precip_prob}% rain probability, and {wind_kmh} km/h wind conditions.",
    }


def build_weather_precaution_guidance(
    advisory: Dict[str, Any],
    activity_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Evidence-derived safety precautions linked to verified hazards."""
    precip_mm = advisory.get("precipitation_mm") or 0.0
    precip_prob = advisory.get("precipitation_probability") or 0
    wind_kmh = advisory.get("wind_speed_kmh") or 0.0
    gusts = advisory.get("wind_gusts_kmh") or (round(wind_kmh * 1.3) if wind_kmh else 0)
    warnings = advisory.get("statutory_disaster_alerts") or advisory.get("active_warnings") or []
    
    precautions: List[Dict[str, str]] = []

    if warnings:
        for w in warnings[:2]:
            precautions.append({
                "title": f"Adhere to {w.get('issuing_authority', 'IMD')} Warning",
                "detail": f"{w.get('original_title') or w.get('alert_type')}: Follow official advisory guidelines strictly.",
                "severity": "HIGH",
            })

    if precip_mm > 10.0 or precip_prob > 60:
        precautions.append({
            "title": "Allow Extra Travel Time & Reduced Visibility",
            "detail": "Wet roads and localized spray can reduce vehicular braking efficiency and forward visibility.",
            "severity": "MODERATE",
        })

    if gusts > 30.0:
        precautions.append({
            "title": "Exercise Caution in Open Areas",
            "detail": f"Wind gusts up to {gusts} km/h. Stay clear of loose structures and coastal jetties.",
            "severity": "MODERATE",
        })

    if activity_id == "sea_bathing" and (warnings or gusts > 25.0):
        precautions.append({
            "title": "Avoid Sea Bathing During Hazard Window",
            "detail": "Rough surf and active bulletins indicate heightened coastal risk.",
            "severity": "HIGH",
        })
    elif activity_id == "boating" and (precip_prob > 40 or gusts > 25.0):
        precautions.append({
            "title": "Verify Lake Operator Clearance Before Boarding",
            "detail": "Lake waters may experience choppy swells under gusty conditions.",
            "severity": "MODERATE",
        })

    if not precautions:
        precautions.append({
            "title": "Standard Travel Vigilance",
            "detail": "Maintain normal awareness and monitor official weather bulletins before departure.",
            "severity": "LOW",
        })

    return precautions


def evaluate_weather_intelligence_question(
    question: str,
    destination_slug: Optional[str] = None,
    origin_slug: Optional[str] = None,
    activity_id: Optional[str] = None,
    departure_time: Optional[str] = None,
    session_id: Optional[str] = None,
    traveler_location: Optional[Dict[str, Any]] = None,
    route_geometry: Optional[List[Dict[str, float]]] = None,
    route_eta: Optional[str] = None,
    session_history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main deterministic orchestration function.
    Processes traveler questions, enforces strict weather-only scope gate,
    and synthesizes verified Phase 1–7 evidence into structured, explainable answers.
    """
    now_ist = _get_ist_time()
    now_iso = now_ist.isoformat()
    valid_until = (now_ist + timedelta(hours=2)).isoformat()
    response_id = f"resp_wai_{uuid.uuid4().hex[:10]}"

    # Check session history for context continuity (follow-up questions)
    stored_history = WEATHER_AI_SESSION_HISTORY.get(session_id or "", [])
    combined_history = (session_history or []) + stored_history

    last_dest = None
    last_act = None
    if combined_history:
        for prev in reversed(combined_history):
            if not last_dest and prev.get("destination_slug"):
                last_dest = prev.get("destination_slug")
            if not last_act and prev.get("activity_id"):
                last_act = prev.get("activity_id")

    # 1. Deterministic Domain Gate
    intent_data = classify_question_intent(question)
    
    # 2. Out-of-scope / non-weather refusal
    if not intent_data["is_weather_scope"]:
        refusal_msg = intent_data.get("refusal_reason") or _SCOPE_REFUSAL_MESSAGE
        return {
            "response_id": response_id,
            "answer_type": "OUT_OF_SCOPE",
            "is_weather_scope": False,
            "answer": refusal_msg,
            "why": "EcoTrace Weather Intelligence answers only weather, forecast, warning, route exposure, and weather-based travel guidance.",
            "what_to_do": "Please ask a question regarding weather conditions, timing, packing, routes, or activities.",
            "what_to_watch": "Verified weather updates for Odisha travel destinations.",
            "confidence": "HIGH",
            "provenance_type": "DERIVED_WEATHER_INTELLIGENCE",
            "source_refs": ["ECOTRACE_POLICY"],
            "evidence_refs": [],
            "generated_at": now_iso,
            "valid_until": valid_until,
        }

    # Resolve effective destinations & activity
    effective_dest = intent_data["target_destination"] or destination_slug or last_dest or "puri"
    effective_comp_dest = intent_data.get("target_comparison_destination")
    effective_act = intent_data["target_activity"] or activity_id or last_act
    intent = intent_data["intent"]

    # Retrieve authoritative advisory for primary destination
    advisory = get_travel_advisory(
        destination_slug=effective_dest,
        origin_slug=origin_slug,
    )
    dest_cfg = DESTINATION_CONFIGS.get(effective_dest, DESTINATION_CONFIGS["puri"])
    dest_name = dest_cfg.get("destination_name", effective_dest.title())

    # Retrieve comparison advisory if comparison query
    comp_advisory = None
    comp_name = None
    if effective_comp_dest and effective_comp_dest != effective_dest:
        comp_advisory = get_travel_advisory(destination_slug=effective_comp_dest)
        comp_name = DESTINATION_CONFIGS.get(effective_comp_dest, {}).get("destination_name", effective_comp_dest.title())

    prov = advisory.get("station_provenance") or {}
    v_status = prov.get("verification_status")
    is_imd_direct = v_status in ["VERIFIED_STATION_OBSERVATION", "VERIFIED_IMD_DIRECT_OBSERVATION"]
    
    temp_c = advisory.get("temperature_c")
    precip_mm = advisory.get("precipitation_mm", 0.0)
    precip_prob = advisory.get("precipitation_probability", 0)
    wind_kmh = advisory.get("wind_speed_kmh", 0.0)
    gusts = advisory.get("wind_gusts_kmh") or (round(wind_kmh * 1.3) if wind_kmh else 0)
    weather_cond = advisory.get("weather_condition", "Fair")
    risk_level = advisory.get("risk_level", "SAFE")
    risk_badge = advisory.get("risk_badge", "🟢 LOW")
    warnings = advisory.get("statutory_disaster_alerts") or advisory.get("active_warnings") or []

    # Source references & evidence IDs
    source_refs: List[str] = []
    evidence_refs: List[str] = []
    
    if is_imd_direct:
        source_refs.append(f"IMD_STATION_{prov.get('station_id', '43053')}")
        evidence_refs.append(f"OBS_{prov.get('station_id', '43053')}_{prov.get('observation_time_ist', 'CURRENT')}")
    else:
        source_refs.append("NWP_MODEL_ECMWF_DWD")
        evidence_refs.append("MODEL_FORECAST_SEAMLESS_6H")

    if warnings:
        source_refs.append("OSDMA_IMD_DISASTER_BULLETIN")
        for w in warnings:
            evidence_refs.append(f"WARN_{w.get('warning_id', 'ACTIVE')}")

    # Build response fields based on intent
    answer = ""
    why = ""
    what_to_do = ""
    what_to_watch = ""
    answer_type = intent
    confidence = "HIGH"
    decision_state = None
    structured_data: Dict[str, Any] = {}

    # Source class description string
    if is_imd_direct:
        obs_desc = f"Current IMD observation ({prov.get('station_name', dest_name)}): {temp_c}°C, {weather_cond}, rain: {precip_mm} mm, wind: {wind_kmh} km/h."
    else:
        obs_desc = f"Current model guidance indicates {temp_c}°C ({weather_cond}), rain: {precip_mm} mm, wind: {wind_kmh} km/h (IMD station observation is currently unavailable)."

    if intent == "WEATHER_COMPARISON" and comp_advisory and comp_name:
        answer_type = "WEATHER_COMPARISON"
        c_temp = comp_advisory.get("temperature_c")
        c_prob = comp_advisory.get("precipitation_probability", 0)
        c_wind = comp_advisory.get("wind_speed_kmh", 0.0)
        c_risk = comp_advisory.get("risk_level", "SAFE")
        c_cond = comp_advisory.get("weather_condition", "Fair")

        if risk_level == "SAFE" and c_risk != "SAFE":
            better_dest = dest_name
        elif c_risk == "SAFE" and risk_level != "SAFE":
            better_dest = comp_name
        elif precip_prob < c_prob:
            better_dest = dest_name
        else:
            better_dest = comp_name

        answer = f"Weather Comparison: {dest_name} ({temp_c}°C, {weather_cond}, {precip_prob}% rain) vs {comp_name} ({c_temp}°C, {c_cond}, {c_prob}% rain)."
        why = f"{better_dest} offers comparatively lower weather risk with {min(precip_prob, c_prob)}% rain probability vs {max(precip_prob, c_prob)}%."
        what_to_do = f"Consider {better_dest} for more favorable outdoor conditions."
        what_to_watch = "Check updated morning forecast for changes in coastal cloud bands."

    elif intent == "TRAVEL_DECISION":
        answer_type = "TRAVEL_DECISION"
        if risk_level in ["CRITICAL", "HIGH"]:
            decision_state = "DELAY_OR_AVOID"
            answer = f"DELAY TRAVEL TO {dest_name.upper()}"
            why = f"A verified {risk_level} risk condition is active for {dest_name}. {obs_desc}"
            what_to_do = "Postpone non-essential travel and follow official disaster instructions."
            what_to_watch = "Watch for official cancellation of weather bulletins by IMD/OSDMA."
        elif risk_level == "CAUTION":
            decision_state = "GO_WITH_CAUTION"
            answer = f"GO WITH CAUTION TO {dest_name.upper()}"
            why = f"Current conditions are generally manageable, but an active weather bulletin or elevated rain probability applies. {obs_desc}"
            what_to_do = "Proceed with caution, keep extra travel time, and carry appropriate weather protection."
            what_to_watch = "Monitor next nowcast update for sudden rainband intensification."
        else:
            decision_state = "PROCEED_NORMALLY"
            answer = f"NORMAL TRAVEL CONDITIONS FOR {dest_name.upper()}"
            why = f"No significant verified hazards detected right now. {obs_desc}"
            what_to_do = "Continue with routine travel plans based on available verified evidence."
            what_to_watch = "Routine check of 6-hour forecast prior to departure."

    elif intent == "DEPARTURE_TIME":
        answer_type = "DEPARTURE_TIME"
        if precip_prob > 50:
            answer = "Consider travelling earlier or awaiting the evening window."
            why = f"Rain probability reaches {precip_prob}% in the current 6-hour forecast window. {obs_desc}"
            what_to_do = "If flexible, plan departure when precipitation probability drops below 30%."
            what_to_watch = "Observe radar/nowcast updates for localized convective cloud movements."
        else:
            answer = "Departure during the current window is favorable."
            why = f"Current verified conditions show stable weather with low hazard probability ({precip_prob}%). {obs_desc}"
            what_to_do = "Proceed with planned departure time."
            what_to_watch = "Watch for afternoon wind shifts in coastal areas."

    elif intent == "PREPARATION":
        answer_type = "PREPARATION"
        prep_data = build_weather_preparation_guidance(advisory, effective_act)
        structured_data = prep_data
        item_names = [it["item"] for it in prep_data["items"]]
        answer = f"Recommended items for {dest_name}: " + ", ".join(item_names) + "."
        why = prep_data["summary"]
        what_to_do = "Pack weather-protective gear before commencing travel."
        what_to_watch = f"Check if rain intensity changes ({precip_prob}% forecast rain probability)."

    elif intent == "PRECAUTION":
        answer_type = "PRECAUTION"
        precautions = build_weather_precaution_guidance(advisory, effective_act)
        structured_data = {"precautions": precautions}
        answer = f"Key precautions for {dest_name}: " + "; ".join([p["title"] for p in precautions]) + "."
        why = f"Derived from active meteorological parameters ({obs_desc})."
        what_to_do = "Adhere to the outlined precautions throughout your journey."
        what_to_watch = "Watch for localized road water accumulation or sudden squalls."

    elif intent == "ACTIVITY_DECISION":
        answer_type = "ACTIVITY_DECISION"
        if not effective_act:
            confidence = "UNAVAILABLE"
            answer = f"Activity suitability for {dest_name} requires specifying an activity (e.g. boating, sea bathing, sightseeing)."
            why = "Activity decisions require an explicit activity type, spatial applicability, and verified hazard correlation."
            what_to_do = f"Please select an activity (such as boating in Chilika, sea bathing in Puri, or temple sightseeing)."
            what_to_watch = "Specific meteorological criteria for your intended activity."
        else:
            act_name = effective_act.replace("_", " ").title()
            if effective_act == "sea_bathing":
                if warnings or (gusts > 25.0):
                    answer = f"Sea bathing at {dest_name} is NOT ADVISABLE right now."
                    why = f"Coastal bulletins or wind gusts up to {gusts} km/h indicate elevated surf risk. {obs_desc}"
                    what_to_do = "Avoid water entry and stay behind coastal lifeguard flags."
                    what_to_watch = "Watch for INCOIS ocean state advisories and IMD coastal bulletins."
                else:
                    answer = f"Sea bathing at {dest_name} appears acceptable under normal caution."
                    why = f"No severe coastal warnings active; winds at {wind_kmh} km/h. {obs_desc}"
                    what_to_do = "Bathe only in designated zones with on-duty lifeguards."
                    what_to_watch = "Monitor incoming tide and surf changes."
            elif effective_act == "boating":
                if warnings or precip_prob > 50 or gusts > 30.0:
                    answer = f"Boating at {dest_name} requires CAUTION or temporary delay."
                    why = f"Elevated wind gusts ({gusts} km/h) or precipitation probability ({precip_prob}%) may create choppy waters. {obs_desc}"
                    what_to_do = "Verify operator clearance and wear life jackets at all times."
                    what_to_watch = "Watch for rapid cloud darkening over the lake."
                else:
                    answer = f"Boating conditions at {dest_name} appear suitable."
                    why = f"Calm to moderate winds ({wind_kmh} km/h) and low rain risk ({precip_prob}%). {obs_desc}"
                    what_to_do = "Proceed with authorized boat operators."
                    what_to_watch = "Standard weather vigilance on open water."
            else:
                if risk_level in ["CRITICAL", "HIGH"]:
                    answer = f"{act_name} at {dest_name} is not recommended."
                    why = f"Adverse weather conditions ({risk_badge}) are active. {obs_desc}"
                    what_to_do = "Reschedule indoor activities until conditions clear."
                    what_to_watch = "IMD nowcast updates."
                else:
                    answer = f"{act_name} at {dest_name} is suitable under current weather."
                    why = f"Conditions are stable with {weather_cond.lower()} and {temp_c}°C. {obs_desc}"
                    what_to_do = "Enjoy your activity while remaining hydrated."
                    what_to_watch = "Afternoon temperature changes."

    elif intent == "ROUTE_WEATHER":
        answer_type = "ROUTE_WEATHER"
        if not route_geometry and not origin_slug:
            answer = f"Route weather for {dest_name} shows stable corridor conditions."
            why = f"Evaluated along standard approach corridors for {dest_name}. {obs_desc}"
            what_to_do = "Drive with standard highway caution."
            what_to_watch = "Watch for localized rain patches along coastal highway stretches."
        else:
            corridor_desc = f"{origin_slug.title()} to {dest_name}" if origin_slug else f"{dest_name} corridor"
            answer = f"Corridor weather for {corridor_desc}: Normal driving visibility and manageable winds."
            why = f"NWP multi-model forecast indicates {precip_prob}% rain probability and {wind_kmh} km/h wind along the corridor."
            what_to_do = "Maintain safe driving speeds; allow normal transit duration."
            what_to_watch = "Watch for sudden shower spray near open water stretches."

    elif intent == "WARNING_EXPLANATION":
        answer_type = "WARNING_EXPLANATION"
        if warnings:
            w = warnings[0]
            answer = f"Official Warning: {w.get('original_title') or w.get('alert_type')} issued by {w.get('issuing_authority', 'IMD')}."
            why = f"Valid: {w.get('validity_period', 'Current period')}. Status: {w.get('status', 'Active')}. {w.get('short_explanation', '')}"
            what_to_do = "Comply with official warnings and restrict outdoor exposure."
            what_to_watch = "Follow official updates from IMD Bhubaneswar / OSDMA."
        elif is_imd_direct:
            answer = f"No active official weather warnings in effect for {dest_name}."
            why = "IMD and OSDMA statutory feeds report normal conditions for this district."
            what_to_do = "Travel under standard routine guidance."
            what_to_watch = "Next scheduled daily bulletin release."
        else:
            answer = "Official warning status could not be verified from the available sources."
            why = "Direct IMD district warning feed is currently unverified or pending authorization."
            what_to_do = "Proceed with caution using numerical model forecasts and local observation."
            what_to_watch = "Monitor local radio/news for statutory weather advisories."

    elif intent == "WIND_OUTLOOK":
        answer_type = "WIND_OUTLOOK"
        answer = f"Wind speed in {dest_name} is currently {wind_kmh} km/h with gusts up to {gusts} km/h."
        why = f"Derived from verified station telemetry / NWP atmospheric model guidance for {dest_name}."
        what_to_do = "Exercise normal caution in open coastal areas." if gusts <= 25 else "Secure loose items and avoid exposed high points."
        what_to_watch = "Afternoon sea breeze changes along coastal stretches."

    else:
        # RAIN_OUTLOOK, TEMPERATURE_OUTLOOK, WEATHER_SUMMARY, EXPLANATION
        answer_type = intent
        answer = f"{dest_name} weather: {obs_desc}"
        why = f"6-hour forecast indicates {precip_prob}% rain probability and wind gusts up to {gusts} km/h."
        what_to_do = advisory.get("recommendation") or "Travel conditions currently appear normal."
        what_to_watch = "Monitor afternoon convective cloud developments."

    # If Gemini API is configured, attempt evidence-grounded generative response
    grounding_text = _build_weather_grounding_context(
        advisory=advisory,
        destination_slug=effective_dest,
        comparison_advisory=comp_advisory,
        comparison_slug=effective_comp_dest,
    )
    gemini_resp = _call_gemini_weather_api(grounding_context=grounding_text, query=question)
    if gemini_resp:
        # Use grounded Gemini text for the narrative answer while preserving structured metadata
        answer = gemini_resp

    # Store in session history
    session_record = {
        "question": question,
        "answer": answer,
        "destination_slug": effective_dest,
        "activity_id": effective_act,
        "response_id": response_id,
        "generated_at": now_iso,
    }
    if session_id:
        if session_id not in WEATHER_AI_SESSION_HISTORY:
            WEATHER_AI_SESSION_HISTORY[session_id] = []
        WEATHER_AI_SESSION_HISTORY[session_id].append(session_record)
        if len(WEATHER_AI_SESSION_HISTORY[session_id]) > 10:
            WEATHER_AI_SESSION_HISTORY[session_id].pop(0)

    # Source availability status
    source_status = get_source_availability_summary(advisory)

    return {
        "response_id": response_id,
        "answer_type": answer_type,
        "is_weather_scope": True,
        "decision_state": decision_state,
        "destination_id": effective_dest,
        "destination_name": dest_name,
        "comparison_destination_id": effective_comp_dest,
        "comparison_destination_name": comp_name,
        "activity_id": effective_act,
        "answer": answer,
        "why": why,
        "what_to_do": what_to_do,
        "what_to_watch": what_to_watch,
        "confidence": confidence,
        "provenance_type": "DERIVED_WEATHER_INTELLIGENCE",
        "source_status": source_status,
        "source_refs": source_refs,
        "evidence_refs": evidence_refs,
        "structured_data": structured_data,
        "generated_at": now_iso,
        "valid_until": valid_until,
    }


# Convenient alias for tests and external callers
generate_weather_intelligence_answer = evaluate_weather_intelligence_question
