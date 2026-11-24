!pip install langgraph langchain-groq requests

import os
os.environ["GROQ_API_KEY"] = " "
os.environ["OPENWEATHER_API_KEY"] = " "  # openweathermap.org

!pip install -q deep-translator langgraph langchain-groq groq requests twilio python-dotenv
"""
Disaster alert agent (LangGraph).
Pipeline: intake (OpenWeather) -> router (rules, LLM for ambiguous cases)
          -> localize (LLM English + Google Translate) -> escalate (SMS/call or simulated)
"""

import os
import time
from typing import TypedDict
from xml.sax.saxutils import escape

import requests
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END


# ---------------------------------------------------------------------------
# Configuration (read from environment; never hardcode secrets)
# ---------------------------------------------------------------------------
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
SIMULATE_TWILIO = os.environ.get("SIMULATE_TWILIO", "true").lower() == "true"
VALID_SEVERITIES = {"SAFE", "WATCH", "WARNING", "EMERGENCY"}

LANG_CODES = {
    "Hindi": "hi",
    "Malayalam": "ml",
    "Tamil": "ta",
    "Telugu": "te",
    "Bengali": "bn",
    "Kannada": "kn",
}

# Emergency numbers are read from the environment so no real phone numbers live in code
ALERT_RECIPIENT = os.environ.get("ALERT_RECIPIENT", "")

llm = ChatGroq(model=GROQ_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class DisasterState(TypedDict, total=False):
    location: str
    temperature: float
    rainfall: float
    wind_speed: float
    severity: str
    reasoning: str
    alert_message: dict
    language: str
    log_entry: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_severity(text: str) -> str:
    """Return a valid severity word. Unparseable output escalates to WATCH, never SAFE."""
    for line in text.splitlines():
        if line.strip().upper().startswith("SEVERITY:"):
            word = line.split(":", 1)[1].strip().upper()
            if word in VALID_SEVERITIES:
                return word
    return "WATCH"


def parse_reasoning(text: str) -> str:
    for line in text.splitlines():
        if line.strip().upper().startswith("REASONING:"):
            return line.split(":", 1)[1].strip()
    return "Model reasoning unavailable; classified conservatively."


def translate_with_retry(text: str, target: str, source: str = "en", retries: int = 3) -> str:
    """Translate with exponential backoff on rate limits. Falls back to the original text."""
    for attempt in range(retries):
        try:
            return GoogleTranslator(source=source, target=target).translate(text)
        except TooManyRequests:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"Translation rate limited. Retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:  # network errors, invalid payloads, etc.
            print(f"Translation error: {e}")
            break
    print("Translation unavailable. Falling back to English.")
    return text


# ---------------------------------------------------------------------------
# Node 1: intake
# ---------------------------------------------------------------------------
def data_intake_node(state: DisasterState):
    location = state["location"]
    api_key = os.environ.get("OPENWEATHER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not set")

    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": location, "appid": api_key, "units": "metric"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    rainfall = data.get("rain", {}).get("1h", 0)           # mm in last hour
    temperature = data["main"]["temp"]
    wind_speed = data["wind"]["speed"] * 3.6               # m/s -> km/h

    print(f"Fetched: {location} | Temp: {temperature}°C | Rain: {rainfall}mm | Wind: {wind_speed:.1f}km/h")

    return {
        "location": location,
        "temperature": temperature,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
    }


# ---------------------------------------------------------------------------
# Node 2: severity router (rules first, LLM only for ambiguous cases)
# ---------------------------------------------------------------------------
def severity_router_node(state: DisasterState):
    rainfall = state["rainfall"]
    wind_speed = state["wind_speed"]

    # Hardcoded thresholds: instant, free, no hallucination risk
    if rainfall > 100 or wind_speed > 90:
        return {
            "severity": "EMERGENCY",
            "reasoning": f"Rule triggered: rainfall={rainfall}mm, wind={wind_speed:.1f}km/h exceeds emergency threshold",
        }

    if rainfall < 10 and wind_speed < 30:
        return {
            "severity": "SAFE",
            "reasoning": f"Rule triggered: rainfall={rainfall}mm, wind={wind_speed:.1f}km/h within safe range",
        }

    # Ambiguous: ask the LLM
    prompt = f"""You are a disaster risk classifier. Based on this data:
Rainfall: {rainfall}mm/hr
Wind speed: {wind_speed:.1f}km/h

Classify the risk as exactly one word: SAFE, WATCH, WARNING, or EMERGENCY.
Then on a new line, give a one-sentence reasoning.

Respond in this exact format:
SEVERITY: <word>
REASONING: <one sentence>
"""
    try:
        text = llm.invoke(prompt).content
    except Exception as e:
        print(f"LLM classification failed: {e}")
        return {
            "severity": "WATCH",
            "reasoning": "Classifier unavailable; escalated conservatively for human review.",
        }

    return {"severity": parse_severity(text), "reasoning": parse_reasoning(text)}


def route_after_severity(state: DisasterState):
    return "skip_to_log" if state["severity"] == "SAFE" else "localize"


# ---------------------------------------------------------------------------
# Node 3: localization (LLM writes English, Google Translate handles the local language)
# ---------------------------------------------------------------------------
def localization_node(state: DisasterState):
    severity = state["severity"]
    reasoning = state["reasoning"]
    location = state["location"]
    language = state.get("language", "Hindi")

    prompt = f"""You are writing a disaster alert for a rural and elderly audience.

Location: {location}
Severity: {severity}
Technical reasoning: {reasoning}

Write a short alert message (2-3 sentences max) using:
- Very simple words (5th-grade reading level)
- A clear action instruction (what the person should do right now)
- No technical jargon (no mm, km/h, or meteorological terms)

Respond with ONLY the English alert message, nothing else.
"""
    try:
        english_msg = llm.invoke(prompt).content.strip()
    except Exception as e:
        print(f"LLM alert writing failed: {e}")
        english_msg = (
            f"{severity} alert for {location}. Stay indoors, keep your phone charged, "
            "and follow instructions from local authorities."
        )

    lang_code = LANG_CODES.get(language, "hi")
    local_msg = translate_with_retry(english_msg, target=lang_code)

    alert_message = {
        "english": english_msg,
        "local": local_msg,
        "local_language": language,
    }
    print(f"English: {english_msg}")
    print(f"{language}: {local_msg}")
    return {"alert_message": alert_message}


# ---------------------------------------------------------------------------
# Node 4: escalation
# ---------------------------------------------------------------------------
def send_live_alerts(severity: str, location: str, alert_message: dict) -> str:
    """Send real SMS/calls via Twilio. Imported lazily so the demo runs without twilio installed."""
    from twilio.rest import Client

    sid = os.environ.get("TWILIO_SID")
    token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_PHONE")
    if not all([sid, token, from_number, ALERT_RECIPIENT]):
        return f"[{severity}] {location}: Live alert skipped, Twilio credentials or ALERT_RECIPIENT missing"

    client = Client(sid, token)
    sms_body = f"{alert_message['english']}\n\n{alert_message['local']}"
    twiml = f"<Response><Say>{escape(alert_message['english'])}</Say></Response>"

    try:
        client.messages.create(body=sms_body, from_=from_number, to=ALERT_RECIPIENT)
        if severity in {"WARNING", "EMERGENCY"}:
            client.calls.create(twiml=twiml, from_=from_number, to=ALERT_RECIPIENT)
    except Exception as e:
        return f"[{severity}] {location}: Escalation attempted but failed: {e}"

    if severity == "EMERGENCY":
        return f"[EMERGENCY] {location}: SMS + Call sent, authority alert logged"
    if severity == "WARNING":
        return f"[WARNING] {location}: SMS + Call sent"
    return f"[WATCH] {location}: SMS sent"


def escalation_node(state: DisasterState):
    severity = state["severity"]
    location = state["location"]

    if severity == "SAFE":
        log_entry = f"[SAFE] {location}: No action needed"
    elif SIMULATE_TWILIO:
        english = state["alert_message"]["english"]
        log_entry = f"[{severity}] {location}: SIMULATED alert -> {english}"
    else:
        log_entry = send_live_alerts(severity, location, state["alert_message"])

    print(log_entry)
    return {"log_entry": log_entry}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(DisasterState)

    graph.add_node("intake", data_intake_node)
    graph.add_node("router", severity_router_node)
    graph.add_node("localize", localization_node)
    graph.add_node("escalate", escalation_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "router")
    graph.add_conditional_edges(
        "router",
        route_after_severity,
        {"localize": "localize", "skip_to_log": "escalate"},
    )
    graph.add_edge("localize", "escalate")
    graph.add_edge("escalate", END)

    return graph.compile()


app = build_graph()


# ---------------------------------------------------------------------------
# Entry point: nothing runs on import
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    final = app.invoke({"location": "Kochi", "language": "Malayalam"})

    print("\n--- FINAL RESULT ---")
    print(f"Location: {final['location']}")
    print(f"Severity: {final['severity']}")
    print(f"Reasoning: {final['reasoning']}")
    if final.get("alert_message"):
        print(f"Alert (EN): {final['alert_message']['english']}")
        print(f"Alert (Local): {final['alert_message']['local']}")
    print(f"Log: {final['log_entry']}")
