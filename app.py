import streamlit as st
import os
import requests
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict
from deep_translator import GoogleTranslator

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["OPENWEATHER_API_KEY"] = st.secrets["OPENWEATHER_API_KEY"]

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
SIMULATE_TWILIO = True

class DisasterState(TypedDict):
    location: str
    temperature: float
    rainfall: float
    wind_speed: float
    severity: str
    reasoning: str
    alert_message: dict
    language: str
    log_entry: str

# ============ PASTE YOUR 4 NODE FUNCTIONS HERE ============
# data_intake_node


def data_intake_node(state: DisasterState):
    location = state["location"]
    api_key = os.environ["OPENWEATHER_API_KEY"]
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    rainfall = data.get("rain", {}).get("1h", 0)  # rainfall in mm in last 1hr, 0 if none
    temperature = data["main"]["temp"]
    wind_speed = data["wind"]["speed"] * 3.6  # convert m/s to km/h

    print(f"Fetched: {location} | Temp: {temperature}°C | Rain: {rainfall}mm | Wind: {wind_speed}km/h")

    return {
        "location": location,
        "temperature": temperature,
        "rainfall": rainfall,
        "wind_speed": wind_speed
    }

# severity_router_node

def severity_router_node(state: DisasterState):
    rainfall = state["rainfall"]
    wind_speed = state["wind_speed"]

    # HARDCODED thresholds first — instant, no LLM cost, no hallucination risk
    if rainfall > 100 or wind_speed > 90:
        severity = "EMERGENCY"
        reasoning = f"Hardcoded rule triggered: rainfall={rainfall}mm, wind={wind_speed}km/h exceeds emergency threshold"
        return {"severity": severity, "reasoning": reasoning}

    if rainfall < 10 and wind_speed < 30:
        severity = "SAFE"
        reasoning = f"Hardcoded rule triggered: rainfall={rainfall}mm, wind={wind_speed}km/h within safe range"
        return {"severity": severity, "reasoning": reasoning}

    # AMBIGUOUS case — only NOW do we call the LLM (saves cost, matches video's chest-pain logic)
    prompt = f"""You are a disaster risk classifier. Based on this data:
    Rainfall: {rainfall}mm/hr
    Wind Speed: {wind_speed}km/h

    Classify the risk as exactly one word: SAFE, WATCH, WARNING, or EMERGENCY.
    Then on a new line, give a one-sentence reasoning.

    Respond in this exact format:
    SEVERITY: <word>
    REASONING: <one sentence>
    """

    response = llm.invoke(prompt)
    text = response.content

    severity = text.split("SEVERITY:")[1].split("\n")[0].strip()
    reasoning = text.split("REASONING:")[1].strip()

    return {"severity": severity, "reasoning": reasoning}



# localization_node  (the GoogleTranslator version)


from deep_translator import GoogleTranslator

def localization_node(state: DisasterState):
    severity = state["severity"]
    reasoning = state["reasoning"]
    location = state["location"]
    language = state.get("language", "Hindi")

    # Step 1: LLM writes ONLY in English — this is what it's actually good at
    prompt = f"""You are writing a disaster alert for a rural/elderly audience.

    Location: {location}
    Severity: {severity}
    Technical reasoning: {reasoning}

    Write a short alert message (2-3 sentences max) using:
    - Very simple words (5th-grade reading level)
    - Clear action instruction (what should the person DO right now)
    - No technical jargon (no mm, km/h, meteorological terms)

    Respond with ONLY the English alert message, nothing else.
    """

    response = llm.invoke(prompt)
    english_msg = response.content.strip()

    # Step 2: Translate using Google Translate (much better grammar than LLM translation)
    lang_codes = {
        "Hindi": "hi", "Malayalam": "ml", "Tamil": "ta",
        "Telugu": "te", "Bengali": "bn", "Kannada": "kn"
    }
    lang_code = lang_codes.get(language, "hi")

    local_msg = GoogleTranslator(source='en', target=lang_code).translate(english_msg)

    alert_message = {
        "english": english_msg,
        "local": local_msg,
        "local_language": language
    }

    print(f"English: {english_msg}")
    print(f"{language}: {local_msg}")

    return {"alert_message": alert_message}

# escalation_node

def escalation_node(state: DisasterState):
    severity = state["severity"]
    location = state["location"]

    # Handle SAFE case FIRST — no alert_message exists since localize was skipped
    if severity == "SAFE":
        log_entry = f"[SAFE] {location}: No action needed"
        print(log_entry)
        return {"log_entry": log_entry}

    # For all other severities, alert_message will exist
    alert_message = state["alert_message"]

    if SIMULATE_TWILIO:
        if severity == "WATCH":
            log_entry = f"[WATCH] {location}: SIMULATED SMS sent -> {alert_message['english']}"
        elif severity == "WARNING":
            log_entry = f"[WARNING] {location}: SIMULATED SMS + Call sent -> {alert_message['english']}"
        else:  # EMERGENCY
            log_entry = f"[EMERGENCY] {location}: SIMULATED SMS + Call + Authority Alert Logged -> {alert_message['english']}"
        print(log_entry)
        return {"log_entry": log_entry}


def route_after_severity(state: DisasterState):
    if state["severity"] == "SAFE":
        return "skip_to_log"
    else:
        return "localize"

graph = StateGraph(DisasterState)
graph.add_node("intake", data_intake_node)
graph.add_node("router", severity_router_node)
graph.add_node("localize", localization_node)
graph.add_node("escalate", escalation_node)
graph.set_entry_point("intake")
graph.add_edge("intake", "router")
graph.add_conditional_edges("router", route_after_severity, {"localize": "localize", "skip_to_log": "escalate"})
graph.add_edge("localize", "escalate")
graph.add_edge("escalate", END)
app_graph = graph.compile()

st.set_page_config(page_title="Disaster Alert AI Agent", page_icon="🌪️")
st.title("🌪️ Disaster Climate Alert AI Agent")
st.write("Autonomous agent for hyperlocal risk detection, multi-language alerts, and automated escalation.")

location = st.text_input("Enter Location (city name)", "Kochi")
language = st.selectbox("Local Language", ["Hindi", "Malayalam", "Tamil", "Telugu", "Bengali", "Kannada"])

if st.button("Run Disaster Risk Check"):
    with st.spinner("Agent analyzing risk..."):
        result = app_graph.invoke({"location": location, "language": language})

    st.subheader("Result")
    st.metric("Severity", result["severity"])
    st.write(f"**Reasoning:** {result['reasoning']}")

    if result.get("alert_message"):
        st.write("**Alert (English):**", result["alert_message"]["english"])
        st.write(f"**Alert ({language}):**", result["alert_message"]["local"])

    st.write("**Action Log:**", result["log_entry"])

#demo

    st.divider()
st.subheader("🧪 Demo: Simulated Emergency Scenario")
st.caption("Since real-time weather changes unpredictably, use this to demonstrate the full pipeline (including translation + escalation).")

if st.button("Run Simulated Emergency Test"):
    with st.spinner("Agent analyzing simulated emergency..."):
        test_state = {"location": location, "language": language, "rainfall": 150, "wind_speed": 40}
        state_after_router = severity_router_node(test_state)
        test_state.update(state_after_router)
        state_after_local = localization_node(test_state)
        test_state.update(state_after_local)
        state_after_escalate = escalation_node(test_state)
        test_state.update(state_after_escalate)

    st.metric("Severity", test_state["severity"])
    st.write(f"**Reasoning:** {test_state['reasoning']}")
    st.write("**Alert (English):**", test_state["alert_message"]["english"])
    st.write(f"**Alert ({language}):**", test_state["alert_message"]["local"])
    st.write("**Action Log:**", test_state["log_entry"])