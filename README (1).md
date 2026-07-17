# 🌪️ Disaster Climate Alert AI Agent

An autonomous AI agent built with **LangGraph** that detects hyperlocal disaster risk, generates accessible multi-language alerts, and automatically triggers escalation actions — without requiring human approval at any step.

## Problem Statement

Government weather warning systems suffer from three core gaps:

1. **Delayed/generic alerts** — Warnings are broad (state/district-level) instead of hyperlocal, and arrive with lag.
2. **Low accessibility** — Alerts assume smartphone access and literacy in the alert's language; elderly, rural, and non-native speaking populations are often missed.
3. **No automated action loop** — Even when a warning exists, there is no system that automatically escalates it (contacting authorities, triggering alerts) — it requires manual follow-through.

This project addresses all three gaps using an agentic pipeline.

## How It Works

The agent is built as a **LangGraph** state machine with 4 nodes:

```
Data Intake Node → Severity Router Node → Localization Node → Escalation Node
        ↓                    ↓                    ↓                  ↓
   Live weather      Hardcoded rules +      Simple, translated    Simulated
   data (OpenWeather)  LLM classification    alert message         SMS/Call/
                       (SAFE/WATCH/           (English + local      Authority
                       WARNING/EMERGENCY)      language)             alert
```

### 1. Data Intake Node
Fetches live weather data (temperature, rainfall, wind speed) for a given location using the **OpenWeather API**.

### 2. Severity Router Node
Classifies risk using a hybrid approach:
- **Hardcoded thresholds** for obvious cases (e.g., rainfall > 100mm/hr → `EMERGENCY`) — instant, no LLM cost, no hallucination risk.
- **LLM reasoning** (via **Groq / Llama 3.3 70B**) for ambiguous cases that don't clearly fall into a threshold.

Every decision includes a **logged reasoning trail** for auditability.

### 3. Localization Node
Generates a short, simple alert message (5th-grade reading level, clear action instructions, no jargon) in **English**, then translates it into the target regional language (Hindi, Malayalam, Tamil, Telugu, Bengali, Kannada) using **Google Translate** — chosen over LLM-based translation after testing showed significantly better grammatical accuracy for low-resource Indian languages.

### 4. Escalation Node
Triggers a graduated response based on severity:
- `WATCH` → SMS alert
- `WARNING` → SMS + automated call
- `EMERGENCY` → SMS + call + simulated authority/siren alert logged

*(Twilio integration is built and testable; simulated in the deployed demo due to trial-account phone verification constraints.)*

## Why This Is an Agent, Not a Chatbot

- Autonomous decision-making via a router node (no human approval required)
- Shared state/memory (`DisasterState`) persists across all steps
- Conditional branching — the graph takes different paths based on model output
- Actions are triggered automatically based on the agent's own classification
- Full reasoning trail logged at every decision point for auditability

## Tech Stack

| Component | Tool |
|---|---|
| Agent orchestration | LangGraph |
| LLM | Groq (Llama 3.3 70B) |
| Weather data | OpenWeather API |
| Translation | Google Translate (via `deep-translator`) |
| Escalation | Twilio (simulated in demo) |
| Frontend + hosting | Streamlit |

## Known Limitations

- Classifies risk based on **real-time weather conditions only** — does not yet account for delayed risks such as river levels or dam releases from rainfall that occurred earlier. A natural extension would be integrating river gauge/reservoir-level APIs.
- Twilio escalation is simulated in the public demo due to free-trial phone verification limits.
- Currently supports 6 Indian regional languages; extensible to more via the `lang_codes` mapping.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add your API keys to `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_groq_key"
OPENWEATHER_API_KEY = "your_openweather_key"
```

## Live Demo

*(Add your Streamlit Cloud link here after deployment)*

## Author

Built as a final-year Computer Science (Data Science specialization) project.
