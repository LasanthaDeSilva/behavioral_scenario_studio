"""
NINOLADES OUTREACH INTELLIGENCE LAB
===================================

A self-contained Streamlit application for:
    - Designing outreach scenarios
    - Modeling likely engagement pathways
    - Predicting real-world outcomes and metrics
    - Running a live outreach copilot with floating real-time AI Voice
    - Recording lightweight observations
    - Comparing predicted vs observed engagement
    - Measuring optional real-world impact
    - Exploring counterfactual interventions
    - Extracting qualitative memory/engagement themes
    - Viewing event-level analytics

IMPORTANT METHODOLOGICAL PRINCIPLES
-----------------------------------
1. AI predictions are hypotheses, not measurements.
2. AI must not diagnose participants.
3. AI must not infer protected/sensitive personal attributes.
4. Participant-stated preferences outrank model inference.
5. Observed behavior is kept separate from interpretation.
6. Deterministic analytics are calculated in Python.
7. Gemini is used for interpretation/generation, not for arithmetic.
8. "No intervention" is a legitimate recommendation.
9. Confidence is not probability of truth.
10. Public-facing results should distinguish:
      OBSERVED
      STATED
      INFERRED
      HYPOTHESIS
11. Optional surveys are supplementary, not the core of the system.
12. The system is intended for exploratory educational/outreach use,
    not clinical or psychological assessment.
"""

import os
import re
import html
import uuid
import math
import sqlite3
import textwrap
from datetime import datetime, timezone
from typing import List, Literal, Optional, Dict, Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Text,
    Float,
    Integer,
    ForeignKey,
    Boolean,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from pydantic import BaseModel, Field, ValidationError

from google import genai
from google.genai import types


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "Outreach Intelligence Lab"
APP_VERSION = "1.2.1"

# Real, currently existing Gemini endpoints preserved strictly
MODEL_FLASH = "gemini-3.6-flash"
MODEL_PRO = "gemini-3.1-pro"
MODEL_LITE = "gemini-3.5-flash-lite"

DEFAULT_MODEL = MODEL_FLASH

DATABASE_URL = os.getenv(
    "OUTREACH_DATABASE_URL",
    "sqlite:///ninolades_outreach_lab.db"
)


# ============================================================
# 2. PAGE CONFIGURATION & USER SESSION ISOLATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ensure each user/browser link has an isolated memory space that survives refresh
if "user_session_token" not in st.session_state:
    query_params = st.query_params
    if "session_id" in query_params:
        st.session_state["user_session_token"] = query_params["session_id"]
    else:
        new_token = f"user_{uuid.uuid4().hex[:12]}"
        st.session_state["user_session_token"] = new_token
        st.query_params["session_id"] = new_token
elif "session_id" not in st.query_params:
    st.query_params["session_id"] = st.session_state["user_session_token"]

USER_SESSION_TOKEN = st.session_state["user_session_token"]


# ============================================================
# 3. PREMIUM MINIMALIST UI & CSS
# ============================================================

PREMIUM_CSS = """
<style>

:root {
    --bg: #0b0b0d;
    --surface: #111114;
    --surface-2: #161619;
    --surface-3: #1b1b20;
    --border: #28282e;
    --border-soft: #202024;
    --text: #f5f5f7;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent: #5b8cff;
    --accent-soft: rgba(91,140,255,.12);
    --success: #4ade80;
    --warning: #fbbf24;
    --danger: #f87171;
}

html, body, [class*="css"] {
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "SF Pro Text",
        "Segoe UI",
        Roboto,
        Helvetica,
        Arial,
        sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 50% -20%,
            rgba(91,140,255,.07),
            transparent 35%
        ),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding-top: 2.5rem;
    padding-bottom: 5rem;
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    font-weight: 500 !important;
    letter-spacing: -0.025em;
}

h1 {
    font-size: 2.6rem !important;
}

h2 {
    font-size: 1.8rem !important;
}

h3 {
    font-size: 1.25rem !important;
}

p, label, span {
    color: var(--text-secondary);
}

[data-testid="stSidebar"] {
    background: #0e0e10;
    border-right: 1px solid var(--border-soft);
}

[data-testid="stSidebar"] * {
    color: var(--text-secondary);
}

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    transition: all 0.2s ease;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
    background: var(--surface-3) !important;
}

/* High-End Clean Buttons */
.stButton button {
    background: rgba(255, 255, 255, 0.03);
    color: var(--text);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    font-weight: 500;
    letter-spacing: 0.3px;
    min-height: 42px;
    transition: all 0.2s ease;
}

.stButton button:hover {
    border-color: rgba(255, 255, 255, 0.25);
    background: rgba(255, 255, 255, 0.06);
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    transform: translateY(-1px);
}

button[data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: #fff !important;
    box-shadow: 0 2px 8px rgba(91,140,255,0.25) !important;
}

button[data-testid="baseButton-primary"]:hover {
    background: #6b9aff !important;
    border-color: #6b9aff !important;
    box-shadow: 0 4px 14px rgba(91,140,255,0.4) !important;
    filter: brightness(1.05);
}

/* High-End Clean Page Selector (Radio) */
div[data-testid="stRadio"] {
    background: transparent;
}
div[data-testid="stRadio"] > div {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}
div[data-testid="stRadio"] label {
    background: rgba(255, 255, 255, 0.02);
    padding: 10px 20px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stRadio"] label:hover {
    border-color: rgba(255, 255, 255, 0.2) !important;
    background: rgba(255, 255, 255, 0.05);
    transform: translateY(-1px);
}
div[data-testid="stRadio"] label[data-checked="true"] {
    background: var(--text) !important;
    border-color: var(--text) !important;
    box-shadow: 0 4px 12px rgba(255,255,255,0.1);
}
div[data-testid="stRadio"] label[data-checked="true"] p {
    color: var(--bg) !important;
    font-weight: 600;
}

.stSlider [data-baseweb="slider"] {
    color: var(--accent) !important;
}

.stAlert {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}

.premium-card {
    background: rgba(255,255,255,.015);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 18px;
    transition: border-color 0.2s ease;
}

.premium-card:hover {
    border-color: rgba(255,255,255,.08);
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 20px;
    min-height: 120px;
}

.metric-label {
    color: var(--text-muted);
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-bottom: 10px;
}

.metric-value {
    color: var(--text);
    font-size: 1.8rem;
    font-weight: 500;
}

.metric-sub {
    color: var(--text-muted);
    font-size: .82rem;
    margin-top: 5px;
}

.section-heading {
    margin-top: 28px;
    margin-bottom: 16px;
    padding-bottom: 11px;
    border-bottom: 1px solid var(--border-soft);
    color: var(--text);
    font-size: 1.15rem;
    font-weight: 500;
}

.eyebrow {
    color: var(--accent);
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .13em;
    font-weight: 600;
    margin-bottom: 8px;
}

.badge {
    display: inline-block;
    padding: 5px 9px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text-secondary);
    font-size: .76rem;
}

.badge-success {
    color: var(--success);
    border-color: rgba(74,222,128,.25);
    background: rgba(74,222,128,.06);
}

.badge-warning {
    color: var(--warning);
    border-color: rgba(251,191,36,.25);
    background: rgba(251,191,36,.06);
}

.badge-danger {
    color: var(--danger);
    border-color: rgba(248,113,113,.25);
    background: rgba(248,113,113,.06);
}

.observation-row {
    padding: 12px 14px;
    margin-bottom: 8px;
    background: #101013;
    border: 1px solid var(--border-soft);
    border-radius: 9px;
}

.hero {
    padding: 18px 0 30px 0;
}

.hero-title {
    color: white;
    font-size: 2.7rem;
    font-weight: 500;
    letter-spacing: -.04em;
}

.hero-subtitle {
    color: var(--text-muted);
    font-size: 1rem;
    max-width: 850px;
    line-height: 1.65;
}

.small-note {
    color: var(--text-muted);
    font-size: .78rem;
    line-height: 1.5;
}

hr {
    border-color: var(--border-soft) !important;
}

/* Ensure Floating AI Voice Iframe Container Stays Fixed on Screen Always */
iframe[title="streamlit_components.v1.html"],
iframe[srcdoc*="voice-fab"] {
    position: fixed !important;
    bottom: 24px !important;
    right: 24px !important;
    width: 360px !important;
    height: 310px !important;
    z-index: 999999 !important;
    border: none !important;
    background: transparent !important;
}

</style>
"""

st.markdown(textwrap.dedent(PREMIUM_CSS), unsafe_allow_html=True)


# ============================================================
# 4. DATABASE & MIGRATIONS
# ============================================================

Base = declarative_base()

@st.cache_resource
def get_db_engine():
    engine_kwargs = {}
    if DATABASE_URL.startswith("sqlite"):
        engine_kwargs["connect_args"] = {
            "check_same_thread": False,
            "timeout": 15
        }
    
    eng = create_engine(DATABASE_URL, **engine_kwargs)
    return eng

@st.cache_resource
def get_session_factory(_engine):
    return sessionmaker(bind=_engine, autoflush=False, autocommit=False)

engine = get_db_engine()

class Event(Base):
    __tablename__ = "events"

    id = Column(String, primary_key=True)
    session_token = Column(String, nullable=False, default="global")
    name = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    objective = Column(String, nullable=False)
    context = Column(Text, nullable=True)
    environment = Column(String, nullable=True)
    sensory_environment = Column(String, nullable=True)
    acoustic_environment = Column(String, nullable=True)
    target_audience = Column(String, nullable=True)

    interactions = relationship(
        "Interaction",
        back_populates="event",
        cascade="all, delete-orphan"
    )


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(String, primary_key=True)
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    participant_code = Column(String, nullable=False)

    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=True)

    phase = Column(
        String,
        default="Approach",
        nullable=False
    )

    stated_preference = Column(Text, nullable=True)

    event = relationship(
        "Event",
        back_populates="interactions"
    )

    observations = relationship(
        "Observation",
        back_populates="interaction",
        cascade="all, delete-orphan"
    )

    surveys = relationship(
        "Survey",
        back_populates="interaction",
        cascade="all, delete-orphan"
    )


class Observation(Base):
    __tablename__ = "observations"

    id = Column(String, primary_key=True)
    interaction_id = Column(
        String,
        ForeignKey("interactions.id"),
        nullable=False
    )

    timestamp = Column(DateTime, nullable=False)
    category = Column(String, nullable=False)
    detail = Column(Text, nullable=False)

    evidence_level = Column(
        String,
        nullable=False,
        default="OBSERVED"
    )

    interaction = relationship(
        "Interaction",
        back_populates="observations"
    )


class Survey(Base):
    __tablename__ = "surveys"

    id = Column(String, primary_key=True)
    interaction_id = Column(
        String,
        ForeignKey("interactions.id"),
        nullable=False
    )

    timing = Column(String, nullable=False)

    curiosity = Column(Float, nullable=True)
    understanding = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)

    recall_text = Column(Text, nullable=True)

    follow_through = Column(Boolean, nullable=True)

    interaction = relationship(
        "Interaction",
        back_populates="surveys"
    )


class RapidStateLog(Base):
    __tablename__ = "rapid_state_logs"

    id = Column(String, primary_key=True)
    session_token = Column(String, nullable=False, default="global")
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    participant_code = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    baseline_level = Column(String, nullable=False)
    current_state = Column(String, nullable=False)

    event = relationship("Event")


Base.metadata.create_all(bind=engine)

# Migration helpers for database columns
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE events ADD COLUMN session_token VARCHAR DEFAULT 'global'"))
        conn.commit()
except Exception:
    pass

try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE rapid_state_logs ADD COLUMN session_token VARCHAR DEFAULT 'global'"))
        conn.commit()
except Exception:
    pass

SessionLocal = get_session_factory(engine)

def db_session():
    return SessionLocal()


# ============================================================
# 5. GEMINI SCHEMAS
# ============================================================

ConfidenceLevel = Literal["Low", "Moderate", "High"]


class CognitiveEstimate(BaseModel):
    bandwidth_pct: int = Field(ge=0, le=100)
    focus_pct: int = Field(ge=0, le=100)
    sensory_load_pct: int = Field(ge=0, le=100)

    rationale: str


class OutreachRecommendation(BaseModel):
    recommended_action: str
    rationale: str

    confidence: ConfidenceLevel

    evidence: List[str] = Field(
        min_length=1,
        max_length=6
    )

    alternative_explanation: str

    next_observation: str

    cognitive_estimate: CognitiveEstimate


class PredictedPathway(BaseModel):
    pathway: str
    mechanism: str
    expected_signal: str
    uncertainty: str


class ForwardModel(BaseModel):
    engagement_state: str

    predicted_pathways: List[PredictedPathway] = Field(
        min_length=3,
        max_length=3
    )

    recommended_outreach_design: List[str] = Field(
        min_length=3,
        max_length=5
    )

    likely_friction_points: List[str] = Field(
        min_length=1,
        max_length=5
    )

    measurement_opportunities: List[str] = Field(
        min_length=2,
        max_length=5
    )


class CounterfactualModel(BaseModel):
    changed_variable: str
    expected_difference: str

    before_state: str
    after_state: str

    predicted_effects: List[str] = Field(
        min_length=3,
        max_length=5
    )

    uncertainty: str


class ThemeModel(BaseModel):
    theme: str
    description: str

    evidence_strength: ConfidenceLevel

    evidence_quotes: List[str] = Field(
        min_length=1,
        max_length=4
    )


class ImpactInterpretation(BaseModel):
    overall_interpretation: str

    strongest_signal: str

    weakest_signal: str

    plausible_mechanisms: List[str] = Field(
        min_length=2,
        max_length=5
    )

    alternative_explanations: List[str] = Field(
        min_length=2,
        max_length=5
    )

    recommended_next_test: str


class OutcomePrediction(BaseModel):
    focus_pct: int = Field(ge=0, le=100, description="Predicted average focus state level (0-100%)")
    stress_reduction_pct: int = Field(ge=0, le=100, description="Predicted stress reduction index (0-100%)")
    cognitive_load_pct: int = Field(ge=0, le=100, description="Predicted mental cognitive load percentage (0-100%)")
    attention_retention_pct: int = Field(ge=0, le=100, description="Predicted visual & auditory retention percentage (0-100%)")
    predicted_curiosity_shift: str
    predicted_understanding_shift: str
    predicted_engagement_rate: str
    overall_outcome_narrative: str
    risk_factors: List[str] = Field(min_length=1, max_length=5)
    success_amplifiers: List[str] = Field(min_length=1, max_length=5)


# ============================================================
# 6. GEMINI CLIENT
# ============================================================

def get_api_key() -> str:
    key = ""

    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    if not key:
        key = os.getenv("GEMINI_API_KEY", "")

    return key


@st.cache_resource
def create_gemini_client(api_key: str):
    if not api_key:
        return None

    try:
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def run_gemini(
    client,
    model_name: str,
    prompt: str,
    schema,
    system_instruction: str,
    temperature: float = 0.2
):
    if client is None:
        raise RuntimeError(
            "Gemini client is unavailable. "
            "Configure GEMINI_API_KEY."
        )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        )
    )

    if getattr(response, "parsed", None) is not None:
        return response.parsed

    text = getattr(response, "text", None)

    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    return schema.model_validate_json(text)


# ============================================================
# 7. SYSTEM INSTRUCTIONS
# ============================================================

AI_SYSTEM = """
You are the reasoning layer of a public-facing science outreach
intelligence system.

Your task is to help facilitators create better human experiences.

STRICT EPISTEMIC RULES:

1. Treat personality, sensory characteristics, motivations, and
   psychological states as hypotheses unless directly stated or
   directly observable.

2. Never diagnose autism, ADHD, anxiety, depression, personality
   disorders, or any other clinical condition.

3. Never infer protected characteristics.

4. Never claim that a behavioral observation proves an internal
   psychological state.

5. Distinguish:
   STATED = participant explicitly said it.
   OBSERVED = facilitator directly observed it.
   INFERRED = reasonable interpretation.
   HYPOTHESIS = speculative explanation.

6. Participant preferences have priority over inference.

7. Prefer low-pressure, reversible, minimally disruptive
   interventions.

8. "No intervention" is always a valid option.

9. Do not manipulate participants.

10. Do not optimize for compliance or persuasion at the expense
    of autonomy.

11. When uncertainty is high, say so.

12. The system estimates possibilities. It does not measure
    cognition directly.

13. Never represent AI-generated percentages as physiological,
    neurological, or clinical measurements.

14. Recommendations should improve:
       clarity,
       accessibility,
       curiosity,
       scientific understanding,
       autonomy,
       comfort,
       meaningful engagement.

15. Avoid deterministic language such as:
       "will do",
       "is definitely",
       "this proves".

Prefer:
       "may",
       "could",
       "is consistent with",
       "one plausible explanation".
"""


# ============================================================
# 8. SESSION STATE INIT & RESTORE
# ============================================================

DEFAULT_STATE = {
    "active_event_id": None,
    "active_interaction_id": None,
    "last_recommendation": None,
    "last_forward_model": None,
    "last_counterfactual": None,
    "last_impact_interpretation": None,
    "last_prediction": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

db = db_session()

# Restore user active session workspace context automatically from database on refresh
if st.session_state.active_event_id is None:
    latest_event = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .first()
    )
    if latest_event:
        st.session_state.active_event_id = latest_event.id

if st.session_state.active_event_id and st.session_state.active_interaction_id is None:
    latest_interaction = (
        db.query(Interaction)
        .filter(Interaction.event_id == st.session_state.active_event_id)
        .order_by(Interaction.started_at.desc())
        .first()
    )
    if latest_interaction:
        st.session_state.active_interaction_id = latest_interaction.id


# ============================================================
# 9. HELPER FUNCTIONS
# ============================================================

def render_html(html_str: str):
    safe_html = "\n".join([line.lstrip() for line in html_str.split("\n")])
    st.markdown(safe_html, unsafe_allow_html=True)


def utc_now():
    return datetime.now(timezone.utc)


def clean_text(value: Any) -> str:
    return html.escape(str(value or ""))


def calculate_change(
    baseline: Optional[float],
    post: Optional[float]
):
    if baseline is None or post is None:
        return None

    return round(post - baseline, 2)


def mean_or_none(series):
    if series is None:
        return None

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if series.empty:
        return None

    return float(series.mean())


def create_event(
    db,
    name,
    objective,
    context,
    environment,
    sensory_environment,
    acoustic_environment,
    target_audience
):
    event = Event(
        id=str(uuid.uuid4()),
        session_token=USER_SESSION_TOKEN,
        name=name.strip(),
        date=utc_now(),
        objective=objective,
        context=context.strip(),
        environment=environment.strip(),
        sensory_environment=sensory_environment.strip(),
        acoustic_environment=acoustic_environment.strip(),
        target_audience=target_audience,
    )

    db.add(event)
    db.commit()

    return event


def create_interaction(
    db,
    event_id
):
    interaction = Interaction(
        id=str(uuid.uuid4()),
        event_id=event_id,
        participant_code=f"P-{uuid.uuid4().hex[:8].upper()}",
        started_at=utc_now(),
        phase="Approach",
    )

    db.add(interaction)
    db.commit()

    return interaction


def log_observation(
    db,
    interaction_id,
    category,
    detail,
    evidence_level="OBSERVED"
):
    observation = Observation(
        id=str(uuid.uuid4()),
        interaction_id=interaction_id,
        timestamp=utc_now(),
        category=category,
        detail=detail,
        evidence_level=evidence_level,
    )

    db.add(observation)
    db.commit()

    return observation


def get_recent_observations(
    db,
    interaction_id,
    limit=12
):
    return (
        db.query(Observation)
        .filter(
            Observation.interaction_id == interaction_id
        )
        .order_by(
            Observation.timestamp.desc()
        )
        .limit(limit)
        .all()
    )


def get_event_interactions(db, event_id):
    return (
        db.query(Interaction)
        .filter(
            Interaction.event_id == event_id
        )
        .all()
    )


# ============================================================
# 10. DETERMINISTIC IMPACT ENGINE
# ============================================================

def calculate_event_impact(db, event_id):

    interactions = get_event_interactions(
        db,
        event_id
    )

    if not interactions:
        return None

    interaction_ids = [
        i.id for i in interactions
    ]

    surveys = (
        db.query(Survey)
        .filter(
            Survey.interaction_id.in_(interaction_ids)
        )
        .all()
    )

    if not surveys:
        return {
            "participants": len(interactions),
            "surveyed": 0,
            "baseline_curiosity": None,
            "post_curiosity": None,
            "curiosity_change": None,
            "baseline_understanding": None,
            "post_understanding": None,
            "understanding_change": None,
            "baseline_confidence": None,
            "post_confidence": None,
            "confidence_change": None,
            "follow_through_rate": None,
            "recall_rate": None,
        }

    rows = []

    for s in surveys:
        rows.append({
            "interaction_id": s.interaction_id,
            "timing": s.timing,
            "curiosity": s.curiosity,
            "understanding": s.understanding,
            "confidence": s.confidence,
            "recall_text": s.recall_text,
            "follow_through": s.follow_through,
        })

    df = pd.DataFrame(rows)

    baseline = (
        df[df["timing"] == "BASELINE"]
        .groupby("interaction_id")
        .first()
    )

    immediate = (
        df[df["timing"] == "IMMEDIATE"]
        .groupby("interaction_id")
        .first()
    )

    paired = baseline.join(
        immediate,
        lsuffix="_baseline",
        rsuffix="_post",
        how="inner"
    )

    curiosity_change = None
    understanding_change = None
    confidence_change = None

    if not paired.empty:

        curiosity_delta = (
            paired["curiosity_post"]
            - paired["curiosity_baseline"]
        )

        understanding_delta = (
            paired["understanding_post"]
            - paired["understanding_baseline"]
        )

        confidence_delta = (
            paired["confidence_post"]
            - paired["confidence_baseline"]
        )

        curiosity_change = mean_or_none(
            curiosity_delta
        )

        understanding_change = mean_or_none(
            understanding_delta
        )

        confidence_change = mean_or_none(
            confidence_delta
        )

    delayed = df[
        df["timing"].isin(
            ["DELAYED_24H", "DELAYED_7D"]
        )
    ]

    follow_through_rate = None

    if not delayed.empty:
        valid = delayed[
            delayed["follow_through"].notna()
        ]

        if not valid.empty:
            follow_through_rate = (
                valid["follow_through"]
                .astype(bool)
                .mean()
                * 100
            )

    recall = df[
        (df["timing"] == "IMMEDIATE")
        & (
            df["recall_text"]
            .fillna("")
            .str.strip()
            .ne("")
        )
    ]

    recall_rate = None

    if len(immediate) > 0:
        recall_rate = (
            len(recall)
            / len(immediate)
            * 100
        )

    return {
        "participants": len(interactions),
        "surveyed": len(
            df["interaction_id"].unique()
        ),
        "baseline_curiosity": mean_or_none(
            baseline.get("curiosity")
        ),
        "post_curiosity": mean_or_none(
            immediate.get("curiosity")
        ),
        "curiosity_change": curiosity_change,

        "baseline_understanding": mean_or_none(
            baseline.get("understanding")
        ),
        "post_understanding": mean_or_none(
            immediate.get("understanding")
        ),
        "understanding_change": understanding_change,

        "baseline_confidence": mean_or_none(
            baseline.get("confidence")
        ),
        "post_confidence": mean_or_none(
            immediate.get("confidence")
        ),
        "confidence_change": confidence_change,

        "follow_through_rate": follow_through_rate,
        "recall_rate": recall_rate,
    }


# ============================================================
# 11. FORWARD MODEL
# ============================================================

def generate_forward_model(
    client,
    model_name,
    event,
    design_data
):

    prompt = f"""
OUTREACH EVENT

Objective:
{event.objective}

Target audience:
{event.target_audience}

Environment:
{event.environment}

Sensory environment:
{event.sensory_environment}

Acoustic environment:
{event.acoustic_environment}

Context:
{event.context}

ADDITIONAL DESIGN PARAMETERS

{design_data}

Produce a forward engagement model.

The model should NOT claim to know what individual participants
will do.

Instead identify three meaningfully different plausible
engagement pathways.

Also identify:
- useful outreach design decisions
- likely friction points
- opportunities for real-world measurement

The purpose is to help a facilitator design a better experience,
not manipulate people.
"""

    return run_gemini(
        client=client,
        model_name=model_name,
        prompt=prompt,
        schema=ForwardModel,
        system_instruction=AI_SYSTEM,
        temperature=0.25,
    )


# ============================================================
# 12. LIVE ADAPTATION MODEL
# ============================================================

def generate_live_recommendation(
    client,
    model_name,
    event,
    interaction,
    observations
):

    observation_text = "\n".join(
        [
            (
                f"[{o.evidence_level}] "
                f"{o.category}: {o.detail}"
            )
            for o in observations
        ]
    )

    prompt = f"""
EVENT:
{event.name}

OBJECTIVE:
{event.objective}

ENVIRONMENT:
{event.environment}

ACOUSTIC ENVIRONMENT:
{event.acoustic_environment}

PARTICIPANT STATED PREFERENCE:
{interaction.stated_preference or "None recorded"}

CURRENT PHASE:
{interaction.phase}

RECENT OBSERVATIONS:
{observation_text or "No observations recorded."}

Determine the most appropriate next outreach action.

The recommendation must be:
- minimally disruptive
- reversible
- autonomy-preserving
- useful for science communication

If the participant appears engaged, "No intervention" is allowed.

Do not diagnose or infer a clinical condition.
Do not treat an observation as proof of an internal state.
"""

    return run_gemini(
        client=client,
        model_name=model_name,
        prompt=prompt,
        schema=OutreachRecommendation,
        system_instruction=AI_SYSTEM,
        temperature=0.2,
    )


# ============================================================
# 13. COUNTERFACTUAL MODEL
# ============================================================

def generate_counterfactual(
    client,
    model_name,
    event,
    design,
    variable_change
):

    prompt = f"""
EVENT:
{event.name}

OBJECTIVE:
{event.objective}

ENVIRONMENT:
{event.environment}

ACOUSTIC ENVIRONMENT:
{event.acoustic_environment}

CURRENT DESIGN:
{design}

COUNTERFACTUAL CHANGE:
{variable_change}

Keep all other major factors conceptually constant.

Explain what could plausibly change because of the
specified modification.

Do not pretend this is experimental evidence.
This is a theoretical counterfactual.
"""

    return run_gemini(
        client=client,
        model_name=model_name,
        prompt=prompt,
        schema=CounterfactualModel,
        system_instruction=AI_SYSTEM,
        temperature=0.3,
    )


# ============================================================
# 14. MEMORY/THEME MODEL
# ============================================================

def generate_theme(
    client,
    model_name,
    text
):

    prompt = f"""
Participant recall response:

{text}

Identify the strongest memory/meaning anchor in the response.

Do not infer personality or diagnosis.

Use only information contained in the response.
"""

    return run_gemini(
        client=client,
        model_name=model_name,
        prompt=prompt,
        schema=ThemeModel,
        system_instruction=AI_SYSTEM,
        temperature=0.1,
    )


# ============================================================
# 15. IMPACT INTERPRETATION & PREDICTION
# ============================================================

def generate_impact_interpretation(
    client,
    model_name,
    metrics
):

    prompt = f"""
Here are deterministic event-level measurements calculated
from participant data:

{metrics}

Interpret these measurements cautiously.

Do NOT claim causation.

Discuss:
- strongest signal
- weakest signal
- plausible mechanisms
- alternative explanations
- one useful next experiment/test

Do not convert the metrics into psychological diagnoses.
"""

    return run_gemini(
        client=client,
        model_name=model_name,
        prompt=prompt,
        schema=ImpactInterpretation,
        system_instruction=AI_SYSTEM,
        temperature=0.2,
    )


def generate_outcome_prediction(
    client,
    model_name,
    event,
    crowd_info,
    situation_info,
    questionnaire_context=""
):
    prompt = f"""
EVENT: {event.name}
OBJECTIVE: {event.objective}
TARGET AUDIENCE: {event.target_audience}
ENVIRONMENT: {event.environment}
SENSORY ENVIRONMENT: {event.sensory_environment}

CROWD DETAILS: {crowd_info}
SITUATIONAL CONTEXT: {situation_info}
ENVIRONMENTAL QUESTIONNAIRE PARAMETERS: {questionnaire_context}

Based on these parameters, predict scientifically grounded estimated effects on participants.
Provide precise, realistic scientific predictions for:
1. focus_pct (estimated average focus percentage, 0-100%)
2. stress_reduction_pct (estimated stress reduction index, 0-100%)
3. cognitive_load_pct (estimated mental cognitive load level, 0-100%)
4. attention_retention_pct (estimated attention retention index, 0-100%)
5. predicted_curiosity_shift (e.g. +2.4 on 10-pt scale)
6. predicted_understanding_shift (e.g. +28% conceptual gain)
7. predicted_engagement_rate (e.g. 82% sustained participation)
8. overall_outcome_narrative (detailed scientific narrative)
9. risk_factors & success_amplifiers

Be realistic and ground estimations in environmental friction and crowd dynamics.
"""
    return run_gemini(
        client=client,
        model_name=model_name,
        prompt=prompt,
        schema=OutcomePrediction,
        system_instruction=AI_SYSTEM,
        temperature=0.3,
    )


# ============================================================
# 16. HEADER & NAVIGATION
# ============================================================

render_html("""
<div class="hero">
    <div class="eyebrow">Ninolades Research Platform</div>
    <div class="hero-title">
        Outreach Intelligence Lab
    </div>
    <div class="hero-subtitle">
        A human-centered intelligence system for designing,
        adapting, and evaluating science outreach experiences.
        It combines structured observation, generative reasoning,
        counterfactual exploration, and real-world impact evidence.
    </div>
</div>
""")

st.markdown("---")

header_col1, header_col2, header_col3 = st.columns([1.5, 1, 3])

with header_col1:
    model_options = {
        "Gemini 3.6 Flash": MODEL_FLASH,
        "Gemini 3.1 Pro": MODEL_PRO,
        "Gemini 3.5 Flash-Lite": MODEL_LITE,
    }
    
    if "mem_model_label" not in st.session_state:
        st.session_state["mem_model_label"] = list(model_options.keys())[0]

    selected_model_label = st.selectbox(
        "Reasoning engine",
        list(model_options.keys()),
        key="mem_model_label",
        help="Choose the Gemini model used for generative analysis.",
        label_visibility="collapsed"
    )
    
    selected_model = model_options[selected_model_label]

    render_html("""
    <div class="small-note" style="margin-top:8px;">
        Flash is default for live. Pro for deep analysis. Flash-Lite for volume.
    </div>
    """)

with header_col2:
    if st.button("Clean Memory", use_container_width=True):
        # Explicitly clear only this user's persistent database entries
        user_events = db.query(Event).filter(Event.session_token == USER_SESSION_TOKEN).all()
        user_event_ids = [e.id for e in user_events]
        
        if user_event_ids:
            user_interactions = db.query(Interaction).filter(Interaction.event_id.in_(user_event_ids)).all()
            user_int_ids = [i.id for i in user_interactions]
            
            if user_int_ids:
                db.query(Observation).filter(Observation.interaction_id.in_(user_int_ids)).delete(synchronize_session=False)
                db.query(Survey).filter(Survey.interaction_id.in_(user_int_ids)).delete(synchronize_session=False)
                db.query(Interaction).filter(Interaction.event_id.in_(user_event_ids)).delete(synchronize_session=False)
            
            db.query(Event).filter(Event.session_token == USER_SESSION_TOKEN).delete(synchronize_session=False)

        db.query(RapidStateLog).filter(RapidStateLog.session_token == USER_SESSION_TOKEN).delete(synchronize_session=False)
        db.commit()

        st.session_state.clear()
        
        # Generate clean new session token
        fresh_token = f"user_{uuid.uuid4().hex[:12]}"
        st.query_params["session_id"] = fresh_token
        st.session_state["user_session_token"] = fresh_token
        st.rerun()

    render_html(f"""
    <div class="small-note" style="margin-top:8px; text-align:center;">
        Version {APP_VERSION} | Session: {USER_SESSION_TOKEN[:8]}
    </div>
    """)

with header_col3:
    page_opts = [
        "Experience Designer",
        "Live Copilot",
        "Outcome Predictor",
        "Scientific Reactions",
        "Impact Observatory",
        "Counterfactual Lab",
        "Methodology",
    ]
    
    if "mem_page" not in st.session_state:
        st.session_state["mem_page"] = page_opts[0]
        
    page = st.radio(
        "Workspace",
        page_opts,
        key="mem_page",
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("---")

api_key = get_api_key()
client = create_gemini_client(api_key)

if client is None:
    st.error(
        "Gemini is not configured. Add GEMINI_API_KEY "
        "to Streamlit secrets or the environment."
    )


# ============================================================
# 17. EXPERIENCE DESIGNER
# ============================================================

if page == "Experience Designer":

    render_html('<div class="section-heading">Design an outreach experience</div>')

    left, right = st.columns(2)

    with left:
        event_name = st.text_input(
            "Experience name",
            key="mem_event_name",
            placeholder="e.g. Science Under the Stars or Local Ecology Walk"
        )

        obj_opts = [
            "Curiosity",
            "Scientific understanding",
            "Awe and wonder",
            "Memory and retention",
            "Question generation",
            "Independent follow-through",
            "General engagement",
        ]
        objective = st.selectbox(
            "Primary objective",
            obj_opts,
            key="mem_objective"
        )

        aud_opts = [
            "General public",
            "Students",
            "Families",
            "Educators",
            "Astronomy enthusiasts",
            "Eco-tourists",
            "Mixed audience",
        ]
        target_audience = st.selectbox(
            "Audience",
            aud_opts,
            key="mem_audience"
        )

    with right:
        environment = st.text_input(
            "Physical environment",
            key="mem_environment",
            placeholder="Dark-sky lawn, school courtyard, museum..."
        )

        acoustic_environment = st.text_input(
            "Acoustic / musical environment",
            key="mem_acoustic",
            placeholder="Silent, ambient sound, live acoustic..."
        )

        sensory_environment = st.text_input(
            "Relevant environmental conditions",
            key="mem_sensory",
            placeholder="Lighting, crowd density, temperature, noise..."
        )

    context = st.text_area(
        "Experience description",
        key="mem_context",
        placeholder=(
            "Describe what participants encounter, "
            "what the facilitator does, and the scientific content."
        ),
        height=130
    )

    render_html('<div class="section-heading">Optional design variables</div>')

    design_col1, design_col2, design_col3 = st.columns(3)

    with design_col1:
        pac_opts = ["Slow and contemplative", "Moderate", "Fast and energetic", "Variable"]
        if "mem_pacing" not in st.session_state:
            st.session_state["mem_pacing"] = pac_opts[0]
        pacing = st.selectbox("Pacing", pac_opts, key="mem_pacing")

    with design_col2:
        style_opts = ["Open observation", "Facilitator-led", "Question-led", "Hands-on", "Story-driven", "Mixed"]
        if "mem_interaction_style" not in st.session_state:
            st.session_state["mem_interaction_style"] = style_opts[0]
        interaction_style = st.selectbox("Interaction style", style_opts, key="mem_interaction_style")

    with design_col3:
        aut_opts = ["High", "Moderate", "Low"]
        if "mem_optional_choice" not in st.session_state:
            st.session_state["mem_optional_choice"] = aut_opts[0]
        optional_choice = st.selectbox("Participant autonomy", aut_opts, key="mem_optional_choice")

    design_data = f"""
Pacing: {pacing}
Interaction style: {interaction_style}
Participant autonomy: {optional_choice}
"""

    if st.button(
        "Create Experience Model",
        type="primary",
        use_container_width=True
    ):

        if not event_name.strip():
            st.error("Enter an experience name.")
        elif not context.strip():
            st.error("Describe the experience.")
        elif client is None:
            st.error("Gemini is unavailable.")
        else:

            try:

                event = create_event(
                    db=db,
                    name=event_name,
                    objective=objective,
                    context=context,
                    environment=environment,
                    sensory_environment=sensory_environment,
                    acoustic_environment=acoustic_environment,
                    target_audience=target_audience,
                )

                st.session_state.active_event_id = event.id

                with st.spinner(
                    "Building engagement model..."
                ):

                    model = generate_forward_model(
                        client,
                        selected_model,
                        event,
                        design_data
                    )

                st.session_state.last_forward_model = (
                    model.model_dump()
                )

                st.success(
                    "Experience initialized and model generated."
                )

            except Exception as exc:

                st.error(
                    f"Could not create the experience: {exc}"
                )

    if st.session_state.last_forward_model:

        model = st.session_state.last_forward_model

        render_html('<div class="section-heading">Engagement architecture</div>')

        render_html(f"""
        <div class="premium-card">
            <div class="eyebrow">Current model</div>
            <div style="color:var(--text); font-size:1.25rem; margin-bottom:10px;">
                {clean_text(model["engagement_state"])}
            </div>
            <div class="small-note">
                This is a generated hypothesis about possible
                engagement dynamics, not a measurement of participants.
            </div>
        </div>
        """)

        cols = st.columns(3)

        for idx, pathway in enumerate(
            model["predicted_pathways"]
        ):

            with cols[idx]:
                render_html(f"""
                <div class="premium-card" style="height:100%;">
                    <div class="eyebrow">
                        Pathway {idx + 1}
                    </div>
                    <h3 style="margin-top:0;">
                        {clean_text(pathway["pathway"])}
                    </h3>
                    <p>
                        {clean_text(pathway["mechanism"])}
                    </p>
                    <div class="small-note">
                        Expected signal:<br>
                        {clean_text(pathway["expected_signal"])}
                    </div>
                    <br>
                    <div class="small-note">
                        Uncertainty:<br>
                        {clean_text(pathway["uncertainty"])}
                    </div>
                </div>
                """)

        c1, c2 = st.columns(2)

        with c1:

            render_html('<div class="section-heading">Design opportunities</div>')

            for item in model[
                "recommended_outreach_design"
            ]:
                st.markdown(
                    f"- {item}"
                )

        with c2:

            render_html('<div class="section-heading">Potential friction</div>')

            for item in model[
                "likely_friction_points"
            ]:
                st.markdown(
                    f"- {item}"
                )

        render_html('<div class="section-heading">What should be measured?</div>')

        for item in model[
            "measurement_opportunities"
        ]:
            st.markdown(
                f"- {item}"
            )


# ============================================================
# 18. OUTCOME PREDICTOR
# ============================================================

elif page == "Outcome Predictor":

    render_html('<div class="section-heading">Predict Outreach Outcomes</div>')

    events = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:
        st.info("Create an experience in Experience Designer first.")
    else:
        event_map = {
            f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})": e
            for e in events
        }

        event_keys = list(event_map.keys())
        selected_name = st.selectbox(
            "Select Experience",
            event_keys,
            key="pred_event_select"
        )
        
        event = event_map[selected_name]

        render_html('<div class="section-heading">Contextual Environment & Crowd Questionnaire</div>')
        
        qc1, qc2 = st.columns(2)
        with qc1:
            baseline_stress_q = st.select_slider(
                "Estimated Baseline Audience Stress Level",
                options=["Very Low / Relaxed", "Moderate Stress", "High Stress / Overwhelmed"],
                value="Moderate Stress",
                key="q_stress"
            )
            noise_sensory_q = st.select_slider(
                "Ambient Distraction & Sensory Noise Level",
                options=["Quiet & Controlled", "Moderate Noise", "High Loudness / Busy Crowd"],
                value="Moderate Noise",
                key="q_noise"
            )
        with qc2:
            duration_q = st.selectbox(
                "Planned Session Duration",
                ["Short (< 15 mins)", "Standard (30-45 mins)", "Extended (60+ mins)"],
                key="q_duration"
            )
            interaction_density_q = st.selectbox(
                "Interactive Touchpoints Density",
                ["Low (Passive listening)", "Medium (Guided Q&A)", "High (Hands-on exploration)"],
                index=1,
                key="q_density"
            )

        c1, c2 = st.columns(2)
        with c1:
            crowd_info = st.text_area(
                "Crowd details & demographics",
                placeholder="e.g. 50 enthusiastic middle schoolers, mostly beginners, excited but easily distracted...",
                key="pred_crowd",
                height=110
            )
        with c2:
            situation_info = st.text_area(
                "Situational context",
                placeholder="e.g. Cloudy weather, noisy street nearby, late evening after a long day...",
                key="pred_situation",
                height=110
            )

        questionnaire_summary = f"Baseline Stress: {baseline_stress_q}, Noise level: {noise_sensory_q}, Duration: {duration_q}, Interaction density: {interaction_density_q}"

        if st.button("Predict Outcome & Metrics", type="primary", use_container_width=True):
            if client is None:
                st.error("Gemini is unavailable.")
            elif not crowd_info.strip() or not situation_info.strip():
                st.error("Please provide both crowd details and situational context.")
            else:
                try:
                    with st.spinner("Analyzing parameters and computing predicted outcomes..."):
                        pred = generate_outcome_prediction(
                            client,
                            selected_model,
                            event,
                            crowd_info,
                            situation_info,
                            questionnaire_summary
                        )
                        st.session_state.last_prediction = pred.model_dump()
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        if st.session_state.get("last_prediction"):
            p = st.session_state.last_prediction

            render_html('<div class="section-heading">Predicted Scientific Effects (Focus, Stress, Load & Retention)</div>')
            
            # Metric cards
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Predicted Focus State", f"{p.get('focus_pct', 75)}%")
            pc2.metric("Stress Reduction Index", f"{p.get('stress_reduction_pct', 60)}%")
            pc3.metric("Cognitive Load Level", f"{p.get('cognitive_load_pct', 45)}%")
            pc4.metric("Attention Retention", f"{p.get('attention_retention_pct', 80)}%")

            # Visual Representation Chart
            st.markdown("**Visual Effect Profile Comparison**")
            chart_data = pd.DataFrame({
                "Metric": ["Focus State", "Stress Reduction", "Cognitive Load", "Attention Retention"],
                "Percentage (%)": [
                    p.get("focus_pct", 75),
                    p.get("stress_reduction_pct", 60),
                    p.get("cognitive_load_pct", 45),
                    p.get("attention_retention_pct", 80)
                ]
            }).set_index("Metric")
            st.bar_chart(chart_data, color="#5b8cff")

            render_html('<div class="section-heading">Predicted Shifts & Outcomes</div>')
            m1, m2, m3 = st.columns(3)
            with m1:
                render_html(f"""
                <div class="metric-card">
                    <div class="metric-label">Curiosity Shift</div>
                    <div class="metric-value" style="font-size:1.4rem;">{clean_text(p['predicted_curiosity_shift'])}</div>
                </div>
                """)
            with m2:
                render_html(f"""
                <div class="metric-card">
                    <div class="metric-label">Understanding Shift</div>
                    <div class="metric-value" style="font-size:1.4rem;">{clean_text(p['predicted_understanding_shift'])}</div>
                </div>
                """)
            with m3:
                render_html(f"""
                <div class="metric-card">
                    <div class="metric-label">Engagement Rate</div>
                    <div class="metric-value" style="font-size:1.4rem;">{clean_text(p['predicted_engagement_rate'])}</div>
                </div>
                """)

            render_html('<div class="section-heading">Outcome Narrative</div>')
            render_html(f"""
            <div class="premium-card">
                <div style="color:var(--text); line-height:1.7;">
                    {clean_text(p['overall_outcome_narrative'])}
                </div>
            </div>
            """)

            r1, r2 = st.columns(2)
            with r1:
                render_html('<div class="section-heading">Risk Factors</div>')
                for r in p["risk_factors"]:
                    st.markdown(f"- {r}")
            with r2:
                render_html('<div class="section-heading">Success Amplifiers</div>')
                for s in p["success_amplifiers"]:
                    st.markdown(f"- {s}")


# ============================================================
# 19. LIVE COPILOT
# ============================================================

elif page == "Live Copilot":

    render_html('<div class="section-heading">Live outreach copilot</div>')

    events = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.info(
            "Create an experience in Experience Designer first."
        )

    else:

        event_map = {
            f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})": e
            for e in events
        }

        event_keys = list(event_map.keys())
        chosen_name = st.selectbox(
            "Active experience",
            event_keys,
            key="live_copilot_event_select"
        )

        event = event_map[chosen_name]

        st.markdown("---")
        render_html('<div class="section-heading">Rapid Participant State Logger</div>')
        render_html('<div class="small-note" style="margin-bottom:15px;">One-click logging for rapid visual analysis. Each click automatically registers as a new participant.</div>')

        rc1, rc2 = st.columns(2)
        baseline_opts = [
            "Calm / Receptive",
            "Neutral / Unengaged",
            "Low Energy / Fatigued",
            "Distracted / Scatterbrained",
            "Anxious / Stressed",
            "High Energy / Excited"
        ]
        state_opts = [
            "Awe / Wonder",
            "Deep Focus / Flow",
            "Curiosity / Inquisitive",
            "Epiphany / Sudden Understanding",
            "Cognitive Overload / Confusion",
            "Disengagement / Boredom",
            "Stress / Frustration",
            "Relaxation / Comfort"
        ]

        with rc1:
            rapid_baseline = st.radio("Baseline Level", baseline_opts, key="rapid_base")
        with rc2:
            rapid_state = st.radio("State of Mind / Reaction", state_opts, key="rapid_state")

        if st.button("Log as New Participant", type="primary", use_container_width=True):
            new_log = RapidStateLog(
                id=str(uuid.uuid4()),
                session_token=USER_SESSION_TOKEN,
                event_id=event.id,
                participant_code=f"RP-{uuid.uuid4().hex[:6].upper()}",
                timestamp=utc_now(),
                baseline_level=rapid_baseline,
                current_state=rapid_state
            )
            db.add(new_log)
            db.commit()
            st.toast("State logged successfully for new participant.")
            st.rerun()

        recent_rapid_logs = db.query(RapidStateLog).filter(
            RapidStateLog.event_id == event.id,
            RapidStateLog.session_token == USER_SESSION_TOKEN
        ).order_by(RapidStateLog.timestamp.desc()).limit(10).all()

        if recent_rapid_logs:
            st.markdown("**Recent Rapid Logs**")
            for rlog in recent_rapid_logs:
                c_time, c_part, c_base, c_state, c_del = st.columns([1.5, 1.5, 2.5, 2.5, 1])
                c_time.caption(rlog.timestamp.strftime("%H:%M:%S UTC"))
                c_part.caption(rlog.participant_code)
                c_base.caption(rlog.baseline_level)
                c_state.caption(rlog.current_state)
                if c_del.button("Delete", key=f"del_rlog_{rlog.id}", help="Delete this log"):
                    db.delete(rlog)
                    db.commit()
                    st.toast(f"Log {rlog.participant_code} deleted.")
                    st.rerun()

        st.markdown("---")

        if (
            st.session_state.active_event_id
            != event.id
        ):
            st.session_state.active_event_id = event.id
            st.session_state.active_interaction_id = None

        # Auto-recover unfinished interaction upon refresh so micro-interactions are not lost
        if not st.session_state.active_interaction_id:
            unfinished_interaction = db.query(Interaction).filter(
                Interaction.event_id == event.id,
                Interaction.ended_at == None
            ).order_by(Interaction.started_at.desc()).first()
            
            if unfinished_interaction:
                st.session_state.active_interaction_id = unfinished_interaction.id

        if not st.session_state.active_interaction_id:

            if st.button(
                "Start participant interaction",
                type="primary",
                use_container_width=True
            ):

                interaction = create_interaction(
                    db,
                    event.id
                )

                st.session_state.active_interaction_id = (
                    interaction.id
                )

                st.rerun()

        else:

            interaction = (
                db.query(Interaction)
                .filter(
                    Interaction.id
                    == st.session_state.active_interaction_id
                )
                .first()
            )

            if interaction is None:
                st.session_state.active_interaction_id = None
                st.rerun()

            render_html(f"""
            <div class="premium-card">
                <div class="eyebrow">
                    Active interaction
                </div>
                <div style="color:var(--text); font-size:1.2rem;">
                    {clean_text(interaction.participant_code)}
                </div>
                <div class="small-note">
                    Anonymous interaction code.
                </div>
            </div>
            """)

            phase_opts = [
                "Approach",
                "Introduction",
                "Waiting",
                "Direct observation",
                "Explanation",
                "Question/discussion",
                "Reflection",
                "Exit",
            ]
            
            if f"phase_{interaction.id}" not in st.session_state:
                st.session_state[f"phase_{interaction.id}"] = interaction.phase

            phase = st.selectbox(
                "Current phase",
                phase_opts,
                key=f"phase_{interaction.id}"
            )

            if phase != interaction.phase:
                interaction.phase = phase
                db.commit()

            if f"pref_{interaction.id}" not in st.session_state:
                st.session_state[f"pref_{interaction.id}"] = interaction.stated_preference or ""

            preference = st.text_input(
                "Participant-stated preference (Press Enter to save)",
                key=f"pref_{interaction.id}",
                placeholder=(
                    "Only record what the participant explicitly states."
                )
            )

            if preference != (interaction.stated_preference or ""):
                interaction.stated_preference = preference
                db.commit()

            render_html('<div class="section-heading">Quick Observations</div>')

            observation_buttons = [
                ("Attention", "Participant appears engaged/focused."),
                ("Attention", "Participant looks away/distracted."),
                ("Participation", "Participant asks a question."),
                ("Participation", "Participant gives a detailed response."),
                ("Participation", "Participant listens without responding."),
                ("Reflection", "Participant pauses to reflect."),
                ("Friction", "Participant has difficulty interacting."),
                ("Friction", "Environmental interruption occurs."),
            ]

            obs_cols = st.columns(4)
            for idx, (category, detail) in enumerate(observation_buttons):
                with obs_cols[idx % 4]:
                    if st.button(detail, key=f"obs_btn_{idx}", use_container_width=True):
                        log_observation(db, interaction.id, category, detail, "OBSERVED")
                        st.toast("Observation recorded.")

            with st.form(key=f"custom_obs_form_{interaction.id}", clear_on_submit=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    custom_obs = st.text_input(
                        "Custom observation",
                        placeholder="Describe only what was directly observed.",
                        label_visibility="collapsed"
                    )
                with c2:
                    submit_obs = st.form_submit_button("Log Observation", use_container_width=True)
                
                if submit_obs and custom_obs.strip():
                    log_observation(db, interaction.id, "Custom", custom_obs.strip(), "OBSERVED")
                    st.toast("Custom observation recorded.")
                    st.rerun()

            render_html('<div class="section-heading">Recent evidence</div>')

            observations = get_recent_observations(
                db,
                interaction.id
            )

            if observations:

                for observation in observations:

                    render_html(f"""
                    <div class="observation-row">
                        <span class="badge">
                            {clean_text(observation.evidence_level)}
                        </span>
                        <span style="margin-left:8px;color:var(--text);">
                            {clean_text(observation.detail)}
                        </span>
                    </div>
                    """)

            else:

                st.caption(
                    "No observations recorded yet."
                )

            render_html('<div class="section-heading">Adaptive guidance</div>')

            if st.button(
                "Generate next best outreach action",
                type="primary",
                use_container_width=True
            ):

                if client is None:

                    st.error(
                        "Gemini is unavailable."
                    )

                else:

                    observations = get_recent_observations(
                        db,
                        interaction.id
                    )

                    try:

                        with st.spinner(
                            "Reasoning over current evidence..."
                        ):

                            recommendation = (
                                generate_live_recommendation(
                                    client,
                                    selected_model,
                                    event,
                                    interaction,
                                    observations
                                )
                            )

                        st.session_state.last_recommendation = (
                            recommendation.model_dump()
                        )

                    except Exception as exc:

                        st.error(
                            f"Recommendation failed: {exc}"
                        )

            if st.session_state.last_recommendation:

                recommendation = (
                    st.session_state.last_recommendation
                )

                render_html(f"""
                <div class="premium-card"
                     style="
                     border-color: rgba(91,140,255,.35);
                     background: linear-gradient(145deg, rgba(91,140,255,.09), rgba(91,140,255,.025));
                     ">
                    <div class="eyebrow">
                        Suggested next move
                    </div>

                    <div style="color:var(--text); font-size:1.35rem; margin-bottom:14px;">
                        {clean_text(recommendation["recommended_action"])}
                    </div>

                    <div style="color:#d4d4d8; line-height:1.6;">
                        {clean_text(recommendation["rationale"])}
                    </div>
                </div>
                """)

                c1, c2 = st.columns(2)

                with c1:

                    render_html(f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            Confidence
                        </div>
                        <div class="metric-value">
                            {clean_text(recommendation["confidence"])}
                        </div>
                    </div>
                    """)

                with c2:

                    estimate = recommendation[
                        "cognitive_estimate"
                    ]

                    render_html(f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            Model-estimated focus
                        </div>
                        <div class="metric-value">
                            {estimate["focus_pct"]}%
                        </div>
                        <div class="metric-sub">
                            Hypothesis only; not a
                            physiological measurement.
                        </div>
                    </div>
                    """)

                render_html('<div class="section-heading">Evidence used</div>')

                for evidence in recommendation[
                    "evidence"
                ]:
                    st.markdown(
                        f"- {evidence}"
                    )

                render_html('<div class="section-heading">Alternative explanation</div>')

                st.write(
                    recommendation[
                        "alternative_explanation"
                    ]
                )

                render_html('<div class="section-heading">Next observation to watch</div>')

                st.write(
                    recommendation[
                        "next_observation"
                    ]
                )

            st.markdown("---")

            if st.button(
                "End interaction and start next participant",
                use_container_width=True
            ):

                interaction.ended_at = utc_now()
                db.commit()

                st.session_state.active_interaction_id = None
                st.session_state.last_recommendation = None

                st.rerun()


# ============================================================
# 20. SCIENTIFIC REACTIONS
# ============================================================

elif page == "Scientific Reactions":

    render_html('<div class="section-heading">Scientific Reaction Analysis</div>')
    render_html('<div class="small-note" style="margin-bottom:15px;">Visualizing accurate, logical, and practical scientific reactions (focus, stress, awe) logged during the experience.</div>')

    events = db.query(Event).filter(Event.session_token == USER_SESSION_TOKEN).order_by(Event.date.desc()).all()
    
    if not events:
        st.info("No experiences available.")
    else:
        event_map = {f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})": e for e in events}
        selected_name = st.selectbox("Select Experience", list(event_map.keys()), key="sci_reac_event")
        event = event_map[selected_name]

        logs = db.query(RapidStateLog).filter(
            RapidStateLog.event_id == event.id,
            RapidStateLog.session_token == USER_SESSION_TOKEN
        ).order_by(RapidStateLog.timestamp.asc()).all()

        if not logs:
            st.info("No rapid reaction data logged for this event yet. Use the Rapid State Logger in the Live Copilot page.")
        else:
            df_logs = pd.DataFrame([{
                "timestamp": l.timestamp,
                "participant": l.participant_code,
                "baseline": l.baseline_level,
                "reaction": l.current_state
            } for l in logs])
            
            df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])

            render_html(f"""
            <div class="metric-card">
                <div class="metric-label">Total Rapid Logs</div>
                <div class="metric-value">{len(logs)}</div>
            </div>
            """)

            c1, c2 = st.columns(2)
            with c1:
                render_html('<div class="section-heading">State of Mind / Reaction Distribution</div>')
                reaction_counts = df_logs['reaction'].value_counts()
                st.bar_chart(reaction_counts, color="#5b8cff")

            with c2:
                render_html('<div class="section-heading">Baseline Level Distribution</div>')
                baseline_counts = df_logs['baseline'].value_counts()
                st.bar_chart(baseline_counts, color="#a1a1aa")

            render_html('<div class="section-heading">Reaction Timeline</div>')
            df_logs['time_minute'] = df_logs['timestamp'].dt.floor('min')
            timeline_df = df_logs.groupby(['time_minute', 'reaction']).size().unstack(fill_value=0)
            st.line_chart(timeline_df)

            render_html('<div class="section-heading">Raw Log Data</div>')
            st.dataframe(df_logs, use_container_width=True)


# ============================================================
# 21. IMPACT OBSERVATORY
# ============================================================

elif page == "Impact Observatory":

    render_html('<div class="section-heading">Real-world impact observatory</div>')

    render_html("""
    <div class="premium-card">
        <div class="eyebrow">Why this exists</div>
        <div style="color:var(--text); line-height:1.7;">
            The system separates what the AI predicts from what
            actually happened. Impact is calculated from recorded
            participant outcomes rather than generated by Gemini.
        </div>
    </div>
    """)

    events = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.info(
            "No experiences have been created yet."
        )

    else:

        event_map = {
            f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})": e
            for e in events
        }

        event_keys = list(event_map.keys())
        selected_name = st.selectbox(
            "Experience",
            event_keys,
            key="obs_event_select"
        )

        event = event_map[selected_name]

        metrics = calculate_event_impact(
            db,
            event.id
        )

        if metrics is None:
            st.info(
                "No interaction data available."
            )

        else:

            cols = st.columns(4)

            cols[0].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Participants</div>
                    <div class="metric-value">{metrics["participants"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            cols[1].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Curiosity change</div>
                    <div class="metric-value">
                        {
                            "—"
                            if metrics["curiosity_change"] is None
                            else f'{metrics["curiosity_change"]:+.2f}'
                        }
                    </div>
                    <div class="metric-sub">Paired baseline → immediate</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            cols[2].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Understanding change</div>
                    <div class="metric-value">
                        {
                            "—"
                            if metrics["understanding_change"] is None
                            else f'{metrics["understanding_change"]:+.2f}'
                        }
                    </div>
                    <div class="metric-sub">Paired baseline → immediate</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            cols[3].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Follow-through</div>
                    <div class="metric-value">
                        {
                            "—"
                            if metrics["follow_through_rate"] is None
                            else f'{metrics["follow_through_rate"]:.1f}%'
                        }
                    </div>
                    <div class="metric-sub">Delayed self-report</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            render_html('<div class="section-heading">Outcome signals</div>')

            signal_data = {
                "Curiosity": metrics[
                    "curiosity_change"
                ],
                "Understanding": metrics[
                    "understanding_change"
                ],
                "Confidence": metrics[
                    "confidence_change"
                ],
            }

            signal_df = pd.DataFrame(
                [
                    {
                        "Signal": name,
                        "Change": value
                    }
                    for name, value
                    in signal_data.items()
                    if value is not None
                ]
            )

            if not signal_df.empty:

                st.dataframe(
                    signal_df,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "Paired baseline/post measurements "
                    "are needed to calculate change."
                )

            render_html('<div class="section-heading">Delayed indicators</div>')

            d1, d2 = st.columns(2)

            d1.metric(
                "Recall response rate",
                (
                    "—"
                    if metrics["recall_rate"] is None
                    else f'{metrics["recall_rate"]:.1f}%'
                )
            )

            d2.metric(
                "Participants with survey data",
                metrics["surveyed"]
            )

            render_html('<div class="section-heading">AI interpretation</div>')

            st.caption(
                "Gemini interprets the measurements below; "
                "it does not calculate them."
            )

            if st.button(
                "Interpret impact",
                type="primary",
                use_container_width=True
            ):

                if client is None:

                    st.error(
                        "Gemini is unavailable."
                    )

                else:

                    try:

                        with st.spinner(
                            "Interpreting outcome patterns..."
                        ):

                            interpretation = (
                                generate_impact_interpretation(
                                    client,
                                    selected_model,
                                    metrics
                                )
                            )

                        st.session_state.last_impact_interpretation = (
                            interpretation.model_dump()
                        )

                    except Exception as exc:

                        st.error(
                            f"Impact interpretation failed: {exc}"
                        )

            if st.session_state.last_impact_interpretation:

                interpretation = (
                    st.session_state.last_impact_interpretation
                )

                render_html(f"""
                <div class="premium-card">
                    <div class="eyebrow">
                        Interpretation
                    </div>
                    <div style="color:var(--text); line-height:1.7;">
                        {clean_text(interpretation["overall_interpretation"])}
                    </div>
                </div>
                """)

                c1, c2 = st.columns(2)

                with c1:

                    render_html('<div class="section-heading">Strongest signal</div>')

                    st.write(
                        interpretation[
                            "strongest_signal"
                        ]
                    )

                with c2:

                    render_html('<div class="section-heading">Weakest signal</div>')

                    st.write(
                        interpretation[
                            "weakest_signal"
                        ]
                    )

                render_html('<div class="section-heading">Plausible mechanisms</div>')

                for item in interpretation[
                    "plausible_mechanisms"
                ]:
                    st.markdown(
                        f"- {item}"
                    )

                render_html('<div class="section-heading">Alternative explanations</div>')

                for item in interpretation[
                    "alternative_explanations"
                ]:
                    st.markdown(
                        f"- {item}"
                    )

                render_html('<div class="section-heading">Recommended next test</div>')

                st.info(
                    interpretation[
                        "recommended_next_test"
                    ]
                )


# ============================================================
# 22. COUNTERFACTUAL LAB
# ============================================================

elif page == "Counterfactual Lab":

    render_html('<div class="section-heading">Counterfactual experiment lab</div>')

    events = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.info(
            "Create an experience first."
        )

    else:

        event_map = {
            f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})": e
            for e in events
        }

        event_keys = list(event_map.keys())
        selected_name = st.selectbox(
            "Experience",
            event_keys,
            key="cf_event_select"
        )
        
        event = event_map[selected_name]

        render_html("""
        <div class="premium-card">
            <div class="eyebrow">
                Counterfactual reasoning
            </div>
            <div class="small-note">
                Change one meaningful variable while keeping
                the conceptual baseline constant. This does
                not create experimental evidence; it helps
                generate testable outreach hypotheses.
            </div>
        </div>
        """)

        variable_change = st.text_area(
            "What would you change?",
            key="cf_variable_change",
            placeholder=(
                "What if the live music stopped during direct "
                "observation?"
            ),
            height=100
        )

        if "cf_design_description" not in st.session_state:
            st.session_state["cf_design_description"] = event.context or ""
            
        design_description = st.text_area(
            "Current design",
            key="cf_design_description",
            height=100
        )

        if st.button(
            "Run counterfactual",
            type="primary",
            use_container_width=True
        ):

            if not variable_change.strip():

                st.error(
                    "Describe the variable you want to change."
                )

            elif client is None:

                st.error(
                    "Gemini is unavailable."
                )

            else:

                try:

                    with st.spinner(
                        "Comparing hypothetical pathways..."
                    ):

                        cf = generate_counterfactual(
                            client,
                            selected_model,
                            event,
                            design_description,
                            variable_change
                        )

                    st.session_state.last_counterfactual = (
                        cf.model_dump()
                    )

                except Exception as exc:

                    st.error(
                        f"Counterfactual failed: {exc}"
                    )

        if st.session_state.last_counterfactual:

            cf = st.session_state.last_counterfactual

            render_html(f"""
            <div class="premium-card">
                <div class="eyebrow">
                    Changed variable
                </div>
                <h3 style="margin-top:0;">
                    {clean_text(cf["changed_variable"])}
                </h3>
                <p>
                    {clean_text(cf["expected_difference"])}
                </p>
            </div>
            """)

            c1, c2 = st.columns(2)

            with c1:

                render_html(f"""
                <div class="premium-card">
                    <div class="eyebrow">
                        Baseline
                    </div>
                    <div style="color:var(--text);">
                        {clean_text(cf["before_state"])}
                    </div>
                </div>
            """)

            with c2:

                render_html(f"""
                <div class="premium-card">
                    <div class="eyebrow">
                        Counterfactual
                    </div>
                    <div style="color:var(--text);">
                        {clean_text(cf["after_state"])}
                    </div>
                </div>
            """)

            render_html('<div class="section-heading">Predicted effects</div>')

            for effect in cf[
                "predicted_effects"
            ]:

                st.markdown(
                    f"- {effect}"
                )

            render_html('<div class="section-heading">Uncertainty</div>')

            st.warning(
                cf["uncertainty"]
            )


# ============================================================
# 23. METHODOLOGY
# ============================================================

elif page == "Methodology":

    render_html('<div class="section-heading">Science and methodology</div>')

    sections = [

        (
            "What the system actually does",
            """
            The platform has four distinct layers.

            First, it helps design an outreach experience.

            Second, it can generate hypotheses about possible
            engagement pathways.

            Third, it can help a facilitator respond to direct
            observations during a live interaction.

            Fourth, it can compare those hypotheses against
            real-world outcome data.
            """
        ),

        (
            "Prediction versus measurement",
            """
            This distinction is fundamental.

            Gemini-generated statements such as "the participant
            may become more focused" are predictions.

            Recorded observations such as "participant asked a
            technical question" are observations.

            Survey-derived changes such as a +1.8 curiosity shift
            are measurements calculated from recorded data.

            These categories should never be silently merged.
            """
        ),

        (
            "Evidence hierarchy",
            """
            STATED:
            Something explicitly reported by the participant.

            OBSERVED:
            Something directly witnessed by the facilitator.

            INFERRED:
            A reasonable interpretation of observed information.

            HYPOTHESIS:
            A speculative explanation requiring additional evidence.
            """
        ),

        (
            "Why the AI is not the measurement engine",
            """
            Generative models are useful for reasoning over complex
            qualitative context, generating alternatives, and
            proposing interventions.

            They should not be trusted to perform the authoritative
            arithmetic of an impact study.

            Therefore this application calculates quantitative
            changes deterministically in Python and uses Gemini
            primarily for interpretation.
            """
        ),

        (
            "What real impact means here",
            """
            A meaningful outreach outcome is not simply that a
            participant looked excited.

            Depending on the objective, useful indicators can include:

            - increased curiosity
            - increased scientific understanding
            - increased confidence asking questions
            - accurate recall
            - generation of new questions
            - voluntary follow-through
            - return engagement
            - participant-described meaning
            - willingness to explore further

            No single metric proves that outreach was successful.
            """
        ),

        (
            "Causality",
            """
            Pre/post changes are useful descriptive evidence but do
            not automatically prove that the outreach caused the
            change.

            Changes can also arise from novelty, prior knowledge,
            social context, selection effects, measurement effects,
            facilitator differences, or unrelated events.

            Strong causal claims require stronger experimental or
            quasi-experimental designs.
            """
        ),

        (
            "Privacy",
            """
            The application intentionally uses anonymous participant
            codes rather than names.

            Public deployments should minimize collection of
            personal information and should implement appropriate
            consent, retention, access-control, and deletion
            policies before collecting real participant data.
            """
        ),

    ]

    for title, body in sections:

        render_html(f"""
        <div class="premium-card">
            <div class="eyebrow">
                Method
            </div>
            <h3 style="margin-top:0;">
                {title}
            </h3>
            <div style="
                color:#a1a1aa;
                line-height:1.75;
                white-space:pre-line;
            ">
                {body}
            </div>
        </div>
        """)


# ============================================================
# 24. OPTIONAL LIGHTWEIGHT IMPACT CAPTURE
# ============================================================

st.markdown("---")

with st.expander(
    "Optional participant outcome capture"
):

    st.markdown(
        """
        This module is intentionally secondary to the outreach
        intelligence system.

        It allows an event team to collect lightweight pre/post
        outcomes when appropriate.
        """
    )

    events = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.caption(
            "Create an experience first."
        )

    else:

        event_map = {
            f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})": e
            for e in events
        }

        event_keys = list(event_map.keys())
        selected_event_name = st.selectbox(
            "Experience",
            event_keys,
            key="survey_event_select"
        )

        event = event_map[
            selected_event_name
        ]

        interactions = (
            db.query(Interaction)
            .filter(
                Interaction.event_id == event.id
            )
            .order_by(
                Interaction.started_at.desc()
            )
            .all()
        )

        if not interactions:

            st.caption(
                "No participant interactions yet."
            )

        else:

            interaction_map = {
                f"{i.participant_code} ({i.started_at.strftime('%H:%M:%S')})": i
                for i in interactions
            }

            int_keys = list(interaction_map.keys())
            selected_participant = st.selectbox(
                "Participant interaction",
                int_keys,
                key="survey_participant_select"
            )

            interaction = interaction_map[
                selected_participant
            ]

            timing_opts = [
                "BASELINE",
                "IMMEDIATE",
                "DELAYED_24H",
                "DELAYED_7D",
            ]
            survey_timing = st.selectbox(
                "Measurement point",
                timing_opts,
                key="survey_timing_select"
            )

            with st.form(
                "outcome_capture",
                clear_on_submit=True
            ):
                
                curiosity = st.slider(
                    "Curiosity",
                    1,
                    10,
                    5
                )

                understanding = st.slider(
                    "Scientific understanding",
                    0,
                    100,
                    50
                )

                confidence = st.slider(
                    "Confidence asking/answering questions",
                    1,
                    10,
                    5
                )

                recall = st.text_area(
                    "What do you remember most?",
                    height=90
                )

                follow_through = st.checkbox(
                    "I voluntarily explored something further afterward"
                )

                submitted = st.form_submit_button(
                    "Save outcome"
                )

                if submitted:

                    survey = Survey(
                        id=str(uuid.uuid4()),
                        interaction_id=interaction.id,
                        timing=survey_timing,
                        curiosity=float(curiosity),
                        understanding=float(
                            understanding
                        ),
                        confidence=float(
                            confidence
                        ),
                        recall_text=(
                            recall.strip()
                            or None
                        ),
                        follow_through=(
                            follow_through
                            if survey_timing.startswith(
                                "DELAYED"
                            )
                            else None
                        ),
                    )

                    db.add(survey)
                    db.commit()

                    st.success(
                        "Outcome recorded."
                    )


# ============================================================
# 25. FLOATING LIVE AI VOICE WIDGET (STRICT MALE VOICE ENFORCEMENT)
# ============================================================
import json

system_prompt_json = json.dumps(AI_SYSTEM)

voice_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}
body {{
    background: transparent;
    overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    justify-content: flex-end;
    padding: 10px;
}}

.voice-container {{
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 10px;
}}

.voice-fab {{
    width: 52px;
    height: 52px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
    user-select: none;
}}

.voice-fab.off {{
    background: rgba(18, 18, 22, 0.90);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}}
.voice-fab.off:hover {{
    background: rgba(30, 30, 38, 0.95);
    transform: scale(1.05);
}}
.voice-fab.off svg {{
    fill: #a1a1aa;
}}

.status-dot {{
    position: absolute;
    top: 2px;
    right: 2px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 2px solid #0b0b0d;
}}
.voice-fab.off .status-dot {{ background-color: #71717a; }}
.voice-fab.on .status-dot {{ background-color: #4ade80; box-shadow: 0 0 6px #4ade80; }}

.voice-fab.on {{
    background: #ffffff;
    border: 1px solid #ffffff;
    box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.2), 0 8px 24px rgba(91, 140, 255, 0.4);
    transform: scale(1.05);
}}
.voice-fab.on svg {{
    fill: #09090b;
}}

.voice-fab svg {{
    width: 22px;
    height: 22px;
}}

.voice-panel {{
    width: 290px;
    background: rgba(15, 15, 18, 0.95);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 12px 14px;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.5);
    display: none;
    color: #f4f4f5;
}}
.voice-panel.visible {{
    display: block;
}}

.panel-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}}

.voice-status-title {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: #94a3b8;
}}

.badge-state {{
    font-size: 0.60rem;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 700;
    text-transform: uppercase;
}}
.badge-off {{ background: rgba(255,255,255,0.08); color: #71717a; }}
.badge-on {{ background: rgba(74, 222, 128, 0.15); color: #4ade80; }}

.voice-text {{
    font-size: 0.75rem;
    color: #d1d5db;
    min-height: 42px;
    max-height: 95px;
    overflow-y: auto;
    line-height: 1.4;
    word-break: break-word;
    font-weight: 400;
}}
</style>
</head>
<body>

<div class="voice-container">
    <div id="voicePanel" class="voice-panel">
        <div class="panel-header">
            <span class="voice-status-title">AI Voice Copilot</span>
            <span id="voiceBadge" class="badge-state badge-off">OFF</span>
        </div>
        <div id="voiceText" class="voice-text">Tap the mic to start speaking...</div>
    </div>

    <div id="voiceFab" class="voice-fab off" onclick="toggleVoiceSession()">
        <div class="status-dot"></div>
        <svg viewBox="0 0 24 24">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
        </svg>
    </div>
</div>

<script>
let isListening = false;
let recognition = null;
let currentVoices = [];
const systemContext = {system_prompt_json};

function loadVoices() {{
    if ('speechSynthesis' in window) {{
        currentVoices = window.speechSynthesis.getVoices();
    }}
}}
if ('speechSynthesis' in window) {{
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
}}

function getStrictMaleVoice() {{
    if (!currentVoices || currentVoices.length === 0) loadVoices();

    const femaleKeywords = ['samantha', 'zira', 'victoria', 'karen', 'aria', 'jenny', 'siri', 'female', 'woman', 'lucy', 'catherine', 'hazel', 'susan', 'fiona', 'veena'];
    const maleKeywords = ['david', 'alex', 'fred', 'daniel', 'george', 'guy', 'mark', 'richard', 'james', 'thomas', 'oliver', 'male', 'man', 'google us english'];

    // Filter out female voices first
    const nonFemaleVoices = currentVoices.filter(v => 
        !femaleKeywords.some(f => v.name.toLowerCase().includes(f))
    );

    // 1. Look for explicit male name matches in English
    let selected = nonFemaleVoices.find(v => 
        v.lang.startsWith('en') && maleKeywords.some(m => v.name.toLowerCase().includes(m))
    );

    // 2. Fallback to any non-female English voice
    if (!selected) {{
        selected = nonFemaleVoices.find(v => v.lang.startsWith('en'));
    }}

    // 3. Absolute fallback
    return selected || currentVoices[0];
}}

function speakText(text, onComplete) {{
    if ('speechSynthesis' in window) {{
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        const maleVoice = getStrictMaleVoice();
        
        if (maleVoice) {{
            utterance.voice = maleVoice;
        }}
        
        utterance.lang = 'en-US';
        utterance.pitch = 0.75; // Lower pitch significantly to force a deeper male tone on any system voice
        utterance.rate = 1.0;

        utterance.onend = () => {{ if (onComplete) onComplete(); }};
        utterance.onerror = () => {{ if (onComplete) onComplete(); }};
        
        window.speechSynthesis.speak(utterance);
    }} else if (onComplete) {{
        onComplete();
    }}
}}

async function queryGeminiVoice(userInput) {{
    const apiKey = "{api_key}";
    const selectedModel = "{selected_model}";
    const textDiv = document.getElementById('voiceText');
    const badge = document.getElementById('voiceBadge');

    if (!apiKey) {{
        textDiv.innerText = "API key missing.";
        return;
    }}

    textDiv.innerText = "Thinking...";
    badge.innerText = "THINKING";

    try {{
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${{selectedModel}}:generateContent?key=${{apiKey}}`, {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{
                system_instruction: {{
                    parts: [{{
                        text: `You are the AI Voice Copilot. System context: ${{systemContext}}. Speak concisely in 1-2 sentences maximum.`
                    }}]
                }},
                contents: [{{ parts: [{{ text: userInput }}] }}]
            }})
        }});

        const data = await response.json();
        const reply = data.candidates?.[0]?.content?.parts?.[0]?.text?.trim() || "I couldn't process that clearly. Please try again.";

        textDiv.innerText = reply;
        badge.innerText = "SPEAKING";
        
        speakText(reply, () => {{
            if (isListening) {{
                badge.innerText = "LISTENING";
                try {{ recognition.start(); }} catch(e){{}}
            }}
        }});

    }} catch (err) {{
        textDiv.innerText = "Connection error. Retrying...";
        badge.innerText = "ERROR";
    }}
}}

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {{
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.lang = 'en-US';

    recognition.onresult = function(event) {{
        if (event.results && event.results[0]) {{
            const transcript = event.results[0][0].transcript;
            document.getElementById('voiceText').innerText = 'You: "' + transcript + '"';
            queryGeminiVoice(transcript);
        }}
    }};

    recognition.onerror = function() {{
        if (isListening) {{
            try {{ recognition.start(); }} catch(e){{}}
        }}
    }};
}}

function toggleVoiceSession() {{
    const panel = document.getElementById('voicePanel');
    const fab = document.getElementById('voiceFab');
    const badge = document.getElementById('voiceBadge');
    const textDiv = document.getElementById('voiceText');

    if (!isListening) {{
        panel.classList.add('visible');
        fab.classList.replace('off', 'on');
        badge.className = "badge-state badge-on";
        badge.innerText = "LISTENING";
        textDiv.innerText = "Listening clearly...";
        
        speakText("Online. How can I help?", () => {{
            if (recognition) {{
                try {{ recognition.start(); }} catch(e){{}}
            }}
        }});

        isListening = true;
    }} else {{
        fab.classList.replace('on', 'off');
        badge.className = "badge-state badge-off";
        badge.innerText = "OFF";
        textDiv.innerText = "Muted.";
        
        if (recognition) {{
            try {{ recognition.stop(); }} catch(e){{}}
        }}
        window.speechSynthesis.cancel();
        isListening = false;
        setTimeout(() => {{ panel.classList.remove('visible'); }}, 1500);
    }}
}}
</script>
</body>
</html>
"""

components.html(voice_html, height=220, width=320)


# ============================================================
# 26. FOOTER
# ============================================================

render_html("""
<div style="
    text-align:center;
    margin-top:70px;
    padding-top:25px;
    border-top:1px solid #202024;
    color:#52525b;
    font-size:.78rem;
    line-height:1.6;
">
    <div style="
        color:#71717a;
        margin-bottom:8px;
    ">
        Outreach Intelligence Lab
    </div>

    <div>
        Exploratory generative modeling and
        evidence-informed science outreach.
    </div>

    <div style="
        max-width:850px;
        margin:12px auto 0 auto;
    ">
        AI-generated predictions are synthetic hypotheses.
        They do not establish psychological, neurological,
        clinical, or causal facts about individuals.
        Real-world impact metrics are calculated from recorded
        observations and participant-reported outcomes.
    </div>
</div>
""")


# ============================================================
# 27. CLEANUP
# ============================================================

db.close()
