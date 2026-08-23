"""
NINOLADES OUTREACH INTELLIGENCE LAB
===================================

A self-contained Streamlit application for:
    - Designing outreach scenarios
    - Modeling likely engagement pathways
    - Running a live outreach copilot
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
from datetime import datetime, timezone
from typing import List, Literal, Optional, Dict, Any

import pandas as pd
import streamlit as st

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
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from pydantic import BaseModel, Field, ValidationError

from google import genai
from google.genai import types


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "Outreach Intelligence Lab"
APP_VERSION = "1.0.0"

# Change ONLY these if your Google account uses different IDs.
MODEL_FLASH = "gemini-3.6-flash"
MODEL_PRO = "gemini-3.1-pro"
MODEL_LITE = "gemini-3.5-flash-lite"

DEFAULT_MODEL = MODEL_FLASH

DATABASE_URL = os.getenv(
    "OUTREACH_DATABASE_URL",
    "sqlite:///ninolades_outreach_lab.db"
)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. PREMIUM MINIMALIST UI
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
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

.stButton button {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 9px;
    min-height: 42px;
    transition: all .18s ease;
}

.stButton button:hover {
    border-color: var(--accent);
    color: white;
    background: #191a20;
}

button[data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: white !important;
}

button[data-testid="baseButton-primary"]:hover {
    filter: brightness(1.08);
}

.stSlider [data-baseweb="slider"] {
    color: var(--accent) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 28px;
    border-bottom: 1px solid var(--border-soft);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: white !important;
    border-bottom: 2px solid var(--accent) !important;
}

.stAlert {
    border-radius: 10px !important;
}

.premium-card {
    background: linear-gradient(
        145deg,
        rgba(255,255,255,.025),
        rgba(255,255,255,.012)
    );
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 18px;
}

.premium-card:hover {
    border-color: #34343c;
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

.progress-track {
    width: 100%;
    height: 7px;
    background: #27272c;
    border-radius: 99px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 99px;
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

</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ============================================================
# 4. DATABASE
# ============================================================

Base = declarative_base()

engine_kwargs = {}

if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    **engine_kwargs
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)


class Event(Base):
    __tablename__ = "events"

    id = Column(String, primary_key=True)
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


Base.metadata.create_all(engine)


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
# 8. SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "active_event_id": None,
    "active_interaction_id": None,
    "last_recommendation": None,
    "last_forward_model": None,
    "last_counterfactual": None,
    "last_impact_interpretation": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 9. HELPER FUNCTIONS
# ============================================================

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
# 15. IMPACT INTERPRETATION
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


# ============================================================
# 16. HEADER
# ============================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


# ============================================================
# 17. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            color:white;
            font-size:1.15rem;
            font-weight:500;
            margin-bottom:5px;
        ">
            Outreach Intelligence
        </div>
        <div class="small-note">
            Human-centered science engagement system
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    model_options = {
        "Gemini 3.6 Flash": MODEL_FLASH,
        "Gemini 3.1 Pro": MODEL_PRO,
        "Gemini 3.5 Flash-Lite": MODEL_LITE,
    }

    selected_model_label = st.selectbox(
        "Reasoning engine",
        list(model_options.keys()),
        index=0,
        help="Choose the Gemini model used for generative analysis."
    )

    selected_model = model_options[
        selected_model_label
    ]

    st.markdown(
        """
        <div class="small-note" style="margin-top:8px;">
            Flash is the default for fast live outreach.
            Pro is intended for deeper analysis.
            Flash-Lite is intended for high-volume lightweight use.
        </div>
        """,
        unsafe_allow_html=True
    )

    api_key = get_api_key()

    client = create_gemini_client(
        api_key
    )

    if client is None:
        st.error(
            "Gemini is not configured. Add GEMINI_API_KEY "
            "to Streamlit secrets or the environment."
        )

    st.markdown("---")

    page = st.radio(
        "Workspace",
        [
            "Experience Designer",
            "Live Copilot",
            "Impact Observatory",
            "Counterfactual Lab",
            "Methodology",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        f"""
        <div class="small-note">
            Version {APP_VERSION}<br>
            AI outputs are hypotheses, not measurements.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 18. DATABASE INSTANCE
# ============================================================

db = db_session()


# ============================================================
# 19. EXPERIENCE DESIGNER
# ============================================================

if page == "Experience Designer":

    st.markdown(
        '<div class="section-heading">Design an outreach experience</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        event_name = st.text_input(
            "Experience name",
            placeholder="e.g. Saturn Under the Southern Sky"
        )

        objective = st.selectbox(
            "Primary objective",
            [
                "Curiosity",
                "Scientific understanding",
                "Awe and wonder",
                "Memory and retention",
                "Question generation",
                "Independent follow-through",
                "General engagement",
            ]
        )

        target_audience = st.selectbox(
            "Audience",
            [
                "General public",
                "Students",
                "Families",
                "Educators",
                "Astronomy enthusiasts",
                "Eco-tourists",
                "Mixed audience",
            ]
        )

    with right:

        environment = st.text_input(
            "Physical environment",
            placeholder="Dark-sky lawn, school courtyard, museum..."
        )

        acoustic_environment = st.text_input(
            "Acoustic / musical environment",
            placeholder="Silent, ambient sound, live ukulele..."
        )

        sensory_environment = st.text_input(
            "Relevant environmental conditions",
            placeholder="Lighting, crowd density, temperature, noise..."
        )

    context = st.text_area(
        "Experience description",
        placeholder=(
            "Describe what participants encounter, "
            "what the facilitator does, and the scientific content."
        ),
        height=130
    )

    st.markdown(
        '<div class="section-heading">Optional design variables</div>',
        unsafe_allow_html=True
    )

    design_col1, design_col2, design_col3 = st.columns(3)

    with design_col1:
        pacing = st.selectbox(
            "Pacing",
            [
                "Slow and contemplative",
                "Moderate",
                "Fast and energetic",
                "Variable"
            ]
        )

    with design_col2:
        interaction_style = st.selectbox(
            "Interaction style",
            [
                "Open observation",
                "Facilitator-led",
                "Question-led",
                "Hands-on",
                "Story-driven",
                "Mixed"
            ]
        )

    with design_col3:
        optional_choice = st.selectbox(
            "Participant autonomy",
            [
                "High",
                "Moderate",
                "Low"
            ]
        )

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

        st.markdown(
            '<div class="section-heading">Engagement architecture</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="premium-card">
                <div class="eyebrow">Current model</div>
                <div style="
                    color:white;
                    font-size:1.25rem;
                    margin-bottom:10px;
                ">
                    {clean_text(model["engagement_state"])}
                </div>
                <div class="small-note">
                    This is a generated hypothesis about possible
                    engagement dynamics, not a measurement of participants.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        cols = st.columns(3)

        for idx, pathway in enumerate(
            model["predicted_pathways"]
        ):

            with cols[idx]:

                st.markdown(
                    f"""
                    <div class="premium-card"
                         style="height:100%;">
                        <div class="eyebrow">
                            Pathway {idx + 1}
                        </div>
                        <h3>
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
                    """,
                    unsafe_allow_html=True
                )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                '<div class="section-heading">Design opportunities</div>',
                unsafe_allow_html=True
            )

            for item in model[
                "recommended_outreach_design"
            ]:
                st.markdown(
                    f"- {item}"
                )

        with c2:

            st.markdown(
                '<div class="section-heading">Potential friction</div>',
                unsafe_allow_html=True
            )

            for item in model[
                "likely_friction_points"
            ]:
                st.markdown(
                    f"- {item}"
                )

        st.markdown(
            '<div class="section-heading">What should be measured?</div>',
            unsafe_allow_html=True
        )

        for item in model[
            "measurement_opportunities"
        ]:
            st.markdown(
                f"- {item}"
            )


# ============================================================
# 20. LIVE COPILOT
# ============================================================

elif page == "Live Copilot":

    st.markdown(
        '<div class="section-heading">Live outreach copilot</div>',
        unsafe_allow_html=True
    )

    events = (
        db.query(Event)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.info(
            "Create an experience in Experience Designer first."
        )

    else:

        event_map = {
            f"{e.name}": e
            for e in events
        }

        chosen_name = st.selectbox(
            "Active experience",
            list(event_map.keys())
        )

        event = event_map[chosen_name]

        if (
            st.session_state.active_event_id
            != event.id
        ):

            st.session_state.active_event_id = event.id
            st.session_state.active_interaction_id = None

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

            st.markdown(
                f"""
                <div class="premium-card">
                    <div class="eyebrow">
                        Active interaction
                    </div>
                    <div style="
                        color:white;
                        font-size:1.2rem;
                    ">
                        {clean_text(interaction.participant_code)}
                    </div>
                    <div class="small-note">
                        Anonymous interaction code.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            phase = st.selectbox(
                "Current phase",
                [
                    "Approach",
                    "Introduction",
                    "Waiting",
                    "Direct observation",
                    "Explanation",
                    "Question/discussion",
                    "Reflection",
                    "Exit",
                ],
                index=[
                    "Approach",
                    "Introduction",
                    "Waiting",
                    "Direct observation",
                    "Explanation",
                    "Question/discussion",
                    "Reflection",
                    "Exit",
                ].index(interaction.phase)
                if interaction.phase in [
                    "Approach",
                    "Introduction",
                    "Waiting",
                    "Direct observation",
                    "Explanation",
                    "Question/discussion",
                    "Reflection",
                    "Exit",
                ]
                else 0
            )

            if phase != interaction.phase:

                interaction.phase = phase
                db.commit()

            preference = st.text_input(
                "Participant-stated preference",
                value=interaction.stated_preference or "",
                placeholder=(
                    "Only record what the participant explicitly states."
                )
            )

            if preference != (
                interaction.stated_preference or ""
            ):

                interaction.stated_preference = preference
                db.commit()

            st.markdown(
                '<div class="section-heading">Rapid observation</div>',
                unsafe_allow_html=True
            )

            observation_buttons = [
                (
                    "Attention",
                    "Participant appears to be observing the target.",
                ),
                (
                    "Attention",
                    "Participant looks away from the target.",
                ),
                (
                    "Participation",
                    "Participant asks a question.",
                ),
                (
                    "Participation",
                    "Participant gives a detailed response.",
                ),
                (
                    "Participation",
                    "Participant listens without responding.",
                ),
                (
                    "Reflection",
                    "Participant pauses after the experience.",
                ),
                (
                    "Friction",
                    "Participant appears to have difficulty hearing.",
                ),
                (
                    "Friction",
                    "Environmental interruption occurs.",
                ),
            ]

            obs_cols = st.columns(4)

            for idx, (category, detail) in enumerate(
                observation_buttons
            ):

                with obs_cols[idx % 4]:

                    if st.button(
                        detail,
                        key=f"obs_{idx}",
                        use_container_width=True
                    ):

                        log_observation(
                            db,
                            interaction.id,
                            category,
                            detail,
                            "OBSERVED"
                        )

                        st.toast(
                            "Observation recorded."
                        )

            custom_obs = st.text_input(
                "Custom observation",
                placeholder=(
                    "Describe only what was directly observed."
                )
            )

            if st.button(
                "Record observation"
            ):

                if custom_obs.strip():

                    log_observation(
                        db,
                        interaction.id,
                        "Custom",
                        custom_obs.strip(),
                        "OBSERVED"
                    )

                    st.success(
                        "Observation recorded."
                    )

            st.markdown(
                '<div class="section-heading">Recent evidence</div>',
                unsafe_allow_html=True
            )

            observations = get_recent_observations(
                db,
                interaction.id
            )

            if observations:

                for observation in observations:

                    st.markdown(
                        f"""
                        <div class="observation-row">
                            <span class="badge">
                                {clean_text(observation.evidence_level)}
                            </span>
                            <span style="margin-left:8px;color:#e5e5e7;">
                                {clean_text(observation.detail)}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            else:

                st.caption(
                    "No observations recorded yet."
                )

            st.markdown(
                '<div class="section-heading">Adaptive guidance</div>',
                unsafe_allow_html=True
            )

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

                st.markdown(
                    f"""
                    <div class="premium-card"
                         style="
                         border-color:
                         rgba(91,140,255,.35);
                         background:
                         linear-gradient(
                           145deg,
                           rgba(91,140,255,.09),
                           rgba(91,140,255,.025)
                         );
                         ">
                        <div class="eyebrow">
                            Suggested next move
                        </div>

                        <div style="
                            color:white;
                            font-size:1.35rem;
                            margin-bottom:14px;
                        ">
                            {clean_text(
                                recommendation[
                                    "recommended_action"
                                ]
                            )}
                        </div>

                        <div style="
                            color:#d4d4d8;
                            line-height:1.6;
                        ">
                            {clean_text(
                                recommendation["rationale"]
                            )}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                Confidence
                            </div>
                            <div class="metric-value">
                                {clean_text(
                                    recommendation["confidence"]
                                )}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with c2:

                    estimate = recommendation[
                        "cognitive_estimate"
                    ]

                    st.markdown(
                        f"""
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
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    '<div class="section-heading">Evidence used</div>',
                    unsafe_allow_html=True
                )

                for evidence in recommendation[
                    "evidence"
                ]:
                    st.markdown(
                        f"- {evidence}"
                    )

                st.markdown(
                    '<div class="section-heading">Alternative explanation</div>',
                    unsafe_allow_html=True
                )

                st.write(
                    recommendation[
                        "alternative_explanation"
                    ]
                )

                st.markdown(
                    '<div class="section-heading">Next observation to watch</div>',
                    unsafe_allow_html=True
                )

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
# 21. IMPACT OBSERVATORY
# ============================================================

elif page == "Impact Observatory":

    st.markdown(
        '<div class="section-heading">Real-world impact observatory</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="premium-card">
            <div class="eyebrow">Why this exists</div>
            <div style="
                color:#e5e5e7;
                line-height:1.7;
            ">
                The system separates what the AI predicts from what
                actually happened. Impact is calculated from recorded
                participant outcomes rather than generated by Gemini.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    events = (
        db.query(Event)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.info(
            "No experiences have been created yet."
        )

    else:

        event_map = {
            e.name: e
            for e in events
        }

        selected_name = st.selectbox(
            "Experience",
            list(event_map.keys())
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
                    <div class="metric-label">
                        Participants
                    </div>
                    <div class="metric-value">
                        {metrics["participants"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            cols[1].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Curiosity change
                    </div>
                    <div class="metric-value">
                        {
                            "—"
                            if metrics["curiosity_change"] is None
                            else f'{metrics["curiosity_change"]:+.2f}'
                        }
                    </div>
                    <div class="metric-sub">
                        Paired baseline → immediate
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            cols[2].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Understanding change
                    </div>
                    <div class="metric-value">
                        {
                            "—"
                            if metrics["understanding_change"] is None
                            else f'{metrics["understanding_change"]:+.2f}'
                        }
                    </div>
                    <div class="metric-sub">
                        Paired baseline → immediate
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            cols[3].markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        Follow-through
                    </div>
                    <div class="metric-value">
                        {
                            "—"
                            if metrics["follow_through_rate"] is None
                            else f'{metrics["follow_through_rate"]:.1f}%'
                        }
                    </div>
                    <div class="metric-sub">
                        Delayed self-report
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-heading">Outcome signals</div>',
                unsafe_allow_html=True
            )

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

            st.markdown(
                '<div class="section-heading">Delayed indicators</div>',
                unsafe_allow_html=True
            )

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

            st.markdown(
                '<div class="section-heading">AI interpretation</div>',
                unsafe_allow_html=True
            )

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

                st.markdown(
                    f"""
                    <div class="premium-card">
                        <div class="eyebrow">
                            Interpretation
                        </div>
                        <div style="
                            color:#e5e5e7;
                            line-height:1.7;
                        ">
                            {clean_text(
                                interpretation[
                                    "overall_interpretation"
                                ]
                            )}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(
                        '<div class="section-heading">Strongest signal</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        interpretation[
                            "strongest_signal"
                        ]
                    )

                with c2:

                    st.markdown(
                        '<div class="section-heading">Weakest signal</div>',
                        unsafe_allow_html=True
                    )

                    st.write(
                        interpretation[
                            "weakest_signal"
                        ]
                    )

                st.markdown(
                    '<div class="section-heading">Plausible mechanisms</div>',
                    unsafe_allow_html=True
                )

                for item in interpretation[
                    "plausible_mechanisms"
                ]:
                    st.markdown(
                        f"- {item}"
                    )

                st.markdown(
                    '<div class="section-heading">Alternative explanations</div>',
                    unsafe_allow_html=True
                )

                for item in interpretation[
                    "alternative_explanations"
                ]:
                    st.markdown(
                        f"- {item}"
                    )

                st.markdown(
                    '<div class="section-heading">Recommended next test</div>',
                    unsafe_allow_html=True
                )

                st.info(
                    interpretation[
                        "recommended_next_test"
                    ]
                )


# ============================================================
# 22. COUNTERFACTUAL LAB
# ============================================================

elif page == "Counterfactual Lab":

    st.markdown(
        '<div class="section-heading">Counterfactual experiment lab</div>',
        unsafe_allow_html=True
    )

    events = (
        db.query(Event)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.info(
            "Create an experience first."
        )

    else:

        event_map = {
            e.name: e
            for e in events
        }

        selected_name = st.selectbox(
            "Experience",
            list(event_map.keys())
        )

        event = event_map[selected_name]

        st.markdown(
            """
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
            """,
            unsafe_allow_html=True
        )

        variable_change = st.text_area(
            "What would you change?",
            placeholder=(
                "What if the live music stopped during direct "
                "telescope observation?"
            ),
            height=100
        )

        design_description = st.text_area(
            "Current design",
            value=(
                event.context
                or ""
            ),
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

            st.markdown(
                f"""
                <div class="premium-card">
                    <div class="eyebrow">
                        Changed variable
                    </div>
                    <h3>
                        {clean_text(cf["changed_variable"])}
                    </h3>
                    <p>
                        {clean_text(cf["expected_difference"])}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            c1, c2 = st.columns(2)

            with c1:

                st.markdown(
                    f"""
                    <div class="premium-card">
                        <div class="eyebrow">
                            Baseline
                        </div>
                        <div style="color:#e5e5e7;">
                            {clean_text(cf["before_state"])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    f"""
                    <div class="premium-card">
                        <div class="eyebrow">
                            Counterfactual
                        </div>
                        <div style="color:#e5e5e7;">
                            {clean_text(cf["after_state"])}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                '<div class="section-heading">Predicted effects</div>',
                unsafe_allow_html=True
            )

            for effect in cf[
                "predicted_effects"
            ]:

                st.markdown(
                    f"- {effect}"
                )

            st.markdown(
                '<div class="section-heading">Uncertainty</div>',
                unsafe_allow_html=True
            )

            st.warning(
                cf["uncertainty"]
            )


# ============================================================
# 23. METHODOLOGY
# ============================================================

elif page == "Methodology":

    st.markdown(
        '<div class="section-heading">Science and methodology</div>',
        unsafe_allow_html=True
    )

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

        st.markdown(
            f"""
            <div class="premium-card">
                <div class="eyebrow">
                    Method
                </div>
                <h3>
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
            """,
            unsafe_allow_html=True
        )


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
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.caption(
            "Create an experience first."
        )

    else:

        event_map = {
            e.name: e
            for e in events
        }

        selected_event_name = st.selectbox(
            "Experience",
            list(event_map.keys()),
            key="survey_event"
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
                i.participant_code: i
                for i in interactions
            }

            selected_participant = st.selectbox(
                "Participant interaction",
                list(
                    interaction_map.keys()
                )
            )

            interaction = interaction_map[
                selected_participant
            ]

            survey_timing = st.selectbox(
                "Measurement point",
                [
                    "BASELINE",
                    "IMMEDIATE",
                    "DELAYED_24H",
                    "DELAYED_7D",
                ]
            )

            with st.form(
                "outcome_capture"
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
# 25. FOOTER
# ============================================================

st.markdown(
    """
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
    """,
    unsafe_allow_html=True
)


# ============================================================
# 26. CLEANUP
# ============================================================

db.close()
