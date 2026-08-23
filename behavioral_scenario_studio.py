# ============================================================
# NINOLADES OUTREACH INTELLIGENCE LAB
# ============================================================
#
# A self-contained Streamlit application for:
#
#   • Outreach scenario simulation
#   • Live facilitator assistance
#   • Human engagement mapping
#   • Impact trajectory analysis
#   • Optional participant feedback
#   • Counterfactual exploration
#   • Behavioral equifinality
#   • Event-level analytics
#   • Longitudinal impact signals
#   • Gemini structured interpretation
#
# IMPORTANT:
# This system is an exploratory outreach-support system.
# It does NOT diagnose people, read minds, measure cognition,
# or establish psychological causality.
#
# Evidence classes used throughout:
#
#   OBSERVED       = directly observed by facilitator/system
#   REPORTED       = explicitly reported by participant
#   COMPUTED       = mathematically calculated from recorded data
#   AI_INTERPRETED = generated interpretation by Gemini
#   HYPOTHESIS     = plausible but unverified explanation
#
# ============================================================

import os
import io
import json
import uuid
import math
import html
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Literal, Dict, Any

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
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_NAME = "Ninolades Outreach Intelligence Lab"
APP_VERSION = "1.0.0"

DB_URL = os.getenv(
    "DB_PATH",
    "sqlite:///ninolades_outreach_intelligence.db"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-2.5-flash-lite"
)


# ============================================================
# 2. DATABASE
# ============================================================

engine = create_engine(
    DB_URL,
    connect_args={"check_same_thread": False}
    if DB_URL.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()


class Event(Base):
    __tablename__ = "events"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    name = Column(String, nullable=False)

    date = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    objective = Column(String, nullable=False)

    setting = Column(String, nullable=True)

    acoustic_setting = Column(String, nullable=True)

    target_audience = Column(String, nullable=True)

    description = Column(Text, nullable=True)

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    interactions = relationship(
        "Interaction",
        back_populates="event",
        cascade="all, delete-orphan"
    )


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    event_id = Column(
        String,
        ForeignKey("events.id"),
        nullable=False
    )

    participant_code = Column(
        String,
        nullable=False
    )

    started_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    ended_at = Column(DateTime, nullable=True)

    phase = Column(
        String,
        default="Approach"
    )

    participant_preference = Column(
        Text,
        nullable=True
    )

    consent_feedback = Column(
        Integer,
        default=0
    )

    event = relationship(
        "Event",
        back_populates="interactions"
    )

    observations = relationship(
        "Observation",
        back_populates="interaction",
        cascade="all, delete-orphan"
    )

    feedback = relationship(
        "ParticipantFeedback",
        back_populates="interaction",
        cascade="all, delete-orphan"
    )

    adaptations = relationship(
        "Adaptation",
        back_populates="interaction",
        cascade="all, delete-orphan"
    )


class Observation(Base):
    __tablename__ = "observations"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    interaction_id = Column(
        String,
        ForeignKey("interactions.id"),
        nullable=False
    )

    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    category = Column(
        String,
        nullable=False
    )

    detail = Column(
        Text,
        nullable=False
    )

    evidence_level = Column(
        String,
        default="OBSERVED"
    )

    interaction = relationship(
        "Interaction",
        back_populates="observations"
    )


class Adaptation(Base):
    __tablename__ = "adaptations"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    interaction_id = Column(
        String,
        ForeignKey("interactions.id"),
        nullable=False
    )

    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    recommended_action = Column(
        Text,
        nullable=False
    )

    rationale = Column(
        Text,
        nullable=True
    )

    confidence = Column(
        String,
        nullable=True
    )

    interaction = relationship(
        "Interaction",
        back_populates="adaptations"
    )


class ParticipantFeedback(Base):
    __tablename__ = "participant_feedback"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )

    interaction_id = Column(
        String,
        ForeignKey("interactions.id"),
        nullable=False
    )

    timing = Column(
        String,
        nullable=False
    )

    curiosity = Column(
        Float,
        nullable=True
    )

    understanding = Column(
        Float,
        nullable=True
    )

    emotional_salience = Column(
        Float,
        nullable=True
    )

    confidence = Column(
        Float,
        nullable=True
    )

    free_text = Column(
        Text,
        nullable=True
    )

    follow_up_action = Column(
        String,
        nullable=True
    )

    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc)
    )

    interaction = relationship(
        "Interaction",
        back_populates="feedback"
    )


Base.metadata.create_all(bind=engine)


# ============================================================
# 3. DATABASE HELPERS
# ============================================================

def db_session():
    return SessionLocal()


def utcnow():
    return datetime.now(timezone.utc)


# ============================================================
# 4. GEMINI STRUCTURED SCHEMAS
# ============================================================

class OutreachRecommendation(BaseModel):
    recommended_action: str = Field(
        description="A specific, minimally disruptive action for the facilitator."
    )

    rationale: str = Field(
        description="Reasoning grounded only in supplied observations and context."
    )

    confidence: Literal[
        "Low",
        "Moderate",
        "High"
    ]

    evidence_used: List[str] = Field(
        min_length=1,
        max_length=6
    )

    alternative_explanation: str = Field(
        description="At least one plausible alternative explanation."
    )

    next_signal_to_watch: str = Field(
        description="Specific observable signal that could update the interpretation."
    )


class ScenarioPrediction(BaseModel):
    predicted_response: str

    mechanism: str

    confidence: Literal[
        "Low",
        "Moderate",
        "High"
    ]

    evidence_basis: List[str]

    uncertainty: str


class ScenarioAnalysis(BaseModel):
    situation_summary: str

    likely_engagement_opportunities: List[str] = Field(
        min_length=3,
        max_length=5
    )

    potential_friction_points: List[str] = Field(
        min_length=3,
        max_length=5
    )

    facilitator_strategy: List[str] = Field(
        min_length=3,
        max_length=5
    )

    predictions: List[ScenarioPrediction] = Field(
        min_length=3,
        max_length=3
    )

    major_uncertainties: List[str] = Field(
        min_length=3,
        max_length=5
    )


class EquifinalityExplanation(BaseModel):
    explanation_name: str

    compatibility: Literal[
        "Strong",
        "Moderate",
        "Weak"
    ]

    possible_mechanism: str

    supporting_evidence: List[str]

    missing_information: List[str]

    alternative_explanation: str


class EquifinalityAnalysis(BaseModel):
    ambiguity_statement: str

    explanations: List[EquifinalityExplanation] = Field(
        min_length=3,
        max_length=3
    )

    cannot_be_inferred: List[str]

    best_next_observation: str


class CounterfactualAnalysis(BaseModel):
    changed_variable: str

    expected_difference: str

    engagement_effect: str

    possible_benefit: str

    possible_cost: str

    confidence: Literal[
        "Low",
        "Moderate",
        "High"
    ]

    what_would_test_this: str


class ImpactInterpretation(BaseModel):
    impact_summary: str

    strongest_positive_signal: str

    weakest_or_uncertain_signal: str

    possible_mechanisms: List[str] = Field(
        min_length=2,
        max_length=4
    )

    alternative_explanations: List[str] = Field(
        min_length=2,
        max_length=4
    )

    recommended_next_measurement: str

    causal_claim_strength: Literal[
        "Descriptive only",
        "Suggestive but non-causal",
        "Stronger evidence needed"
    ]


class ThemeAnalysis(BaseModel):
    themes: List[str] = Field(
        min_length=1,
        max_length=5
    )

    memorable_elements: List[str] = Field(
        min_length=1,
        max_length=5
    )

    curiosity_signals: List[str]

    limitations: List[str]


# ============================================================
# 5. GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or st.secrets.get("GEMINI_API_KEY", "")
    )

    if not api_key:
        return None

    try:
        return genai.Client(
            api_key=api_key
        )
    except Exception:
        return None


gemini = get_gemini_client()


# ============================================================
# 6. AI SYSTEM INSTRUCTION
# ============================================================

AI_SYSTEM = """
You are the interpretation engine of the Ninolades Outreach Intelligence Lab.

This is an exploratory science-communication and human-engagement system.

STRICT EPISTEMIC RULES:

1. Never diagnose a participant.
2. Never infer autism, ADHD, personality disorder, intelligence,
   trauma, mental illness, neurological conditions, or other
   clinical/personality identities from behavior.
3. Never claim to read internal mental states.
4. Never convert behavioral observations into psychological facts.
5. Treat facilitator observations as observations, not mind-reading.
6. Treat participant statements as participant-reported evidence.
7. Prefer explicit participant preferences over inferred preferences.
8. Always provide alternative explanations when interpreting behavior.
9. Use uncertainty explicitly.
10. Recommendations must be minimally disruptive and reversible.
11. "Do nothing" or "continue normally" is a legitimate recommendation.
12. Do not optimize for engagement at the expense of participant autonomy.
13. Do not manipulate participants.
14. Do not use coercive persuasion.
15. Do not infer demographic or protected attributes.
16. Never claim causality from a single interaction.
17. Distinguish:
      OBSERVED
      REPORTED
      COMPUTED
      AI_INTERPRETED
      HYPOTHESIS
18. If evidence is insufficient, explicitly say so.
19. The goal is better outreach, not behavioral control.
"""


# ============================================================
# 7. GENERIC GEMINI CALL
# ============================================================

def gemini_structured(
    prompt: str,
    schema,
    temperature: float = 0.2
):

    if gemini is None:
        raise RuntimeError(
            "Gemini client unavailable. "
            "Set GEMINI_API_KEY."
        )

    models_to_try = [
        GEMINI_MODEL,
        GEMINI_FALLBACK_MODEL
    ]

    last_error = None

    for model_name in models_to_try:

        try:

            response = gemini.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=AI_SYSTEM,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature
                )
            )

            if hasattr(response, "parsed") and response.parsed:
                return response.parsed

            if getattr(response, "text", None):
                return schema.model_validate_json(
                    response.text
                )

        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Gemini generation failed: {last_error}"
    )


# ============================================================
# 8. AI FUNCTIONS
# ============================================================

def generate_live_recommendation(
    event: Event,
    interaction: Interaction,
    observations: List[Observation]
):

    observation_text = "\n".join(
        f"- [{o.evidence_level}] "
        f"{o.category}: {o.detail}"
        for o in observations
    )

    prompt = f"""
EVENT:
Name: {event.name}
Objective: {event.objective}
Setting: {event.setting}
Acoustic setting: {event.acoustic_setting}
Audience: {event.target_audience}

CURRENT INTERACTION:
Phase: {interaction.phase}
Participant-stated preference:
{interaction.participant_preference or "Not provided"}

RECENT OBSERVATIONS:
{observation_text or "No observations yet."}

TASK:

Recommend the next facilitator action.

The recommendation should be:
- practical
- immediately usable
- minimally disruptive
- reversible
- respectful of participant autonomy

Do not assume why the participant behaved this way.

Return a structured recommendation.
"""

    return gemini_structured(
        prompt,
        OutreachRecommendation,
        temperature=0.15
    )


def generate_scenario_analysis(
    objective,
    setting,
    audience,
    sensory_context,
    facilitator_style,
    situation
):

    prompt = f"""
OUTREACH SCENARIO

Objective:
{objective}

Setting:
{setting}

Audience:
{audience}

Sensory/context factors:
{sensory_context}

Facilitator style:
{facilitator_style}

Situation:
{situation}

Analyze this scenario as an outreach-planning problem.

Identify:
1. engagement opportunities
2. possible friction
3. practical facilitator strategies
4. three meaningfully different plausible response pathways
5. major uncertainties

Do not claim that any predicted behavior is certain.
"""

    return gemini_structured(
        prompt,
        ScenarioAnalysis,
        temperature=0.25
    )


def generate_equifinality(
    situation,
    observed_behavior,
    context
):

    prompt = f"""
SITUATION:
{situation}

OBSERVED BEHAVIOR:
{observed_behavior}

KNOWN CONTEXT:
{context}

Perform behavioral equifinality analysis.

The same behavior may have multiple explanations.

Generate exactly three genuinely different plausible explanations.

For each:
- identify the mechanism
- identify evidence
- identify missing information
- provide an alternative explanation

Do not diagnose the participant.
Do not infer personality or neurotype.
Do not treat one behavior as proof of an internal state.
"""

    return gemini_structured(
        prompt,
        EquifinalityAnalysis,
        temperature=0.3
    )


def generate_counterfactual(
    baseline,
    changed_variable
):

    prompt = f"""
BASELINE OUTREACH CONFIGURATION:
{baseline}

COUNTERFACTUAL CHANGE:
{changed_variable}

Assume all other factors remain constant.

Analyze the plausible difference this single change could produce.

Do not claim certainty or causality.

Include:
- expected difference
- possible benefit
- possible cost
- confidence
- what future observation would test the hypothesis
"""

    return gemini_structured(
        prompt,
        CounterfactualAnalysis,
        temperature=0.25
    )


def generate_impact_interpretation(
    metrics,
    qualitative_data
):

    prompt = f"""
EVENT IMPACT DATA:

{json.dumps(metrics, indent=2, default=str)}

QUALITATIVE MATERIAL:

{json.dumps(qualitative_data, indent=2, default=str)}

Interpret the impact data.

IMPORTANT:

Do not claim the outreach caused the measured changes.

Distinguish:
- descriptive evidence
- participant-reported evidence
- computed changes
- plausible mechanisms
- alternative explanations

Identify the strongest signal and the weakest/most uncertain signal.

Recommend the next useful measurement.
"""

    return gemini_structured(
        prompt,
        ImpactInterpretation,
        temperature=0.2
    )


def generate_theme_analysis(texts):

    joined = "\n\n".join(
        f"Response {i+1}: {text}"
        for i, text in enumerate(texts)
    )

    prompt = f"""
PARTICIPANT-REPORTED TEXT:

{joined}

Identify recurring themes and memorable elements.

Only summarize what appears in the text.

Do not infer personality, diagnosis, intelligence,
or hidden psychological states.
"""

    return gemini_structured(
        prompt,
        ThemeAnalysis,
        temperature=0.1
    )


# ============================================================
# 9. DETERMINISTIC ANALYTICS
# ============================================================

def safe_mean(series):

    if series is None:
        return None

    series = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(series) == 0:
        return None

    return float(series.mean())


def calculate_event_metrics(
    db,
    event_id
):

    interactions = (
        db.query(Interaction)
        .filter(
            Interaction.event_id == event_id
        )
        .all()
    )

    if not interactions:
        return {
            "participants": 0
        }

    interaction_ids = [
        i.id
        for i in interactions
    ]

    feedback = (
        db.query(ParticipantFeedback)
        .filter(
            ParticipantFeedback.interaction_id.in_(
                interaction_ids
            )
        )
        .all()
    )

    observations = (
        db.query(Observation)
        .filter(
            Observation.interaction_id.in_(
                interaction_ids
            )
        )
        .all()
    )

    adaptations = (
        db.query(Adaptation)
        .filter(
            Adaptation.interaction_id.in_(
                interaction_ids
            )
        )
        .all()
    )

    metrics = {
        "participants": len(interactions),
        "feedback_records": len(feedback),
        "observations": len(observations),
        "adaptations": len(adaptations),
    }

    if feedback:

        rows = []

        for f in feedback:

            rows.append({
                "interaction_id": f.interaction_id,
                "timing": f.timing,
                "curiosity": f.curiosity,
                "understanding": f.understanding,
                "emotional_salience": f.emotional_salience,
                "confidence": f.confidence,
                "follow_up_action": f.follow_up_action,
            })

        df = pd.DataFrame(rows)

        baseline = df[
            df["timing"] == "BASELINE"
        ]

        immediate = df[
            df["timing"] == "IMMEDIATE"
        ]

        delayed = df[
            df["timing"].isin(
                ["DELAYED_24H", "DELAYED_7D"]
            )
        ]

        metrics["baseline_curiosity"] = safe_mean(
            baseline["curiosity"]
        )

        metrics["immediate_curiosity"] = safe_mean(
            immediate["curiosity"]
        )

        metrics["baseline_understanding"] = safe_mean(
            baseline["understanding"]
        )

        metrics["immediate_understanding"] = safe_mean(
            immediate["understanding"]
        )

        metrics["baseline_salience"] = safe_mean(
            baseline["emotional_salience"]
        )

        metrics["immediate_salience"] = safe_mean(
            immediate["emotional_salience"]
        )

        metrics["curiosity_change"] = None

        if (
            metrics["baseline_curiosity"] is not None
            and metrics["immediate_curiosity"] is not None
        ):
            metrics["curiosity_change"] = round(
                metrics["immediate_curiosity"]
                - metrics["baseline_curiosity"],
                2
            )

        metrics["understanding_change"] = None

        if (
            metrics["baseline_understanding"] is not None
            and metrics["immediate_understanding"] is not None
        ):
            metrics["understanding_change"] = round(
                metrics["immediate_understanding"]
                - metrics["baseline_understanding"],
                2
            )

        metrics["follow_up_rate"] = None

        if not delayed.empty:

            values = delayed[
                "follow_up_action"
            ].dropna()

            if len(values):

                positive = values.astype(str).str.lower().isin(
                    [
                        "yes",
                        "looked something up",
                        "searched",
                        "continued",
                        "returned",
                    ]
                )

                metrics["follow_up_rate"] = round(
                    float(positive.mean() * 100),
                    1
                )

    return metrics


# ============================================================
# 10. IMPACT SIGNAL MODEL
# ============================================================

def build_impact_signals(metrics):

    signals = []

    if metrics.get("curiosity_change") is not None:

        change = metrics["curiosity_change"]

        if change > 1:
            status = "Positive signal"
        elif change < -1:
            status = "Decline signal"
        else:
            status = "Little measured change"

        signals.append({
            "name": "Curiosity",
            "value": change,
            "status": status,
            "evidence": "COMPUTED"
        })

    if metrics.get("understanding_change") is not None:

        change = metrics["understanding_change"]

        if change > 5:
            status = "Positive signal"
        elif change < -5:
            status = "Decline signal"
        else:
            status = "Little measured change"

        signals.append({
            "name": "Understanding",
            "value": change,
            "status": status,
            "evidence": "COMPUTED"
        })

    if metrics.get("follow_up_rate") is not None:

        signals.append({
            "name": "Voluntary follow-through",
            "value": metrics["follow_up_rate"],
            "status": (
                "Observed follow-through"
                if metrics["follow_up_rate"] > 0
                else "No recorded follow-through"
            ),
            "evidence": "REPORTED"
        })

    return signals


# ============================================================
# 11. CSS
# ============================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>

:root {
    --bg: #090a0c;
    --panel: #111318;
    --panel2: #15171c;
    --border: #252932;
    --text: #f4f5f7;
    --muted: #9297a3;
    --accent: #6ea8ff;
    --green: #62d49a;
    --orange: #f3b562;
    --red: #ef7272;
    --purple: #a995ff;
}

.stApp {
    background:
        radial-gradient(
            circle at 80% 0%,
            rgba(60,90,150,0.08),
            transparent 35%
        ),
        var(--bg);
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background: #0c0d10;
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4 {
    color: var(--text) !important;
    letter-spacing: -0.025em;
}

.hero {
    padding: 32px 0 24px 0;
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 300;
    margin-bottom: 6px;
}

.hero-subtitle {
    color: var(--muted);
    font-size: 1rem;
    max-width: 850px;
    line-height: 1.6;
}

.card {
    background: rgba(17,19,24,0.88);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px;
    margin-bottom: 18px;
}

.metric-card {
    background: #111318;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
    min-height: 110px;
}

.metric-label {
    color: var(--muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.metric-value {
    color: white;
    font-size: 1.8rem;
    font-weight: 500;
    margin-top: 7px;
}

.metric-caption {
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 4px;
}

.signal {
    background: #101217;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}

.signal-title {
    font-weight: 600;
    color: white;
}

.signal-meta {
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 4px;
}

.evidence {
    display: inline-block;
    border-radius: 5px;
    padding: 3px 7px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-right: 5px;
}

.observed {
    color: #c9b8ff;
    background: #211d30;
}

.reported {
    color: #72d8ff;
    background: #14242c;
}

.computed {
    color: #6fe0a5;
    background: #14251d;
}

.ai {
    color: #7eaeff;
    background: #162238;
}

.hypothesis {
    color: #ffc879;
    background: #2b2114;
}

.section {
    color: white;
    font-size: 1.25rem;
    font-weight: 450;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin: 24px 0 18px 0;
}

.timeline {
    border-left: 2px solid #28303d;
    padding-left: 20px;
    margin-left: 8px;
}

.timeline-item {
    margin-bottom: 22px;
    position: relative;
}

.timeline-dot {
    position: absolute;
    left: -28px;
    top: 3px;
    width: 12px;
    height: 12px;
    background: var(--accent);
    border-radius: 50%;
}

.small-muted {
    color: var(--muted);
    font-size: 0.85rem;
}

.warning-box {
    background: #211b10;
    border: 1px solid #4b3b20;
    border-radius: 10px;
    padding: 14px 16px;
    color: #e6d3ad;
}

.info-box {
    background: #101c2d;
    border: 1px solid #263d61;
    border-radius: 10px;
    padding: 14px 16px;
    color: #c7d9f7;
}

.success-box {
    background: #102118;
    border: 1px solid #244d36;
    border-radius: 10px;
    padding: 14px 16px;
    color: #c4e8d3;
}

button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 12. HEADER
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">
        🔭 Outreach Intelligence Lab
    </div>
    <div class="hero-subtitle">
        A human-centered intelligence layer for designing,
        adapting, and measuring science outreach experiences.
        It separates what was observed from what was reported,
        calculated, interpreted, and hypothesized.
    </div>
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# 13. SIDEBAR
# ============================================================

db = db_session()

pages = [
    "◈ Mission Control",
    "✦ Scenario Studio",
    "⚡ Live Outreach Copilot",
    "◉ Human Response Map",
    "↗ Impact Trajectory",
    "◇ Counterfactual Lab",
    "∿ Equifinality Engine",
    "◎ Event Observatory",
    "☷ Methodology",
]

with st.sidebar:

    st.markdown(
        "### 🔭 Outreach Intelligence"
    )

    page = st.radio(
        "Navigate",
        pages,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        """
        <div class="small-muted">
        <b>Evidence architecture</b><br><br>
        OBSERVED · directly seen<br>
        REPORTED · participant stated<br>
        COMPUTED · mathematical result<br>
        AI · model interpretation<br>
        HYPOTHESIS · unverified possibility
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    if gemini:
        st.success(
            f"Gemini online · {GEMINI_MODEL}"
        )
    else:
        st.warning(
            "Gemini offline"
        )


# ============================================================
# 14. HELPER FUNCTIONS
# ============================================================

def get_events():

    return (
        db.query(Event)
        .order_by(Event.created_at.desc())
        .all()
    )


def get_event_map():

    events = get_events()

    return {
        e.name: e
        for e in events
    }


def create_interaction(event_id):

    participant_code = (
        "P-"
        + uuid.uuid4().hex[:6].upper()
    )

    interaction = Interaction(
        event_id=event_id,
        participant_code=participant_code
    )

    db.add(interaction)
    db.commit()

    return interaction


def current_interaction():

    interaction_id = st.session_state.get(
        "active_interaction_id"
    )

    if not interaction_id:
        return None

    return (
        db.query(Interaction)
        .filter(
            Interaction.id == interaction_id
        )
        .first()
    )


def render_metric(
    label,
    value,
    caption=""
):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
            <div class="metric-caption">{html.escape(str(caption))}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def evidence_badge(level):

    mapping = {
        "OBSERVED": "observed",
        "REPORTED": "reported",
        "COMPUTED": "computed",
        "AI_INTERPRETED": "ai",
        "HYPOTHESIS": "hypothesis"
    }

    cls = mapping.get(
        level,
        "observed"
    )

    return (
        f'<span class="evidence {cls}">'
        f'{html.escape(level)}'
        f'</span>'
    )


# ============================================================
# 15. MISSION CONTROL
# ============================================================

if page == "◈ Mission Control":

    events = get_events()

    st.markdown(
        "## Mission Control"
    )

    st.markdown(
        """
        <div class="info-box">
        <b>The core idea</b><br><br>
        This system is not designed to "read" people.
        It is designed to help outreach teams notice useful
        signals, respond respectfully, and learn whether an
        experience produced meaningful downstream effects.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section">System Architecture</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(6)

    stages = [
        ("01", "Design", "Plan the experience"),
        ("02", "Observe", "Record what actually happens"),
        ("03", "Adapt", "Choose a reversible response"),
        ("04", "Experience", "Deliver the outreach"),
        ("05", "Impact", "Measure meaningful signals"),
        ("06", "Learn", "Improve future outreach"),
    ]

    for col, (num, title, desc) in zip(
        cols,
        stages
    ):

        with col:

            st.markdown(
                f"""
                <div class="card">
                    <div style="color:#6ea8ff;
                    font-size:0.75rem">{num}</div>
                    <div style="font-size:1.05rem;
                    font-weight:600;margin-top:8px">
                    {title}
                    </div>
                    <div class="small-muted"
                    style="margin-top:7px">
                    {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section">Current Observatory</div>',
        unsafe_allow_html=True
    )

    total_events = len(events)

    total_interactions = (
        db.query(Interaction).count()
    )

    total_observations = (
        db.query(Observation).count()
    )

    total_feedback = (
        db.query(ParticipantFeedback).count()
    )

    cols = st.columns(4)

    with cols[0]:
        render_metric(
            "Events",
            total_events,
            "Created in this installation"
        )

    with cols[1]:
        render_metric(
            "Interactions",
            total_interactions,
            "Participant encounters"
        )

    with cols[2]:
        render_metric(
            "Observations",
            total_observations,
            "Recorded behavioral/context signals"
        )

    with cols[3]:
        render_metric(
            "Feedback records",
            total_feedback,
            "Optional participant reports"
        )

    st.markdown(
        '<div class="section">What Makes This Different</div>',
        unsafe_allow_html=True
    )

    features = [
        (
            "Human Response Map",
            "Tracks observable engagement without pretending to know what somebody is thinking."
        ),
        (
            "Impact Trajectory",
            "Moves beyond immediate satisfaction toward curiosity, understanding, memory and voluntary follow-through."
        ),
        (
            "Equifinality",
            "Shows that identical behavior can emerge from very different causes."
        ),
        (
            "Counterfactual Lab",
            "Lets facilitators ask what might change if one outreach variable changed."
        ),
        (
            "Evidence Architecture",
            "Makes the boundary between measurement and interpretation visible."
        ),
        (
            "Gemini Copilot",
            "Provides contextual interpretation while Python remains responsible for deterministic analytics."
        ),
    ]

    for title, description in features:

        st.markdown(
            f"""
            <div class="card">
                <b>{html.escape(title)}</b>
                <div class="small-muted"
                style="margin-top:6px">
                {html.escape(description)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 16. SCENARIO STUDIO
# ============================================================

elif page == "✦ Scenario Studio":

    st.markdown("## Scenario Studio")

    st.caption(
        "Plan an outreach experience before people arrive."
    )

    col1, col2 = st.columns(2)

    with col1:

        objective = st.selectbox(
            "Primary objective",
            [
                "Curiosity",
                "Scientific understanding",
                "Awe / wonder",
                "Question generation",
                "Long-term science interest",
                "Memory / retention"
            ]
        )

        audience = st.selectbox(
            "Audience",
            [
                "General public",
                "Students / youth",
                "Families",
                "Tourists",
                "Astronomy enthusiasts",
                "Mixed audience"
            ]
        )

        setting = st.text_input(
            "Setting",
            "Outdoor dark-sky telescope event"
        )

    with col2:

        sensory_context = st.text_area(
            "Sensory / environmental context",
            "Dark outdoor setting, telescope queue, "
            "moderate background conversation, acoustic music.",
            height=120
        )

        facilitator_style = st.text_area(
            "Planned facilitator style",
            "Short explanations followed by direct observation.",
            height=120
        )

    situation = st.text_area(
        "Scenario",
        "Participants are waiting to see Saturn. "
        "Some are talking while others are watching the telescope."
    )

    if st.button(
        "Generate Outreach Strategy",
        type="primary",
        use_container_width=True
    ):

        if not gemini:

            st.error(
                "Gemini is unavailable. "
                "Set GEMINI_API_KEY."
            )

        else:

            with st.spinner(
                "Building scenario model..."
            ):

                try:

                    result = generate_scenario_analysis(
                        objective,
                        setting,
                        audience,
                        sensory_context,
                        facilitator_style,
                        situation
                    )

                    st.session_state[
                        "scenario_result"
                    ] = result.model_dump()

                except Exception as exc:

                    st.error(str(exc))

    result = st.session_state.get(
        "scenario_result"
    )

    if result:

        st.markdown(
            '<div class="section">Scenario Readout</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                {html.escape(result["situation_summary"])}
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("### Engagement opportunities")

            for item in result[
                "likely_engagement_opportunities"
            ]:

                st.markdown(
                    f"• {item}"
                )

        with c2:

            st.markdown("### Potential friction")

            for item in result[
                "potential_friction_points"
            ]:

                st.markdown(
                    f"• {item}"
                )

        st.markdown(
            '<div class="section">Facilitator Strategy</div>',
            unsafe_allow_html=True
        )

        for i, item in enumerate(
            result["facilitator_strategy"],
            1
        ):

            st.markdown(
                f"""
                <div class="card">
                    <b>{i}. {html.escape(item)}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section">Three Plausible Response Pathways</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(3)

        for col, prediction in zip(
            cols,
            result["predictions"]
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="card">
                        <h4>
                        {html.escape(prediction["predicted_response"])}
                        </h4>
                        {evidence_badge("AI_INTERPRETED")}
                        <p class="small-muted">
                        <b>Mechanism:</b>
                        {html.escape(prediction["mechanism"])}
                        </p>
                        <p class="small-muted">
                        <b>Confidence:</b>
                        {html.escape(prediction["confidence"])}
                        </p>
                        <p class="small-muted">
                        <b>Uncertainty:</b>
                        {html.escape(prediction["uncertainty"])}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# 17. LIVE OUTREACH COPILOT
# ============================================================

elif page == "⚡ Live Outreach Copilot":

    st.markdown("## Live Outreach Copilot")

    events = get_events()

    if not events:

        st.warning(
            "Create an event first."
        )

        st.stop()

    event_map = get_event_map()

    selected_name = st.selectbox(
        "Active event",
        list(event_map.keys())
    )

    event = event_map[selected_name]

    active = current_interaction()

    if active is None:

        st.markdown(
            f"""
            <div class="card">
                <b>{html.escape(event.name)}</b><br>
                <span class="small-muted">
                {html.escape(event.objective)}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Start New Participant Interaction",
            type="primary",
            use_container_width=True
        ):

            interaction = create_interaction(
                event.id
            )

            st.session_state[
                "active_interaction_id"
            ] = interaction.id

            st.rerun()

        st.stop()

    st.markdown(
        f"""
        <div class="info-box">
        <b>Active interaction:</b>
        {html.escape(active.participant_code)}
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        phase = st.selectbox(
            "Interaction phase",
            [
                "Approach",
                "Waiting",
                "Introduction",
                "Direct observation",
                "Explanation",
                "Question / discussion",
                "Reflection",
                "Exit"
            ],
            index=[
                "Approach",
                "Waiting",
                "Introduction",
                "Direct observation",
                "Explanation",
                "Question / discussion",
                "Reflection",
                "Exit"
            ].index(
                active.phase
            )
        )

        if phase != active.phase:

            active.phase = phase
            db.commit()

    with col2:

        preference = st.text_input(
            "Participant-stated preference",
            active.participant_preference or "",
            placeholder=(
                "e.g. 'I'd rather look first than hear an explanation.'"
            )
        )

        if preference != (
            active.participant_preference or ""
        ):

            active.participant_preference = preference
            db.commit()

    st.markdown(
        '<div class="section">Rapid Observation Logging</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "These buttons record observations. "
        "They do not assert what the participant is thinking."
    )

    def log_observation(
        category,
        detail,
        level="OBSERVED"
    ):

        observation = Observation(
            interaction_id=active.id,
            category=category,
            detail=detail,
            evidence_level=level
        )

        db.add(observation)
        db.commit()

        st.toast(
            f"Recorded: {detail}"
        )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        if st.button(
            "👁 Observing target",
            use_container_width=True
        ):
            log_observation(
                "Attention",
                "Observed looking toward the outreach target."
            )

        if st.button(
            "↔ Attention shifts",
            use_container_width=True
        ):
            log_observation(
                "Attention",
                "Observed attention shifting away from target."
            )

    with c2:

        if st.button(
            "❓ Technical question",
            use_container_width=True
        ):
            log_observation(
                "Participation",
                "Participant asked a technical/scientific question."
            )

        if st.button(
            "💬 Follow-up question",
            use_container_width=True
        ):
            log_observation(
                "Participation",
                "Participant initiated a follow-up question."
            )

    with c3:

        if st.button(
            "🔭 Voluntary observation",
            use_container_width=True
        ):
            log_observation(
                "Engagement",
                "Participant voluntarily continued observing."
            )

        if st.button(
            "🚶 Voluntary exit",
            use_container_width=True
        ):
            log_observation(
                "Exit",
                "Participant voluntarily ended interaction."
            )

    with c4:

        if st.button(
            "🔊 Environmental noise",
            use_container_width=True
        ):
            log_observation(
                "Environment",
                "Environmental noise increased."
            )

        if st.button(
            "🌙 Reduced visual interference",
            use_container_width=True
        ):
            log_observation(
                "Environment",
                "Visual environment became less distracting."
            )

    custom_observation = st.text_input(
        "Custom observation"
    )

    if st.button(
        "Record custom observation"
    ):

        if custom_observation.strip():

            log_observation(
                "Custom",
                custom_observation.strip()
            )

    st.markdown(
        '<div class="section">Recent Interaction Timeline</div>',
        unsafe_allow_html=True
    )

    observations = (
        db.query(Observation)
        .filter(
            Observation.interaction_id == active.id
        )
        .order_by(
            Observation.timestamp.desc()
        )
        .limit(12)
        .all()
    )

    if observations:

        st.markdown(
            '<div class="timeline">',
            unsafe_allow_html=True
        )

        for obs in observations:

            st.markdown(
                f"""
                <div class="timeline-item">
                    <div class="timeline-dot"></div>
                    <div>
                        <b>{html.escape(obs.detail)}</b>
                        <div class="small-muted">
                        {html.escape(obs.category)}
                        · {obs.timestamp.strftime("%H:%M:%S")}
                        </div>
                        <div style="margin-top:5px">
                        {evidence_badge(obs.evidence_level)}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section">AI Adaptation</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "⚡ Ask Copilot What To Do Next",
        type="primary",
        use_container_width=True
    ):

        if not gemini:

            st.error(
                "Gemini is unavailable."
            )

        else:

            with st.spinner(
                "Interpreting current interaction..."
            ):

                try:

                    recommendation = (
                        generate_live_recommendation(
                            event,
                            active,
                            observations
                        )
                    )

                    st.session_state[
                        "last_recommendation"
                    ] = recommendation.model_dump()

                    adaptation = Adaptation(
                        interaction_id=active.id,
                        recommended_action=(
                            recommendation.recommended_action
                        ),
                        rationale=(
                            recommendation.rationale
                        ),
                        confidence=(
                            recommendation.confidence
                        )
                    )

                    db.add(adaptation)
                    db.commit()

                except Exception as exc:

                    st.error(str(exc))

    recommendation = st.session_state.get(
        "last_recommendation"
    )

    if recommendation:

        st.markdown(
            f"""
            <div class="success-box">
                <b>DO THIS</b><br><br>
                {html.escape(
                    recommendation["recommended_action"]
                )}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                <b>Why?</b><br><br>
                {html.escape(
                    recommendation["rationale"]
                )}
                <br><br>
                <span class="small-muted">
                Confidence:
                {html.escape(
                    recommendation["confidence"]
                )}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "### Evidence used"
        )

        for item in recommendation[
            "evidence_used"
        ]:

            st.write(
                f"• {item}"
            )

        st.markdown(
            f"""
            <div class="warning-box">
            <b>Alternative explanation</b><br><br>
            {html.escape(
                recommendation["alternative_explanation"]
            )}
            <br><br>
            <b>Watch next:</b>
            {html.escape(
                recommendation["next_signal_to_watch"]
            )}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    if st.button(
        "End Interaction & Start Next Participant"
    ):

        active.ended_at = utcnow()
        db.commit()

        st.session_state.pop(
            "active_interaction_id",
            None
        )

        st.session_state.pop(
            "last_recommendation",
            None
        )

        st.rerun()


# ============================================================
# 18. HUMAN RESPONSE MAP
# ============================================================

elif page == "◉ Human Response Map":

    st.markdown(
        "## Human Response Map"
    )

    st.caption(
        "A visual model of what happened during an interaction."
    )

    events = get_events()

    if not events:

        st.info(
            "Create an event first."
        )

        st.stop()

    event_map = get_event_map()

    selected = st.selectbox(
        "Event",
        list(event_map.keys())
    )

    event = event_map[selected]

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

        st.info(
            "No interactions recorded."
        )

        st.stop()

    rows = []

    for interaction in interactions:

        obs = interaction.observations

        rows.append({
            "Participant": interaction.participant_code,
            "Phase": interaction.phase,
            "Observations": len(obs),
            "Questions": sum(
                1
                for o in obs
                if "question" in o.detail.lower()
            ),
            "Voluntary continuation": sum(
                1
                for o in obs
                if "continued" in o.detail.lower()
            ),
            "Exit": any(
                "ended interaction" in o.detail.lower()
                or "voluntarily ended" in o.detail.lower()
                for o in obs
            )
        })

    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        '<div class="section">Engagement Signal Distribution</div>',
        unsafe_allow_html=True
    )

    signal_counts = {
        "Target observation": 0,
        "Technical question": 0,
        "Follow-up question": 0,
        "Voluntary continuation": 0,
        "Environmental friction": 0
    }

    for interaction in interactions:

        for obs in interaction.observations:

            text = obs.detail.lower()

            if "target" in text:
                signal_counts[
                    "Target observation"
                ] += 1

            if "technical" in text:
                signal_counts[
                    "Technical question"
                ] += 1

            if "follow-up" in text:
                signal_counts[
                    "Follow-up question"
                ] += 1

            if "continued" in text:
                signal_counts[
                    "Voluntary continuation"
                ] += 1

            if "noise" in text:
                signal_counts[
                    "Environmental friction"
                ] += 1

    chart_df = pd.DataFrame(
        {
            "Signal": list(
                signal_counts.keys()
            ),
            "Count": list(
                signal_counts.values()
            )
        }
    ).set_index("Signal")

    st.bar_chart(
        chart_df
    )

    st.markdown(
        """
        <div class="info-box">
        <b>Important:</b> these are engagement signals,
        not measurements of internal mental states.
        Looking away does not necessarily mean boredom.
        Silence does not necessarily mean awe.
        Asking a question does not necessarily prove learning.
        The system intentionally preserves that ambiguity.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 19. IMPACT TRAJECTORY
# ============================================================

elif page == "↗ Impact Trajectory":

    st.markdown(
        "## Impact Trajectory"
    )

    st.caption(
        "Measure what changed after the experience — "
        "without pretending that every change was caused by it."
    )

    events = get_events()

    if not events:

        st.info(
            "Create an event first."
        )

        st.stop()

    event_map = get_event_map()

    selected = st.selectbox(
        "Event",
        list(event_map.keys())
    )

    event = event_map[selected]

    interactions = (
        db.query(Interaction)
        .filter(
            Interaction.event_id == event.id
        )
        .all()
    )

    if not interactions:

        st.info(
            "No participant interactions yet."
        )

        st.stop()

    interaction_map = {
        i.participant_code: i
        for i in interactions
    }

    selected_participant = st.selectbox(
        "Participant interaction",
        list(interaction_map.keys())
    )

    interaction = interaction_map[
        selected_participant
    ]

    st.markdown(
        '<div class="section">Impact Timeline</div>',
        unsafe_allow_html=True
    )

    phases = [
        (
            "BASELINE",
            "Before",
            "What was true before the outreach?"
        ),
        (
            "IMMEDIATE",
            "Immediate",
            "What changed immediately afterward?"
        ),
        (
            "DELAYED_24H",
            "24 hours",
            "Did anything persist or continue?"
        ),
        (
            "DELAYED_7D",
            "7 days",
            "Was there later voluntary continuation?"
        ),
    ]

    for timing, title, description in phases:

        feedback = (
            db.query(ParticipantFeedback)
            .filter(
                ParticipantFeedback.interaction_id
                == interaction.id,
                ParticipantFeedback.timing
                == timing
            )
            .first()
        )

        if feedback:

            st.markdown(
                f"""
                <div class="card">
                    <h3>{title}</h3>
                    <div class="small-muted">
                    {description}
                    </div>
                    <br>
                    Curiosity:
                    <b>{feedback.curiosity or "—"}</b>
                    &nbsp;&nbsp;
                    Understanding:
                    <b>{feedback.understanding or "—"}</b>
                    &nbsp;&nbsp;
                    Salience:
                    <b>{feedback.emotional_salience or "—"}</b>
                    <br><br>
                    {evidence_badge("REPORTED")}
                    </div>
                """,
                unsafe_allow_html=True
            )

            if feedback.free_text:

                st.markdown(
                    f"""
                    <div class="card">
                    <b>Participant reflection</b><br><br>
                    "{html.escape(feedback.free_text)}"
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.markdown(
                f"""
                <div class="card"
                style="opacity:0.55">
                <h3>{title}</h3>
                <span class="small-muted">
                {description}
                </span><br><br>
                No data recorded.
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section">Optional Participant Micro-Feedback</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "This is deliberately optional. "
        "A participant should never need to complete a survey "
        "to receive the outreach experience."
    )

    with st.form(
        "feedback_form"
    ):

        timing = st.selectbox(
            "Feedback point",
            [
                "BASELINE",
                "IMMEDIATE",
                "DELAYED_24H",
                "DELAYED_7D"
            ]
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            curiosity = st.slider(
                "Curiosity",
                1.0,
                10.0,
                5.0,
                0.5
            )

        with c2:

            understanding = st.slider(
                "Understanding",
                1.0,
                10.0,
                5.0,
                0.5
            )

        with c3:

            salience = st.slider(
                "How memorable was it?",
                1.0,
                10.0,
                5.0,
                0.5
            )

        confidence = st.slider(
            "Confidence that you understood the main idea",
            1.0,
            10.0,
            5.0,
            0.5
        )

        free_text = st.text_area(
            "Optional: What stayed with you?",
            max_chars=1000
        )

        follow_up = st.selectbox(
            "Optional follow-through",
            [
                "Not asked",
                "Looked something up",
                "Talked about it",
                "Tried something related",
                "Returned / continued",
                "None"
            ]
        )

        submitted = st.form_submit_button(
            "Save Participant Feedback"
        )

        if submitted:

            feedback = ParticipantFeedback(
                interaction_id=interaction.id,
                timing=timing,
                curiosity=curiosity,
                understanding=understanding,
                emotional_salience=salience,
                confidence=confidence,
                free_text=free_text,
                follow_up_action=follow_up
            )

            db.add(feedback)
            db.commit()

            st.success(
                "Feedback recorded."
            )

    st.markdown(
        '<div class="section">Event-Level Impact</div>',
        unsafe_allow_html=True
    )

    metrics = calculate_event_metrics(
        db,
        event.id
    )

    cols = st.columns(4)

    with cols[0]:
        render_metric(
            "Participants",
            metrics.get(
                "participants",
                0
            ),
            "Recorded interactions"
        )

    with cols[1]:
        render_metric(
            "Curiosity change",
            (
                f'{metrics["curiosity_change"]:+.2f}'
                if metrics.get(
                    "curiosity_change"
                ) is not None
                else "—"
            ),
            "Immediate minus baseline"
        )

    with cols[2]:
        render_metric(
            "Understanding change",
            (
                f'{metrics["understanding_change"]:+.2f}'
                if metrics.get(
                    "understanding_change"
                ) is not None
                else "—"
            ),
            "Immediate minus baseline"
        )

    with cols[3]:
        render_metric(
            "Follow-through",
            (
                f'{metrics["follow_up_rate"]:.1f}%'
                if metrics.get(
                    "follow_up_rate"
                ) is not None
                else "—"
            ),
            "Reported voluntary continuation"
        )

    signals = build_impact_signals(
        metrics
    )

    if signals:

        st.markdown(
            '<div class="section">Impact Signals</div>',
            unsafe_allow_html=True
        )

        for signal in signals:

            st.markdown(
                f"""
                <div class="signal">
                    <div class="signal-title">
                    {html.escape(signal["name"])}
                    </div>
                    <div class="signal-meta">
                    Value: {html.escape(str(signal["value"]))}
                    · {html.escape(signal["status"])}
                    · {evidence_badge(signal["evidence"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# 20. COUNTERFACTUAL LAB
# ============================================================

elif page == "◇ Counterfactual Lab":

    st.markdown(
        "## Counterfactual Lab"
    )

    st.caption(
        "Change one variable and explore what might plausibly differ."
    )

    baseline = st.text_area(
        "Baseline outreach configuration",
        """
Outdoor telescope event.
Saturn observation.
Live acoustic ukulele.
Short scientific explanations.
Moderate background conversation.
Participant waits approximately 5 minutes.
""",
        height=180
    )

    changed = st.text_input(
        "Change ONE variable",
        "Remove live music during direct telescope observation."
    )

    if st.button(
        "Run Counterfactual",
        type="primary",
        use_container_width=True
    ):

        if not gemini:

            st.error(
                "Gemini is unavailable."
            )

        else:

            with st.spinner(
                "Exploring divergence..."
            ):

                try:

                    result = generate_counterfactual(
                        baseline,
                        changed
                    )

                    st.session_state[
                        "counterfactual"
                    ] = result.model_dump()

                except Exception as exc:

                    st.error(str(exc))

    result = st.session_state.get(
        "counterfactual"
    )

    if result:

        st.markdown(
            '<div class="section">Divergence</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="card">
                <b>Changed variable</b><br>
                {html.escape(result["changed_variable"])}
                <br><br>
                <b>Expected difference</b><br>
                {html.escape(result["expected_difference"])}
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown("### Possible benefit")

            st.write(
                result["possible_benefit"]
            )

        with c2:

            st.markdown("### Possible cost")

            st.write(
                result["possible_cost"]
            )

        st.markdown(
            f"""
            <div class="info-box">
            <b>Confidence:</b>
            {html.escape(result["confidence"])}
            <br><br>
            <b>How to test it:</b><br>
            {html.escape(result["what_would_test_this"])}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 21. EQUFINALITY ENGINE
# ============================================================

elif page == "∿ Equifinality Engine":

    st.markdown(
        "## Equifinality Engine"
    )

    st.caption(
        "One observed behavior can have many possible explanations."
    )

    situation = st.text_area(
        "Situation",
        "Participant observes Saturn through a telescope "
        "and remains silent afterward."
    )

    observed = st.text_area(
        "Observed behavior",
        "Participant remains silent for approximately 30 seconds "
        "and then asks a detailed question about planetary formation."
    )

    context = st.text_area(
        "Known context",
        "First telescope observation. Participant voluntarily "
        "joined the queue."
    )

    if st.button(
        "Generate Three Explanations",
        type="primary",
        use_container_width=True
    ):

        if not gemini:

            st.error(
                "Gemini unavailable."
            )

        else:

            with st.spinner(
                "Generating divergent explanations..."
            ):

                try:

                    result = generate_equifinality(
                        situation,
                        observed,
                        context
                    )

                    st.session_state[
                        "equifinality"
                    ] = result.model_dump()

                except Exception as exc:

                    st.error(str(exc))

    result = st.session_state.get(
        "equifinality"
    )

    if result:

        st.markdown(
            f"""
            <div class="warning-box">
            <b>Structural ambiguity</b><br><br>
            {html.escape(result["ambiguity_statement"])}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section">Three Divergent Hypotheses</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(3)

        for col, explanation in zip(
            cols,
            result["explanations"]
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="card">
                        <h3>
                        {html.escape(
                            explanation["explanation_name"]
                        )}
                        </h3>

                        <b>
                        Compatibility:
                        {html.escape(
                            explanation["compatibility"]
                        )}
                        </b>

                        <br><br>

                        <b>Possible mechanism</b><br>
                        <span class="small-muted">
                        {html.escape(
                            explanation["possible_mechanism"]
                        )}
                        </span>

                        <br><br>

                        <b>Evidence</b><br>
                        <span class="small-muted">
                        {"<br>".join(
                            html.escape(x)
                            for x in explanation[
                                "supporting_evidence"
                            ]
                        )}
                        </span>

                        <br><br>

                        <b>Missing information</b><br>
                        <span class="small-muted">
                        {"<br>".join(
                            html.escape(x)
                            for x in explanation[
                                "missing_information"
                            ]
                        )}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
            f"""
            <div class="info-box">
            <b>Best next observation</b><br><br>
            {html.escape(result["best_next_observation"])}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "### What cannot be inferred"
        )

        for item in result[
            "cannot_be_inferred"
        ]:

            st.write(
                f"• {item}"
            )


# ============================================================
# 22. EVENT OBSERVATORY
# ============================================================

elif page == "◎ Event Observatory":

    st.markdown(
        "## Event Observatory"
    )

    events = get_events()

    if not events:

        st.info(
            "No events yet."
        )

        st.stop()

    event_map = get_event_map()

    selected = st.selectbox(
        "Event",
        list(event_map.keys())
    )

    event = event_map[selected]

    metrics = calculate_event_metrics(
        db,
        event.id
    )

    cols = st.columns(5)

    with cols[0]:
        render_metric(
            "Participants",
            metrics.get(
                "participants",
                0
            )
        )

    with cols[1]:
        render_metric(
            "Observations",
            metrics.get(
                "observations",
                0
            )
        )

    with cols[2]:
        render_metric(
            "Adaptations",
            metrics.get(
                "adaptations",
                0
            )
        )

    with cols[3]:
        render_metric(
            "Curiosity Δ",
            (
                f'{metrics["curiosity_change"]:+.2f}'
                if metrics.get(
                    "curiosity_change"
                ) is not None
                else "—"
            )
        )

    with cols[4]:
        render_metric(
            "Understanding Δ",
            (
                f'{metrics["understanding_change"]:+.2f}'
                if metrics.get(
                    "understanding_change"
                ) is not None
                else "—"
            )
        )

    st.markdown(
        '<div class="section">Event Description</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card">
            <b>Objective:</b>
            {html.escape(event.objective)}
            <br><br>
            <b>Setting:</b>
            {html.escape(event.setting or "—")}
            <br><br>
            <b>Acoustic setting:</b>
            {html.escape(event.acoustic_setting or "—")}
            <br><br>
            <b>Audience:</b>
            {html.escape(event.target_audience or "—")}
            <br><br>
            {html.escape(event.description or "")}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section">Impact Interpretation</div>',
        unsafe_allow_html=True
    )

    qualitative = []

    interactions = (
        db.query(Interaction)
        .filter(
            Interaction.event_id == event.id
        )
        .all()
    )

    for interaction in interactions:

        for feedback in interaction.feedback:

            if feedback.free_text:

                qualitative.append({
                    "participant": (
                        interaction.participant_code
                    ),
                    "timing": feedback.timing,
                    "text": feedback.free_text
                })

    if st.button(
        "Generate Event Impact Interpretation",
        type="primary",
        use_container_width=True
    ):

        if not gemini:

            st.error(
                "Gemini unavailable."
            )

        else:

            with st.spinner(
                "Interpreting event-level evidence..."
            ):

                try:

                    interpretation = (
                        generate_impact_interpretation(
                            metrics,
                            qualitative
                        )
                    )

                    st.session_state[
                        "impact_interpretation"
                    ] = interpretation.model_dump()

                except Exception as exc:

                    st.error(str(exc))

    interpretation = st.session_state.get(
        "impact_interpretation"
    )

    if interpretation:

        st.markdown(
            f"""
            <div class="card">
                <b>Impact summary</b><br><br>
                {html.escape(
                    interpretation["impact_summary"]
                )}
            </div>
            """,
            unsafe_allow_html=True
        )

        c1, c2 = st.columns(2)

        with c1:

            st.markdown(
                "### Strongest positive signal"
            )

            st.success(
                interpretation[
                    "strongest_positive_signal"
                ]
            )

        with c2:

            st.markdown(
                "### Weakest / uncertain signal"
            )

            st.warning(
                interpretation[
                    "weakest_or_uncertain_signal"
                ]
            )

        st.markdown(
            "### Possible mechanisms"
        )

        for item in interpretation[
            "possible_mechanisms"
        ]:

            st.write(
                f"• {item}"
            )

        st.markdown(
            "### Alternative explanations"
        )

        for item in interpretation[
            "alternative_explanations"
        ]:

            st.write(
                f"• {item}"
            )

        st.markdown(
            f"""
            <div class="info-box">
            <b>Recommended next measurement</b><br><br>
            {html.escape(
                interpretation[
                    "recommended_next_measurement"
                ]
            )}
            <br><br>
            <b>Causal claim strength:</b>
            {html.escape(
                interpretation[
                    "causal_claim_strength"
                ]
            )}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section">Participant Memory Themes</div>',
        unsafe_allow_html=True
    )

    texts = [
        item["text"]
        for item in qualitative
        if item.get("text")
    ]

    if texts:

        if st.button(
            "Analyze Memory Themes"
        ):

            if gemini:

                with st.spinner(
                    "Analyzing qualitative material..."
                ):

                    try:

                        theme_result = (
                            generate_theme_analysis(
                                texts
                            )
                        )

                        st.session_state[
                            "theme_result"
                        ] = theme_result.model_dump()

                    except Exception as exc:

                        st.error(str(exc))

            else:

                st.error(
                    "Gemini unavailable."
                )

        theme_result = st.session_state.get(
            "theme_result"
        )

        if theme_result:

            st.markdown(
                "#### Recurring themes"
            )

            for item in theme_result[
                "themes"
            ]:

                st.write(
                    f"• {item}"
                )

            st.markdown(
                "#### Memorable elements"
            )

            for item in theme_result[
                "memorable_elements"
            ]:

                st.write(
                    f"• {item}"
                )

            st.markdown(
                "#### Curiosity signals"
            )

            for item in theme_result[
                "curiosity_signals"
            ]:

                st.write(
                    f"• {item}"
                )

    else:

        st.caption(
            "No participant-written reflections available."
        )

    st.markdown(
        '<div class="section">Export</div>',
        unsafe_allow_html=True
    )

    export_rows = []

    for interaction in interactions:

        for feedback in interaction.feedback:

            export_rows.append({
                "participant_code": (
                    interaction.participant_code
                ),
                "timing": feedback.timing,
                "curiosity": feedback.curiosity,
                "understanding": feedback.understanding,
                "emotional_salience": (
                    feedback.emotional_salience
                ),
                "confidence": feedback.confidence,
                "follow_up": (
                    feedback.follow_up_action
                ),
                "free_text": feedback.free_text
            })

    export_df = pd.DataFrame(
        export_rows
    )

    csv_data = export_df.to_csv(
        index=False
    )

    st.download_button(
        "Download Event CSV",
        data=csv_data,
        file_name=(
            f"outreach_{event.id[:8]}.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# 23. METHODOLOGY
# ============================================================

elif page == "☷ Methodology":

    st.markdown(
        "## Methodology"
    )

    st.markdown(
        """
        <div class="info-box">
        <b>The central principle:</b>
        the application should be useful without pretending
        to know more than the evidence supports.
        </div>
        """,
        unsafe_allow_html=True
    )

    sections = {

        "1. Observation is not mind-reading":
        """
        A facilitator can observe that somebody looked away,
        asked a question, continued watching, left, smiled,
        became quiet, or interacted with another person.

        Those observations do not establish why the behavior occurred.
        """,

        "2. Participant report is a separate evidence class":
        """
        If a participant says "I loved seeing Saturn",
        that is participant-reported evidence.

        It should not be rewritten as:
        "The participant experienced profound awe."

        The first is evidence.
        The second is interpretation.
        """,

        "3. Computation stays deterministic":
        """
        Changes, averages, rates and counts are calculated by
        Python rather than delegated to the language model.

        Gemini is therefore not responsible for arithmetic.
        """,

        "4. Gemini is an interpretation layer":
        """
        Gemini can synthesize observations, identify plausible
        mechanisms, propose alternative explanations and suggest
        facilitator actions.

        It cannot establish psychological facts from sparse behavior.
        """,

        "5. Impact is longitudinal":
        """
        A successful outreach experience should not be reduced to
        whether somebody clicked a smiley face immediately afterward.

        Useful downstream signals can include:

        • increased curiosity
        • improved self-reported understanding
        • memorable concepts
        • questions generated after the event
        • voluntary science-seeking
        • talking about the experience
        • returning
        • attempting something related
        """,

        "6. Causality is deliberately constrained":
        """
        If curiosity increases after an event, the system reports
        a change.

        It does not automatically report:

        "The outreach caused the increase."

        Other explanations may exist:
        prior interest, social influence, novelty, another experience,
        selection effects, measurement effects, etc.
        """,

        "7. Equifinality":
        """
        One behavior can emerge from multiple pathways.

        Silence could represent concentration, uncertainty,
        awe, fatigue, social hesitation, or simply personal style.

        The system therefore actively generates alternative explanations.
        """,

        "8. Counterfactuals are hypotheses":
        """
        If removing music might improve direct observation,
        that is a hypothesis.

        It becomes stronger only when tested repeatedly under
        appropriate conditions.
        """,

        "9. Autonomy":
        """
        The participant is never treated as an optimization target.

        Recommendations should remain minimally disruptive,
        reversible, and respectful.

        The system should be allowed to recommend:
        "Do nothing."
        """
    }

    for title, text in sections.items():

        st.markdown(
            f"""
            <div class="card">
                <h3>{html.escape(title)}</h3>
                <div style="white-space:pre-line;
                color:#b6bac4;
                line-height:1.7">
                {html.escape(text)}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 24. EVENT CREATION — SIDEBAR UTILITY
# ============================================================

with st.sidebar:

    st.divider()

    with st.expander(
        "＋ Create Outreach Event"
    ):

        event_name = st.text_input(
            "Event name",
            key="new_event_name"
        )

        event_objective = st.selectbox(
            "Objective",
            [
                "Curiosity",
                "Scientific understanding",
                "Awe / wonder",
                "Question generation",
                "Long-term science interest",
                "Memory / retention"
            ],
            key="new_event_objective"
        )

        event_setting = st.text_input(
            "Setting",
            "Outdoor astronomy outreach",
            key="new_event_setting"
        )

        event_acoustic = st.selectbox(
            "Acoustic setting",
            [
                "None",
                "Ambient sound",
                "Live acoustic music",
                "Storytelling + music"
            ],
            key="new_event_acoustic"
        )

        event_audience = st.selectbox(
            "Audience",
            [
                "General public",
                "Students / youth",
                "Families",
                "Tourists",
                "Astronomy enthusiasts",
                "Mixed"
            ],
            key="new_event_audience"
        )

        event_description = st.text_area(
            "Description",
            key="new_event_description"
        )

        if st.button(
            "Initialize Event",
            use_container_width=True
        ):

            if not event_name.strip():

                st.error(
                    "Event name required."
                )

            else:

                event = Event(
                    name=event_name.strip(),
                    objective=event_objective,
                    setting=event_setting,
                    acoustic_setting=event_acoustic,
                    target_audience=event_audience,
                    description=event_description
                )

                db.add(event)
                db.commit()

                st.success(
                    "Event created."
                )

                st.rerun()


# ============================================================
# 25. FOOTER
# ============================================================

st.markdown(
    """
    <br><br>
    <div style="
        text-align:center;
        color:#555b66;
        border-top:1px solid #20232a;
        padding:25px;
        font-size:0.78rem;
        line-height:1.6;
    ">
        <b>Ninolades Outreach Intelligence Lab</b>
        · Predictive Generative Architecture
        · Human-Centered Science Communication
        <br><br>
        This system is for exploratory, educational and outreach-support
        purposes. AI-generated interpretations are hypotheses rather than
        psychological, clinical or causal determinations. Participant
        autonomy and uncertainty should take precedence over optimization.
        <br><br>
        Version {APP_VERSION}
    </div>
    """.format(
        APP_VERSION=APP_VERSION
    ),
    unsafe_allow_html=True
)


# ============================================================
# 26. CLEANUP
# ============================================================

db.close()
