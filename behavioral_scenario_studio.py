import os
import uuid
import json
import math
from datetime import datetime, timezone
from typing import List, Literal, Optional

import pandas as pd
import streamlit as st

from sqlalchemy import (
    create_engine,
    Column,
    String,
    DateTime,
    Text,
    Float,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ninolades Outreach Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GEMINI MODEL CONFIGURATION
# ============================================================

MODEL_OPTIONS = {
    "Gemini 3.6 Flash — Recommended": {
        "id": "gemini-3.6-flash",
        "description": (
            "Balanced reasoning, speed, and responsiveness. "
            "Recommended for most outreach sessions."
        ),
    },
    "Gemini 3.1 Pro — Deep Analysis": {
        "id": "gemini-3.1-pro-preview",
        "description": (
            "Higher-depth reasoning for complex interpretation, "
            "competing explanations, and difficult outreach scenarios."
        ),
    },
    "Gemini 3.5 Flash-Lite — Fast": {
        "id": "gemini-3.5-flash-lite",
        "description": (
            "Fast and economical processing for rapid or high-volume "
            "outreach interactions."
        ),
    },
}


# ============================================================
# PREMIUM MINIMALIST DESIGN SYSTEM
# ============================================================

PREMIUM_CSS = """
<style>

:root {
    --bg: #0b0b0c;
    --surface: #111113;
    --surface-2: #151517;
    --surface-3: #1a1a1d;
    --border: #27272a;
    --border-soft: #202023;

    --text: #f5f5f7;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;

    --accent: #4f83f5;
    --accent-soft: rgba(79, 131, 245, 0.12);

    --positive: #5bd68a;
    --warning: #e7b85b;
    --negative: #e06b6b;

    --radius: 14px;
}

/* Global */

.stApp {
    background:
        radial-gradient(
            circle at top left,
            rgba(79,131,245,0.035),
            transparent 32%
        ),
        var(--bg);
    color: var(--text);
}

html,
body,
[class*="css"] {
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "SF Pro Text",
        "Helvetica Neue",
        Arial,
        sans-serif;
}

.block-container {
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    max-width: 1500px;
}

/* Typography */

h1 {
    font-size: 2.35rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.04em !important;
}

h2 {
    font-size: 1.65rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.03em !important;
}

h3 {
    font-size: 1.25rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em !important;
}

h4 {
    font-weight: 500 !important;
}

p {
    color: var(--text-secondary);
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #0d0d0f;
    border-right: 1px solid var(--border-soft);
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

section[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-secondary);
}

/* Inputs */

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"] {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    box-shadow: none !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stSelectbox div[data-baseweb="select"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent-soft) !important;
}

/* Sliders */

.stSlider [data-baseweb="slider"] div {
    background-color: var(--accent) !important;
}

/* Buttons */

.stButton button {
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 9px;
    min-height: 42px;
    font-weight: 500;
    transition:
        background 0.18s ease,
        border-color 0.18s ease,
        transform 0.18s ease;
}

.stButton button:hover {
    background: var(--surface-3);
    border-color: #3a3a3f;
    color: var(--text);
    transform: translateY(-1px);
}

button[data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: white !important;
}

button[data-testid="baseButton-primary"]:hover {
    background: #3f73dd !important;
    border-color: #3f73dd !important;
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 30px;
    border-bottom: 1px solid var(--border-soft);
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    padding: 13px 0;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    color: var(--text) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Cards */

.premium-card {
    background: linear-gradient(
        145deg,
        #151517,
        #111113
    );
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 18px;
    box-shadow:
        0 12px 40px rgba(0,0,0,0.22);
}

.section-title {
    color: var(--text);
    font-size: 1.25rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin: 20px 0 20px 0;
}

.eyebrow {
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.muted {
    color: var(--text-muted);
    font-size: 0.9rem;
}

.metric-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    min-height: 120px;
}

.metric-label {
    color: var(--text-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 10px;
}

.metric-value {
    color: var(--text);
    font-size: 1.85rem;
    font-weight: 500;
    letter-spacing: -0.035em;
}

.metric-detail {
    color: var(--text-secondary);
    font-size: 0.82rem;
    margin-top: 5px;
}

/* Progress */

.progress-background {
    background: #242428;
    border-radius: 99px;
    height: 7px;
    width: 100%;
    overflow: hidden;
}

.progress-fill {
    background: var(--accent);
    height: 100%;
    border-radius: 99px;
}

/* Evidence */

.evidence {
    background: #101012;
    border: 1px solid var(--border-soft);
    border-radius: 9px;
    padding: 14px 16px;
    margin-bottom: 9px;
}

.evidence-title {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 5px;
}

.evidence-body {
    color: #d4d4d8;
    font-size: 0.9rem;
    line-height: 1.55;
}

.observed {
    border-left: 3px solid #a78bfa;
}

.interpreted {
    border-left: 3px solid var(--accent);
}

.speculative {
    border-left: 3px solid #d99b54;
}

/* Status */

.status {
    display: inline-block;
    border-radius: 7px;
    padding: 5px 9px;
    font-size: 0.74rem;
    font-weight: 600;
}

.status-positive {
    color: var(--positive);
    border: 1px solid rgba(91,214,138,0.25);
    background: rgba(91,214,138,0.08);
}

.status-warning {
    color: var(--warning);
    border: 1px solid rgba(231,184,91,0.25);
    background: rgba(231,184,91,0.08);
}

.status-negative {
    color: var(--negative);
    border: 1px solid rgba(224,107,107,0.25);
    background: rgba(224,107,107,0.08);
}

/* Alerts */

div[data-testid="stAlert"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Tables */

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

/* Hide Streamlit branding */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ============================================================
# DATABASE
# ============================================================

DB_PATH = os.getenv(
    "DB_PATH",
    "sqlite:///ninolades_outreach.db"
)

engine = create_engine(
    DB_PATH,
    connect_args={"check_same_thread": False}
    if DB_PATH.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class Event(Base):
    __tablename__ = "events"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    name = Column(String, nullable=False)
    date = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    primary_objective = Column(String)
    acoustic_setting = Column(String)
    environment = Column(String)
    audience = Column(String)
    description = Column(Text)

    interactions = relationship(
        "Interaction",
        back_populates="event",
        cascade="all, delete-orphan",
    )


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    event_id = Column(
        String,
        ForeignKey("events.id"),
        nullable=False,
    )

    participant_id = Column(
        String,
        default=lambda: str(uuid.uuid4()),
    )

    timestamp_start = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    timestamp_end = Column(DateTime)

    phase = Column(
        String,
        default="Approach",
    )

    stated_preference = Column(
        String,
        nullable=True,
    )

    event = relationship(
        "Event",
        back_populates="interactions",
    )

    observations = relationship(
        "Observation",
        back_populates="interaction",
        cascade="all, delete-orphan",
    )

    surveys = relationship(
        "Survey",
        back_populates="interaction",
        cascade="all, delete-orphan",
    )


class Observation(Base):
    __tablename__ = "observations"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    interaction_id = Column(
        String,
        ForeignKey("interactions.id"),
        nullable=False,
    )

    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
    )

    category = Column(String, nullable=False)

    detail = Column(String, nullable=False)

    evidence_level = Column(
        String,
        default="OBSERVED",
    )

    interaction = relationship(
        "Interaction",
        back_populates="observations",
    )


class Survey(Base):
    __tablename__ = "surveys"

    id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    interaction_id = Column(
        String,
        ForeignKey("interactions.id"),
        nullable=False,
    )

    timing = Column(
        String,
        nullable=False,
    )

    curiosity_score = Column(
        Float,
        nullable=True,
    )

    knowledge_score = Column(
        Float,
        nullable=True,
    )

    memorability_text = Column(
        Text,
        nullable=True,
    )

    follow_through = Column(
        String,
        nullable=True,
    )

    interaction = relationship(
        "Interaction",
        back_populates="surveys",
    )


Base.metadata.create_all(bind=engine)


# ============================================================
# DATABASE HELPER
# ============================================================

def get_db():
    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


db = next(get_db())


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_ai_client():

    api_key = (
        st.secrets.get("GEMINI_API_KEY", None)
        if hasattr(st, "secrets")
        else None
    )

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    try:
        return genai.Client(
            api_key=api_key
        )

    except Exception:
        return None


ai_client = get_ai_client()


# ============================================================
# PYDANTIC AI SCHEMAS
# ============================================================

class OutreachRecommendation(BaseModel):

    recommended_action: str = Field(
        description=(
            "The most appropriate immediate outreach action. "
            "May be 'No intervention' when the participant appears engaged."
        )
    )

    rationale: str = Field(
        description=(
            "Evidence-grounded explanation for the recommendation."
        )
    )

    confidence: Literal[
        "Low",
        "Moderate",
        "High",
    ]

    evidence: List[str] = Field(
        description=(
            "Specific observed or participant-stated signals "
            "used to reach the recommendation."
        )
    )

    alternative_explanation: str = Field(
        description=(
            "A plausible alternative interpretation that prevents "
            "overconfidence."
        )
    )

    next_observation: str = Field(
        description=(
            "The next observable signal that would help determine "
            "whether the recommendation is working."
        )
    )


class QualitativeTheme(BaseModel):

    theme: str

    evidence: List[str]

    confidence: Literal[
        "Low",
        "Moderate",
        "High",
    ]


class ImpactInterpretation(BaseModel):

    summary: str = Field(
        description="Concise interpretation of the measured engagement impact."
    )

    strongest_signal: str = Field(
        description="The strongest observed impact signal."
    )

    weakest_signal: str = Field(
        description="The weakest or least reliable signal."
    )

    plausible_mechanisms: List[str] = Field(
        min_length=2,
        max_length=4,
        description="Plausible mechanisms that could explain the observed pattern."
    )

    limitations: List[str] = Field(
        min_length=2,
        max_length=5,
        description="Important limitations and alternative explanations."
    )

    next_measurement: str = Field(
        description="Most useful next measurement to improve confidence."
    )


# ============================================================
# GEMINI SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTION = """
You are the interpretation layer of the Ninolades Outreach Intelligence
platform.

Your role is to help science communicators improve real-world outreach.

CORE PRINCIPLES:

1. Treat all participant characteristics as hypothetical or uncertain
   unless directly stated by the participant.

2. Never diagnose autism, ADHD, anxiety, personality disorders, or any
   other medical, psychiatric, neurological, or clinical condition.

3. Never claim to know a participant's internal mental state from behavior.

4. Distinguish clearly between:
   - DIRECT: participant explicitly stated something.
   - OBSERVED: facilitator directly observed something.
   - INFERRED: reasonable interpretation of observations.
   - HYPOTHESIS: speculative explanation.

5. Participant-stated preferences take priority over inferred preferences.

6. Do not pathologize silence, eye contact, movement, facial expression,
   social behavior, or communication style.

7. No intervention is often the correct intervention when engagement
   appears healthy.

8. Prefer minimally disruptive, reversible outreach actions.

9. Do not manipulate participants.

10. Recommendations should optimize:
    - comprehension
    - curiosity
    - autonomy
    - comfort
    - meaningful engagement
    - scientific understanding

11. Always acknowledge uncertainty.

12. Do not treat model-generated probabilities as empirical probabilities.

13. Do not infer stable personality traits from a small number of behaviors.

14. Focus on the interaction and the outreach environment rather than
    labeling the person.

15. If evidence is insufficient, explicitly say so.

16. Never fabricate evidence.
"""


# ============================================================
# GEMINI ADAPTATION
# ============================================================

def get_adaptation(
    context_data: dict,
    client,
    model_name: str,
) -> OutreachRecommendation:

    prompt = f"""
You are assisting a science outreach facilitator in real time.

EVENT OBJECTIVE:
{context_data.get("objectives", "Unknown")}

CURRENT PHASE:
{context_data.get("phase", "Unknown")}

PARTICIPANT-STATED PREFERENCE:
{context_data.get("preference", "Not provided")}

RECENT OBSERVATIONS:
{json.dumps(context_data.get("observations", []), indent=2)}

TASK:

Recommend the most appropriate next outreach action.

Prioritize direct evidence and participant autonomy.

Do not diagnose or infer hidden psychological traits.

If the participant appears adequately engaged, "No intervention" may be
the best recommendation.

Provide one concrete next action and explain why.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=OutreachRecommendation,
        ),
    )

    if hasattr(response, "parsed") and response.parsed:
        return response.parsed

    return OutreachRecommendation.model_validate_json(
        response.text
    )


# ============================================================
# GEMINI MEMORY THEME EXTRACTION
# ============================================================

def extract_theme(
    memory_text: str,
    client,
    model_name: str,
) -> QualitativeTheme:

    prompt = f"""
Analyze this participant's response:

"{memory_text}"

Identify the primary memory or engagement theme.

Do not infer personality, diagnosis, intelligence, or hidden psychology.

Only describe what the response itself supports.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=QualitativeTheme,
        ),
    )

    if hasattr(response, "parsed") and response.parsed:
        return response.parsed

    return QualitativeTheme.model_validate_json(
        response.text
    )


# ============================================================
# GEMINI IMPACT INTERPRETATION
# ============================================================

def interpret_impact(
    metrics: dict,
    client,
    model_name: str,
) -> ImpactInterpretation:

    prompt = f"""
Interpret the following deterministic outreach measurements.

MEASUREMENTS:

{json.dumps(metrics, indent=2)}

Explain the pattern conservatively.

Do not claim causality.

Do not treat aggregate measurements as psychological diagnoses.

Distinguish:
- what the numbers directly show
- plausible explanations
- what remains unknown

Identify the strongest useful signal and the most useful next measurement.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=ImpactInterpretation,
        ),
    )

    if hasattr(response, "parsed") and response.parsed:
        return response.parsed

    return ImpactInterpretation.model_validate_json(
        response.text
    )


# ============================================================
# EVENT ANALYTICS
# ============================================================

def calculate_impact_fingerprint(
    db,
    event_id: str,
):

    query = (
        db.query(Survey)
        .join(Interaction)
        .filter(Interaction.event_id == event_id)
    )

    df = pd.read_sql(
        query.statement,
        db.bind,
    )

    if df.empty:
        return None

    participant_count = (
        df["interaction_id"]
        .nunique()
    )

    baseline = df[
        df["timing"] == "BASELINE"
    ]

    immediate = df[
        df["timing"] == "IMMEDIATE"
    ]

    delayed = df[
        df["timing"].isin(
            [
                "DELAYED_24H",
                "DELAYED_7D",
            ]
        )
    ]

    baseline_curiosity = (
        baseline["curiosity_score"].mean()
        if not baseline.empty
        else None
    )

    post_curiosity = (
        immediate["curiosity_score"].mean()
        if not immediate.empty
        else None
    )

    baseline_knowledge = (
        baseline["knowledge_score"].mean()
        if not baseline.empty
        else None
    )

    post_knowledge = (
        immediate["knowledge_score"].mean()
        if not immediate.empty
        else None
    )

    curiosity_change = (
        post_curiosity - baseline_curiosity
        if baseline_curiosity is not None
        and post_curiosity is not None
        else None
    )

    knowledge_change = (
        post_knowledge - baseline_knowledge
        if baseline_knowledge is not None
        and post_knowledge is not None
        else None
    )

    follow_through_rate = None

    if (
        not delayed.empty
        and "follow_through" in delayed.columns
    ):

        valid_follow = delayed[
            delayed["follow_through"].isin(
                ["Yes", "No"]
            )
        ]

        if not valid_follow.empty:

            follow_through_rate = (
                (
                    valid_follow[
                        "follow_through"
                    ] == "Yes"
                ).mean()
                * 100
            )

    return {
        "total_participants": int(
            participant_count
        ),

        "baseline_curiosity": (
            round(baseline_curiosity, 2)
            if baseline_curiosity is not None
            else None
        ),

        "post_curiosity": (
            round(post_curiosity, 2)
            if post_curiosity is not None
            else None
        ),

        "curiosity_change": (
            round(curiosity_change, 2)
            if curiosity_change is not None
            else None
        ),

        "baseline_knowledge": (
            round(baseline_knowledge, 2)
            if baseline_knowledge is not None
            else None
        ),

        "post_knowledge": (
            round(post_knowledge, 2)
            if post_knowledge is not None
            else None
        ),

        "knowledge_change": (
            round(knowledge_change, 2)
            if knowledge_change is not None
            else None
        ),

        "follow_through_rate": (
            round(follow_through_rate, 2)
            if follow_through_rate is not None
            else None
        ),
    }


# ============================================================
# SESSION STATE
# ============================================================

SESSION_DEFAULTS = {
    "current_interaction": None,
    "last_recommendation": None,
    "selected_event_id": None,
}

for key, value in SESSION_DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        margin-bottom: 34px;
    ">
        <div class="eyebrow">
            Ninolades Intelligence
        </div>

        <h1>
            Outreach Intelligence
        </h1>

        <p style="
            font-size: 1.05rem;
            max-width: 760px;
            line-height: 1.6;
            color: #a1a1aa;
        ">
            A real-time system for designing, adapting, and measuring
            meaningful science engagement.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size: 1.1rem;
            font-weight: 500;
            color: #f5f5f7;
            margin-bottom: 24px;
        ">
            Outreach Intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="eyebrow">
            AI Engine
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_choice = st.selectbox(
        "Reasoning engine",
        list(MODEL_OPTIONS.keys()),
        index=0,
    )

    selected_model = MODEL_OPTIONS[
        model_choice
    ]["id"]

    st.caption(
        MODEL_OPTIONS[
            model_choice
        ]["description"]
    )

    st.markdown("---")

    page = st.radio(
        "Workspace",
        [
            "Event Builder",
            "Live Copilot",
            "Impact & Surveys",
            "Impact Observatory",
        ],
    )

    st.markdown("---")

    st.caption(
        "AI-generated interpretations are hypotheses, "
        "not measurements of internal psychological states."
    )


# ============================================================
# EVENT BUILDER
# ============================================================

if page == "Event Builder":

    st.markdown(
        """
        <div class="section-title">
            Event Builder
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <p class="muted">
            Define the outreach environment before participants arrive.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.form("event_builder"):

        name = st.text_input(
            "Event name",
            placeholder="e.g. Saturn Under the Southern Sky",
        )

        objective = st.selectbox(
            "Primary objective",
            [
                "Curiosity",
                "Scientific understanding",
                "Awe and wonder",
                "Long-term retention",
                "Scientific participation",
            ],
        )

        acoustic = st.selectbox(
            "Acoustic setting",
            [
                "No music",
                "Ambient music",
                "Live acoustic music",
                "Storytelling with music",
                "Variable / experimental",
            ],
        )

        environment = st.text_input(
            "Environment",
            placeholder=(
                "e.g. Dark-sky lawn, telescope station, "
                "moderate crowd"
            ),
        )

        audience = st.selectbox(
            "Audience",
            [
                "General public",
                "Students / youth",
                "Families",
                "Eco-tourists",
                "Astronomy enthusiasts",
                "Mixed audience",
            ],
        )

        description = st.text_area(
            "Event description",
            placeholder=(
                "Describe the experience, scientific content, "
                "setting, and intended engagement."
            ),
        )

        submitted = st.form_submit_button(
            "Create Event",
            type="primary",
            use_container_width=True,
        )

        if submitted:

            if not name.strip():

                st.error(
                    "An event name is required."
                )

            else:

                event = Event(
                    name=name.strip(),
                    primary_objective=objective,
                    acoustic_setting=acoustic,
                    environment=environment,
                    audience=audience,
                    description=description,
                )

                db.add(event)
                db.commit()

                st.success(
                    "Event created successfully."
                )

    st.markdown(
        "<div class='section-title'>Existing Events</div>",
        unsafe_allow_html=True,
    )

    events = (
        db.query(Event)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.info(
            "No events have been created yet."
        )

    else:

        for event in events:

            interactions_count = (
                db.query(Interaction)
                .filter(
                    Interaction.event_id
                    == event.id
                )
                .count()
            )

            st.markdown(
                f"""
                <div class="premium-card">

                    <div class="eyebrow">
                        Outreach Event
                    </div>

                    <h3>
                        {event.name}
                    </h3>

                    <p>
                        {event.description or "No description provided."}
                    </p>

                    <div style="
                        display:grid;
                        grid-template-columns:
                        repeat(3,1fr);
                        gap:12px;
                    ">

                        <div class="metric-card">
                            <div class="metric-label">
                                Objective
                            </div>
                            <div style="color:#f5f5f7;">
                                {event.primary_objective}
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-label">
                                Audience
                            </div>
                            <div style="color:#f5f5f7;">
                                {event.audience}
                            </div>
                        </div>

                        <div class="metric-card">
                            <div class="metric-label">
                                Interactions
                            </div>
                            <div class="metric-value">
                                {interactions_count}
                            </div>
                        </div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# LIVE COPILOT
# ============================================================

elif page == "Live Copilot":

    st.markdown(
        """
        <div class="section-title">
            Live Outreach Copilot
        </div>

        <p class="muted">
            Record what is actually happening. The AI recommends
            minimally disruptive next steps without treating
            observations as diagnoses.
        </p>
        """,
        unsafe_allow_html=True,
    )

    events = (
        db.query(Event)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.info(
            "Create an event before starting live outreach."
        )
        st.stop()

    event_dict = {
        e.name: e
        for e in events
    }

    selected_event_name = st.selectbox(
        "Active event",
        list(event_dict.keys()),
    )

    event_record = event_dict[
        selected_event_name
    ]

    if (
        st.session_state.current_interaction
        is None
    ):

        interaction = Interaction(
            event_id=event_record.id
        )

        db.add(interaction)
        db.commit()

        st.session_state.current_interaction = (
            interaction.id
        )

    current_id = (
        st.session_state.current_interaction
    )

    interaction = (
        db.query(Interaction)
        .filter(
            Interaction.id == current_id
        )
        .first()
    )

    if (
        interaction is None
        or interaction.event_id != event_record.id
    ):

        interaction = Interaction(
            event_id=event_record.id
        )

        db.add(interaction)
        db.commit()

        st.session_state.current_interaction = (
            interaction.id
        )

        current_id = interaction.id

    st.markdown(
        f"""
        <div class="premium-card">

            <div class="eyebrow">
                Active Interaction
            </div>

            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
            ">

                <div>
                    <div style="
                        color:#f5f5f7;
                        font-size:1.1rem;
                    ">
                        Participant session
                    </div>

                    <div class="muted">
                        Session {current_id[:8]}
                    </div>
                </div>

                <div class="status status-positive">
                    Active
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    preference = st.text_input(
        "Participant-stated preference",
        placeholder=(
            "Optional. Record only what the participant "
            "actually tells you."
        ),
    )

    if preference.strip():

        interaction.stated_preference = (
            preference.strip()
        )

        db.commit()

    st.markdown(
        "<div class='section-title'>Rapid Evidence Logging</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        "These controls record observations. They do not claim to "
        "explain why the participant behaved that way."
    )

    c1, c2, c3 = st.columns(3)

    def log_obs(
        category,
        detail,
        level="OBSERVED",
    ):

        obs = Observation(
            interaction_id=current_id,
            category=category,
            detail=detail,
            evidence_level=level,
        )

        db.add(obs)
        db.commit()

        st.toast(
            "Observation recorded."
        )

    with c1:

        if st.button(
            "Observing target",
            use_container_width=True,
        ):
            log_obs(
                "Attention",
                "Participant is observing the target.",
            )

        if st.button(
            "Looking elsewhere",
            use_container_width=True,
        ):
            log_obs(
                "Attention",
                "Participant is looking elsewhere.",
            )

        if st.button(
            "Asks technical question",
            use_container_width=True,
        ):
            log_obs(
                "Participation",
                "Participant asks a technical question.",
            )

    with c2:

        if st.button(
            "Listening",
            use_container_width=True,
        ):
            log_obs(
                "Participation",
                "Participant appears to be listening.",
            )

        if st.button(
            "Requests more information",
            use_container_width=True,
        ):
            log_obs(
                "Curiosity",
                "Participant explicitly requests additional information.",
                "DIRECT",
            )

        if st.button(
            "Requests a change",
            use_container_width=True,
        ):
            log_obs(
                "Preference",
                "Participant explicitly requests a change.",
                "DIRECT",
            )

    with c3:

        if st.button(
            "Voluntarily leaves",
            use_container_width=True,
        ):
            log_obs(
                "Exit",
                "Participant voluntarily leaves the interaction.",
                "DIRECT",
            )

        if st.button(
            "Environmental noise",
            use_container_width=True,
        ):
            log_obs(
                "Environment",
                "Environmental noise is present.",
                "OBSERVED",
            )

        if st.button(
            "Long pause",
            use_container_width=True,
        ):
            log_obs(
                "Timing",
                "Participant pauses before responding.",
                "OBSERVED",
            )

    st.markdown(
        "<div class='section-title'>Custom Observation</div>",
        unsafe_allow_html=True,
    )

    custom_category = st.selectbox(
        "Category",
        [
            "Attention",
            "Participation",
            "Curiosity",
            "Preference",
            "Environment",
            "Timing",
            "Exit",
            "Other",
        ],
    )

    custom_detail = st.text_input(
        "Observation",
        placeholder=(
            "Describe only what was actually observed."
        ),
    )

    custom_level = st.selectbox(
        "Evidence level",
        [
            "DIRECT",
            "OBSERVED",
            "INFERRED",
            "HYPOTHESIS",
        ],
    )

    if st.button(
        "Record Observation",
        use_container_width=True,
    ):

        if custom_detail.strip():

            log_obs(
                custom_category,
                custom_detail.strip(),
                custom_level,
            )

        else:

            st.warning(
                "Enter an observation first."
            )

    st.markdown(
        "<div class='section-title'>Recent Evidence</div>",
        unsafe_allow_html=True,
    )

    recent_obs = (
        db.query(Observation)
        .filter(
            Observation.interaction_id
            == current_id
        )
        .order_by(
            Observation.timestamp.desc()
        )
        .limit(12)
        .all()
    )

    if not recent_obs:

        st.info(
            "No observations recorded yet."
        )

    else:

        for obs in recent_obs:

            css_class = (
                "observed"
                if obs.evidence_level
                in ["DIRECT", "OBSERVED"]
                else "interpreted"
                if obs.evidence_level == "INFERRED"
                else "speculative"
            )

            st.markdown(
                f"""
                <div class="evidence {css_class}">

                    <div class="evidence-title">
                        {obs.evidence_level}
                        ·
                        {obs.category}
                    </div>

                    <div class="evidence-body">
                        {obs.detail}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        "<div class='section-title'>AI Adaptation</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        "Generate Next Outreach Action",
        type="primary",
        use_container_width=True,
    ):

        if ai_client is None:

            st.error(
                "Gemini is unavailable. "
                "Add GEMINI_API_KEY to Streamlit secrets "
                "or the environment."
            )

        else:

            observations = [
                (
                    f"[{o.evidence_level}] "
                    f"{o.category}: "
                    f"{o.detail}"
                )
                for o in recent_obs
            ]

            context = {
                "phase": interaction.phase,
                "preference": (
                    interaction.stated_preference
                    or "Not provided"
                ),
                "observations": observations,
                "objectives": event_record.primary_objective,
            }

            with st.spinner(
                "Analyzing current outreach evidence..."
            ):

                try:

                    recommendation = get_adaptation(
                        context,
                        ai_client,
                        selected_model,
                    )

                    st.session_state[
                        "last_recommendation"
                    ] = recommendation.model_dump()

                    st.markdown(
                        f"""
                        <div class="premium-card">

                            <div class="eyebrow">
                                Recommended Action
                            </div>

                            <h2>
                                {recommendation.recommended_action}
                            </h2>

                            <p>
                                {recommendation.rationale}
                            </p>

                            <div class="status status-positive">
                                Confidence:
                                {recommendation.confidence}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        "<div class='section-title'>Evidence Used</div>",
                        unsafe_allow_html=True,
                    )

                    for evidence in recommendation.evidence:

                        st.markdown(
                            f"""
                            <div class="evidence observed">
                                <div class="evidence-body">
                                    {evidence}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f"""
                        <div class="premium-card">

                            <div class="eyebrow">
                                Alternative Explanation
                            </div>

                            <p>
                                {recommendation.alternative_explanation}
                            </p>

                            <div class="eyebrow"
                                 style="margin-top:18px;">
                                Next Signal To Watch
                            </div>

                            <p>
                                {recommendation.next_observation}
                            </p>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                except Exception as error:

                    st.error(
                        "The selected Gemini model could not "
                        "complete the analysis."
                    )

                    st.caption(
                        str(error)
                    )

    st.markdown(
        "<div class='section-title'>Interaction Controls</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        "End Interaction",
        use_container_width=True,
    ):

        interaction.timestamp_end = (
            datetime.now(timezone.utc)
        )

        db.commit()

        st.session_state.current_interaction = None
        st.session_state.last_recommendation = None

        st.success(
            "Interaction closed."
        )

        st.rerun()


# ============================================================
# IMPACT & SURVEYS
# ============================================================

elif page == "Impact & Surveys":

    st.markdown(
        """
        <div class="section-title">
            Participant Impact
        </div>

        <p class="muted">
            Optional participant feedback can complement behavioral
            observations with direct self-report.
        </p>
        """,
        unsafe_allow_html=True,
    )

    interactions = (
        db.query(Interaction)
        .order_by(
            Interaction.timestamp_start.desc()
        )
        .limit(50)
        .all()
    )

    if not interactions:

        st.info(
            "No participant interactions have been recorded."
        )
        st.stop()

    interaction_labels = {}

    for i in interactions:

        event = (
            db.query(Event)
            .filter(Event.id == i.event_id)
            .first()
        )

        event_name = (
            event.name
            if event
            else "Unknown event"
        )

        interaction_labels[
            f"{event_name} · {i.id[:8]}"
        ] = i

    selected_label = st.selectbox(
        "Participant interaction",
        list(interaction_labels.keys()),
    )

    selected_int = interaction_labels[
        selected_label
    ]

    tabs = st.tabs(
        [
            "Baseline",
            "Immediate",
            "Delayed Follow-up",
        ]
    )

    with tabs[0]:

        with st.form("baseline_form"):

            curiosity = st.slider(
                "Curiosity before engagement",
                1,
                10,
                5,
            )

            knowledge = st.slider(
                "Self-rated knowledge before engagement",
                0,
                100,
                0,
            )

            submitted = st.form_submit_button(
                "Save Baseline",
                use_container_width=True,
            )

            if submitted:

                db.add(
                    Survey(
                        interaction_id=selected_int.id,
                        timing="BASELINE",
                        curiosity_score=curiosity,
                        knowledge_score=knowledge,
                    )
                )

                db.commit()

                st.success(
                    "Baseline recorded."
                )

    with tabs[1]:

        with st.form("immediate_form"):

            curiosity_post = st.slider(
                "Curiosity after engagement",
                1,
                10,
                8,
            )

            knowledge_post = st.slider(
                "Self-rated knowledge after engagement",
                0,
                100,
                50,
            )

            memory_text = st.text_area(
                "What is the one thing you expect to remember?",
                placeholder=(
                    "Optional participant response."
                ),
            )

            submitted = st.form_submit_button(
                "Save Immediate Response",
                use_container_width=True,
            )

            if submitted:

                db.add(
                    Survey(
                        interaction_id=selected_int.id,
                        timing="IMMEDIATE",
                        curiosity_score=curiosity_post,
                        knowledge_score=knowledge_post,
                        memorability_text=memory_text,
                    )
                )

                db.commit()

                st.success(
                    "Immediate response recorded."
                )

    with tabs[2]:

        with st.form("delayed_form"):

            follow = st.radio(
                "Did you independently engage with the topic afterward?",
                [
                    "Yes",
                    "No",
                    "Not sure",
                ],
            )

            submitted = st.form_submit_button(
                "Save Follow-up",
                use_container_width=True,
            )

            if submitted:

                db.add(
                    Survey(
                        interaction_id=selected_int.id,
                        timing="DELAYED_24H",
                        follow_through=follow,
                    )
                )

                db.commit()

                st.success(
                    "Follow-up recorded."
                )


# ============================================================
# IMPACT OBSERVATORY
# ============================================================

elif page == "Impact Observatory":

    st.markdown(
        """
        <div class="section-title">
            Impact Observatory
        </div>

        <p class="muted">
            Measure what changed after outreach. Deterministic metrics
            are calculated independently from Gemini interpretation.
        </p>
        """,
        unsafe_allow_html=True,
    )

    events = (
        db.query(Event)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:

        st.info(
            "Create an event first."
        )
        st.stop()

    event_dict = {
        e.name: e
        for e in events
    }

    event_name = st.selectbox(
        "Event",
        list(event_dict.keys()),
    )

    event_record = event_dict[
        event_name
    ]

    metrics = calculate_impact_fingerprint(
        db,
        event_record.id,
    )

    if not metrics:

        st.info(
            "There is not enough survey data to calculate an impact fingerprint yet."
        )

    else:

        st.markdown(
            "<div class='eyebrow'>Deterministic Measurements</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Participants
                </div>
                <div class="metric-value">
                    {metrics["total_participants"]}
                </div>
                <div class="metric-detail">
                    Unique interaction records
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        curiosity_change = (
            "—"
            if metrics["curiosity_change"] is None
            else f'{metrics["curiosity_change"]:+.2f}'
        )

        c2.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Curiosity Shift
                </div>
                <div class="metric-value">
                    {curiosity_change}
                </div>
                <div class="metric-detail">
                    Pre → post
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        knowledge_change = (
            "—"
            if metrics["knowledge_change"] is None
            else f'{metrics["knowledge_change"]:+.2f}%'
        )

        c3.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Knowledge Shift
                </div>
                <div class="metric-value">
                    {knowledge_change}
                </div>
                <div class="metric-detail">
                    Self-rated pre → post
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        follow_rate = (
            "—"
            if metrics["follow_through_rate"] is None
            else f'{metrics["follow_through_rate"]:.1f}%'
        )

        c4.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Follow-through
                </div>
                <div class="metric-value">
                    {follow_rate}
                </div>
                <div class="metric-detail">
                    Delayed self-report
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div class='section-title'>Impact Interpretation</div>",
            unsafe_allow_html=True,
        )

        if ai_client is not None:

            if st.button(
                "Interpret Impact With Gemini",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Interpreting measured engagement patterns..."
                ):

                    try:

                        interpretation = interpret_impact(
                            metrics,
                            ai_client,
                            selected_model,
                        )

                        st.markdown(
                            f"""
                            <div class="premium-card">

                                <div class="eyebrow">
                                    Summary
                                </div>

                                <p style="
                                    color:#f5f5f7;
                                    font-size:1.05rem;
                                ">
                                    {interpretation.summary}
                                </p>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        c1, c2 = st.columns(2)

                        with c1:

                            st.markdown(
                                f"""
                                <div class="premium-card">

                                    <div class="eyebrow">
                                        Strongest Signal
                                    </div>

                                    <p>
                                        {interpretation.strongest_signal}
                                    </p>

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with c2:

                            st.markdown(
                                f"""
                                <div class="premium-card">

                                    <div class="eyebrow">
                                        Weakest Signal
                                    </div>

                                    <p>
                                        {interpretation.weakest_signal}
                                    </p>

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            "<div class='section-title'>Plausible Mechanisms</div>",
                            unsafe_allow_html=True,
                        )

                        for item in interpretation.plausible_mechanisms:

                            st.markdown(
                                f"""
                                <div class="evidence interpreted">
                                    <div class="evidence-body">
                                        {item}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            "<div class='section-title'>Limitations</div>",
                            unsafe_allow_html=True,
                        )

                        for item in interpretation.limitations:

                            st.markdown(
                                f"""
                                <div class="evidence speculative">
                                    <div class="evidence-body">
                                        {item}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown(
                            f"""
                            <div class="premium-card">

                                <div class="eyebrow">
                                    Recommended Next Measurement
                                </div>

                                <p>
                                    {interpretation.next_measurement}
                                </p>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    except Exception as error:

                        st.error(
                            "Gemini could not interpret the impact data."
                        )

                        st.caption(
                            str(error)
                        )

        else:

            st.info(
                "Add GEMINI_API_KEY to enable AI interpretation."
            )

        # ----------------------------------------------------
        # MEMORY THEMES
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-title'>Memory Anchors</div>",
            unsafe_allow_html=True,
        )

        surveys = (
            db.query(Survey)
            .join(Interaction)
            .filter(
                Interaction.event_id
                == event_record.id,
                Survey.timing
                == "IMMEDIATE",
                Survey.memorability_text.isnot(None),
            )
            .all()
        )

        valid_surveys = [
            s
            for s in surveys
            if s.memorability_text
            and s.memorability_text.strip()
        ]

        if not valid_surveys:

            st.info(
                "No memory responses have been collected."
            )

        elif ai_client is None:

            st.info(
                "Add GEMINI_API_KEY to enable memory-theme analysis."
            )

        else:

            if st.button(
                "Analyze Memory Themes",
                use_container_width=True,
            ):

                for survey in valid_surveys:

                    try:

                        theme = extract_theme(
                            survey.memorability_text,
                            ai_client,
                            selected_model,
                        )

                        st.markdown(
                            f"""
                            <div class="premium-card">

                                <div class="eyebrow">
                                    {theme.confidence} confidence
                                </div>

                                <h3>
                                    {theme.theme}
                                </h3>

                                <p>
                                    Participant response:
                                    "{survey.memorability_text}"
                                </p>

                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    except Exception as error:

                        st.warning(
                            "One memory response could not be analyzed."
                        )

        # ----------------------------------------------------
        # RAW DATA EXPORT
        # ----------------------------------------------------

        st.markdown(
            "<div class='section-title'>Event Data</div>",
            unsafe_allow_html=True,
        )

        export_query = (
            db.query(Survey)
            .join(Interaction)
            .filter(
                Interaction.event_id
                == event_record.id
            )
        )

        export_df = pd.read_sql(
            export_query.statement,
            db.bind,
        )

        if not export_df.empty:

            csv_data = export_df.to_csv(
                index=False
            )

            st.download_button(
                "Export Event Data",
                data=csv_data,
                file_name=(
                    f"{event_record.name}"
                    .replace(" ", "_")
                    .lower()
                    + "_impact.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:80px;
        padding-top:25px;
        border-top:1px solid #202023;
        text-align:center;
        color:#52525b;
        font-size:0.78rem;
        line-height:1.6;
    ">

        <strong style="color:#71717a;">
            Ninolades Outreach Intelligence
        </strong>

        <br>

        Evidence-guided science communication and engagement research.

        <br><br>

        AI-generated interpretations are synthetic hypotheses.
        They do not establish psychological, clinical, neurological,
        or causal facts about individual participants.

    </div>
    """,
    unsafe_allow_html=True,
)
