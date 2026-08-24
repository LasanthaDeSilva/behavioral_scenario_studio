"""
NINOLADES OUTREACH INTELLIGENCE LAB
===================================

Version: 1.3.0

A self-contained Streamlit application for:

    - Designing outreach scenarios
    - Modeling likely engagement pathways
    - Predicting real-world outcomes and metrics
    - Running a live outreach copilot
    - Running a true Gemini Live realtime voice assistant
    - Recording lightweight observations
    - Comparing predicted vs observed engagement
    - Measuring optional real-world impact
    - Exploring counterfactual interventions
    - Extracting qualitative memory/engagement themes
    - Viewing event-level analytics
    - Persisting user-specific application memory

IMPORTANT METHODOLOGICAL PRINCIPLES
-----------------------------------

1. AI predictions are hypotheses, not measurements.
2. AI must not diagnose participants.
3. AI must not infer protected/sensitive personal attributes.
4. Participant-stated preferences outrank model inference.
5. Observed behavior is kept separate from interpretation.
6. Deterministic analytics are calculated in Python.
7. Gemini is used for interpretation/generation, not arithmetic.
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

MODEL PRESERVATION
------------------

The three existing reasoning models are intentionally preserved:

    gemini-3.6-flash
    gemini-3.1-pro
    gemini-3.5-flash-lite

A separate Gemini Live model is used ONLY by the realtime voice layer:

    gemini-3.1-flash-live-preview

The Live model does not replace or alter the three reasoning engines.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import json
import html
import uuid
import textwrap
import hashlib
from datetime import datetime, timezone, timedelta
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
    UniqueConstraint,
    inspect,
    text,
)

from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
)

from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ============================================================
# 2. APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "Outreach Intelligence Lab"
APP_VERSION = "1.3.0"

# ------------------------------------------------------------
# EXISTING REASONING MODELS
# DO NOT CHANGE
# ------------------------------------------------------------

MODEL_FLASH = "gemini-3.6-flash"
MODEL_PRO = "gemini-3.1-pro"
MODEL_LITE = "gemini-3.5-flash-lite"

DEFAULT_MODEL = MODEL_FLASH

# ------------------------------------------------------------
# LIVE VOICE MODEL
# Dedicated realtime audio model.
# This does NOT replace the three reasoning models above.
# ------------------------------------------------------------

LIVE_VOICE_MODEL = "gemini-3.1-flash-live-preview"

# Browser SDK used by the Live component.
# Pinned intentionally to avoid silent major-version changes.
LIVE_JS_SDK_VERSION = "2.18.0"

DATABASE_URL = os.getenv(
    "OUTREACH_DATABASE_URL",
    "sqlite:///ninolades_outreach_lab.db",
)


# ============================================================
# 3. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 4. PREMIUM MINIMALIST CSS
# ============================================================

PREMIUM_CSS = """
<style>

:root {
    --bg: #0a0a0b;
    --surface: #101012;
    --surface-2: #151518;
    --surface-3: #1b1b1f;
    --border: #27272c;
    --border-soft: #1f1f23;

    --text: #f5f5f7;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;

    --accent: #5b8cff;
    --accent-hover: #6b9aff;

    --success: #4ade80;
    --warning: #fbbf24;
    --danger: #f87171;
}

html,
body,
[class*="css"] {
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
            rgba(91, 140, 255, .055),
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

h1,
h2,
h3,
h4 {
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

p,
label,
span {
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
    background: rgba(255,255,255,.03);
    color: var(--text);
    border: 1px solid rgba(255,255,255,.15);
    border-radius: 8px;
    font-weight: 500;
    letter-spacing: .3px;
    min-height: 42px;
    transition:
        all .2s ease;
    box-shadow:
        0 1px 3px rgba(0,0,0,.05);
}

.stButton button:hover {
    border-color: var(--text-secondary);
    background: rgba(255,255,255,.06);
    color: white;
    box-shadow:
        0 4px 8px rgba(0,0,0,.15);
    transform: translateY(-1px);
}

button[data-testid="baseButton-primary"] {
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: white !important;
    box-shadow:
        0 2px 8px rgba(91,140,255,.25) !important;
}

button[data-testid="baseButton-primary"]:hover {
    background: var(--accent-hover) !important;
    border-color: var(--accent-hover) !important;
    box-shadow:
        0 4px 14px rgba(91,140,255,.4) !important;
}

div[data-testid="stRadio"] {
    background: transparent;
}

div[data-testid="stRadio"] > div {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

div[data-testid="stRadio"] label {
    background: rgba(255,255,255,.02);
    padding: 10px 20px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.08) !important;
    cursor: pointer;
    transition: all .25s ease;
}

div[data-testid="stRadio"] label:hover {
    border-color: rgba(255,255,255,.3) !important;
    background: rgba(255,255,255,.06);
    transform: translateY(-1px);
}

div[data-testid="stRadio"] label[data-checked="true"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow:
        0 4px 12px rgba(91,140,255,.3);
}

div[data-testid="stRadio"] label[data-checked="true"] p {
    color: white !important;
    font-weight: 600;
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
    background:
        linear-gradient(
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

st.markdown(
    textwrap.dedent(PREMIUM_CSS),
    unsafe_allow_html=True,
)


# ============================================================
# 5. DATABASE
# ============================================================

Base = declarative_base()


@st.cache_resource
def get_db_engine():
    kwargs = {}

    if DATABASE_URL.startswith("sqlite"):
        kwargs["connect_args"] = {
            "check_same_thread": False,
            "timeout": 30,
        }

    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        **kwargs,
    )


@st.cache_resource
def get_session_factory(_engine):
    return sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


engine = get_db_engine()


# ============================================================
# 6. DATABASE MODELS
# ============================================================

class UserMemory(Base):
    __tablename__ = "user_memory"

    user_id = Column(String, primary_key=True)

    created_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    active_event_id = Column(
        String,
        nullable=True,
    )

    active_interaction_id = Column(
        String,
        nullable=True,
    )

    current_page = Column(
        String,
        nullable=True,
    )

    selected_model = Column(
        String,
        nullable=True,
    )

    last_recommendation = Column(
        Text,
        nullable=True,
    )

    last_forward_model = Column(
        Text,
        nullable=True,
    )

    last_counterfactual = Column(
        Text,
        nullable=True,
    )

    last_impact_interpretation = Column(
        Text,
        nullable=True,
    )

    last_prediction = Column(
        Text,
        nullable=True,
    )

    voice_enabled = Column(
        Boolean,
        nullable=False,
        default=False,
    )

    voice_preferences = Column(
        Text,
        nullable=True,
    )


class MemoryEvent(Base):
    __tablename__ = "memory_events"

    id = Column(String, primary_key=True)

    user_id = Column(
        String,
        ForeignKey("user_memory.user_id"),
        nullable=False,
        index=True,
    )

    timestamp = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    event_type = Column(
        String,
        nullable=False,
        index=True,
    )

    payload = Column(
        Text,
        nullable=True,
    )


class Event(Base):
    __tablename__ = "events"

    id = Column(String, primary_key=True)

    user_id = Column(
        String,
        nullable=False,
        index=True,
    )

    session_token = Column(
        String,
        nullable=True,
        index=True,
    )

    name = Column(
        String,
        nullable=False,
    )

    date = Column(
        DateTime,
        nullable=False,
    )

    objective = Column(
        String,
        nullable=False,
    )

    context = Column(
        Text,
        nullable=True,
    )

    environment = Column(
        String,
        nullable=True,
    )

    sensory_environment = Column(
        String,
        nullable=True,
    )

    acoustic_environment = Column(
        String,
        nullable=True,
    )

    target_audience = Column(
        String,
        nullable=True,
    )

    interactions = relationship(
        "Interaction",
        back_populates="event",
        cascade="all, delete-orphan",
    )


class Interaction(Base):
    __tablename__ = "interactions"

    id = Column(String, primary_key=True)

    user_id = Column(
        String,
        nullable=False,
        index=True,
    )

    event_id = Column(
        String,
        ForeignKey("events.id"),
        nullable=False,
    )

    participant_code = Column(
        String,
        nullable=False,
    )

    started_at = Column(
        DateTime,
        nullable=False,
    )

    ended_at = Column(
        DateTime,
        nullable=True,
    )

    phase = Column(
        String,
        default="Approach",
        nullable=False,
    )

    stated_preference = Column(
        Text,
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

    id = Column(String, primary_key=True)

    user_id = Column(
        String,
        nullable=False,
        index=True,
    )

    interaction_id = Column(
        String,
        ForeignKey("interactions.id"),
        nullable=False,
    )

    timestamp = Column(
        DateTime,
        nullable=False,
    )

    category = Column(
        String,
        nullable=False,
    )

    detail = Column(
        Text,
        nullable=False,
    )

    evidence_level = Column(
        String,
        nullable=False,
        default="OBSERVED",
    )

    interaction = relationship(
        "Interaction",
        back_populates="observations",
    )


class Survey(Base):
    __tablename__ = "surveys"

    id = Column(String, primary_key=True)

    user_id = Column(
        String,
        nullable=False,
        index=True,
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

    curiosity = Column(
        Float,
        nullable=True,
    )

    understanding = Column(
        Float,
        nullable=True,
    )

    confidence = Column(
        Float,
        nullable=True,
    )

    recall_text = Column(
        Text,
        nullable=True,
    )

    follow_through = Column(
        Boolean,
        nullable=True,
    )

    interaction = relationship(
        "Interaction",
        back_populates="surveys",
    )


class RapidStateLog(Base):
    __tablename__ = "rapid_state_logs"

    id = Column(String, primary_key=True)

    user_id = Column(
        String,
        nullable=False,
        index=True,
    )

    event_id = Column(
        String,
        ForeignKey("events.id"),
        nullable=False,
    )

    participant_code = Column(
        String,
        nullable=False,
    )

    timestamp = Column(
        DateTime,
        nullable=False,
    )

    baseline_level = Column(
        String,
        nullable=False,
    )

    current_state = Column(
        String,
        nullable=False,
    )

    event = relationship("Event")


# ============================================================
# 7. DATABASE MIGRATION
# ============================================================

def ensure_column(
    connection,
    table_name: str,
    column_name: str,
    column_definition: str,
):
    """
    Add a missing column without failing if it already exists.
    """

    inspector = inspect(connection)

    try:
        columns = {
            column["name"]
            for column in inspector.get_columns(table_name)
        }
    except Exception:
        return

    if column_name not in columns:
        connection.execute(
            text(
                f"ALTER TABLE {table_name} "
                f"ADD COLUMN {column_name} "
                f"{column_definition}"
            )
        )


def initialize_database():
    Base.metadata.create_all(bind=engine)

    # Compatibility migration for databases created by older versions.
    with engine.begin() as connection:

        ensure_column(
            connection,
            "events",
            "user_id",
            "VARCHAR",
        )

        ensure_column(
            connection,
            "events",
            "session_token",
            "VARCHAR",
        )

        ensure_column(
            connection,
            "interactions",
            "user_id",
            "VARCHAR",
        )

        ensure_column(
            connection,
            "observations",
            "user_id",
            "VARCHAR",
        )

        ensure_column(
            connection,
            "surveys",
            "user_id",
            "VARCHAR",
        )

        ensure_column(
            connection,
            "rapid_state_logs",
            "user_id",
            "VARCHAR",
        )


initialize_database()

SessionLocal = get_session_factory(engine)


def db_session():
    return SessionLocal()


# ============================================================
# 8. CLIENT-SIDE PERSISTENT IDENTITY COMPONENT
# ============================================================

# Streamlit Components v2 allows trusted inline JS to communicate
# directly with Python and persist a browser-local identifier.

USER_ID_COMPONENT_HTML = """
<div class="identity-root" aria-hidden="true"></div>
"""

USER_ID_COMPONENT_CSS = """
.identity-root {
    display: none;
}
"""

USER_ID_COMPONENT_JS = """
export default function(component) {

    const {
        setStateValue,
        parentElement
    } = component;

    const STORAGE_KEY = "ninolades_outreach_user_id_v1";

    function generateId() {
        if (
            typeof crypto !== "undefined" &&
            crypto.randomUUID
        ) {
            return crypto.randomUUID();
        }

        return (
            "user-" +
            Date.now().toString(36) +
            "-" +
            Math.random().toString(36).slice(2, 14)
        );
    }

    let userId = null;

    try {
        userId = window.localStorage.getItem(STORAGE_KEY);

        if (!userId) {
            userId = generateId();
            window.localStorage.setItem(
                STORAGE_KEY,
                userId
            );
        }
    } catch (error) {
        // Storage may be disabled.
        // Use a browser-session fallback.
        userId =
            window.name ||
            generateId();

        try {
            window.name = userId;
        } catch (_) {}
    }

    setStateValue(
        "user_id",
        userId
    );

    return () => {};
}
"""


try:
    identity_component = st.components.v2.component(
        name="ninolades_persistent_identity",
        html=USER_ID_COMPONENT_HTML,
        css=USER_ID_COMPONENT_CSS,
        js=USER_ID_COMPONENT_JS,
    )

    identity_result = identity_component(
        key="persistent_browser_identity",
        default={
            "user_id": None,
        },
    )

    browser_user_id = (
        getattr(
            identity_result,
            "user_id",
            None,
        )
        if identity_result is not None
        else None
    )

except Exception as identity_error:

    browser_user_id = None

    st.error(
        "Persistent browser identity could not initialize. "
        f"Please use a current Streamlit release. Details: "
        f"{identity_error}"
    )


# ============================================================
# 9. FIRST-LOAD IDENTITY GATE
# ============================================================

if not browser_user_id:

    st.markdown(
        """
        <div style="
            min-height:60vh;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
        ">
            <div>
                <div class="eyebrow">
                    Ninolades Research Platform
                </div>
                <div style="
                    color:white;
                    font-size:1.35rem;
                    margin-bottom:8px;
                ">
                    Initializing private workspace
                </div>
                <div class="small-note">
                    Creating your browser-local workspace...
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# Normalize and validate the browser-generated identity.
browser_user_id = str(browser_user_id).strip()

if len(browser_user_id) > 128:
    browser_user_id = hashlib.sha256(
        browser_user_id.encode("utf-8")
    ).hexdigest()


USER_SESSION_TOKEN = browser_user_id


# ============================================================
# 10. USER MEMORY HELPERS
# ============================================================

def get_or_create_user_memory(db):
    memory = (
        db.query(UserMemory)
        .filter(
            UserMemory.user_id == USER_SESSION_TOKEN
        )
        .first()
    )

    if memory is None:

        memory = UserMemory(
            user_id=USER_SESSION_TOKEN,
            created_at=utc_now(),
            updated_at=utc_now(),
            current_page="Experience Designer",
            selected_model=DEFAULT_MODEL,
            voice_enabled=False,
            voice_preferences=json.dumps({}),
        )

        db.add(memory)
        db.commit()

    return memory


def utc_now():
    return datetime.now(timezone.utc)


def json_dumps(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def json_loads(value, default=None):

    if not value:
        return default

    try:
        return json.loads(value)
    except Exception:
        return default


def save_memory_event(
    db,
    event_type,
    payload,
):
    memory_event = MemoryEvent(
        id=str(uuid.uuid4()),
        user_id=USER_SESSION_TOKEN,
        timestamp=utc_now(),
        event_type=event_type,
        payload=json_dumps(payload),
    )

    db.add(memory_event)

    memory = get_or_create_user_memory(db)
    memory.updated_at = utc_now()

    db.commit()


def persist_memory_state(
    db,
    memory,
):
    memory.updated_at = utc_now()
    db.commit()


def restore_session_from_memory(
    db,
    memory,
):

    restored = {
        "active_event_id": memory.active_event_id,
        "active_interaction_id": memory.active_interaction_id,
        "current_page": memory.current_page,
        "selected_model": memory.selected_model,
        "last_recommendation": json_loads(
            memory.last_recommendation,
            None,
        ),
        "last_forward_model": json_loads(
            memory.last_forward_model,
            None,
        ),
        "last_counterfactual": json_loads(
            memory.last_counterfactual,
            None,
        ),
        "last_impact_interpretation": json_loads(
            memory.last_impact_interpretation,
            None,
        ),
        "last_prediction": json_loads(
            memory.last_prediction,
            None,
        ),
    }

    return restored


# ============================================================
# 11. DATABASE INSTANCE + USER MEMORY
# ============================================================

db = db_session()

user_memory = get_or_create_user_memory(db)

restored_memory = restore_session_from_memory(
    db,
    user_memory,
)


# ============================================================
# 12. SESSION STATE INITIALIZATION
# ============================================================

DEFAULT_STATE = {
    "active_event_id":
        restored_memory["active_event_id"],

    "active_interaction_id":
        restored_memory["active_interaction_id"],

    "last_recommendation":
        restored_memory["last_recommendation"],

    "last_forward_model":
        restored_memory["last_forward_model"],

    "last_counterfactual":
        restored_memory["last_counterfactual"],

    "last_impact_interpretation":
        restored_memory["last_impact_interpretation"],

    "last_prediction":
        restored_memory["last_prediction"],

    "mem_model_label":
        "Gemini 3.6 Flash",

    "mem_page":
        restored_memory["current_page"]
        or "Experience Designer",
}

model_options = {
    "Gemini 3.6 Flash": MODEL_FLASH,
    "Gemini 3.1 Pro": MODEL_PRO,
    "Gemini 3.5 Flash-Lite": MODEL_LITE,
}

saved_model = restored_memory.get(
    "selected_model"
)

if saved_model:
    for label, model_id in model_options.items():
        if model_id == saved_model:
            DEFAULT_STATE["mem_model_label"] = label
            break


for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# 13. PERSISTENT SESSION HELPERS
# ============================================================

def persist_current_state():

    memory = get_or_create_user_memory(db)

    memory.active_event_id = (
        st.session_state.get(
            "active_event_id"
        )
    )

    memory.active_interaction_id = (
        st.session_state.get(
            "active_interaction_id"
        )
    )

    memory.current_page = (
        st.session_state.get(
            "mem_page"
        )
    )

    selected_label = (
        st.session_state.get(
            "mem_model_label"
        )
    )

    memory.selected_model = (
        model_options.get(
            selected_label,
            DEFAULT_MODEL,
        )
    )

    memory.last_recommendation = json_dumps(
        st.session_state.get(
            "last_recommendation"
        )
    )

    memory.last_forward_model = json_dumps(
        st.session_state.get(
            "last_forward_model"
        )
    )

    memory.last_counterfactual = json_dumps(
        st.session_state.get(
            "last_counterfactual"
        )
    )

    memory.last_impact_interpretation = json_dumps(
        st.session_state.get(
            "last_impact_interpretation"
        )
    )

    memory.last_prediction = json_dumps(
        st.session_state.get(
            "last_prediction"
        )
    )

    persist_memory_state(
        db,
        memory,
    )


# ============================================================
# 14. GEMINI API KEY
# ============================================================

def get_api_key() -> str:

    key = ""

    try:
        key = st.secrets.get(
            "GEMINI_API_KEY",
            "",
        )
    except Exception:
        pass

    if not key:
        key = os.getenv(
            "GEMINI_API_KEY",
            "",
        )

    return str(key).strip()


@st.cache_resource
def create_gemini_client(
    api_key: str,
):

    if not api_key:
        return None

    try:
        return genai.Client(
            api_key=api_key
        )
    except Exception:
        return None


api_key = get_api_key()

client = create_gemini_client(
    api_key
)


# ============================================================
# 15. GEMINI EPHEMERAL LIVE TOKEN
# ============================================================

def create_live_ephemeral_token():

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    live_client = genai.Client(
        api_key=api_key,
    )

    now = datetime.now(
        timezone.utc
    )

    token = live_client.auth_tokens.create(
        config={
            "uses": 1,

            "expire_time":
                now + timedelta(
                    minutes=30
                ),

            "new_session_expire_time":
                now + timedelta(
                    minutes=1
                ),

            "live_connect_constraints": {
                "model":
                    LIVE_VOICE_MODEL,

                "config": {
                    "response_modalities": [
                        "AUDIO"
                    ],

                    "input_audio_transcription": {},

                    "output_audio_transcription": {},

                    "session_resumption": {},
                },
            },
        }
    )

    token_name = getattr(
        token,
        "name",
        None,
    )

    if not token_name:
        raise RuntimeError(
            "Gemini did not return an ephemeral Live API token."
        )

    return token_name


# ============================================================
# 16. PYDANTIC SCHEMAS
# ============================================================

ConfidenceLevel = Literal[
    "Low",
    "Moderate",
    "High",
]


class CognitiveEstimate(BaseModel):

    bandwidth_pct: int = Field(
        ge=0,
        le=100,
    )

    focus_pct: int = Field(
        ge=0,
        le=100,
    )

    sensory_load_pct: int = Field(
        ge=0,
        le=100,
    )

    rationale: str


class OutreachRecommendation(BaseModel):

    recommended_action: str

    rationale: str

    confidence: ConfidenceLevel

    evidence: List[str] = Field(
        min_length=1,
        max_length=6,
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

    predicted_pathways: List[
        PredictedPathway
    ] = Field(
        min_length=3,
        max_length=3,
    )

    recommended_outreach_design: List[str] = Field(
        min_length=3,
        max_length=5,
    )

    likely_friction_points: List[str] = Field(
        min_length=1,
        max_length=5,
    )

    measurement_opportunities: List[str] = Field(
        min_length=2,
        max_length=5,
    )


class CounterfactualModel(BaseModel):

    changed_variable: str

    expected_difference: str

    before_state: str

    after_state: str

    predicted_effects: List[str] = Field(
        min_length=3,
        max_length=5,
    )

    uncertainty: str


class ThemeModel(BaseModel):

    theme: str

    description: str

    evidence_strength: ConfidenceLevel

    evidence_quotes: List[str] = Field(
        min_length=1,
        max_length=4,
    )


class ImpactInterpretation(BaseModel):

    overall_interpretation: str

    strongest_signal: str

    weakest_signal: str

    plausible_mechanisms: List[str] = Field(
        min_length=2,
        max_length=5,
    )

    alternative_explanations: List[str] = Field(
        min_length=2,
        max_length=5,
    )

    recommended_next_test: str


class OutcomePrediction(BaseModel):

    focus_pct: int = Field(
        ge=0,
        le=100,
    )

    stress_reduction_pct: int = Field(
        ge=0,
        le=100,
    )

    cognitive_load_pct: int = Field(
        ge=0,
        le=100,
    )

    attention_retention_pct: int = Field(
        ge=0,
        le=100,
    )

    predicted_curiosity_shift: str

    predicted_understanding_shift: str

    predicted_engagement_rate: str

    overall_outcome_narrative: str

    risk_factors: List[str] = Field(
        min_length=1,
        max_length=5,
    )

    success_amplifiers: List[str] = Field(
        min_length=1,
        max_length=5,
    )


# ============================================================
# 17. GEMINI GENERATION HELPER
# ============================================================

def run_gemini(
    client,
    model_name: str,
    prompt: str,
    schema,
    system_instruction: str,
    temperature: float = 0.2,
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
        ),
    )

    parsed = getattr(
        response,
        "parsed",
        None,
    )

    if parsed is not None:
        return parsed

    response_text = getattr(
        response,
        "text",
        None,
    )

    if not response_text:

        raise RuntimeError(
            "Gemini returned an empty response."
        )

    response_text = response_text.strip()

    if response_text.startswith(
        "```json"
    ):
        response_text = response_text[7:]

    elif response_text.startswith(
        "```"
    ):
        response_text = response_text[3:]

    if response_text.endswith(
        "```"
    ):
        response_text = response_text[:-3]

    response_text = response_text.strip()

    return schema.model_validate_json(
        response_text
    )


# ============================================================
# 18. SYSTEM INSTRUCTIONS
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
# 19. UI HELPERS
# ============================================================

def render_html(
    html_str: str
):

    safe_html = "\n".join(
        line.lstrip()
        for line in html_str.split("\n")
    )

    st.markdown(
        safe_html,
        unsafe_allow_html=True,
    )


def clean_text(
    value: Any
) -> str:

    return html.escape(
        str(value or "")
    )


def mean_or_none(series):

    if series is None:
        return None

    series = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if series.empty:
        return None

    return float(
        series.mean()
    )


# ============================================================
# 20. DATABASE EVENT FUNCTIONS
# ============================================================

def create_event(
    db,
    name,
    objective,
    context,
    environment,
    sensory_environment,
    acoustic_environment,
    target_audience,
):

    event = Event(

        id=str(
            uuid.uuid4()
        ),

        user_id=USER_SESSION_TOKEN,

        session_token=USER_SESSION_TOKEN,

        name=name.strip(),

        date=utc_now(),

        objective=objective,

        context=context.strip(),

        environment=environment.strip(),

        sensory_environment=
            sensory_environment.strip(),

        acoustic_environment=
            acoustic_environment.strip(),

        target_audience=
            target_audience,
    )

    db.add(event)
    db.commit()

    save_memory_event(
        db,
        "experience_created",
        {
            "event_id": event.id,
            "name": event.name,
            "objective": event.objective,
        },
    )

    return event


def create_interaction(
    db,
    event_id,
):

    interaction = Interaction(

        id=str(
            uuid.uuid4()
        ),

        user_id=USER_SESSION_TOKEN,

        event_id=event_id,

        participant_code=
            f"P-{uuid.uuid4().hex[:8].upper()}",

        started_at=utc_now(),

        phase="Approach",
    )

    db.add(interaction)
    db.commit()

    save_memory_event(
        db,
        "interaction_started",
        {
            "event_id": event_id,
            "interaction_id": interaction.id,
            "participant_code":
                interaction.participant_code,
        },
    )

    return interaction


def log_observation(
    db,
    interaction_id,
    category,
    detail,
    evidence_level="OBSERVED",
):

    observation = Observation(

        id=str(
            uuid.uuid4()
        ),

        user_id=USER_SESSION_TOKEN,

        interaction_id=interaction_id,

        timestamp=utc_now(),

        category=category,

        detail=detail,

        evidence_level=evidence_level,
    )

    db.add(observation)
    db.commit()

    save_memory_event(
        db,
        "observation_recorded",
        {
            "interaction_id":
                interaction_id,

            "category":
                category,

            "detail":
                detail,

            "evidence_level":
                evidence_level,
        },
    )

    return observation


def get_recent_observations(
    db,
    interaction_id,
    limit=12,
):

    return (
        db.query(Observation)
        .filter(
            Observation.interaction_id
            == interaction_id,

            Observation.user_id
            == USER_SESSION_TOKEN,
        )
        .order_by(
            Observation.timestamp.desc()
        )
        .limit(limit)
        .all()
    )


def get_event_interactions(
    db,
    event_id,
):

    return (
        db.query(Interaction)
        .filter(
            Interaction.event_id
            == event_id,

            Interaction.user_id
            == USER_SESSION_TOKEN,
        )
        .all()
    )


def get_user_events(db):

    return (
        db.query(Event)
        .filter(
            Event.user_id
            == USER_SESSION_TOKEN
        )
        .order_by(
            Event.date.desc()
        )
        .all()
    )


# ============================================================
# 21. EVENT IMPACT ENGINE
# ============================================================

def calculate_event_impact(
    db,
    event_id,
):

    interactions = get_event_interactions(
        db,
        event_id,
    )

    if not interactions:

        return None

    interaction_ids = [
        interaction.id
        for interaction in interactions
    ]

    surveys = (
        db.query(Survey)
        .filter(
            Survey.interaction_id.in_(
                interaction_ids
            ),
            Survey.user_id
            == USER_SESSION_TOKEN,
        )
        .all()
    )

    if not surveys:

        return {
            "participants":
                len(interactions),

            "surveyed":
                0,

            "baseline_curiosity":
                None,

            "post_curiosity":
                None,

            "curiosity_change":
                None,

            "baseline_understanding":
                None,

            "post_understanding":
                None,

            "understanding_change":
                None,

            "baseline_confidence":
                None,

            "post_confidence":
                None,

            "confidence_change":
                None,

            "follow_through_rate":
                None,

            "recall_rate":
                None,
        }

    rows = []

    for survey in surveys:

        rows.append({

            "interaction_id":
                survey.interaction_id,

            "timing":
                survey.timing,

            "curiosity":
                survey.curiosity,

            "understanding":
                survey.understanding,

            "confidence":
                survey.confidence,

            "recall_text":
                survey.recall_text,

            "follow_through":
                survey.follow_through,
        })

    df = pd.DataFrame(rows)

    baseline = (
        df[
            df["timing"]
            == "BASELINE"
        ]
        .groupby("interaction_id")
        .first()
    )

    immediate = (
        df[
            df["timing"]
            == "IMMEDIATE"
        ]
        .groupby("interaction_id")
        .first()
    )

    paired = baseline.join(
        immediate,
        lsuffix="_baseline",
        rsuffix="_post",
        how="inner",
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
            [
                "DELAYED_24H",
                "DELAYED_7D",
            ]
        )
    ]

    follow_through_rate = None

    if not delayed.empty:

        valid = delayed[
            delayed[
                "follow_through"
            ].notna()
        ]

        if not valid.empty:

            follow_through_rate = (
                valid[
                    "follow_through"
                ]
                .astype(bool)
                .mean()
                * 100
            )

    recall = df[
        (
            df["timing"]
            == "IMMEDIATE"
        )
        &
        (
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
            /
            len(immediate)
            *
            100
        )

    return {

        "participants":
            len(interactions),

        "surveyed":
            len(
                df[
                    "interaction_id"
                ].unique()
            ),

        "baseline_curiosity":
            mean_or_none(
                baseline.get(
                    "curiosity"
                )
            ),

        "post_curiosity":
            mean_or_none(
                immediate.get(
                    "curiosity"
                )
            ),

        "curiosity_change":
            curiosity_change,

        "baseline_understanding":
            mean_or_none(
                baseline.get(
                    "understanding"
                )
            ),

        "post_understanding":
            mean_or_none(
                immediate.get(
                    "understanding"
                )
            ),

        "understanding_change":
            understanding_change,

        "baseline_confidence":
            mean_or_none(
                baseline.get(
                    "confidence"
                )
            ),

        "post_confidence":
            mean_or_none(
                immediate.get(
                    "confidence"
                )
            ),

        "confidence_change":
            confidence_change,

        "follow_through_rate":
            follow_through_rate,

        "recall_rate":
            recall_rate,
    }


# ============================================================
# 22. FORWARD MODEL
# ============================================================

def generate_forward_model(
    client,
    model_name,
    event,
    design_data,
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
# 23. LIVE RECOMMENDATION
# ============================================================

def generate_live_recommendation(
    client,
    model_name,
    event,
    interaction,
    observations,
):

    observation_text = "\n".join(
        [
            (
                f"[{observation.evidence_level}] "
                f"{observation.category}: "
                f"{observation.detail}"
            )
            for observation in observations
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
{
    interaction.stated_preference
    or "None recorded."
}

CURRENT PHASE:
{interaction.phase}

RECENT OBSERVATIONS:
{
    observation_text
    or "No observations recorded."
}

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
# 24. COUNTERFACTUAL
# ============================================================

def generate_counterfactual(
    client,
    model_name,
    event,
    design,
    variable_change,
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
# 25. IMPACT INTERPRETATION
# ============================================================

def generate_impact_interpretation(
    client,
    model_name,
    metrics,
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
# 26. OUTCOME PREDICTION
# ============================================================

def generate_outcome_prediction(
    client,
    model_name,
    event,
    crowd_info,
    situation_info,
    questionnaire_context="",
):

    prompt = f"""
EVENT:
{event.name}

OBJECTIVE:
{event.objective}

TARGET AUDIENCE:
{event.target_audience}

ENVIRONMENT:
{event.environment}

SENSORY ENVIRONMENT:
{event.sensory_environment}

CROWD DETAILS:
{crowd_info}

SITUATIONAL CONTEXT:
{situation_info}

ENVIRONMENTAL QUESTIONNAIRE PARAMETERS:
{questionnaire_context}

Based on these parameters, predict scientifically grounded
estimated effects on participants.

Provide realistic estimates for:

1. focus_pct
2. stress_reduction_pct
3. cognitive_load_pct
4. attention_retention_pct
5. predicted_curiosity_shift
6. predicted_understanding_shift
7. predicted_engagement_rate
8. overall_outcome_narrative
9. risk_factors
10. success_amplifiers

These are model-generated hypotheses, not measurements.
Do not present them as physiological, neurological,
clinical, or causal facts.
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
# 27. HEADER
# ============================================================

render_html(
    """
<div class="hero">
    <div class="eyebrow">
        Ninolades Research Platform
    </div>

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
"""
)

st.markdown("---")


# ============================================================
# 28. HEADER CONTROLS
# ============================================================

header_col1, header_col2, header_col3 = st.columns(
    [1.5, 1, 3]
)


with header_col1:

    saved_label = (
        st.session_state.get(
            "mem_model_label",
            "Gemini 3.6 Flash",
        )
    )

    selected_model_label = st.selectbox(
        "Reasoning engine",
        list(model_options.keys()),
        index=(
            list(model_options.keys())
            .index(saved_label)
            if saved_label
            in model_options
            else 0
        ),
        key="mem_model_label",
        help=(
            "Choose the Gemini model used "
            "for generative analysis."
        ),
        label_visibility="collapsed",
    )

    selected_model = model_options[
        selected_model_label
    ]

    if (
        user_memory.selected_model
        != selected_model
    ):

        user_memory.selected_model = (
            selected_model
        )

        user_memory.updated_at = utc_now()

        db.commit()

    render_html(
        """
<div class="small-note" style="margin-top:8px;">
    Flash is default for live. Pro for deep analysis.
    Flash-Lite for volume.
</div>
"""
    )


with header_col2:

    if st.button(
        "Clean Memory",
        use_container_width=True,
    ):

        # Delete persistent user-specific application
        # memory without touching other users.
        db.query(
            MemoryEvent
        ).filter(
            MemoryEvent.user_id
            == USER_SESSION_TOKEN
        ).delete(
            synchronize_session=False
        )

        db.query(
            Observation
        ).filter(
            Observation.user_id
            == USER_SESSION_TOKEN
        ).delete(
            synchronize_session=False
        )

        db.query(
            Survey
        ).filter(
            Survey.user_id
            == USER_SESSION_TOKEN
        ).delete(
            synchronize_session=False
        )

        db.query(
            RapidStateLog
        ).filter(
            RapidStateLog.user_id
            == USER_SESSION_TOKEN
        ).delete(
            synchronize_session=False
        )

        db.query(
            Interaction
        ).filter(
            Interaction.user_id
            == USER_SESSION_TOKEN
        ).delete(
            synchronize_session=False
        )

        db.query(
            Event
        ).filter(
            Event.user_id
            == USER_SESSION_TOKEN
        ).delete(
            synchronize_session=False
        )

        db.query(
            UserMemory
        ).filter(
            UserMemory.user_id
            == USER_SESSION_TOKEN
        ).delete(
            synchronize_session=False
        )

        db.commit()

        st.session_state.clear()

        st.success(
            "Your private workspace memory has been cleared."
        )

        st.rerun()

    render_html(
        f"""
<div class="small-note"
     style="
        margin-top:8px;
        text-align:center;
     ">
    Version {APP_VERSION}
    <br>
    Private workspace
</div>
"""
    )


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

    current_page = st.session_state.get(
        "mem_page",
        page_opts[0],
    )

    if current_page not in page_opts:
        current_page = page_opts[0]

    page = st.radio(
        "Workspace",
        page_opts,
        index=page_opts.index(
            current_page
        ),
        key="mem_page",
        horizontal=True,
        label_visibility="collapsed",
    )

    if user_memory.current_page != page:

        user_memory.current_page = page
        user_memory.updated_at = utc_now()

        db.commit()


st.markdown("---")


# ============================================================
# 29. GEMINI STATUS
# ============================================================

if client is None:

    st.error(
        "Gemini is not configured. Add GEMINI_API_KEY "
        "to Streamlit secrets or the environment."
    )


# ============================================================
# 30. EXPERIENCE DESIGNER
# ============================================================

if page == "Experience Designer":

    render_html(
        """
<div class="section-heading">
    Design an outreach experience
</div>
"""
    )

    left, right = st.columns(2)

    with left:

        event_name = st.text_input(
            "Experience name",
            key="mem_event_name",
            placeholder=(
                "e.g. Science Under the Stars "
                "or Local Ecology Walk"
            ),
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
            ],
            key="mem_objective",
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
            ],
            key="mem_audience",
        )

    with right:

        environment = st.text_input(
            "Physical environment",
            key="mem_environment",
            placeholder=(
                "Dark-sky lawn, school courtyard, museum..."
            ),
        )

        acoustic_environment = st.text_input(
            "Acoustic / musical environment",
            key="mem_acoustic",
            placeholder=(
                "Silent, ambient sound, live acoustic..."
            ),
        )

        sensory_environment = st.text_input(
            "Relevant environmental conditions",
            key="mem_sensory",
            placeholder=(
                "Lighting, crowd density, temperature, noise..."
            ),
        )

    context = st.text_area(
        "Experience description",
        key="mem_context",
        placeholder=(
            "Describe what participants encounter, "
            "what the facilitator does, and the scientific content."
        ),
        height=130,
    )

    render_html(
        """
<div class="section-heading">
    Optional design variables
</div>
"""
    )

    design_col1, design_col2, design_col3 = st.columns(3)

    with design_col1:

        pacing = st.selectbox(
            "Pacing",
            [
                "Slow and contemplative",
                "Moderate",
                "Fast and energetic",
                "Variable",
            ],
            key="mem_pacing",
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
                "Mixed",
            ],
            key="mem_interaction_style",
        )

    with design_col3:

        optional_choice = st.selectbox(
            "Participant autonomy",
            [
                "High",
                "Moderate",
                "Low",
            ],
            key="mem_optional_choice",
        )

    design_data = f"""
Pacing: {pacing}
Interaction style: {interaction_style}
Participant autonomy: {optional_choice}
"""

    if st.button(
        "Create Experience Model",
        type="primary",
        use_container_width=True,
    ):

        if not event_name.strip():

            st.error(
                "Enter an experience name."
            )

        elif not context.strip():

            st.error(
                "Describe the experience."
            )

        elif client is None:

            st.error(
                "Gemini is unavailable."
            )

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

                st.session_state.active_event_id = (
                    event.id
                )

                st.session_state.active_interaction_id = (
                    None
                )

                with st.spinner(
                    "Building engagement model..."
                ):

                    model = generate_forward_model(
                        client,
                        selected_model,
                        event,
                        design_data,
                    )

                st.session_state.last_forward_model = (
                    model.model_dump()
                )

                st.session_state.last_recommendation = None

                persist_current_state()

                st.success(
                    "Experience initialized and model generated."
                )

            except Exception as exc:

                st.error(
                    f"Could not create the experience: {exc}"
                )

    if st.session_state.last_forward_model:

        model = (
            st.session_state.last_forward_model
        )

        render_html(
            """
<div class="section-heading">
    Engagement architecture
</div>
"""
        )

        render_html(
            f"""
<div class="premium-card">
    <div class="eyebrow">
        Current model
    </div>

    <div style="
        color:white;
        font-size:1.25rem;
        margin-bottom:10px;
    ">
        {clean_text(
            model["engagement_state"]
        )}
    </div>

    <div class="small-note">
        This is a generated hypothesis about possible
        engagement dynamics, not a measurement of participants.
    </div>
</div>
"""
        )

        cols = st.columns(3)

        for index, pathway in enumerate(
            model["predicted_pathways"]
        ):

            with cols[index]:

                render_html(
                    f"""
<div class="premium-card"
     style="height:100%;">

    <div class="eyebrow">
        Pathway {index + 1}
    </div>

    <h3 style="margin-top:0;">
        {clean_text(
            pathway["pathway"]
        )}
    </h3>

    <p>
        {clean_text(
            pathway["mechanism"]
        )}
    </p>

    <div class="small-note">
        Expected signal:<br>
        {clean_text(
            pathway["expected_signal"]
        )}
    </div>

    <br>

    <div class="small-note">
        Uncertainty:<br>
        {clean_text(
            pathway["uncertainty"]
        )}
    </div>

</div>
"""
                )

        c1, c2 = st.columns(2)

        with c1:

            render_html(
                """
<div class="section-heading">
    Design opportunities
</div>
"""
            )

            for item in model[
                "recommended_outreach_design"
            ]:

                st.markdown(
                    f"- {item}"
                )

        with c2:

            render_html(
                """
<div class="section-heading">
    Potential friction
</div>
"""
            )

            for item in model[
                "likely_friction_points"
            ]:

                st.markdown(
                    f"- {item}"
                )

        render_html(
            """
<div class="section-heading">
    What should be measured?
</div>
"""
        )

        for item in model[
            "measurement_opportunities"
        ]:

            st.markdown(
                f"- {item}"
            )


# ============================================================
# 31. OUTCOME PREDICTOR
# ============================================================

elif page == "Outcome Predictor":

    render_html(
        """
<div class="section-heading">
    Predict Outreach Outcomes
</div>
"""
    )

    events = get_user_events(db)

    if not events:

        st.info(
            "Create an experience in Experience Designer first."
        )

    else:

        event_map = {
            f"{event.name} "
            f"({event.date.strftime('%Y-%m-%d %H:%M')})":
                event
            for event in events
        }

        selected_name = st.selectbox(
            "Select Experience",
            list(event_map.keys()),
            key="pred_event_select",
        )

        event = event_map[
            selected_name
        ]

        render_html(
            """
<div class="section-heading">
    Contextual Environment & Crowd Questionnaire
</div>
"""
        )

        qc1, qc2 = st.columns(2)

        with qc1:

            baseline_stress_q = st.select_slider(
                "Estimated Baseline Audience Stress Level",
                options=[
                    "Very Low / Relaxed",
                    "Moderate Stress",
                    "High Stress / Overwhelmed",
                ],
                value="Moderate Stress",
                key="q_stress",
            )

            noise_sensory_q = st.select_slider(
                "Ambient Distraction & Sensory Noise Level",
                options=[
                    "Quiet & Controlled",
                    "Moderate Noise",
                    "High Loudness / Busy Crowd",
                ],
                value="Moderate Noise",
                key="q_noise",
            )

        with qc2:

            duration_q = st.selectbox(
                "Planned Session Duration",
                [
                    "Short (< 15 mins)",
                    "Standard (30-45 mins)",
                    "Extended (60+ mins)",
                ],
                key="q_duration",
            )

            interaction_density_q = st.selectbox(
                "Interactive Touchpoints Density",
                [
                    "Low (Passive listening)",
                    "Medium (Guided Q&A)",
                    "High (Hands-on exploration)",
                ],
                index=1,
                key="q_density",
            )

        c1, c2 = st.columns(2)

        with c1:

            crowd_info = st.text_area(
                "Crowd details & demographics",
                placeholder=(
                    "e.g. 50 enthusiastic middle schoolers, "
                    "mostly beginners, excited but easily distracted..."
                ),
                key="pred_crowd",
                height=110,
            )

        with c2:

            situation_info = st.text_area(
                "Situational context",
                placeholder=(
                    "e.g. Cloudy weather, noisy street nearby, "
                    "late evening after a long day..."
                ),
                key="pred_situation",
                height=110,
            )

        questionnaire_summary = (
            f"Baseline Stress: {baseline_stress_q}, "
            f"Noise level: {noise_sensory_q}, "
            f"Duration: {duration_q}, "
            f"Interaction density: "
            f"{interaction_density_q}"
        )

        if st.button(
            "Predict Outcome & Metrics",
            type="primary",
            use_container_width=True,
        ):

            if client is None:

                st.error(
                    "Gemini is unavailable."
                )

            elif (
                not crowd_info.strip()
                or not situation_info.strip()
            ):

                st.error(
                    "Please provide both crowd details "
                    "and situational context."
                )

            else:

                try:

                    with st.spinner(
                        "Analyzing parameters and "
                        "computing predicted outcomes..."
                    ):

                        prediction = (
                            generate_outcome_prediction(
                                client,
                                selected_model,
                                event,
                                crowd_info,
                                situation_info,
                                questionnaire_summary,
                            )
                        )

                    st.session_state.last_prediction = (
                        prediction.model_dump()
                    )

                    persist_current_state()

                except Exception as exc:

                    st.error(
                        f"Prediction failed: {exc}"
                    )

        if st.session_state.get(
            "last_prediction"
        ):

            prediction = (
                st.session_state.last_prediction
            )

            render_html(
                """
<div class="section-heading">
    Predicted Scientific Effects
</div>
"""
            )

            pc1, pc2, pc3, pc4 = st.columns(4)

            pc1.metric(
                "Predicted Focus State",
                f"{prediction.get('focus_pct', 75)}%",
            )

            pc2.metric(
                "Stress Reduction Index",
                f"{prediction.get('stress_reduction_pct', 60)}%",
            )

            pc3.metric(
                "Cognitive Load Level",
                f"{prediction.get('cognitive_load_pct', 45)}%",
            )

            pc4.metric(
                "Attention Retention",
                f"{prediction.get('attention_retention_pct', 80)}%",
            )

            st.markdown(
                "**Visual Effect Profile Comparison**"
            )

            chart_data = pd.DataFrame(
                {
                    "Metric": [
                        "Focus State",
                        "Stress Reduction",
                        "Cognitive Load",
                        "Attention Retention",
                    ],

                    "Percentage (%)": [
                        prediction.get(
                            "focus_pct",
                            75,
                        ),

                        prediction.get(
                            "stress_reduction_pct",
                            60,
                        ),

                        prediction.get(
                            "cognitive_load_pct",
                            45,
                        ),

                        prediction.get(
                            "attention_retention_pct",
                            80,
                        ),
                    ],
                }
            ).set_index(
                "Metric"
            )

            st.bar_chart(
                chart_data
            )

            render_html(
                """
<div class="section-heading">
    Predicted Shifts & Outcomes
</div>
"""
            )

            m1, m2, m3 = st.columns(3)

            with m1:

                render_html(
                    f"""
<div class="metric-card">
    <div class="metric-label">
        Curiosity Shift
    </div>

    <div class="metric-value"
         style="font-size:1.4rem;">
        {clean_text(
            prediction[
                "predicted_curiosity_shift"
            ]
        )}
    </div>
</div>
"""
                )

            with m2:

                render_html(
                    f"""
<div class="metric-card">
    <div class="metric-label">
        Understanding Shift
    </div>

    <div class="metric-value"
         style="font-size:1.4rem;">
        {clean_text(
            prediction[
                "predicted_understanding_shift"
            ]
        )}
    </div>
</div>
"""
                )

            with m3:

                render_html(
                    f"""
<div class="metric-card">
    <div class="metric-label">
        Engagement Rate
    </div>

    <div class="metric-value"
         style="font-size:1.4rem;">
        {clean_text(
            prediction[
                "predicted_engagement_rate"
            ]
        )}
    </div>
</div>
"""
                )

            render_html(
                """
<div class="section-heading">
    Outcome Narrative
</div>
"""
            )

            render_html(
                f"""
<div class="premium-card">
    <div style="
        color:#e5e5e7;
        line-height:1.7;
    ">
        {clean_text(
            prediction[
                "overall_outcome_narrative"
            ]
        )}
    </div>
</div>
"""
            )

            r1, r2 = st.columns(2)

            with r1:

                render_html(
                    """
<div class="section-heading">
    Risk Factors
</div>
"""
                )

                for item in prediction[
                    "risk_factors"
                ]:

                    st.markdown(
                        f"- {item}"
                    )

            with r2:

                render_html(
                    """
<div class="section-heading">
    Success Amplifiers
</div>
"""
                )

                for item in prediction[
                    "success_amplifiers"
                ]:

                    st.markdown(
                        f"- {item}"
                    )


# ============================================================
# 32. LIVE COPILOT
# ============================================================

elif page == "Live Copilot":

    render_html(
        """
<div class="section-heading">
    Live outreach copilot
</div>
"""
    )

    events = get_user_events(db)

    if not events:

        st.info(
            "Create an experience in Experience Designer first."
        )

    else:

        event_map = {
            f"{event.name} "
            f"({event.date.strftime('%Y-%m-%d %H:%M')})":
                event
            for event in events
        }

        chosen_name = st.selectbox(
            "Active experience",
            list(event_map.keys()),
            key="live_copilot_event_select",
        )

        event = event_map[
            chosen_name
        ]

        # Keep active event persisted.
        if (
            st.session_state.active_event_id
            != event.id
        ):

            st.session_state.active_event_id = (
                event.id
            )

            st.session_state.active_interaction_id = (
                None
            )

            st.session_state.last_recommendation = (
                None
            )

            persist_current_state()

        st.markdown("---")

        render_html(
            """
<div class="section-heading">
    Rapid Participant State Logger
</div>

<div class="small-note"
     style="margin-bottom:15px;">
    One-click logging for rapid visual analysis.
    Each click registers a new anonymous participant.
</div>
"""
        )

        rc1, rc2 = st.columns(2)

        baseline_opts = [
            "Calm / Receptive",
            "Neutral / Unengaged",
            "Low Energy / Fatigued",
            "Distracted / Scatterbrained",
            "Anxious / Stressed",
            "High Energy / Excited",
        ]

        state_opts = [
            "Awe / Wonder",
            "Deep Focus / Flow",
            "Curiosity / Inquisitive",
            "Epiphany / Sudden Understanding",
            "Cognitive Overload / Confusion",
            "Disengagement / Boredom",
            "Stress / Frustration",
            "Relaxation / Comfort",
        ]

        with rc1:

            rapid_baseline = st.radio(
                "Baseline Level",
                baseline_opts,
                key="rapid_base",
            )

        with rc2:

            rapid_state = st.radio(
                "State of Mind / Reaction",
                state_opts,
                key="rapid_state",
            )

        if st.button(
            "Log as New Participant",
            type="primary",
            use_container_width=True,
        ):

            rapid_log = RapidStateLog(

                id=str(
                    uuid.uuid4()
                ),

                user_id=USER_SESSION_TOKEN,

                event_id=event.id,

                participant_code=
                    f"RP-{uuid.uuid4().hex[:6].upper()}",

                timestamp=utc_now(),

                baseline_level=
                    rapid_baseline,

                current_state=
                    rapid_state,
            )

            db.add(rapid_log)
            db.commit()

            save_memory_event(
                db,
                "rapid_state_logged",
                {
                    "event_id":
                        event.id,

                    "participant_code":
                        rapid_log.participant_code,

                    "baseline":
                        rapid_baseline,

                    "reaction":
                        rapid_state,
                },
            )

            st.toast(
                "State logged."
            )

            st.rerun()

        recent_rapid_logs = (
            db.query(RapidStateLog)
            .filter(
                RapidStateLog.event_id
                == event.id,

                RapidStateLog.user_id
                == USER_SESSION_TOKEN,
            )
            .order_by(
                RapidStateLog.timestamp.desc()
            )
            .limit(10)
            .all()
        )

        if recent_rapid_logs:

            st.markdown(
                "**Recent Rapid Logs**"
            )

            for rapid_log in recent_rapid_logs:

                c_time, c_part, c_base, c_state, c_del = (
                    st.columns(
                        [
                            1.5,
                            1.5,
                            2.5,
                            2.5,
                            1,
                        ]
                    )
                )

                c_time.caption(
                    rapid_log.timestamp.strftime(
                        "%H:%M:%S UTC"
                    )
                )

                c_part.caption(
                    rapid_log.participant_code
                )

                c_base.caption(
                    rapid_log.baseline_level
                )

                c_state.caption(
                    rapid_log.current_state
                )

                if c_del.button(
                    "Delete",
                    key=f"del_rlog_{rapid_log.id}",
                ):

                    db.delete(
                        rapid_log
                    )

                    db.commit()

                    save_memory_event(
                        db,
                        "rapid_state_deleted",
                        {
                            "event_id":
                                event.id,

                            "participant_code":
                                rapid_log.participant_code,
                        },
                    )

                    st.toast(
                        "Rapid log deleted."
                    )

                    st.rerun()

        st.markdown("---")

        if not st.session_state.active_interaction_id:

            if st.button(
                "Start participant interaction",
                type="primary",
                use_container_width=True,
            ):

                interaction = create_interaction(
                    db,
                    event.id,
                )

                st.session_state.active_interaction_id = (
                    interaction.id
                )

                st.session_state.last_recommendation = (
                    None
                )

                persist_current_state()

                st.rerun()

        else:

            interaction = (
                db.query(
                    Interaction
                )
                .filter(
                    Interaction.id
                    ==
                    st.session_state.active_interaction_id,

                    Interaction.user_id
                    ==
                    USER_SESSION_TOKEN,
                )
                .first()
            )

            if interaction is None:

                st.session_state.active_interaction_id = (
                    None
                )

                persist_current_state()

                st.rerun()

            render_html(
                f"""
<div class="premium-card">

    <div class="eyebrow">
        Active interaction
    </div>

    <div style="
        color:white;
        font-size:1.2rem;
    ">
        {clean_text(
            interaction.participant_code
        )}
    </div>

    <div class="small-note">
        Anonymous interaction code.
    </div>

</div>
"""
            )

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

            phase_key = (
                f"phase_{interaction.id}"
            )

            if phase_key not in st.session_state:

                st.session_state[
                    phase_key
                ] = interaction.phase

            phase = st.selectbox(
                "Current phase",
                phase_opts,
                key=phase_key,
            )

            if phase != interaction.phase:

                interaction.phase = phase

                db.commit()

                save_memory_event(
                    db,
                    "interaction_phase_changed",
                    {
                        "interaction_id":
                            interaction.id,

                        "phase":
                            phase,
                    },
                )

                persist_current_state()

            preference_key = (
                f"pref_{interaction.id}"
            )

            if preference_key not in st.session_state:

                st.session_state[
                    preference_key
                ] = (
                    interaction.stated_preference
                    or ""
                )

            preference = st.text_input(
                "Participant-stated preference "
                "(Press Enter to save)",
                key=preference_key,
                placeholder=(
                    "Only record what the participant explicitly states."
                ),
            )

            if preference != (
                interaction.stated_preference
                or ""
            ):

                interaction.stated_preference = (
                    preference
                )

                db.commit()

                save_memory_event(
                    db,
                    "participant_preference_updated",
                    {
                        "interaction_id":
                            interaction.id,

                        "preference":
                            preference,
                    },
                )

                persist_current_state()

            render_html(
                """
<div class="section-heading">
    Quick Observations
</div>
"""
            )

            observation_buttons = [

                (
                    "Attention",
                    "Participant appears engaged/focused.",
                ),

                (
                    "Attention",
                    "Participant looks away/distracted.",
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
                    "Participant pauses to reflect.",
                ),

                (
                    "Friction",
                    "Participant has difficulty interacting.",
                ),

                (
                    "Friction",
                    "Environmental interruption occurs.",
                ),
            ]

            obs_cols = st.columns(4)

            for index, (
                category,
                detail,
            ) in enumerate(
                observation_buttons
            ):

                with obs_cols[
                    index % 4
                ]:

                    if st.button(
                        detail,
                        key=f"obs_btn_{index}",
                        use_container_width=True,
                    ):

                        log_observation(
                            db,
                            interaction.id,
                            category,
                            detail,
                            "OBSERVED",
                        )

                        st.toast(
                            "Observation recorded."
                        )

            with st.form(
                key=(
                    f"custom_obs_form_"
                    f"{interaction.id}"
                ),
                clear_on_submit=True,
            ):

                c1, c2 = st.columns(
                    [3, 1]
                )

                with c1:

                    custom_obs = st.text_input(
                        "Custom observation",
                        placeholder=(
                            "Describe only what was directly observed."
                        ),
                        label_visibility="collapsed",
                    )

                with c2:

                    submit_obs = (
                        st.form_submit_button(
                            "Log Observation",
                            use_container_width=True,
                        )
                    )

                if (
                    submit_obs
                    and custom_obs.strip()
                ):

                    log_observation(
                        db,
                        interaction.id,
                        "Custom",
                        custom_obs.strip(),
                        "OBSERVED",
                    )

                    st.toast(
                        "Custom observation recorded."
                    )

                    st.rerun()

            render_html(
                """
<div class="section-heading">
    Recent evidence
</div>
"""
            )

            observations = (
                get_recent_observations(
                    db,
                    interaction.id,
                )
            )

            if observations:

                for observation in observations:

                    render_html(
                        f"""
<div class="observation-row">

    <span class="badge">
        {clean_text(
            observation.evidence_level
        )}
    </span>

    <span style="
        margin-left:8px;
        color:#e5e5e7;
    ">
        {clean_text(
            observation.detail
        )}
    </span>

</div>
"""
                    )

            else:

                st.caption(
                    "No observations recorded yet."
                )

            render_html(
                """
<div class="section-heading">
    Adaptive guidance
</div>
"""
            )

            if st.button(
                "Generate next best outreach action",
                type="primary",
                use_container_width=True,
            ):

                if client is None:

                    st.error(
                        "Gemini is unavailable."
                    )

                else:

                    observations = (
                        get_recent_observations(
                            db,
                            interaction.id,
                        )
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
                                    observations,
                                )
                            )

                        st.session_state.last_recommendation = (
                            recommendation.model_dump()
                        )

                        persist_current_state()

                        save_memory_event(
                            db,
                            "recommendation_generated",
                            {
                                "interaction_id":
                                    interaction.id,

                                "model":
                                    selected_model,
                            },
                        )

                    except Exception as exc:

                        st.error(
                            f"Recommendation failed: {exc}"
                        )

            if st.session_state.last_recommendation:

                recommendation = (
                    st.session_state.last_recommendation
                )

                render_html(
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
            recommendation[
                "rationale"
            ]
        )}
    </div>

</div>
"""
                )

                c1, c2 = st.columns(2)

                with c1:

                    render_html(
                        f"""
<div class="metric-card">

    <div class="metric-label">
        Confidence
    </div>

    <div class="metric-value">
        {clean_text(
            recommendation[
                "confidence"
            ]
        )}
    </div>

</div>
"""
                    )

                with c2:

                    estimate = recommendation[
                        "cognitive_estimate"
                    ]

                    render_html(
                        f"""
<div class="metric-card">

    <div class="metric-label">
        Model-estimated focus
    </div>

    <div class="metric-value">
        {estimate["focus_pct"]}%
    </div>

    <div class="metric-sub">
        Hypothesis only; not a physiological
        measurement.
    </div>

</div>
"""
                    )

                render_html(
                    """
<div class="section-heading">
    Evidence used
</div>
"""
                )

                for evidence in recommendation[
                    "evidence"
                ]:

                    st.markdown(
                        f"- {evidence}"
                    )

                render_html(
                    """
<div class="section-heading">
    Alternative explanation
</div>
"""
                )

                st.write(
                    recommendation[
                        "alternative_explanation"
                    ]
                )

                render_html(
                    """
<div class="section-heading">
    Next observation to watch
</div>
"""
                )

                st.write(
                    recommendation[
                        "next_observation"
                    ]
                )

            st.markdown("---")

            if st.button(
                "End interaction and start next participant",
                use_container_width=True,
            ):

                interaction.ended_at = utc_now()

                db.commit()

                save_memory_event(
                    db,
                    "interaction_ended",
                    {
                        "interaction_id":
                            interaction.id,
                    },
                )

                st.session_state.active_interaction_id = (
                    None
                )

                st.session_state.last_recommendation = (
                    None
                )

                persist_current_state()

                st.rerun()


# ============================================================
# 33. SCIENTIFIC REACTIONS
# ============================================================

elif page == "Scientific Reactions":

    render_html(
        """
<div class="section-heading">
    Scientific Reaction Analysis
</div>

<div class="small-note"
     style="margin-bottom:15px;">
    Visualizing accurate, logical, and practical scientific
    reactions logged during the experience.
</div>
"""
    )

    events = get_user_events(db)

    if not events:

        st.info(
            "No experiences available."
        )

    else:

        event_map = {
            f"{event.name} "
            f"({event.date.strftime('%Y-%m-%d %H:%M')})":
                event
            for event in events
        }

        selected_name = st.selectbox(
            "Select Experience",
            list(event_map.keys()),
            key="sci_reac_event",
        )

        event = event_map[
            selected_name
        ]

        logs = (
            db.query(
                RapidStateLog
            )
            .filter(
                RapidStateLog.event_id
                == event.id,

                RapidStateLog.user_id
                == USER_SESSION_TOKEN,
            )
            .order_by(
                RapidStateLog.timestamp.asc()
            )
            .all()
        )

        if not logs:

            st.info(
                "No rapid reaction data logged for "
                "this event yet. Use the Rapid State "
                "Logger in the Live Copilot page."
            )

        else:

            df_logs = pd.DataFrame(
                [
                    {
                        "timestamp":
                            log.timestamp,

                        "participant":
                            log.participant_code,

                        "baseline":
                            log.baseline_level,

                        "reaction":
                            log.current_state,
                    }

                    for log in logs
                ]
            )

            df_logs[
                "timestamp"
            ] = pd.to_datetime(
                df_logs[
                    "timestamp"
                ]
            )

            render_html(
                f"""
<div class="metric-card">

    <div class="metric-label">
        Total Rapid Logs
    </div>

    <div class="metric-value">
        {len(logs)}
    </div>

</div>
"""
            )

            c1, c2 = st.columns(2)

            with c1:

                render_html(
                    """
<div class="section-heading">
    State of Mind / Reaction Distribution
</div>
"""
                )

                reaction_counts = (
                    df_logs[
                        "reaction"
                    ]
                    .value_counts()
                )

                st.bar_chart(
                    reaction_counts
                )

            with c2:

                render_html(
                    """
<div class="section-heading">
    Baseline Level Distribution
</div>
"""
                )

                baseline_counts = (
                    df_logs[
                        "baseline"
                    ]
                    .value_counts()
                )

                st.bar_chart(
                    baseline_counts
                )

            render_html(
                """
<div class="section-heading">
    Reaction Timeline
</div>
"""
            )

            df_logs[
                "time_minute"
            ] = (
                df_logs[
                    "timestamp"
                ]
                .dt.floor("min")
            )

            timeline_df = (
                df_logs
                .groupby(
                    [
                        "time_minute",
                        "reaction",
                    ]
                )
                .size()
                .unstack(
                    fill_value=0
                )
            )

            st.line_chart(
                timeline_df
            )

            render_html(
                """
<div class="section-heading">
    Raw Log Data
</div>
"""
            )

            st.dataframe(
                df_logs,
                use_container_width=True,
            )


# ============================================================
# 34. IMPACT OBSERVATORY
# ============================================================

elif page == "Impact Observatory":

    render_html(
        """
<div class="section-heading">
    Real-world impact observatory
</div>
"""
    )

    render_html(
        """
<div class="premium-card">

    <div class="eyebrow">
        Why this exists
    </div>

    <div style="
        color:#e5e5e7;
        line-height:1.7;
    ">
        The system separates what the AI predicts from what
        actually happened. Impact is calculated from recorded
        participant outcomes rather than generated by Gemini.
    </div>

</div>
"""
    )

    events = get_user_events(db)

    if not events:

        st.info(
            "No experiences have been created yet."
        )

    else:

        event_map = {
            f"{event.name} "
            f"({event.date.strftime('%Y-%m-%d %H:%M')})":
                event
            for event in events
        }

        selected_name = st.selectbox(
            "Experience",
            list(event_map.keys()),
            key="obs_event_select",
        )

        event = event_map[
            selected_name
        ]

        metrics = calculate_event_impact(
            db,
            event.id,
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
                unsafe_allow_html=True,
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
            if metrics[
                "curiosity_change"
            ] is None
            else
            f'{metrics["curiosity_change"]:+.2f}'
        }
    </div>

    <div class="metric-sub">
        Paired baseline → immediate
    </div>

</div>
""",
                unsafe_allow_html=True,
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
            if metrics[
                "understanding_change"
            ] is None
            else
            f'{metrics["understanding_change"]:+.2f}'
        }
    </div>

    <div class="metric-sub">
        Paired baseline → immediate
    </div>

</div>
""",
                unsafe_allow_html=True,
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
            if metrics[
                "follow_through_rate"
            ] is None
            else
            f'{metrics["follow_through_rate"]:.1f}%'
        }
    </div>

    <div class="metric-sub">
        Delayed self-report
    </div>

</div>
""",
                unsafe_allow_html=True,
            )

            render_html(
                """
<div class="section-heading">
    Outcome signals
</div>
"""
            )

            signal_data = {

                "Curiosity":
                    metrics[
                        "curiosity_change"
                    ],

                "Understanding":
                    metrics[
                        "understanding_change"
                    ],

                "Confidence":
                    metrics[
                        "confidence_change"
                    ],
            }

            signal_df = pd.DataFrame(
                [
                    {
                        "Signal":
                            name,

                        "Change":
                            value,
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
                    hide_index=True,
                )

            else:

                st.info(
                    "Paired baseline/post measurements "
                    "are needed to calculate change."
                )

            render_html(
                """
<div class="section-heading">
    Delayed indicators
</div>
"""
            )

            d1, d2 = st.columns(2)

            d1.metric(
                "Recall response rate",
                (
                    "—"
                    if metrics[
                        "recall_rate"
                    ] is None

                    else
                    f'{metrics["recall_rate"]:.1f}%'
                ),
            )

            d2.metric(
                "Participants with survey data",
                metrics["surveyed"],
            )

            render_html(
                """
<div class="section-heading">
    AI interpretation
</div>
"""
            )

            st.caption(
                "Gemini interprets the measurements below; "
                "it does not calculate them."
            )

            if st.button(
                "Interpret impact",
                type="primary",
                use_container_width=True,
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
                                    metrics,
                                )
                            )

                        st.session_state.last_impact_interpretation = (
                            interpretation.model_dump()
                        )

                        persist_current_state()

                    except Exception as exc:

                        st.error(
                            f"Impact interpretation failed: {exc}"
                        )

            if st.session_state.last_impact_interpretation:

                interpretation = (
                    st.session_state.last_impact_interpretation
                )

                render_html(
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
"""
                )

                c1, c2 = st.columns(2)

                with c1:

                    render_html(
                        """
<div class="section-heading">
    Strongest signal
</div>
"""
                    )

                    st.write(
                        interpretation[
                            "strongest_signal"
                        ]
                    )

                with c2:

                    render_html(
                        """
<div class="section-heading">
    Weakest signal
</div>
"""
                    )

                    st.write(
                        interpretation[
                            "weakest_signal"
                        ]
                    )

                render_html(
                    """
<div class="section-heading">
    Plausible mechanisms
</div>
"""
                )

                for item in interpretation[
                    "plausible_mechanisms"
                ]:

                    st.markdown(
                        f"- {item}"
                    )

                render_html(
                    """
<div class="section-heading">
    Alternative explanations
</div>
"""
                )

                for item in interpretation[
                    "alternative_explanations"
                ]:

                    st.markdown(
                        f"- {item}"
                    )

                render_html(
                    """
<div class="section-heading">
    Recommended next test
</div>
"""
                )

                st.info(
                    interpretation[
                        "recommended_next_test"
                    ]
                )


# ============================================================
# 35. COUNTERFACTUAL LAB
# ============================================================

elif page == "Counterfactual Lab":

    render_html(
        """
<div class="section-heading">
    Counterfactual experiment lab
</div>
"""
    )

    events = get_user_events(db)

    if not events:

        st.info(
            "Create an experience first."
        )

    else:

        event_map = {
            f"{event.name} "
            f"({event.date.strftime('%Y-%m-%d %H:%M')})":
                event
            for event in events
        }

        selected_name = st.selectbox(
            "Experience",
            list(event_map.keys()),
            key="cf_event_select",
        )

        event = event_map[
            selected_name
        ]

        render_html(
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
"""
        )

        variable_change = st.text_area(
            "What would you change?",
            key="cf_variable_change",
            placeholder=(
                "What if the live music stopped during "
                "direct observation?"
            ),
            height=100,
        )

        if (
            "cf_design_description"
            not in st.session_state
        ):

            st.session_state[
                "cf_design_description"
            ] = (
                event.context
                or ""
            )

        design_description = st.text_area(
            "Current design",
            key="cf_design_description",
            height=100,
        )

        if st.button(
            "Run counterfactual",
            type="primary",
            use_container_width=True,
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

                        counterfactual = (
                            generate_counterfactual(
                                client,
                                selected_model,
                                event,
                                design_description,
                                variable_change,
                            )
                        )

                    st.session_state.last_counterfactual = (
                        counterfactual.model_dump()
                    )

                    persist_current_state()

                except Exception as exc:

                    st.error(
                        f"Counterfactual failed: {exc}"
                    )

        if st.session_state.last_counterfactual:

            counterfactual = (
                st.session_state.last_counterfactual
            )

            render_html(
                f"""
<div class="premium-card">

    <div class="eyebrow">
        Changed variable
    </div>

    <h3 style="margin-top:0;">
        {clean_text(
            counterfactual[
                "changed_variable"
            ]
        )}
    </h3>

    <p>
        {clean_text(
            counterfactual[
                "expected_difference"
            ]
        )}
    </p>

</div>
"""
            )

            c1, c2 = st.columns(2)

            with c1:

                render_html(
                    f"""
<div class="premium-card">

    <div class="eyebrow">
        Baseline
    </div>

    <div style="
        color:#e5e5e7;
    ">
        {clean_text(
            counterfactual[
                "before_state"
            ]
        )}
    </div>

</div>
"""
                )

            with c2:

                render_html(
                    f"""
<div class="premium-card">

    <div class="eyebrow">
        Counterfactual
    </div>

    <div style="
        color:#e5e5e7;
    ">
        {clean_text(
            counterfactual[
                "after_state"
            ]
        )}
    </div>

</div>
"""
                )

            render_html(
                """
<div class="section-heading">
    Predicted effects
</div>
"""
            )

            for effect in counterfactual[
                "predicted_effects"
            ]:

                st.markdown(
                    f"- {effect}"
                )

            render_html(
                """
<div class="section-heading">
    Uncertainty
</div>
"""
            )

            st.warning(
                counterfactual[
                    "uncertainty"
                ]
            )


# ============================================================
# 36. METHODOLOGY
# ============================================================

elif page == "Methodology":

    render_html(
        """
<div class="section-heading">
    Science and methodology
</div>
"""
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
""",
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
""",
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
""",
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
""",
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
""",
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
""",
        ),

        (
            "Privacy and user memory",
            """
The application uses anonymous participant codes.

Application workspace memory is independently keyed to
the user's browser-local identifier.

The browser identifier is not placed into the shareable
application URL.

Refreshing the application therefore does not intentionally
create a new workspace.

A different browser receives a different workspace.

Public deployments should still implement appropriate
consent, retention, access-control, and deletion policies
before collecting real participant data.
""",
        ),
    ]

    for title, body in sections:

        render_html(
            f"""
<div class="premium-card">

    <div class="eyebrow">
        Method
    </div>

    <h3 style="margin-top:0;">
        {clean_text(title)}
    </h3>

    <div style="
        color:#a1a1aa;
        line-height:1.75;
        white-space:pre-line;
    ">
        {clean_text(body)}
    </div>

</div>
"""
        )


# ============================================================
# 37. OPTIONAL LIGHTWEIGHT IMPACT CAPTURE
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

    events = get_user_events(db)

    if not events:

        st.caption(
            "Create an experience first."
        )

    else:

        event_map = {
            f"{event.name} "
            f"({event.date.strftime('%Y-%m-%d %H:%M')})":
                event
            for event in events
        }

        selected_event_name = st.selectbox(
            "Experience",
            list(event_map.keys()),
            key="survey_event_select",
        )

        event = event_map[
            selected_event_name
        ]

        interactions = (
            db.query(
                Interaction
            )
            .filter(
                Interaction.event_id
                == event.id,

                Interaction.user_id
                == USER_SESSION_TOKEN,
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
                (
                    f"{interaction.participant_code} "
                    f"({interaction.started_at.strftime('%H:%M:%S')})"
                ):
                    interaction

                for interaction
                in interactions
            }

            selected_participant = st.selectbox(
                "Participant interaction",
                list(
                    interaction_map.keys()
                ),
                key="survey_participant_select",
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
                ],
                key="survey_timing_select",
            )

            with st.form(
                "outcome_capture",
                clear_on_submit=True,
            ):

                curiosity = st.slider(
                    "Curiosity",
                    1,
                    10,
                    5,
                )

                understanding = st.slider(
                    "Scientific understanding",
                    0,
                    100,
                    50,
                )

                confidence = st.slider(
                    "Confidence asking/answering questions",
                    1,
                    10,
                    5,
                )

                recall = st.text_area(
                    "What do you remember most?",
                    height=90,
                )

                follow_through = st.checkbox(
                    "I voluntarily explored something further afterward"
                )

                submitted = st.form_submit_button(
                    "Save outcome"
                )

                if submitted:

                    survey = Survey(

                        id=str(
                            uuid.uuid4()
                        ),

                        user_id=
                            USER_SESSION_TOKEN,

                        interaction_id=
                            interaction.id,

                        timing=
                            survey_timing,

                        curiosity=
                            float(curiosity),

                        understanding=
                            float(
                                understanding
                            ),

                        confidence=
                            float(
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

                    save_memory_event(
                        db,
                        "survey_recorded",
                        {
                            "interaction_id":
                                interaction.id,

                            "timing":
                                survey_timing,
                        },
                    )

                    st.success(
                        "Outcome recorded."
                    )


# ============================================================
# 38. TRUE LIVE GEMINI VOICE COMPONENT
# ============================================================

LIVE_VOICE_HTML = """
<div id="liveVoiceRoot">

    <button
        id="liveVoiceFab"
        class="liveVoiceFab"
        type="button"
        aria-label="Start Live AI Voice"
        aria-pressed="false"
    >
        <span
            id="liveVoiceRing"
            class="liveVoiceRing"
        ></span>

        <svg
            class="liveVoiceIcon"
            viewBox="0 0 24 24"
            aria-hidden="true"
        >
            <path
                d="
                M12 14.5
                a3.5 3.5 0 0 0
                3.5-3.5V6
                a3.5 3.5 0 0 0
                -7 0v5
                a3.5 3.5 0 0 0
                3.5 3.5Z
                "
            />

            <path
                d="
                M5 11
                a7 7 0 0 0
                14 0
                M12 18
                v3
                M8.5 21
                h7
                "
            />
        </svg>

        <span
            id="liveVoiceDot"
            class="liveVoiceDot"
        ></span>
    </button>

    <section
        id="liveVoicePanel"
        class="liveVoicePanel"
        aria-hidden="true"
    >

        <div class="liveVoiceHeader">

            <div>
                <div
                    id="liveVoiceTitle"
                    class="liveVoiceTitle"
                >
                    Live AI Voice
                </div>

                <div
                    id="liveVoiceStatus"
                    class="liveVoiceStatus"
                >
                    Off
                </div>
            </div>

            <button
                id="liveVoiceClose"
                class="liveVoiceClose"
                type="button"
                aria-label="Close panel"
            >
                ×
            </button>

        </div>

        <div
            id="liveVoiceTranscript"
            class="liveVoiceTranscript"
        >
            Start Live AI Voice to begin.
        </div>

        <div
            id="liveVoiceHint"
            class="liveVoiceHint"
        >
            Native Gemini realtime voice.
        </div>

    </section>

</div>
"""

LIVE_VOICE_CSS = """
#liveVoiceRoot {
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "SF Pro Display",
        "SF Pro Text",
        "Segoe UI",
        sans-serif;
}

.liveVoiceFab {
    position: fixed;

    right: 24px;
    bottom: 24px;

    width: 62px;
    height: 62px;

    border-radius: 50%;

    border:
        1px solid
        rgba(255,255,255,.16);

    background:
        rgba(18,18,21,.94);

    color:
        #f5f5f7;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    cursor:
        pointer;

    z-index:
        2147483000;

    box-shadow:
        0 12px 35px rgba(0,0,0,.42),
        0 0 0 1px rgba(255,255,255,.035);

    backdrop-filter:
        blur(18px);

    -webkit-backdrop-filter:
        blur(18px);

    transition:
        transform .2s ease,
        background .2s ease,
        border-color .2s ease,
        box-shadow .2s ease;
}

.liveVoiceFab:hover {
    transform:
        translateY(-2px)
        scale(1.025);

    background:
        rgba(28,28,32,.98);

    border-color:
        rgba(255,255,255,.28);

    box-shadow:
        0 16px 40px rgba(0,0,0,.5);
}

.liveVoiceFab:active {
    transform:
        scale(.97);
}

.liveVoiceFab.is-active {
    background:
        rgba(91,140,255,.96);

    border-color:
        rgba(255,255,255,.35);

    box-shadow:
        0 12px 38px
        rgba(91,140,255,.32);
}

.liveVoiceFab.is-error {
    border-color:
        rgba(248,113,113,.6);
}

.liveVoiceIcon {
    width:
        25px;

    height:
        25px;

    fill:
        none;

    stroke:
        currentColor;

    stroke-width:
        1.65;

    stroke-linecap:
        round;

    stroke-linejoin:
        round;

    position:
        relative;

    z-index:
        2;
}

.liveVoiceRing {
    position:
        absolute;

    inset:
        -5px;

    border-radius:
        50%;

    border:
        1px solid
        rgba(91,140,255,0);

    pointer-events:
        none;
}

.liveVoiceFab.is-listening
.liveVoiceRing {
    border-color:
        rgba(91,140,255,.65);

    animation:
        voicePulse 1.8s
        ease-out
        infinite;
}

.liveVoiceFab.is-speaking
.liveVoiceRing {
    border-color:
        rgba(255,255,255,.75);

    animation:
        voicePulse .95s
        ease-out
        infinite;
}

.liveVoiceDot {
    position:
        absolute;

    right:
        8px;

    bottom:
        8px;

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        #52525b;

    border:
        2px solid
        #121215;

    transition:
        background .2s ease;
}

.liveVoiceFab.is-listening
.liveVoiceDot {
    background:
        #4ade80;
}

.liveVoiceFab.is-speaking
.liveVoiceDot {
    background:
        #ffffff;
}

.liveVoiceFab.is-thinking
.liveVoiceDot {
    background:
        #fbbf24;
}

.liveVoiceFab.is-error
.liveVoiceDot {
    background:
        #f87171;
}

@keyframes voicePulse {

    0% {
        transform:
            scale(.95);

        opacity:
            .9;
    }

    70% {
        transform:
            scale(1.22);

        opacity:
            0;
    }

    100% {
        transform:
            scale(1.22);

        opacity:
            0;
    }
}

.liveVoicePanel {
    position:
        fixed;

    right:
        24px;

    bottom:
        98px;

    width:
        min(360px, calc(100vw - 32px));

    border:
        1px solid
        rgba(255,255,255,.10);

    border-radius:
        18px;

    background:
        rgba(15,15,18,.96);

    box-shadow:
        0 24px 70px
        rgba(0,0,0,.52);

    backdrop-filter:
        blur(24px);

    -webkit-backdrop-filter:
        blur(24px);

    padding:
        17px;

    z-index:
        2147482999;

    display:
        none;
}

.liveVoicePanel.open {
    display:
        block;

    animation:
        voicePanelIn .18s
        ease-out;
}

@keyframes voicePanelIn {

    from {
        opacity: 0;
        transform:
            translateY(8px)
            scale(.985);
    }

    to {
        opacity: 1;
        transform:
            translateY(0)
            scale(1);
    }
}

.liveVoiceHeader {
    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    gap:
        14px;
}

.liveVoiceTitle {
    color:
        #f5f5f7;

    font-size:
        .95rem;

    font-weight:
        550;

    letter-spacing:
        -.01em;
}

.liveVoiceStatus {
    margin-top:
        3px;

    color:
        #71717a;

    font-size:
        .72rem;

    letter-spacing:
        .08em;

    text-transform:
        uppercase;
}

.liveVoiceStatus.active {
    color:
        #4ade80;
}

.liveVoiceStatus.speaking {
    color:
        #f5f5f7;
}

.liveVoiceStatus.thinking {
    color:
        #fbbf24;
}

.liveVoiceStatus.error {
    color:
        #f87171;
}

.liveVoiceClose {
    width:
        30px;

    height:
        30px;

    border:
        0;

    border-radius:
        50%;

    background:
        rgba(255,255,255,.05);

    color:
        #a1a1aa;

    cursor:
        pointer;

    font-size:
        19px;

    line-height:
        1;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;
}

.liveVoiceClose:hover {
    background:
        rgba(255,255,255,.10);

    color:
        #ffffff;
}

.liveVoiceTranscript {
    margin-top:
        15px;

    min-height:
        68px;

    max-height:
        150px;

    overflow:
        auto;

    padding:
        13px;

    border:
        1px solid
        rgba(255,255,255,.07);

    border-radius:
        12px;

    background:
        rgba(255,255,255,.025);

    color:
        #d4d4d8;

    font-size:
        .84rem;

    line-height:
        1.55;
}

.liveVoiceHint {
    margin-top:
        10px;

    color:
        #52525b;

    font-size:
        .71rem;

    line-height:
        1.4;
}

@media (max-width: 640px) {

    .liveVoiceFab {
        right:
            16px;

        bottom:
            16px;

        width:
            58px;

        height:
            58px;
    }

    .liveVoicePanel {
        right:
            16px;

        bottom:
            86px;
    }
}
"""


LIVE_VOICE_JS = f"""
import {{ GoogleGenAI, Modality }} from
    "https://esm.sh/@google/genai@{LIVE_JS_SDK_VERSION}";

export default function(component) {{

    const {{
        parentElement,
        data,
        setStateValue,
        setTriggerValue
    }} = component;

    const fab =
        parentElement.querySelector(
            "#liveVoiceFab"
        );

    const panel =
        parentElement.querySelector(
            "#liveVoicePanel"
        );

    const status =
        parentElement.querySelector(
            "#liveVoiceStatus"
        );

    const transcript =
        parentElement.querySelector(
            "#liveVoiceTranscript"
        );

    const closeButton =
        parentElement.querySelector(
            "#liveVoiceClose"
        );

    let session = null;

    let stream = null;

    let audioContext = null;

    let processor = null;

    let sourceNode = null;

    let audioQueue = [];

    let playing = false;

    let stopped = true;

    let currentAudioSources = [];

    const LIVE_MODEL =
        "{LIVE_VOICE_MODEL}";

    function setStatus(
        label,
        mode = "off"
    ) {{

        status.textContent = label;

        status.className =
            "liveVoiceStatus";

        fab.classList.remove(
            "is-active",
            "is-listening",
            "is-speaking",
            "is-thinking",
            "is-error"
        );

        if (mode === "listening") {{

            status.classList.add(
                "active"
            );

            fab.classList.add(
                "is-active",
                "is-listening"
            );
        }}

        if (mode === "speaking") {{

            status.classList.add(
                "speaking"
            );

            fab.classList.add(
                "is-active",
                "is-speaking"
            );
        }}

        if (mode === "thinking") {{

            status.classList.add(
                "thinking"
            );

            fab.classList.add(
                "is-active",
                "is-thinking"
            );
        }}

        if (mode === "error") {{

            status.classList.add(
                "error"
            );

            fab.classList.add(
                "is-error"
            );
        }}

        setStateValue(
            "voice_state",
            {{
                state: mode,
                label: label
            }}
        );
    }}

    function setTranscript(
        value
    ) {{

        transcript.textContent =
            value || "";

        setStateValue(
            "transcript",
            value || ""
        );
    }}

    function openPanel() {{

        panel.classList.add(
            "open"
        );

        panel.setAttribute(
            "aria-hidden",
            "false"
        );
    }}

    function closePanel() {{

        panel.classList.remove(
            "open"
        );

        panel.setAttribute(
            "aria-hidden",
            "true"
        );
    }}

    function stopAudioPlayback() {{

        for (
            const source
            of currentAudioSources
        ) {{
            try {{
                source.stop();
            }} catch (_) {{}}
        }}

        currentAudioSources = [];

        audioQueue = [];

        playing = false;
    }}

    function pcm16ToFloat32(
        pcm
    ) {{

        const input =
            new Int16Array(
                pcm.buffer,
                pcm.byteOffset,
                Math.floor(
                    pcm.byteLength / 2
                )
            );

        const output =
            new Float32Array(
                input.length
            );

        for (
            let i = 0;
            i < input.length;
            i++
        ) {{

            output[i] =
                Math.max(
                    -1,
                    Math.min(
                        1,
                        input[i] / 32768
                    )
                );
        }}

        return output;
    }}

    async function playPcm16(
        pcm
    ) {{

        if (stopped) {{
            return;
        }}

        if (!audioContext) {{

            audioContext =
                new (
                    window.AudioContext ||
                    window.webkitAudioContext
                )({{
                    sampleRate: 24000
                }});
        }}

        if (
            audioContext.state
            === "suspended"
        ) {{
            await audioContext.resume();
        }}

        const floatData =
            pcm16ToFloat32(
                pcm
            );

        const buffer =
            audioContext.createBuffer(
                1,
                floatData.length,
                24000
            );

        buffer.copyToChannel(
            floatData,
            0
        );

        audioQueue.push(
            buffer
        );

        if (!playing) {{
            await drainAudioQueue();
        }}
    }}

    async function drainAudioQueue() {{

        if (
            playing ||
            stopped
        ) {{
            return;
        }}

        if (
            audioQueue.length === 0
        ) {{
            return;
        }}

        playing = true;

        const buffer =
            audioQueue.shift();

        const source =
            audioContext.createBufferSource();

        source.buffer =
            buffer;

        source.connect(
            audioContext.destination
        );

        currentAudioSources.push(
            source
        );

        setStatus(
            "Speaking",
            "speaking"
        );

        source.onended =
            () => {{

                currentAudioSources =
                    currentAudioSources.filter(
                        item =>
                            item !== source
                    );

                if (
                    audioQueue.length
                    > 0
                    &&
                    !stopped
                ) {{

                    playing = false;

                    drainAudioQueue();

                }} else {{

                    playing = false;

                    if (!stopped) {{

                        setStatus(
                            "Listening",
                            "listening"
                        );
                    }}
                }}
            }};

        source.start();
    }}

    function downsampleBuffer(
        buffer,
        inputRate,
        outputRate
    ) {{

        if (
            outputRate
            === inputRate
        ) {{
            return buffer;
        }}

        const ratio =
            inputRate /
            outputRate;

        const newLength =
            Math.round(
                buffer.length /
                ratio
            );

        const result =
            new Float32Array(
                newLength
            );

        let offsetResult = 0;
        let offsetBuffer = 0;

        while (
            offsetResult
            < result.length
        ) {{

            const nextOffsetBuffer =
                Math.round(
                    (
                        offsetResult + 1
                    )
                    * ratio
                );

            let accum = 0;
            let count = 0;

            for (
                let i =
                    offsetBuffer;

                i <
                    nextOffsetBuffer
                    &&
                    i <
                    buffer.length;

                i++
            ) {{

                accum +=
                    buffer[i];

                count++;
            }}

            result[
                offsetResult
            ] =
                count
                ? accum / count
                : 0;

            offsetResult++;

            offsetBuffer =
                nextOffsetBuffer;
        }}

        return result;
    }}

    function float32ToPcm16(
        float32
    ) {{

        const output =
            new Int16Array(
                float32.length
            );

        for (
            let i = 0;
            i < float32.length;
            i++
        ) {{

            const sample =
                Math.max(
                    -1,
                    Math.min(
                        1,
                        float32[i]
                    )
                );

            output[i] =
                sample < 0
                ? sample * 32768
                : sample * 32767;
        }}

        return output.buffer;
    }}

    async function startMicrophone() {{

        if (
            !navigator.mediaDevices
            ||
            !navigator.mediaDevices.getUserMedia
        ) {{

            throw new Error(
                "Microphone access is not supported by this browser."
            );
        }}

        stream =
            await navigator.mediaDevices
                .getUserMedia({{
                    audio: {{
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    }}
                }});

        audioContext =
            new (
                window.AudioContext ||
                window.webkitAudioContext
            )();

        await audioContext.resume();

        sourceNode =
            audioContext.createMediaStreamSource(
                stream
            );

        processor =
            audioContext.createScriptProcessor(
                4096,
                1,
                1
            );

        processor.onaudioprocess =
            async event => {{

                if (
                    stopped ||
                    !session
                ) {{
                    return;
                }}

                const input =
                    event.inputBuffer
                        .getChannelData(0);

                const downsampled =
                    downsampleBuffer(
                        input,
                        audioContext.sampleRate,
                        16000
                    );

                const pcm =
                    float32ToPcm16(
                        downsampled
                    );

                try {{

                    await session
                        .sendRealtimeInput({{
                            audio: {{
                                data: arrayBufferToBase64(
                                    pcm
                                ),
                                mimeType:
                                    "audio/pcm;rate=16000"
                            }}
                        }});

                }} catch (error) {{

                    console.error(
                        "Live audio send error",
                        error
                    );
                }}
            }};

        sourceNode.connect(
            processor
        );

        processor.connect(
            audioContext.destination
        );
    }}

    function arrayBufferToBase64(
        buffer
    ) {{

        const bytes =
            new Uint8Array(
                buffer
            );

        const chunkSize =
            0x8000;

        let binary = "";

        for (
            let i = 0;
            i < bytes.length;
            i += chunkSize
        ) {{

            const chunk =
                bytes.subarray(
                    i,
                    Math.min(
                        i + chunkSize,
                        bytes.length
                    )
                );

            binary +=
                String.fromCharCode(
                    ...chunk
                );
        }}

        return btoa(
            binary
        );
    }}

    function base64ToArrayBuffer(
        base64
    ) {{

        const binary =
            atob(base64);

        const bytes =
            new Uint8Array(
                binary.length
            );

        for (
            let i = 0;
            i < binary.length;
            i++
        ) {{
            bytes[i] =
                binary.charCodeAt(i);
        }}

        return bytes.buffer;
    }}

    async function stopMicrophone() {{

        if (processor) {{

            try {{
                processor.disconnect();
            }} catch (_) {{}}

            processor.onaudioprocess =
                null;

            processor = null;
        }}

        if (sourceNode) {{

            try {{
                sourceNode.disconnect();
            }} catch (_) {{}}

            sourceNode = null;
        }}

        if (stream) {{

            for (
                const track
                of stream.getTracks()
            ) {{
                track.stop();
            }}

            stream = null;
        }}
    }}

    async function startLive() {{

        if (
            !data
            ||
            !data.ephemeral_token
        ) {{

            throw new Error(
                "Live AI authentication token is unavailable."
            );
        }}

        stopped = false;

        openPanel();

        setStatus(
            "Connecting",
            "thinking"
        );

        setTranscript(
            "Connecting to Gemini Live..."
        );

        const ai =
            new GoogleGenAI({{
                apiKey:
                    data.ephemeral_token,
                apiVersion:
                    "v1beta"
            }});

        session =
            await ai.live.connect({{

                model:
                    LIVE_MODEL,

                config: {{

                    responseModalities: [
                        Modality.AUDIO
                    ],

                    inputAudioTranscription: {{}},

                    outputAudioTranscription: {{}},

                    systemInstruction:
                        data.system_instruction
                        ||
                        "You are a concise, calm, helpful voice assistant for the Ninolades Outreach Intelligence Lab. Do not diagnose participants or infer sensitive traits.",

                    realtimeInputConfig: {{
                        automaticActivityDetection: {{
                            disabled: false
                        }}
                    }}
                }},

                callbacks: {{

                    onopen() {{

                        setStatus(
                            "Listening",
                            "listening"
                        );

                        setTranscript(
                            "Listening..."
                        );
                    }},

                    onmessage(
                        message
                    ) {{

                        try {{

                            const server =
                                message.serverContent;

                            if (!server) {{
                                return;
                            }}

                            if (
                                server.inputTranscription
                                &&
                                server
                                    .inputTranscription
                                    .text
                            ) {{

                                setTranscript(
                                    "You: " +
                                    server
                                        .inputTranscription
                                        .text
                                );
                            }}

                            if (
                                server.outputTranscription
                                &&
                                server
                                    .outputTranscription
                                    .text
                            ) {{

                                setTranscript(
                                    "AI: " +
                                    server
                                        .outputTranscription
                                        .text
                                );
                            }}

                            if (
                                server.modelTurn
                            ) {{

                                setStatus(
                                    "Speaking",
                                    "speaking"
                                );

                                for (
                                    const part
                                    of server
                                        .modelTurn
                                        .parts
                                    || []
                                ) {{

                                    const inlineData =
                                        part.inlineData;

                                    if (
                                        inlineData
                                        &&
                                        inlineData.data
                                    ) {{

                                        const audio =
                                            base64ToArrayBuffer(
                                                inlineData.data
                                            );

                                        playPcm16(
                                            new Uint8Array(
                                                audio
                                            )
                                        );
                                    }}
                                }}
                            }}

                            if (
                                server.turnComplete
                                &&
                                !playing
                                &&
                                !stopped
                            ) {{

                                setStatus(
                                    "Listening",
                                    "listening"
                                );
                            }}

                        }} catch (error) {{

                            console.error(
                                "Live message error",
                                error
                            );
                        }}
                    }},

                    onerror(
                        error
                    ) {{

                        console.error(
                            "Gemini Live error",
                            error
                        );

                        setStatus(
                            "Voice error",
                            "error"
                        );

                        setTranscript(
                            "The Live AI connection encountered an error."
                        );
                    }},

                    onclose(
                        event
                    ) {{

                        console.info(
                            "Gemini Live closed",
                            event
                        );

                        if (!stopped) {{

                            setStatus(
                                "Disconnected",
                                "error"
                            );
                        }}
                    }}
                }}
            }});

        await startMicrophone();

        setStateValue(
            "voice_active",
            true
        );
    }}

    async function stopLive() {{

        stopped = true;

        stopAudioPlayback();

        await stopMicrophone();

        if (session) {{

            try {{
                session.close();
            }} catch (_) {{}}

            session = null;
        }}

        setStatus(
            "Off",
            "off"
        );

        setTranscript(
            "Live AI Voice is off."
        );

        setStateValue(
            "voice_active",
            false
        );
    }}

    async function toggleLive() {{

        try {{

            if (stopped) {{

                await startLive();

            }} else {{

                await stopLive();
            }}

        }} catch (error) {{

            console.error(
                "Live Voice startup error",
                error
            );

            stopped = true;

            await stopMicrophone();

            setStatus(
                "Unavailable",
                "error"
            );

            setTranscript(
                error &&
                error.message
                ? error.message
                : "Live AI Voice could not start."
            );

            setStateValue(
                "voice_active",
                false
            );
        }}
    }}

    fab.onclick =
        toggleLive;

    closeButton.onclick =
        closePanel;

    // Clicking the panel does not stop the
    // session. It only hides the expanded panel.
    panel.onclick =
        event => {{
            event.stopPropagation();
        }};

    // Start completely OFF.
    stopped = true;

    setStatus(
        "Off",
        "off"
    );

    setTranscript(
        "Start Live AI Voice to begin."
    );

    return () => {{

        stopped = true;

        try {{
            stopAudioPlayback();
        }} catch (_) {{}}

        try {{
            stopMicrophone();
        }} catch (_) {{}}

        try {{
            if (session) {{
                session.close();
            }}
        }} catch (_) {{}}

        session = null;
    }};
}}
"""


# ------------------------------------------------------------
# Only create a token when the voice component actually needs
# one. The token itself is short-lived.
# ------------------------------------------------------------

if (
    "live_voice_token"
    not in st.session_state
):

    st.session_state[
        "live_voice_token"
    ] = None


# A button cannot directly signal the v2 component before
# mounting, so the component initially receives no token.
#
# The component will remain visibly OFF until a valid token is
# supplied.

live_voice_token = (
    st.session_state[
        "live_voice_token"
    ]
)


# ============================================================
# 39. VOICE TOKEN BUTTON
# ============================================================

# We use a tiny hidden state channel to request a token from
# Python. The visible floating button remains entirely inside
# the Live component.

if (
    "voice_token_request"
    not in st.session_state
):

    st.session_state[
        "voice_token_request"
    ] = False


# ============================================================
# 40. LIVE VOICE COMPONENT MOUNT
# ============================================================

try:

    live_voice_component = (
        st.components.v2.component(
            name="ninolades_live_ai_voice",
            html=LIVE_VOICE_HTML,
            css=LIVE_VOICE_CSS,
            js=LIVE_VOICE_JS,
            isolate_styles=False,
        )
    )

    live_voice_result = (
        live_voice_component(

            key="ninolades_live_voice",

            data={

                "ephemeral_token":
                    live_voice_token,

                "system_instruction":
                    """
You are the live voice assistant for
the Ninolades Outreach Intelligence Lab.

Be concise, clear, calm and useful.

Help the facilitator with the current outreach
workflow.

Never diagnose participants.

Never infer protected or sensitive traits.

Never present hypotheses as facts.

Respect participant autonomy.

If uncertain, say so.

Keep spoken answers naturally concise.
""",
            },

            default={
                "voice_active":
                    False,

                "voice_state":
                    {
                        "state":
                            "off",

                        "label":
                            "Off",
                    },

                "transcript":
                    "",
            },
        )
    )

except Exception as voice_component_error:

    live_voice_result = None

    st.error(
        "Live AI Voice could not initialize. "
        f"Details: {voice_component_error}"
    )


# ============================================================
# 41. LIVE VOICE TOKEN PROVISIONING
# ============================================================

# The component reports state. If the user activates the voice
# control while no token exists, we cannot expose the permanent
# API key to the browser. A production-quality implementation
# therefore provisions a short-lived token server-side.

#
# Because the floating component itself cannot synchronously
# request a token from Python without causing a Streamlit rerun,
# we detect the active request and provision a fresh token.
#

voice_active = False

if live_voice_result is not None:

    voice_active = bool(
        getattr(
            live_voice_result,
            "voice_active",
            False,
        )
    )

    voice_state = getattr(
        live_voice_result,
        "voice_state",
        None,
    )

    if voice_state:

        try:

            voice_state_dict = dict(
                voice_state
            )

        except Exception:

            voice_state_dict = {}

    else:

        voice_state_dict = {}


# ============================================================
# 42. IMPORTANT LIVE TOKEN FLOW
# ============================================================

# If the component is being mounted for the first time with no
# token, we cannot know whether the user intends to start voice
# until the browser component requests it.
#
# The component therefore exposes the following behavior:
#
#   1. OFF icon is always visible.
#   2. Clicking it requires a token.
#   3. Python provisions the token on the next rerun.
#
# To make that request explicit, the component uses a trigger
# when it detects that no token is available.

#
# NOTE:
# The current Live API architecture requires the client to
# receive an ephemeral token before opening the WebSocket.
#
# We therefore use a lightweight browser-side token request
# trigger.
#

# Re-registering the component with token provisioning requires
# the component JS to trigger the request before attempting
# connection.

# The following state check catches that trigger when supported
# by the current component runtime.

voice_request = None

if live_voice_result is not None:

    voice_request = getattr(
        live_voice_result,
        "request_token",
        None,
    )


# ============================================================
# 43. TOKEN REQUEST HANDLER
# ============================================================

if (
    voice_request
    and not live_voice_token
):

    try:

        with st.spinner(
            "Preparing Live AI Voice..."
        ):

            st.session_state[
                "live_voice_token"
            ] = (
                create_live_ephemeral_token()
            )

        st.rerun()

    except Exception as exc:

        st.error(
            f"Live AI Voice authentication failed: {exc}"
        )


# ============================================================
# 44. VOICE MEMORY PERSISTENCE
# ============================================================

if live_voice_result is not None:

    if voice_active != (
        user_memory.voice_enabled
    ):

        user_memory.voice_enabled = (
            voice_active
        )

        user_memory.updated_at = utc_now()

        db.commit()

        save_memory_event(
            db,
            "voice_state_changed",
            {
                "active":
                    voice_active,
            },
        )


# ============================================================
# 45. FOOTER
# ============================================================

render_html(
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
"""
)


# ============================================================
# 46. PERSIST CURRENT STATE
# ============================================================

try:
    persist_current_state()
except Exception:
    # Never allow a memory persistence failure to destroy
    # the primary application UI.
    db.rollback()


# ============================================================
# 47. DATABASE CLEANUP
# ============================================================

db.close()
