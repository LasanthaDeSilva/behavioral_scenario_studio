"""
NINOLADES OUTREACH INTELLIGENCE LAB
===================================

A self-contained Streamlit application for:
    - Designing outreach scenarios
    - Modeling likely engagement pathways
    - Predicting real-world outcomes and metrics
    - Running a live outreach copilot
    - Recording lightweight observations
    - Comparing predicted vs observed engagement
    - Measuring optional real-world impact
    - Exploring counterfactual interventions
    - Predicting scenario impact across HEXACO personality profiles
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
import hashlib
import secrets
import hmac
import json
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

# PDF reporting is additive: it does not alter any existing analysis,
# storage, navigation, or interaction logic.
from io import BytesIO
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

APP_TITLE = "Outreach Intelligence Lab"
APP_VERSION = "1.4.1"

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

# Account authentication below provides the durable identity boundary.
# The previous anonymous localStorage/session-token mechanism is intentionally
# superseded so shared links cannot select another account's workspace.

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
    """Create a database engine that is safe for Streamlit reruns.

    SQLite uses NullPool so connections cannot accumulate across concurrent
    Streamlit sessions/reruns and exhaust a QueuePool.
    """
    engine_kwargs = {"pool_pre_ping": True}
    if DATABASE_URL.startswith("sqlite"):
        from sqlalchemy.pool import NullPool

        engine_kwargs["poolclass"] = NullPool
        engine_kwargs["connect_args"] = {
            "check_same_thread": False,
            "timeout": 30,
        }

    return create_engine(DATABASE_URL, **engine_kwargs)

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
    pacing = Column(String, nullable=True)
    interaction_style = Column(String, nullable=True)
    participant_autonomy = Column(String, nullable=True)

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

    # Exact save time used to select the most recent record when a participant
    # has more than one record at the same measurement point. Nullable so
    # existing survey records remain fully compatible.
    recorded_at = Column(DateTime, nullable=True)

    curiosity = Column(Float, nullable=True)
    understanding = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)

    recall_text = Column(Text, nullable=True)

    # Optional descriptive state records captured separately before and after
    # the outreach interaction. These are not clinical or physiological
    # measurements; they may contain participant-reported descriptions or
    # directly observed descriptions only.
    before_state = Column(Text, nullable=True)
    after_state = Column(Text, nullable=True)

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

class PersonalityProfile(Base):
    __tablename__ = "personality_profiles"
    
    id = Column(String, primary_key=True)
    session_token = Column(String, nullable=False, default="global")
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    
    # HEXACO + Resilience Metrics
    hexaco_h = Column(Integer, default=50) # Honesty-Humility
    hexaco_e = Column(Integer, default=50) # Emotionality
    hexaco_x = Column(Integer, default=50) # Extraversion
    hexaco_a = Column(Integer, default=50) # Agreeableness
    hexaco_c = Column(Integer, default=50) # Conscientiousness
    hexaco_o = Column(Integer, default=50) # Openness
    resilience_baseline = Column(Integer, default=50) 


class UserAccount(Base):
    __tablename__ = "user_accounts"
    id = Column(String, primary_key=True)
    username = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    last_login_at = Column(DateTime, nullable=True)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("user_accounts.id"), nullable=False, index=True)
    token_hash = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, nullable=False)
    last_seen_at = Column(DateTime, nullable=False)
    revoked = Column(Boolean, nullable=False, default=False)


class AppState(Base):
    __tablename__ = "app_states"
    user_id = Column(String, ForeignKey("user_accounts.id"), primary_key=True)
    state_json = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, nullable=False)


def _initialize_database_schema():
    """Create the existing schema while tolerating brief SQLite lock contention."""
    last_error = None
    for attempt in range(4):
        try:
            Base.metadata.create_all(bind=engine)
            return
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                import time
                time.sleep(0.5 * (attempt + 1))
    raise last_error


_initialize_database_schema()

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

try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE personality_profiles ADD COLUMN session_token VARCHAR DEFAULT 'global'"))
        conn.commit()
except Exception:
    pass

# Experience Designer persistence migrations. These columns store the
# existing optional design variables so saved experiences can be reopened
# and edited without changing their existing interactions, observations,
# surveys, rapid-state logs, or generated analysis elsewhere in the app.
for _column, _default in (
    ("pacing", "Slow and contemplative"),
    ("interaction_style", "Open observation"),
    ("participant_autonomy", "High"),
):
    try:
        # SQLite does not accept bound parameters in ALTER TABLE ADD COLUMN
        # DEFAULT clauses, so quote these fixed migration defaults explicitly.
        _safe_default = _default.replace("'", "''")
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"ALTER TABLE events ADD COLUMN {_column} VARCHAR "
                    f"DEFAULT '{_safe_default}'"
                )
            )
            conn.commit()
    except Exception:
        pass

# Optional participant outcome state-capture migrations. These are additive:
# existing survey records remain valid and the new fields are nullable.
for _column, _sql_type in (
    ("before_state", "TEXT"),
    ("after_state", "TEXT"),
    ("recorded_at", "DATETIME"),
):
    try:
        with engine.connect() as conn:
            conn.execute(
                text(
                    f"ALTER TABLE surveys ADD COLUMN {_column} {_sql_type}"
                )
            )
            conn.commit()
    except Exception:
        # The column already exists on an upgraded installation.
        pass

SessionLocal = get_session_factory(engine)

def db_session():
    return SessionLocal()


# ============================================================
# 4B. ACCOUNT SECURITY & PERSISTENT STATE
# ============================================================

PASSWORD_ITERATIONS = 600_000
PASSWORD_SALT_BYTES = 16
AUTH_TOKEN_BYTES = 32
MIN_PASSWORD_LENGTH = 8


def _normalize_username(username: str) -> str:
    return (username or "").strip().lower()


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"),
            bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(actual, bytes.fromhex(digest_hex))
    except (ValueError, TypeError):
        return False


def _hash_auth_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _new_auth_session(db, user_id: str) -> str:
    raw = secrets.token_urlsafe(AUTH_TOKEN_BYTES)
    now = datetime.now(timezone.utc)
    db.add(AuthSession(
        id=str(uuid.uuid4()), user_id=user_id,
        token_hash=_hash_auth_token(raw), created_at=now,
        last_seen_at=now, revoked=False
    ))
    db.commit()
    return raw


def _revoke_auth_token(db, raw_token: str):
    if not raw_token:
        return
    auth = db.query(AuthSession).filter(
        AuthSession.token_hash == _hash_auth_token(raw_token),
        AuthSession.revoked == False
    ).first()
    if auth:
        auth.revoked = True
        db.commit()


def _authenticate_token(db, raw_token: str):
    if not raw_token:
        return None
    auth = db.query(AuthSession).filter(
        AuthSession.token_hash == _hash_auth_token(raw_token),
        AuthSession.revoked == False
    ).first()
    if not auth:
        return None
    auth.last_seen_at = datetime.now(timezone.utc)
    db.commit()
    return db.query(UserAccount).filter(UserAccount.id == auth.user_id).first()


def _safe_state_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_state_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_state_value(v) for k, v in value.items()}
    return None


def _is_persistable_state_key(key: str) -> bool:
    """Return True only for application state safe to restore.

    Streamlit also stores buttons and form-submit widgets in session_state.
    Restoring those transient values is forbidden and can raise
    StreamlitValueAssignmentNotAllowedError on the next render.
    """
    key = str(key)

    protected = {
        "user_session_token",
        "authenticated_user_id",
        "authenticated_username",
        "auth_token",
        "session_verified_locally",
        "ls_synced",
        "_trigger_ls_update",
        "confirm_clean_memory",
        "confirm_clean_memory_checkbox",
        "persistent_state_restored",
        "experience_saved_selector_pending",
        "experience_editor_load_pending",
        "experience_new_program_pending",
    }
    if key in protected or key.startswith("_"):
        return False

    if key.startswith("FormSubmitter:") or key.startswith("$$"):
        return False

    transient_prefixes = (
        "edit_saved_event_",
        "del_prof_",
        "del_rlog_",
        "obs_btn_",
    )
    if key.startswith(transient_prefixes):
        return False

    return True


def _persistent_state_snapshot():
    result = {}
    for key, value in st.session_state.items():
        if not _is_persistable_state_key(str(key)):
            continue
        safe = _safe_state_value(value)
        try:
            json.dumps(safe, ensure_ascii=False, allow_nan=False)
            result[str(key)] = safe
        except (TypeError, ValueError):
            pass
    return result


def restore_persistent_state(db, user_id: str):
    record = db.query(AppState).filter(AppState.user_id == user_id).first()
    if not record or not record.state_json:
        return
    try:
        payload = json.loads(record.state_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if _is_persistable_state_key(str(key)):
            if key not in st.session_state:
                st.session_state[key] = value


def persist_app_state(db, user_id: str):
    payload = json.dumps(
        _persistent_state_snapshot(), ensure_ascii=False,
        separators=(",", ":"), allow_nan=False
    )
    record = db.query(AppState).filter(AppState.user_id == user_id).first()
    now = datetime.now(timezone.utc)
    if record:
        record.state_json = payload
        record.updated_at = now
    else:
        db.add(AppState(
            user_id=user_id, state_json=payload, updated_at=now
        ))
    db.commit()


def _set_auth_query_token(raw_token: str):
    st.query_params["auth_session"] = raw_token
    try:
        del st.query_params["session_id"]
    except Exception:
        pass


def _clear_auth_query_token():
    for key in ("auth_session", "session_id"):
        try:
            del st.query_params[key]
        except Exception:
            pass


# ============================================================
# 4C. LOGIN / REGISTRATION GATE
# ============================================================

auth_db = db_session()
raw_auth_token = st.query_params.get("auth_session")
authenticated_user = _authenticate_token(auth_db, raw_auth_token)

if authenticated_user:
    # 1. User is valid: Persist the session to localStorage so they stay logged in across refreshes/restarts.
    components.html(f"""
        <script>
        try {{
            window.parent.localStorage.setItem("outreach_auth_session", "{raw_auth_token}");
        }} catch (e) {{}}
        </script>
    """, height=0, width=0)

else:
    if raw_auth_token:
        # 2. An invalid or expired token was found in the URL. Clear it out so the user starts cleanly logged out.
        _clear_auth_query_token()
        components.html("""
            <script>
            (function () {
                try {
                    window.parent.localStorage.removeItem("outreach_auth_session");
                    const url = new URL(window.parent.location);
                    url.searchParams.delete("auth_session");
                    window.parent.history.replaceState({}, '', url);
                } catch (e) {}
            })();
            </script>
        """, height=0, width=0)
    else:
        # 3. No token present. Check if the device has a valid saved session to auto-login.
        components.html("""
            <script>
            (function () {
                try {
                    const saved = window.parent.localStorage.getItem("outreach_auth_session");
                    if (saved) {
                        const url = new URL(window.parent.location);
                        url.searchParams.set("auth_session", saved);
                        window.parent.location.replace(url.toString());
                    }
                } catch (e) {}
            })();
            </script>
        """, height=0, width=0)

    st.markdown("## Account")
    st.caption("Log in or register to keep your Outreach Intelligence Lab workspace saved independently across refreshes and browser restarts.")
  

    login_tab, register_tab = st.tabs(["Log in", "Register"])

    with login_tab:
        with st.form("account_login_form", clear_on_submit=False):
            login_username = st.text_input("Username")
            login_password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button(
                "Log in", type="primary", use_container_width=True
            )
        if login_submitted:
            account = auth_db.query(UserAccount).filter(
                UserAccount.username == _normalize_username(login_username)
            ).first()
            if account and _verify_password(login_password, account.password_hash):
                token = _new_auth_session(auth_db, account.id)
                account.last_login_at = datetime.now(timezone.utc)
                auth_db.commit()
                _set_auth_query_token(token)
                auth_db.close()
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        with st.form("account_register_form", clear_on_submit=False):
            register_username = st.text_input("Choose a username")
            register_password = st.text_input(
                "Create a password", type="password"
            )
            register_confirm = st.text_input(
                "Confirm password", type="password"
            )
            register_submitted = st.form_submit_button(
                "Create account", type="primary", use_container_width=True
            )
        if register_submitted:
            username = _normalize_username(register_username)
            exists = auth_db.query(UserAccount).filter(
                UserAccount.username == username
            ).first()
            if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,63}", username):
                st.error(
                    "Username must be 3-64 characters and use only letters, "
                    "numbers, periods, underscores, or hyphens."
                ) 
            elif register_password != register_confirm:
                st.error("Passwords do not match.")
            elif exists is not None:
                st.error("That username is already registered.")
            else:
                account = UserAccount(
                    id=str(uuid.uuid4()),
                    username=username,
                    password_hash=_hash_password(
                        register_password,
                        secrets.token_bytes(PASSWORD_SALT_BYTES)
                    ),
                    created_at=datetime.now(timezone.utc),
                )
                auth_db.add(account)
                auth_db.commit()
                token = _new_auth_session(auth_db, account.id)
                account.last_login_at = datetime.now(timezone.utc)
                auth_db.commit()
                _set_auth_query_token(token)
                auth_db.close()
                st.rerun()

    auth_db.close()
    st.stop()

USER_ACCOUNT_ID = authenticated_user.id
USER_USERNAME = authenticated_user.username
USER_SESSION_TOKEN = f"user_{USER_ACCOUNT_ID}"

st.session_state["authenticated_user_id"] = USER_ACCOUNT_ID
st.session_state["authenticated_username"] = USER_USERNAME
st.session_state["auth_token"] = raw_auth_token

if "persistent_state_restored" not in st.session_state:
    restore_persistent_state(auth_db, USER_ACCOUNT_ID)
    st.session_state["persistent_state_restored"] = True

auth_db.close()


def app_rerun():
    """Persist current workspace state, release DB resources, then rerun."""
    try:
        persist_app_state(db, USER_ACCOUNT_ID)
    except Exception:
        try:
            db.rollback()
        except Exception:
            pass
    finally:
        try:
            db.close()
        except Exception:
            pass
    st.rerun()


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


class TraitImpact(BaseModel):
    personality_name: str
    focus_shift_pct: int = Field(ge=-100, le=100, description="Change in focus level percentage")
    stress_level_pct: int = Field(ge=0, le=100, description="Predicted final stress level percentage")
    resilience_activation_pct: int = Field(ge=0, le=100, description="Percentage of resilience capacity utilized to adapt")
    cognitive_load_pct: int = Field(ge=0, le=100, description="Predicted cognitive load during the scenario")
    behavioral_response: str = Field(description="Scientific narrative of how this profile responds behaviorally")
    friction_points: List[str] = Field(min_length=1, max_length=3)

class PersonalityPredictorResponse(BaseModel):
    overall_scenario_dynamics: str = Field(description="Summary of how the diverse personalities interact with the event overall")
    impacts: List[TraitImpact]


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
    "last_personality_prediction": None,
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
    target_audience,
    pacing="Slow and contemplative",
    interaction_style="Open observation",
    participant_autonomy="High",
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
        pacing=pacing,
        interaction_style=interaction_style,
        participant_autonomy=participant_autonomy,
    )

    db.add(event)
    db.commit()

    return event


def update_event(
    db,
    event,
    name,
    objective,
    context,
    environment,
    sensory_environment,
    acoustic_environment,
    target_audience,
    pacing,
    interaction_style,
    participant_autonomy,
):
    """Update only the saved Experience Designer fields for one user's event."""
    if event is None or event.session_token != USER_SESSION_TOKEN:
        raise ValueError("The selected experience is not available in this workspace.")

    event.name = name.strip()
    event.objective = objective
    event.context = context.strip()
    event.environment = environment.strip()
    event.sensory_environment = sensory_environment.strip()
    event.acoustic_environment = acoustic_environment.strip()
    event.target_audience = target_audience
    event.pacing = pacing
    event.interaction_style = interaction_style
    event.participant_autonomy = participant_autonomy
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


def delete_observation(db, interaction_id, observation_id):
    """Delete one observation only when it belongs to the supplied interaction."""
    observation = (
        db.query(Observation)
        .filter(
            Observation.id == observation_id,
            Observation.interaction_id == interaction_id,
        )
        .first()
    )
    if observation is None:
        return False

    db.delete(observation)
    db.commit()
    return True


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
    """Calculate descriptive event outcomes from recorded participant data.

    Baseline and immediate values are paired within the same interaction.
    If multiple records exist at one measurement point, the most recently
    recorded complete record is used. Missing values are not treated as zero.
    These calculations describe the recorded sample and do not establish
    causation.
    """

    interactions = get_event_interactions(db, event_id)

    if not interactions:
        return None

    interaction_ids = [i.id for i in interactions]

    surveys = (
        db.query(Survey)
        .filter(Survey.interaction_id.in_(interaction_ids))
        .order_by(Survey.id.asc())
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
            "recorded_at": s.recorded_at,
            "survey_id": s.id,
            "curiosity": s.curiosity,
            "understanding": s.understanding,
            "confidence": s.confidence,
            "recall_text": s.recall_text,
            "follow_through": s.follow_through,
        })

    df = pd.DataFrame(rows)

    # Prefer the newest explicitly timestamped record. For legacy records
    # without recorded_at, survey_id provides a stable deterministic fallback.
    df["_recorded_sort"] = pd.to_datetime(
        df["recorded_at"], errors="coerce", utc=True
    )
    df = df.sort_values(
        ["interaction_id", "timing", "_recorded_sort", "survey_id"],
        kind="mergesort",
        na_position="first",
    )

    baseline = (
        df[df["timing"] == "BASELINE"]
        .groupby("interaction_id", sort=False)
        .tail(1)
        .set_index("interaction_id")
    )

    immediate = (
        df[df["timing"] == "IMMEDIATE"]
        .groupby("interaction_id", sort=False)
        .tail(1)
        .set_index("interaction_id")
    )

    paired = baseline.join(
        immediate,
        lsuffix="_baseline",
        rsuffix="_post",
        how="inner",
    )

    def paired_mean_change(column):
        if paired.empty:
            return None
        left = paired[f"{column}_baseline"]
        right = paired[f"{column}_post"]
        valid = left.notna() & right.notna()
        if not valid.any():
            return None
        return mean_or_none(right[valid] - left[valid])

    curiosity_change = paired_mean_change("curiosity")
    understanding_change = paired_mean_change("understanding")
    confidence_change = paired_mean_change("confidence")

    delayed = df[df["timing"].isin(["DELAYED_24H", "DELAYED_7D"])]

    follow_through_rate = None
    if not delayed.empty:
        valid = delayed[delayed["follow_through"].notna()]
        if not valid.empty:
            follow_through_rate = (
                valid["follow_through"].astype(bool).mean() * 100
            )

    immediate_with_recall = immediate[
        immediate["recall_text"].fillna("").astype(str).str.strip().ne("")
    ] if not immediate.empty else immediate

    recall_rate = None
    if len(immediate) > 0:
        recall_rate = len(immediate_with_recall) / len(immediate) * 100

    return {
        "participants": len(interactions),
        "surveyed": len(df["interaction_id"].unique()),
        "baseline_curiosity": mean_or_none(baseline.get("curiosity")),
        "post_curiosity": mean_or_none(immediate.get("curiosity")),
        "curiosity_change": curiosity_change,
        "baseline_understanding": mean_or_none(baseline.get("understanding")),
        "post_understanding": mean_or_none(immediate.get("understanding")),
        "understanding_change": understanding_change,
        "baseline_confidence": mean_or_none(baseline.get("confidence")),
        "post_confidence": mean_or_none(immediate.get("confidence")),
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

def generate_personality_impact(
    client,
    model_name,
    event,
    personalities_data
):
    prompt = f"""
EVENT: {event.name}
OBJECTIVE: {event.objective}
TARGET AUDIENCE: {event.target_audience}
ENVIRONMENT: {event.environment}
SENSORY & ACOUSTIC: {event.sensory_environment} / {event.acoustic_environment}
CONTEXT: {event.context}

PERSONALITIES TO MODEL (HEXACO + Resilience Base [0-100 scales]):
{personalities_data}

Model the specific behavioral impact, cognitive load, focus shift, and stress levels for these specific profiles as they undergo the defined scenario. Use accurate sociological and behavioral frameworks to predict how their distinct traits and resilience capacities interact with the event's environmental and operational design. Ensure responses are grounded and realistic.
"""
    return run_gemini(
        client=client,
        model_name=model_name,
        prompt=prompt,
        schema=PersonalityPredictorResponse,
        system_instruction=AI_SYSTEM,
        temperature=0.3,
    )



# ============================================================
# 15B. MINIMALIST PDF REPORTING
# ============================================================

def _pdf_text(value: Any) -> str:
    """Convert arbitrary values to safe, readable ReportLab text."""
    if value is None:
        return "Not recorded"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "Not recorded"
        return f"{value:g}"
    return str(value)


def _pdf_paragraph(text: Any, style) -> Paragraph:
    """Create a paragraph while safely escaping user/model-generated text."""
    normalized = _pdf_text(text).replace("\r\n", "\n").replace("\r", "\n")
    # Preserve line breaks without allowing model/user text to inject markup.
    safe = xml_escape(normalized).replace("\n", "<br/>")
    return Paragraph(safe, style)


def _pdf_add_heading(story, title: str, styles, level: int = 1):
    style = styles["ReportH1"] if level == 1 else styles["ReportH2"]
    story.append(Spacer(1, 5 * mm if level == 1 else 3 * mm))
    story.append(Paragraph(xml_escape(title), style))
    story.append(Spacer(1, 2 * mm))


def _pdf_add_kv_table(story, rows, styles):
    data = [
        [
            _pdf_paragraph(label, styles["TableLabel"]),
            _pdf_paragraph(value, styles["TableValue"]),
        ]
        for label, value in rows
    ]
    if not data:
        return
    table = Table(data, colWidths=[52 * mm, 130 * mm], hAlign="LEFT", repeatRows=0)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F6F8")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9DDE3")),
        ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E5E7EB")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    story.append(Spacer(1, 3 * mm))


def _pdf_add_bullets(story, items, styles):
    items = items or []
    for item in items:
        story.append(
            Paragraph(
                "• " + xml_escape(_pdf_text(item)).replace("\n", "<br/>"),
                styles["BulletReport"],
            )
        )


def _pdf_add_json_section(story, title, payload, styles):
    if payload is None:
        return
    _pdf_add_heading(story, title, styles, level=2)
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, list):
                story.append(
                    Paragraph(xml_escape(str(key).replace("_", " ").title()), styles["FieldLabel"])
                )
                _pdf_add_bullets(story, value, styles)
            elif isinstance(value, dict):
                story.append(
                    Paragraph(xml_escape(str(key).replace("_", " ").title()), styles["FieldLabel"])
                )
                _pdf_add_json_section(story, "", value, styles)
            else:
                _pdf_add_kv_table(story, [
                    (str(key).replace("_", " ").title(), _pdf_text(value))
                ], styles)
    elif isinstance(payload, list):
        _pdf_add_bullets(story, payload, styles)
    else:
        story.append(_pdf_paragraph(payload, styles["BodyReport"]))


def _build_program_pdf(
    event,
    interactions,
    rapid_logs,
    metrics,
    model_label,
    session_snapshot,
) -> bytes:
    """Build a complete, print-ready report for one selected program.

    The report is a presentation/export layer only. It reads the existing
    database records and current workspace analysis state; it does not mutate
    records and it does not recompute or reinterpret AI outputs.
    """
    buffer = BytesIO()

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=29,
        textColor=colors.HexColor("#111318"),
        spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#68707D"),
        spaceAfter=7 * mm,
    ))
    styles.add(ParagraphStyle(
        name="ReportH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=19,
        textColor=colors.HexColor("#111318"),
        spaceBefore=4 * mm,
        spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="ReportH2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#242832"),
        spaceBefore=3 * mm,
        spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="BodyReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#343943"),
        spaceAfter=2.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="BulletReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.7,
        leading=13,
        leftIndent=5 * mm,
        firstLineIndent=-3 * mm,
        textColor=colors.HexColor("#343943"),
        spaceAfter=1.2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="TableLabel",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor("#4B5563"),
    ))
    styles.add(ParagraphStyle(
        name="TableValue",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.1,
        leading=12,
        textColor=colors.HexColor("#20242C"),
    ))
    styles.add(ParagraphStyle(
        name="FieldLabel",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4B5563"),
        spaceBefore=1.5 * mm,
        spaceAfter=1 * mm,
    ))
    styles.add(ParagraphStyle(
        name="ReportNote",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=12,
        textColor=colors.HexColor("#68707D"),
        spaceAfter=2.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="ReportFooter",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        textColor=colors.HexColor("#8A909A"),
        alignment=TA_CENTER,
    ))

    def page_decor(canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setStrokeColor(colors.HexColor("#E5E7EB"))
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, height - 13 * mm, width - 18 * mm, height - 13 * mm)
        canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#8A909A"))
        canvas.drawString(18 * mm, 8 * mm, "Outreach Intelligence Lab")
        canvas.drawRightString(
            width - 18 * mm,
            8 * mm,
            f"Page {doc.page}",
        )
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title=f"{event.name} — Outreach Intelligence Lab Report",
        author="Outreach Intelligence Lab",
        subject="Comprehensive outreach program report",
    )

    story = []

    generated_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("OUTREACH INTELLIGENCE LAB", styles["ReportSubtitle"]))
    story.append(Paragraph(xml_escape(event.name), styles["ReportTitle"]))
    story.append(Paragraph(
        "Comprehensive program record • analysis • evidence • outcome metrics",
        styles["ReportSubtitle"],
    ))

    story.append(Table(
        [[
            _pdf_paragraph("REPORT", styles["TableLabel"]),
            _pdf_paragraph("SELECTED PROGRAM", styles["TableLabel"]),
            _pdf_paragraph("REASONING ENGINE", styles["TableLabel"]),
        ], [
            _pdf_paragraph("Comprehensive PDF export", styles["TableValue"]),
            _pdf_paragraph(event.name, styles["TableValue"]),
            _pdf_paragraph(model_label, styles["TableValue"]),
        ]],
        colWidths=[55 * mm, 73 * mm, 54 * mm],
        hAlign="LEFT",
        style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111318")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9DDE3")),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E5E7EB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]),
    ))
    story.append(Spacer(1, 4 * mm))
    story.append(_pdf_paragraph(
        f"Generated {generated_at}. This document is a read-only presentation of "
        "the selected program's existing records and the analysis currently held "
        "in the workspace. No database records are changed by exporting.",
        styles["ReportNote"],
    ))

    _pdf_add_heading(story, "1. Program specification", styles)
    _pdf_add_kv_table(story, [
        ("Program name", event.name),
        ("Created / recorded", event.date),
        ("Objective", event.objective),
        ("Target audience", event.target_audience),
        ("Context / experience description", event.context),
        ("Environment", event.environment),
        ("Sensory environment", event.sensory_environment),
        ("Acoustic environment", event.acoustic_environment),
        ("Pacing", event.pacing),
        ("Interaction style", event.interaction_style),
        ("Participant autonomy", event.participant_autonomy),
    ], styles)

    _pdf_add_heading(story, "2. Evidence inventory", styles)
    _pdf_add_kv_table(story, [
        ("Participant interactions", len(interactions)),
        ("Recorded observations", sum(len(i.observations or []) for i in interactions)),
        ("Survey records", sum(len(i.surveys or []) for i in interactions)),
        ("Rapid-state logs", len(rapid_logs)),
        ("Surveyed participants", metrics.get("surveyed") if metrics else 0),
    ], styles)
    story.append(_pdf_paragraph(
        "Participant identifiers shown below are the application's anonymous "
        "codes. The report intentionally excludes account credentials, session "
        "tokens, and authentication secrets.",
        styles["ReportNote"],
    ))

    _pdf_add_heading(story, "3. Deterministic outcome metrics", styles)
    if metrics is None:
        story.append(_pdf_paragraph("No event-level metrics are available.", styles["BodyReport"]))
    else:
        metric_rows = [
            ("Participants", metrics.get("participants")),
            ("Participants with survey data", metrics.get("surveyed")),
            ("Baseline curiosity", metrics.get("baseline_curiosity")),
            ("Post / immediate curiosity", metrics.get("post_curiosity")),
            ("Curiosity change", metrics.get("curiosity_change")),
            ("Baseline understanding", metrics.get("baseline_understanding")),
            ("Post / immediate understanding", metrics.get("post_understanding")),
            ("Understanding change", metrics.get("understanding_change")),
            ("Baseline confidence", metrics.get("baseline_confidence")),
            ("Post / immediate confidence", metrics.get("post_confidence")),
            ("Confidence change", metrics.get("confidence_change")),
            ("Follow-through rate", metrics.get("follow_through_rate")),
            ("Recall response rate", metrics.get("recall_rate")),
        ]
        _pdf_add_kv_table(story, metric_rows, styles)
        story.append(_pdf_paragraph(
            "These values are descriptive calculations from recorded participant "
            "data. A pre/post change does not by itself establish causation.",
            styles["ReportNote"],
        ))

    _pdf_add_heading(story, "4. Participant interactions and evidence", styles)
    if not interactions:
        story.append(_pdf_paragraph("No participant interactions are recorded for this program.", styles["BodyReport"]))
    else:
        for idx, interaction in enumerate(interactions, 1):
            story.append(KeepTogether([
                Paragraph(
                    xml_escape(f"Interaction {idx} · {interaction.participant_code}"),
                    styles["ReportH2"],
                ),
                _pdf_paragraph(
                    f"Started: {_pdf_text(interaction.started_at)}   •   "
                    f"Ended: {_pdf_text(interaction.ended_at)}   •   "
                    f"Phase: {_pdf_text(interaction.phase)}",
                    styles["ReportNote"],
                ),
                _pdf_paragraph(
                    f"Participant-stated preference: {_pdf_text(interaction.stated_preference)}",
                    styles["BodyReport"],
                ),
            ]))

            observations = sorted(
                list(interaction.observations or []),
                key=lambda x: x.timestamp or datetime.min,
            )
            if observations:
                story.append(Paragraph("Observed evidence", styles["FieldLabel"]))
                _pdf_add_bullets(
                    story,
                    [
                        f"{o.timestamp} · [{o.evidence_level}] {o.category}: {o.detail}"
                        for o in observations
                    ],
                    styles,
                )
            else:
                story.append(_pdf_paragraph("Observed evidence: none recorded.", styles["ReportNote"]))

            surveys = sorted(
                list(interaction.surveys or []),
                key=lambda x: x.timing or "",
            )
            if surveys:
                story.append(Paragraph("Outcome records", styles["FieldLabel"]))
                survey_rows = [[
                    _pdf_paragraph("Timing", styles["TableLabel"]),
                    _pdf_paragraph("Curiosity", styles["TableLabel"]),
                    _pdf_paragraph("Understanding", styles["TableLabel"]),
                    _pdf_paragraph("Confidence", styles["TableLabel"]),
                    _pdf_paragraph("Before state", styles["TableLabel"]),
                    _pdf_paragraph("After state", styles["TableLabel"]),
                    _pdf_paragraph("Follow-through", styles["TableLabel"]),
                    _pdf_paragraph("Recall", styles["TableLabel"]),
                ]]
                for s in surveys:
                    survey_rows.append([
                        _pdf_paragraph(s.timing, styles["TableValue"]),
                        _pdf_paragraph(s.curiosity, styles["TableValue"]),
                        _pdf_paragraph(s.understanding, styles["TableValue"]),
                        _pdf_paragraph(s.confidence, styles["TableValue"]),
                        _pdf_paragraph(getattr(s, "before_state", None) or "—", styles["TableValue"]),
                        _pdf_paragraph(getattr(s, "after_state", None) or "—", styles["TableValue"]),
                        _pdf_paragraph(
                            "Yes" if s.follow_through is True else
                            "No" if s.follow_through is False else "—",
                            styles["TableValue"],
                        ),
                        _pdf_paragraph(s.recall_text or "—", styles["TableValue"]),
                    ])
                survey_table = Table(
                    survey_rows,
                    colWidths=[19*mm, 17*mm, 21*mm, 19*mm, 34*mm, 34*mm, 25*mm, 26*mm],
                    repeatRows=1,
                    hAlign="LEFT",
                )
                survey_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6F8")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9DDE3")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E5E7EB")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(survey_table)
            story.append(Spacer(1, 3 * mm))

    _pdf_add_heading(story, "5. Rapid participant-state records", styles)
    if not rapid_logs:
        story.append(_pdf_paragraph("No rapid-state records are recorded for this program.", styles["BodyReport"]))
    else:
        rapid_rows = [[
            _pdf_paragraph("Time", styles["TableLabel"]),
            _pdf_paragraph("Participant code", styles["TableLabel"]),
            _pdf_paragraph("Baseline", styles["TableLabel"]),
            _pdf_paragraph("Current state / reaction", styles["TableLabel"]),
        ]]
        for log in rapid_logs:
            rapid_rows.append([
                _pdf_paragraph(log.timestamp, styles["TableValue"]),
                _pdf_paragraph(log.participant_code, styles["TableValue"]),
                _pdf_paragraph(log.baseline_level, styles["TableValue"]),
                _pdf_paragraph(log.current_state, styles["TableValue"]),
            ])
        rapid_table = Table(
            rapid_rows,
            colWidths=[35*mm, 35*mm, 48*mm, 76*mm],
            repeatRows=1,
            hAlign="LEFT",
        )
        rapid_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F6F8")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9DDE3")),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#E5E7EB")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(rapid_table)

    # Current workspace analysis is included in full when present. It is
    # explicitly labeled as a workspace snapshot so the exporter never
    # fabricates an event-to-analysis association that the existing schema
    # does not persist.
    _pdf_add_heading(story, "6. Current workspace analysis snapshot", styles)
    story.append(_pdf_paragraph(
        "The following sections reproduce analysis objects currently held in "
        "the active Streamlit workspace. Generated AI content remains a hypothesis "
        "rather than a measurement. If an analysis was generated for another "
        "program earlier in the same browser session, this section does not "
        "silently relabel it as evidence for the selected program.",
        styles["ReportNote"],
    ))

    analysis_items = [
        ("Engagement architecture / forward model", session_snapshot.get("last_forward_model")),
        ("Outcome prediction", session_snapshot.get("last_prediction")),
        ("Personality predictor", session_snapshot.get("last_personality_prediction")),
        ("Live copilot recommendation", session_snapshot.get("last_recommendation")),
        ("Impact interpretation", session_snapshot.get("last_impact_interpretation")),
        ("Counterfactual analysis", session_snapshot.get("last_counterfactual")),
    ]
    any_analysis = False
    for title, payload in analysis_items:
        if payload:
            any_analysis = True
            _pdf_add_json_section(story, title, payload, styles)

    if not any_analysis:
        story.append(_pdf_paragraph(
            "No generated analysis objects are currently held in the workspace.",
            styles["BodyReport"],
        ))

    # Include the inputs currently visible to analysis pages where available.
    _pdf_add_heading(story, "7. Current analysis inputs", styles)
    input_rows = []
    input_map = [
        ("Crowd details", "pred_crowd"),
        ("Situational context", "pred_situation"),
        ("Baseline audience stress", "q_stress"),
        ("Ambient distraction / sensory noise", "q_noise"),
        ("Planned session duration", "q_duration"),
        ("Interactive touchpoint density", "q_density"),
        ("Counterfactual variable change", "cf_variable_change"),
        ("Counterfactual current design", "cf_design_description"),
    ]
    for label, key in input_map:
        value = session_snapshot.get(key)
        if value not in (None, ""):
            input_rows.append((label, value))
    if input_rows:
        _pdf_add_kv_table(story, input_rows, styles)
    else:
        story.append(_pdf_paragraph(
            "No additional page-level analysis inputs are currently populated.",
            styles["BodyReport"],
        ))

    _pdf_add_heading(story, "8. Methodological guardrails", styles)
    _pdf_add_bullets(story, [
        "AI predictions are hypotheses, not measurements.",
        "The system does not diagnose participants or infer protected/sensitive personal attributes.",
        "Participant-stated preferences outrank model inference.",
        "Observed behavior is kept separate from interpretation.",
        "Deterministic analytics are calculated in Python; Gemini is used for interpretation/generation.",
        "\"No intervention\" is a legitimate recommendation.",
        "Confidence is not probability of truth.",
        "Public-facing results should distinguish OBSERVED, STATED, INFERRED, and HYPOTHESIS.",
        "The system is intended for exploratory educational/outreach use, not clinical or psychological assessment.",
        "Pre/post changes are descriptive evidence and do not automatically establish causation.",
    ], styles)

    _pdf_add_heading(story, "9. Evidence labels", styles)
    _pdf_add_kv_table(story, [
        ("STATED", "Something explicitly reported by the participant."),
        ("OBSERVED", "Something directly witnessed by the facilitator."),
        ("INFERRED", "A reasonable interpretation of observed information."),
        ("HYPOTHESIS", "A speculative explanation requiring additional evidence."),
    ], styles)

    _pdf_add_heading(story, "10. Report provenance", styles)
    _pdf_add_kv_table(story, [
        ("Application", APP_TITLE),
        ("Application version", APP_VERSION),
        ("Selected reasoning engine", model_label),
        ("Report generated", generated_at),
        ("Selected program ID", event.id),
        ("Data scope", "Selected program and current workspace analysis snapshot"),
        ("Export behavior", "Read-only; no records modified"),
    ], styles)

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(
        "AI-generated predictions are synthetic hypotheses. They do not establish "
        "psychological, neurological, clinical, or causal facts about individuals. "
        "Real-world impact metrics are calculated from recorded observations and "
        "participant-reported outcomes.",
        styles["ReportNote"],
    ))

    doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
    return buffer.getvalue()


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

header_col1, header_col2, header_col3 = st.columns([1.5, 1, 3.5])

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
        st.session_state["confirm_clean_memory"] = True

    if st.session_state.get("confirm_clean_memory", False):
        st.warning(
            "This permanently erases this account's saved application memory: "
            "experiences, interactions, observations, surveys, rapid-state logs, "
            "personality profiles, and saved interface state. Your account remains."
        )
        confirmed = st.checkbox(
            "I understand this saved application memory cannot be recovered.",
            key="confirm_clean_memory_checkbox",
        )
        if st.button(
            "Permanently erase saved memory",
            type="primary",
            use_container_width=True,
            disabled=not confirmed,
        ):
            events = db.query(Event).filter(
                Event.session_token == USER_SESSION_TOKEN
            ).all()
            event_ids = [e.id for e in events]

            if event_ids:
                interactions = db.query(Interaction).filter(
                    Interaction.event_id.in_(event_ids)
                ).all()
                interaction_ids = [i.id for i in interactions]

                if interaction_ids:
                    db.query(Observation).filter(
                        Observation.interaction_id.in_(interaction_ids)
                    ).delete(synchronize_session=False)
                    db.query(Survey).filter(
                        Survey.interaction_id.in_(interaction_ids)
                    ).delete(synchronize_session=False)
                    db.query(Interaction).filter(
                        Interaction.event_id.in_(event_ids)
                    ).delete(synchronize_session=False)

                db.query(Event).filter(
                    Event.session_token == USER_SESSION_TOKEN
                ).delete(synchronize_session=False)

            db.query(RapidStateLog).filter(
                RapidStateLog.session_token == USER_SESSION_TOKEN
            ).delete(synchronize_session=False)
            db.query(PersonalityProfile).filter(
                PersonalityProfile.session_token == USER_SESSION_TOKEN
            ).delete(synchronize_session=False)
            db.query(AppState).filter(
                AppState.user_id == USER_ACCOUNT_ID
            ).delete(synchronize_session=False)
            db.commit()

            keep = {
                "auth_token": st.session_state.get("auth_token"),
                "authenticated_username": st.session_state.get("authenticated_username"),
                "authenticated_user_id": st.session_state.get("authenticated_user_id"),
            }
            st.session_state.clear()
            st.session_state.update(keep)
            st.session_state["persistent_state_restored"] = True
            try:
                db.close()
            except Exception:
                pass
            st.rerun()

    if st.button("Log out", use_container_width=True):
        current_token = (
            st.session_state.get("auth_token")
            or st.query_params.get("auth_session")
        )
        _revoke_auth_token(db, current_token)
        _clear_auth_query_token()
        components.html("""
            <script>
            try {
                window.parent.localStorage.removeItem("outreach_auth_session");
            } catch (e) {}
            </script>
        """, height=0, width=0)
        try:
            db.close()
        except Exception:
            pass
        st.session_state.clear()
        st.rerun()

    render_html(f"""
    <div class="small-note" style="margin-top:8px; text-align:center;">
        Account: {clean_text(USER_USERNAME)} | Version {APP_VERSION}
    </div>
    """)

with header_col3:
    page_opts = [
        "Experience Designer",
        "Live Copilot",
        "Outcome Predictor",
        "Personality Predictor",
        "Scientific Reactions",
        "Impact Observatory",
        "Counterfactual Lab",
        "PDF Report",
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

    # Saved experiences are separate, durable records. Creating another
    # experience never overwrites an existing one; editing updates only the
    # selected experience record.
    saved_events = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .all()
    )

    saved_event_map = {
        f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')}) · {e.id[:8]}": e
        for e in saved_events
    }

    manager_col1, manager_col2 = st.columns([3, 1])

    # Streamlit widget-backed session-state keys must be prepared BEFORE the
    # corresponding widget is instantiated in a script run.  All requests that
    # need to load/reset the designer therefore use a pending action.  This
    # keeps the existing editor controls intact while making program switching
    # safe on current Streamlit releases.
    pending_selector = st.session_state.pop(
        "experience_saved_selector_pending",
        None,
    )
    pending_editor_id = st.session_state.pop(
        "experience_editor_load_pending",
        None,
    )
    pending_new_program = st.session_state.pop(
        "experience_new_program_pending",
        False,
    )

    if pending_editor_id is not None:
        pending_event = saved_event_map.get(
            next(
                (
                    label
                    for label, saved_event in saved_event_map.items()
                    if saved_event.id == pending_editor_id
                ),
                "",
            )
        )

        if pending_event is not None:
            obj_opts_for_load = [
                "Curiosity",
                "Scientific understanding",
                "Awe and wonder",
                "Memory and retention",
                "Question generation",
                "Independent follow-through",
                "General engagement",
            ]
            aud_opts_for_load = [
                "General public",
                "Students",
                "Families",
                "Educators",
                "Astronomy enthusiasts",
                "Eco-tourists",
                "Mixed audience",
            ]
            pac_opts_for_load = [
                "Slow and contemplative",
                "Moderate",
                "Fast and energetic",
                "Variable",
            ]
            style_opts_for_load = [
                "Open observation",
                "Facilitator-led",
                "Question-led",
                "Hands-on",
                "Story-driven",
                "Mixed",
            ]
            aut_opts_for_load = ["High", "Moderate", "Low"]

            st.session_state["experience_editor_mode"] = "edit"
            st.session_state["experience_editor_event_id"] = pending_event.id
            st.session_state["active_event_id"] = pending_event.id
            st.session_state["active_interaction_id"] = None
            st.session_state["last_forward_model"] = None

            st.session_state["mem_event_name"] = pending_event.name or ""
            st.session_state["mem_objective"] = (
                pending_event.objective
                if pending_event.objective in obj_opts_for_load
                else "Curiosity"
            )
            st.session_state["mem_audience"] = (
                pending_event.target_audience
                if pending_event.target_audience in aud_opts_for_load
                else "General public"
            )
            st.session_state["mem_environment"] = pending_event.environment or ""
            st.session_state["mem_acoustic"] = pending_event.acoustic_environment or ""
            st.session_state["mem_sensory"] = pending_event.sensory_environment or ""
            st.session_state["mem_context"] = pending_event.context or ""
            st.session_state["mem_pacing"] = (
                pending_event.pacing
                if pending_event.pacing in pac_opts_for_load
                else pac_opts_for_load[0]
            )
            st.session_state["mem_interaction_style"] = (
                pending_event.interaction_style
                if pending_event.interaction_style in style_opts_for_load
                else style_opts_for_load[0]
            )
            st.session_state["mem_optional_choice"] = (
                pending_event.participant_autonomy
                if pending_event.participant_autonomy in aut_opts_for_load
                else aut_opts_for_load[0]
            )

            pending_selector = next(
                (
                    label
                    for label, saved_event in saved_event_map.items()
                    if saved_event.id == pending_event.id
                ),
                "New program",
            )

    elif pending_new_program:
        st.session_state["experience_editor_mode"] = "new"
        st.session_state["experience_editor_event_id"] = None
        st.session_state["active_event_id"] = None
        st.session_state["active_interaction_id"] = None
        st.session_state["last_forward_model"] = None

        st.session_state["mem_event_name"] = ""
        st.session_state["mem_objective"] = "Curiosity"
        st.session_state["mem_audience"] = "General public"
        st.session_state["mem_environment"] = ""
        st.session_state["mem_acoustic"] = ""
        st.session_state["mem_sensory"] = ""
        st.session_state["mem_context"] = ""
        st.session_state["mem_pacing"] = "Slow and contemplative"
        st.session_state["mem_interaction_style"] = "Open observation"
        st.session_state["mem_optional_choice"] = "High"
        pending_selector = "New program"

    if pending_selector is not None:
        if pending_selector == "New program" or pending_selector in saved_event_map:
            st.session_state["experience_saved_selector"] = pending_selector
        else:
            st.session_state["experience_saved_selector"] = "New program"

    if "experience_saved_selector" not in st.session_state:
        initial_editor_id = st.session_state.get("experience_editor_event_id")
        initial_label = next(
            (
                label
                for label, saved_event in saved_event_map.items()
                if saved_event.id == initial_editor_id
            ),
            "New program",
        )
        st.session_state["experience_saved_selector"] = initial_label

    with manager_col1:
        if saved_events:
            selected_saved_label = st.selectbox(
                "Saved programs",
                ["New program"] + list(saved_event_map.keys()),
                key="experience_saved_selector",
                help="Choose a saved program to edit, or choose New program to design another program.",
            )
        else:
            selected_saved_label = "New program"
            st.caption("No saved programs yet. Design your first program below.")

    with manager_col2:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button(
            "New program",
            use_container_width=True,
            help="Start a separate program without changing any saved program.",
        ):
            st.session_state["experience_saved_selector_pending"] = "New program"
            st.session_state["experience_new_program_pending"] = True
            st.rerun()

    # A selector change happens after the selectbox has been instantiated, so
    # never write to any of the designer widget keys here. Stage the requested
    # program and let the next run load it before the widgets are created.
    if selected_saved_label == "New program":
        if st.session_state.get("experience_editor_event_id") is not None:
            st.session_state["experience_new_program_pending"] = True
            st.session_state["experience_saved_selector_pending"] = "New program"
            st.rerun()
    elif selected_saved_label in saved_event_map:
        selected_event = saved_event_map[selected_saved_label]
        if st.session_state.get("experience_editor_event_id") != selected_event.id:
            st.session_state["experience_editor_load_pending"] = selected_event.id
            st.session_state["experience_saved_selector_pending"] = selected_saved_label
            st.rerun()

    editor_mode = st.session_state.get("experience_editor_mode", "new")
    editing_event_id = st.session_state.get("experience_editor_event_id")

    if editor_mode == "edit" and editing_event_id:
        editing_event = (
            db.query(Event)
            .filter(
                Event.id == editing_event_id,
                Event.session_token == USER_SESSION_TOKEN,
            )
            .first()
        )

        if editing_event is None:
            st.session_state["experience_editor_mode"] = "new"
            st.session_state["experience_editor_event_id"] = None
            st.session_state["last_forward_model"] = None
            st.rerun()
    else:
        editing_event = None

    if editing_event is not None:
        render_html(
            f"""
            <div class="premium-card" style="border-color:rgba(91,140,255,.30);">
                <div class="eyebrow">Editing saved program</div>
                <div style="color:var(--text);font-size:1.2rem;">
                    {clean_text(editing_event.name)}
                </div>
                <div class="small-note" style="margin-top:6px;">
                    Changes are saved to this program only. Existing
                    participant interactions, observations, surveys, and
                    rapid-state logs are retained.
                </div>
            </div>
            """
        )

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

    action_label = (
        "Save Changes & Regenerate Model"
        if editing_event is not None
        else "Create Experience Model"
    )

    if st.button(
        action_label,
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
                if editing_event is not None:
                    event = update_event(
                        db=db,
                        event=editing_event,
                        name=event_name,
                        objective=objective,
                        context=context,
                        environment=environment,
                        sensory_environment=sensory_environment,
                        acoustic_environment=acoustic_environment,
                        target_audience=target_audience,
                        pacing=pacing,
                        interaction_style=interaction_style,
                        participant_autonomy=optional_choice,
                    )
                    save_message = "Program changes saved."
                else:
                    event = create_event(
                        db=db,
                        name=event_name,
                        objective=objective,
                        context=context,
                        environment=environment,
                        sensory_environment=sensory_environment,
                        acoustic_environment=acoustic_environment,
                        target_audience=target_audience,
                        pacing=pacing,
                        interaction_style=interaction_style,
                        participant_autonomy=optional_choice,
                    )
                    save_message = "New program saved."

                st.session_state.active_event_id = event.id
                st.session_state.active_interaction_id = None
                st.session_state.experience_editor_mode = "edit"
                st.session_state.experience_editor_event_id = event.id

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
                st.session_state["experience_saved_selector_pending"] = (
                    f"{event.name} ({event.date.strftime('%Y-%m-%d %H:%M')}) · {event.id[:8]}"
                )

                st.success(
                    f"{save_message} Engagement model generated."
                )

                # Refresh the saved-program selector so the durable record
                # and the editor remain synchronized without touching any
                # other workspace data.
                app_rerun()

            except Exception as exc:

                db.rollback()

                st.error(
                    f"Could not save the experience: {exc}"
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

    # Saved-program management remains inside the Experience Designer.
    # It provides visibility that multiple independent programs can coexist
    # and lets the user reopen any one of them for editing.
    st.markdown("---")
    render_html('<div class="section-heading">Saved programs</div>')

    current_saved_events = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .all()
    )

    if not current_saved_events:
        st.caption("No saved programs yet.")
    else:
        for saved_event in current_saved_events:
            with st.expander(
                f"{saved_event.name} · {saved_event.date.strftime('%Y-%m-%d %H:%M')}"
            ):
                ec1, ec2 = st.columns([4, 1])

                with ec1:
                    st.caption(
                        f"Objective: {saved_event.objective} · "
                        f"Audience: {saved_event.target_audience or 'Not specified'}"
                    )
                    st.caption(
                        f"Pacing: {saved_event.pacing or 'Slow and contemplative'} · "
                        f"Interaction: {saved_event.interaction_style or 'Open observation'} · "
                        f"Autonomy: {saved_event.participant_autonomy or 'High'}"
                    )

                with ec2:
                    if st.button(
                        "Edit",
                        key=f"edit_saved_event_{saved_event.id}",
                        use_container_width=True,
                    ):
                        st.session_state["experience_saved_selector_pending"] = (
                            f"{saved_event.name} ({saved_event.date.strftime('%Y-%m-%d %H:%M')}) · {saved_event.id[:8]}"
                        )
                        st.session_state["experience_editor_load_pending"] = saved_event.id
                        st.rerun()


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
            stress_options = [
                "Very Low / Relaxed",
                "Moderate Stress",
                "High Stress / Overwhelmed",
            ]
            if "q_stress" not in st.session_state:
                st.session_state["q_stress"] = stress_options[1]
            baseline_stress_q = st.select_slider(
                "Estimated Baseline Audience Stress Level",
                options=stress_options,
                key="q_stress",
            )

            noise_options = [
                "Quiet & Controlled",
                "Moderate Noise",
                "High Loudness / Busy Crowd",
            ]
            if "q_noise" not in st.session_state:
                st.session_state["q_noise"] = noise_options[1]
            noise_sensory_q = st.select_slider(
                "Ambient Distraction & Sensory Noise Level",
                options=noise_options,
                key="q_noise",
            )
        with qc2:
            duration_q = st.selectbox(
                "Planned Session Duration",
                ["Short (< 15 mins)", "Standard (30-45 mins)", "Extended (60+ mins)"],
                key="q_duration"
            )
            density_options = [
                "Low (Passive listening)",
                "Medium (Guided Q&A)",
                "High (Hands-on exploration)",
            ]
            if "q_density" not in st.session_state:
                st.session_state["q_density"] = density_options[1]
            interaction_density_q = st.selectbox(
                "Interactive Touchpoints Density",
                density_options,
                key="q_density",
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
# 18.5 PERSONALITY PREDICTOR
# ============================================================

elif page == "Personality Predictor":
    
    render_html('<div class="section-heading">Behavioral Personality Lab (HEXACO + Resilience)</div>')
    render_html('<div class="small-note" style="margin-bottom:15px;">Model how highly specific psychological profiles, defined by their HEXACO traits and baseline behavioral resilience, react dynamically to your outreach scenarios.</div>')

    col_create, col_list = st.columns([1.5, 1])
    
    with col_create:
        render_html('<div class="section-heading" style="margin-top:0;">Create Profile</div>')
        with st.form("create_personality", clear_on_submit=True):
            p_name = st.text_input("Profile Name", placeholder="e.g. High-Stress Introvert or The Resilient Explorer")
            p_desc = st.text_input("Brief Description", placeholder="Short contextual note about this profile")
            
            c1, c2 = st.columns(2)
            with c1:
                h_val = st.slider("Honesty-Humility (H)", 0, 100, 50, help="Sincerity, fairness, greed avoidance.")
                e_val = st.slider("Emotionality (E)", 0, 100, 50, help="Fearfulness, anxiety, dependence vs. bravery, toughness.")
                x_val = st.slider("Extraversion (X)", 0, 100, 50, help="Social self-esteem, social boldness, sociability.")
            with c2:
                a_val = st.slider("Agreeableness (A)", 0, 100, 50, help="Forgivingness, gentleness, flexibility.")
                c_val = st.slider("Conscientiousness (C)", 0, 100, 50, help="Organization, diligence, perfectionism.")
                o_val = st.slider("Openness (O)", 0, 100, 50, help="Aesthetic appreciation, inquisitiveness, creativity.")
            
            res_val = st.slider("Resilience Baseline Capacity", 0, 100, 50, help="The individual's built-in capacity to bounce back from environmental or cognitive stress.")
            
            if st.form_submit_button("Save Profile", use_container_width=True):
                if p_name.strip():
                    new_profile = PersonalityProfile(
                        id=str(uuid.uuid4()),
                        session_token=USER_SESSION_TOKEN,
                        name=p_name.strip(),
                        description=p_desc.strip(),
                        hexaco_h=h_val, hexaco_e=e_val, hexaco_x=x_val,
                        hexaco_a=a_val, hexaco_c=c_val, hexaco_o=o_val,
                        resilience_baseline=res_val
                    )
                    db.add(new_profile)
                    db.commit()
                    st.toast(f"Profile '{p_name}' saved.")
                    app_rerun()
                else:
                    st.error("Profile Name is required.")

    with col_list:
        render_html('<div class="section-heading" style="margin-top:0;">Saved Profiles</div>')
        saved_profiles = db.query(PersonalityProfile).filter(PersonalityProfile.session_token == USER_SESSION_TOKEN).all()
        
        if not saved_profiles:
            st.info("No saved profiles. Create one to begin.")
        else:
            for prof in saved_profiles:
                with st.expander(f"{prof.name}"):
                    st.caption(f"Desc: {prof.description}")
                    st.caption(f"H:{prof.hexaco_h} E:{prof.hexaco_e} X:{prof.hexaco_x} A:{prof.hexaco_a} C:{prof.hexaco_c} O:{prof.hexaco_o} | Res:{prof.resilience_baseline}")
                    if st.button("Delete", key=f"del_prof_{prof.id}"):
                        db.delete(prof)
                        db.commit()
                        st.toast("Profile deleted.")
                        app_rerun()

    st.markdown("---")
    
    events = db.query(Event).filter(Event.session_token == USER_SESSION_TOKEN).order_by(Event.date.desc()).all()
    if not events:
        st.info("You must create an experience in the Experience Designer before testing personalities.")
    elif not saved_profiles:
        st.info("You must create at least one Personality Profile to run a prediction.")
    else:
        render_html('<div class="section-heading">Simulate Event Impact on Personalities</div>')
        event_map = {f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})": e for e in events}
        selected_event_name = st.selectbox("Select Experience Scenario", list(event_map.keys()), key="pers_event_select")
        event = event_map[selected_event_name]
        
        selected_profiles = st.multiselect(
            "Select Profiles to Test",
            options=[p.name for p in saved_profiles],
            default=[p.name for p in saved_profiles][:3] # Select up to 3 by default
        )
        
        if st.button("Predict Personality Impacts", type="primary", use_container_width=True):
            if client is None:
                st.error("Gemini is unavailable.")
            elif not selected_profiles:
                st.error("Select at least one profile.")
            else:
                target_profiles = [p for p in saved_profiles if p.name in selected_profiles]
                profile_data_str = ""
                for tp in target_profiles:
                    profile_data_str += f"- {tp.name}: H({tp.hexaco_h}), E({tp.hexaco_e}), X({tp.hexaco_x}), A({tp.hexaco_a}), C({tp.hexaco_c}), O({tp.hexaco_o}), Resilience({tp.resilience_baseline})\n"
                
                try:
                    with st.spinner("Modeling behavioral impacts across diverse traits..."):
                        impact_pred = generate_personality_impact(
                            client,
                            selected_model,
                            event,
                            profile_data_str
                        )
                        st.session_state.last_personality_prediction = impact_pred.model_dump()
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
                    
        if st.session_state.get("last_personality_prediction"):
            result = st.session_state.last_personality_prediction
            
            render_html('<div class="section-heading">Overall Crowd Dynamics</div>')
            render_html(f"""
            <div class="premium-card">
                <div style="color:var(--text); line-height:1.7;">
                    {clean_text(result['overall_scenario_dynamics'])}
                </div>
            </div>
            """)
            
            render_html('<div class="section-heading">Individual Profile Outcomes</div>')
            for imp in result['impacts']:
                st.markdown(f"### {clean_text(imp['personality_name'])}")
                mc1, mc2, mc3, mc4 = st.columns(4)
                
                # Format focus shift with +/- sign
                focus_shift = imp['focus_shift_pct']
                focus_str = f"+{focus_shift}%" if focus_shift > 0 else f"{focus_shift}%"
                
                mc1.metric("Focus Shift", focus_str)
                mc2.metric("Final Stress Level", f"{imp['stress_level_pct']}%")
                mc3.metric("Cognitive Load", f"{imp['cognitive_load_pct']}%")
                mc4.metric("Resilience Activated", f"{imp['resilience_activation_pct']}%")
                
                render_html(f"""
                <div class="premium-card" style="margin-top: 15px;">
                    <div class="eyebrow">Behavioral Response Narrative</div>
                    <div style="color:var(--text-secondary); line-height:1.6;">
                        {clean_text(imp['behavioral_response'])}
                    </div>
                </div>
                """)
                
                if imp['friction_points']:
                    st.markdown("**Expected Friction Points:**")
                    for fp in imp['friction_points']:
                        st.markdown(f"- *{clean_text(fp)}*")
                
                st.markdown("---")

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
            app_rerun()

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
                    app_rerun()

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

                app_rerun()

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
                app_rerun()

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
                    app_rerun()

            render_html('<div class="section-heading">Recent evidence</div>')

            observations = get_recent_observations(
                db,
                interaction.id
            )

            if observations:
                st.markdown(
                    f"**Logged observations for {clean_text(interaction.participant_code)}**"
                )

                for observation in observations:
                    obs_time = (
                        observation.timestamp.strftime("%H:%M:%S UTC")
                        if observation.timestamp
                        else "—"
                    )
                    obs_col1, obs_col2 = st.columns([5, 1])

                    with obs_col1:
                        render_html(f"""
                        <div class="observation-row">
                            <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                                <span class="badge">
                                    {clean_text(observation.evidence_level)}
                                </span>
                                <span class="badge">
                                    {clean_text(observation.category)}
                                </span>
                                <span class="small-note">
                                    {clean_text(obs_time)}
                                </span>
                            </div>
                            <div style="margin-top:8px;color:var(--text);line-height:1.5;">
                                {clean_text(observation.detail)}
                            </div>
                        </div>
                        """)

                    with obs_col2:
                        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                        if st.button(
                            "Delete",
                            key=f"del_obs_{interaction.id}_{observation.id}",
                            help="Permanently delete this logged observation.",
                            use_container_width=True,
                        ):
                            if delete_observation(
                                db,
                                interaction.id,
                                observation.id,
                            ):
                                st.toast("Observation deleted.")
                                app_rerun()
                            else:
                                st.warning("Observation was not found.")
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

                app_rerun()


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
                st.caption(
                    "To obtain a valid paired change: select the same participant "
                    "interaction in Optional participant outcome capture, save a "
                    "BASELINE record before the interaction, then save an IMMEDIATE "
                    "record after the interaction. The system pairs records within "
                    "the same participant interaction; missing values are not treated as zero."
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
# 23. PDF REPORT
# ============================================================

elif page == "PDF Report":
    render_html('<div class="section-heading">Comprehensive PDF report</div>')
    render_html("""
    <div class="premium-card">
        <div class="eyebrow">Print / export</div>
        <div style="color:var(--text); font-size:1.05rem; line-height:1.6;">
            Choose a saved program and generate a clean, print-ready PDF containing
            its full specification, recorded evidence, deterministic outcome metrics,
            participant outcome records, rapid-state records, and the analysis
            currently held in this workspace.
        </div>
        <div class="small-note" style="margin-top:10px;">
            Exporting is read-only. It does not modify programs, observations,
            surveys, predictions, or other workspace data.
        </div>
    </div>
    """)

    report_events = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .all()
    )

    if not report_events:
        st.info("Create an experience in Experience Designer first.")
    else:
        report_event_map = {
            f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')}) · {e.id[:8]}": e
            for e in report_events
        }

        report_event_keys = list(report_event_map.keys())
        if "pdf_report_event_select" not in st.session_state:
            default_event_id = st.session_state.get("active_event_id")
            default_label = next(
                (
                    label for label, saved_event in report_event_map.items()
                    if saved_event.id == default_event_id
                ),
                report_event_keys[0],
            )
            st.session_state["pdf_report_event_select"] = default_label

        selected_report_label = st.selectbox(
            "Program to print",
            report_event_keys,
            key="pdf_report_event_select",
            help="Choose exactly which saved program should be included in the PDF.",
        )
        report_event = report_event_map[selected_report_label]

        report_interactions = (
            db.query(Interaction)
            .filter(Interaction.event_id == report_event.id)
            .order_by(Interaction.started_at.asc())
            .all()
        )

        report_rapid_logs = (
            db.query(RapidStateLog)
            .filter(
                RapidStateLog.event_id == report_event.id,
                RapidStateLog.session_token == USER_SESSION_TOKEN,
            )
            .order_by(RapidStateLog.timestamp.asc())
            .all()
        )

        report_metrics = calculate_event_impact(db, report_event.id)

        report_session_keys = [
            "last_forward_model",
            "last_prediction",
            "last_personality_prediction",
            "last_recommendation",
            "last_impact_interpretation",
            "last_counterfactual",
            "pred_crowd",
            "pred_situation",
            "q_stress",
            "q_noise",
            "q_duration",
            "q_density",
            "cf_variable_change",
            "cf_design_description",
        ]
        report_session_snapshot = {
            key: st.session_state.get(key)
            for key in report_session_keys
        }

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Interactions", len(report_interactions))
        rc2.metric("Rapid-state logs", len(report_rapid_logs))
        rc3.metric(
            "Surveyed participants",
            report_metrics.get("surveyed", 0) if report_metrics else 0,
        )

        st.caption(
            "The PDF contains the selected program's stored records. "
            "AI analysis is reproduced as analysis/hypothesis content, not as measured fact."
        )

        if st.button(
            "Generate comprehensive PDF",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Building the comprehensive minimalist PDF..."):
                    pdf_bytes = _build_program_pdf(
                        event=report_event,
                        interactions=report_interactions,
                        rapid_logs=report_rapid_logs,
                        metrics=report_metrics,
                        model_label=selected_model_label,
                        session_snapshot=report_session_snapshot,
                    )

                safe_filename = re.sub(
                    r"[^A-Za-z0-9._-]+",
                    "_",
                    (report_event.name or "outreach_program").strip(),
                ).strip("._") or "outreach_program"
                filename = f"{safe_filename}_outreach_report.pdf"

                st.download_button(
                    "Print / download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success(
                    "PDF generated successfully. Use the print/download control above."
                )
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")

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
    "Optional participant outcome capture",
    expanded=True,
):
    st.markdown(
        """
        This module is intentionally secondary to the outreach
        intelligence system.

        It allows an event team to collect lightweight pre/post outcomes
        when appropriate. The participant's descriptive state is recorded
        at its own time point: **Before state = BASELINE** and
        **After state = IMMEDIATE**. These are descriptive records, not
        clinical, neurological, or psychological diagnoses.
        """
    )

    events = (
        db.query(Event)
        .filter(Event.session_token == USER_SESSION_TOKEN)
        .order_by(Event.date.desc())
        .all()
    )

    if not events:
        st.caption("Create an experience first.")

    else:
        event_map = {
            f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})": e
            for e in events
        }

        event_keys = list(event_map.keys())
        selected_event_name = st.selectbox(
            "Experience",
            event_keys,
            key="survey_event_select",
        )

        event = event_map[selected_event_name]

        interactions = (
            db.query(Interaction)
            .filter(Interaction.event_id == event.id)
            .order_by(Interaction.started_at.desc())
            .all()
        )

        if not interactions:
            st.caption("No participant interactions yet.")

        else:
            interaction_map = {
                f"{i.participant_code} ({i.started_at.strftime('%H:%M:%S')})": i
                for i in interactions
            }

            int_keys = list(interaction_map.keys())
            selected_participant = st.selectbox(
                "Participant interaction",
                int_keys,
                key="survey_participant_select",
            )

            interaction = interaction_map[selected_participant]

            timing_opts = [
                "BASELINE",
                "IMMEDIATE",
                "DELAYED_24H",
                "DELAYED_7D",
            ]
            survey_timing = st.selectbox(
                "Measurement point",
                timing_opts,
                key="survey_timing_select",
                help=(
                    "BASELINE records the before-interaction state. "
                    "IMMEDIATE records the after-interaction state. "
                    "Delayed points are for later outcomes."
                ),
            )

            # Make the two state fields unmistakable. Only the state that
            # belongs to the selected measurement point is editable. This
            # prevents an 'after' description from being accidentally stored
            # as baseline evidence (or vice versa).
            render_html('<div class="section-heading">Participant state records</div>')
            st.caption(
                "Record the two time points separately. Choose BASELINE to record "
                "the participant's state before the interaction; choose IMMEDIATE "
                "to record the participant's state immediately after it. Use only "
                "participant-reported wording or directly observable concrete "
                "descriptions. Do not diagnose or infer a clinical or psychological condition."
            )

            if survey_timing == "BASELINE":
                st.markdown("**BEFORE STATE · BASELINE**")
                before_state_help = (
                    "Enabled because BASELINE is selected. Describe the participant "
                    "before the outreach interaction."
                )
                after_state_help = (
                    "Select IMMEDIATE above to record the separate after-interaction state."
                )
            elif survey_timing == "IMMEDIATE":
                st.markdown("**AFTER STATE · IMMEDIATE**")
                before_state_help = (
                    "Select BASELINE above to record the separate before-interaction state."
                )
                after_state_help = (
                    "Enabled because IMMEDIATE is selected. Describe the participant "
                    "immediately after the outreach interaction."
                )
            else:
                st.markdown("**BEFORE / AFTER STATE**")
                before_state_help = (
                    "State descriptions are captured only at BASELINE and IMMEDIATE. "
                    "Select one of those measurement points to record a state."
                )
                after_state_help = before_state_help

            with st.form("outcome_capture", clear_on_submit=True):
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

                before_state = st.text_area(
                    "Before state (BASELINE only)",
                    height=90,
                    disabled=survey_timing != "BASELINE",
                    help=before_state_help,
                    placeholder=(
                        "Example: Participant says they are unfamiliar with the topic; "
                        "asks what the demonstration is about."
                    ),
                )

                after_state = st.text_area(
                    "After state (IMMEDIATE only)",
                    height=90,
                    disabled=survey_timing != "IMMEDIATE",
                    help=after_state_help,
                    placeholder=(
                        "Example: Participant says the main idea is clearer and asks "
                        "a follow-up question about the demonstration."
                    ),
                )

                recall = st.text_area(
                    "What do you remember most?",
                    height=90,
                )

                follow_through = st.checkbox(
                    "I voluntarily explored something further afterward"
                )

                submitted = st.form_submit_button("Save outcome")

                if submitted:
                    before_value = (
                        before_state.strip()
                        if survey_timing == "BASELINE" and before_state.strip()
                        else None
                    )
                    after_value = (
                        after_state.strip()
                        if survey_timing == "IMMEDIATE" and after_state.strip()
                        else None
                    )

                    survey = Survey(
                        id=str(uuid.uuid4()),
                        interaction_id=interaction.id,
                        timing=survey_timing,
                        recorded_at=utc_now(),
                        curiosity=float(curiosity),
                        understanding=float(understanding),
                        confidence=float(confidence),
                        recall_text=recall.strip() or None,
                        before_state=before_value,
                        after_state=after_value,
                        follow_through=(
                            follow_through
                            if survey_timing.startswith("DELAYED")
                            else None
                        ),
                    )

                    db.add(survey)
                    db.commit()

                    if survey_timing == "BASELINE":
                        message = "Baseline outcome recorded. The separate After state is recorded at IMMEDIATE."
                    elif survey_timing == "IMMEDIATE":
                        message = "Immediate outcome recorded. The separate Before state is recorded at BASELINE."
                    else:
                        message = "Delayed outcome recorded. Before/After state fields are intentionally limited to BASELINE and IMMEDIATE."

                    st.success(message)

            # Show the latest stored state records for this exact participant,
            # so the user can verify that the two time points are actually
            # present even after the form is submitted.
            stored_surveys = (
                db.query(Survey)
                .filter(Survey.interaction_id == interaction.id)
                .order_by(Survey.recorded_at.desc(), Survey.id.desc())
                .all()
            )

            latest_baseline = next(
                (s for s in stored_surveys if s.timing == "BASELINE"),
                None,
            )
            latest_immediate = next(
                (s for s in stored_surveys if s.timing == "IMMEDIATE"),
                None,
            )

            render_html('<div class="section-heading">Recorded before / after states</div>')

            state_col1, state_col2 = st.columns(2)
            with state_col1:
                st.markdown("**BEFORE · BASELINE**")
                st.write(
                    (latest_baseline.before_state if latest_baseline else None)
                    or "Not recorded yet."
                )
            with state_col2:
                st.markdown("**AFTER · IMMEDIATE**")
                st.write(
                    (latest_immediate.after_state if latest_immediate else None)
                    or "Not recorded yet."
                )

            st.caption(
                "These two records are stored independently. The Impact Observatory "
                "does not use these free-text state descriptions as quantitative scores; "
                "quantitative change is calculated only from the numeric measurements."
            )


# ============================================================
# 25A. PERSIST CURRENT WORKSPACE STATE
# ============================================================
try:
    persist_app_state(db, USER_ACCOUNT_ID)
except Exception:
    db.rollback()


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
        <div style="margin-top:12px;">
        Designed &amp; Engineered by Nikolai de Silva &bull; &copy; 2026 ninolades.com
    </div>
</div>
""")
