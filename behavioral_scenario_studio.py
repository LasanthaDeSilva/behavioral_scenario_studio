
import os
import uuid
import html
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import streamlit as st

from sqlalchemy import (
    create_engine, Column, String, DateTime, Text, Float, Integer,
    ForeignKey, Boolean, text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

# Optional dependency used only for persistent browser identity.
# Install with: pip install streamlit-cookies-controller
try:
    from streamlit_cookies_controller import CookieController
except Exception:
    CookieController = None

APP_TITLE = "Outreach Intelligence Lab"
APP_VERSION = "1.3.0"

# EXISTING REASONING MODELS — deliberately unchanged.
MODEL_FLASH = "gemini-3.6-flash"
MODEL_PRO = "gemini-3.1-pro"
MODEL_LITE = "gemini-3.5-flash-lite"
DEFAULT_MODEL = MODEL_FLASH

# Separate Gemini Live API model. This does NOT replace or alter the
# three existing reasoning models above.
LIVE_MODEL = "gemini-3.1-flash-live-preview"

DATABASE_URL = os.getenv(
    "OUTREACH_DATABASE_URL",
    "sqlite:///ninolades_outreach_lab.db",
)

COOKIE_NAME = "ninolades_outreach_user"
COOKIE_MAX_AGE = 60 * 60 * 24 * 365 * 5

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------
# Persistent per-browser identity
# ---------------------------------------------------------------------
def new_user_id() -> str:
    return f"user_{uuid.uuid4().hex}"

def get_persistent_user_id() -> str:
    """
    Identity is deliberately NOT stored in the URL.

    A shared URL therefore does not carry another user's memory.
    The browser cookie is the stable per-browser namespace and survives
    Streamlit reruns and normal refreshes.
    """
    if "persistent_user_id" in st.session_state:
        return st.session_state["persistent_user_id"]

    if CookieController is not None:
        controller = CookieController(key="outreach_identity")
        existing = controller.get(COOKIE_NAME)
        if existing:
            user_id = str(existing)
        else:
            user_id = new_user_id()
            controller.set(COOKIE_NAME, user_id, max_age=COOKIE_MAX_AGE)
    else:
        # Safe fallback for environments that did not install the optional
        # cookie component. This preserves isolation for the current run,
        # but only the cookie-enabled path provides refresh persistence.
        user_id = new_user_id()

    st.session_state["persistent_user_id"] = user_id
    return user_id

USER_ID = get_persistent_user_id()

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
PREMIUM_CSS = """
<style>
:root{
 --bg:#0b0b0d;--surface:#111114;--surface2:#161619;--surface3:#1b1b20;
 --border:#28282e;--borderSoft:#202024;--text:#f5f5f7;--secondary:#a1a1aa;
 --muted:#71717a;--accent:#5b8cff;--success:#4ade80;--warning:#fbbf24;
 --danger:#f87171;
}
html,body,[class*="css"]{
 font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text",
 "Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
.stApp{background:radial-gradient(circle at 50% -20%,rgba(91,140,255,.07),transparent 35%),var(--bg);color:var(--text)}
.block-container{max-width:1500px;padding-top:2.5rem;padding-bottom:5rem}
h1,h2,h3,h4{color:var(--text)!important;font-weight:500!important;letter-spacing:-.025em}
h1{font-size:2.6rem!important}h2{font-size:1.8rem!important}h3{font-size:1.25rem!important}
p,label,span{color:var(--secondary)}
[data-testid="stSidebar"]{background:#0e0e10;border-right:1px solid var(--borderSoft)}
[data-testid="stSidebar"] *{color:var(--secondary)}
.stTextInput input,.stTextArea textarea,.stSelectbox div[data-baseweb="select"],
.stMultiSelect div[data-baseweb="select"]{
 background:var(--surface2)!important;color:var(--text)!important;
 border:1px solid var(--border)!important;border-radius:9px!important
}
.stTextInput input:focus,.stTextArea textarea:focus{
 border-color:var(--accent)!important;box-shadow:0 0 0 1px var(--accent)!important
}
.stButton button{
 background:rgba(255,255,255,.03);color:var(--text);
 border:1px solid rgba(255,255,255,.15);border-radius:8px;font-weight:500;
 min-height:42px;transition:all .2s ease
}
.stButton button:hover{border-color:var(--secondary);background:rgba(255,255,255,.06);
 color:#fff;box-shadow:0 4px 8px rgba(0,0,0,.15);transform:translateY(-1px)}
button[data-testid="baseButton-primary"]{
 background:var(--accent)!important;border:1px solid var(--accent)!important;color:#fff!important;
 box-shadow:0 2px 8px rgba(91,140,255,.25)!important
}
button[data-testid="baseButton-primary"]:hover{background:#6b9aff!important;border-color:#6b9aff!important}
div[data-testid="stRadio"]>div{display:flex;gap:12px;flex-wrap:wrap}
div[data-testid="stRadio"] label{
 background:rgba(255,255,255,.02);padding:10px 20px;border-radius:12px;
 border:1px solid rgba(255,255,255,.08)!important;cursor:pointer;transition:.25s
}
div[data-testid="stRadio"] label:hover{border-color:rgba(255,255,255,.3)!important;background:rgba(255,255,255,.06)}
div[data-testid="stRadio"] label[data-checked="true"]{
 background:var(--accent)!important;border-color:var(--accent)!important
}
div[data-testid="stRadio"] label[data-checked="true"] p{color:#fff!important;font-weight:600}
.stTabs [data-baseweb="tab-list"]{gap:28px;border-bottom:1px solid var(--borderSoft)}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;background:transparent!important}
.stTabs [aria-selected="true"]{color:#fff!important;border-bottom:2px solid var(--accent)!important}
.stAlert{border-radius:10px!important}
.premium-card{background:linear-gradient(145deg,rgba(255,255,255,.025),rgba(255,255,255,.012));
 border:1px solid var(--border);border-radius:14px;padding:24px;margin-bottom:18px}
.premium-card:hover{border-color:#34343c}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:13px;padding:20px;min-height:120px}
.metric-label{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
.metric-value{color:var(--text);font-size:1.8rem;font-weight:500}
.metric-sub{color:var(--muted);font-size:.82rem;margin-top:5px}
.section-heading{margin-top:28px;margin-bottom:16px;padding-bottom:11px;border-bottom:1px solid var(--borderSoft);
 color:var(--text);font-size:1.15rem;font-weight:500}
.eyebrow{color:var(--accent);font-size:.72rem;text-transform:uppercase;letter-spacing:.13em;font-weight:600;margin-bottom:8px}
.badge{display:inline-block;padding:5px 9px;border-radius:6px;border:1px solid var(--border);
 background:var(--surface2);color:var(--secondary);font-size:.76rem}
.observation-row{padding:12px 14px;margin-bottom:8px;background:#101013;border:1px solid var(--borderSoft);border-radius:9px}
.hero{padding:18px 0 30px}.hero-title{color:#fff;font-size:2.7rem;font-weight:500;letter-spacing:-.04em}
.hero-subtitle{color:var(--muted);font-size:1rem;max-width:850px;line-height:1.65}
.small-note{color:var(--muted);font-size:.78rem;line-height:1.5}
hr{border-color:var(--borderSoft)!important}
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

Base = declarative_base()

class UserMemory(Base):
    __tablename__ = "user_memory"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, nullable=False)
    event_type = Column(String, nullable=False)
    payload = Column(Text, nullable=False)

class Event(Base):
    __tablename__ = "events"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    session_token = Column(String, nullable=False, default="")  # legacy-compatible
    name = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    objective = Column(String, nullable=False)
    context = Column(Text, nullable=True)
    environment = Column(String, nullable=True)
    sensory_environment = Column(String, nullable=True)
    acoustic_environment = Column(String, nullable=True)
    target_audience = Column(String, nullable=True)
    interactions = relationship("Interaction", back_populates="event", cascade="all, delete-orphan")

class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(String, primary_key=True)
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    participant_code = Column(String, nullable=False)
    started_at = Column(DateTime, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    phase = Column(String, default="Approach", nullable=False)
    stated_preference = Column(Text, nullable=True)
    event = relationship("Event", back_populates="interactions")
    observations = relationship("Observation", back_populates="interaction", cascade="all, delete-orphan")
    surveys = relationship("Survey", back_populates="interaction", cascade="all, delete-orphan")

class Observation(Base):
    __tablename__ = "observations"
    id = Column(String, primary_key=True)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    category = Column(String, nullable=False)
    detail = Column(Text, nullable=False)
    evidence_level = Column(String, nullable=False, default="OBSERVED")
    interaction = relationship("Interaction", back_populates="observations")

class Survey(Base):
    __tablename__ = "surveys"
    id = Column(String, primary_key=True)
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    timing = Column(String, nullable=False)
    curiosity = Column(Float, nullable=True)
    understanding = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    recall_text = Column(Text, nullable=True)
    follow_through = Column(Boolean, nullable=True)
    interaction = relationship("Interaction", back_populates="surveys")

class RapidStateLog(Base):
    __tablename__ = "rapid_state_logs"
    id = Column(String, primary_key=True)
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    participant_code = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    baseline_level = Column(String, nullable=False)
    current_state = Column(String, nullable=False)
    event = relationship("Event")

class AIResult(Base):
    __tablename__ = "ai_results"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    event_id = Column(String, nullable=True, index=True)
    interaction_id = Column(String, nullable=True, index=True)
    created_at = Column(DateTime, nullable=False)
    kind = Column(String, nullable=False)
    model = Column(String, nullable=False)
    payload = Column(Text, nullable=False)

@st.cache_resource
def get_db_engine():
    kwargs = {}
    if DATABASE_URL.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    return create_engine(DATABASE_URL, **kwargs)

@st.cache_resource
def get_session_factory(_engine):
    return sessionmaker(bind=_engine, autoflush=False, autocommit=False)

engine = get_db_engine()
Base.metadata.create_all(bind=engine)
SessionLocal = get_session_factory(engine)

def db_session():
    return SessionLocal()

def migrate_legacy_events():
    # Existing databases may have events without user_id.
    # SQLite cannot safely infer ownership, so legacy global rows are not
    # silently assigned to a new browser. New writes are always isolated.
    try:
        with engine.begin() as conn:
            cols = [r[1] for r in conn.execute(text("PRAGMA table_info(events)"))]
            if "user_id" not in cols:
                conn.execute(text("ALTER TABLE events ADD COLUMN user_id VARCHAR"))
            if "session_token" not in cols:
                conn.execute(text("ALTER TABLE events ADD COLUMN session_token VARCHAR DEFAULT ''"))
            for table in ["interactions","observations","surveys","rapid_state_logs"]:
                cols2 = [r[1] for r in conn.execute(text(f"PRAGMA table_info({table})"))]
                if "user_id" not in cols2:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN user_id VARCHAR"))
    except Exception:
        pass

migrate_legacy_events()
db = db_session()

# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
ConfidenceLevel = Literal["Low","Moderate","High"]

class CognitiveEstimate(BaseModel):
    bandwidth_pct:int=Field(ge=0,le=100); focus_pct:int=Field(ge=0,le=100)
    sensory_load_pct:int=Field(ge=0,le=100); rationale:str

class OutreachRecommendation(BaseModel):
    recommended_action:str; rationale:str; confidence:ConfidenceLevel
    evidence:List[str]=Field(min_length=1,max_length=6)
    alternative_explanation:str; next_observation:str
    cognitive_estimate:CognitiveEstimate

class PredictedPathway(BaseModel):
    pathway:str; mechanism:str; expected_signal:str; uncertainty:str

class ForwardModel(BaseModel):
    engagement_state:str
    predicted_pathways:List[PredictedPathway]=Field(min_length=3,max_length=3)
    recommended_outreach_design:List[str]=Field(min_length=3,max_length=5)
    likely_friction_points:List[str]=Field(min_length=1,max_length=5)
    measurement_opportunities:List[str]=Field(min_length=2,max_length=5)

class CounterfactualModel(BaseModel):
    changed_variable:str; expected_difference:str; before_state:str; after_state:str
    predicted_effects:List[str]=Field(min_length=3,max_length=5); uncertainty:str

class ThemeModel(BaseModel):
    theme:str; description:str; evidence_strength:ConfidenceLevel
    evidence_quotes:List[str]=Field(min_length=1,max_length=4)

class ImpactInterpretation(BaseModel):
    overall_interpretation:str; strongest_signal:str; weakest_signal:str
    plausible_mechanisms:List[str]=Field(min_length=2,max_length=5)
    alternative_explanations:List[str]=Field(min_length=2,max_length=5)
    recommended_next_test:str

class OutcomePrediction(BaseModel):
    focus_pct:int=Field(ge=0,le=100); stress_reduction_pct:int=Field(ge=0,le=100)
    cognitive_load_pct:int=Field(ge=0,le=100); attention_retention_pct:int=Field(ge=0,le=100)
    predicted_curiosity_shift:str; predicted_understanding_shift:str
    predicted_engagement_rate:str; overall_outcome_narrative:str
    risk_factors:List[str]=Field(min_length=1,max_length=5)
    success_amplifiers:List[str]=Field(min_length=1,max_length=5)

AI_SYSTEM = """
You are the reasoning layer of a public-facing science outreach intelligence system.
Treat internal states as hypotheses unless directly stated or directly observable.
Never diagnose or infer protected/sensitive characteristics. Never claim observations
prove internal psychology. Distinguish STATED, OBSERVED, INFERRED and HYPOTHESIS.
Participant preferences outrank inference. Prefer low-pressure, reversible,
autonomy-preserving actions. "No intervention" is valid. Do not manipulate.
AI percentages are hypotheses, not physiological, neurological or clinical measurements.
"""

def get_api_key():
    key=""
    try: key=st.secrets.get("GEMINI_API_KEY","")
    except Exception: pass
    return key or os.getenv("GEMINI_API_KEY","")

@st.cache_resource
def create_gemini_client(api_key):
    if not api_key: return None
    try: return genai.Client(api_key=api_key)
    except Exception: return None

def run_gemini(client, model_name, prompt, schema, system_instruction=AI_SYSTEM, temperature=.2):
    if client is None: raise RuntimeError("Gemini client is unavailable. Configure GEMINI_API_KEY.")
    response=client.models.generate_content(
        model=model_name, contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema, temperature=temperature,
        )
    )
    if getattr(response,"parsed",None) is not None: return response.parsed
    raw=getattr(response,"text",None)
    if not raw: raise RuntimeError("Gemini returned an empty response.")
    raw=raw.strip()
    if raw.startswith("```json"): raw=raw[7:]
    elif raw.startswith("```"): raw=raw[3:]
    if raw.endswith("```"): raw=raw[:-3]
    return schema.model_validate_json(raw.strip())

def utc_now(): return datetime.now(timezone.utc)
def clean_text(v): return html.escape(str(v or ""))
def memory_event(kind, payload, event_id=None, interaction_id=None, model="local"):
    db.add(UserMemory(id=str(uuid.uuid4()),user_id=USER_ID,created_at=utc_now(),
                      event_type=kind,payload=str(payload)))
    if model != "local":
        db.add(AIResult(id=str(uuid.uuid4()),user_id=USER_ID,event_id=event_id,
                        interaction_id=interaction_id,created_at=utc_now(),
                        kind=kind,model=model,payload=str(payload)))
    db.commit()

def save_ai_result(kind, result, model, event_id=None, interaction_id=None):
    payload=result.model_dump() if hasattr(result,"model_dump") else result
    db.add(AIResult(id=str(uuid.uuid4()),user_id=USER_ID,event_id=event_id,
                    interaction_id=interaction_id,created_at=utc_now(),
                    kind=kind,model=model,payload=repr(payload)))
    db.commit()

def create_event(name,objective,context,environment,sensory_environment,acoustic_environment,target_audience):
    e=Event(id=str(uuid.uuid4()),user_id=USER_ID,session_token=USER_ID,name=name.strip(),
            date=utc_now(),objective=objective,context=context.strip(),
            environment=environment.strip(),sensory_environment=sensory_environment.strip(),
            acoustic_environment=acoustic_environment.strip(),target_audience=target_audience)
    db.add(e); db.commit()
    memory_event("experience_created",name,event_id=e.id)
    return e

def create_interaction(event_id):
    i=Interaction(id=str(uuid.uuid4()),event_id=event_id,user_id=USER_ID,
                  participant_code=f"P-{uuid.uuid4().hex[:8].upper()}",
                  started_at=utc_now(),phase="Approach")
    db.add(i); db.commit(); memory_event("interaction_started",i.participant_code,event_id=event_id,interaction_id=i.id)
    return i

def log_observation(interaction_id,category,detail,evidence="OBSERVED"):
    o=Observation(id=str(uuid.uuid4()),interaction_id=interaction_id,user_id=USER_ID,
                  timestamp=utc_now(),category=category,detail=detail,evidence_level=evidence)
    db.add(o); db.commit(); memory_event("observation",f"{category}|{detail}",interaction_id=interaction_id)

def get_recent_observations(interaction_id,limit=12):
    return (db.query(Observation).filter(Observation.user_id==USER_ID,
            Observation.interaction_id==interaction_id).order_by(Observation.timestamp.desc()).limit(limit).all())

def get_events():
    return db.query(Event).filter(Event.user_id==USER_ID).order_by(Event.date.desc()).all()

def get_interactions(event_id):
    return db.query(Interaction).filter(Interaction.user_id==USER_ID,Interaction.event_id==event_id).all()

def calculate_event_impact(event_id):
    interactions=get_interactions(event_id)
    if not interactions: return None
    ids=[i.id for i in interactions]
    surveys=db.query(Survey).filter(Survey.user_id==USER_ID,Survey.interaction_id.in_(ids)).all()
    if not surveys:
        return {"participants":len(interactions),"surveyed":0,"baseline_curiosity":None,"post_curiosity":None,
                "curiosity_change":None,"baseline_understanding":None,"post_understanding":None,
                "understanding_change":None,"baseline_confidence":None,"post_confidence":None,
                "confidence_change":None,"follow_through_rate":None,"recall_rate":None}
    df=pd.DataFrame([{"interaction_id":s.interaction_id,"timing":s.timing,"curiosity":s.curiosity,
                      "understanding":s.understanding,"confidence":s.confidence,
                      "recall_text":s.recall_text,"follow_through":s.follow_through} for s in surveys])
    base=df[df.timing=="BASELINE"].groupby("interaction_id").first()
    post=df[df.timing=="IMMEDIATE"].groupby("interaction_id").first()
    paired=base.join(post,lsuffix="_baseline",rsuffix="_post",how="inner")
    def mean(x):
        x=pd.to_numeric(x,errors="coerce").dropna() if x is not None else pd.Series(dtype=float)
        return None if x.empty else float(x.mean())
    curiosity_change=mean(paired.curiosity_post-paired.curiosity_baseline) if not paired.empty else None
    understanding_change=mean(paired.understanding_post-paired.understanding_baseline) if not paired.empty else None
    confidence_change=mean(paired.confidence_post-paired.confidence_baseline) if not paired.empty else None
    delayed=df[df.timing.isin(["DELAYED_24H","DELAYED_7D"])]
    ftr=None
    if not delayed.empty:
        valid=delayed[delayed.follow_through.notna()]
        if not valid.empty: ftr=float(valid.follow_through.astype(bool).mean()*100)
    immediate=df[df.timing=="IMMEDIATE"]
    recall=df[(df.timing=="IMMEDIATE") & df.recall_text.fillna("").str.strip().ne("")]
    rr=None if len(immediate)==0 else float(len(recall)/len(immediate)*100)
    return {"participants":len(interactions),"surveyed":len(df.interaction_id.unique()),
            "baseline_curiosity":mean(base.get("curiosity")),"post_curiosity":mean(post.get("curiosity")),
            "curiosity_change":curiosity_change,"baseline_understanding":mean(base.get("understanding")),
            "post_understanding":mean(post.get("understanding")),"understanding_change":understanding_change,
            "baseline_confidence":mean(base.get("confidence")),"post_confidence":mean(post.get("confidence")),
            "confidence_change":confidence_change,"follow_through_rate":ftr,"recall_rate":rr}

def forward_model(client,model,event,design):
    return run_gemini(client,model,f"""
EVENT:{event.name}\nOBJECTIVE:{event.objective}\nTARGET:{event.target_audience}
ENVIRONMENT:{event.environment}\nSENSORY:{event.sensory_environment}\nACOUSTIC:{event.acoustic_environment}
CONTEXT:{event.context}\nDESIGN:{design}
Create three plausible engagement pathways, useful design decisions,
friction points and measurable signals. Do not predict individual certainty.
""",ForwardModel,.25)

def live_recommendation(client,model,event,interaction,observations):
    obs="\n".join(f"[{o.evidence_level}] {o.category}: {o.detail}" for o in observations)
    return run_gemini(client,model,f"""
EVENT:{event.name}\nOBJECTIVE:{event.objective}\nENVIRONMENT:{event.environment}
ACOUSTIC:{event.acoustic_environment}\nSTATED:{interaction.stated_preference or "None"}
PHASE:{interaction.phase}\nOBSERVATIONS:\n{obs or "None"}
Give the most appropriate minimally disruptive, reversible, autonomy-preserving
next outreach action. No diagnosis. No-intervention is allowed.
""",OutreachRecommendation,.2)

def counterfactual(client,model,event,design,change):
    return run_gemini(client,model,f"""
EVENT:{event.name}\nOBJECTIVE:{event.objective}\nENVIRONMENT:{event.environment}
CURRENT DESIGN:{design}\nCHANGE:{change}
Explain plausible differences while clearly treating this as a theoretical counterfactual.
""",CounterfactualModel,.3)

def impact_interpretation(client,model,metrics):
    return run_gemini(client,model,f"""Deterministic event measurements:\n{metrics}\n
Interpret cautiously. Do not claim causation. Discuss strongest/weakest signal,
plausible mechanisms, alternative explanations and one next test.""",
        ImpactInterpretation,.2)

def outcome_prediction(client,model,event,crowd,situation,questionnaire):
    return run_gemini(client,model,f"""
EVENT:{event.name}\nOBJECTIVE:{event.objective}\nTARGET:{event.target_audience}
ENVIRONMENT:{event.environment}\nSENSORY:{event.sensory_environment}
CROWD:{crowd}\nSITUATION:{situation}\nQUESTIONNAIRE:{questionnaire}
Estimate focus, stress-reduction index, cognitive-load index, retention,
curiosity/understanding shifts, engagement rate, risks and amplifiers.
Treat all percentages as synthetic hypotheses, not measurements.
""",OutcomePrediction,.3)

def render_html(s):
    st.markdown(textwrap.dedent(s),unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
defaults={
 "active_event_id":None,"active_interaction_id":None,"last_recommendation":None,
 "last_forward_model":None,"last_counterfactual":None,"last_impact_interpretation":None,
 "last_prediction":None,"page":"Experience Designer","model_label":"Gemini 3.6 Flash",
}
for k,v in defaults.items(): st.session_state.setdefault(k,v)

render_html("""
<div class="hero"><div class="eyebrow">Ninolades Research Platform</div>
<div class="hero-title">Outreach Intelligence Lab</div>
<div class="hero-subtitle">A human-centered intelligence system for designing,
adapting, and evaluating science outreach experiences. It combines structured
observation, generative reasoning, counterfactual exploration, and real-world impact evidence.</div></div>
""")
st.markdown("---")

model_options={"Gemini 3.6 Flash":MODEL_FLASH,"Gemini 3.1 Pro":MODEL_PRO,"Gemini 3.5 Flash-Lite":MODEL_LITE}
c1,c2,c3=st.columns([1.5,1,3])
with c1:
    label=st.selectbox("Reasoning engine",list(model_options),key="model_label",label_visibility="collapsed")
    selected_model=model_options[label]
    st.caption("Flash is default for live. Pro for deep analysis. Flash-Lite for volume.")
with c2:
    if st.button("Clean Memory",use_container_width=True):
        # This now truly clears only THIS browser's persisted application memory.
        for cls in [AIResult,UserMemory,Observation,Survey,Interaction,RapidStateLog,Event]:
            if cls in [Observation,Survey,Interaction,RapidStateLog]:
                # children are deleted through event/interactions below
                pass
        event_ids=[e.id for e in get_events()]
        interaction_ids=[i.id for e in get_events() for i in get_interactions(e.id)]
        if interaction_ids:
            db.query(Observation).filter(Observation.user_id==USER_ID).delete(synchronize_session=False)
            db.query(Survey).filter(Survey.user_id==USER_ID).delete(synchronize_session=False)
            db.query(Interaction).filter(Interaction.user_id==USER_ID).delete(synchronize_session=False)
        db.query(RapidStateLog).filter(RapidStateLog.user_id==USER_ID).delete(synchronize_session=False)
        db.query(AIResult).filter(AIResult.user_id==USER_ID).delete(synchronize_session=False)
        db.query(UserMemory).filter(UserMemory.user_id==USER_ID).delete(synchronize_session=False)
        db.query(Event).filter(Event.user_id==USER_ID).delete(synchronize_session=False)
        db.commit()
        for k in ["active_event_id","active_interaction_id","last_recommendation","last_forward_model","last_counterfactual","last_impact_interpretation","last_prediction"]:
            st.session_state[k]=defaults[k]
        st.success("This browser's memory was cleared. Other users are untouched.")
        st.rerun()
    st.caption(f"Private browser memory · {USER_ID[:12]}")
with c3:
    pages=["Experience Designer","Live Copilot","Outcome Predictor","Scientific Reactions","Impact Observatory","Counterfactual Lab","Methodology"]
    page=st.radio("Workspace",pages,key="page",horizontal=True,label_visibility="collapsed")
st.markdown("---")

# Persist meaningful UI-state changes in this user's own audit trail.
_previous_page = st.session_state.get("_audit_page")
if _previous_page != page:
    memory_event("workspace_changed", page)
    st.session_state["_audit_page"] = page

_previous_model = st.session_state.get("_audit_model")
if _previous_model != selected_model:
    memory_event("reasoning_model_changed", selected_model)
    st.session_state["_audit_model"] = selected_model

api_key=get_api_key()
client=create_gemini_client(api_key)
if client is None: st.error("Gemini is not configured. Add GEMINI_API_KEY to Streamlit secrets or the environment.")

# ---------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------
if page=="Experience Designer":
    render_html('<div class="section-heading">Design an outreach experience</div>')
    a,b=st.columns(2)
    with a:
        name=st.text_input("Experience name",key="event_name",placeholder="e.g. Science Under the Stars or Local Ecology Walk")
        objective=st.selectbox("Primary objective",["Curiosity","Scientific understanding","Awe and wonder","Memory and retention","Question generation","Independent follow-through","General engagement"],key="objective")
        audience=st.selectbox("Audience",["General public","Students","Families","Educators","Astronomy enthusiasts","Eco-tourists","Mixed audience"],key="audience")
    with b:
        environment=st.text_input("Physical environment",key="environment",placeholder="Dark-sky lawn, school courtyard, museum...")
        acoustic=st.text_input("Acoustic / musical environment",key="acoustic",placeholder="Silent, ambient sound, live acoustic...")
        sensory=st.text_input("Relevant environmental conditions",key="sensory",placeholder="Lighting, crowd density, temperature, noise...")
    context=st.text_area("Experience description",key="context",height=130)
    render_html('<div class="section-heading">Optional design variables</div>')
    x,y,z=st.columns(3)
    pacing=x.selectbox("Pacing",["Slow and contemplative","Moderate","Fast and energetic","Variable"],key="pacing")
    style=y.selectbox("Interaction style",["Open observation","Facilitator-led","Question-led","Hands-on","Story-driven","Mixed"],key="interaction_style")
    autonomy=z.selectbox("Participant autonomy",["High","Moderate","Low"],key="optional_choice")
    design=f"Pacing: {pacing}\nInteraction style: {style}\nParticipant autonomy: {autonomy}"
    if st.button("Create Experience Model",type="primary",use_container_width=True):
        if not name.strip(): st.error("Enter an experience name.")
        elif not context.strip(): st.error("Describe the experience.")
        elif client is None: st.error("Gemini is unavailable.")
        else:
            try:
                e=create_event(name,objective,context,environment,sensory,acoustic,audience)
                st.session_state.active_event_id=e.id
                with st.spinner("Building engagement model..."):
                    m=forward_model(client,selected_model,e,design)
                st.session_state.last_forward_model=m.model_dump()
                save_ai_result("forward_model",m,selected_model,event_id=e.id)
                st.success("Experience initialized and model generated.")
            except Exception as exc: st.error(f"Could not create the experience: {exc}")
    if st.session_state.last_forward_model:
        m=st.session_state.last_forward_model
        render_html(f'<div class="premium-card"><div class="eyebrow">Current model</div><div style="color:white;font-size:1.25rem">{clean_text(m["engagement_state"])}</div><div class="small-note">Generated hypothesis, not participant measurement.</div></div>')
        cols=st.columns(3)
        for idx,p in enumerate(m["predicted_pathways"]):
            with cols[idx]:
                render_html(f'<div class="premium-card" style="height:100%"><div class="eyebrow">Pathway {idx+1}</div><h3>{clean_text(p["pathway"])}</h3><p>{clean_text(p["mechanism"])}</p><div class="small-note">Expected signal:<br>{clean_text(p["expected_signal"])}</div><br><div class="small-note">Uncertainty:<br>{clean_text(p["uncertainty"])}</div></div>')
        q,r=st.columns(2)
        with q:
            render_html('<div class="section-heading">Design opportunities</div>')
            for item in m["recommended_outreach_design"]: st.markdown(f"- {item}")
        with r:
            render_html('<div class="section-heading">Potential friction</div>')
            for item in m["likely_friction_points"]: st.markdown(f"- {item}")
        render_html('<div class="section-heading">What should be measured?</div>')
        for item in m["measurement_opportunities"]: st.markdown(f"- {item}")

elif page=="Outcome Predictor":
    render_html('<div class="section-heading">Predict Outreach Outcomes</div>')
    events=get_events()
    if not events: st.info("Create an experience in Experience Designer first.")
    else:
        emap={f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})":e for e in events}
        e=emap[st.selectbox("Select Experience",list(emap),key="pred_event_select")]
        render_html('<div class="section-heading">Contextual Environment & Crowd Questionnaire</div>')
        q1,q2=st.columns(2)
        stress=q1.select_slider("Estimated Baseline Audience Stress Level",["Very Low / Relaxed","Moderate Stress","High Stress / Overwhelmed"],value="Moderate Stress",key="q_stress")
        noise=q1.select_slider("Ambient Distraction & Sensory Noise Level",["Quiet & Controlled","Moderate Noise","High Loudness / Busy Crowd"],value="Moderate Noise",key="q_noise")
        duration=q2.selectbox("Planned Session Duration",["Short (< 15 mins)","Standard (30-45 mins)","Extended (60+ mins)"],key="q_duration")
        density=q2.selectbox("Interactive Touchpoints Density",["Low (Passive listening)","Medium (Guided Q&A)","High (Hands-on exploration)"],index=1,key="q_density")
        crowd=st.text_area("Crowd details & demographics",key="pred_crowd",height=110)
        situation=st.text_area("Situational context",key="pred_situation",height=110)
        questionnaire=f"Baseline Stress: {stress}, Noise level: {noise}, Duration: {duration}, Interaction density: {density}"
        if st.button("Predict Outcome & Metrics",type="primary",use_container_width=True):
            if client is None: st.error("Gemini is unavailable.")
            elif not crowd.strip() or not situation.strip(): st.error("Please provide both crowd details and situational context.")
            else:
                try:
                    with st.spinner("Analyzing parameters and computing predicted outcomes..."):
                        p=outcome_prediction(client,selected_model,e,crowd,situation,questionnaire)
                    st.session_state.last_prediction=p.model_dump()
                    save_ai_result("outcome_prediction",p,selected_model,event_id=e.id)
                except Exception as exc: st.error(f"Prediction failed: {exc}")
        if st.session_state.last_prediction:
            p=st.session_state.last_prediction
            render_html('<div class="section-heading">Predicted Scientific Effects (Focus, Stress, Load & Retention)</div>')
            a,b,c,d=st.columns(4)
            a.metric("Predicted Focus State",f'{p["focus_pct"]}%'); b.metric("Stress Reduction Index",f'{p["stress_reduction_pct"]}%')
            c.metric("Cognitive Load Level",f'{p["cognitive_load_pct"]}%'); d.metric("Attention Retention",f'{p["attention_retention_pct"]}%')
            chart=pd.DataFrame({"Metric":["Focus State","Stress Reduction","Cognitive Load","Attention Retention"],"Percentage (%)":[p["focus_pct"],p["stress_reduction_pct"],p["cognitive_load_pct"],p["attention_retention_pct"]]}).set_index("Metric")
            st.bar_chart(chart)
            render_html('<div class="section-heading">Predicted Shifts & Outcomes</div>')
            x,y,z=st.columns(3)
            for col,label,key in [(x,"Curiosity Shift","predicted_curiosity_shift"),(y,"Understanding Shift","predicted_understanding_shift"),(z,"Engagement Rate","predicted_engagement_rate")]:
                with col: render_html(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="font-size:1.4rem">{clean_text(p[key])}</div></div>')
            render_html(f'<div class="section-heading">Outcome Narrative</div><div class="premium-card"><div style="color:#e5e5e7;line-height:1.7">{clean_text(p["overall_outcome_narrative"])}</div></div>')
            x,y=st.columns(2)
            with x:
                render_html('<div class="section-heading">Risk Factors</div>')
                for r in p["risk_factors"]: st.markdown(f"- {r}")
            with y:
                render_html('<div class="section-heading">Success Amplifiers</div>')
                for s in p["success_amplifiers"]: st.markdown(f"- {s}")

elif page=="Live Copilot":
    render_html('<div class="section-heading">Live outreach copilot</div>')
    events=get_events()
    if not events: st.info("Create an experience in Experience Designer first.")
    else:
        emap={f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})":e for e in events}
        e=emap[st.selectbox("Active experience",list(emap),key="live_copilot_event_select")]
        render_html('<div class="section-heading">Rapid Participant State Logger</div>')
        st.caption("One-click logging for rapid visual analysis. Each click automatically registers as a new participant.")
        base=st.radio("Baseline Level",["Calm / Receptive","Neutral / Unengaged","Low Energy / Fatigued","Distracted / Scatterbrained","Anxious / Stressed","High Energy / Excited"],key="rapid_base")
        state=st.radio("State of Mind / Reaction",["Awe / Wonder","Deep Focus / Flow","Curiosity / Inquisitive","Epiphany / Sudden Understanding","Cognitive Overload / Confusion","Disengagement / Boredom","Stress / Frustration","Relaxation / Comfort"],key="rapid_state")
        if st.button("Log as New Participant",type="primary",use_container_width=True):
            r=RapidStateLog(id=str(uuid.uuid4()),event_id=e.id,user_id=USER_ID,participant_code=f"RP-{uuid.uuid4().hex[:6].upper()}",timestamp=utc_now(),baseline_level=base,current_state=state)
            db.add(r); db.commit(); memory_event("rapid_state",f"{base}|{state}",event_id=e.id); st.toast("State logged successfully for new participant."); st.rerun()
        logs=db.query(RapidStateLog).filter(RapidStateLog.user_id==USER_ID,RapidStateLog.event_id==e.id).order_by(RapidStateLog.timestamp.desc()).limit(10).all()
        for r in logs:
            a,b,c,d,x=st.columns([1.5,1.5,2.5,2.5,1])
            a.caption(r.timestamp.strftime("%H:%M:%S UTC")); b.caption(r.participant_code); c.caption(r.baseline_level); d.caption(r.current_state)
            if x.button("Delete",key=f"del_rlog_{r.id}"):
                db.delete(r); db.commit(); memory_event("rapid_state_deleted",r.participant_code,event_id=e.id); st.rerun()
        if st.session_state.active_event_id!=e.id:
            st.session_state.active_event_id=e.id; st.session_state.active_interaction_id=None
        if not st.session_state.active_interaction_id:
            if st.button("Start participant interaction",type="primary",use_container_width=True):
                st.session_state.active_interaction_id=create_interaction(e.id).id; st.rerun()
        else:
            i=db.query(Interaction).filter(Interaction.user_id==USER_ID,Interaction.id==st.session_state.active_interaction_id).first()
            if i is None: st.session_state.active_interaction_id=None; st.rerun()
            render_html(f'<div class="premium-card"><div class="eyebrow">Active interaction</div><div style="color:white;font-size:1.2rem">{clean_text(i.participant_code)}</div><div class="small-note">Anonymous interaction code.</div></div>')
            phases=["Approach","Introduction","Waiting","Direct observation","Explanation","Question/discussion","Reflection","Exit"]
            phase=st.selectbox("Current phase",phases,index=phases.index(i.phase) if i.phase in phases else 0,key=f"phase_{i.id}")
            if phase!=i.phase: i.phase=phase; db.commit(); memory_event("phase_change",phase,interaction_id=i.id)
            pref=st.text_input("Participant-stated preference (Press Enter to save)",value=i.stated_preference or "",key=f"pref_{i.id}",placeholder="Only record what the participant explicitly states.")
            if pref!=(i.stated_preference or ""): i.stated_preference=pref; db.commit(); memory_event("stated_preference",pref,interaction_id=i.id)
            render_html('<div class="section-heading">Quick Observations</div>')
            buttons=[("Attention","Participant appears engaged/focused."),("Attention","Participant looks away/distracted."),("Participation","Participant asks a question."),("Participation","Participant gives a detailed response."),("Participation","Participant listens without responding."),("Reflection","Participant pauses to reflect."),("Friction","Participant has difficulty interacting."),("Friction","Environmental interruption occurs.")]
            cols=st.columns(4)
            for n,(cat,detail) in enumerate(buttons):
                if cols[n%4].button(detail,key=f"obs_btn_{n}",use_container_width=True): log_observation(i.id,cat,detail); st.toast("Observation recorded.")
            with st.form(f"custom_obs_{i.id}",clear_on_submit=True):
                custom=st.text_input("Custom observation",placeholder="Describe only what was directly observed.")
                if st.form_submit_button("Log Observation") and custom.strip(): log_observation(i.id,"Custom",custom.strip()); st.toast("Custom observation recorded."); st.rerun()
            render_html('<div class="section-heading">Recent evidence</div>')
            obs=get_recent_observations(i.id)
            for o in obs: render_html(f'<div class="observation-row"><span class="badge">{clean_text(o.evidence_level)}</span><span style="margin-left:8px;color:#e5e5e7">{clean_text(o.detail)}</span></div>')
            render_html('<div class="section-heading">Adaptive guidance</div>')
            if st.button("Generate next best outreach action",type="primary",use_container_width=True):
                if client is None: st.error("Gemini is unavailable.")
                else:
                    try:
                        with st.spinner("Reasoning over current evidence..."): rec=live_recommendation(client,selected_model,e,i,obs)
                        st.session_state.last_recommendation=rec.model_dump(); save_ai_result("live_recommendation",rec,selected_model,event_id=e.id,interaction_id=i.id)
                    except Exception as exc: st.error(f"Recommendation failed: {exc}")
            if st.session_state.last_recommendation:
                r=st.session_state.last_recommendation
                render_html(f'<div class="premium-card" style="border-color:rgba(91,140,255,.35)"><div class="eyebrow">Suggested next move</div><div style="color:white;font-size:1.35rem;margin-bottom:14px">{clean_text(r["recommended_action"])}</div><div style="color:#d4d4d8;line-height:1.6">{clean_text(r["rationale"])}</div></div>')
                a,b=st.columns(2)
                a.metric("Confidence",r["confidence"]); b.metric("Model-estimated focus",f'{r["cognitive_estimate"]["focus_pct"]}%')
                render_html('<div class="section-heading">Evidence used</div>')
                for q in r["evidence"]: st.markdown(f"- {q}")
                render_html('<div class="section-heading">Alternative explanation</div>'); st.write(r["alternative_explanation"])
                render_html('<div class="section-heading">Next observation to watch</div>'); st.write(r["next_observation"])
            if st.button("End interaction and start next participant",use_container_width=True):
                i.ended_at=utc_now(); db.commit(); memory_event("interaction_ended",i.participant_code,event_id=e.id,interaction_id=i.id)
                st.session_state.active_interaction_id=None; st.session_state.last_recommendation=None; st.rerun()

elif page=="Scientific Reactions":
    render_html('<div class="section-heading">Scientific Reaction Analysis</div>')
    st.caption("Visualizing accurate, logical, and practical scientific reactions logged during the experience.")
    events=get_events()
    if not events: st.info("No experiences available.")
    else:
        emap={f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})":e for e in events}; e=emap[st.selectbox("Select Experience",list(emap),key="sci_reac_event")]
        logs=db.query(RapidStateLog).filter(RapidStateLog.user_id==USER_ID,RapidStateLog.event_id==e.id).order_by(RapidStateLog.timestamp.asc()).all()
        if not logs: st.info("No rapid reaction data logged for this event yet. Use the Rapid State Logger in the Live Copilot page.")
        else:
            df=pd.DataFrame([{"timestamp":l.timestamp,"participant":l.participant_code,"baseline":l.baseline_level,"reaction":l.current_state} for l in logs])
            render_html(f'<div class="metric-card"><div class="metric-label">Total Rapid Logs</div><div class="metric-value">{len(logs)}</div></div>')
            a,b=st.columns(2)
            with a:
                render_html('<div class="section-heading">State of Mind / Reaction Distribution</div>'); st.bar_chart(df.reaction.value_counts())
            with b:
                render_html('<div class="section-heading">Baseline Level Distribution</div>'); st.bar_chart(df.baseline.value_counts())
            render_html('<div class="section-heading">Reaction Timeline</div>')
            df["timestamp"]=pd.to_datetime(df.timestamp); df["time_minute"]=df.timestamp.dt.floor("min")
            st.line_chart(df.groupby(["time_minute","reaction"]).size().unstack(fill_value=0))
            render_html('<div class="section-heading">Raw Log Data</div>'); st.dataframe(df,use_container_width=True,hide_index=True)

elif page=="Impact Observatory":
    render_html('<div class="section-heading">Real-world impact observatory</div>')
    render_html('<div class="premium-card"><div class="eyebrow">Why this exists</div><div style="color:#e5e5e7;line-height:1.7">The system separates AI predictions from what actually happened. Impact is calculated from recorded participant outcomes.</div></div>')
    events=get_events()
    if not events: st.info("No experiences have been created yet.")
    else:
        emap={f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})":e for e in events}; e=emap[st.selectbox("Experience",list(emap),key="obs_event_select")]
        metrics=calculate_event_impact(e.id)
        if metrics:
            a,b,c,d=st.columns(4)
            a.metric("Participants",metrics["participants"]); b.metric("Curiosity change","—" if metrics["curiosity_change"] is None else f'{metrics["curiosity_change"]:+.2f}')
            c.metric("Understanding change","—" if metrics["understanding_change"] is None else f'{metrics["understanding_change"]:+.2f}')
            d.metric("Follow-through","—" if metrics["follow_through_rate"] is None else f'{metrics["follow_through_rate"]:.1f}%')
            render_html('<div class="section-heading">Outcome signals</div>')
            sig={k:v for k,v in {"Curiosity":metrics["curiosity_change"],"Understanding":metrics["understanding_change"],"Confidence":metrics["confidence_change"]}.items() if v is not None}
            if sig: st.dataframe(pd.DataFrame([{"Signal":k,"Change":v} for k,v in sig.items()]),use_container_width=True,hide_index=True)
            else: st.info("Paired baseline/post measurements are needed to calculate change.")
            render_html('<div class="section-heading">Delayed indicators</div>')
            a,b=st.columns(2); a.metric("Recall response rate","—" if metrics["recall_rate"] is None else f'{metrics["recall_rate"]:.1f}%'); b.metric("Participants with survey data",metrics["surveyed"])
            render_html('<div class="section-heading">AI interpretation</div>'); st.caption("Gemini interprets the measurements; Python calculates them.")
            if st.button("Interpret impact",type="primary",use_container_width=True):
                if client is None: st.error("Gemini is unavailable.")
                else:
                    try:
                        with st.spinner("Interpreting outcome patterns..."): it=impact_interpretation(client,selected_model,metrics)
                        st.session_state.last_impact_interpretation=it.model_dump(); save_ai_result("impact_interpretation",it,selected_model,event_id=e.id)
                    except Exception as exc: st.error(f"Impact interpretation failed: {exc}")
            if st.session_state.last_impact_interpretation:
                it=st.session_state.last_impact_interpretation
                render_html(f'<div class="premium-card"><div class="eyebrow">Interpretation</div><div style="color:#e5e5e7;line-height:1.7">{clean_text(it["overall_interpretation"])}</div></div>')
                a,b=st.columns(2); a.write(it["strongest_signal"]); b.write(it["weakest_signal"])
                render_html('<div class="section-heading">Plausible mechanisms</div>')
                for x in it["plausible_mechanisms"]: st.markdown(f"- {x}")
                render_html('<div class="section-heading">Alternative explanations</div>')
                for x in it["alternative_explanations"]: st.markdown(f"- {x}")
                render_html('<div class="section-heading">Recommended next test</div>'); st.info(it["recommended_next_test"])

elif page=="Counterfactual Lab":
    render_html('<div class="section-heading">Counterfactual experiment lab</div>')
    events=get_events()
    if not events: st.info("Create an experience first.")
    else:
        emap={f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})":e for e in events}; e=emap[st.selectbox("Experience",list(emap),key="cf_event_select")]
        render_html('<div class="premium-card"><div class="eyebrow">Counterfactual reasoning</div><div class="small-note">Change one meaningful variable while keeping the conceptual baseline constant. This creates testable hypotheses, not experimental evidence.</div></div>')
        change=st.text_area("What would you change?",key="cf_variable_change",height=100)
        design=st.text_area("Current design",value=e.context or "",key="cf_design_description",height=100)
        if st.button("Run counterfactual",type="primary",use_container_width=True):
            if not change.strip(): st.error("Describe the variable you want to change.")
            elif client is None: st.error("Gemini is unavailable.")
            else:
                try:
                    with st.spinner("Comparing hypothetical pathways..."): cf=counterfactual(client,selected_model,e,design,change)
                    st.session_state.last_counterfactual=cf.model_dump(); save_ai_result("counterfactual",cf,selected_model,event_id=e.id)
                except Exception as exc: st.error(f"Counterfactual failed: {exc}")
        if st.session_state.last_counterfactual:
            cf=st.session_state.last_counterfactual
            render_html(f'<div class="premium-card"><div class="eyebrow">Changed variable</div><h3>{clean_text(cf["changed_variable"])}</h3><p>{clean_text(cf["expected_difference"])}</p></div>')
            a,b=st.columns(2); a.write(cf["before_state"]); b.write(cf["after_state"])
            render_html('<div class="section-heading">Predicted effects</div>')
            for x in cf["predicted_effects"]: st.markdown(f"- {x}")
            render_html('<div class="section-heading">Uncertainty</div>'); st.warning(cf["uncertainty"])

elif page=="Methodology":
    render_html('<div class="section-heading">Science and methodology</div>')
    sections=[
      ("What the system actually does","The platform designs outreach, generates hypotheses, supports live observation, and compares hypotheses with real-world outcome data."),
      ("Prediction versus measurement","Gemini statements are predictions. Direct observations are observations. Survey-derived changes are deterministic measurements. These categories are not silently merged."),
      ("Evidence hierarchy","STATED means explicitly reported. OBSERVED means directly witnessed. INFERRED means interpretation. HYPOTHESIS means speculative explanation requiring more evidence."),
      ("Why the AI is not the measurement engine","Generative models are used for qualitative reasoning and alternatives. Quantitative impact changes are calculated deterministically in Python."),
      ("What real impact means here","Useful indicators can include curiosity, scientific understanding, confidence, recall, voluntary follow-through, return engagement and participant-described meaning."),
      ("Causality","Pre/post changes are descriptive evidence and do not automatically prove causation. Strong causal claims require stronger experimental or quasi-experimental designs."),
      ("Privacy","The application uses anonymous participant codes and a browser-specific app-memory namespace. Production deployments should still implement consent, retention, access control and deletion policies.")
    ]
    for title,body in sections:
        render_html(f'<div class="premium-card"><div class="eyebrow">Method</div><h3>{title}</h3><div style="color:#a1a1aa;line-height:1.75">{clean_text(body)}</div></div>')

# ---------------------------------------------------------------------
# Optional outcome capture
# ---------------------------------------------------------------------
st.markdown("---")
with st.expander("Optional participant outcome capture"):
    events=get_events()
    if not events: st.caption("Create an experience first.")
    else:
        emap={f"{e.name} ({e.date.strftime('%Y-%m-%d %H:%M')})":e for e in events}; e=emap[st.selectbox("Experience",list(emap),key="survey_event_select")]
        ints=get_interactions(e.id)
        if not ints: st.caption("No participant interactions yet.")
        else:
            imap={f"{i.participant_code} ({i.started_at.strftime('%H:%M:%S')})":i for i in ints}; i=imap[st.selectbox("Participant interaction",list(imap),key="survey_participant_select")]
            timing=st.selectbox("Measurement point",["BASELINE","IMMEDIATE","DELAYED_24H","DELAYED_7D"],key="survey_timing_select")
            with st.form("outcome_capture",clear_on_submit=True):
                curiosity=st.slider("Curiosity",1,10,5); understanding=st.slider("Scientific understanding",0,100,50)
                confidence=st.slider("Confidence asking/answering questions",1,10,5); recall=st.text_area("What do you remember most?",height=90)
                follow=st.checkbox("I voluntarily explored something further afterward")
                if st.form_submit_button("Save outcome"):
                    s=Survey(id=str(uuid.uuid4()),interaction_id=i.id,user_id=USER_ID,timing=timing,curiosity=float(curiosity),understanding=float(understanding),confidence=float(confidence),recall_text=recall.strip() or None,follow_through=follow if timing.startswith("DELAYED") else None)
                    db.add(s); db.commit(); memory_event("survey",f"{timing}|{curiosity}|{understanding}|{confidence}|{recall}",event_id=e.id,interaction_id=i.id); st.success("Outcome recorded.")

# ---------------------------------------------------------------------
# Real Gemini Live Voice
# ---------------------------------------------------------------------
def issue_live_ephemeral_token():
    """
    Server-side token minting. The long-lived GEMINI_API_KEY never reaches
    browser JavaScript. The token is restricted to the dedicated Live model.
    """
    if not api_key: return None
    try:
        live_client=genai.Client(api_key=api_key)
        now=utc_now()
        token=live_client.auth_tokens.create(config={
            "uses":1,
            "expire_time":now + __import__("datetime").timedelta(minutes=30),
            "new_session_expire_time":now + __import__("datetime").timedelta(minutes=1),
            "live_connect_constraints":{
                "model":LIVE_MODEL,
                "config":{
                    "response_modalities":["AUDIO"],
                    "input_audio_transcription":{},
                    "output_audio_transcription":{},
                    "session_resumption":{},
                    "context_window_compression":{"sliding_window":{}},
                },
            },
        })
        return getattr(token,"name",None)
    except Exception:
        return None

live_token=issue_live_ephemeral_token() if client is not None else None

# The component is deliberately rendered as an app-level fixed layer.
# It is not placed inside the normal Streamlit layout, so it remains visible
# while navigating between workspaces.
VOICE_HTML = r"""
<div id="nil-voice">
  <button id="nil-fab" aria-label="Live AI Voice" title="Live AI Voice">
    <span class="nil-ring"></span>
    <span class="nil-core">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 14.5a3.5 3.5 0 0 0 3.5-3.5V6a3.5 3.5 0 0 0-7 0v5a3.5 3.5 0 0 0 3.5 3.5Z"/>
        <path d="M18 11a6 6 0 0 1-12 0M12 17v4M9 21h6"/>
      </svg>
    </span>
  </button>
  <div id="nil-panel" aria-live="polite">
    <div class="nil-top"><span id="nil-dot"></span><span id="nil-status">Live AI Voice</span></div>
    <div id="nil-transcript">Off. Tap the voice control to start.</div>
    <div class="nil-actions">
      <button id="nil-stop">Stop</button>
    </div>
  </div>
</div>
"""
VOICE_CSS = r"""
#nil-voice{position:fixed;right:24px;bottom:24px;z-index:2147483647;font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","Segoe UI",sans-serif}
#nil-fab{position:relative;width:58px;height:58px;border-radius:50%;border:1px solid rgba(255,255,255,.18);
background:rgba(17,17,20,.94);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
box-shadow:0 12px 34px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,255,255,.08);cursor:pointer;padding:0}
.nil-core{position:absolute;inset:0;display:grid;place-items:center}
.nil-core svg{width:23px;height:23px;fill:none;stroke:#f5f5f7;stroke-width:1.65;stroke-linecap:round;stroke-linejoin:round}
.nil-ring{position:absolute;inset:-4px;border-radius:50%;border:1px solid transparent;transition:.2s}
#nil-fab.live{border-color:rgba(91,140,255,.8);box-shadow:0 0 0 1px rgba(91,140,255,.15),0 12px 34px rgba(0,0,0,.42),0 0 28px rgba(91,140,255,.22)}
#nil-fab.live .nil-ring{border-color:rgba(91,140,255,.55);animation:nilpulse 1.8s ease-out infinite}
@keyframes nilpulse{0%{transform:scale(.98);opacity:.8}70%{transform:scale(1.13);opacity:0}100%{transform:scale(1.13);opacity:0}}
#nil-panel{position:absolute;right:0;bottom:70px;width:310px;padding:14px;border:1px solid #28282e;border-radius:15px;
background:rgba(17,17,20,.96);backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
box-shadow:0 18px 45px rgba(0,0,0,.45);display:none;color:#f5f5f7}
#nil-panel.open{display:block}.nil-top{display:flex;align-items:center;gap:8px;font-size:11px;text-transform:uppercase;letter-spacing:.12em;font-weight:600}
#nil-dot{width:7px;height:7px;border-radius:50%;background:#71717a}
#nil-dot.live{background:#5b8cff;box-shadow:0 0 10px rgba(91,140,255,.75)}
#nil-transcript{margin-top:12px;color:#a1a1aa;font-size:13px;line-height:1.5;min-height:42px;max-height:150px;overflow:auto}
.nil-actions{display:flex;justify-content:flex-end;margin-top:12px}
#nil-stop{background:transparent;color:#a1a1aa;border:1px solid #28282e;border-radius:8px;padding:7px 11px;cursor:pointer}
@media(max-width:600px){#nil-voice{right:16px;bottom:16px}#nil-panel{right:0;width:min(310px,calc(100vw - 32px))}}
"""
# Streamlit V2 components execute in the app DOM rather than inside the
# old V1 iframe. That is important: a position:fixed control can therefore
# genuinely float over the whole Streamlit viewport.
import streamlit.components.v2 as components_v2

def mount_live_voice():
    if live_token:
        token_literal = repr(live_token)
        model_literal = repr(LIVE_MODEL)
        js = f"""
export default function(component) {{
  const root = component.parentElement;
  const TOKEN = {token_literal};
  const LIVE_MODEL = {model_literal};
  const panel = root.querySelector("#nil-panel");
  const fab = root.querySelector("#nil-fab");
  const dot = root.querySelector("#nil-dot");
  const status = root.querySelector("#nil-status");
  const transcript = root.querySelector("#nil-transcript");
  const stop = root.querySelector("#nil-stop");

  let session = null;
  let stream = null;
  let audioContext = null;
  let processor = null;
  let source = null;
  let active = false;
  let loading = false;
  let nextAudioTime = 0;
  const audioSources = new Set();

  const setUI = (on, message) => {{
    active = on;
    fab.classList.toggle("live", on);
    dot.classList.toggle("live", on);
    status.textContent = on ? "Live AI Voice · Listening" : "Live AI Voice · Off";
    transcript.textContent = message || (on ? "Listening…" : "Off. Tap the voice control to start.");
  }};

  const toBase64 = (buffer) => {{
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {{
      binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
    }}
    return btoa(binary);
  }};

  const toPCM16 = (samples) => {{
    const output = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) {{
      const x = Math.max(-1, Math.min(1, samples[i]));
      output[i] = x < 0 ? x * 32768 : x * 32767;
    }}
    return output;
  }};

  const resample = (input, fromRate, toRate) => {{
    if (fromRate === toRate) return input;
    const ratio = fromRate / toRate;
    const length = Math.max(1, Math.round(input.length / ratio));
    const output = new Float32Array(length);
    for (let i = 0; i < length; i++) {{
      const position = i * ratio;
      const left = Math.floor(position);
      const right = Math.min(left + 1, input.length - 1);
      const fraction = position - left;
      output[i] = input[left] * (1 - fraction) + input[right] * fraction;
    }}
    return output;
  }};

  const stopOutputAudio = () => {{
    for (const node of audioSources) {{
      try {{ node.stop(); }} catch (_) {{}}
      try {{ node.disconnect(); }} catch (_) {{}}
    }}
    audioSources.clear();
    nextAudioTime = 0;
  }};

  const playPCM = (encoded) => {{
    const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
    const pcm = new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));
    if (!audioContext) audioContext = new AudioContext();
    const buffer = audioContext.createBuffer(1, pcm.length, 24000);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;
    const node = audioContext.createBufferSource();
    node.buffer = buffer;
    node.connect(audioContext.destination);
    const startAt = Math.max(audioContext.currentTime + 0.01, nextAudioTime);
    node.start(startAt);
    nextAudioTime = startAt + buffer.duration;
    audioSources.add(node);
    node.onended = () => {{
      audioSources.delete(node);
      try {{ node.disconnect(); }} catch (_) {{}}
    }};
  }};

  const stopAll = async () => {{
    active = false;
    try {{ processor?.disconnect(); }} catch (_) {{}}
    try {{ source?.disconnect(); }} catch (_) {{}}
    try {{ stream?.getTracks().forEach(track => track.stop()); }} catch (_) {{}}
    try {{ if (audioContext && audioContext.state !== "closed") await audioContext.close(); }} catch (_) {{}}
    stopOutputAudio();
    try {{ session?.close(); }} catch (_) {{}}
    session = null;
    stream = null;
    processor = null;
    source = null;
    audioContext = null;
    setUI(false);
  }};

  const start = async () => {{
    if (loading || active) return;
    loading = true;
    panel.classList.add("open");
    setUI(false, "Connecting to Gemini Live…");

    try {{
      if (!TOKEN) throw new Error("No Live API token.");
      const {{ GoogleGenAI, Modality }} = await import("https://esm.sh/@google/genai");
      const ai = new GoogleGenAI({{ apiKey: TOKEN }});

      session = await ai.live.connect({{
        model: LIVE_MODEL,
        config: {{
          responseModalities: [Modality.AUDIO],
          inputAudioTranscription: {{}},
          outputAudioTranscription: {{}},
          sessionResumption: {{}},
          contextWindowCompression: {{ slidingWindow: {{}} }},
          systemInstruction: {{
            parts: [{{ text:
              "You are the live voice copilot for the Ninolades Outreach Intelligence Lab. " +
              "Be concise, calm and useful. Treat participant psychology as uncertain. " +
              "Never diagnose. Preserve participant autonomy. Give practical outreach guidance."
            }}]
          }}
        }},
        callbacks: {{
          onopen: () => setUI(true, "Listening…"),
          onmessage: (message) => {{
            const content = message.serverContent;
            if (!content) return;
            if (content.inputTranscription?.text) {{
              transcript.textContent = "You: " + content.inputTranscription.text;
            }}
            if (content.outputTranscription?.text) {{
              transcript.textContent = "Gemini: " + content.outputTranscription.text;
            }}
            if (content.interrupted) {{
              stopOutputAudio();
              transcript.textContent = "Listening…";
            }}
            if (content.modelTurn?.parts) {{
              for (const part of content.modelTurn.parts) {{
                if (part.inlineData?.data) playPCM(part.inlineData.data);
              }}
            }}
          }},
          onerror: () => {{
            active = false;
            stopAll();
            panel.classList.add("open");
            transcript.textContent = "Live voice error. Tap to reconnect.";
          }},
          onclose: () => {{
            if (active) {{
              active = false;
              stopAll();
              panel.classList.add("open");
              transcript.textContent = "Connection closed. Tap to reconnect.";
            }}
          }}
        }}
      }});

      stream = await navigator.mediaDevices.getUserMedia({{
        audio: {{
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }}
      }});

      audioContext = new AudioContext();
      await audioContext.resume();
      source = audioContext.createMediaStreamSource(stream);
      processor = audioContext.createScriptProcessor(4096, 1, 1);

      processor.onaudioprocess = (event) => {{
        if (!active || !session) return;
        const input = event.inputBuffer.getChannelData(0);
        const downsampled = resample(input, event.inputBuffer.sampleRate, 16000);
        const pcm = toPCM16(downsampled);
        session.sendRealtimeInput({{
          audio: {{
            data: toBase64(pcm.buffer),
            mimeType: "audio/pcm;rate=16000"
          }}
        }});
      }};

      source.connect(processor);
      processor.connect(audioContext.destination);
      setUI(true, "Listening…");
    }} catch (error) {{
      console.error("Live voice:", error);
      await stopAll();
      panel.classList.add("open");
      transcript.textContent =
        "Live voice could not start. Check microphone permission and Gemini Live availability.";
    }} finally {{
      loading = false;
    }}
  }};

  fab.onclick = () => {{
    panel.classList.toggle("open");
    if (!active) start();
  }};

  stop.onclick = async () => {{
    await stopAll();
    panel.classList.add("open");
  }};

  setUI(false, "Off. Tap the voice control to start.");

  return () => {{
    stopAll();
  }};
}}
"""
    else:
        js = """
export default function(component) {
  const root = component.parentElement;
  const panel = root.querySelector("#nil-panel");
  const transcript = root.querySelector("#nil-transcript");
  const fab = root.querySelector("#nil-fab");
  panel.classList.add("open");
  transcript.textContent = "Live voice unavailable until GEMINI_API_KEY is configured.";
  fab.disabled = true;
  return () => {};
}
"""

    renderer = components_v2.component(
        "ninolades_live_ai_voice",
        html=VOICE_HTML,
        css=VOICE_CSS,
        js=js,
        isolate_styles=False,
    )
    renderer(key="ninolades_live_ai_voice_instance")

mount_live_voice()

render_html("""
<div style="text-align:center;margin-top:70px;padding-top:25px;border-top:1px solid #202024;color:#52525b;font-size:.78rem;line-height:1.6">
<div style="color:#71717a;margin-bottom:8px">Outreach Intelligence Lab</div>
<div>Exploratory generative modeling and evidence-informed science outreach.</div>
<div style="max-width:850px;margin:12px auto 0">AI-generated predictions are synthetic hypotheses. They do not establish psychological, neurological, clinical, or causal facts about individuals. Real-world impact metrics are calculated from recorded observations and participant-reported outcomes.</div>
</div>
""")
db.close()
