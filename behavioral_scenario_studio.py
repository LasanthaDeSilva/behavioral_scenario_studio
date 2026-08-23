import os
import uuid
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, DateTime, Text, Float, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from pydantic import BaseModel, Field
from typing import List, Literal
from google import genai
from google.genai import types

# ==========================================
# 1. DETERMINISTIC DATABASE LAYER (SQLAlchemy)
# ==========================================
DB_PATH = os.getenv("DB_PATH", "sqlite:///ninolades_outreach.db")
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Event(Base):
    __tablename__ = "events"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    date = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    primary_objective = Column(String)
    acoustic_setting = Column(String)
    interactions = relationship("Interaction", back_populates="event", cascade="all, delete-orphan")

class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String, ForeignKey("events.id"), nullable=False)
    participant_id = Column(String, default=lambda: str(uuid.uuid4()))
    timestamp_start = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    phase = Column(String, default="Phase 1 - Approach")
    stated_preference = Column(String, nullable=True)
    
    event = relationship("Event", back_populates="interactions")
    observations = relationship("Observation", back_populates="interaction", cascade="all, delete-orphan")
    surveys = relationship("Survey", back_populates="interaction", cascade="all, delete-orphan")

class Observation(Base):
    __tablename__ = "observations"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    category = Column(String, nullable=False)
    detail = Column(String, nullable=False)
    evidence_level = Column(String, default="OBSERVED") # DIRECT, OBSERVED, INFERRED, HYPOTHESIS
    
    interaction = relationship("Interaction", back_populates="observations")

class Survey(Base):
    __tablename__ = "surveys"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    interaction_id = Column(String, ForeignKey("interactions.id"), nullable=False)
    timing = Column(String, nullable=False) # 'BASELINE', 'IMMEDIATE', 'DELAYED_24H', 'DELAYED_7D'
    curiosity_score = Column(Float, nullable=True) # 1-10
    knowledge_score = Column(Float, nullable=True) # Percentage 0-100
    memorability_text = Column(Text, nullable=True)
    follow_through = Column(String, nullable=True) # 'Yes' or 'No'
    
    interaction = relationship("Interaction", back_populates="surveys")

# Initialize database
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 2. AI INTERPRETATION LAYER (Gemini & Pydantic)
# ==========================================
class OutreachRecommendation(BaseModel):
    recommended_action: str = Field(..., description="Specific action. 'No intervention' is valid if engaged.")
    rationale: str = Field(..., description="Evidence-based reasoning.")
    confidence: Literal["Low", "Moderate", "High"]
    evidence: List[str] = Field(..., description="Observed signals used.")
    alternative_explanation: str = Field(..., description="Alternative interpretation.")
    next_observation: str = Field(..., description="What to watch for next.")

class QualitativeTheme(BaseModel):
    theme: str
    evidence: List[str]
    confidence: Literal["Low", "Moderate", "High"]

SYSTEM_INSTRUCTION = """
You are the interpretation layer for the Ninolades Outreach Intelligence platform.
RULES:
1. Treat participant attributes as hypotheses, never facts.
2. Never infer a diagnosis or clinical condition.
3. Prefer participant-stated preferences over inferred ones.
4. "No intervention" is valid if the participant is engaged.
5. Recommend minimally disruptive and reversible interventions.
"""

@st.cache_resource
def get_ai_client():
    try:
        return genai.Client()
    except Exception:
        return None

def get_adaptation(context_data: dict, client) -> OutreachRecommendation:
    prompt = f"""
    Context:
    Phase: {context_data.get('phase')}
    Preference: {context_data.get('preference', 'Unknown')}
    Recent Observations: {context_data.get('observations')}
    Event Objectives: {context_data.get('objectives')}
    
    Provide the next best action using the strict schema.
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=OutreachRecommendation,
            temperature=0.2
        )
    )
    return response.parsed

def extract_theme(memory_text: str, client) -> QualitativeTheme:
    prompt = f"Extract the primary memory anchor from this participant response: '{memory_text}'"
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction="Categorize the memory into a concise theme.",
            response_mime_type="application/json",
            response_schema=QualitativeTheme,
            temperature=0.1
        )
    )
    return response.parsed

# ==========================================
# 3. DETERMINISTIC ANALYTICS ENGINE (Pandas)
# ==========================================
def calculate_impact_fingerprint(db, event_id: str):
    query = db.query(Survey).join(Interaction).filter(Interaction.event_id == event_id)
    df = pd.read_sql(query.statement, db.bind)
    
    if df.empty:
        return None

    baseline = df[df['timing'] == 'BASELINE']
    immediate = df[df['timing'] == 'IMMEDIATE']
    delayed = df[df['timing'].isin(['DELAYED_24H', 'DELAYED_7D'])]
    
    metrics = {
        "total_participants": len(df['interaction_id'].unique()),
        "baseline_curiosity": baseline['curiosity_score'].mean() if not baseline.empty else 0,
        "post_curiosity": immediate['curiosity_score'].mean() if not immediate.empty else 0,
        "baseline_knowledge": baseline['knowledge_score'].mean() if not baseline.empty else 0,
        "post_knowledge": immediate['knowledge_score'].mean() if not immediate.empty else 0,
    }
    
    metrics["curiosity_change"] = round(metrics["post_curiosity"] - metrics["baseline_curiosity"], 2)
    metrics["knowledge_change"] = round(metrics["post_knowledge"] - metrics["baseline_knowledge"], 2)
    
    if not delayed.empty and 'follow_through' in delayed.columns:
        metrics["follow_through_rate"] = (delayed['follow_through'] == 'Yes').mean() * 100
    else:
        metrics["follow_through_rate"] = 0.0
        
    return metrics

# ==========================================
# 4. STREAMLIT UI LAYER
# ==========================================
st.set_page_config(page_title="Ninolades Outreach", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: #f8f9fa; }
    .stButton>button { border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

db = next(get_db())
ai_client = get_ai_client()

st.sidebar.title("🔭 Ninolades Intelligence")
page = st.sidebar.radio("Modules", ["A: Event Builder", "B: Live Copilot", "C: Impact & Surveys", "E: Observatory"])

def get_active_events():
    return db.query(Event).all()

# --- MODULE A: EVENT BUILDER ---
if page == "A: Event Builder":
    st.title("Create Outreach Event")
    with st.form("event_builder"):
        name = st.text_input("Event Name")
        objective = st.selectbox("Primary Objective", ["Curiosity", "Scientific Understanding", "Awe/Wonder", "Retention"])
        acoustic = st.selectbox("Acoustic Setting", ["No music", "Ambient music", "Live acoustic music", "Storytelling with music"])
        
        if st.form_submit_button("Initialize Event"):
            if name.strip():
                new_event = Event(name=name, primary_objective=objective, acoustic_setting=acoustic)
                db.add(new_event)
                db.commit()
                st.success(f"Event '{name}' initialized.")
            else:
                st.error("Please provide an Event Name.")

# --- MODULE B: LIVE COPILOT ---
elif page == "B: Live Copilot":
    st.title("Live Outreach Copilot")
    events = get_active_events()
    if not events:
        st.warning("Create an event in Module A first.")
        st.stop()
        
    event_dict = {e.name: e for e in events}
    active_event_name = st.selectbox("Active Event", list(event_dict.keys()))
    event_record = event_dict[active_event_name]
    
    if "current_interaction" not in st.session_state:
        new_int = Interaction(event_id=event_record.id)
        db.add(new_int)
        db.commit()
        st.session_state.current_interaction = new_int.id
        
    st.caption(f"Interaction ID: `{st.session_state.current_interaction}`")
    
    st.markdown("### Rapid Observation Logging")
    c1, c2, c3 = st.columns(3)
    def log_obs(cat, det, level="OBSERVED"):
        obs = Observation(interaction_id=st.session_state.current_interaction, category=cat, detail=det, evidence_level=level)
        db.add(obs)
        db.commit()
        st.toast(f"Logged: {det}")
        
    with c1:
        if st.button("👁️ Observing Target", use_container_width=True): log_obs("Attention", "observing target")
        if st.button("👀 Looking Elsewhere", use_container_width=True): log_obs("Attention", "looking elsewhere")
    with c2:
        if st.button("❓ Asks Technical Q", use_container_width=True): log_obs("Participation", "asks technical question")
        if st.button("👂 Listening Intently", use_container_width=True): log_obs("Participation", "listening")
    with c3:
        if st.button("🛑 Voluntarily Leaves", use_container_width=True): log_obs("Exit", "voluntarily leaves", "DIRECT")
        if st.button("🔊 Noise Friction", use_container_width=True): log_obs("Friction", "environmental noise", "INFERRED")

    st.markdown("---")
    
    if st.button("⚡ ADAPT", type="primary", use_container_width=True):
        if not ai_client:
            st.error("AI client unavailable. Check GEMINI_API_KEY.")
        else:
            with st.spinner("Analyzing observations..."):
                recent_obs = db.query(Observation).filter(
                    Observation.interaction_id == st.session_state.current_interaction
                ).order_by(Observation.timestamp.desc()).limit(5).all()
                
                context = {
                    "phase": "Direct Experience",
                    "observations": [f"[{o.evidence_level}] {o.category}: {o.detail}" for o in recent_obs],
                    "objectives": event_record.primary_objective
                }
                
                try:
                    rec = get_adaptation(context, ai_client)
                    st.success(f"**DO THIS:** {rec.recommended_action}")
                    st.info(f"**BECAUSE:** {rec.rationale}")
                    st.caption(f"Confidence: **{rec.confidence}** | Watch for: *{rec.next_observation}*")
                except Exception as e:
                    st.error(f"AI Error: {e}")

    if st.button("Next Participant ➡️"):
        del st.session_state.current_interaction
        st.rerun()

# --- MODULE C & D: IMPACT & SURVEYS ---
elif page == "C: Impact & Surveys":
    st.title("Participant Surveys")
    interactions = db.query(Interaction).order_by(Interaction.timestamp_start.desc()).limit(10).all()
    if not interactions:
        st.warning("No interactions recorded yet. Log one in the Live Copilot.")
        st.stop()
        
    selected_int = st.selectbox("Select Recent Interaction ID", [i.id for i in interactions])
    
    tab1, tab2, tab3 = st.tabs(["Baseline (Pre)", "Immediate (Post)", "Delayed (Follow-up)"])
    
    with tab1:
        with st.form("baseline"):
            curiosity = st.slider("Pre-Curiosity (1-10)", 1, 10, 5)
            knowledge = st.slider("Pre-Knowledge (%)", 0, 100, 0)
            if st.form_submit_button("Save Baseline"):
                db.add(Survey(interaction_id=selected_int, timing="BASELINE", curiosity_score=curiosity, knowledge_score=knowledge))
                db.commit()
                st.success("Baseline saved.")
                
    with tab2:
        with st.form("immediate"):
            curiosity_post = st.slider("Post-Curiosity (1-10)", 1, 10, 8)
            knowledge_post = st.slider("Post-Knowledge (%)", 0, 100, 50)
            mem_text = st.text_area("What is the one thing you'll remember most?")
            if st.form_submit_button("Save Immediate Post"):
                db.add(Survey(interaction_id=selected_int, timing="IMMEDIATE", curiosity_score=curiosity_post, knowledge_score=knowledge_post, memorability_text=mem_text))
                db.commit()
                st.success("Post-survey saved.")
                
    with tab3:
        with st.form("delayed"):
            follow = st.radio("Did you look something up afterward?", ["Yes", "No"])
            if st.form_submit_button("Save Delayed Impact"):
                db.add(Survey(interaction_id=selected_int, timing="DELAYED_24H", follow_through=follow))
                db.commit()
                st.success("Delayed metric saved.")

# --- MODULE E: IMPACT OBSERVATORY ---
elif page == "E: Observatory":
    st.title("Impact Observatory")
    events = get_active_events()
    if events:
        event_dict = {e.name: e for e in events}
        active_event_name = st.selectbox("Analyze Event", list(event_dict.keys()))
        event_record = event_dict[active_event_name]
        
        metrics = calculate_impact_fingerprint(db, event_record.id)
        
        if metrics:
            st.markdown("### 📊 Deterministic Impact Fingerprint")
            st.caption("Calculated strictly via Python/Pandas.")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Participants", metrics["total_participants"])
            c2.metric("Curiosity Shift", f"{metrics['curiosity_change']:+}", f"{metrics['baseline_curiosity']:.1f} → {metrics['post_curiosity']:.1f}")
            c3.metric("Knowledge Shift", f"{metrics['knowledge_change']:+}%", f"{metrics['baseline_knowledge']:.1f}% → {metrics['post_knowledge']:.1f}%")
            c4.metric("Follow-Through", f"{metrics['follow_through_rate']:.1f}%")
            
            st.markdown("### 🧠 Qualitative Memory Anchors")
            st.caption("Extracted via Gemini from free-text surveys.")
            surveys = db.query(Survey).join(Interaction).filter(Interaction.event_id == event_record.id, Survey.timing == 'IMMEDIATE', Survey.memorability_text != "").all()
            
            if st.button("Synthesize Memory Themes"):
                if not ai_client:
                    st.error("AI client unavailable.")
                else:
                    with st.spinner("Classifying memory anchors..."):
                        for s in surveys:
                            if s.memorability_text and str(s.memorability_text).strip():
                                try:
                                    theme_data = extract_theme(s.memorability_text, ai_client)
                                    st.write(f"- **{theme_data.theme}** (Confidence: {theme_data.confidence})")
                                    st.caption(f"Evidence: '{s.memorability_text}'")
                                except Exception as e:
                                    st.write(f"- Failed to extract: {s.memorability_text}")
                            else:
                                st.write("- (Empty response skipped)")
        else:
            st.info("No survey data available for this event yet.")
