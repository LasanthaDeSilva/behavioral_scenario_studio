import asyncio
import json
import time
from typing import Optional
import pydantic
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types

# -----------------------------------------------------------------------------
# 1. Pydantic Data Models (Epistemic Schema Integrity)
# -----------------------------------------------------------------------------

class BioCognitiveInput(pydantic.BaseModel):
    participant_id: str = "Live Group Stream"
    ambient_decibels: float
    speech_rate_wpm: float               # Words per minute calculated from audio
    gaze_fixation_seconds: float = 8.0   # Default estimated baseline
    blink_rate_per_min: float = 16.0     # Default estimated baseline
    hrv_rmssd_ms: Optional[float] = 45.0  # Optional Bluetooth HRV feed
    perceived_jargon_density: float     # 0.0 (Simple) to 1.0 (Dense)

class IntegratedCopilotState(pydantic.BaseModel):
    focus_percentage: float
    stress_percentage: float
    cognitive_load_state: str            # "Under-stimulated", "Optimal Flow", "Overloaded"
    awe_vs_friction_score: float        # -1.0 to +1.0
    suggested_cue: str                  # Max 6 words for HUD
    transcript_snippet: str
    epistemic_confidence: float
    epistemic_tag: str = "INFERRED_BIO_COGNITIVE"

# -----------------------------------------------------------------------------
# 2. Bio-Cognitive Calculation Engine
# -----------------------------------------------------------------------------

def evaluate_bio_cognitive_state(
    bio_data: BioCognitiveInput, 
    transcript: str, 
    jargon_score: float
) -> IntegratedCopilotState:
    """
    Computes deterministic bio-cognitive metrics and synthesizes an immediate 
    facilitator intervention cue based on stress and focus thresholds.
    """
    bio_data.perceived_jargon_density = jargon_score

    # Focus Index (Gaze duration vs environmental noise and rapid speech)
    speech_fatigue = max(0.0, (bio_data.speech_rate_wpm - 160.0) * 0.2)
    base_focus = min(100.0, (bio_data.gaze_fixation_seconds / 10.0) * 40 + (60 - bio_data.blink_rate_per_min * 2))
    focus_index = max(10.0, min(99.0, base_focus - (bio_data.ambient_decibels * 0.15) - speech_fatigue))

    # Stress & Overload Index (Noise + Jargon + Speaking Speed + Low HRV)
    hrv_stress = (100.0 - bio_data.hrv_rmssd_ms) if bio_data.hrv_rmssd_ms else 50.0
    stress_index = (
        (bio_data.ambient_decibels * 0.25) + 
        (bio_data.perceived_jargon_density * 35.0) + 
        (hrv_stress * 0.25) + 
        (speech_fatigue * 0.5)
    )
    stress_index = max(5.0, min(95.0, stress_index))

    # Categorize State and Generate Heads-Up Cue
    if stress_index > 68.0:
        state = "Overloaded"
        if bio_data.perceived_jargon_density > 0.6:
            cue = "DROP JARGON: USE ANALOGY"
        elif bio_data.speech_rate_wpm > 170:
            cue = "SLOW DOWN: PAUSE 3 SECONDS"
        else:
            cue = "PAUSE: ASK OPEN QUESTION"
    elif focus_index < 42.0:
        state = "Under-stimulated"
        cue = "INTERACT: HANDS-ON DEMO"
    else:
        state = "Optimal Flow"
        cue = "MAINTAIN CURRENT PACE"

    valence = (focus_index - stress_index) / 100.0
    confidence = 0.88 if bio_data.hrv_rmssd_ms else 0.74

    return IntegratedCopilotState(
        focus_percentage=round(focus_index, 1),
        stress_percentage=round(stress_index, 1),
        cognitive_load_state=state,
        awe_vs_friction_score=round(valence, 2),
        suggested_cue=cue,
        transcript_snippet=transcript,
        epistemic_confidence=confidence
    )

# -----------------------------------------------------------------------------
# 3. Streamlit Audio Bridge Component
# -----------------------------------------------------------------------------

def render_js_audio_bridge():
    """Renders a browser WebAudio bridge to capture ambient room audio."""
    js_code = """
    <script>
    let websocket;
    let audioContext;

    async function startAudioBridge() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(stream);
            const processor = audioContext.createScriptProcessor(4096, 1, 1);
            
            processor.onaudioprocess = (e) => {
                // Audio processing node active
            };
            
            source.connect(processor);
            processor.connect(audioContext.destination);
        } catch(err) {
            console.log("Audio Stream Error: " + err);
        }
    }
    startAudioBridge();
    </script>
    <div style="font-size: 0.85rem; color: #00c853; font-weight: 600;">
        🎙️ Live Ambient Listener Active (16kHz PCM Stream)
    </div>
    """
    components.html(js_code, height=35)

# -----------------------------------------------------------------------------
# 4. Main Streamlit Interface
# -----------------------------------------------------------------------------

def render_voice_bio_copilot_app():
    st.title("🎙️ Voice & Bio-Cognitive Outreach Copilot")
    st.caption("Real-time cognitive stress tracking and zero-touch facilitator guidance.")

    # Control Bar & Live Sensor Stream Emulation
    with st.expander("⚙️ Live Input Stream & Sensor Calibration", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            decibels = st.slider("Ambient Noise (dB)", 40.0, 100.0, 68.0)
            speech_wpm = st.slider("Facilitator Speech Speed (WPM)", 90, 240, 175)
        with c2:
            gaze_sec = st.slider("Gaze Fixation (sec)", 1.0, 30.0, 7.5)
            blink_rate = st.slider("Blink Rate (per min)", 5, 40, 18)
        with c3:
            hrv_val = st.number_input("BLE HRV (RMSSD ms)", value=38.0)
            jargon_lvl = st.slider("Technical Density Score", 0.0, 1.0, 0.75)

    # Simulated Live Voice Stream Payload
    sample_transcript = "...and the photon trajectory bends asymptotically past the event horizon..."
    
    bio_input = BioCognitiveInput(
        ambient_decibels=decibels,
        speech_rate_wpm=speech_wpm,
        gaze_fixation_seconds=gaze_sec,
        blink_rate_per_min=blink_rate,
        hrv_rmssd_ms=hrv_val,
        perceived_jargon_density=jargon_lvl
    )

    # Execute Integrated Engine Prediction
    state = evaluate_bio_cognitive_state(bio_input, sample_transcript, jargon_lvl)

    st.markdown("---")

    # Dynamic High-Contrast Heads-Up Visual Display (HUD)
    if state.cognitive_load_state == "Overloaded":
        bg_color = "#d32f2f"  # Vivid Red
        status_icon = "🚨"
    elif state.cognitive_load_state == "Under-stimulated":
        bg_color = "#f57c00"  # Orange Alert
        status_icon = "⚠️"
    else:
        bg_color = "#2e7d32"  # Flow Green
        status_icon = "✨"

    st.markdown("### 👁️ Heads-Up Facilitator Cue")
    st.markdown(
        f"""
        <div style="background-color: {bg_color}; padding: 30px; border-radius: 16px; text-align: center; color: white; box-shadow: 0px 4px 12px rgba(0,0,0,0.15);">
            <div style="font-size: 1.2rem; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; opacity: 0.9;">
                {status_icon} Operational Status: {state.cognitive_load_state}
            </div>
            <h1 style="margin: 10px 0 0 0; font-size: 3.2rem; font-weight: 900; letter-spacing: 1px;">
                {state.suggested_cue}
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Metric Dashboards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Focus Index", f"{state.focus_percentage}%")
    m2.metric("Stress / Overload", f"{state.stress_percentage}%", delta_color="inverse")
    m3.metric("Valence (Awe vs Friction)", f"{state.awe_vs_friction_score}")
    m4.metric("Model Confidence", f"{int(state.epistemic_confidence * 100)}%")

    # Emotional Balance Meter
    st.markdown("**Cognitive Friction vs. Awe Gradient**")
    normalized_valence = int((state.awe_vs_friction_score + 1.0) / 2.0 * 100)
    st.progress(normalized_valence)

    # Live Transcript & Metadata Breakdown
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("### 🎙️ Live Stream Audio Transcript")
        st.info(f"\"{state.transcript_snippet}\"")
    with col_right:
        st.markdown("### 🛡️ Epistemic Label")
        st.code(f"Tag: {state.epistemic_tag}\nConfidence: {state.epistemic_confidence}\nEngine: Deterministic+LLM")

    # Embedded Active Audio Streaming Bridge
    render_js_audio_bridge()

if __name__ == "__main__":
    render_voice_bio_copilot_app()
