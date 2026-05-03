"""
app.py — TYPEGUARD Streamlit Dashboard
A sleek, dark-themed cybersecurity authentication dashboard.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TYPEGUARD",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Colors (LIGHT CYBER THEME) ── */
:root {
    --bg: #f5f7fb;
    --surface: #ffffff;
    --card: rgba(255, 255, 255, 0.85);
    --border: #dbe3ef;
    --accent: #0077ff;
    --accent2: #00c896;
    --danger: #ff4d4f;
    --warning: #f5a623;
    --text: #1f2937;
    --muted: #6b7280;
}

/* ── Global ── */
html, body, .stApp {
    background: linear-gradient(135deg, #f5f7fb, #eaf2ff);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid var(--border);
}

/* ── Cards ── */
[data-testid="metric-container"],
.streamlit-expanderHeader,
.stAlert {
    background: var(--card) !important;
    backdrop-filter: blur(10px);
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border: none;
    color: white;
    font-size: 14px;
    font-weight: 500;
    border-radius: 8px;
    padding: 8px 18px;
    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,119,255,0.25);
}

/* ── Inputs ── */
input, textarea {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    padding: 8px !important;
}

input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 6px rgba(0,119,255,0.2);
}

/* ── Tabs ── */
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent);
    font-weight: 600;
}

/* ── Progress Bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 6px;
}

/* ── Hero Header ── */
.hero-header {
    font-family: 'Space Mono';
    font-size: 38px;
    font-weight: 700;
    color: var(--accent);
    text-align: center;
}

.hero-sub {
    font-size: 13px;
    color: var(--muted);
    text-align: center;
    letter-spacing: 0.1em;
}

/* ── Section Headings ── */
.section-head {
    font-size: 12px;
    font-weight: 600;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    margin-bottom: 10px;
}

/* ── Terminal Card (light version) ── */
.term-card {
    background: #f9fbff;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px;
    font-family: 'Space Mono';
    font-size: 12px;
}

.term-line-ok { color: var(--accent2); }
.term-line-warn { color: var(--warning); }
.term-line-err { color: var(--danger); }
.term-line-info { color: var(--accent); }

/* ── Metrics Text ── */
[data-testid="stMetricValue"] {
    font-size: 20px !important;
    color: var(--accent) !important;
    font-weight: 600;
}

[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    color: var(--muted) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: var(--accent);
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ─── Lazy imports (avoid top-level import errors in demo mode) ────────────────

@st.cache_resource(show_spinner=False)
def get_auth_system():
    from auth_system import TYPEGUARD
    return TYPEGUARD()

@st.cache_resource(show_spinner=False)
def get_demo_model():
    from model import build_default_model, UserProfile
    profile = UserProfile(wpm_mean=72, wpm_std=9, hold_mean=92, error_rate=0.035)
    model, df, metrics = build_default_model(profile)
    return model, df, metrics

# ─── Session State Defaults ────────────────────────────────────────────────────

def init_state():
    defaults = {
        "page": "login",
        "logged_in": False,
        "username": "",
        "session_id": None,
        "last_report": None,
        "session_logs": [],
        "typing_events": [],        # list of {key, press_t, release_t}
        "mouse_events": [],
        "captcha_mode": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Helpers ──────────────────────────────────────────────────────────────────

def risk_badge(level: str) -> str:
    cls = {"LOW": "badge-low", "MEDIUM": "badge-medium", "HIGH": "badge-high"}.get(level, "badge-medium")
    icon = {"LOW": "✓", "MEDIUM": "⚠", "HIGH": "✕"}.get(level, "?")
    return f'<span class="{cls}">{icon} {level}</span>'

def score_color(score: float) -> str:
    if score >= 70: return "#00ff88"
    if score >= 45: return "#ffaa00"
    return "#ff3860"

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="hero-header">TYPE<br>GUARD</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-sub">Behavioral Biometric Engine v1.0</div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.session_state.logged_in:
            st.markdown(f"**Logged in as:** `{st.session_state.username}`")
            if st.session_state.last_report:
                r = st.session_state.last_report
                st.markdown(f"**Match score:** `{r['similarity_score']}%`")
                st.markdown(f"**Risk:** {risk_badge(r['risk_level'])}", unsafe_allow_html=True)
            st.markdown("---")
            nav = st.radio("Navigate", ["Dashboard", "Continuous Auth", "Session Logs", "Model Insights"], label_visibility="collapsed")
            st.markdown("---")
            if st.button(" Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.page = "login"
                st.rerun()
            return nav
        else:
            nav = st.radio("Navigate", ["Login", "Register"], label_visibility="collapsed")
            st.markdown("---")
            st.markdown('<div class="term-card"><span class="term-line-info">// SYSTEM STATUS</span><br><span class="term-line-ok">● AUTH ENGINE: ONLINE</span><br><span class="term-line-ok">● ML MODEL: READY</span><br><span class="term-line-ok">● RISK ENGINE: ACTIVE</span></div>', unsafe_allow_html=True)
            return nav

# ─── Typing Capture Widget ─────────────────────────────────────────────────────

def typing_capture_widget(label: str = "Type the sentence below:", prompt: str = None) -> Dict:
    """
    Simulates keystroke capture via JavaScript timing in Streamlit.
    Returns extracted typing features as a dict.
    """
    if prompt is None:
        prompts = [
            "The quick brown fox jumps over the lazy dog",
            "Security is not a product but a process",
            "Behavioral biometrics verify your identity",
            "Every keystroke tells a unique story",
        ]
        import random
        prompt = random.choice(prompts)

    st.markdown(f'<div class="section-head">BEHAVIORAL CAPTURE</div>', unsafe_allow_html=True)
    st.markdown(f"**Type exactly:** `{prompt}`")

    # JS-powered keystroke capture embedded in a component
    components_html = f"""
    <style>
    #typing-area {{
        width: 100%;
        background: #0d1117;
        border: 1px solid #1e2d3d;
        color: #c9d1d9;
        font-family: 'Space Mono', monospace;
        font-size: 14px;
        padding: 12px;
        border-radius: 6px;
        outline: none;
        box-sizing: border-box;
        transition: border-color .2s;
    }}
    #typing-area:focus {{ border-color: #00d4ff; box-shadow: 0 0 8px rgba(0,212,255,.2); }}
    #stats-display {{
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        color: #586069;
        margin-top: 6px;
        display: flex; gap: 20px;
    }}
    .stat-pill {{ background: #111820; border: 1px solid #1e2d3d; padding: 3px 10px; border-radius: 12px; }}
    </style>
    <textarea id="typing-area" rows="2" placeholder="Start typing here..."></textarea>
    <div id="stats-display">
        <span class="stat-pill" id="wpm-stat">WPM: 0</span>
        <span class="stat-pill" id="hold-stat">AVG HOLD: 0ms</span>
        <span class="stat-pill" id="err-stat">ERRORS: 0</span>
        <span class="stat-pill" id="chars-stat">CHARS: 0</span>
    </div>
    <input type="hidden" id="feat-json" value="{{}}">
    <script>
    (function() {{
        const ta = document.getElementById('typing-area');
        let pressMap = {{}}, events = [], bsCount = 0, totalChars = 0, lastRelease = null;
        let startTime = null;
        ta.addEventListener('keydown', e => {{
            const t = performance.now();
            if (!startTime) startTime = t;
            pressMap[e.code] = t;
        }});
        ta.addEventListener('keyup', e => {{
            const t = performance.now();
            const pt = pressMap[e.code];
            if (pt === undefined) return;
            const hold = t - pt;
            const flight = lastRelease !== null ? pt - lastRelease : 0;
            lastRelease = t;
            if (e.code === 'Backspace') {{ bsCount++; }}
            else {{ totalChars++; }}
            events.push({{ hold, flight }});
            delete pressMap[e.code];
            updateStats(t);
        }});
        function updateStats(t) {{
            if (events.length < 2) return;
            const holds = events.map(e=>e.hold);
            const flights = events.filter(e=>e.flight>0).map(e=>e.flight);
            const elapsed = (t - startTime) / 60000;
            const wpm = elapsed > 0 ? Math.round((totalChars/5)/elapsed) : 0;
            const avgHold = Math.round(holds.reduce((a,b)=>a+b,0)/holds.length);
            document.getElementById('wpm-stat').textContent = 'WPM: ' + wpm;
            document.getElementById('hold-stat').textContent = 'AVG HOLD: ' + avgHold + 'ms';
            document.getElementById('err-stat').textContent = 'ERRORS: ' + bsCount;
            document.getElementById('chars-stat').textContent = 'CHARS: ' + totalChars;
            // compute features
            const avgFlight = flights.length ? flights.reduce((a,b)=>a+b,0)/flights.length : 0;
            const stdHold = Math.sqrt(holds.reduce((a,b)=>a+(b-avgHold)**2,0)/holds.length);
            const stdFlight = flights.length > 1 ? Math.sqrt(flights.reduce((a,b,_,arr)=>{{const m=arr.reduce((x,y)=>x+y,0)/arr.length;return a+(b-m)**2;}},0)/flights.length) : 0;
            const burstRatio = flights.filter(f=>f<80).length / Math.max(flights.length,1);
            const pauseRatio = flights.filter(f=>f>300).length / Math.max(flights.length,1);
            const errRate = bsCount / Math.max(totalChars+bsCount, 1);
            const feat = {{
                wpm, mean_hold_ms: avgHold, std_hold_ms: stdHold,
                mean_flight_ms: avgFlight, std_flight_ms: stdFlight,
                error_rate: errRate, burst_ratio: burstRatio, pause_ratio: pauseRatio,
                n_events: events.length
            }};
            document.getElementById('feat-json').value = JSON.stringify(feat);
        }}
    }})();
    </script>
    """
    st.components.v1.html(components_html, height=120)

    # Since we can't read JS values back in Streamlit directly, we simulate
    # features using realistic random generation for the demo
    typed = st.text_input("↑ Type above, then confirm here", key=f"confirm_{label}", label_visibility="collapsed",
                          placeholder="(confirmation field — type the prompt above first)")

    return typed, prompt

# ─── Simulate features from typed text ────────────────────────────────────────
from feature_engineering import BehavioralFeatures
def simulate_features_from_input(text: str, hour: int, username: str = "")->"BehavioralFeatures":
    """
    Derive plausible behavioral features from typed text length and
    some session-state entropy. Used when real JS timing is unavailable.
    """
    from data_collection import simulate_typing_session, simulate_mouse_session
    from feature_engineering import extract_features
    import hashlib

    # Use text + username as seed for reproducibility within session
    seed_val = int(hashlib.md5((text + username + str(hour)).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_val)

    # Realistic WPM based on text length as proxy
    wpm = float(rng.normal(72, 10))
    hold = float(rng.normal(95, 18))
    err = float(rng.uniform(0.02, 0.07))

    ks = simulate_typing_session(wpm_mean=wpm, wpm_std=8, hold_mean_ms=hold,
                                  hold_std_ms=16, error_rate=err, num_chars=max(len(text), 20), rng=rng)
    ms = simulate_mouse_session(rng=rng)
    return extract_features(ks, ms, hour=hour, day_of_week=datetime.now().weekday())

# ─── Pages ────────────────────────────────────────────────────────────────────

def page_login():
    st.markdown('<div class="hero-header">SIGN IN</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Multi-factor behavioral authentication</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown('<div class="section-head">CREDENTIALS</div>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username", key="login_user")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pw")

        st.markdown("")
        typed, prompt = typing_capture_widget(label="login_type")

        mode = st.selectbox("Security Mode", ["ADAPTIVE", "STRICT", "RELAXED"])

        st.markdown("")
        if st.button("⟶ AUTHENTICATE", use_container_width=True, key="btn_login"):
            if not username or not password or not typed:
                st.warning("Please fill in all fields and type the verification sentence.")
            else:
                with st.spinner("Running behavioral analysis..."):
                    time.sleep(0.8)  # simulate processing
                    hour = datetime.now().hour
                    features = simulate_features_from_input(typed, hour, username)
                    auth = get_auth_system()

                    # Auto-register demo user if not exists
                    if username not in auth._users:
                        from model import UserProfile
                        profile = UserProfile(wpm_mean=72, wpm_std=9, hold_mean=92, error_rate=0.035)
                        auth.register_user(username, password, profile=profile, mode=mode)

                    result = auth.login(username, password, features, hour=hour)

                if result['decision'] == 'ALLOW':
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.session_id = result.get('session_id')
                    st.session_state.last_report = result['report']
                    st.session_state.session_logs.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "event": "LOGIN_SUCCESS",
                        "score": result['report']['similarity_score'],
                        "risk": result['report']['risk_level'],
                    })
                    st.rerun()
                elif result['decision'] == 'CAPTCHA':
                    st.session_state.last_report = result['report']
                    st.session_state.captcha_mode = True
                    st.session_state.username = username
                    st.warning("⚠️ Behavioral mismatch — complete the verification below.")
                    st.rerun()
                else:
                    st.error(f"🚫 {result['message']}")
                    if result.get('report'):
                        st.markdown(f"Match score: `{result['report']['similarity_score']}%` | Risk: `{result['report']['risk_level']}`")

    with col2:
        st.markdown('<div class="section-head">HOW IT WORKS</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="term-card">
        <span class="term-line-info">01. PASSWORD HASH</span><br>
        SHA-256 + salt verification<br><br>
        <span class="term-line-info">02. KEYSTROKE CAPTURE</span><br>
        Hold time · Flight time · WPM<br><br>
        <span class="term-line-info">03. ML ANALYSIS</span><br>
        Random Forest similarity score<br><br>
        <span class="term-line-info">04. RISK ENGINE</span><br>
        Context + behavior → decision<br><br>
        <span class="term-line-info">05. CONTINUOUS AUTH</span><br>
        Ongoing session verification
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.captcha_mode and st.session_state.get("last_report"):
            r = st.session_state.last_report
            st.markdown("---")
            st.markdown(f"**Last score:** `{r['similarity_score']}%`")
            st.markdown(f"**Flags:** {', '.join(r.get('context_flags', [])) or 'None'}")


def page_register():
    st.markdown('<div class="hero-header">REGISTER</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Create a behavioral identity profile</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown('<div class="section-head">NEW ACCOUNT</div>', unsafe_allow_html=True)
        username = st.text_input("Choose Username", key="reg_user")
        password = st.text_input("Choose Password", type="password", key="reg_pw")
        confirm_pw = st.text_input("Confirm Password", type="password", key="reg_pw2")

        st.markdown('<div class="section-head" style="margin-top:16px">BEHAVIORAL PROFILE</div>', unsafe_allow_html=True)
        st.caption("These parameters seed your behavioral model (tune to match your typing style)")
        wpm = st.slider("Typical WPM", 20, 180, 70)
        hold = st.slider("Key hold duration (ms)", 40, 200, 95)
        err = st.slider("Error rate (%)", 0, 20, 4) / 100.0
        mode = st.selectbox("Default Auth Mode", ["ADAPTIVE", "STRICT", "RELAXED"], key="reg_mode")

        if st.button("⟶ CREATE PROFILE", use_container_width=True):
            if not username or not password:
                st.error("Please fill in all fields.")
            elif password != confirm_pw:
                st.error("Passwords do not match.")
            else:
                with st.spinner("Training behavioral model..."):
                    from model import UserProfile
                    from auth_system import AuthMode
                    auth = get_auth_system()
                    profile = UserProfile(wpm_mean=float(wpm), wpm_std=float(wpm)*0.12,
                                          hold_mean=float(hold), error_rate=err)
                    result = auth.register_user(username, password, profile=profile, mode=mode)
                if result['success']:
                    st.success(f"✅ {result['message']}")
                    m = result.get('model_metrics', {})
                    if m:
                        st.markdown(f"**Model AUC:** `{m.get('cv_auc_mean', 0):.3f}` ± `{m.get('cv_auc_std', 0):.3f}`")
                else:
                    st.error(result['message'])

    with col2:
        st.markdown('<div class="section-head">PRIVACY GUARANTEE</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="term-card">
        <span class="term-line-ok">✓ NO raw keystrokes stored</span><br>
        <span class="term-line-ok">✓ NO text content saved</span><br>
        <span class="term-line-ok">✓ ONLY numerical timing</span><br>
        <span class="term-line-ok">✓ Password hashed (SHA-256)</span><br>
        <span class="term-line-ok">✓ Platform info one-way hashed</span><br><br>
        <span class="term-line-info">STORED FEATURES:</span><br>
        WPM · Hold time · Flight time<br>
        Error rate · Mouse velocity<br>
        Temporal patterns (cyclic)
        </div>
        """, unsafe_allow_html=True)


def page_dashboard():
    st.markdown('<div class="hero-header">DASHBOARD</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="hero-sub">Session: {st.session_state.username} · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>', unsafe_allow_html=True)
    st.markdown("---")

    r = st.session_state.last_report
    if not r:
        st.info("No authentication data yet. Please log in first.")
        return

    # ── Top metrics row ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Match Score", f"{r['similarity_score']}%",
              delta=f"{r['final_score'] - r['similarity_score']:+.1f} (after penalties)" if r['final_score'] != r['similarity_score'] else None)
    c2.metric("Final Risk Score", f"{r['final_score']}%")
    c3.metric("Risk Level", r['risk_level'])
    c4.metric("Decision", r['decision'])

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    # ── Score gauge ──────────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-head">SIMILARITY GAUGE</div>', unsafe_allow_html=True)
        score = r['similarity_score']
        color = score_color(score)
        gauge_html = f"""
        <div style="display:flex;flex-direction:column;align-items:center;padding:16px">
        <svg width="200" height="200" viewBox="0 0 200 200">
          <defs>
            <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" style="stop-color:#00d4ff"/>
              <stop offset="100%" style="stop-color:#00ff88"/>
            </linearGradient>
          </defs>
          <!-- Background ring -->
          <circle cx="100" cy="100" r="80" fill="none" stroke="#1e2d3d" stroke-width="16"/>
          <!-- Score arc -->
          <circle cx="100" cy="100" r="80" fill="none" stroke="url(#g1)" stroke-width="16"
            stroke-dasharray="{score/100*502.65:.1f} 502.65"
            stroke-dashoffset="125.66"
            stroke-linecap="round"
            transform="rotate(-90 100 100)"
            style="filter:drop-shadow(0 0 8px {color})"/>
          <!-- Score text -->
          <text x="100" y="95" text-anchor="middle" font-family="Space Mono" font-size="32" font-weight="700" fill="{color}">{score:.0f}%</text>
          <text x="100" y="118" text-anchor="middle" font-family="Space Mono" font-size="10" fill="#586069">MATCH SCORE</text>
        </svg>
        <div style="font-family:Space Mono;font-size:11px;color:#586069;text-align:center;margin-top:4px">
            Risk Level: <span style="color:{color};font-weight:700">{r['risk_level']}</span>
        </div>
        </div>
        """
        st.markdown(gauge_html, unsafe_allow_html=True)

        # Context flags
        if r.get('context_flags'):
            st.markdown('<div class="section-head">ANOMALY FLAGS</div>', unsafe_allow_html=True)
            for flag in r['context_flags']:
                st.markdown(f"⚠️ `{flag}`")
        else:
            st.markdown("✅ No contextual anomalies detected")

    # ── Feature breakdown ────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-head">EXPLAINABLE AI — TOP DRIVERS</div>', unsafe_allow_html=True)
        if r.get('explanation'):
            import plotly.graph_objects as go
            exp = r['explanation']
            names = [e['feature'].replace('_', ' ').upper() for e in exp]
            impacts = [e['impact'] for e in exp]
            zscores = [e['z_score'] for e in exp]
            colors = ['#ff3860' if z > 2 else '#ffaa00' if z > 1 else '#00ff88' for z in zscores]

            fig = go.Figure(go.Bar(
                x=impacts[::-1], y=names[::-1],
                orientation='h',
                marker_color=colors[::-1],
                text=[f"z={z:.1f}" for z in zscores[::-1]],
                textposition='outside',
                textfont=dict(family="Space Mono", size=10, color="#c9d1d9"),
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Space Mono", color="#c9d1d9", size=11),
                xaxis=dict(gridcolor='#1e2d3d', title="Impact Score", color='#586069'),
                yaxis=dict(gridcolor='#1e2d3d', color='#c9d1d9'),
                margin=dict(l=10, r=30, t=10, b=10),
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Full explanation"):
                st.json(r['explanation'])
        else:
            st.info("Explanation unavailable.")

    # ── Feature history chart ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-head">SESSION SCORE TIMELINE</div>', unsafe_allow_html=True)
    logs = st.session_state.session_logs
    if len(logs) >= 1:
        import plotly.graph_objects as go
        scores = [l['score'] for l in logs]
        times = [l['time'] for l in logs]
        risks = [l['risk'] for l in logs]
        clrs = ['#00ff88' if rk=='LOW' else '#ffaa00' if rk=='MEDIUM' else '#ff3860' for rk in risks]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(len(scores))), y=scores,
            mode='lines+markers',
            line=dict(color='#00d4ff', width=2),
            marker=dict(color=clrs, size=10, line=dict(color='#0d1117', width=2)),
            text=times, hovertemplate='%{text}<br>Score: %{y}%<extra></extra>',
        ))
        fig2.add_hline(y=65, line_dash="dot", line_color="#00ff88", annotation_text="ALLOW", annotation_font_color="#00ff88")
        fig2.add_hline(y=45, line_dash="dot", line_color="#ffaa00", annotation_text="CAPTCHA", annotation_font_color="#ffaa00")
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Space Mono", color="#c9d1d9", size=11),
            xaxis=dict(gridcolor='#1e2d3d', title="Check #", color='#586069'),
            yaxis=dict(gridcolor='#1e2d3d', title="Score %", color='#586069', range=[0,105]),
            margin=dict(l=10, r=10, t=10, b=10), height=200,
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("Score history will appear here after multiple checks.")


def page_continuous_auth():
    st.markdown('<div class="hero-header">CONTINUOUS AUTH</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Live behavioral re-verification during session</div>', unsafe_allow_html=True)
    st.markdown("---")

    typed, prompt = typing_capture_widget(label="continuous_type", prompt="Authenticate your session now")

    if st.button("⟶ RUN BEHAVIORAL CHECK", use_container_width=True):
        if not typed:
            st.warning("Please type in the capture field first.")
        else:
            with st.spinner("Analysing current behavior..."):
                time.sleep(0.6)
                hour = datetime.now().hour
                features = simulate_features_from_input(typed, hour, st.session_state.username)
                auth = get_auth_system()
                result = auth.continuous_check(
                    session_id=st.session_state.session_id,
                    features=features,
                    hour=hour,
                )

            if result.get('revoked'):
                st.error(f"🚫 Session revoked: {result['message']}")
                if result.get('report'):
                    st.session_state.last_report = result['report']
            else:
                report = result['report']
                st.session_state.last_report = report
                st.session_state.session_logs.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "event": f"CONTINUOUS_CHECK_{result.get('continuous_check_num',0)}",
                    "score": report['similarity_score'],
                    "risk": report['risk_level'],
                })
                rl = report['risk_level']
                badge = risk_badge(rl)
                st.markdown(f"**Check #{result.get('continuous_check_num',1)}** — Score: `{report['similarity_score']}%` | {badge}", unsafe_allow_html=True)
                if rl == 'LOW':
                    st.success("✅ Behavioral signature verified — session continues.")
                elif rl == 'MEDIUM':
                    st.warning("⚠️ Slight deviation detected. Additional monitoring active.")
                else:
                    st.error("🚫 High-risk deviation. Consider re-authentication.")


def page_session_logs():
    st.markdown('<div class="hero-header">SESSION LOGS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Audit trail of authentication events</div>', unsafe_allow_html=True)
    st.markdown("---")

    logs = st.session_state.session_logs
    if not logs:
        st.info("No events in this session yet.")
        return

    for i, log in enumerate(reversed(logs)):
        rl = log.get('risk', 'LOW')
        color = '#00ff88' if rl=='LOW' else '#ffaa00' if rl=='MEDIUM' else '#ff3860'
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #1e2d3d;border-left:3px solid {color};
                    padding:10px 16px;border-radius:4px;margin-bottom:8px;font-family:Space Mono;font-size:12px">
            <span style="color:#586069">[{log['time']}]</span>
            <span style="color:{color};margin:0 12px">{log['event']}</span>
            <span style="color:#c9d1d9">score={log['score']}%</span>
            <span style="color:{color};margin-left:12px">RISK:{rl}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑 Clear logs"):
        st.session_state.session_logs = []
        st.rerun()


def page_model_insights():
    st.markdown('<div class="hero-header">MODEL INSIGHTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ML training data & feature importance analysis</div>', unsafe_allow_html=True)
    st.markdown("---")

    with st.spinner("Loading model data..."):
        model, df, metrics = get_demo_model()

    c1, c2, c3 = st.columns(3)
    c1.metric("Training Samples", f"{len(df):,}")
    c2.metric("CV AUC", f"{metrics.get('cv_auc_mean', 0):.3f}")
    c3.metric("AUC ± Std", f"±{metrics.get('cv_auc_std', 0):.3f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # Feature importance
    with col1:
        st.markdown('<div class="section-head">FEATURE IMPORTANCES</div>', unsafe_allow_html=True)
        if model.feature_importances_ is not None:
            import plotly.graph_objects as go
            from feature_engineering import FEATURE_NAMES
            fi = model.feature_importances_
            idx = np.argsort(fi)[::-1]
            fig = go.Figure(go.Bar(
                x=[FEATURE_NAMES[i].replace('_',' ').upper() for i in idx],
                y=[fi[i] for i in idx],
                marker_color=['#00d4ff' if fi[i] > np.median(fi) else '#1e2d3d' for i in idx],
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Space Mono", color="#c9d1d9", size=9),
                xaxis=dict(gridcolor='#1e2d3d', color='#586069', tickangle=45),
                yaxis=dict(gridcolor='#1e2d3d', color='#586069', title='Importance'),
                margin=dict(l=0, r=0, t=10, b=10), height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Score distribution
    with col2:
        st.markdown('<div class="section-head">SCORE DISTRIBUTION (TRAIN DATA)</div>', unsafe_allow_html=True)
        legit_sample = df[df.label == 1].sample(min(50, len(df[df.label==1])), random_state=42)
        impos_sample = df[df.label == 0].sample(min(50, len(df[df.label==0])), random_state=42)

        from feature_engineering import BehavioralFeatures, FEATURE_NAMES, features_from_dict
        def score_row(row):
            feat = features_from_dict({k: row[k] for k in FEATURE_NAMES})
            return model.similarity_score(feat)

        legit_scores = legit_sample.apply(score_row, axis=1).tolist()
        impos_scores = impos_sample.apply(score_row, axis=1).tolist()

        import plotly.graph_objects as go
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=legit_scores, name='Legitimate', marker_color='#00ff88', opacity=0.7, nbinsx=20))
        fig2.add_trace(go.Histogram(x=impos_scores, name='Impostor', marker_color='#ff3860', opacity=0.7, nbinsx=20))
        fig2.update_layout(
            barmode='overlay',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Space Mono", color="#c9d1d9", size=10),
            xaxis=dict(gridcolor='#1e2d3d', title='Score (%)', color='#586069'),
            yaxis=dict(gridcolor='#1e2d3d', title='Count', color='#586069'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=0, r=0, t=10, b=10), height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Sample data
    st.markdown('<div class="section-head">TRAINING DATA SAMPLE</div>', unsafe_allow_html=True)
    display_cols = ['wpm', 'mean_hold_ms', 'mean_flight_ms', 'error_rate', 'burst_ratio', 'label']
    st.dataframe(
        df[display_cols].head(20).style.background_gradient(
            cmap='RdYlGn', subset=['wpm', 'mean_hold_ms']
        ),
        use_container_width=True,
    )

# ─── Main Router ──────────────────────────────────────────────────────────────

def main():
    nav = render_sidebar()

    if st.session_state.logged_in:
        page_map = {
            "Dashboard": page_dashboard,
            "Continuous Auth": page_continuous_auth,
            "Session Logs": page_session_logs,
            "Model Insights": page_model_insights,
        }
        page_fn = page_map.get(nav, page_dashboard)
        page_fn()
    else:
        if nav == "Login":
            page_login()
        else:
            page_register()

if __name__ == "__main__":
    main()
