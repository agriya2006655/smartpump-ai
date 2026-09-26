"""
app.py
------
SmartPump AI dashboard. Run with:
    streamlit run app.py

Needs pump_data.csv, pump_model.pkl, and pump_hero.jpg in the same folder
(pump_data.csv / pump_model.pkl are created by generate_data.py and
train_model.py; pump_hero.jpg is the hero banner image).
"""

import io
import os
import base64
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="SmartPump AI", page_icon="🔧", layout="wide")


def render_fig(fig):
    """Render a matplotlib figure as a crisp transparent PNG (avoids the
    st.pyplot savefig-kwargs deprecation warning)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def get_base64_image(path):
    """Read a local image file and return it as a base64 string, or None
    if the file isn't there yet (so the app never crashes on a missing image)."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


FEATURES = [
    "Temperature_C", "Vibration_mm_s", "Inlet_Pressure_bar",
    "Outlet_Pressure_bar", "Flow_Rate_Lmin", "RPM", "Power_kW",
    "Operating_Hours"
]

FEATURE_LABELS = {
    "Temperature_C": "Temperature (°C)",
    "Vibration_mm_s": "Vibration (mm/s)",
    "Inlet_Pressure_bar": "Inlet Pressure (bar)",
    "Outlet_Pressure_bar": "Outlet Pressure (bar)",
    "Flow_Rate_Lmin": "Flow Rate (L/min)",
    "RPM": "RPM",
    "Power_kW": "Power (kW)",
    "Operating_Hours": "Operating Hours",
}

# ---------- Custom CSS: hero, glow effects, chips, credit bar ----------
st.markdown("""
<style>

/* ---------- Hero banner ---------- */
.hero-section {
    background-size: cover;
    background-position: center;
    border-radius: 18px;
    border: 1px solid #00C2CB55;
    min-height: 300px;
    display: flex;
    align-items: flex-end;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 0 45px rgba(0,194,203,0.25), inset 0 0 60px rgba(0,0,0,0.35);
    animation: heroPulse 3.5s ease-in-out infinite;
}
@keyframes heroPulse {
    0%, 100% { box-shadow: 0 0 30px rgba(0,194,203,0.20), inset 0 0 60px rgba(0,0,0,0.35); }
    50%      { box-shadow: 0 0 55px rgba(0,194,203,0.45), inset 0 0 60px rgba(0,0,0,0.35); }
}
.hero-content { max-width: 620px; }
.hero-badges { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.9rem; }
.glass-badge {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(0,194,203,0.5);
    color: #E5F9FA;
    padding: 0.32rem 0.85rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.hero-title {
    font-size: 2.7rem;
    margin: 0;
    color: #FAFAFA;
    animation: titleGlow 2.4s ease-in-out infinite;
}
@keyframes titleGlow {
    0%, 100% { text-shadow: 0 0 12px rgba(0,194,203,0.75), 0 0 32px rgba(0,194,203,0.35); }
    50%      { text-shadow: 0 0 26px rgba(0,194,203,1), 0 0 60px rgba(0,194,203,0.65), 0 0 90px rgba(46,230,140,0.25); }
}
.hero-tag { color: #C7CED6; font-size: 0.98rem; margin-top: 0.5rem; line-height: 1.5; }

/* ---------- Metric cards ---------- */
div[data-testid="stMetric"] {
    background-color: #1C2128;
    border: 1px solid #2D333B;
    border-radius: 10px;
    padding: 1rem;
    transition: all 0.25s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: #00C2CB;
    box-shadow: 0 0 20px rgba(0,194,203,0.45);
    transform: translateY(-2px);
}

/* ---------- Status pill ---------- */
.status-pill {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
    animation: pulse 1.8s infinite;
}
@keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(0,194,203,0.45); }
    70%  { box-shadow: 0 0 0 14px rgba(0,194,203,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,194,203,0); }
}

/* ---------- Flow diagram animation ---------- */
.flow-line { stroke-dasharray: 6,4; animation: flow 0.9s linear infinite; }
@keyframes flow { to { stroke-dashoffset: -20; } }
.pump-box { animation: glow 2s ease-in-out infinite; }
@keyframes glow {
    0%, 100% { filter: drop-shadow(0 0 3px #00C2CB); }
    50%      { filter: drop-shadow(0 0 14px #00C2CB); }
}

/* ---------- Bottom credit bar (shown under every tab) ---------- */
.credit-wrap { text-align: center; margin: 2rem 0 0.5rem 0; }
.credit-bar {
    display: inline-block;
    padding: 0.5rem 1.3rem;
    background: #05070A;
    border: 1px solid #1F2937;
    border-radius: 8px;
    font-family: "Courier New", monospace;
    font-size: 0.78rem;
    color: #9CA3AF;
    letter-spacing: 0.3px;
}
.credit-bar b { color: #00C2CB; }
/* ---------- Hero Stats ---------- */

.hero-mini-stats {
    display: flex;
    gap: 32px;
    margin-top: 24px;
}

.hero-stat {
    display: flex;
    align-items: baseline;
    gap: 6px;
}

.hero-stat strong {
    font-size: 18px;
    font-weight: 700;
    color: #ffffff;
}

.hero-stat span {
    font-size: 15px;
    color: #c8cbd3;
}
/* ---------- Interactive Tabs ---------- */

.stTabs [data-baseweb="tab"] {
    padding: 10px 18px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

/* Hover */
.stTabs [data-baseweb="tab"]:hover {
    color: #00E5EF !important;
    background: rgba(0, 194, 203, 0.14) !important;
    box-shadow: 0 0 18px rgba(0, 194, 203, 0.30) !important;
}

/* Selected tab */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #00E5EF !important;
    background: rgba(0, 194, 203, 0.10) !important;
    box-shadow: 0 0 20px rgba(0, 194, 203, 0.22) !important;
}
/* ---------- Pump Visual Glow ---------- */

div[data-testid="stImage"] img {
    border-radius: 12px;

    box-shadow:
        0 0 18px rgba(0, 194, 203, 0.20),
        0 0 40px rgba(0, 194, 203, 0.10);

    transition:
        transform 0.35s ease,
        box-shadow 0.35s ease;

    animation: pumpGlow 4s ease-in-out infinite;
}

div[data-testid="stImage"] img:hover {
    transform: scale(1.012);

    box-shadow:
        0 0 25px rgba(0, 194, 203, 0.45),
        0 0 55px rgba(0, 194, 203, 0.22),
        0 0 90px rgba(0, 194, 203, 0.10);

    animation: none;
}

@keyframes pumpGlow {
    0%, 100% {
        box-shadow:
            0 0 18px rgba(0, 194, 203, 0.18),
            0 0 40px rgba(0, 194, 203, 0.08);
    }

    50% {
        box-shadow:
            0 0 24px rgba(0, 194, 203, 0.28),
            0 0 50px rgba(0, 194, 203, 0.14);
    }
}

</style>
""", unsafe_allow_html=True)

STATUS_COLORS = {
    "Normal": ("#1DB954", "#0E2A17"),
    "Warning": ("#F5A623", "#2A210E"),
    "Failure Risk": ("#E5484D", "#2A0E0E"),
}
STATUS_EMOJI = {"Normal": "🟢", "Warning": "🟡", "Failure Risk": "🔴"}


@st.cache_resource
def load_model():
    return joblib.load("pump_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("pump_data.csv")


model = load_model()
df = load_data()

# ---------- Hero banner (pump render + title + info badges) ----------
pump_img_b64 = get_base64_image("pump_hero.jpg")

if pump_img_b64:

    with st.container(border=True):

        col1, col2 = st.columns([0.38, 0.62], gap="large")

        with col1:

            st.markdown(
                '<div class="hero-badges">'
                '<span class="glass-badge">🟢 Live Monitoring</span>'
                '<span class="glass-badge">🟢 Random Forest AI</span>'
                '<span class="glass-badge">🟢 8 Sensors Tracked</span>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<h1 class="hero-title">⚙ SmartPump AI</h1>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<p class="hero-tag">'
                'AI-Assisted Performance Monitoring & Predictive Maintenance'
                '</p>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="hero-mini-stats">'
                '<div class="hero-stat">'
                '<strong>1200</strong>'
                ' <span>Sensor Readings</span>'
                '</div>'
                '<div class="hero-stat">'
                '<strong>200</strong>'
                ' <span>Random Forest Trees</span>'
                '</div>'
                '<div class="hero-stat">'
                '<strong>3</strong>'
                ' <span>Health Conditions</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div style="margin-top:18px;">'
                '<a href="https://github.com/agriya2006655/smartpump-ai" '
                'target="_blank" '
                'style="color:#00C2CB; text-decoration:none; '
                'font-weight:600; font-size:14px;">'
                '> View Project on GitHub'
                '</a>'
                '</div>',
                unsafe_allow_html=True
            )

        with col2:
            st.image("pump_hero.jpg", use_container_width=True)

else:
    st.info("Add a file named **pump_hero.jpg** next to app.py.")

tab1, tab2, tab3 = st.tabs(
    ["Performance Monitoring", "AI Diagnosis", "What-If Simulator"]
)
# ---------- TAB 1: Performance graphs over time ----------
with tab1:
    st.subheader("Process flow")
    st.markdown("""
    <div style="background:#1C2128;border:1px solid #2D333B;border-radius:12px;
    padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
    <svg viewBox="0 0 900 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;display:block;margin:0 auto;">
      <rect x="10" y="45" width="130" height="45" rx="6" fill="#0E4B4F" stroke="#00C2CB" stroke-width="1.5"/>
      <text x="75" y="72" fill="#FAFAFA" font-size="14" text-anchor="middle" font-family="sans-serif">Condenser</text>

      <line class="flow-line" x1="145" y1="67" x2="215" y2="67" stroke="#00C2CB" stroke-width="2"/>
      <polygon points="215,61 227,67 215,73" fill="#00C2CB"/>
      <text x="185" y="55" fill="#9CA3AF" font-size="11" text-anchor="middle" font-family="sans-serif">Low P</text>

      <rect class="pump-box" x="230" y="35" width="150" height="65" rx="6" fill="#0E4B4F" stroke="#00C2CB" stroke-width="2"/>
      <text x="305" y="60" fill="#00C2CB" font-size="14" text-anchor="middle" font-family="sans-serif" font-weight="bold">FEEDWATER</text>
      <text x="305" y="80" fill="#00C2CB" font-size="14" text-anchor="middle" font-family="sans-serif" font-weight="bold">PUMP</text>

      <line class="flow-line" x1="385" y1="67" x2="455" y2="67" stroke="#00C2CB" stroke-width="2"/>
      <polygon points="455,61 467,67 455,73" fill="#00C2CB"/>
      <text x="420" y="55" fill="#9CA3AF" font-size="11" text-anchor="middle" font-family="sans-serif">High P</text>

      <rect x="470" y="45" width="130" height="45" rx="6" fill="#0E4B4F" stroke="#00C2CB" stroke-width="1.5"/>
      <text x="535" y="72" fill="#FAFAFA" font-size="14" text-anchor="middle" font-family="sans-serif">Boiler</text>

      <line x1="610" y1="67" x2="660" y2="67" stroke="#2D333B" stroke-width="1" stroke-dasharray="4,3"/>
      <text x="760" y="30" fill="#9CA3AF" font-size="11" text-anchor="middle" font-family="sans-serif">Monitored:</text>
      <text x="760" y="55" fill="#FAFAFA" font-size="11" text-anchor="middle" font-family="sans-serif">Temp · Vibration · Pressure</text>
      <text x="760" y="72" fill="#FAFAFA" font-size="11" text-anchor="middle" font-family="sans-serif">Flow · RPM · Power</text>
      <text x="760" y="89" fill="#FAFAFA" font-size="11" text-anchor="middle" font-family="sans-serif">Operating Hours</text>
    </svg>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Sensor readings over time")
    st.caption("All 8 monitored parameters — this is what the AI actually sees for every reading.")

    cols = st.columns(2)
    plt.style.use("dark_background")

    for i, col in enumerate(FEATURES):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.plot(df["Reading_ID"], df[col], linewidth=0.8, color="#00C2CB")
            ax.set_title(FEATURE_LABELS[col], color="#FAFAFA")
            ax.set_xlabel("Reading #")
            ax.tick_params(colors="#9CA3AF")
            for spine in ax.spines.values():
                spine.set_color("#2D333B")
            render_fig(fig)

    st.subheader("Condition breakdown (simulated dataset)")
    st.bar_chart(df["Condition"].value_counts())

# ---------- TAB 2: AI Diagnosis on a chosen reading ----------
with tab2:
    st.write("Select which sensor reading to diagnose (defaults to the most recent):")
    reading_num = st.slider(
        "Reading #", min_value=1, max_value=len(df), value=len(df), key="diag_reading"
    )
    latest = df.iloc[reading_num - 1]
    x_latest = latest[FEATURES].values.reshape(1, -1)

    pred = model.predict(x_latest)[0]
    proba = model.predict_proba(x_latest)[0]
    classes = model.classes_
    risk_idx = list(classes).index("Failure Risk") if "Failure Risk" in classes else None
    failure_risk_pct = proba[risk_idx] * 100 if risk_idx is not None else 0

    fg, bg = STATUS_COLORS.get(pred, ("#FAFAFA", "#1C2128"))
    st.markdown(
        f'<span class="status-pill" style="color:{fg};background:{bg};">'
        f'{STATUS_EMOJI.get(pred,"")} {pred}</span>',
        unsafe_allow_html=True
    )
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        m1, m2 = st.columns(2)
        m1.metric("Failure Risk", f"{failure_risk_pct:.1f}%")
        m2.metric("Pump Health", f"{100 - failure_risk_pct:.1f}%")
        st.write("Latest sensor reading:")
        pretty = latest[FEATURES].to_frame(name="Value")
        pretty.index = [FEATURE_LABELS[i] for i in pretty.index]
        st.dataframe(pretty, use_container_width=True)

    with col2:
        st.write("**Key factors influencing this prediction** (from the trained model)")
        importances = pd.Series(model.feature_importances_, index=FEATURES)
        importances.index = [FEATURE_LABELS[i] for i in importances.index]
        importances = importances.sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        importances.plot(kind="barh", ax=ax, color="#00C2CB")
        ax.invert_yaxis()
        ax.tick_params(colors="#9CA3AF")
        for spine in ax.spines.values():
            spine.set_color("#2D333B")
        render_fig(fig)

    st.subheader("Recommended action")
    if pred == "Normal":
        st.success("Continue normal operation. No action needed.")
    elif pred == "Warning":
        st.warning("Schedule an inspection soon — vibration/temperature trending up. Possible early bearing wear.")
    else:
        st.error("High failure risk. Inspect the bearing and shaft alignment before next operating cycle.")

# ---------- TAB 3: What-if simulator ----------
with tab3:
    st.subheader("Adjust sensor values and re-run the AI diagnosis")
    st.caption("Create your own scenario — move the sliders, then run the diagnostic to see the model react live.")

    c1, c2 = st.columns(2)
    with c1:
        temp = st.slider("Temperature (°C)", 50.0, 110.0, 72.0)
        vib = st.slider("Vibration (mm/s)", 0.0, 12.0, 3.5)
        inlet_p = st.slider("Inlet Pressure (bar)", 0.5, 1.6, 1.2)
        outlet_p = st.slider("Outlet Pressure (bar)", 3.5, 6.0, 5.2)
    with c2:
        flow = st.slider("Flow Rate (L/min)", 70.0, 140.0, 118.0)
        rpm = st.slider("RPM", 1300.0, 1500.0, 1450.0)
        power = st.slider("Power (kW)", 3.5, 6.5, 4.3)
        op_hours = st.slider("Operating Hours since service", 0.0, 12000.0, 1000.0)

    if st.button("RUN AI DIAGNOSTIC", type="primary", use_container_width=True):
        x_new = np.array([[temp, vib, inlet_p, outlet_p, flow, rpm, power, op_hours]])
        pred = model.predict(x_new)[0]
        proba = model.predict_proba(x_new)[0]
        classes = model.classes_
        risk_idx = list(classes).index("Failure Risk") if "Failure Risk" in classes else None
        failure_risk_pct = proba[risk_idx] * 100 if risk_idx is not None else 0

        fg, bg = STATUS_COLORS.get(pred, ("#FAFAFA", "#1C2128"))
        st.markdown(
            f'<span class="status-pill" style="color:{fg};background:{bg};">'
            f'{STATUS_EMOJI.get(pred,"")} {pred}</span>',
            unsafe_allow_html=True
        )
        m1, m2 = st.columns(2)
        m1.metric("Failure Risk", f"{failure_risk_pct:.1f}%")
        m2.metric("Pump Health", f"{100 - failure_risk_pct:.1f}%")

        st.toast(f"Diagnosis complete: {pred}", icon=STATUS_EMOJI.get(pred, "✅"))
        if pred == "Normal":
            st.balloons()

# ---------- Bottom credit bar — shows under every tab ----------
st.markdown("""
<div class="credit-wrap">
  <span class="credit-bar">Agriya Rajsi Sinha — <b>Mechanical & Automation Engineering</b>, IGDTUW</span>
</div>
""", unsafe_allow_html=True)