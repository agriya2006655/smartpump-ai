"""
app.py
------
SmartPump AI dashboard. Run with:
    streamlit run app.py

Needs pump_data.csv and pump_model.pkl in the same folder
(created by generate_data.py and train_model.py).
"""

import io
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

FEATURES = [
    "Temperature_C", "Vibration_mm_s", "Inlet_Pressure_bar",
    "Outlet_Pressure_bar", "Flow_Rate_Lmin", "RPM", "Power_kW",
    "Operating_Hours"
]

# ---------- Custom CSS for a cleaner, more "product-like" look ----------
st.markdown("""
<style>
    .main-header {
        padding: 1.5rem 2rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #0E4B4F 0%, #0E1117 100%);
        border: 1px solid #00C2CB40;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header p {
        margin: 0.3rem 0 0 0;
        color: #9CA3AF;
        font-size: 0.95rem;
    }
    div[data-testid="stMetric"] {
        background-color: #1C2128;
        border: 1px solid #2D333B;
        border-radius: 10px;
        padding: 1rem;
    }
    .status-pill {
        display: inline-block;
        padding: 0.3rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
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

# ---------- Header banner ----------
st.markdown("""
<div class="main-header">
    <h1>🔧 SmartPump AI</h1>
    <p>AI-Assisted Performance Monitoring & Predictive Maintenance · Boiler Feedwater Pump</p>
</div>
""", unsafe_allow_html=True)

st.caption("⚠️ All sensor data on this dashboard is SIMULATED for demonstration purposes.")

# ---------- Simple pump flow diagram (engineering context) ----------
st.markdown("""
<div style="background:#1C2128;border:1px solid #2D333B;border-radius:12px;
            padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
<svg viewBox="0 0 900 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;display:block;margin:0 auto;">
  <!-- Condenser -->
  <rect x="10" y="45" width="130" height="45" rx="6" fill="#0E4B4F" stroke="#00C2CB" stroke-width="1.5"/>
  <text x="75" y="72" fill="#FAFAFA" font-size="14" text-anchor="middle" font-family="sans-serif">Condenser</text>

  <!-- Arrow 1 -->
  <line x1="145" y1="67" x2="215" y2="67" stroke="#9CA3AF" stroke-width="2"/>
  <polygon points="215,61 227,67 215,73" fill="#9CA3AF"/>
  <text x="185" y="55" fill="#9CA3AF" font-size="11" text-anchor="middle" font-family="sans-serif">Low P</text>

  <!-- Pump -->
  <rect x="230" y="35" width="150" height="65" rx="6" fill="#0E4B4F" stroke="#00C2CB" stroke-width="2"/>
  <text x="305" y="60" fill="#00C2CB" font-size="14" text-anchor="middle" font-family="sans-serif" font-weight="bold">FEEDWATER</text>
  <text x="305" y="80" fill="#00C2CB" font-size="14" text-anchor="middle" font-family="sans-serif" font-weight="bold">PUMP</text>

  <!-- Arrow 2 -->
  <line x1="385" y1="67" x2="455" y2="67" stroke="#9CA3AF" stroke-width="2"/>
  <polygon points="455,61 467,67 455,73" fill="#9CA3AF"/>
  <text x="420" y="55" fill="#9CA3AF" font-size="11" text-anchor="middle" font-family="sans-serif">High P</text>

  <!-- Boiler -->
  <rect x="470" y="45" width="130" height="45" rx="6" fill="#0E4B4F" stroke="#00C2CB" stroke-width="1.5"/>
  <text x="535" y="72" fill="#FAFAFA" font-size="14" text-anchor="middle" font-family="sans-serif">Boiler</text>

  <!-- Sensors row -->
  <line x1="610" y1="67" x2="660" y2="67" stroke="#2D333B" stroke-width="1" stroke-dasharray="4,3"/>
  <text x="760" y="30" fill="#9CA3AF" font-size="11" text-anchor="middle" font-family="sans-serif">Monitored:</text>
  <text x="760" y="55" fill="#FAFAFA" font-size="11" text-anchor="middle" font-family="sans-serif">Temp · Vibration</text>
  <text x="760" y="72" fill="#FAFAFA" font-size="11" text-anchor="middle" font-family="sans-serif">Pressure · Flow</text>
  <text x="760" y="89" fill="#FAFAFA" font-size="11" text-anchor="middle" font-family="sans-serif">RPM · Power</text>
</svg>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📊 Performance Monitoring", "🧠 AI Diagnosis", "🎛️ What-If Simulator"])

# ---------- TAB 1: Performance graphs over time ----------
with tab1:
    st.subheader("Sensor readings over time")
    cols = st.columns(2)
    plot_cols = ["Temperature_C", "Vibration_mm_s", "Outlet_Pressure_bar",
                 "Flow_Rate_Lmin", "Power_kW", "RPM"]
    plt.style.use("dark_background")
    for i, col in enumerate(plot_cols):
        with cols[i % 2]:
            fig, ax = plt.subplots(figsize=(5, 2.5))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            ax.plot(df["Reading_ID"], df[col], linewidth=0.8, color="#00C2CB")
            ax.set_title(col, color="#FAFAFA")
            ax.set_xlabel("Reading #")
            ax.tick_params(colors="#9CA3AF")
            for spine in ax.spines.values():
                spine.set_color("#2D333B")
            render_fig(fig)

    st.subheader("Condition breakdown (simulated dataset)")
    st.bar_chart(df["Condition"].value_counts())

# ---------- TAB 2: AI Diagnosis on the most recent row ----------
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
        st.dataframe(latest[FEATURES].to_frame(name="Value"), use_container_width=True)

    with col2:
        st.write("**Key factors influencing this prediction** (from the trained model)")
        importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
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
