"""
generate_data.py
-----------------
Creates a realistic SIMULATED dataset for a boiler feedwater pump.
Run this first. It creates pump_data.csv which everything else uses.

Unlike a purely random dataset, this simulates the pump running
continuously over time and SLOWLY DEGRADING — the way a real machine
would. That gives you three honest phases in the data:

  Healthy (stable) -> Gradual degradation (vibration/temp creep up)
  -> Warning -> Failure Risk (abnormal, noisy readings)

Small random noise is added on top so it still looks like real sensor
data, not a perfectly smooth line.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N_ROWS = 1200

t = np.arange(N_ROWS)

# ---------- Build a slow degradation trend over time ----------
# Vibration and temperature slowly climb as operating hours accumulate,
# with small day-to-day noise on top (this is what makes it look real
# instead of randomly jumping between extremes every single reading).
degradation = np.clip((t / N_ROWS) ** 1.8, 0, 1)  # 0 -> 1 over time, accelerating near the end

base_vibration = 3.0 + degradation * 7.0          # 3.0 mm/s -> ~10 mm/s
vibration = base_vibration + np.random.normal(0, 0.4, N_ROWS)
vibration = np.clip(vibration, 1.5, 13)

base_temp = 70 + degradation * 28                 # 70C -> ~98C
temperature = base_temp + np.random.normal(0, 2.0, N_ROWS)

base_power = 4.3 + degradation * 1.4
power = base_power + np.random.normal(0, 0.15, N_ROWS)

# As the pump degrades, flow and outlet pressure gradually DROP
base_flow = 118 - degradation * 22
flow = base_flow + np.random.normal(0, 3, N_ROWS)

base_outlet_p = 5.2 - degradation * 0.65
outlet_p = base_outlet_p + np.random.normal(0, 0.12, N_ROWS)

base_inlet_p = 1.2 - degradation * 0.28
inlet_p = base_inlet_p + np.random.normal(0, 0.05, N_ROWS)

base_rpm = 1450 - degradation * 65
rpm = base_rpm + np.random.normal(0, 12, N_ROWS)

operating_hours = t * (12000 / N_ROWS) + np.random.normal(0, 30, N_ROWS)
operating_hours = np.clip(operating_hours, 0, None)

# ---------- Label each reading based on how degraded it is ----------
def label_condition(d):
    if d < 0.45:
        return "Normal"
    elif d < 0.75:
        return "Warning"
    else:
        return "Failure Risk"

conditions = [label_condition(d) for d in degradation]

df = pd.DataFrame({
    "Reading_ID": t + 1,
    "Temperature_C": temperature.round(1),
    "Vibration_mm_s": vibration.round(2),
    "Inlet_Pressure_bar": inlet_p.round(2),
    "Outlet_Pressure_bar": outlet_p.round(2),
    "Flow_Rate_Lmin": flow.round(1),
    "RPM": rpm.round(0),
    "Power_kW": power.round(2),
    "Operating_Hours": operating_hours.round(0),
    "Condition": conditions
})

df.to_csv("pump_data.csv", index=False)

print(f"Created pump_data.csv with {len(df)} rows (realistic degradation trend)")
print(df["Condition"].value_counts())
print("\nFirst 5 rows:")
print(df.head())
print("\nLast 5 rows (should show degraded/failure-risk values):")
print(df.tail())
