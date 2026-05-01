import streamlit as st
import pandas as pd

# ================= DATA =================
# Precomputed results from your evaluation
DATA = {
    "Low Traffic": [0.55, 0.23, 0.38],
    "Medium Traffic": [1.74, 15.33, 0.46],
    "High Traffic": [3.58, 6.10, 0.88],
    "Imbalanced Traffic": [2.30, 13.37, 0.51],
}

# ================= UI =================
st.set_page_config(page_title="Traffic RL Dashboard", layout="centered")

st.title("🚦 Smart Traffic Signal Optimization Dashboard")
st.markdown("### Reinforcement Learning (DQN) vs Traditional Traffic Control")

scenario = st.selectbox("Select Traffic Scenario", list(DATA.keys()))

fixed, greedy, rl = DATA[scenario]

# ================= METRICS =================
st.subheader("📊 Results")

col1, col2, col3 = st.columns(3)
col1.metric("Fixed", f"{fixed:.2f}")
col2.metric("Greedy", f"{greedy:.2f}")
col3.metric("RL", f"{rl:.2f}")

# ================= IMPROVEMENT =================
improvement = ((fixed - rl) / fixed) * 100 if fixed != 0 else 0
st.metric("🚀 RL Improvement over Fixed", f"{improvement:.2f}%")

# ================= BAR GRAPH =================
st.subheader("📊 Method Comparison (Bar Chart)")
df = pd.DataFrame({
    "Method": ["Fixed", "Greedy", "RL"],
    "Waiting Time": [fixed, greedy, rl]
})
st.bar_chart(df.set_index("Method"))

# ================= LINE GRAPH =================
st.subheader("📈 Trend Comparison")
st.line_chart(df.set_index("Method"))

# ================= AREA GRAPH =================
st.subheader("📉 Performance Distribution")
st.area_chart({
    "Fixed": [fixed],
    "Greedy": [greedy],
    "RL": [rl]
})

# ================= SCENARIO COMPARISON =================
st.subheader("🌍 Scenario-wise RL Performance")

scenario_df = pd.DataFrame({
    "Scenario": list(DATA.keys()),
    "RL Waiting Time": [v[2] for v in DATA.values()]
})
st.bar_chart(scenario_df.set_index("Scenario"))

# ================= TRAINING GRAPHS =================
st.subheader("📚 Training Analysis")

try:
    st.image("reward_curve.png", caption="Reward Curve")
    st.image("waiting_curve.png", caption="Waiting Time Curve")
    st.image("smoothed_curve.png", caption="Smoothed Curve")
except:
    st.warning("Training graphs not found in repo.")

# ================= INSIGHTS =================
st.subheader("🧠 Insights")

if rl < fixed and rl < greedy:
    st.success("RL performs best and significantly reduces congestion 🚀")
elif rl < fixed:
    st.warning("RL improves over fixed signals but can be optimized further")
else:
    st.error("RL needs improvement in this scenario")

# ================= FOOTER =================
st.markdown("---")
st.markdown("🚀 Built with DQN + SUMO + Streamlit")