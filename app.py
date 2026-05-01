import streamlit as st
import pandas as pd

# ================= DATA =================
DATA = {
    "Low Traffic": [0.55, 0.23, 0.38],
    "Medium Traffic": [1.74, 15.33, 0.46],
    "High Traffic": [3.58, 6.10, 0.88],
    "Imbalanced Traffic": [2.30, 13.37, 0.51],
}

# ================= UI =================
st.set_page_config(page_title="Traffic RL Dashboard", layout="centered")

st.title("🚦 Traffic Signal Optimization Dashboard")
st.markdown("### Waiting Time & Performance Analysis")

scenario = st.selectbox("Select Scenario", list(DATA.keys()))

fixed, greedy, rl = DATA[scenario]

# ================= WAITING TIME GRAPH =================
st.subheader("📊 Waiting Time Comparison")

df = pd.DataFrame({
    "Method": ["Fixed", "Greedy", "RL"],
    "Waiting Time": [fixed, greedy, rl]
})

st.bar_chart(df.set_index("Method"))

# ================= PERFORMANCE GRAPH =================
st.subheader("🚀 Performance Improvement (RL vs Fixed)")

improvement = ((fixed - rl) / fixed) * 100 if fixed != 0 else 0

perf_df = pd.DataFrame({
    "Metric": ["Improvement %"],
    "Value": [improvement]
})

st.bar_chart(perf_df.set_index("Metric"))

# ================= SIMPLE INSIGHT =================
st.subheader("🧠 Insight")

if rl < fixed and rl < greedy:
    st.success("RL significantly reduces waiting time and performs best 🚀")
else:
    st.warning("RL shows improvement but can be further optimized")

# ================= FOOTER =================
st.markdown("---")
st.markdown("Powered by Reinforcement Learning (DQN) + SUMO")