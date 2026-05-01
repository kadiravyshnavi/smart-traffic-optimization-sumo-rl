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

st.title("🚦 Traffic Signal Optimization using RL")

# ================= SELECT SCENARIO =================
scenario = st.selectbox("Select Traffic Scenario", list(DATA.keys()))

fixed, greedy, rl = DATA[scenario]

# ================= SCENARIO VIEW =================
st.subheader(f"📊 Results for {scenario}")

single_df = pd.DataFrame({
    "Method": ["Fixed", "Greedy", "RL"],
    "Waiting Time": [fixed, greedy, rl]
})

st.bar_chart(single_df.set_index("Method"))

# ================= STRONG GLOBAL GRAPH =================
st.subheader("📈 Overall Comparison Across All Scenarios")

global_df = pd.DataFrame(DATA, index=["Fixed", "Greedy", "RL"]).T

st.line_chart(global_df)

# ================= INSIGHT =================
st.subheader("🧠 Insight")

if rl < fixed and rl < greedy:
    st.success("RL consistently performs best and reduces waiting time 🚀")
else:
    st.warning("RL shows improvement but may need tuning")

# ================= FOOTER =================
st.markdown("---")
st.markdown("Powered by DQN + SUMO + Streamlit")