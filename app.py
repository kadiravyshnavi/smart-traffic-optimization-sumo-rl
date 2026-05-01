import streamlit as st
import pandas as pd

# ================= DATA =================
DATA = {
    "Low Traffic": [0.55, 0.23, 0.38],
    "Medium Traffic": [1.74, 15.33, 0.46],
    "High Traffic": [3.58, 6.10, 0.88],
    "Imbalanced Traffic": [2.30, 13.37, 0.51],
}

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Traffic RL Dashboard", layout="wide")

# ================= HEADER =================
st.markdown("""
# 🚦 Smart Traffic Optimization Dashboard
### Reinforcement Learning vs Traditional Traffic Control
""")

# ================= SELECT =================
scenario = st.selectbox("📍 Select Traffic Scenario", list(DATA.keys()))

fixed, greedy, rl = DATA[scenario]

# ================= KPI CARDS =================
st.markdown("## 📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("🚗 Fixed", f"{fixed:.2f}")
col2.metric("⚡ Greedy", f"{greedy:.2f}")
col3.metric("🤖 RL", f"{rl:.2f}")

# ================= IMPROVEMENT =================
improvement = ((fixed - rl) / fixed) * 100 if fixed != 0 else 0

st.markdown("## 🚀 Performance Gain")
st.metric("RL Improvement over Fixed", f"{improvement:.2f}%")

# ================= MAIN GRAPH =================
st.markdown("## 📈 Waiting Time Comparison (All Scenarios)")

global_df = pd.DataFrame(DATA, index=["Fixed", "Greedy", "RL"]).T
st.line_chart(global_df)

# ================= SCENARIO GRAPH =================
st.markdown(f"## 📊 Selected Scenario: {scenario}")

scenario_df = pd.DataFrame({
    "Method": ["Fixed", "Greedy", "RL"],
    "Waiting Time": [fixed, greedy, rl]
})

st.bar_chart(scenario_df.set_index("Method"))

# ================= RANKING =================
st.markdown("## 🏆 Ranking")

rank_df = scenario_df.sort_values("Waiting Time")
st.dataframe(rank_df, use_container_width=True)

# ================= INSIGHT =================
st.markdown("## 🧠 Insight")

if rl < fixed and rl < greedy:
    st.success("RL consistently achieves the lowest waiting time and adapts efficiently 🚀")
elif rl < fixed:
    st.warning("RL improves over fixed signals but greedy is competitive")
else:
    st.error("RL performance needs improvement")

# ================= FOOTER =================
st.markdown("---")
st.markdown("✨ Built with DQN + SUMO + Streamlit")