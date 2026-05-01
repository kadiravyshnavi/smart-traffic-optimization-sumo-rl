import streamlit as st
import numpy as np
import torch
import traci
import pandas as pd

from train import DQN, TrafficEnv, MAX_STEPS

# ================= CONFIG =================
SCENARIOS = {
    "Low Traffic": "low.rou.xml",
    "Medium Traffic": "medium.rou.xml",
    "High Traffic": "high.rou.xml",
    "Imbalanced Traffic": "imbalanced.rou.xml"
}

# ================= FUNCTION =================
def run_model(mode, route_file):
    SUMO_CMD = [
        "sumo",
        "-c", "config.sumocfg",
        "--route-files", route_file
    ]

    if not traci.isLoaded():
        traci.start(SUMO_CMD)

    env = TrafficEnv(SUMO_CMD)

    state = env.reset()
    total_wait = 0

    model = None
    if mode == "RL":
        model = DQN(9, 4)
        model.load_state_dict(torch.load("best_model.pth"))
        model.eval()

    for step in range(MAX_STEPS):

        if mode == "Fixed":
            action = (step // 10) % 4

        elif mode == "Greedy":
            action = np.argmax(state[:4])

        elif mode == "RL":
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action = torch.argmax(model(state_tensor)).item()

        next_state, _, done = env.step(action)
        total_wait += sum(next_state[4:8])
        state = next_state

        if done:
            break

    traci.close()
    return total_wait


# ================= UI =================
st.set_page_config(page_title="Traffic RL Dashboard", layout="centered")

st.title("🚦 Smart Traffic Signal Control Dashboard")
st.markdown("Compare Fixed, Greedy, and RL strategies")

scenario = st.selectbox("Select Traffic Scenario", list(SCENARIOS.keys()))

if st.button("Run Simulation 🚀"):

    route_file = SCENARIOS[scenario]

    with st.spinner("Running SUMO simulation... ⏳"):
        fixed = run_model("Fixed", route_file)
        greedy = run_model("Greedy", route_file)
        rl = run_model("RL", route_file)

    st.success("Simulation Completed ✅")

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
    st.subheader("📊 Comparison (Bar Chart)")
    bar_data = {
        "Method": ["Fixed", "Greedy", "RL"],
        "Waiting Time": [fixed, greedy, rl]
    }
    st.bar_chart(bar_data, x="Method", y="Waiting Time")

    # ================= LINE GRAPH =================
    st.subheader("📈 Method Comparison (Line Chart)")
    df = pd.DataFrame({
        "Method": ["Fixed", "Greedy", "RL"],
        "Waiting Time": [fixed, greedy, rl]
    })
    st.line_chart(df.set_index("Method"))

    # ================= AREA GRAPH =================
    st.subheader("📉 Performance Distribution")
    st.area_chart({
        "Fixed": [fixed],
        "Greedy": [greedy],
        "RL": [rl]
    })

    # ================= TRAINING GRAPHS =================
    st.subheader("📚 Training Analysis")

    try:
        st.image("reward_curve.png", caption="Reward Curve")
        st.image("waiting_curve.png", caption="Waiting Time Curve")
        st.image("smoothed_curve.png", caption="Smoothed Performance")
    except:
        st.warning("Training graphs not found. Run train.py to generate them.")

    # ================= INSIGHTS =================
    st.subheader("🧠 Insights")

    if rl < fixed and rl < greedy:
        st.success("RL significantly reduces congestion and performs best 🚀")
    elif rl < fixed:
        st.warning("RL improves over fixed signals but greedy is competitive")
    else:
        st.error("RL performance needs further tuning")

# ================= FOOTER =================
st.markdown("---")
st.markdown("Built with SUMO + DQN + Streamlit")