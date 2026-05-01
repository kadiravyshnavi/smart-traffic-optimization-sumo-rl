import torch
import numpy as np
import traci
import matplotlib.pyplot as plt   # ✅ ADDED

from train import DQN, TrafficEnv, MAX_STEPS

# ================= SCENARIOS =================
ROUTE_FILES = [
    "low.rou.xml",
    "medium.rou.xml",
    "high.rou.xml",
    "imbalanced.rou.xml",
    "random.rou.xml"
]


# ================= RL =================
def run_rl(SUMO_CMD):
    if not traci.isLoaded():
        traci.start(SUMO_CMD)

    env = TrafficEnv(SUMO_CMD)

    model = DQN(9, 4)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    state = env.reset()
    total_wait = 0

    for _ in range(MAX_STEPS):
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


# ================= FIXED =================
def run_fixed(SUMO_CMD):
    if not traci.isLoaded():
        traci.start(SUMO_CMD)

    env = TrafficEnv(SUMO_CMD)
    state = env.reset()

    total_wait = 0

    for step in range(MAX_STEPS):
        action = (step // 10) % 4

        next_state, _, done = env.step(action)
        total_wait += sum(next_state[4:8])
        state = next_state

        if done:
            break

    traci.close()
    return total_wait


# ================= GREEDY =================
def run_greedy(SUMO_CMD):
    if not traci.isLoaded():
        traci.start(SUMO_CMD)

    env = TrafficEnv(SUMO_CMD)
    state = env.reset()

    total_wait = 0

    for _ in range(MAX_STEPS):
        action = np.argmax(state[:4])

        next_state, _, done = env.step(action)
        total_wait += sum(next_state[4:8])
        state = next_state

        if done:
            break

    traci.close()
    return total_wait


# ================= MAIN =================
if __name__ == "__main__":

    fixed_scores = []
    greedy_scores = []
    rl_scores = []

    scenario_names = ["Low", "Medium", "High", "Imbalanced", "Random"]

    for i, route in enumerate(ROUTE_FILES):

        print(f"\n===== SCENARIO {i+1}: {route} =====")

        SUMO_CMD = [
            "sumo",
            "-c", "config.sumocfg",
            "--route-files", route
        ]

        f = run_fixed(SUMO_CMD)
        g = run_greedy(SUMO_CMD)
        r = run_rl(SUMO_CMD)

        fixed_scores.append(f)
        greedy_scores.append(g)
        rl_scores.append(r)

        print(f"Fixed  : {f:.2f}")
        print(f"Greedy : {g:.2f}")
        print(f"RL     : {r:.2f}")

    print("\n===== FINAL AVERAGE RESULTS =====")
    print(f"Fixed  : {np.mean(fixed_scores):.2f}")
    print(f"Greedy : {np.mean(greedy_scores):.2f}")
    print(f"RL     : {np.mean(rl_scores):.2f}")

    # ================= 📊 GRAPHS =================

    # 1️⃣ BAR CHART (MAIN RESULT)
    labels = ['Fixed', 'Greedy', 'RL']
    values = [
        np.mean(fixed_scores),
        np.mean(greedy_scores),
        np.mean(rl_scores)
    ]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Average Waiting Time")
    plt.title("Overall Performance Comparison")
    plt.savefig("comparison_bar.png")


    # 2️⃣ SCENARIO-WISE LINE GRAPH
    plt.figure()
    plt.plot(scenario_names, fixed_scores, marker='o', label="Fixed")
    plt.plot(scenario_names, greedy_scores, marker='o', label="Greedy")
    plt.plot(scenario_names, rl_scores, marker='o', label="RL")

    plt.xlabel("Scenarios")
    plt.ylabel("Waiting Time")
    plt.title("Scenario-wise Comparison")
    plt.legend()
    plt.savefig("scenario_comparison.png")


    # 3️⃣ IMPROVEMENT GRAPH (RL vs Fixed)
    improvement = [
        (f - r) / f * 100 if f != 0 else 0
        for f, r in zip(fixed_scores, rl_scores)
    ]

    plt.figure()
    plt.bar(scenario_names, improvement)
    plt.ylabel("% Improvement over Fixed")
    plt.title("RL Improvement (%)")
    plt.savefig("improvement.png")


    # 4️⃣ GROUPED BAR (VERY PROFESSIONAL)
    x = np.arange(len(scenario_names))
    width = 0.25

    plt.figure()
    plt.bar(x - width, fixed_scores, width, label="Fixed")
    plt.bar(x, greedy_scores, width, label="Greedy")
    plt.bar(x + width, rl_scores, width, label="RL")

    plt.xticks(x, scenario_names)
    plt.ylabel("Waiting Time")
    plt.title("Detailed Comparison")
    plt.legend()
    plt.savefig("grouped_comparison.png")

    plt.show()