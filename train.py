import random
import numpy as np
import traci
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt   # ✅ ADDED

memory = deque(maxlen=5000)
BATCH_SIZE = 32

# ================= CONFIG =================
SUMO_CMD = ["sumo", "-c", "config.sumocfg"]
EPISODES = 150
MAX_STEPS = 200
GAMMA = 0.99
LR = 0.001
EPSILON = 1.0
EPSILON_DECAY = 0.95
EPSILON_MIN = 0.05

# ================= LANES =================
LANES = [
    "n2c_0", "n2c_1", "n2c_2",
    "s2c_0", "s2c_1", "s2c_2",
    "e2c_0", "e2c_1", "e2c_2",
    "w2c_0", "w2c_1", "w2c_2"
]

# ================= DQN =================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.net(x)

# ================= ENV =================
class TrafficEnv:
    def __init__(self, sumo_cmd):
        self.sumo_cmd = sumo_cmd
        self.last_action = 0
        self.phase_duration = 0
        self.min_green = 10

    def reset(self):
        if not traci.isLoaded():
            traci.start(self.sumo_cmd)
        else:
            traci.close()                
            traci.start(self.sumo_cmd)    

        self.last_action = 0
        self.phase_duration = 0

        return self.get_state()

    def step(self, action):
        actual_action = action

        if action != self.last_action and self.phase_duration < self.min_green:
            actual_action = self.last_action

        # store previous action
        prev_action = self.last_action

        # enforce min green again
        actual_action = action
        if action != prev_action and self.phase_duration < self.min_green:
            actual_action = prev_action

        # update phase duration
        if actual_action == prev_action:
            self.phase_duration += 1
        else:
            self.phase_duration = 0

        # apply action
        traci.trafficlight.setPhase("center", actual_action)

        # simulate
        for _ in range(5):
            traci.simulationStep()

        # next state
        next_state = self.get_state()

        # reward
        queues = next_state[:4]
        waits = next_state[4:8]

        reward = - (3.0 * sum(queues) + 1.5 * sum(waits))

        if actual_action != prev_action:
            reward -= 2

        self.last_action = actual_action

        done = traci.simulation.getMinExpectedNumber() <= 0

        return next_state, reward, done

    def get_state(self):
        queues = []
        waits = []

        for i in range(0, len(LANES), 3):
            q = 0
            w = 0
            for lane in LANES[i:i+3]:
                q += traci.lane.getLastStepHaltingNumber(lane)
                w += traci.lane.getWaitingTime(lane) / 1000.0
            queues.append(q / 20.0)
            waits.append(w / 50.0)

        return np.array(queues + waits + [self.last_action], dtype=np.float32)

# ================= TRAIN =================
def train():
    env = TrafficEnv(SUMO_CMD)

    state_size = 9
    action_size = 4

    model = DQN(state_size, action_size)
    target_model = DQN(state_size, action_size)
    target_model.load_state_dict(model.state_dict())
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    epsilon = EPSILON

    best_wait = float("inf")
    best_episode = 0

    # ✅ ADDED (for graphs)
    waiting_history = []
    reward_history = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        total_waiting_time = 0

        for step in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(state)

            if random.random() < epsilon:
                action = random.randint(0, action_size - 1)
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()

            next_state, reward, done = env.step(action)

            total_waiting_time += sum(next_state[4:8])

            memory.append((state, action, reward, next_state, done))

            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)

                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                next_states = torch.FloatTensor(np.array(next_states))
                actions = torch.LongTensor(actions)
                rewards = torch.FloatTensor(rewards)
                dones = torch.FloatTensor(dones)

                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze()

                next_q = target_model(next_states).max(1)[0]
                target_q = rewards + GAMMA * next_q * (1 - dones)

                loss = loss_fn(current_q, target_q.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward

            if done:
                break

        # save best model
        if total_waiting_time < best_wait:
            best_wait = total_waiting_time
            best_episode = episode + 1
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Waiting: {total_waiting_time:.2f}")

        # ✅ store for graphs
        waiting_history.append(total_waiting_time)
        reward_history.append(total_reward)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % 10 == 0:
            target_model.load_state_dict(model.state_dict())

    # ================= PLOTS =================

    # Waiting Curve
    plt.figure()
    plt.plot(waiting_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Waiting Time")
    plt.title("Training Curve - Waiting Time")
    plt.savefig("waiting_curve.png")

    # Reward Curve
    plt.figure()
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Curve - Reward")
    plt.savefig("reward_curve.png")

    # Smoothed Curve
    def moving_average(data, window=10):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    smooth_wait = moving_average(waiting_history)

    plt.figure()
    plt.plot(smooth_wait)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Waiting Time")
    plt.title("Smoothed Training Curve")
    plt.savefig("smoothed_curve.png")

    plt.show()

    traci.close()

    print(f"🔥 Best Waiting Time: {best_wait} (Episode {best_episode})")
    print("✅ Training complete")

# ================= MAIN =================
if __name__ == "__main__":
    train()