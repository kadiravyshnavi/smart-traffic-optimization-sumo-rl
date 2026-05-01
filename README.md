# 🚦 Smart Traffic Signal Optimization using Reinforcement Learning (DQN + SUMO)

## 🌐 Live Demo

👉 https://smart-rl-traffic-optimization.streamlit.app/

---

## 📌 Overview

This project implements an intelligent traffic signal control system using **Deep Reinforcement Learning (DQN)** and **SUMO (Simulation of Urban Mobility)**.

The system learns adaptive signal policies to **minimize vehicle waiting time** and **reduce congestion**, outperforming traditional fixed and greedy approaches.

---

## 🎯 Problem Statement

Traditional traffic signals use fixed timings, which leads to:

* Traffic congestion 🚗
* Increased waiting time ⏳
* Inefficient road utilization

---

## 💡 Solution

This project uses:

* **SUMO** → realistic traffic simulation
* **Deep Q-Network (DQN)** → adaptive signal control
* **Evaluation system** → compares RL with baseline methods

---

## ⚙️ Technologies Used

* Python
* PyTorch
* SUMO (Simulation of Urban Mobility)
* Streamlit

---

## 🚀 Features

* DQN-based traffic signal optimization
* Experience replay + target network
* Multi-scenario evaluation:

  * Low traffic
  * Medium traffic
  * High traffic
  * Imbalanced traffic
* Comparison with:

  * Fixed timing strategy
  * Greedy strategy
* Interactive dashboard for visualization

---

## 📊 Results

### 🔹 Waiting Time Comparison

![Comparison](comparison_bar.png)

### 🔹 Scenario-wise Analysis

![Scenario](scenario_comparison.png)

---

## 📈 Performance Summary

| Method | Avg Waiting Time |
| ------ | ---------------- |
| Fixed  | ~2.03            |
| Greedy | ~7.10            |
| RL     | ~0.54 ✅          |

👉 RL significantly reduces congestion and waiting time.

---

## 📁 Project Structure

```
traffic_project/
├── train.py              # RL training
├── evaluate.py           # Evaluation & comparison
├── dashboard.py          # Local SUMO live dashboard
├── app.py                # Deployed Streamlit app
├── config.sumocfg        # SUMO configuration
├── *.rou.xml             # Traffic scenarios
├── *.net.xml             # Road network
├── best_model.pth        # Trained model
├── *.png                 # Result graphs
```

---

## ▶️ How to Run

### 🔹 Train Model

```bash
python train.py
```

### 🔹 Evaluate Model

```bash
python evaluate.py
```

### 🔹 Run Local Dashboard (with SUMO)

```bash
streamlit run dashboard.py
```

---

## 🎓 Key Learning Outcomes

* Applied Reinforcement Learning to real-world traffic optimization
* Integrated ML models with SUMO simulation
* Built evaluation and visualization pipeline
* Developed interactive dashboard for insights

---

## 🔮 Future Improvements

* Multi-intersection traffic control
* Real-time API integration (weather/traffic)
* Graph Neural Networks (GNN)
* Smart city deployment

---

## ⭐ Support

If you like this project, give it a ⭐ and share!
