# Deep Reinforcement Learning for Elevator Group Control

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)

An intelligent elevator scheduling system that uses Deep Q-Networks (DQN) to reduce passenger wait times by up to 25% compared to traditional algorithms.

## 📖 Project Overview

Modern skyscrapers require efficient vertical transportation. Traditional algorithms like **"Nearest Car"** rely on fixed rules that often fail during rush hours, leading to long wait times and energy waste.

This project implements a **Deep Reinforcement Learning (RL)** agent that learns to control a group of elevators. By treating the building as a *Markov Decision Process (MDP)*, the agent learns complex dispatching strategies through trial-and-error, optimizing for minimal passenger waiting time and fairness.

---

## 📈 Results

We compared the trained DQN agent against the Nearest-Car baseline over 100 test episodes. The agent demonstrates significant improvements, particularly during high-stress "Up-Peak" scenarios.

| Traffic Pattern | Baseline (Avg Wait) | DQN Agent (Avg Wait) | Improvement |
|------------------|----------------------|------------------------|-------------|
| Uniform          | 24.5s               | 19.8s                 | ~19%        |
| Up-Peak          | 35.2s               | 25.2s                 | ~28%        |
| Mixed            | 31.8s               | 23.5s                 | ~26%        |


### Performance Visualizations

| Reward Curve | Average Waiting Time |
| :---: | :---: |
| ![Reward Curve](assets/total_reward_per_episode.png) | ![Average Wait Time](assets/average_wait_time_per_step.png) |

---

## ✨ Key Features

* **🧠 Deep Q-Network (DQN):** Uses a Multi-Layer Perceptron to approximate Q-values for high-dimensional state spaces.
* **🏢 Custom Simulation Environment:** A discrete-event simulator modeling elevator physics, passenger arrival patterns, and building dynamics.
* **📊 Comparison Baseline:** Benchmarked against the industry-standard "Nearest-Car" (NC) heuristic algorithm.
* **🚦 Dynamic Traffic Handling:** Robust performance across **Uniform**, **Up-Peak** (Morning Rush), and **Mixed** (Lunch Hour) traffic patterns.

---

## 🛠️ Tech Stack

* **Core:** ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
* **Deep Learning:** ![PyTorch](https://img.shields.io/badge/PyTorch-v1.9%2B-ee4c2c?logo=pytorch&logoColor=white)
* **Simulation & Viz:** ![Pygame](https://img.shields.io/badge/Pygame-v2.0-yellow?logo=python&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-Computation-013243?logo=numpy&logoColor=white)
* **Analysis:** ![Matplotlib](https://img.shields.io/badge/Matplotlib-Plotting-11557c?logo=python&logoColor=white)

---

## ⚙️ How It Works

```bash
+---------------------------+                     +-------------------------+
|   ELEVATOR ENVIRONMENT    |                     |        DQN AGENT        |
|---------------------------|                     |-------------------------|
|  Current Floor & Load     |   State Vector (s)  |  Input: State size (32) |
|  Active Hall Calls (U/D)  |-------------------->|  [   Neural Network   ] |
|  Elapsed Wait Times       |                     |  Output: Q-Values       |
|                           |                     |                         |
|  Execute: Physics Step    |   Select Action (a) |  Policy: Epsilon-Greedy |
|  Calculate: Wait Penalty  |<--------------------|                         |
+---------------------------+                     +-------------------------+
             |                                                 ^
             |          Reward (r) & Next State (s')           |
             +-------------------------------------------------+
                                     |
                          (Store in Replay Buffer)
```

### 1. The Environment (MDP)
The building is modeled as a state grid where the agent observes:
* **Elevator States:** Position, direction, current load, and internal button presses.
* **Hall Calls:** Boolean vector indicating waiting passengers (Up/Down) at each floor.
* **Wait Times:** Elapsed time for active calls (used to prevent starvation).

### 2. The Agent (DQN)
Instead of hard-coded rules, the agent optimizes a custom Reward Function:

$$R_t = - \sum (w_p)^2 - \lambda \cdot E_{cost}$$

* **Squared Waiting Time ($w_p^2$):** Penalizes long waits heavily to ensure fairness.
* **Energy Cost ($E_{cost}$):** Penalizes excessive movement.

The network takes the state vector as input and outputs the Q-value for assigning a new hall call to a specific elevator.

---

## 📂 Project Structure

```text
Elevator-Scheduling-using-Reinforcement-Learning/
├── src/
│   ├── config.py        
│   ├── dqn_agent.py     # Deep Q-Network model & Replay Buffer
│   └── elevator_env.py  # Custom discrete-event simulation environment
|
├── models/              # Directory for saved model weights (.pth)
├── assets/              # Reports, visualizations, performance plots
|
├── train.py             # Entry point for training the agent
├── visualize.py         # Entry point for running the graphical demo
├── requirements.txt     
└── README.md
```

---

## 💻 Installation & Usage

### Prerequisites

* Python 3.8+
* PyTorch (or TensorFlow, depending on your implementation)
* NumPy, Matplotlib

### Setup

1. **Clone the repository:**
	```bash
		git clone [https://github.com/yourusername/Elevator-Scheduling-using-Reinforcement-Learning.git](https://github.com/yourusername/Elevator-Scheduling-using-Reinforcement-Learning.git)
 		cd Elevator-Scheduling-using-Reinforcement-Learning
 	```
2. Install dependencies:
	```bash
		pip install -r requirements.txt
	```
 ### 🚀 Quick Start (Visualization)
To see the trained agent in action immediately using pre-trained weights:
```bash
python visualize.py
```
### 🏋️‍♂️ Training from Scratch

To train the DQN agent from scratch:
```bash
	python train.py
```
_This will generate training logs and save the model weights to `/models`._

---

## 📄 Documentation

For a detailed technical explanation of the mathematical formulation, network architecture, and experimental setup, please refer to the Project Report (IEEE Format) included in this repository.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 🙏 Acknowledgements

* **DeepMind DQN:** The Deep Q-Network implementation is based on the foundational paper by [Mnih et al. (2015)](https://www.nature.com/articles/nature14236).
* **Pygame:** Visualization engine used for the rendering module.
* **OpenAI Gym:** The environment structure was inspired by the Gym API design patterns.

---

## 👤 Author

Yashwant Patnaikuni

📧 yashwantpatnaikuni@gmail.com <br>
ℹ️ www.linkedin.com/in/yashwant-patnaikuni

Nirup Koyilada

📧 nirupkoyilada@gmail.com <br>
ℹ️ https://www.linkedin.com/in/nirup-koyilada
