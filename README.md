# Deep Reinforcement Learning for Elevator Group Control

An intelligent elevator scheduling system that uses Deep Q-Networks (DQN) to reduce passenger wait times by up to 25% compared to traditional algorithms.

## Project Overview

Modern skyscrapers require efficient vertical transportation. Traditional elevator algorithms (like "Nearest Car") rely on fixed rules that often fail during rush hours, leading to long wait times and energy waste.

This project implements a Deep Reinforcement Learning (RL) agent that learns to control a group of elevators. By treating the building as a Markov Decision Process (MDP), the agent learns complex dispatching strategies through trial-and-error, optimizing for minimal passenger waiting time and fair service.

### Key Features

- **Deep Q-Network (DQN):** Uses a neural network to approximate the Q-value function for high-dimensional state spaces.

- **Custom Simulation Environment:** A discrete-event simulator modeling elevator physics, passenger arrivals, and building dynamics.

- **Comparison Baseline:** Benchmarked against the industry-standard "Nearest-Car" (NC) heuristic algorithm.

- **Dynamic Traffic Handling:** robust performance across Uniform, Up-Peak (morning rush), and Mixed (lunch hour) traffic patterns.

## How It Works

### 1. The Environment (MDP)

The building is modeled as a grid where the agent observes:

- **Elevator States:** Position, direction, current load, and pressed buttons.

- **Hall Calls:** Which floors have passengers waiting (Up/Down).

- **Wait Times:** How long active calls have been waiting (to prevent starvation).

### 2. The Agent (DQN)

Instead of hard-coded rules, the agent receives a Reward based on:

- Negative sum of squared waiting times (penalizes long waits heavily).

- Energy consumption penalties (distance traveled).

The network takes the state vector as input and outputs the Q-value for assigning a new hall call to each specific elevator.

## Results

We compared the trained DQN agent against the Nearest-Car baseline over 100 test episodes.

| Traffic Pattern | Baseline (Avg Wait) | DQN Agent (Avg Wait) | Improvement |
|------------------|----------------------|------------------------|-------------|
| Uniform          | 24.5s               | 19.8s                 | ~19%        |
| Up-Peak          | 35.2s               | 25.2s                 | ~28%        |
| Mixed            | 31.8s               | 23.5s                 | ~26%        |


### Performance Visualizations

Reward Curve: 

![Reward Curve](assets/total_reward_per_episode.png)


Average Waiting Time Comparison:

![Average Wait Time](assets/average_wait_time_per_step.png)



## Installation & Usage

### Prerequisites

- Python 3.8+

- PyTorch (or TensorFlow, depending on your implementation)

- NumPy, Matplotlib

### Setup

1. Clone the repository:

		git clone [https://github.com/yourusername/elevator-rl.git](https://github.com/yourusername/elevator-rl.git) cd elevator-rl

2. Install dependencies:

		pip install -r requirements.txt

### Running the Training

To train the DQN agent from scratch:

	python train.py

_This will generate training logs and save the model weights to `/models`._

## Documentation

For a detailed technical explanation of the mathematical formulation, network architecture, and experimental setup, please refer to the Project Report (IEEE Format) included in this repository.

---

About 'iterations' Folder
	
	'iterations' is a folder that contains the .png files that we collected form the various iterations of making and improving the model.

## Authors

- Yashwant Patnaikuni

- Nirup Koyilada
