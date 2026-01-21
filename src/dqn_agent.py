import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from src import config
from collections import deque

# --- Hyperparameters ---
BUFFER_SIZE = 100000  # How many experiences to store in memory
BATCH_SIZE = 64       # How many experiences to use for each learning step
GAMMA = 0.99          # Discount factor (how much to value future rewards)
LR = 0.0005           # Learning rate for the optimizer
EPSILON_START = 1.0   # Starting value for exploration
EPSILON_END = 0.01    # Minimum value for exploration
EPSILON_DECAY = 0.995 # How fast to reduce exploration
TARGET_UPDATE = 10    # How often (in episodes) to update the target network

class QNetwork(nn.Module):
    """The neural network that estimates Q-values."""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # A simple Multi-Layer Perceptron (MLP)
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        return self.layer3(x) # Output Q-values for each action

class ReplayBuffer:
    """A fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        # Convert to torch tensors
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float()
        
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """The agent that interacts with and learns from the environment."""
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = config.EPSILON_START
        
        # --- Q-Network Initialization ---
        # Main network (gets trained)
        self.q_network = QNetwork(state_size, action_size)
        # Target network (used for stable Q-value targets)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict()) # Sync them
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.LR)
        
        # --- Replay Memory ---
        self.memory = ReplayBuffer(config.BUFFER_SIZE, config.BATCH_SIZE)
        
        # --- Loss Function ---
        self.criterion = nn.MSELoss() # Mean Squared Error Loss

    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        # Epsilon-Greedy:
        # With prob. epsilon, take a random action (explore)
        # With prob. 1-epsilon, take the best action from the network (exploit)
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0) # Prep state for network
            self.q_network.eval() # Set network to evaluation mode
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train() # Set network back to training mode
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size)) # Random action

    def learn(self):
        """
        Learn from a batch of experiences stored in memory.
        This is the core Bellman equation update.
        """
        if len(self.memory) < BATCH_SIZE:
            return # Don't learn until we have enough experiences
            
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # --- 1. Get Q-values for next states from the *target* network ---
        # We want max_a' Q_target(s', a')
        Q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # --- 2. Calculate the "target" Q-value (y) ---
        # y = r + gamma * Q_target(s', a')   (if not done)
        # y = r                               (if done)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        
        # --- 3. Get Q-values for current states from the *main* network ---
        # We want Q(s, a)
        Q_expected = self.q_network(states).gather(1, actions)
        
        # --- 4. Calculate Loss ---
        loss = self.criterion(Q_expected, Q_targets)
        
        # --- 5. Optimize the main network ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self, state, action, reward, next_state, done):
        """Called at every step of the simulation."""
        # Store the experience in memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Trigger the learning step
        self.learn()

    def end_of_episode(self, episode_num):
        """Called at the end of each episode."""
        # Decay epsilon (exploration)
        self.epsilon = max(config.EPSILON_END, config.EPSILON_DECAY * self.epsilon)
        
        # Update the target network
        if episode_num % TARGET_UPDATE == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"*** Updated target network at episode {episode_num} ***")

    def save(self, filename):
        """Saves the trained network."""
        torch.save(self.q_network.state_dict(), filename)    


