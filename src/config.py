# --- Simulation Parameters ---
NUM_FLOORS = 10
ELEVATOR_CAPACITY = 10
NUM_EPISODES = 500
STEPS_PER_EPISODE = 2000

# --- Agent / Training Hyperparameters ---
BATCH_SIZE = 64         # Batch size for replay buffer sampling
BUFFER_SIZE = 100000    # Total size of replay memory
GAMMA = 0.99            # Discount factor
LR = 0.0005             # Learning rate
TARGET_UPDATE = 10      # Update target network every N episodes

# --- Exploration (Epsilon-Greedy) ---
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

# --- Rewards & Penalties (Reward Shaping) ---
# These drive the agent's behavior. Tweak these to change policies.
REWARD_DELIVERY = 20.0       # Bonus for delivering a passenger
PENALTY_MOVE = 0.75          # Energy cost per floor move
PENALTY_STOP = 0.0           # (Optional) Cost for stopping
PENALTY_WRONG_DIR = 10.0     # Penalty for moving away from request
PENALTY_WAIT_TIME = 1.0      # Per-step penalty for waiting passengers
PENALTY_RIDE_TIME = 0.2      # Per-step penalty for riding passengers
PENALTY_SQUARED_WAIT = 0.0001 # Coefficient for squared waiting time (fairness)

# --- Visualization ---
SCREEN_WIDTH = 800
FLOOR_HEIGHT = 60
