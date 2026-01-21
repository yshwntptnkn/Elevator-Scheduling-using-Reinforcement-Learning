import numpy as np
from elevator_env import ElevatorEnv
from dqn_agent import DQNAgent
from collections import deque
import matplotlib.pyplot as plt

# --- Main Training Loop ---

# --- 1. Initialize Environment and Agent ---
env = ElevatorEnv(num_floors=10)
state_size = env.get_state().shape[0] # Get state size from env
action_size = 3 # 0:Down, 1:Stop, 2:Up

agent = DQNAgent(state_size=state_size, action_size=action_size)

# --- 2. Training Parameters ---
NUM_EPISODES = 500  # Total number of "simulations" to run
STEPS_PER_EPISODE = 2000 # Max steps per simulation

# --- 3. Metrics Tracking ---
# These lists are what you will use for your "Results" graphs
episode_rewards = []
episode_wait_times = []
episode_passengers_delivered = []
rewards_window = deque(maxlen=100) # For a moving average

print(f"Starting training for {NUM_EPISODES} episodes...")
print(f"State size: {state_size}, Action size: {action_size}")

# --- 4. Run the Loop ---
for episode in range(1, NUM_EPISODES + 1):
    state = env.reset()
    total_reward = 0
    
    for step in range(STEPS_PER_EPISODE):
        # 1. Agent chooses an action
        action = agent.choose_action(state)
        
        # 2. Environment executes the action
        next_state, reward, done, info = env.step(action)
        
        # 3. Agent stores experience and learns
        agent.step(state, action, reward, next_state, done)
        
        # 4. Update state and total reward
        state = next_state
        total_reward += reward
        
        if done: # (Our env doesn't really have a 'done', but good to keep)
            break
            
    # --- End of Episode ---
    agent.end_of_episode(episode)
    
    # --- 5. Log Metrics ---
    rewards_window.append(total_reward)
    episode_rewards.append(total_reward)
    episode_wait_times.append(info['total_wait_time'] / (step + 1))
    episode_passengers_delivered.append(info['passengers_delivered'])
    
    # Print stats
    if episode % 20 == 0:
        print(f"Episode: {episode}/{NUM_EPISODES}")
        print(f"  Avg Reward (last 100): {np.mean(rewards_window):.2f}")
        print(f"  Total Reward (this ep): {total_reward:.2f}")
        print(f"  Avg Wait Time (this ep): {info['total_wait_time'] / (step + 1):.2f}")
        print(f"  Passengers Delivered (this ep): {info['passengers_delivered']}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        
# --- 6. Save the trained model ---
agent.save("elevator_dqn.pth")
print("\nTraining complete. Model saved to 'elevator_dqn.pth'")


# --- 7. Plot Results (for your PPT) ---
def plot_results(scores, label):
    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(label)
    plt.xlabel('Episode #')
    # Plot moving average
    moving_avg = [np.mean(scores[max(0, i-50):i+1]) for i in range(len(scores))]
    plt.plot(np.arange(len(moving_avg)), moving_avg, label='Moving Avg (50 eps)', color='red')
    plt.legend()
    plt.savefig(f"{label.lower().replace(' ', '_')}.png")
    print(f"Saved plot: {label.lower().replace(' ', '_')}.png")

plot_results(episode_rewards, "Total Reward per Episode")
plot_results(episode_wait_times, "Average Wait Time per Step")
plot_results(episode_passengers_delivered, "Passengers Delivered per Episode")

print("All plots saved. Your 'Results' are ready.")
