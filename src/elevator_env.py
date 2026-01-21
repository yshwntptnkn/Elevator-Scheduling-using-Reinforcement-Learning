import random
import numpy as np
from src import config
from collections import deque

class Passenger:
    """A simple class for a passenger."""
    def __init__(self, start_floor, dest_floor):
        self.start_floor = start_floor
        self.dest_floor = dest_floor
        self.wait_time = 0
        self.ride_time = 0

    def get_direction(self):
        return 1 if self.dest_floor > self.start_floor else -1

class Elevator:
    """A simple class for an elevator."""
    def __init__(self, num_floors, capacity=10):
        self.num_floors = num_floors
        self.current_floor = 0  # Start at ground floor
        self.direction = 0  # 0: idle, 1: up, -1: down
        self.capacity = capacity
        self.passengers = []
        self.floor_requests = set() # Floors requested from *inside* the elevator

class ElevatorEnv:
    """
    The main Elevator Simulation Environment for the RL Agent.
    
    Actions:
    - 0: Move Down
    - 1: Stop (and open/close doors to load/unload)
    - 2: Move Up
    """
    def __init__(self, num_floors=10):
        self.num_floors = num_floors
        # Hallway call buttons:
        # self.calls_up[floor] = 1 if 'up' is pressed, 0 otherwise
        self.calls_up = np.zeros(self.num_floors, dtype=int)
        self.calls_down = np.zeros(self.num_floors, dtype=int)
        
        # Passengers waiting at each floor
        # self.waiting_passengers[floor] = [list of Passenger objects]
        self.waiting_passengers = [deque() for _ in range(self.num_floors)]
        
        self.elevator = Elevator(num_floors)
        self.total_wait_time = 0
        self.total_ride_time = 0
        self.passengers_delivered = 0

    def _generate_passenger(self):
        """Randomly generates a new passenger and adds them to the waiting list."""
        start = random.randint(0, self.num_floors - 1)
        dest = random.randint(0, self.num_floors - 1)
        
        if start == dest:
            return # Don't generate a passenger going nowhere
            
        passenger = Passenger(start, dest)
        direction = passenger.get_direction()
        
        self.waiting_passengers[start].append(passenger)
        if direction == 1:
            self.calls_up[start] = 1
        else:
            self.calls_down[start] = 1

    def get_state(self):
        """
        Returns the current state representation for the RL agent.
        This is a critical part of your "Methodology".
        """
        # A simple state:
        # 1. Elevator's current floor (1 number)
        # 2. Elevator's direction (1 number)
        # 3. Internal floor requests (num_floors numbers)
        # 4. Hallway 'up' calls (num_floors numbers)
        # 5. Hallway 'down' calls (num_floors numbers)
        
        internal_requests = np.zeros(self.num_floors, dtype=int)
        for floor in self.elevator.floor_requests:
            internal_requests[floor] = 1
            
        state = [
            self.elevator.current_floor,
            self.elevator.direction
        ]
        state = np.concatenate([
            state,
            internal_requests,
            self.calls_up,
            self.calls_down
        ]).astype(float)
        
        # You could also use a more complex state, like a hashable tuple
        # for Q-table keys, but a NumPy array is good for NNs.
        return state

    def reset(self):
        """Resets the environment to a starting state."""
        self.calls_up.fill(0)
        self.calls_down.fill(0)
        self.waiting_passengers = [deque() for _ in range(self.num_floors)]
        self.elevator = Elevator(self.num_floors)
        
        self.total_wait_time = 0
        self.total_ride_time = 0
        self.passengers_delivered = 0
        
        # Generate a few starting passengers
        for _ in range(5):
            self._generate_passenger()
            
        return self.get_state()

    def step(self, action):
        """
        Runs one time-step of the simulation based on the agent's action.
        
        Returns:
        - next_state: The state after the action is taken.
        - reward: The reward (or penalty) for this step.
        - done: (Always False for a continuous simulation, but can be modified)
        - info: A dictionary for metrics (e.g., total wait time).
        """
        
        # --- 1. Initialize Reward ---
        reward = 0
        
        # --- 2. Apply Action ---
        if action == 0: # Move Down
            self.elevator.direction = -1
            if self.elevator.current_floor > 0:
                self.elevator.current_floor -= 1
            reward -= config.STATE_MOVE_PENALTY
            # Penalize moving wrong way
            if self.calls_up[self.elevator.current_floor] or \
               any(p.get_direction() == 1 for p in self.elevator.passengers):
                reward -= config.STATE_WRONG_DIR_PENALTY

        elif action == 1: # Stop and Load/Unload
            self.elevator.direction = 0 # Set to idle
            
            # --- Unload Passengers ---
            unloaded_passengers = []
            for p in self.elevator.passengers:
                if p.dest_floor == self.elevator.current_floor:
                    reward += config.STATE_DROPOFF_REWARD # Big reward!
                    self.passengers_delivered += 1
                else:
                    unloaded_passengers.append(p)
            self.elevator.passengers = unloaded_passengers
            self.elevator.floor_requests.discard(self.elevator.current_floor)

            # --- Load Passengers ---
            waiting_now = self.waiting_passengers[self.elevator.current_floor]
            passengers_to_board = []
            passengers_still_waiting = deque()
            
            # Check 'up' calls
            if self.calls_up[self.elevator.current_floor]:
                for p in waiting_now:
                    if p.get_direction() == 1 and len(self.elevator.passengers) < self.elevator.capacity:
                        passengers_to_board.append(p)
                    else:
                        passengers_still_waiting.append(p)
                # If we picked up 'up' passengers, clear the call
                if passengers_to_board: 
                    self.calls_up[self.elevator.current_floor] = 0
                waiting_now = passengers_still_waiting
                passengers_still_waiting = deque()

            # Check 'down' calls
            if self.calls_down[self.elevator.current_floor]:
                for p in waiting_now:
                    if p.get_direction() == -1 and len(self.elevator.passengers) < self.elevator.capacity:
                        passengers_to_board.append(p)
                    else:
                        passengers_still_waiting.append(p)
                if passengers_to_board: # This logic is simple, assumes we take all
                    self.calls_down[self.elevator.current_floor] = 0
                waiting_now = passengers_still_waiting
            
            # Board the chosen passengers
            for p in passengers_to_board:
                self.elevator.passengers.append(p)
                self.elevator.floor_requests.add(p.dest_floor)
            
            self.waiting_passengers[self.elevator.current_floor] = waiting_now

        elif action == 2: # Move Up
            self.elevator.direction = 1
            if self.elevator.current_floor < self.num_floors - 1:
                self.elevator.current_floor += 1
            reward -= config.STATE_MOVE_PENALTY
            # Penalize moving wrong way
            if self.calls_down[self.elevator.current_floor] or \
               any(p.get_direction() == -1 for p in self.elevator.passengers):
                reward -= config.STATE_WRONG_DIR_PENALTY
                
        # --- 3. Update Timers and Penalties ---
        
        # Penalize passengers for riding
        for p in self.elevator.passengers:
            p.ride_time += 1
            reward -= config.STATE_RIDE_TIME
            
        # Penalize passengers for waiting
        for floor_queue in self.waiting_passengers:
            for p in floor_queue:
                p.wait_time += 1
                reward -= config.STATE_WAIT_TIME
                reward -= (p.wait_time ** 2) * 0.0001 #squared pen

        # Update total metrics
        self.total_wait_time += sum(len(q) for q in self.waiting_passengers)
        self.total_ride_time += len(self.elevator.passengers)

        # --- 4. Randomly add a new passenger ---
        if random.random() < 0.1: # 10% chance each step
            self._generate_passenger()
            
        # --- 5. Return Results ---
        next_state = self.get_state()
        done = False # This simulation runs forever
        info = {
            "total_wait_time": self.total_wait_time,
            "total_ride_time": self.total_ride_time,
            "passengers_delivered": self.passengers_delivered
        }
        
        return next_state, reward, done, info

    def render(self):
        """A simple text-based visualization of the environment."""
        print(f"\n--- Step ---")
        for f in range(self.num_floors - 1, -1, -1):
            # Elevator
            e_str = "| E |" if self.elevator.current_floor == f else "|   |"
            
            # Passengers in elevator
            p_in = len(self.elevator.passengers)
            e_str = f"|{str(p_in).center(3)}|" if self.elevator.current_floor == f else "|   |"
            
            if self.elevator.current_floor == f:
                if self.elevator.direction == 1: e_str = "| ^ |"
                elif self.elevator.direction == -1: e_str = "| v |"
                elif self.elevator.direction == 0: e_str = "|[ ]|"
                
            # Calls
            up = "U" if self.calls_up[f] else " "
            down = "D" if self.calls_down[f] else " "
            
            # Waiting
            wait = len(self.waiting_passengers[f])
            
            print(f"Floor {f:2} [{up} {down}] {e_str} Waiting: {wait}")
        
        print(f"Internal Requests: {sorted(list(self.elevator.floor_requests))}")
        print(f"Passengers Riding: {len(self.elevator.passengers)}")


# --- Main block to test the environment ---
if __name__ == "__main__":
    env = ElevatorEnv(num_floors=10)
    state = env.reset()
    
    total_reward = 0
    
    # Run for 100 steps with a random agent
    for step in range(100):
        # Your RL Agent will go here.
        # For now, we use a "random" agent.
        action = random.randint(0, 2) # 0:Down, 1:Stop, 2:Up
        
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        
        # env.render() # Uncomment this to see the text visualization
        
        if (step + 1) % 20 == 0:
            print(f"\n--- Step {step + 1} ---")
            print(f"Action: {['Down', 'Stop', 'Up'][action]}")
            print(f"Total Reward: {total_reward:.2f}")
            print(f"Info: {info}")
            env.render() # Render every 20 steps
            
        state = next_state

