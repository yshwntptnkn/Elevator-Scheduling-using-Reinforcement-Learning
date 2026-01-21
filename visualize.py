import pygame
import sys
import torch
import numpy as np
import src.config
from src.elevator_env import ElevatorEnv  # Import your environment
from src.dqn_agent import DQNAgent        # Import your agent
import os

# --- 1. Setup ---
pygame.init()
pygame.font.init()

# --- 2. Constants ---
SCREEN_HEIGHT = config.NUM_FLOORS * config.FLOOR_HEIGHT
BUILDING_PANEL_WIDTH = 350
ELEVATOR_SHAFT_WIDTH = 100
INFO_PANEL_WIDTH = config.SCREEN_WIDTH - config.BUILDING_PANEL_WIDTH

# Colors
COLORS = {
    'bg': (20, 30, 40),
    'text': (220, 220, 220),
    'shaft': (10, 10, 10),
    'floor_line': (50, 60, 70),
    'elevator': (214, 122, 127),
    'elevator_text': (0, 0, 0),
    'btn_off': (60, 70, 80),
    'btn_on': (46, 204, 113),
    'wait_text': (241, 196, 15)
}

# Fonts
try:
    FONT_S = pygame.font.SysFont('Arial', 16)
    FONT_M = pygame.font.SysFont('Arial', 18, bold=True)
    FONT_L = pygame.font.SysFont('Arial', 24, bold=True)
except IOError:
    FONT_S = pygame.font.SysFont(None, 18)
    FONT_M = pygame.font.SysFont(None, 22)
    FONT_L = pygame.font.SysFont(None, 28)

# --- 3. Helper Functions ---

def get_floor_y(floor):
    """Calculates the Y-coordinate for a floor (Pygame Y is inverted)"""
    return SCREEN_HEIGHT - (floor + 1) * FLOOR_HEIGHT

def draw_text(surface, text, pos, font, color, center=False):
    """Helper to draw text"""
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    if center:
        text_rect.center = pos
    else:
        text_rect.topleft = pos
    surface.blit(text_surface, text_rect)

def draw_building(surface, env):
    """Draws the floors, buttons, and waiting passengers."""
    
    # Draw shaft background
    shaft_x = (BUILDING_PANEL_WIDTH - ELEVATOR_SHAFT_WIDTH) // 2
    pygame.draw.rect(surface, COLORS['shaft'], (shaft_x, 0, ELEVATOR_SHAFT_WIDTH, SCREEN_HEIGHT))
    
    for f in range(env.num_floors):
        floor_y = get_floor_y(f)
        
        # Draw floor line
        pygame.draw.line(surface, COLORS['floor_line'], (0, floor_y), (BUILDING_PANEL_WIDTH, floor_y), 2)
        
        # Draw floor label
        draw_text(surface, f"FLOOR {f}", (10, floor_y + 5), FONT_S, COLORS['text'])
        
        # --- Draw Call Buttons (Visual) ---
        btn_x = 100
        btn_y_up = floor_y + 20
        btn_y_down = floor_y + 40
        
        # Up button
        up_color = COLORS['btn_on'] if env.calls_up[f] else COLORS['btn_off']
        pygame.draw.polygon(surface, up_color, [(btn_x, btn_y_up), (btn_x + 10, btn_y_up), (btn_x + 5, btn_y_up - 8)])
        
        # Down button
        down_color = COLORS['btn_on'] if env.calls_down[f] else COLORS['btn_off']
        pygame.draw.polygon(surface, down_color, [(btn_x, btn_y_down), (btn_x + 10, btn_y_down), (btn_x + 5, btn_y_down + 8)])

        # --- Draw Waiting Passengers ---
        wait_count = len(env.waiting_passengers[f])
        if wait_count > 0:
            draw_text(surface, f"WAITING: {wait_count}", (btn_x + 25, floor_y + 25), FONT_M, COLORS['wait_text'])

def draw_elevator(surface, env, action_str):
    """Draws the elevator car, passenger count, and current action."""
    e = env.elevator
    shaft_x = (BUILDING_PANEL_WIDTH - ELEVATOR_SHAFT_WIDTH) // 2
    
    # Calculate elevator position
    e_floor_y = get_floor_y(e.current_floor)
    e_rect = (shaft_x + 5, e_floor_y + 5, ELEVATOR_SHAFT_WIDTH - 10, FLOOR_HEIGHT - 10)
    
    # Draw elevator car
    pygame.draw.rect(surface, COLORS['elevator'], e_rect, border_radius=4)
    
    # --- FIX IS HERE ---
    # Calculate the center (x, y) coordinate of the rectangle
    center_x = e_rect[0] + e_rect[2] // 2
    center_y = e_rect[1] + e_rect[3] // 2
    center_pos = (center_x, center_y)
    
    # Draw passenger count
    # Pass the (x, y) center_pos tuple, not the (x, y, w, h) e_rect tuple
    draw_text(surface, str(len(e.passengers)), center_pos, FONT_L, COLORS['elevator_text'], center=True)
    
    # Draw action text
    draw_text(surface, action_str, (e_rect[0] + e_rect[2] // 2, e_rect[1] - 10), FONT_S, COLORS['elevator'], center=True)
    
def draw_info_panel(surface, env, action_str, total_steps, avg_wait_time, avg_ride_time):
    """Draws the separate panel on the right with all text info."""
    panel_x = BUILDING_PANEL_WIDTH
    
    # Panel background
    pygame.draw.rect(surface, (10, 15, 20), (panel_x, 0, INFO_PANEL_WIDTH, SCREEN_HEIGHT))
    
    # Title
    draw_text(surface, "ELEVATOR SIMULATION", (panel_x + INFO_PANEL_WIDTH // 2, 30), FONT_L, COLORS['text'], center=True)
    
    # --- Info ---
    y_pos = 80
    info_x = panel_x + 20
    
    def draw_info(label, value):
        nonlocal y_pos
        draw_text(surface, f"{label}:", (info_x, y_pos), FONT_M, COLORS['text'])
        draw_text(surface, str(value), (info_x + 200, y_pos), FONT_M, COLORS['wait_text'])
        y_pos += 30

    draw_info("Model", "elevator_dqn.pth") # <-- Changed
    y_pos += 20
    
    draw_info("Current Action", action_str)
    draw_info("Current Floor", env.elevator.current_floor)
    draw_info("Passengers Inside", len(env.elevator.passengers))
    y_pos += 20
    
    draw_info("Total Delivered", env.passengers_delivered)
    draw_info("Total Sim Steps", total_steps)
    draw_info("Avg Wait Time", f"{avg_wait_time:.2f} steps")
    draw_info("Avg Ride Time", f"{avg_ride_time:.2f} steps")
    
    # --- Internal Requests ---
    y_pos += 30
    draw_text(surface, "Internal Requests:", (info_x, y_pos), FONT_M, COLORS['text'])
    req_str = ", ".join(map(str, sorted(list(env.elevator.floor_requests))))
    if not req_str:
        req_str = "None"
    
    # Handle long request lists by wrapping text
    max_width = INFO_PANEL_WIDTH - 40
    words = req_str.split(', ')
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + word + ", "
        if FONT_S.size(test_line)[0] < max_width:
            current_line = test_line
        else:
            lines.append(current_line.rstrip(', '))
            current_line = word + ", "
    lines.append(current_line.rstrip(', '))
    
    y_pos += 30
    for line in lines:
        draw_text(surface, line, (info_x, y_pos), FONT_S, COLORS['wait_text'])
        y_pos += 20


# --- 4. Main Execution ---
def main():
    # --- Initialize Env and Agent ---
    env = ElevatorEnv(num_floors=config.NUM_FLOORS)
    state_size = env.get_state().shape[0]
    action_size = 3
    
    agent = DQNAgent(state_size=state_size, action_size=action_size)
    
    # --- Load the TRAINED Model ---
    possible_paths = [
        os.path.join("models", "elevator_dqn.pth"), # Priority: clean models folder
        "elevator_dqn.pth",                         # Fallback: root directory
        os.path.join(os.path.dirname(__file__), "models", "elevator_dqn.pth") # Absolute path safety
    ]
    model_load_path = None
    for path in possible_paths:
    if os.path.exists(path):
        model_load_path = path
        break

    if model_load_path is None:
        print("\n❌ ERROR: Could not find model file 'elevator_dqn.pth'.")
        print(f"   Checked locations: {', '.join(possible_paths)}")
        print("   Please run 'train.py' first to generate the model file.\n")
        sys.exit()

    print(f"✅ Loading model from: {model_load_path}")
    
    try:
        # Load model, mapping to CPU (works for both CPU and GPU-trained models)
        agent.q_network.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    except FileNotFoundError:
        print(f"ERROR: Could not find model at '{model_load_path}'.") # <-- Changed
        print("Please run 'train.py' first.")
        sys.exit()
        
    # --- THIS IS THE FIRST CORRECTED LINE ---
    agent.q_network.eval() 
    # ----------------------------------
    
    print(f"Loaded model from '{model_load_path}'. Starting visualization...")

    # --- Initialize Pygame ---
    screen = pygame.display.set_mode((config.SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Elevator Scheduling Visualization")
    clock = pygame.time.Clock()

    # --- Simulation Loop ---
    state = env.reset()
    action = 1 # Start with a "Stop" action
    action_str = "STOPPED"
    
    total_steps = 0
    
    running = True
    while running:
        # --- Handle Events ---
        # --- THIS IS THE SECOND CORRECTED LINE ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # --- 1. Agent Chooses Best Action ---
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = agent.q_network(state_tensor)
        action = np.argmax(action_values.cpu().data.numpy())
        
        if action == 0: action_str = "v MOVING DOWN v"
        elif action == 1: action_str = "[ STOPPED ]"
        elif action == 2: action_str = "^ MOVING UP ^"

        # --- 2. Environment Steps ---
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_steps += 1
        
        # --- 3. Calculate Live Metrics ---
        # This is the corrected logic for live averages
        avg_wait_time = info['total_wait_time'] / total_steps if total_steps > 0 else 0
        avg_ride_time = info['total_ride_time'] / total_steps if total_steps > 0 else 0
        
        if info['passengers_delivered'] > 0:
            # A more accurate metric: avg time per *delivered* passenger
            # We can't get this easily without modifying the env, 
            # so we'll stick to the simpler overall average for now.
            pass

        # --- 4. Draw Everything ---
        screen.fill(COLORS['bg'])
        draw_building(screen, env)
        draw_elevator(screen, env, action_str)
        draw_info_panel(screen, env, action_str, total_steps, avg_wait_time, avg_ride_time)
        
        # --- 5. Update Display ---
        pygame.display.flip()
        
        # --- 6. Control Speed (5 ticks per second) ---
        clock.tick(2) 

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()


