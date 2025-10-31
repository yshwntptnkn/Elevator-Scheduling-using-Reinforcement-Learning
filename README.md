# Elevator-Scheduling-using-Reinforcement-Learning
README.txt

Project Files

	elevator_env.py: The simulation. Defines the elevator, passengers, and building 	logic. This is where you change rewards or simulation rules.

	dqn_agent.py: The "brain." Defines the neural network (QNetwork) and the agent's 	learning logic (DQNAgent).

	train.py: Script to run for training. It trains the agent using the environment 	and saves the model to elevator_dqn.pth.

	visualize.py: Script to run for visualization. It loads the trained 	elevator_dqn.pth and shows it working in a Pygame window.

	requirements.txt: All the libraries you need.

How to Run

	Step 1: Install Dependencies

	"pip install -r requirements.txt"


	Step 2: Train the Agent

	runs the training and saves model as elevator_dqn.pth.

	"python train.py"


	Step 3: See the Trained Agent Work

	loads saved elevator_dqn.pth and opens pygame window.

	"python visualize.py"
