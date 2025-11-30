# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization

# Step 1.1: (Additional) Imports for Deep Q-Learning
import tensorflow as tf
from tensorflow import keras
from keras import layers

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  # Static network information (such as reading and analyzing network files)

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo-gui',
    '-c', 'cros.sumocfg',
    '--start',
    '--step-length', '0.10',
    '--delay', '500',
    '--lateral-resolution', '0.20'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

required_files = ["cros.sumocfg", "cros.net.xml", "cros.rou.xml", "cros.add.xml"]
for file in required_files:
    if not os.path.exists(file):
        print(f"Ошибка: файл {file} не найден в {script_dir}!")
        print("Доступные файлы:", os.listdir())
        sys.exit(1)

# -------------------------
# Step 6: Define Variables
# -------------------------

# Variables for RL State (queue lengths from detectors and current phase)
q_db_0 = 0
q_db_1 = 0
q_dl_0 = 0
q_dl_1 = 0
q_dl_2 = 0
q_dr_0 = 0
q_dr_1 = 0
q_dr_2 = 0
q_dt_0 = 0
q_dt_1 = 0
current_phase = 0

# ---- Reinforcement Learning Hyperparameters ----
TOTAL_STEPS = 10000    # The total number of simulation steps for continuous (online) training.

ALPHA = 0.1 # Learning rate (α) between[0, 1]   #If α = 1, you fully replace the old Q-value with the newly computed estimate.
                                                #If α = 0, you ignore the new estimate and never update the Q-value.
GAMMA = 0.9 # Discount factor (γ) between[0, 1] #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
                                                #If γ = 1, the agent cares equally about current and future rewards.
EPSILON = 0.1 # Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random

ACTIONS = [0, 1, 2, 3, 4, 5] # The discrete action space (0 = keep phase, 1 = switch phase on one, switch phase on two, ...)

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

# ---- Settings for Model Managing ----
MODEL_SAVE = "traffic_light_model.keras"
MODEL_LOAD = "traffic_light_model.keras"
build_load_switch = 0 # 0 - build / 1 - load

# -------------------------
# Step 7: Define Functions
# -------------------------

def build_model(state_size, action_size):
    """
    Build a simple feedforward neural network that approximates Q-values.
    """
    model = keras.Sequential()                                 # Feedforward neural network
    model.add(layers.Input(shape=(state_size,)))               # Input layer
    model.add(layers.Dense(32, activation='relu'))             # First hidden layer
    model.add(layers.Dense(32, activation='relu'))             # Second hidden layer
    model.add(layers.Dense(action_size, activation='linear'))  # Output layer
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    return model

def to_array(state_tuple):
    """
    Convert the state tuple into a NumPy array for neural network input.
    """
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))

def save_model(model, filename=MODEL_SAVE):
    model.save(filename)
    print(f"Model was saved: {filename}")

def load_model(filename=MODEL_LOAD):
    if os.path.exists(filename):
        model = keras.models.load_model(filename)
        print(f"Model was loaded: {filename}")
        return model
    else:
        print(f"Model file {filename} was not found!")
        return None

    # Create the DQN model
state_size = 11   # (q_db_0, q_db_1, q_dl_0, q_dl_1, q_dl_2, q_dr_0, q_dr_1, q_dr_2, q_dt_0, q_dt_1, current_phase)
action_size = len(ACTIONS)

	# Building or loading DQN model
if build_load_switch == 0:
    dqn_model = build_model(state_size, action_size)
else:
    dqn_model = load_model()

def get_max_Q_value_of_state(s): #1. Objective Function
    state_array = to_array(s)
    Q_values = dqn_model.predict(state_array, verbose=0)[0]  # shape: (action_size,)
    return np.max(Q_values)

def get_reward(state): #2. Constraint 2 
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    total_queue = sum(x ** 2 for x in state[:-1])  # Exclude the current_phase element
    reward = -float(total_queue)
    return reward

def get_state():  #3&4. Constraint 3 & 4
    global q_db_0, q_db_1, q_dl_0, q_dl_1, q_dl_2, q_dr_0, q_dr_1, q_dr_2, q_dt_0, q_dt_1, current_phase
    
    # Detector IDs for left
    detector_dl_0 = "D_l_0"
    detector_dl_1 = "D_l_1"
    detector_dl_2 = "D_l_2"
    
    # Detector IDs for right
    detector_dr_0 = "D_r_0"
    detector_dr_1 = "D_r_1"
    detector_dr_2 = "D_r_2"
    
    # Detector IDs for top
    detector_dt_0 = "D_t_0"
    detector_dt_1 = "D_t_1"
    
    # Detector IDs for bottom
    detector_db_0 = "D_b_0"
    detector_db_1 = "D_b_1"
    
    # Traffic light ID
    traffic_light_id = "tl"
    
    # Get queue lengths from each detector
    q_dl_0 = get_queue_length(detector_dl_0)
    q_dl_1 = get_queue_length(detector_dl_1)
    q_dl_2 = get_queue_length(detector_dl_2)
    
    q_dr_0 = get_queue_length(detector_dr_0)
    q_dr_1 = get_queue_length(detector_dr_1)
    q_dr_2 = get_queue_length(detector_dr_2)
    
    q_dt_0 = get_queue_length(detector_dt_0)
    q_dt_1 = get_queue_length(detector_dt_1)
    
    q_db_0 = get_queue_length(detector_db_0)
    q_db_1 = get_queue_length(detector_db_1)
    
    # Get current phase index
    current_phase = get_current_phase(traffic_light_id)
    
    return (q_db_0, q_db_1, q_dl_0, q_dl_1, q_dl_2, q_dr_0, q_dr_1, q_dr_2, q_dt_0, q_dt_1, current_phase)

def apply_action(action, tls_id="tl"): #5. Constraint 5
    """
    Executes the chosen action on the traffic light, combining:
      - Min Green Time check
      - Switching to the next phase if allowed
    Constraint #5: Ensure at least MIN_GREEN_STEPS pass before switching again.
    """
    global last_switch_step
    
    if action == 0:
        # Do nothing (keep current phase)
        return
    # Check if minimum green time has passed before switching
    elif current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        num_phases = len(program.phases)
        next_phase = (get_current_phase(tls_id) + action) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        # Record when the switch happened
        last_switch_step = current_simulation_step

def update_Q_table(old_state, action, reward, new_state): #6. Constraint 6
    """
    In DQN, we do a single-step gradient update instead of a table update.
    """
    # 1) Predict current Q-values from old_state (current state)
    old_state_array = to_array(old_state)
    Q_values_old = dqn_model.predict(old_state_array, verbose=0)[0]
    # 2) Predict Q-values for new_state to get max future Q (new state)
    new_state_array = to_array(new_state)
    Q_values_new = dqn_model.predict(new_state_array, verbose=0)[0]
    best_future_q = np.max(Q_values_new)
        
    # 3) Incorporate ALPHA to partially update the Q-value
    Q_values_old[action] = Q_values_old[action] + ALPHA * (reward + GAMMA * best_future_q - Q_values_old[action])
    
    # 4) Train (fit) the DQN on this single sample
    dqn_model.fit(old_state_array, np.array([Q_values_old]), verbose=0)

def get_action_from_policy(state): #7. Constraint 7
    """
    Epsilon-greedy strategy using the DQN's predicted Q-values.
    """
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))

def get_queue_length(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id): #8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)

# -------------------------
# Step 8: Fully Online Continuous Learning Loop
# -------------------------

# Lists to record data for plotting
step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step  # keep this variable for apply_action usage
    
    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)
    
    traci.simulationStep()  # Advance simulation by one step
    
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward
    
    update_Q_table(state, action, reward, new_state)
    
    # Print Q-values for the old_state right after update
    updated_q_vals = dqn_model.predict(to_array(state), verbose=0)[0]

    # Record data every 100 steps
    if step % 1 == 0:
        updated_q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, \
              Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}, Q-values(current_state): {updated_q_vals}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))  # sum of queue lengths

# -------------------------
# Step 9: Close connection between SUMO and Traci
# -------------------------

# Also saving out model
save_model(dqn_model)

traci.close()

# ~~~ Print final model summary (replacing Q-table info) ~~~
print("\nOnline Training completed.")
print("DQN Model Summary:")
dqn_model.summary()

# -------------------------
# Visualization of Results
# -------------------------

# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (DQN): Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.show()

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training (DQN): Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.show()