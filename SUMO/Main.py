import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import traci

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

Sumo_config = [
    'sumo-gui',
    '-c', 'cros.sumocfg',
    '--start',
    '--step-length', '0.10',
    '--delay', '500',
    '--lateral-resolution', '0.20'
]

traci.start(Sumo_config)
traci.gui.setSchema("View #0", "real world")

required_files = ["cros.sumocfg", "cros.net.xml", "cros.rou.xml", "cros.add.xml"]
for file in required_files:
    if not os.path.exists(file):
        print(f"Ошибка: файл {file} не найден в {script_dir}!")
        print("Доступные файлы:", os.listdir())
        sys.exit(1)

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

TOTAL_STEPS = 5000

ACTIONS = [0, 1, 2, 3, 4, 5] # The discrete action space (0 = keep phase, 1 = switch phase on one, switch phase on two, ...)

MIN_GREEN_STEPS = 100
last_switch_step = -MIN_GREEN_STEPS

MODEL_LOAD = "traffic_light_model.keras"

def to_array(state_tuple):
    """
    Convert the state tuple into a NumPy array for neural network input.
    """
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))

def load_model(filename=MODEL_LOAD):
    if os.path.exists(filename):
        model = keras.models.load_model(filename)
        print(f"Model was loaded: {filename}")
        return model
    else:
        print(f"Model file {filename} was not found!")
        return None

dqn_model = load_model()

def get_state():
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

def apply_action(action, tls_id="tl"):
    global last_switch_step
    
    if action == 0:
        # Do nothing (keep current phase)
        return
    else:
        program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
        num_phases = len(program.phases)
        next_phase = (get_current_phase(tls_id) + action) % num_phases
        traci.trafficlight.setPhase(tls_id, next_phase)
        # Record when the switch happened
        last_switch_step = current_simulation_step

def get_action_from_policy(state, step):
    if ((step % MIN_GREEN_STEPS != 0)):
        return 0
    else:
        state_array = to_array(state)
        Q_values = dqn_model.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

print("\n=== Starting Simulation of Smart Traffic Lights ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step  # keep this variable for apply_action usage
    
    state = get_state()
    action = get_action_from_policy(state, step)
    apply_action(action)
    
    traci.simulationStep()  # Advance simulation by one step
    
    new_state = get_state()

traci.close()

print("\nOnline Simulation completed.")