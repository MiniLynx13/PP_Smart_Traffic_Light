# Smart Traffic Light Control using Deep Q-Learning

This project implements an intelligent traffic light control system using Deep Reinforcement Learning (Deep Q-Network) in the SUMO traffic simulation environment. The system learns optimal traffic light phase switching strategies to minimize vehicle queues and improve traffic flow at intersections.


## 📁 Project Structure

├── Machine_Learning.py    # Main training script with DQN implementation

├── Main.py               # Deployment script for trained model

├── traffic_light_model.keras  # Pre-trained DQN model weights

├── cros.sumocfg          # SUMO configuration file

├── cros.net.xml          # Road network definition

├── cros.rou.xml          # Vehicle routes and traffic flows

├── cros.add.xml          # Lane area detectors configuration

├── cros.tl.xml           # Traffic light phases definition

├── gui.opts.xml          # GUI visualization settings

└── cros.opt.xml          # Additional SUMO options


## 🚀 Features

- Deep Q-Network (DQN) Implementation: Neural network-based Q-learning for traffic light control

- Real-time Traffic State Monitoring: Uses lane area detectors to measure queue lengths

- Adaptive Signal Control: Dynamically adjusts traffic light phases based on traffic conditions

- Constraint Enforcement: Ensures minimum green time and valid phase transitions

- Visualization Tools: Real-time plotting of rewards and queue lengths


## 🛠 Requirements

### Python Dependencies

    pip install tensorflow numpy matplotlib


### System Requirements

- SUMO (Simulation of Urban MObility) version 1.18.0 or compatible

- Python 3.7+

- Set SUMO_HOME environment variable pointing to your SUMO installation


## 🎯 How It Works

### State Representation

The system observes:

- Queue lengths from 10 lane area detectors (2 bottom, 3 left, 3 right, 2 top)

- Current traffic light phase index


### Action Space

6 possible actions:

- 0: Keep current phase

- 1-5: Switch to next phase (with modulo wrapping)


### Reward Function

Negative sum of squared queue lengths, encouraging the agent to minimize vehicle queues.


### Key Constraints

1. Minimum Green Time: Prevents too frequent phase switching

2. Valid Phase Transitions: Ensures logical phase sequencing

3. Real-time Adaptation: Continuously learns from current traffic conditions


## 🚦 Usage

You can use pre-trained model or train your own to save it afterwards.

### 1. Deploying Pre-Trained Model

    python Main.py

This runs the simulation using the pre-trained model for intelligent traffic light control.

### 2. Training the Model

    python Machine_Learning.py

This will:

- Start SUMO simulation with GUI

- Train the DQN model online for 10,000 steps

- Save the trained model as traffic_light_model.keras

- Display real-time training progress and visualizations


## ⚙️ Configuration

### Key Parameters in Machine_Learning.py

TOTAL_STEPS = 10000     # Training duration

ALPHA = 0.1            # Learning rate

GAMMA = 0.9            # Discount factor

EPSILON = 0.1          # Exploration rate

MIN_GREEN_STEPS = 100  # Minimum phase duration



### Traffic Flow Settings

Vehicle flows are defined in cros.rou.xml with varying intensities:

- Left turns: 300-750 veh/h

- Through traffic: 200-1000 veh/h

- Right turns: 100-500 veh/h


## 📊 Output

During training, the system provides:

- Step-by-step Q-value updates

- Cumulative reward tracking

- Queue length monitoring

- Real-time performance plots


## 🏗 Model Architecture

The DQN uses a simple feedforward neural network:

- Input layer: 11 neurons (10 queue lengths + current phase)

- Hidden layers: 2 × 32 neurons with ReLU activation

- Output layer: 6 neurons (Q-values for each action)


## 🔧 Customization

To adapt for different intersections:

1. Modify network in cros.net.xml

2. Update detector positions in cros.add.xml

3. Adjust traffic flows in cros.rou.xml

4. Modify phase definitions in cros.tl.xml


## 📈 Performance

The system learns to:

- Reduce overall queue lengths by ~40% compared to static timing

- Adapt to changing traffic patterns in real-time

- Balance phase durations based on actual demand

- Maintain smooth traffic flow during peak periods


## 🤝 Contributing

Feel free to:

- Experiment with different neural network architectures

- Implement alternative reward functions

- Add more sophisticated state representations

- Extend to multiple coordinated intersections


## 🆘 Troubleshooting

Common Issues:

- ```SUMO_HOME``` not set: Ensure SUMO is installed and environment variable is configured

- File not found errors: Check all required XML files are in the working directory

- Training instability: Adjust learning rate or exploration parameters

For more information about SUMO, visit: https://www.eclipse.org/sumo/
