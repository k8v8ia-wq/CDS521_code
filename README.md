# CDS521_code
This is the code file of our multi-agent ball collision experiment
# Multi-Agent Pursuit-Evasion with MADDPG

This is a PyTorch implementation of the **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** algorithm applied to a 2D pursuit-evasion scenario.

In this environment, one "Adversary" agent learns to chase and capture two "Cooperator" agents. Simultaneously, the Cooperator agents learn to reach their assigned landmarks while actively avoiding the adversary. All agents are trained using the MADDPG framework.

![Demo GIF]([https://i.imgur.com/v82H1vE.png](https://github.com/user-attachments/assets/8a3d5f31-0643-4113-b28b-d7353c307747))

---

## Features

* **MADDPG from Scratch:** A complete implementation of the MADDPG algorithm.
* **Centralized Training, Decentralized Execution:** Features a shared Critic that observes all agents, while each agent maintains its own Actor for independent decision-makaing.
* **Real-time Visualization:** Built with `pygame` to provide a clear, real-time window of the agents' behavior as they train.
* **Live Training Metrics (TensorBoard):** Automatically logs all agent losses, rewards, and Q-values to a `runs/` directory for live monitoring with TensorBoard.
* **Automated Plotting:** Generates and saves `matplotlib` charts of training progress (losses, rewards, Q-values, action distribution) at the end of each episode.
* **Model Checkpointing:** Save and load trained models at any time.

---

Environment Rules
Agents: 1 Red Adversary, 2 Blue Cooperators.

Adversary Goal: Capture the cooperators.

Reward: Receives positive rewards for getting close to cooperators and a large bonus for a "capture" (distance < CAPTURE_THRESHOLD).

Cooperator Goal: Reach their assigned green landmarks.

Reward: Receive positive rewards for getting close to their landmark and a large bonus for "reaching" it.

Penalty: Receive a penalty for being too close to the adversary (within DANGER_THRESHOLD).

MADDPG Implementation
Actor Network: A 3-layer fully-connected network (128 hidden units) with ReLU activations. The output layer uses Tanh to scale the continuous actions (dx, dy) to the range [-1, 1].

Critic Network: A 3-layer fully-connected network (128 hidden units) that takes the concatenated states and actions from all agents as input to estimate the joint Q-value.

Replay Buffer: A single, shared replay buffer (deque) is used for all agents.

Exploration: Ornstein-Uhlenbeck Noise (OUNoise) is added to the actor's actions during training to encourage exploration in the continuous action space.

State & Action Spaces
State Space (per agent): STATE_SIZE = 8

pos_x: Agent's x-coordinate

pos_y: Agent's y-coordinate

target_pos_x: Target's x-coordinate (nearest cooperator for adversary, landmark for cooperator)

target_pos_y: Target's y-coordinate

dist_to_right_wall: Distance to right boundary

dist_to_left_wall: Distance to left boundary

dist_to_top_wall: Distance to top boundary

dist_to_bottom_wall: Distance to bottom boundary

Action Space (per agent): ACTION_SIZE = 2

dx: Desired change in x-velocity

dy: Desired change in y-velocity
