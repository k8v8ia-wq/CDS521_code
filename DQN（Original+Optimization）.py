import pygame
import numpy as np
import time
import math
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

print("=" * 60)
print("Multi-Agent Environment - Pursuit Evasion Game with DQN")
print("=" * 60)

# Initialize pygame
pygame.init()

# Set up window
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Multi-Agent Environment - Pursuit Evasion Game with DQN")

# Color definitions
BACKGROUND = (240, 240, 240)
RED = (255, 80, 80)  # Adversary
BLUE = (80, 80, 255)  # Cooperator
GREEN = (80, 180, 80)  # Landmark
YELLOW = (255, 200, 0)  # Highlight
PURPLE = (180, 80, 200)  # Interaction effect
ORANGE = (255, 150, 50)  # Warning
BLACK = (40, 40, 40)
GRAY = (200, 200, 200)
LIGHT_GRAY = (220, 220, 220)  # For finer grid
WHITE = (255, 255, 255)
DARK_RED = (180, 0, 0)  # DQN agent highlight

# Fonts
font_large = pygame.font.Font(None, 32)
font_medium = pygame.font.Font(None, 24)
font_small = pygame.font.Font(None, 18)

# Environment parameters
SCALE = 180
CENTER_X = WIDTH // 3
CENTER_Y = HEIGHT // 2

# Size parameters
AGENT_RADIUS = 8
LANDMARK_RADIUS = 6
DANGER_THRESHOLD = 0.4
TARGET_THRESHOLD = 0.08
CAPTURE_THRESHOLD = 0.15

# UI parameters
INFO_PANEL_WIDTH = 400
INFO_PANEL_HEIGHT = 600
INFO_PANEL_MARGIN = 20
MINIMAP_HEIGHT = 150

# Movement speeds - adjusted for better visualization
COOPERATOR_SPEED = 0.018  # Significantly slower for more challenging gameplay
ADVERSARY_SPEED = COOPERATOR_SPEED * 1.8  # Increased ratio to make chase more intense

# DQN Parameters - adjusted for increased difficulty
BATCH_SIZE = 64 # Increased batch size for more stable learning
LEARNING_RATE = 0.0003  # Reduced learning rate for better convergence with harder gameplay
GAMMA = 0.96  # Slightly reduced discount factor to prioritize immediate rewards
EPSILON_START = 1.0
EPSILON_END = 0.02  # Slightly higher exploration rate for more challenging environment
EPSILON_DECAY = 0.996  # Slower decay to maintain exploration longer
TARGET_UPDATE = 10  # Less frequent updates for more stable target network
MEMORY_SIZE = 40000  # Increased memory capacity for more diverse experiences

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DQN(nn.Module):
    """Deep Q-Network for the adversary agent"""

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    """DQN Agent for the adversary"""

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.learning_step = 0

        # Neural Networks
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

        # Initialize target network with policy network weights
        self.update_target_network()

        # Training statistics
        self.losses = []
        self.rewards = []
        self.episode_rewards = []
        self.average_rewards = []  # 平均回合奖励
        self.q_values = []  # Q值记录
        self.epsilons = []  # 探索率记录
        self.td_errors = []  # 时序差分误差记录
        self.action_counts = np.zeros(action_size)  # 动作分布记录

    def update_target_network(self):
        """Update the target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Select action using epsilon-greedy policy"""
        # Ensure state is the correct size
        if len(state) != self.state_size:
            state = state[:self.state_size]  # Truncate if too long
            if len(state) < self.state_size:
                # Pad if too short
                state = np.pad(state, (0, self.state_size - len(state)), 'constant')

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            # 记录平均Q值
            avg_q = q_values.mean().item()
            self.q_values.append(avg_q)

            if random.random() <= self.epsilon:
                action = random.randrange(self.action_size)
            else:
                action = np.argmax(q_values.cpu().data.numpy())

            # 记录动作分布
            self.action_counts[action] += 1
            return action

    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < BATCH_SIZE:
            return 0

        # Sample batch from memory
        batch = random.sample(self.memory, BATCH_SIZE)

        # Process states to ensure correct dimensions
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for state, action, reward, next_state, done in batch:
            # Ensure state dimensions are correct
            if len(state) != self.state_size:
                state = state[:self.state_size]
                if len(state) < self.state_size:
                    state = np.pad(state, (0, self.state_size - len(state)), 'constant')

            if len(next_state) != self.state_size:
                next_state = next_state[:self.state_size]
                if len(next_state) < self.state_size:
                    next_state = np.pad(next_state, (0, self.state_size - len(next_state)), 'constant')

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        dones = torch.BoolTensor(dones).to(device)

        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            next_q_values[dones] = 0.0

        # Compute target Q values
        target_q_values = rewards + (GAMMA * next_q_values)

        # Compute TD error
        with torch.no_grad():
            td_error = target_q_values.unsqueeze(1) - current_q_values
            # 计算平均TD误差并记录
            avg_td_error = td_error.abs().mean().item()
            self.td_errors.append(avg_td_error)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.epsilons.append(self.epsilon)

        # Update target network periodically
        self.learning_step += 1
        if self.learning_step % TARGET_UPDATE == 0:
            self.update_target_network()

        # Store loss for tracking
        self.losses.append(loss.item())
        return loss.item()

    def save_model(self, filename):
        """Save the model weights"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'memory': list(self.memory),
            'losses': self.losses,
            'episode_rewards': self.episode_rewards,
            'average_rewards': self.average_rewards,
            'q_values': self.q_values,
            'epsilons': self.epsilons,
            'td_errors': self.td_errors,
            'action_counts': self.action_counts
        }, filename)

    def load_model(self, filename):
        """Load the model weights"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.memory = deque(checkpoint['memory'], maxlen=MEMORY_SIZE)
        self.losses = checkpoint['losses']
        self.episode_rewards = checkpoint['episode_rewards']
        self.average_rewards = checkpoint.get('average_rewards', [])
        self.q_values = checkpoint.get('q_values', [])
        self.epsilons = checkpoint.get('epsilons', [])
        self.td_errors = checkpoint.get('td_errors', [])
        self.action_counts = checkpoint.get('action_counts', np.zeros(self.action_size))


def create_simulation_data():
    """Create simulation data"""
    agents = [
        {'pos': [-1.0, 0.6], 'type': 'adversary', 'target_landmark': None, 'score': 0},
        {'pos': [0.3, -0.8], 'type': 'cooperator', 'target_landmark': 0, 'score': 0},
        {'pos': [0.8, 0.2], 'type': 'cooperator', 'target_landmark': 1, 'score': 0}
    ]
    landmarks = [
        {'pos': [-1.2, -1.0], 'occupied': False, 'value': 10},
        {'pos': [1.1, 1.0], 'occupied': False, 'value': 10}
    ]
    return agents, landmarks


def get_random_position():
    """Generate random position within world bounds"""
    WORLD_BOUNDS = 1.4
    x = np.random.uniform(-WORLD_BOUNDS, WORLD_BOUNDS)
    y = np.random.uniform(-WORLD_BOUNDS, WORLD_BOUNDS)
    return [x, y]


def get_state(adversary_pos, cooperator1_pos, cooperator2_pos, landmarks):
    """Get state representation for DQN - Fixed to 15 dimensions"""
    state = []

    # Adversary position (normalized) - 2 dimensions
    state.extend(adversary_pos)

    # Relative positions to cooperators - 4 dimensions
    state.extend([cooperator1_pos[0] - adversary_pos[0], cooperator1_pos[1] - adversary_pos[1]])
    state.extend([cooperator2_pos[0] - adversary_pos[0], cooperator2_pos[1] - adversary_pos[1]])

    # Distances to cooperators - 2 dimensions
    state.append(calculate_distance(adversary_pos, cooperator1_pos))
    state.append(calculate_distance(adversary_pos, cooperator2_pos))

    # Relative positions to landmarks - 4 dimensions (only 2 landmarks)
    for i, landmark in enumerate(landmarks[:2]):  # Ensure only 2 landmarks
        state.extend([landmark['pos'][0] - adversary_pos[0], landmark['pos'][1] - adversary_pos[1]])

    # Distances to landmarks - 2 dimensions
    for i, landmark in enumerate(landmarks[:2]):  # Ensure only 2 landmarks
        state.append(calculate_distance(adversary_pos, landmark['pos']))

    # Which cooperator is closer to landmarks (threat assessment) - 1 dimension
    coop1_min_landmark_dist = min(calculate_distance(cooperator1_pos, landmark['pos']) for landmark in landmarks)
    coop2_min_landmark_dist = min(calculate_distance(cooperator2_pos, landmark['pos']) for landmark in landmarks)

    # Use difference instead of both values to save dimensions
    threat_difference = coop1_min_landmark_dist - coop2_min_landmark_dist
    state.append(threat_difference)

    # Ensure exactly 15 dimensions
    if len(state) != 15:
        # Truncate or pad to exactly 15 dimensions
        state = state[:15]
        if len(state) < 15:
            state.extend([0] * (15 - len(state)))

    return np.array(state, dtype=np.float32)


def calculate_distance(pos1, pos2):
    """Calculate distance between two positions"""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def world_to_screen(pos):
    """Convert world coordinates to screen coordinates"""
    x = int(pos[0] * SCALE + CENTER_X)
    y = int(pos[1] * SCALE + CENTER_Y)
    return x, y


def draw_grid():
    """Draw grid"""
    grid_right_boundary = WIDTH - INFO_PANEL_WIDTH - 10

    # Draw dense grid lines
    dense_grid_range = 40
    for i in range(-dense_grid_range, dense_grid_range + 1):
        x = CENTER_X + i * (SCALE / 10)
        pygame.draw.line(screen, LIGHT_GRAY, (x, 50), (x, HEIGHT - 50), 1)
        y = CENTER_Y + i * (SCALE / 10)
        pygame.draw.line(screen, LIGHT_GRAY, (50, y), (grid_right_boundary, y), 1)

    # Draw main grid lines
    main_grid_range = 4
    for i in range(-main_grid_range, main_grid_range + 1):
        x = CENTER_X + i * SCALE
        pygame.draw.line(screen, GRAY, (x, 50), (x, HEIGHT - 50), 2)
        y = CENTER_Y + i * SCALE
        pygame.draw.line(screen, GRAY, (50, y), (grid_right_boundary, y), 2)

    # Draw axes
    pygame.draw.line(screen, BLACK, (CENTER_X, 50), (CENTER_X, HEIGHT - 50), 3)
    pygame.draw.line(screen, BLACK, (50, CENTER_Y), (grid_right_boundary, CENTER_Y), 3)

    # Draw coordinate labels
    for i in range(-main_grid_range, main_grid_range + 1):
        if i != 0:
            x = CENTER_X + i * SCALE
            label_value = i * 10
            label = font_small.render(str(label_value), True, BLACK)
            screen.blit(label, (x - 8, CENTER_Y + 12))

            y = CENTER_Y + i * SCALE
            label = font_small.render(str(label_value), True, BLACK)
            screen.blit(label, (CENTER_X + 12, y - 8))

    # Draw origin label
    origin_label = font_small.render("0", True, BLACK)
    screen.blit(origin_label, (CENTER_X + 5, CENTER_Y + 12))


def draw_landmark(pos, index, is_highlighted=False, is_occupied=False):
    """Draw landmark"""
    x, y = world_to_screen(pos)

    if is_occupied:
        color = PURPLE
    elif is_highlighted:
        color = YELLOW
    else:
        color = GREEN

    pygame.draw.circle(screen, color, (x, y), LANDMARK_RADIUS)
    pygame.draw.circle(screen, WHITE, (x, y), LANDMARK_RADIUS // 2)

    text = font_small.render(f"LM{index + 1}", True, BLACK)
    screen.blit(text, (x - 12, y - LANDMARK_RADIUS - 12))


def draw_agent(pos, is_adversary, agent_id, target_distance=None, is_success=False, is_danger=False,
               is_dqn_agent=False):
    """Draw agent"""
    x, y = world_to_screen(pos)

    if is_success:
        color = PURPLE
    elif is_danger:
        color = ORANGE
    elif is_dqn_agent:
        color = DARK_RED
    else:
        color = RED if is_adversary else BLUE

    radius = AGENT_RADIUS + 2 if is_adversary else AGENT_RADIUS
    pygame.draw.circle(screen, color, (x, y), radius)
    pygame.draw.circle(screen, WHITE, (x, y), radius, 2)

    # Add special indicator for DQN agent
    if is_dqn_agent:
        pygame.draw.circle(screen, YELLOW, (x, y), radius + 4, 2)

    # Agent label
    agent_type = "Adv" if is_adversary else f"C{agent_id}"
    text = font_small.render(agent_type, True, BLACK)
    screen.blit(text, (x - 8, y - radius - 10))


def draw_danger_zone(agent_pos, adversary_pos):
    """Draw danger zone"""
    if calculate_distance(agent_pos, adversary_pos) < DANGER_THRESHOLD:
        x1, y1 = world_to_screen(agent_pos)
        x2, y2 = world_to_screen(adversary_pos)
        pygame.draw.line(screen, ORANGE, (x1, y1), (x2, y2), 2)


def draw_dqn_panel(step, rewards, total_rewards, game_status, dqn_agent, current_state, current_action,
                   success_count, danger_count, capture_count, episode):
    """Draw DQN information panel"""
    panel_x = WIDTH - INFO_PANEL_WIDTH - INFO_PANEL_MARGIN
    panel_y = INFO_PANEL_MARGIN

    # Panel background
    pygame.draw.rect(screen, WHITE, (panel_x, panel_y, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT))
    pygame.draw.rect(screen, BLACK, (panel_x, panel_y, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT), 2)

    # Title
    title = font_large.render("DQN Pursuit Evasion", True, BLACK)
    screen.blit(title, (panel_x + 10, panel_y + 10))

    # Episode and step information
    episode_text = font_medium.render(f"Episode: {episode}", True, BLACK)
    screen.blit(episode_text, (panel_x + 10, panel_y + 50))

    step_text = font_medium.render(f"Step: {step}/300", True, BLACK)
    screen.blit(step_text, (panel_x + 150, panel_y + 50))

    # Game status
    status_text = font_medium.render(f"Status: {game_status}", True,
                                     (0, 150, 0) if "Leading" in game_status else
                                     ORANGE if "Danger" in game_status or "Captured" in game_status else
                                     (0, 150, 0) if "Success" in game_status else BLACK)
    screen.blit(status_text, (panel_x + 10, panel_y + 80))

    # DQN Agent Information
    dqn_y = panel_y + 110
    dqn_title = font_medium.render("DQN Adversary Info:", True, DARK_RED)
    screen.blit(dqn_title, (panel_x + 10, dqn_y))

    epsilon_text = font_small.render(f"Epsilon: {dqn_agent.epsilon:.3f}", True, BLACK)
    screen.blit(epsilon_text, (panel_x + 20, dqn_y + 25))

    memory_text = font_small.render(f"Memory: {len(dqn_agent.memory)}/{MEMORY_SIZE}", True, BLACK)
    screen.blit(memory_text, (panel_x + 20, dqn_y + 45))

    action_text = font_small.render(f"Action: {get_action_description(current_action)}", True, BLACK)
    screen.blit(action_text, (panel_x + 20, dqn_y + 65))

    learning_text = font_small.render(f"Learning Steps: {dqn_agent.learning_step}", True, BLACK)
    screen.blit(learning_text, (panel_x + 20, dqn_y + 85))

    avg_loss = np.mean(dqn_agent.losses[-100:]) if dqn_agent.losses else 0
    loss_text = font_small.render(f"Avg Loss: {avg_loss:.4f}", True, BLACK)
    screen.blit(loss_text, (panel_x + 20, dqn_y + 105))

    # Statistics
    stats_y = dqn_y + 135
    stats_title = font_medium.render("Statistics:", True, BLACK)
    screen.blit(stats_title, (panel_x + 10, stats_y))

    success_text = font_small.render(f"Success: {success_count}", True, (0, 150, 0))
    screen.blit(success_text, (panel_x + 20, stats_y + 30))

    danger_count_text = font_small.render(f"Danger: {danger_count}", True, ORANGE)
    screen.blit(danger_count_text, (panel_x + 20, stats_y + 50))

    capture_text = font_small.render(f"Captures: {capture_count}", True, RED)
    screen.blit(capture_text, (panel_x + 20, stats_y + 70))

    # Reward information
    rewards_y = stats_y + 110
    rewards_title = font_medium.render("Rewards:", True, BLACK)
    screen.blit(rewards_title, (panel_x + 10, rewards_y))

    agent_names = ["Adversary (DQN)", "Cooperator1", "Cooperator2"]
    for i, (name, reward, total) in enumerate(zip(agent_names, rewards, total_rewards)):
        color = DARK_RED if i == 0 else BLUE
        text = f"{name}: {reward:+.2f}"
        reward_text = font_small.render(text, True, color)
        screen.blit(reward_text, (panel_x + 20, rewards_y + 30 + i * 20))

        total_text = font_small.render(f"Total: {total:+.2f}", True, color)
        screen.blit(total_text, (panel_x + 180, rewards_y + 30 + i * 20))

    # Victory condition
    victory_y = rewards_y + 110
    adv_abs = abs(total_rewards[0])
    coop_total = total_rewards[1] + total_rewards[2]
    condition_text = font_medium.render("Victory Condition:", True, BLACK)
    screen.blit(condition_text, (panel_x + 10, victory_y))

    condition_details = font_small.render("Coop Total > Adv Abs", True, BLACK)
    screen.blit(condition_details, (panel_x + 20, victory_y + 30))

    condition_math = font_small.render(f"{coop_total:.2f} > {adv_abs:.2f}", True,
                                       (0, 150, 0) if coop_total > adv_abs else RED)
    screen.blit(condition_math, (panel_x + 20, victory_y + 50))

    # Game Legend
    legend_y = victory_y + 90
    legend_title = font_medium.render("Legend", True, BLACK)
    screen.blit(legend_title, (panel_x + 10, legend_y))

    items = [
        ("Adv (DQN)", DARK_RED, "DQN-controlled chaser"),
        ("Coop", BLUE, "Reach landmarks"),
        ("LM", GREEN, "Landmark"),
        ("Yellow", YELLOW, "Target/Highlight"),
        ("Orange", ORANGE, "Danger")
    ]

    for i, (symbol, color, desc) in enumerate(items):
        y_pos = legend_y + 30 + i * 24
        pygame.draw.circle(screen, color, (panel_x + 15, y_pos), 6)
        desc_text = font_small.render(f"{symbol}: {desc}", True, BLACK)
        screen.blit(desc_text, (panel_x + 25, y_pos - 6))


def get_action_description(action):
    """Get human-readable description of action"""
    actions_desc = {
        0: "Chase Cooperator 1",
        1: "Chase Cooperator 2",
        2: "Patrol Center",
        3: "Ambush Strategy",
        4: "Random Exploration"
    }
    return actions_desc.get(action, "Unknown Action")


def execute_action(action, adversary_pos, cooperator1_pos, cooperator2_pos):
    """Execute the selected action and return new position"""
    new_pos = adversary_pos.copy()

    if action == 0:  # Chase cooperator 1
        target_pos = cooperator1_pos
    elif action == 1:  # Chase cooperator 2
        target_pos = cooperator2_pos
    elif action == 2:  # Patrol center
        target_pos = [0, 0]
    elif action == 3:  # Ambush - move toward midpoint
        target_pos = [(cooperator1_pos[0] + cooperator2_pos[0]) / 2,
                      (cooperator1_pos[1] + cooperator2_pos[1]) / 2]
    else:  # Random exploration
        move_x = np.random.uniform(-ADVERSARY_SPEED, ADVERSARY_SPEED)
        move_y = np.random.uniform(-ADVERSARY_SPEED, ADVERSARY_SPEED)
        new_pos[0] += move_x
        new_pos[1] += move_y
        return new_pos

    # Move toward target
    direction = [target_pos[0] - adversary_pos[0],
                 target_pos[1] - adversary_pos[1]]
    dist = calculate_distance(adversary_pos, target_pos)
    if dist > 0:
        new_pos[0] += direction[0] / dist * ADVERSARY_SPEED
        new_pos[1] += direction[1] / dist * ADVERSARY_SPEED

    return new_pos


def calculate_reward(adversary_pos, cooperator1_pos, cooperator2_pos, capture_occurred, step_reward):
    """Calculate reward for DQN agent"""
    reward = step_reward

    # Check for captures
    if capture_occurred:
        reward += 15.0

    # Distance-based rewards
    dist_to_coop1 = calculate_distance(adversary_pos, cooperator1_pos)
    dist_to_coop2 = calculate_distance(adversary_pos, cooperator2_pos)

    # Reward for getting close to cooperators
    if dist_to_coop1 < 0.3:
        reward += 2.0
    elif dist_to_coop1 < 0.6:
        reward += 0.5

    if dist_to_coop2 < 0.3:
        reward += 2.0
    elif dist_to_coop2 < 0.6:
        reward += 0.5

    # Small penalty for not making progress
    reward -= 0.1

    return reward


def plot_training_progress(dqn_agent, episode):
    """Plot training progress with enhanced visualizations"""
    if not dqn_agent.losses:
        return

    plt.figure(figsize=(15, 20))

    # Plot 1: Overall Loss Trend
    plt.subplot(4, 2, 1)
    plt.plot(dqn_agent.losses)
    plt.title('Overall Loss Trend')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Plot 3: Episode Rewards and Average Rewards
    plt.subplot(4, 2, 2)
    if dqn_agent.episode_rewards:
        plt.plot(dqn_agent.episode_rewards, label='Total Reward per Episode')
        if hasattr(dqn_agent, 'average_rewards') and dqn_agent.average_rewards:
            plt.plot(dqn_agent.average_rewards, label='Avg Reward (10 episodes)', color='red', linewidth=2)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No episode rewards yet', ha='center', va='center')
        plt.axis('off')

    # Plot 4: Average Q Values
    plt.subplot(4, 2, 3)
    if hasattr(dqn_agent, 'q_values') and dqn_agent.q_values:
        # 计算移动平均Q值以减少噪声
        window_size = max(1, len(dqn_agent.q_values) // 50)
        if window_size > 0:
            smoothed_q = []
            for i in range(len(dqn_agent.q_values)):
                start_idx = max(0, i - window_size + 1)
                end_idx = i + 1
                avg = np.mean(dqn_agent.q_values[start_idx:end_idx])
                smoothed_q.append(avg)
            plt.plot(smoothed_q)
        else:
            plt.plot(dqn_agent.q_values)
        plt.title('Average Q Values')
        plt.xlabel('Training Step')
        plt.ylabel('Average Q Value')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Q value data available', ha='center', va='center')
        plt.axis('off')

    # Plot 7: Action Distribution
    plt.subplot(4, 2, 4)
    if hasattr(dqn_agent, 'action_counts') and np.sum(dqn_agent.action_counts) > 0:
        action_names = [get_action_description(i) for i in range(len(dqn_agent.action_counts))]
        plt.bar(range(len(dqn_agent.action_counts)), dqn_agent.action_counts)
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.xticks(range(len(dqn_agent.action_counts)), action_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
    else:
        plt.text(0.5, 0.5, 'No action data available', ha='center', va='center')
        plt.axis('off')

    plt.tight_layout()

    # Save the figure
    plt.savefig(f'training_progress_episode_{episode}.png')

    # Also save a simplified version for the latest progress
    plt.figure(figsize=(10, 6))
    plt.plot(dqn_agent.losses)
    plt.title(f'Training Loss (Episode {episode})')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('latest_training_progress.png')
    plt.close('all')

    print(f"📊 Loss visualization updated and saved")


def main():
    print("✅ Pygame initialized successfully")

    # Create simulation data
    agents, landmarks = create_simulation_data()

    # Initialize DQN agent with correct state size (15)
    state_size = 15
    action_size = 5
    dqn_agent = DQNAgent(state_size, action_size)

    print("📦 Environment configured")
    print(f"   Agents: {len(agents)} (1 DQN Adversary + 2 Cooperators)")
    print(f"   Landmarks: {len(landmarks)}")
    print(f"   State size: {state_size}, Action size: {action_size}")

    print("\n🤖 DQN Configuration:")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Discount Factor: {GAMMA}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Memory Size: {MEMORY_SIZE}")
    print(f"   Epsilon: {EPSILON_START}->{EPSILON_END} (decay: {EPSILON_DECAY})")

    print("\n🎯 Game Rules:")
    print("  🔴 Adversary (DQN): Learns optimal chasing strategy")
    print("  🔵 Cooperator: Reach landmarks for positive points")
    print("  🏆 Victory: Cooperator total > |Adversary total|")
    print("  ⚠️  Press 'S' to save model, 'L' to load model")

    print("\n🎮 Starting DQN training...")
    print("💡 Press ESC to exit")
    print("-" * 50)

    running = True
    clock = pygame.time.Clock()

    # Training parameters - Increased for better learning
    max_episodes = 500  # 增加训练轮数
    max_steps = 500  # 增加每轮的步数
    episode = 0

    # World boundaries
    WORLD_BOUNDS = 1.4

    while running and episode < max_episodes:
        # Reset environment for new episode
        agents, landmarks = create_simulation_data()
        total_rewards = [0, 0, 0]
        episode_reward = 0
        step = 0
        episode_start_time = time.time()
        success_count = 0
        danger_count = 0
        capture_count = 0

        # Get initial state
        current_state = get_state(
            agents[0]['pos'],
            agents[1]['pos'],
            agents[2]['pos'],
            landmarks
        )

        for step in range(max_steps):
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        # Save model
                        dqn_agent.save_model(f'dqn_model_episode_{episode}.pth')
                        print(f"💾 Model saved at episode {episode}")
                    elif event.key == pygame.K_l:
                        # Load model
                        try:
                            dqn_agent.load_model('dqn_model.pth')
                            print("📂 Model loaded successfully")
                        except:
                            print("❌ No model found to load")

            if not running:
                break

            # DQN Action selection
            current_action = dqn_agent.act(current_state)

            # Execute action
            old_adversary_pos = agents[0]['pos'].copy()
            agents[0]['pos'] = execute_action(
                current_action,
                agents[0]['pos'],
                agents[1]['pos'],
                agents[2]['pos']
            )

            # Apply boundary limits
            agents[0]['pos'][0] = max(-WORLD_BOUNDS, min(WORLD_BOUNDS, agents[0]['pos'][0]))
            agents[0]['pos'][1] = max(-WORLD_BOUNDS, min(WORLD_BOUNDS, agents[0]['pos'][1]))

            # Move cooperators
            rewards = [0, 0, 0]
            current_capture = False

            for i, agent in enumerate(agents[1:], 1):  # Cooperators only
                target_idx = agent['target_landmark']
                target_pos = landmarks[target_idx]['pos']

                # Cooperator movement logic
                target_direction = [target_pos[0] - agent['pos'][0],
                                    target_pos[1] - agent['pos'][1]]
                target_distance = calculate_distance(agent['pos'], target_pos)

                # Enhanced evasion behavior with predictive components
                avoid_direction = [0, 0]
                dist_to_adversary = calculate_distance(agent['pos'], agents[0]['pos'])

                # Calculate adversary movement direction for predictive evasion
                adv_movement_dir = [agents[0]['pos'][0] - old_adversary_pos[0],
                                    agents[0]['pos'][1] - old_adversary_pos[1]]
                adv_movement_magnitude = math.sqrt(adv_movement_dir[0] ** 2 + adv_movement_dir[1] ** 2)

                if dist_to_adversary < DANGER_THRESHOLD:
                    # Basic avoidance direction (directly away from adversary)
                    avoid_dir = [agent['pos'][0] - agents[0]['pos'][0],
                                 agent['pos'][1] - agents[0]['pos'][1]]
                    avoid_dist = calculate_distance(agent['pos'], agents[0]['pos'])

                    if avoid_dist > 0:
                        # Distance-based avoidance strength - reduced for more challenging gameplay
                        base_avoid_strength = 0.04 * (DANGER_THRESHOLD - dist_to_adversary) / DANGER_THRESHOLD

                        # Add predictive avoidance component
                        if adv_movement_magnitude > 0:
                            # Predict where adversary is moving and avoid that path
                            predicted_avoid_dir = [-adv_movement_dir[0] / adv_movement_magnitude,
                                                   -adv_movement_dir[1] / adv_movement_magnitude]
                            # Scale prediction influence based on distance (closer = more prediction influence) - reduced for more challenging gameplay
                            prediction_factor = 0.015 * (DANGER_THRESHOLD - dist_to_adversary) / DANGER_THRESHOLD
                            avoid_direction[0] += predicted_avoid_dir[0] * prediction_factor
                            avoid_direction[1] += predicted_avoid_dir[1] * prediction_factor

                        # Add base avoidance
                        avoid_direction[0] += avoid_dir[0] / avoid_dist * base_avoid_strength
                        avoid_direction[1] += avoid_dir[1] / avoid_dist * base_avoid_strength

                        # Add randomness to prevent predictable patterns - reduced randomness
                        if dist_to_adversary < DANGER_THRESHOLD * 0.7:
                            avoid_direction[0] += np.random.uniform(-0.005, 0.005)
                            avoid_direction[1] += np.random.uniform(-0.005, 0.005)

                # Combined movement
                if target_distance > 0.1:
                    agent['pos'][0] += target_direction[0] / target_distance * COOPERATOR_SPEED
                    agent['pos'][1] += target_direction[1] / target_distance * COOPERATOR_SPEED
                    agent['pos'][0] += avoid_direction[0]
                    agent['pos'][1] += avoid_direction[1]
                else:
                    agent['pos'][0] += np.random.uniform(-0.008, 0.008) + avoid_direction[0]
                agent['pos'][1] += np.random.uniform(-0.008, 0.008) + avoid_direction[1]

                # Boundary limits
                agent['pos'][0] = max(-WORLD_BOUNDS, min(WORLD_BOUNDS, agent['pos'][0]))
                agent['pos'][1] = max(-WORLD_BOUNDS, min(WORLD_BOUNDS, agent['pos'][1]))

                # Check for capture
                if calculate_distance(agents[0]['pos'], agent['pos']) < CAPTURE_THRESHOLD:
                    current_capture = True
                    capture_count += 1
                    agent['pos'] = get_random_position()

                # Cooperator rewards
                base_reward = 0
                if target_distance < TARGET_THRESHOLD * 1.5:
                    base_reward = 3.5 - (target_distance * 20)
                    if target_distance < TARGET_THRESHOLD:
                        base_reward += 8.0
                        success_count += 1
                        # Refresh landmark position when reached
                        landmarks[target_idx]['pos'] = get_random_position()
                else:
                    base_reward = 1.2 - (target_distance * 2.5)

                if dist_to_adversary < DANGER_THRESHOLD * 0.8:
                    danger_penalty = 2.5 * (DANGER_THRESHOLD * 0.8 - dist_to_adversary) / (DANGER_THRESHOLD * 0.8)
                    base_reward -= danger_penalty
                    danger_count += 1

                rewards[i] = base_reward
                agent['score'] += base_reward

            # Calculate DQN reward
            dqn_reward = calculate_reward(
                agents[0]['pos'],
                agents[1]['pos'],
                agents[2]['pos'],
                current_capture,
                rewards[0]
            )
            rewards[0] = dqn_reward
            agents[0]['score'] += dqn_reward
            episode_reward += dqn_reward

            # Get next state
            next_state = get_state(
                agents[0]['pos'],
                agents[1]['pos'],
                agents[2]['pos'],
                landmarks
            )

            # Check if episode should end
            done = (step == max_steps - 1)

            # Remember experience
            dqn_agent.remember(current_state, current_action, dqn_reward, next_state, done)

            # Train DQN
            loss = dqn_agent.replay()

            # Update total rewards for display
            for i in range(len(rewards)):
                total_rewards[i] += rewards[i]

            # Update current state
            current_state = next_state

            # Determine game status for display
            adv_abs = abs(total_rewards[0])
            coop_total = total_rewards[1] + total_rewards[2]

            if coop_total > adv_abs and coop_total > 15:
                game_status = "Cooperators Leading!"
            elif step >= max_steps - 50:
                game_status = "Ending Soon"
            elif current_capture:
                game_status = "DQN Capture!"
            else:
                game_status = f"Training - Episode {episode}"

            # Visualization
            screen.fill(BACKGROUND)
            draw_grid()

            # Draw landmarks
            for i, landmark in enumerate(landmarks):
                is_highlighted = any(agent['target_landmark'] == i for agent in agents if agent['type'] == 'cooperator')
                is_occupied = any(calculate_distance(agent['pos'], landmark['pos']) < TARGET_THRESHOLD
                                  for agent in agents if agent['type'] == 'cooperator')
                draw_landmark(landmark['pos'], i, is_highlighted, is_occupied)

            # Draw danger zones
            for agent in agents[1:]:
                draw_danger_zone(agent['pos'], agents[0]['pos'])

            # Draw agents
            for i, agent in enumerate(agents):
                is_adversary = agent['type'] == 'adversary'
                is_dqn_agent = (i == 0)

                target_distance = None
                is_success = False
                is_danger = False

                if not is_adversary:
                    target_idx = agent['target_landmark']
                    target_pos = landmarks[target_idx]['pos']
                    target_distance = calculate_distance(agent['pos'], target_pos)
                    is_success = target_distance < TARGET_THRESHOLD

                    dist_to_adv = calculate_distance(agent['pos'], agents[0]['pos'])
                    if dist_to_adv < DANGER_THRESHOLD * 0.8:
                        is_danger = True

                draw_agent(agent['pos'], is_adversary, i, target_distance, is_success, is_danger, is_dqn_agent)

            # Draw DQN panel
            draw_dqn_panel(step + 1, rewards, total_rewards, game_status, dqn_agent,
                           current_state, current_action, success_count, danger_count,
                           capture_count, episode)

            pygame.display.flip()
            clock.tick(30)

            # Console output every 50 steps
            if (step + 1) % 50 == 0:
                print(f"Episode {episode}, Step {step + 1}:")
                print(f"  DQN Reward: {dqn_reward:+.2f} (Total: {total_rewards[0]:+.2f})")
                print(f"  Action: {get_action_description(current_action)}")
                print(f"  Epsilon: {dqn_agent.epsilon:.3f}")
                print(f"  Memory: {len(dqn_agent.memory)}")
                print(f"  Captures: {capture_count}")

        # End of episode
        dqn_agent.episode_rewards.append(episode_reward)
        # 计算平均回合奖励（最近10个回合）
        recent_rewards = dqn_agent.episode_rewards[-min(10, len(dqn_agent.episode_rewards)):]
        avg_reward = np.mean(recent_rewards)
        dqn_agent.average_rewards.append(avg_reward)

        print(f"\n🎯 Episode {episode} completed:")
        print(f"   Total Reward: {episode_reward:.2f}")
        print(f"   Captures: {capture_count}")
        print(f"   Successes: {success_count}")
        print(f"   Epsilon: {dqn_agent.epsilon:.4f}")
        print(f"   Memory Size: {len(dqn_agent.memory)}")
        print("-" * 40)

        # 在每个回合结束时生成所有可视化图表
        # 记录当前回合奖励
        dqn_agent.episode_rewards.append(episode_reward)

        # 计算并记录平均回合奖励
        if len(dqn_agent.episode_rewards) > 0:
            window_size = min(10, len(dqn_agent.episode_rewards))
            avg_reward = np.mean(dqn_agent.episode_rewards[-window_size:])
            dqn_agent.average_rewards.append(avg_reward)

        # 每回合进行可视化
        plot_training_progress(dqn_agent, episode)

        episode_duration = time.time() - episode_start_time
        print(
            f"📈 Episode {episode + 1}/{max_episodes} completed in {episode_duration:.2f}s - Total Reward: {episode_reward:.2f}")

        episode += 1

    # Save final model
    dqn_agent.save_model('dqn_model_final.pth')

    # Final training plot
    plot_training_progress(dqn_agent, episode)

    pygame.quit()
    print("✅ DQN training completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
        pygame.quit()
        sys.exit()

