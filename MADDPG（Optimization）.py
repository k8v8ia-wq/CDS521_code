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
from torch.utils.tensorboard import SummaryWriter
import sys

print("=" * 60)
print("Multi-Agent Environment - Pursuit Evasion Game with MADDPG")
print("=" * 60)

# Initialize pygame
pygame.init()

# Set up window
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Multi-Agent Environment - Pursuit Evasion Game with MADDPG")

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
DARK_RED = (180, 0, 0)  # Adversary agent highlight

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

# Movement speeds
COOPERATOR_SPEED = 0.018
ADVERSARY_SPEED = COOPERATOR_SPEED * 1.8

# MADDPG Parameters 
NUM_AGENTS = 3
STATE_SIZE = 8 
ACTION_SIZE = 2  
BATCH_SIZE = 256  
LEARNING_RATE = 0.0003  
GAMMA = 0.99  
TAU = 0.005  
MEMORY_SIZE = 50000  
TRAIN_EVERY = 4  

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# MADDPG Network Definitions
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.tanh = nn.Tanh()
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, full_state_size, full_action_size, hidden_size=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(full_state_size + full_action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, full_state, full_action):
        x = torch.cat([full_state, full_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Ornstein-Uhlenbeck noise for exploration - 优化参数
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.1, sigma=0.1):  
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
        
    def reset(self):
        self.state = np.copy(self.mu)
        
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

# MADDPG Agent class
class MADDPGAgent:
    def __init__(self, state_size, action_size, num_agents, agent_index):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.agent_index = agent_index
        
        # Actor networks
        self.actor = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        
        # Critic networks
        full_state_size = state_size * num_agents
        full_action_size = action_size * num_agents
        self.critic = Critic(full_state_size, full_action_size).to(device)
        self.critic_target = Critic(full_state_size, full_action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        
        # Initialize target networks with same weights
        self.soft_update(self.actor, self.actor_target, 1.0)
        self.soft_update(self.critic, self.critic_target, 1.0)
        
        # Training parameters
        self.gamma = GAMMA
        self.tau = TAU
        self.noise = OUNoise(action_size)
        
        # 🚨 探索衰减参数
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.1
        
        # Training statistics
        self.actor_losses = []
        self.critic_losses = []
        self.episode_rewards = []
        self.average_rewards = []
        self.q_values = []
        self.recent_actions = deque(maxlen=1000)
        self.action_counts = np.zeros(8)
        
    def act(self, state, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state_tensor).squeeze(0).detach().cpu().numpy()
        
        if add_noise:
            # 🚨 应用衰减的探索噪声
            noise = self.noise.sample() * self.epsilon
            action += noise
            action = np.clip(action, -1, 1)
        
        # 记录动作用于可视化
        self.recent_actions.append(action.copy())
        
        # 将连续动作离散化为方向类别
        dx, dy = action
        angle = math.atan2(dy, dx)
        direction = int((angle + math.pi) / (math.pi / 4)) % 8
        self.action_counts[direction] += 1
        
        return action
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def replay(self, minibatch):
        """从外部传入的batch进行训练"""
        if minibatch is None or len(minibatch) == 0:
            return 0.0, 0.0
            
        batch_size = len(minibatch)
        
        # 数据处理
        states_batch = torch.FloatTensor([experience[0] for experience in minibatch]).to(device)
        actions_batch = torch.FloatTensor([experience[1] for experience in minibatch]).to(device)
        rewards_batch = torch.FloatTensor([experience[2] for experience in minibatch]).to(device)
        next_states_batch = torch.FloatTensor([experience[3] for experience in minibatch]).to(device)
        dones_batch = torch.FloatTensor([experience[4] for experience in minibatch]).to(device)
        
        # 🚨 根据agent_index提取对应的reward和done
        rewards_batch = rewards_batch[:, self.agent_index].unsqueeze(1)
        dones_batch = dones_batch[:, self.agent_index].unsqueeze(1)
        
        # Reshape for multi-agent
        states_batch = states_batch.view(batch_size, -1)
        actions_batch = actions_batch.view(batch_size, -1)
        next_states_batch = next_states_batch.view(batch_size, -1)
        
        # Update critic
        with torch.no_grad():
            next_actions = []
            for i in range(self.num_agents):
                next_agent_state = next_states_batch[:, i*self.state_size:(i+1)*self.state_size]
                next_agent_action = self.actor_target(next_agent_state)
                next_actions.append(next_agent_action)
            next_actions_full = torch.cat(next_actions, dim=1)
            target_q = self.critic_target(next_states_batch, next_actions_full)
            target_q = rewards_batch + self.gamma * target_q * (1 - dones_batch)
        
        current_q = self.critic(states_batch, actions_batch)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor
        current_actions = []
        for i in range(self.num_agents):
            agent_state = states_batch[:, i*self.state_size:(i+1)*self.state_size]
            if i == self.agent_index:
                agent_action = self.actor(agent_state)
            else:
                agent_action = actions_batch[:, i*self.action_size:(i+1)*self.action_size]
            current_actions.append(agent_action)
        current_actions_full = torch.cat(current_actions, dim=1)
        
        actor_loss = -self.critic(states_batch, current_actions_full).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # 记录Q值
        with torch.no_grad():
            avg_q = current_q.mean().item()
            self.q_values.append(avg_q)
        
        # Soft update target networks
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        return actor_loss.item(), critic_loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, filename):
        """Save the model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'episode_rewards': self.episode_rewards,
            'average_rewards': self.average_rewards,
            'q_values': self.q_values,
            'action_counts': self.action_counts,
            'epsilon': self.epsilon,  # 保存探索率
        }, filename)

    def load_model(self, filename):
        """Load the model weights"""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_losses = checkpoint.get('actor_losses', [])
        self.critic_losses = checkpoint.get('critic_losses', [])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.average_rewards = checkpoint.get('average_rewards', [])
        self.q_values = checkpoint.get('q_values', [])
        self.action_counts = checkpoint.get('action_counts', np.zeros(8))
        self.epsilon = checkpoint.get('epsilon', 1.0)  

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

def get_agent_state(agent, agents, landmarks):
    """Get state for a single agent with boundary awareness"""
    state = []
    WORLD_BOUNDS = 1.4
    
    # Agent's own position
    state.extend(agent['pos'])
    
    # Target position
    if agent['type'] == 'adversary':
        min_dist = float('inf')
        target_pos = [0, 0]
        for other_agent in agents:
            if other_agent['type'] == 'cooperator':
                dist = calculate_distance(agent['pos'], other_agent['pos'])
                if dist < min_dist:
                    min_dist = dist
                    target_pos = other_agent['pos']
        state.extend(target_pos)
    else:
        target_idx = agent['target_landmark']
        target_pos = landmarks[target_idx]['pos']
        state.extend(target_pos)
    
    # Boundary distance information
    state.extend([
        WORLD_BOUNDS - agent['pos'][0],  # Right boundary distance
        agent['pos'][0] + WORLD_BOUNDS,  # Left boundary distance
        WORLD_BOUNDS - agent['pos'][1],  # Upper boundary distance
        agent['pos'][1] + WORLD_BOUNDS   # Lower boundary distance
    ])
    
    return np.array(state, dtype=np.float32)

def get_full_state(agents, landmarks):
    """Get full state for all agents"""
    full_state = []
    for agent in agents:
        agent_state = get_agent_state(agent, agents, landmarks)
        full_state.extend(agent_state)
    return np.array(full_state, dtype=np.float32)

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

def draw_agent(pos, is_adversary, agent_id, target_distance=None, is_success=False, is_danger=False):
    """Draw agent"""
    x, y = world_to_screen(pos)

    if is_success:
        color = PURPLE
    elif is_danger:
        color = ORANGE
    else:
        color = DARK_RED if is_adversary else BLUE

    radius = AGENT_RADIUS + 2 if is_adversary else AGENT_RADIUS
    pygame.draw.circle(screen, color, (x, y), radius)
    pygame.draw.circle(screen, WHITE, (x, y), radius, 2)

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

def apply_continuous_action(pos, action, speed, agent_type):
    """Apply continuous action to agent position"""
    dx, dy = action
    move_magnitude = math.sqrt(dx**2 + dy**2)
    
    if move_magnitude > 0:
        # Normalize and scale by speed
        dx = dx / move_magnitude * speed
        dy = dy / move_magnitude * speed
    else:
        dx, dy = 0, 0
    
    new_pos = [pos[0] + dx, pos[1] + dy]
    
    # Ensure new position is within bounds
    WORLD_BOUNDS = 1.4
    new_pos[0] = max(-WORLD_BOUNDS, min(WORLD_BOUNDS, new_pos[0]))
    new_pos[1] = max(-WORLD_BOUNDS, min(WORLD_BOUNDS, new_pos[1]))
    
    return new_pos

def calculate_rewards(agents, landmarks, capture_occurred):
    """🚨 增强的奖励函数，防止卡边界"""
    rewards = [0, 0, 0]
    WORLD_BOUNDS = 1.4
    
    adversary = agents[0]
    for i, coop in enumerate(agents[1:], 1):
        dist = calculate_distance(adversary['pos'], coop['pos'])
        
        if dist < 1.0:
            proximity_reward = (1.0 - dist) * 2.0
            rewards[0] += proximity_reward
        
        if capture_occurred and dist < CAPTURE_THRESHOLD:
            rewards[0] += 15.0
    
    for i, agent in enumerate(agents[1:], 1):
        target_idx = agent['target_landmark']
        target_pos = landmarks[target_idx]['pos']
        target_dist = calculate_distance(agent['pos'], target_pos)
        
        if target_dist < 1.0:
            progress_reward = (1.0 - target_dist) * 3.0
            rewards[i] += progress_reward
        
        if target_dist < TARGET_THRESHOLD:
            rewards[i] += 20.0
        
        dist_to_adv = calculate_distance(agent['pos'], agents[0]['pos'])
        if dist_to_adv < 0.4:
            danger_penalty = (0.4 - dist_to_adv) * 8.0
            rewards[i] -= danger_penalty
        
        pos = agent['pos']
        boundary_penalty = 0
        if abs(pos[0]) > WORLD_BOUNDS * 0.7:
            boundary_distance = WORLD_BOUNDS - abs(pos[0])
            boundary_penalty += (0.3 - boundary_distance) * 5.0
        if abs(pos[1]) > WORLD_BOUNDS * 0.7:
            boundary_distance = WORLD_BOUNDS - abs(pos[1])
            boundary_penalty += (0.3 - boundary_distance) * 5.0
            
        rewards[i] -= boundary_penalty
        
        if capture_occurred and dist_to_adv < CAPTURE_THRESHOLD:
            rewards[i] -= 15.0
    
    adv_pos = agents[0]['pos']
    adv_boundary_penalty = 0
    if abs(adv_pos[0]) > WORLD_BOUNDS * 0.7:
        boundary_distance = WORLD_BOUNDS - abs(adv_pos[0])
        adv_boundary_penalty += (0.3 - boundary_distance) * 3.0
    if abs(adv_pos[1]) > WORLD_BOUNDS * 0.7:
        boundary_distance = WORLD_BOUNDS - abs(adv_pos[1])
        adv_boundary_penalty += (0.3 - boundary_distance) * 3.0
    rewards[0] -= adv_boundary_penalty
    
    return rewards

def draw_maddpg_panel(step, rewards, total_rewards, game_status, maddpg_agents,
                     success_count, danger_count, capture_count, episode, shared_memory_size):
    """Draw MADDPG information panel"""
    panel_x = WIDTH - INFO_PANEL_WIDTH - INFO_PANEL_MARGIN
    panel_y = INFO_PANEL_MARGIN

    # Panel background
    pygame.draw.rect(screen, WHITE, (panel_x, panel_y, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT))
    pygame.draw.rect(screen, BLACK, (panel_x, panel_y, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT), 2)

    # Title
    title = font_large.render("MADDPG Pursuit Evasion", True, BLACK)
    screen.blit(title, (panel_x + 10, panel_y + 10))

    # Episode and step information
    episode_text = font_medium.render(f"Episode: {episode}", True, BLACK)
    screen.blit(episode_text, (panel_x + 10, panel_y + 50))

    step_text = font_medium.render(f"Step: {step}/1000", True, BLACK)
    screen.blit(step_text, (panel_x + 150, panel_y + 50))

    # Game status
    status_text = font_medium.render(f"Status: {game_status}", True,
                                     (0, 150, 0) if "Leading" in game_status else
                                     ORANGE if "Danger" in game_status or "Captured" in game_status else
                                     (0, 150, 0) if "Success" in game_status else BLACK)
    screen.blit(status_text, (panel_x + 10, panel_y + 80))

    # MADDPG Agent Information
    maddpg_y = panel_y + 110
    maddpg_title = font_medium.render("MADDPG Agents Info:", True, DARK_RED)
    screen.blit(maddpg_title, (panel_x + 10, maddpg_y))

    for i, agent in enumerate(maddpg_agents):
        agent_y = maddpg_y + 25 + i * 60
        agent_type = "Adversary" if i == 0 else f"Cooperator{i}"
        agent_title = font_small.render(f"{agent_type}:", True, DARK_RED if i == 0 else BLUE)
        screen.blit(agent_title, (panel_x + 20, agent_y))
        
        memory_text = font_small.render(f"Shared Memory: {shared_memory_size}", True, BLACK)
        screen.blit(memory_text, (panel_x + 40, agent_y + 15))
        
        if agent.actor_losses:
            avg_actor_loss = np.mean(agent.actor_losses[-50:])
            actor_loss_text = font_small.render(f"Actor Loss: {avg_actor_loss:.4f}", True, BLACK)
            screen.blit(actor_loss_text, (panel_x + 40, agent_y + 30))

    # Statistics
    stats_y = maddpg_y + 200
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

    agent_names = ["Adversary (MADDPG)", "Cooperator1 (MADDPG)", "Cooperator2 (MADDPG)"]
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

def plot_training_progress(maddpg_agents, episode):
    """Plot training progress with only 4 essential plots"""
    plt.figure(figsize=(16, 12))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 红，青，蓝
    
    # Plot 1: Losses (Critic)
    plt.subplot(2, 2, 1)
    for i, agent in enumerate(maddpg_agents):
        agent_type = "Adversary" if i == 0 else f"Cooperator{i}"
        
        if agent.critic_losses:
            window_size = max(1, len(agent.critic_losses) // 50)
            if window_size > 0:
                smoothed_losses = []
                for j in range(len(agent.critic_losses)):
                    start_idx = max(0, j - window_size + 1)
                    end_idx = j + 1
                    avg = np.mean(agent.critic_losses[start_idx:end_idx])
                    smoothed_losses.append(avg)
                plt.plot(smoothed_losses, label=f'{agent_type} Critic', color=colors[i], linestyle='-', alpha=0.8)

    plt.title('Critic Losses')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Episode Rewards 
    plt.subplot(2, 2, 2)
    for i, agent in enumerate(maddpg_agents):
        if agent.episode_rewards:
            agent_type = "Adversary" if i == 0 else f"Cooperator{i}"
            
            if len(agent.episode_rewards) > 10:
                window_size = min(10, len(agent.episode_rewards))
                moving_avg = []
                for j in range(len(agent.episode_rewards)):
                    start_idx = max(0, j - window_size + 1)
                    end_idx = j + 1
                    avg = np.mean(agent.episode_rewards[start_idx:end_idx])
                    moving_avg.append(avg)
                
                plt.plot(moving_avg, label=f'{agent_type} (MA)', color=colors[i], linewidth=2)
            else:
                plt.plot(agent.episode_rewards, label=agent_type, color=colors[i], linewidth=2)

    plt.title('Average Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Q Values
    plt.subplot(2, 2, 3)
    for i, agent in enumerate(maddpg_agents):
        if hasattr(agent, 'q_values') and agent.q_values:
            agent_type = "Adversary" if i == 0 else f"Cooperator{i}"
            
            q_values = agent.q_values
            window_size = max(1, len(q_values) // 100)
            if window_size > 0:
                smoothed_q = []
                for j in range(len(q_values)):
                    start_idx = max(0, j - window_size + 1)
                    end_idx = j + 1
                    avg = np.mean(q_values[start_idx:end_idx])
                    smoothed_q.append(avg)
                plt.plot(smoothed_q[-1000:], label=agent_type, color=colors[i], alpha=0.8)
            else:
                plt.plot(q_values[-1000:], label=agent_type, color=colors[i], alpha=0.8)
    
    plt.title('Q Values (Recent 1000 steps)')
    plt.xlabel('Training Step')
    plt.ylabel('Average Q Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Action Distribution
    plt.subplot(2, 2, 4)
    
    directions = ['→', '↗', '↑', '↖', '←', '↙', '↓', '↘']
    
    bar_width = 0.25
    x_pos = np.arange(len(directions))
    
    for i, agent in enumerate(maddpg_agents):
        if hasattr(agent, 'action_counts') and np.sum(agent.action_counts) > 0:
            agent_type = "Adversary" if i == 0 else f"Cooperator{i}"
            total_actions = np.sum(agent.action_counts)
            if total_actions > 0:
                percentages = (agent.action_counts / total_actions) * 100
                plt.bar(x_pos + i * bar_width, percentages, bar_width, 
                       label=agent_type, color=colors[i], alpha=0.8)
    
    plt.title('Action Distribution by Direction')
    plt.xlabel('Movement Direction')
    plt.ylabel('Percentage (%)')
    plt.xticks(x_pos + bar_width, directions)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'maddpg_training_progress_episode_{episode}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 MADDPG training visualization updated - 4 essential plots saved")

def check_training_progress(maddpg_agents, episode):
    """检查训练进度"""
    progress_indicators = []
    
    for i, agent in enumerate(maddpg_agents):
        if len(agent.episode_rewards) > 10:
            recent_rewards = agent.episode_rewards[-10:]
            avg_recent = np.mean(recent_rewards)
            avg_previous = np.mean(agent.episode_rewards[-20:-10]) if len(agent.episode_rewards) > 20 else avg_recent
            
            if avg_recent > avg_previous:
                progress_indicators.append(f"Agent{i}✓")
            else:
                progress_indicators.append(f"Agent{i}↓")
        
        if hasattr(agent, 'q_values') and len(agent.q_values) > 100:
            recent_q = agent.q_values[-100:]
            q_std = np.std(recent_q)
            if q_std < 5.0:  
                progress_indicators.append(f"Agent{i}_Q_stable")
    
    print(f"Episode {episode} Progress: {', '.join(progress_indicators)}")

def main():
    print("✅ Pygame initialized successfully")

    # Create simulation data
    agents, landmarks = create_simulation_data()

    # Initialize MADDPG agents
    maddpg_agents = []
    for i in range(NUM_AGENTS):
        agent = MADDPGAgent(STATE_SIZE, ACTION_SIZE, NUM_AGENTS, i)
        maddpg_agents.append(agent)

    shared_memory = deque(maxlen=MEMORY_SIZE)

    # Initialize TensorBoard
    writer = SummaryWriter('runs/maddpg_pursuit_evasion')
    print("📊 TensorBoard logging enabled")

    print("📦 Environment configured")
    print(f"   Agents: {NUM_AGENTS} (All using MADDPG)")
    print(f"   Landmarks: {len(landmarks)}")
    print(f"   State size: {STATE_SIZE}, Action size: {ACTION_SIZE}")

    print("\n🤖 MADDPG Configuration:")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Discount Factor: {GAMMA}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Memory Size: {MEMORY_SIZE}")
    print(f"   Train Every: {TRAIN_EVERY} steps")

    print("\n🎯 Game Rules:")
    print("  🔴 Adversary (MADDPG): Learns optimal chasing strategy")
    print("  🔵 Cooperators (MADDPG): Learn to reach landmarks while avoiding adversary")
    print("  🏆 Victory: Cooperator total > |Adversary total|")
    print("  ⚠️  Press 'S' to save model, 'L' to load model")

    print("\n🎮 Starting MADDPG training...")
    print("💡 Press ESC to exit")
    print("-" * 50)

    running = True
    clock = pygame.time.Clock()

    # Training parameters
    max_episodes = 500
    max_steps = 1000
    episode = 0

    # World boundaries
    WORLD_BOUNDS = 1.4

    while running and episode < max_episodes:
        # Reset environment for new episode
        agents, landmarks = create_simulation_data()
        total_rewards = [0, 0, 0]
        episode_rewards = [0, 0, 0]
        step = 0
        episode_start_time = time.time()
        success_count = 0
        danger_count = 0
        capture_count = 0
        
        last_positions = [agent['pos'].copy() for agent in agents]

        for step in range(max_steps):
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        # Save models
                        for i, agent in enumerate(maddpg_agents):
                            agent.save_model(f'maddpg_agent_{i}_episode_{episode}.pth')
                        print(f"💾 Models saved at episode {episode}")
                    elif event.key == pygame.K_l:
                        # Load models
                        try:
                            for i, agent in enumerate(maddpg_agents):
                                agent.load_model(f'maddpg_agent_{i}.pth')
                            print("📂 Models loaded successfully")
                        except:
                            print("❌ No models found to load")

            if not running:
                break

            # Get current full state
            current_full_state = get_full_state(agents, landmarks)
            
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(agents):
                agent_state = get_agent_state(agent, agents, landmarks)
                action = maddpg_agents[i].act(agent_state, add_noise=True)
                actions.append(action)
            
            # Apply actions
            capture_occurred = False
            for i, agent in enumerate(agents):
                speed = ADVERSARY_SPEED if agent['type'] == 'adversary' else COOPERATOR_SPEED
                agent['pos'] = apply_continuous_action(agent['pos'], actions[i], speed, agent['type'])
                
                # Check for capture (adversary only)
                if agent['type'] == 'adversary':
                    for j, other_agent in enumerate(agents):
                        if other_agent['type'] == 'cooperator':
                            dist = calculate_distance(agent['pos'], other_agent['pos'])
                            if dist < CAPTURE_THRESHOLD:
                                capture_occurred = True
                                capture_count += 1
                                other_agent['pos'] = get_random_position()

            # Check for success (cooperators only)
            for agent in agents:
                if agent['type'] == 'cooperator':
                    target_idx = agent['target_landmark']
                    target_pos = landmarks[target_idx]['pos']
                    if calculate_distance(agent['pos'], target_pos) < TARGET_THRESHOLD:
                        success_count += 1
                        agent['pos'] = get_random_position()

            if step % 100 == 0 and step > 200:  
                for i, agent in enumerate(agents):
                    if random.random() < 0.2:  # 
                        random_action = np.random.uniform(-1, 1, 2)
                        speed = ADVERSARY_SPEED if agent['type'] == 'adversary' else COOPERATOR_SPEED
                        agent['pos'] = apply_continuous_action(agent['pos'], random_action, speed, agent['type'])
                        print(f"🔄 Step {step}: Forced exploration for agent {i}")

            if step > 100 and step % 50 == 0:
                current_positions = [agent['pos'].copy() for agent in agents]
                position_changes = [calculate_distance(current_pos, last_pos) 
                                  for current_pos, last_pos in zip(current_positions, last_positions)]
                
                max_change = max(position_changes)
                all_near_boundary = all(
                    abs(pos[0]) > WORLD_BOUNDS * 0.8 or abs(pos[1]) > WORLD_BOUNDS * 0.8 
                    for pos in current_positions
                )
                
                if max_change < 0.02 and all_near_boundary:
                    print(f"🚨 Step {step}: Detected stuck agents, performing reset...")
                    for i in range(NUM_AGENTS):
                        agents[i]['pos'] = get_random_position()
                
                last_positions = current_positions

            # Calculate rewards
            rewards = calculate_rewards(agents, landmarks, capture_occurred)
            
            # Accumulate rewards
            for i in range(len(rewards)):
                total_rewards[i] += rewards[i]
                episode_rewards[i] += rewards[i]
                agents[i]['score'] += rewards[i]

            # Get next state
            next_full_state = get_full_state(agents, landmarks)
            
            # Check if episode should end
            done = (step == max_steps - 1)
            
            shared_memory.append((
                current_full_state.copy(), 
                np.array(actions, dtype=np.float32), 
                np.array(rewards, dtype=np.float32), 
                next_full_state.copy(), 
                np.array([done] * NUM_AGENTS, dtype=np.float32)
            ))

            if step % TRAIN_EVERY == 0 and len(shared_memory) > BATCH_SIZE:
                minibatch = random.sample(shared_memory, BATCH_SIZE)
                
                for i in range(NUM_AGENTS):
                    actor_loss, critic_loss = maddpg_agents[i].replay(minibatch)
                    
                    maddpg_agents[i].decay_epsilon()
                    
                    if actor_loss > 0:  
                        writer.add_scalar(f'Agent_{i}/Actor_Loss', actor_loss, episode * max_steps + step)
                        writer.add_scalar(f'Agent_{i}/Critic_Loss', critic_loss, episode * max_steps + step)
                        writer.add_scalar(f'Agent_{i}/Step_Reward', rewards[i], episode * max_steps + step)
                        writer.add_scalar(f'Agent_{i}/Epsilon', maddpg_agents[i].epsilon, episode * max_steps + step)

            # Log team rewards to TensorBoard
            writer.add_scalar('Team/Adversary_Reward', rewards[0], episode * max_steps + step)
            writer.add_scalar('Team/Cooperator_Team_Reward', rewards[1] + rewards[2], episode * max_steps + step)
            writer.add_scalar('Statistics/Captures', capture_count, episode * max_steps + step)
            writer.add_scalar('Statistics/Successes', success_count, episode * max_steps + step)
            writer.add_scalar('Training/Shared_Memory_Size', len(shared_memory), episode * max_steps + step)

            # Determine game status for display
            adv_abs = abs(total_rewards[0])
            coop_total = total_rewards[1] + total_rewards[2]

            if coop_total > adv_abs and coop_total > 15:
                game_status = "Cooperators Leading!"
            elif step >= max_steps - 50:
                game_status = "Ending Soon"
            elif capture_occurred:
                game_status = "Adversary Capture!"
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

                draw_agent(agent['pos'], is_adversary, i, target_distance, is_success, is_danger)

            # Draw MADDPG panel
            draw_maddpg_panel(step + 1, rewards, total_rewards, game_status, maddpg_agents,
                             success_count, danger_count, capture_count, episode, len(shared_memory))

            pygame.display.flip()
            clock.tick(30)

            # Console output every 50 steps
            if (step + 1) % 50 == 0:
                print(f"Episode {episode}, Step {step + 1}:")
                print(f"  Adversary Reward: {rewards[0]:+.2f} (Total: {total_rewards[0]:+.2f})")
                print(f"  Cooperator1 Reward: {rewards[1]:+.2f} (Total: {total_rewards[1]:+.2f})")
                print(f"  Cooperator2 Reward: {rewards[2]:+.2f} (Total: {total_rewards[2]:+.2f})")
                print(f"  Captures: {capture_count}, Successes: {success_count}")
                print(f"  Shared Memory: {len(shared_memory)}/{MEMORY_SIZE}")

        # End of episode
        for i, agent in enumerate(maddpg_agents):
            agent.episode_rewards.append(episode_rewards[i])
            # Calculate average episode rewards
            if len(agent.episode_rewards) > 0:
                window_size = min(10, len(agent.episode_rewards))
                avg_reward = np.mean(agent.episode_rewards[-window_size:])
                agent.average_rewards.append(avg_reward)
            
            # Log episode rewards to TensorBoard
            writer.add_scalar(f'Agent_{i}/Episode_Reward', episode_rewards[i], episode)
            writer.add_scalar(f'Agent_{i}/Average_Reward', agent.average_rewards[-1] if agent.average_rewards else 0, episode)

        print(f"\n🎯 Episode {episode} completed:")
        print(f"   Adversary Total Reward: {episode_rewards[0]:.2f}")
        print(f"   Cooperator1 Total Reward: {episode_rewards[1]:.2f}")
        print(f"   Cooperator2 Total Reward: {episode_rewards[2]:.2f}")
        print(f"   Captures: {capture_count}, Successes: {success_count}")
        print(f"   Shared Memory Size: {len(shared_memory)}")
        print(f"   Avg Epsilon: {np.mean([agent.epsilon for agent in maddpg_agents]):.3f}")
        print("-" * 40)

        check_training_progress(maddpg_agents, episode)
        
        # Plot training progress
        # plot_training_progress(maddpg_agents, episode)
        
        episode_duration = time.time() - episode_start_time
        print(f"📈 Episode {episode+1}/{max_episodes} completed in {episode_duration:.2f}s")
        
        episode += 1

    # Save final models
    for i, agent in enumerate(maddpg_agents):
        agent.save_model(f'maddpg_agent_{i}_final.pth')

    # Final training plot
    plot_training_progress(maddpg_agents, episode)

    # Close TensorBoard writer
    writer.close()

    pygame.quit()
    print("✅ MADDPG training completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit()