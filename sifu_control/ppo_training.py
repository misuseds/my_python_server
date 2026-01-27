"""
PPO训练实现
使用PPO算法在游戏环境中搜索目标
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import cv2
from PIL import Image
import base64
from collections import deque
import logging
from pathlib import Path
from ultralytics import YOLO

# 导入移动控制器
from control_api_tool import ImprovedMovementController

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义PPO网络
class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class PPOCritic(nn.Module):
    def __init__(self, state_dim):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=3e-4, gamma=0.99, eps_clip=0.2):
        self.actor = PPOActor(state_dim, action_dim)
        self.critic = PPOCritic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.memory = deque(maxlen=10000)
    
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), action_probs.detach().numpy()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        self.memory.append((state, action, reward, next_state, done, log_prob))
    
    def update(self, batch_size=64, epochs=4):
        if len(self.memory) < batch_size:
            return
        
        for _ in range(epochs):
            states, actions, rewards, next_states, dones, old_log_probs = zip(*random.sample(self.memory, batch_size))
            
            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions)
            rewards = torch.FloatTensor(rewards)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones)
            old_log_probs = torch.FloatTensor(old_log_probs)
            
            # 计算优势函数
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            advantages = []
            gae = 0
            
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
                gae = delta + self.gamma * 0.95 * (1 - dones[i]) * gae
                advantages.insert(0, gae)
            
            advantages = torch.FloatTensor(advantages)
            returns = advantages + values
            
            # 计算新的动作概率
            action_probs = self.actor(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # 计算PPO损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = nn.MSELoss()(values, returns)
            
            # 更新网络
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()

# 定义环境类
class GameEnvironment:
    def __init__(self):
        self.controller = ImprovedMovementController()
        self.yolo_model = YOLO('yolov8n.pt')  # 加载YOLO模型用于目标检测
        self.action_space = [0, 1, 2]  # 0: 左转30度, 1: 右转30度, 2: 不动
        self.state_dim = 10  # 简化的状态维度
    
    def get_state(self):
        # 简化的状态获取，实际应用中可能需要更复杂的状态表示
        state = np.random.rand(self.state_dim)
        return state
    
    def step(self, action):
        # 执行动作
        if action == 0:
            # 左转30度
            self.controller.turn_left(angle=30, duration=0.1)
        elif action == 1:
            # 右转30度
            self.controller.turn_right(angle=30, duration=0.1)
        elif action == 2:
            # 不动
            time.sleep(0.1)
        
        # 向前移动一小步
        self.controller.move_forward(duration=0.5)
        
        # 获取新状态
        next_state = self.get_state()
        
        # 简化的奖励函数
        reward = random.uniform(-1, 1)
        
        # 检测是否达到目标
        done = False
        # 实际应用中需要检测是否出现攀爬按钮提示
        
        return next_state, reward, done
    
    def reset(self):
        # 重置环境
        return self.get_state()

# 训练函数
def train_ppo_agent(episodes=1000, batch_size=64, save_interval=100):
    env = GameEnvironment()
    state_dim = env.state_dim
    action_dim = len(env.action_space)
    
    agent = PPOAgent(state_dim, action_dim)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, action_probs = agent.select_action(state)
            log_prob = np.log(action_probs[0, action])
            
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done, log_prob)
            
            state = next_state
            total_reward += reward
        
        # 更新网络
        agent.update(batch_size)
        
        logger.info(f"Episode {episode+1}, Total Reward: {total_reward:.2f}")
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            torch.save(agent.actor.state_dict(), f"ppo_actor_episode_{episode+1}.pt")
            torch.save(agent.critic.state_dict(), f"ppo_critic_episode_{episode+1}.pt")

if __name__ == "__main__":
    train_ppo_agent()
