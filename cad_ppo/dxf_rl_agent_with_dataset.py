# dxf_rl_agent_with_dataset.py

import requests
import json
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from io import BytesIO
import cv2
from flask import Flask, render_template_string
import base64
from datetime import datetime
import time

# 创建Flask应用用于实时监控
app = Flask(__name__)
latest_training_info = {
    "episode": 0,
    "score": 0,
    "accuracy": 0,
    "image": None,
    "highlighted_info": {},
    "action": None,
    "reward": 0,
    "was_correct": False,
    "signal_signatures": [],
    "reward_history": [],
    "decision_history": [],
    "recent_images": [],
    "training_time": 0,
    "episode_time": 0
}

@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE, data=latest_training_info)

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DXF PPO Training Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: #333; color: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .stats { display: flex; justify-content: space-between; margin-bottom: 20px; }
        .stat-card { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); flex: 1; margin: 0 10px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #333; }
        .image-container { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; margin-bottom: 20px; }
        .recent-images-container { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; margin-bottom: 20px; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .recent-image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }
        .recent-image-item { border: 1px solid #ddd; border-radius: 5px; padding: 10px; }
        .recent-image-item img { max-width: 100%; height: auto; }
        .recent-image-info { font-size: 12px; margin-top: 5px; text-align: left; }
        .info-panel { background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .highlighted-info { background: #e3f2fd; padding: 10px; border-left: 3px solid #2196f3; margin: 10px 0; }
        .action-info { background: #e8f5e9; padding: 10px; border-left: 3px solid #4caf50; margin: 10px 0; }
        .correct { background: #e8f5e9; border-left: 3px solid #4caf50; }
        .incorrect { background: #ffebee; border-left: 3px solid #f44336; }
        .signal-object { background: #fff3e0; border-left: 3px solid #ff9800; }
        .signal-signatures { background: #fce4ec; padding: 15px; border-left: 5px solid #e91e63; margin-bottom: 15px; }
        .reward-history { background: #f3e5f5; padding: 15px; border-left: 5px solid #9c27b0; margin-bottom: 15px; }
        .decision-history { background: #e0f2f1; padding: 15px; border-left: 5px solid #009688; margin-bottom: 15px; }
        .refresh-btn { background: #2196f3; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        .refresh-btn:hover { background: #1976d2; }
        .signature-item { padding: 8px; margin: 5px 0; background: #f9f9f9; border-left: 3px solid #e91e63; border-radius: 3px; }
        .reward-chart { height: 100px; display: flex; align-items: flex-end; gap: 2px; margin-top: 10px; }
        .reward-bar { flex: 1; background: #9c27b0; text-align: center; color: white; font-size: 10px; }
        .decision-item { padding: 10px; margin: 5px 0; border-radius: 3px; }
        .decision-item.correct { background: #e8f5e9; border-left: 5px solid #4caf50; }
        .decision-item.incorrect { background: #ffebee; border-left: 5px solid #f44336; }
        .decision-item.signal-object { background: #fff3e0; border-left: 5px solid #ff9800; }
        .time-info { background: #fff8e1; padding: 15px; border-left: 5px solid #ffc107; margin-bottom: 15px; }
        .current-decision-info { margin-top: 10px; text-align: left; }
        .recent-image-decision { margin-top: 8px; text-align: left; font-size: 12px; }
        .section-title { margin-top: 20px; color: #333; border-bottom: 2px solid #333; padding-bottom: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>DXF PPO Training Dashboard</h1>
            <p>实时监控CAD对象删除决策训练过程</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <h3>当前回合</h3>
                <div class="stat-value">{{ data.episode }}</div>
            </div>
            <div class="stat-card">
                <h3>平均得分</h3>
                <div class="stat-value">{{ "%.2f"|format(data.score) }}</div>
            </div>
            <div class="stat-card">
                <h3>准确率</h3>
                <div class="stat-value">{{ "%.2f"|format(data.accuracy * 100) }}%</div>
            </div>
        </div>
        
        <!-- 新增：时间信息 -->
        <div class="time-info">
            <h3>训练时间信息</h3>
            <p><strong>总训练时间:</strong> {{ "%.2f"|format(data.training_time) }} 秒</p>
            <p><strong>当前Episode耗时:</strong> {{ "%.2f"|format(data.episode_time) }} 秒</p>
        </div>
        
        <div class="recent-images-container">
            <h2 class="section-title">决策图像与信息</h2>
            {% if data.recent_images %}
                <div class="recent-image-grid">
                    {% for img_data in data.recent_images|reverse %}
                        <div class="recent-image-item">
                            <img src="data:image/png;base64,{{ img_data.image }}" alt="Decision {{ img_data.step }}">
                            <div class="recent-image-info">
                                <p><strong>步骤:</strong> {{ img_data.step }}</p>
                                <p><strong>奖励:</strong> {{ img_data.reward }}</p>
                            </div>
                            <div class="recent-image-decision {{ 'correct' if img_data.was_correct else 'incorrect' if img_data.was_correct != None else '' }}">
                                <p><strong>决策:</strong> {% if img_data.action == 1 %}删除{% else %}保留{% endif %}</p>
                                <p><strong>正确性:</strong> 
                                    {% if img_data.was_correct == True %}正确{% elif img_data.was_correct == False %}错误{% else %}未知{% endif %}
                                </p>
                            </div>
                        </div>
                    {% endfor %}
                </div>
            {% else %}
                <p>暂无图像数据</p>
            {% endif %}
        </div>
        
        <div class="info-panel">
            <h2 class="section-title">详细信息</h2>
            
            <!-- 新增：信号签名信息 -->
            <div class="signal-signatures">
                <h3>信号签名对象 (来自 label/1.dxf)</h3>
                {% if data.signal_signatures %}
                    <div style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                        {% for signature in data.signal_signatures %}
                            <div class="signature-item">
                                <strong>类型:</strong> {{ signature.type }}
                                {% if signature.type == "LINE" %}
                                    <br><strong>起点:</strong> ({{ "%.2f"|format(signature.start[0]) }}, {{ "%.2f"|format(signature.start[1]) }})
                                    <br><strong>终点:</strong> ({{ "%.2f"|format(signature.end[0]) }}, {{ "%.2f"|format(signature.end[1]) }})
                                {% elif signature.type == "CIRCLE" %}
                                    <br><strong>圆心:</strong> ({{ "%.2f"|format(signature.center[0]) }}, {{ "%.2f"|format(signature.center[1]) }})
                                    <br><strong>半径:</strong> {{ "%.2f"|format(signature.radius) }}
                                {% elif signature.type == "ARC" %}
                                    <br><strong>圆心:</strong> ({{ "%.2f"|format(signature.center[0]) }}, {{ "%.2f"|format(signature.center[1]) }})
                                    <br><strong>半径:</strong> {{ "%.2f"|format(signature.radius) }}
                                {% endif %}
                            </div>
                        {% endfor %}
                    </div>
                    <p style="margin-top: 10px;"><strong>总计:</strong> {{ data.signal_signatures|length }} 个信号对象</p>
                {% else %}
                    <p>暂无信号签名信息</p>
                {% endif %}
            </div>
            
            <!-- 奖励历史信息 -->
            <div class="reward-history">
                <h3>奖励历史</h3>
                <p><strong>当前奖励:</strong> {{ data.reward }}</p>
                {% if data.reward_history %}
                    <div class="reward-chart">
                        {% for reward in data.reward_history %}
                            <div class="reward-bar" style="height: {{ (reward + 5) * 10 }}px;">
                                {{ "%.1f"|format(reward) }}
                            </div>
                        {% endfor %}
                    </div>
                    <p style="margin-top: 10px;"><strong>平均奖励 (最近20次):</strong> {{ "%.2f"|format(data.reward_history|sum / data.reward_history|length) }}</p>
                {% else %}
                    <p>暂无奖励历史数据</p>
                {% endif %}
            </div>
            
            <!-- 决策历史信息 -->
            <div class="decision-history">
                <h3>本轮决策历史</h3>
                {% if data.decision_history %}
                    <div style="max-height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; border-radius: 5px;">
                        {% for decision in data.decision_history|reverse %}
                            <div class="decision-item {{ 'correct' if decision.was_correct else 'incorrect' if decision.was_correct != None else '' }} {{ 'signal-object' if decision.is_in_signal else '' }}">
                                <p><strong>步骤 {{ loop.index }}:</strong></p>
                                <p><strong>对象类型:</strong> {{ decision.highlighted_info.type }}</p>
                                <p><strong>AI决策:</strong> 
                                    {% if decision.action == 1 %}删除{% elif decision.action == 0 %}保留{% else %}未知{% endif %}
                                    {% if decision.is_in_signal %}<span style="color: #ff9800;"> [信号对象]</span>{% endif %}
                                </p>
                                <p><strong>应删除:</strong> {{ decision.should_delete|default("未知") }}</p>
                                <p><strong>奖励:</strong> {{ decision.reward }}</p>
                                <p><strong>正确性:</strong> 
                                    {% if decision.was_correct == True %}正确{% elif decision.was_correct == False %}错误{% else %}未知{% endif %}
                                </p>
                            </div>
                        {% endfor %}
                    </div>
                    <p style="margin-top: 10px;"><strong>总计:</strong> {{ data.decision_history|length }} 个决策</p>
                {% else %}
                    <p>暂无决策历史数据</p>
                {% endif %}
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 20px;">
            <button class="refresh-btn" onclick="location.reload()">刷新</button>
        </div>
    </div>
    
    <!-- 自动刷新 -->
    <script>
        // 每5秒自动刷新页面（降低刷新频率）
        setInterval(function() {
            location.reload();
        }, 5000);
    </script>
</body>
</html>
"""

class JSONDeletionEnv:
    def __init__(self, dataset_path="output", max_steps=5):
        self.dataset_path = os.path.abspath(dataset_path)
        self.combined_labels_file = os.path.join(self.dataset_path, "combined_labels.json")
        
        # 加载所有样本
        self.samples = self._load_samples()
        self.signal_signatures = self._load_signal_signatures()
        self.current_sample_idx = 0
        self.current_step = 0
        self.max_steps = max_steps
        
        print(f"Dataset path: {self.dataset_path}")
        print(f"Combined labels file: {self.combined_labels_file}")
        print(f"Found {len(self.samples)} samples")
        print(f"Loaded {len(self.signal_signatures)} signal signatures")
        
    def _load_samples(self):
        """从combined_labels.json加载所有样本"""
        samples = []
        if os.path.exists(self.combined_labels_file):
            try:
                with open(self.combined_labels_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    samples = data.get("samples", [])
            except Exception as e:
                print(f"Error loading samples from {self.combined_labels_file}: {e}")
        return samples
        
    def _load_signal_signatures(self):
        """
        从combined_labels.json加载信号签名信息
        """
        signal_signatures = []
        
        if os.path.exists(self.combined_labels_file):
            try:
                with open(self.combined_labels_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 从任意样本中提取信号签名（假设所有样本都使用相同的信号签名）
                    if data.get("samples"):
                        # 从第一个样本中获取信号签名
                        sample = data["samples"][0]
                        # 实际上，我们需要从训练数据生成过程中获取参考签名
                        # 这里我们假设信号签名存储在单独的字段中
                        if "signal_signatures" in data:
                            signal_signatures = data["signal_signatures"]
                        else:
                            # 如果没有单独的字段，尝试从样本中提取
                            # 这里我们简单地返回空列表，实际应该根据需求实现
                            signal_signatures = []
            except Exception as e:
                print(f"Error loading signal signatures from {self.combined_labels_file}: {e}")
        
        return signal_signatures
        
    def reset(self):
        # 随机选择一个样本
        if not self.samples:
            raise Exception("No samples found in combined_labels.json")
            
        self.current_sample_idx = np.random.choice(len(self.samples))
        self.current_step = 0
        sample = self.samples[self.current_sample_idx]
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Selected sample: {sample.get('sample_id', 'Unknown')}")
        
        return self._get_state()
    
    def _get_state(self):
        start_time = time.time()
        sample = self.samples[self.current_sample_idx]
        image_path = sample.get("image_path")
        
        if image_path and os.path.exists(image_path):
            # 读取图像并预处理为固定大小
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (128, 128))
                image = image.transpose(2, 0, 1)  # HWC to CHW
                image = image.astype(np.float32) / 255.0  # 归一化
                process_time = time.time() - start_time
                print(f"  Image processing time: {process_time:.2f}s")
                return image
        process_time = time.time() - start_time
        print(f"  Image processing time (failed): {process_time:.2f}s")
        return np.zeros((3, 128, 128), dtype=np.float32)
    
    def step(self, action):
        """
        执行动作并返回奖励
        action: 0 - 不删除, 1 - 删除
        """
        reward = 0
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        sample = self.samples[self.current_sample_idx]
        entity_info = sample.get("entity_info", {})
        entity_signature = entity_info.get("signature", {})
        should_delete = sample.get("label", {}).get("should_delete", False)
        
        # 检查对象是否在信号签名中
        is_in_signal = self._is_in_signal_signatures(entity_signature)
        
        # 根据动作和实际情况计算奖励
        if action == 1:  # 选择删除
            if should_delete:  # 正确删除了应该删除的元素
                reward = 1
            elif is_in_signal:  # 错误删除了信号对象
                reward = -2  # 更严厉的惩罚
            else:  # 错误删除了不应删除的元素
                reward = -1
        else:  # 选择保留
            if not should_delete:  # 正确保留了不应删除的元素
                reward = 1
            elif is_in_signal:  # 正确保留了信号对象（奖励）
                reward = 2  # 额外奖励
            else:  # 错误保留了应该删除的元素
                reward = -1
                
        # 获取下一个样本（如果未完成）
        if not done:
            self.current_sample_idx = np.random.choice(len(self.samples))
        
        return self._get_state(), reward, done, {
            "highlighted_type": entity_info.get("type"),
            "was_correct": (action == 1) == should_delete,
            "should_delete": should_delete,
            "is_in_signal": is_in_signal,
            "highlighted_entity": entity_info
        }
    
    def _is_in_signal_signatures(self, highlighted_signature):
        """
        检查高亮对象是否在信号签名列表中
        """
        for signal_sig in self.signal_signatures:
            if self._signatures_match(highlighted_signature, signal_sig):
                return True
        return False
    
    def _signatures_match(self, sig1, sig2):
        """
        比较两个对象签名是否匹配
        """
        if sig1.get("type") != sig2.get("type"):
            return False
            
        # 根据不同类型的对象比较关键属性
        obj_type = sig1.get("type")
        tolerance = 1e-3  # 容忍度
        
        if obj_type == "LINE":
            return (abs(sig1.get("start", [0,0])[0] - sig2.get("start", [0,0])[0]) < tolerance and
                    abs(sig1.get("start", [0,0])[1] - sig2.get("start", [0,0])[1]) < tolerance and
                    abs(sig1.get("end", [0,0])[0] - sig2.get("end", [0,0])[0]) < tolerance and
                    abs(sig1.get("end", [0,0])[1] - sig2.get("end", [0,0])[1]) < tolerance)
        elif obj_type == "CIRCLE":
            return (abs(sig1.get("center", [0,0])[0] - sig2.get("center", [0,0])[0]) < tolerance and
                    abs(sig1.get("center", [0,0])[1] - sig2.get("center", [0,0])[1]) < tolerance and
                    abs(sig1.get("radius", 0) - sig2.get("radius", 0)) < tolerance)
        elif obj_type == "ARC":
            return (abs(sig1.get("center", [0,0])[0] - sig2.get("center", [0,0])[0]) < tolerance and
                    abs(sig1.get("center", [0,0])[1] - sig2.get("center", [0,0])[1]) < tolerance and
                    abs(sig1.get("radius", 0) - sig2.get("radius", 0)) < tolerance)
            
        return False

class PPOAgent(nn.Module):
    def __init__(self, input_shape=(3, 128, 128), action_space=2):
        super(PPOAgent, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积层输出大小
        conv_out_size = self._get_conv_output(input_shape)
        
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv_layers(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

class PPOTrainer:
    def __init__(self, agent, lr=1e-3, gamma=0.99, eps_clip=0.2, update_timestep=100):
        self.agent = agent
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_timestep = update_timestep
        self.timestep = 0
        
        self.memory = []
        
    def store_transition(self, state, action, prob, reward, next_state, done):
        self.memory.append((state, action, prob, reward, next_state, done))
        self.timestep += 1
        
    def clear_memory(self):
        self.memory = []
        self.timestep = 0
        
    def update(self):
        if len(self.memory) == 0:
            return
            
        print("Starting network update...")
        update_start_time = time.time()
        
        states = torch.FloatTensor(np.array([e[0] for e in self.memory]))
        actions = torch.LongTensor([e[1] for e in self.memory])
        old_probs = torch.FloatTensor([e[2] for e in self.memory])
        rewards = [e[3] for e in self.memory]
        next_states = torch.FloatTensor(np.array([e[4] for e in self.memory]))
        dones = torch.BoolTensor([e[5] for e in self.memory])
        
        # 计算折扣回报
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0
            G = reward + self.gamma * G
            returns.insert(0, G)
            
        returns = torch.FloatTensor(returns)
        
        # 标准化回报
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        
        # 获取新概率
        action_probs, state_values = self.agent(states)
        new_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        old_probs = old_probs.detach()
        
        # 计算优势函数
        advantages = returns - state_values.squeeze(1).detach()
        
        # PPO损失计算
        ratios = new_probs / (old_probs + 1e-5)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Critic损失
        critic_loss = nn.MSELoss()(state_values.squeeze(1), returns)
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        update_time = time.time() - update_start_time
        print(f"Network update completed in {update_time:.2f}s")
        
        self.clear_memory()

def update_dashboard(episode, score, accuracy, image_path=None, highlighted_info=None, action=None, reward=0, was_correct=None, is_in_signal=None, signal_signatures=None, decision_history=None, recent_images=None, training_time=0, episode_time=0):
    """更新仪表板数据"""
    latest_training_info["episode"] = episode
    latest_training_info["score"] = score
    latest_training_info["accuracy"] = accuracy
    latest_training_info["training_time"] = training_time
    latest_training_info["episode_time"] = episode_time
    
    # 处理图像 - 使用最新图像而非缓存
    if image_path and os.path.exists(image_path):
        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
                latest_training_info["image"] = base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            latest_training_info["image"] = None
    else:
        latest_training_info["image"] = None
    
    # 处理高亮信息
    if highlighted_info:
        latest_training_info["highlighted_info"] = highlighted_info
        # 添加信号对象信息
        if is_in_signal is not None:
            latest_training_info["highlighted_info"]["is_in_signal"] = is_in_signal
    else:
        latest_training_info["highlighted_info"] = {}
        
    latest_training_info["action"] = action
    latest_training_info["reward"] = reward
    latest_training_info["was_correct"] = was_correct
    
    # 更新信号签名信息
    if signal_signatures is not None:
        latest_training_info["signal_signatures"] = signal_signatures
        
    # 更新奖励历史（保留最近20个奖励值）
    latest_training_info["reward_history"].append(reward)
    if len(latest_training_info["reward_history"]) > 20:
        latest_training_info["reward_history"].pop(0)
        
    # 更新决策历史
    if decision_history is not None:
        latest_training_info["decision_history"] = decision_history
        
    # 更新最近图像历史
    if recent_images is not None:
        latest_training_info["recent_images"] = recent_images

# 修改训练函数以使用新的环境
def train_ppo_agent_with_json(dataset_path="output", episodes=1000):
    # 初始化环境和代理
    env = JSONDeletionEnv(dataset_path, max_steps=5)
    agent = PPOAgent()
    trainer = PPOTrainer(agent, lr=1e-3, update_timestep=20)
    
    # 在训练开始时将信号签名信息传递给仪表板
    update_dashboard(
        episode=0,
        score=0,
        accuracy=0,
        signal_signatures=env.signal_signatures
    )
    
    scores = deque(maxlen=100)
    correct_decisions = deque(maxlen=100)
    
    # 记录训练开始时间
    start_time = time.time()
    
    for episode in range(episodes):
        # 记录episode开始时间
        episode_start_time = time.time()
        
        try:
            state = env.reset()
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error resetting environment: {e}")
            continue
            
        total_reward = 0
        done = False
        correct_count = 0
        total_steps = 0
        
        # 收集用于仪表板的信息
        highlighted_info = {}
        
        step_count = 0
        step_times = []
        action_selection_times = []
        env_step_times = []
        image_processing_times = []
        
        while not done:
            step_start_time = time.time()
            
            # 选择动作
            action_select_start = time.time()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, _ = agent(state_tensor)
            action_probs_np = action_probs.detach().numpy()[0]
            action = np.random.choice(len(action_probs_np), p=action_probs_np)
            prob = action_probs_np[action]
            action_select_time = time.time() - action_select_start
            action_selection_times.append(action_select_time)
            
            env_step_start = time.time()
            next_state, reward, done, info = env.step(action)
            env_step_time = time.time() - env_step_start
            env_step_times.append(env_step_time)
            
            trainer.store_transition(state, action, prob, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            total_steps += 1
            step_count += 1
            if info.get("was_correct", False):
                correct_count += 1
                
            step_time = time.time() - step_start_time
            step_times.append(step_time)
        
        accuracy = correct_count / total_steps if total_steps > 0 else 0
        scores.append(total_reward)
        correct_decisions.append(accuracy)
        
        avg_score = np.mean(scores)
        avg_accuracy = np.mean(correct_decisions)
        
        # 计算episode耗时
        episode_time = time.time() - episode_start_time
        training_time = time.time() - start_time
        
        # 打印详细的性能分析
        if step_times:
            print(f"\n=== Episode {episode} Performance Analysis ===")
            print(f"Total time: {episode_time:.2f}s")
            print(f"Average time per step: {np.mean(step_times):.2f}s")
            print(f"Action selection average time: {np.mean(action_selection_times):.4f}s")
            print(f"Environment step average time: {np.mean(env_step_times):.2f}s")
            print("=" * 45)
        
        # 降低仪表板更新频率（每5个episode更新一次）
        if episode % 5 == 0:
            update_dashboard(
                episode=episode,
                score=avg_score,
                accuracy=avg_accuracy,
                training_time=training_time,
                episode_time=episode_time
            )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Episode {episode}, "
              f"Score: {total_reward:.2f}, "
              f"Avg Score: {avg_score:.2f}, "
              f"Accuracy: {avg_accuracy:.2f}, "
              f"Action: {'Delete' if action == 1 else 'Keep'}, "
              f"Reward: {reward}, "
              f"Episode Time: {episode_time:.2f}s")
        
        # 更新代理（根据trainer的设定决定何时更新）
        if trainer.timestep >= trainer.update_timestep:
            trainer.update()
            print(f'[{datetime.now().strftime("%H:%M:%S")}] Network updated at episode {episode}')
        
        if episode % 20 == 0:
            print(f'[{datetime.now().strftime("%H:%M:%S")}] Saving model at episode {episode}')
            # 保存模型
            torch.save(agent.state_dict(), f'pth/dxf_ppo_agent_ep{episode}.pth')
            
    # 最终保存模型
    torch.save(agent.state_dict(), 'pth/dxf_ppo_agent_final.pth')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training completed. Final model saved.")

def start_dashboard():
    """启动仪表板"""
    app.run(host='0.0.0.0', port=5302, debug=False, use_reloader=False)

# 修改主函数以使用新的训练函数
if __name__ == "__main__":
    import threading
    
    # 在单独的线程中启动仪表板
    dashboard_thread = threading.Thread(target=start_dashboard)
    dashboard_thread.daemon = True
    dashboard_thread.start()
    
    print("Dashboard started at: http://localhost:5302")
    
    # 启动训练（使用JSON数据集）
    train_ppo_agent_with_json("output", episodes=1000)
    
    print("DXF PPO Agent with Dataset ready. Dashboard available at http://localhost:5302")