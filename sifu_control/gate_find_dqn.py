"""
目标搜索强化学习AI - 门搜索专用版
使用Grounding DINO进行目标检测，结合移动控制器进行强化学习训练
适配VLMWindow格式
"""
import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from PIL import Image
import base64
from collections import deque

# 添加当前目录到Python路径，确保能正确导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有的移动控制器
from control_api_tool import ImprovedMovementController

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入现有的目标检测功能
from grounding_dino.dino import detect_objects_with_text_transformers

class TargetSearchEnvironment:
    """
    目标搜索环境
    """
    def __init__(self, target_description="gate"):
        self.controller = ImprovedMovementController()
        self.target_description = target_description
        self.step_count = 0
        self.max_steps = 50
        self.last_detection_result = None
        self.last_center_distance = float('inf')
        
    def capture_screen(self):
        """
        截取当前屏幕画面
        """
        # 使用现有的浏览器截图功能
        try:
            from browser_server.prtsc import capture_screen_region
            screenshot = capture_screen_region()
            return screenshot
        except ImportError:
            # 如果没有截图功能，模拟返回一张图片
            print("截图功能不可用，使用模拟图片")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def detect_target(self, image):
        """
        使用Grounding DINO检测目标
        """
        # 将numpy数组转换为base64
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # 调用现有的检测函数
        try:
            detection_results = detect_objects_with_text_transformers(base64_image, self.target_description)
            return detection_results
        except Exception as e:
            print(f"检测过程中出错: {e}")
            return []
    
    def calculate_reward(self, detection_results, prev_distance):
        """
        计算奖励值
        """
        if not detection_results:
            # 没有检测到目标，给予小的负奖励
            return -1.0
        
        # 找到最近的检测框
        min_distance = float('inf')
        for detection in detection_results:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            # 计算到图像中心的距离
            img_center_x = 320  # 假设图像宽度为640
            img_center_y = 240  # 假设图像高度为480
            distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
        
        # 如果距离比之前更近，给予正奖励
        if min_distance < prev_distance:
            reward = 10.0 * (prev_distance - min_distance) / max(prev_distance, 1)
        else:
            reward = -5.0  # 距离变远，给予负奖励
        
        # 如果目标在中心附近，给予额外奖励
        if min_distance < 50:
            reward += 20.0  # 找到目标，给予大奖励
        
        return reward
    
    def step(self, action):
        """
        执行动作并返回新的状态、奖励和是否结束
        动作空间: 0-forward, 1-backward, 2-turn_left, 3-turn_right, 4-strafe_left, 5-strafe_right
        """
        # 执行动作
        if action == 0:  # forward
            self.controller.move_forward(1)
        elif action == 1:  # backward
            self.controller.move_backward(1)
        elif action == 2:  # turn_left
            self.controller.turn_left(30)
        elif action == 3:  # turn_right
            self.controller.turn_right(30)
        elif action == 4:  # strafe_left
            self.controller.strafe_left(1)
        elif action == 5:  # strafe_right
            self.controller.strafe_right(1)
        
        time.sleep(0.5)  # 等待动作完成
        
        # 获取新状态（截图）
        new_state = self.capture_screen()
        
        # 检测目标
        detection_results = self.detect_target(new_state)
        
        # 计算奖励
        current_distance = self.last_center_distance
        if detection_results:
            # 计算当前最近目标到中心的距离
            min_distance = float('inf')
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                img_center_x = new_state.shape[1] / 2
                img_center_y = new_state.shape[0] / 2
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
            current_distance = min_distance
        
        reward = self.calculate_reward(detection_results, self.last_center_distance)
        self.last_center_distance = current_distance
        self.last_detection_result = detection_results
        
        # 更新步数
        self.step_count += 1
        
        # 检查是否结束
        done = self.step_count >= self.max_steps or (detection_results and current_distance < 50)
        
        return new_state, reward, done, detection_results
    
    def reset(self):
        """
        重置环境
        """
        self.step_count = 0
        self.last_center_distance = float('inf')
        self.last_detection_result = None
        initial_state = self.capture_screen()
        return initial_state

class DQNNetwork(nn.Module):
    """
    深度Q网络
    """
    def __init__(self, input_channels=3, action_space=6):
        super(DQNNetwork, self).__init__()
        
        # 卷积层处理图像输入
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积输出大小
        conv_out_size = self._get_conv_output((input_channels, 480, 640))
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )
    
    def _get_conv_output(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv_layers(x)
            return int(np.prod(x.size()))
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        return x

class TargetSearchDQNAgent:
    """
    目标搜索DQN智能体
    """
    def __init__(self, action_space=6, learning_rate=1e-4, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 神经网络
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork(action_space=action_space).to(self.device)
        self.target_network = DQNNetwork(action_space=action_space).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        
        # 更新目标网络
        self.update_target_network()
    
    def update_target_network(self):
        """
        更新目标网络权重
        """
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """
        存储经验
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        根据当前策略选择动作
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        
        # 预处理状态
        state_tensor = self._preprocess_state(state)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)  # 添加批次维度
        
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def _preprocess_state(self, state):
        """
        预处理状态（图像）
        """
        # 将BGR转为RGB
        if len(state.shape) == 3:
            state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        else:
            state_rgb = state
        
        # 转换为tensor并归一化
        state_tensor = torch.FloatTensor(state_rgb).permute(2, 0, 1) / 255.0
        return state_tensor
    
    def replay(self, batch_size=32):
        """
        经验回放训练
        """
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.stack([self._preprocess_state(e[0]) for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1]] for e in batch).to(self.device)
        rewards = torch.FloatTensor([e[2]] for e in batch).to(self.device)
        next_states = torch.stack([self._preprocess_state(e[3]) for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4]] for e in batch).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_gate_search_agent():
    """
    训练门搜索智能体
    """
    print("开始训练门搜索智能体...")
    
    env = TargetSearchEnvironment("gate")
    agent = TargetSearchDQNAgent()
    
    scores = deque(maxlen=100)
    
    for episode in range(200):  # 训练200轮
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, detection_results = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                print(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                scores.append(step_count)
                break
        
        if len(agent.memory) > 32:
            agent.replay(32)
        
        # 每100轮更新一次目标网络
        if episode % 100 == 0:
            agent.update_target_network()
        
        # 每50轮打印一次平均分数
        if episode % 50 == 0 and episode > 0:
            avg_score = np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
    
    print("训练完成！")
    
    # 保存模型
    model_path = "gate_search_dqn_model.pth"
    torch.save(agent.q_network.state_dict(), model_path)
    print(f"模型已保存为 {model_path}")
    
    return agent

def find_gate_with_dqn():
    """
    使用DQN算法寻找门
    """
    print("开始训练并寻找门...")
    
    # 训练智能体
    trained_agent = train_gate_search_agent()
    
    # 测试智能体
    env = TargetSearchEnvironment("gate")
    state = env.reset()
    
    total_reward = 0
    step_count = 0
    done = False
    
    print("开始测试智能体...")
    
    while not done and step_count < 50:
        action = trained_agent.act(state)
        
        # 打印动作
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        print(f"Step {step_count}: Taking action - {action_names[action]}")
        
        state, reward, done, detection_results = env.step(action)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            print(f"Detected {len(detection_results)} gates")
        
        if done:
            if detection_results:
                print("成功找到门！")
                return "成功找到门！"
            else:
                print("未能找到门")
                return "未能找到门"
    
    result = f"测试完成 - Steps: {step_count}, Total Reward: {total_reward}"
    print(result)
    return result

def execute_tool(tool_name, *args):
    """
    根据工具名称执行对应的DQN操作
    
    Args:
        tool_name (str): 工具名称
        *args: 工具参数
        
    Returns:
        dict: API响应结果
    """
    tool_functions = {
        "find_gate_with_dqn": find_gate_with_dqn,
        "train_gate_search_agent": train_gate_search_agent
    }
    
    if tool_name in tool_functions:
        try:
            result = tool_functions[tool_name]()
            return {"status": "success", "result": str(result)}
        except Exception as e:
            return {"status": "error", "message": f"执行工具时出错: {str(e)}"}
    else:
        print(f"错误: 未知的DQN工具 '{tool_name}'")
        return {"status": "error", "message": f"未知的DQN工具 '{tool_name}'"}

def main():
    """
    主函数，用于直接运行此脚本
    """
    if len(sys.argv) < 2:
        print("用法: python gate_find_dqn.py <tool_name>")
        print("可用工具: find_gate_with_dqn, train_gate_search_agent")
        return

    tool_name = sys.argv[1]
    
    # 执行对应的工具
    response = execute_tool(tool_name)
    print(response)

if __name__ == "__main__":
    main()