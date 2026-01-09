"""
门搜索PPO实现
使用PPO算法在虚拟环境中搜索目标
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import cv2
from PIL import Image
import base64
from collections import deque
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# 添加当前目录到Python路径，确保能正确导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有的移动控制器
from control_api_tool import ImprovedMovementController

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 导入现有的目标检测功能
from grounding_dino.dino import detect_objects_with_text_transformers


# 配置日志
def setup_logging():
    """设置日志配置"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"gate_find_ppo_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    return logging.getLogger(__name__)


class OccupancyMap:
    """
    占用地图类，用于记录已探索区域
    """
    def __init__(self, map_size=100, resolution=1.0):
        """
        初始化占用地图
        :param map_size: 地图大小 (单位: 格子)
        :param resolution: 每个格子代表的实际距离
        """
        self.map_size = map_size
        self.resolution = resolution
        # 地图初始化: -1-未知, 0-自由, 1-占用
        self.occupancy_map = np.full((map_size, map_size), -1, dtype=np.int8)
        # 探索地图: 0-未探索, 1-已探索
        self.explored_map = np.zeros((map_size, map_size), dtype=np.int8)
        # 当前位置在地图中的坐标
        self.current_pos = np.array([map_size // 2, map_size // 2], dtype=np.int32)  # 初始在地图中心
        self.logger = logging.getLogger(__name__)
        
    def update_position(self, new_pos):
        """
        更新当前位置
        :param new_pos: 新位置 [x, y]
        """
        self.current_pos = new_pos
        
    def mark_explored_around(self, radius=5):
        """
        标记当前位置周围区域为已探索
        :param radius: 探索半径
        """
        x, y = self.current_pos
        x_min = max(0, x - radius)
        x_max = min(self.map_size, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.map_size, y + radius + 1)
        
        self.explored_map[x_min:x_max, y_min:y_max] = 1
        
    def get_newly_explored_ratio(self, prev_explored_cells, radius=5):
        """
        计算新探索区域的比例
        :param prev_explored_cells: 之前已探索的格子数量
        :param radius: 探索半径
        :return: 新探索区域的比例
        """
        x, y = self.current_pos
        x_min = max(0, x - radius)
        x_max = min(self.map_size, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(self.map_size, y + radius + 1)
        
        current_explored_cells = np.sum(self.explored_map[x_min:x_max, y_min:y_max])
        new_explored_cells = current_explored_cells - prev_explored_cells
        
        return new_explored_cells / ((x_max - x_min) * (y_max - y_min)) if (x_max - x_min) * (y_max - y_min) > 0 else 0.0
    
    def export_map_image(self, filepath=None):
        """
        导出占用地图为图片
        :param filepath: 保存路径，如果不提供则自动生成
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"./map_export/map_{timestamp}.png"
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 创建地图的可视化表示
        # -1 (未知) -> 白色 (255, 255, 255)
        # 0 (自由) -> 浅灰色 (200, 200, 200)
        # 1 (占用) -> 黑色 (0, 0, 0)
        # 已探索但未知 -> 深灰色 (150, 150, 150)
        vis_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # 根据占用情况设置颜色
        vis_map[self.occupancy_map == -1] = [255, 255, 255]  # 未知区域白色
        vis_map[self.occupancy_map == 0] = [200, 200, 200]   # 自由区域浅灰
        vis_map[self.occupancy_map == 1] = [0, 0, 0]         # 占用区域黑色
        
        # 已探索但未知的区域设置为深灰色
        unexplored_known = (self.occupancy_map == -1) & (self.explored_map == 1)
        vis_map[unexplored_known] = [150, 150, 150]
        
        # 标记当前位置
        pos_x, pos_y = self.current_pos
        if 0 <= pos_x < self.map_size and 0 <= pos_y < self.map_size:
            # 用红色标记当前位置
            vis_map[pos_x, pos_y] = [255, 0, 0]
            # 增加标记范围使更明显
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = pos_x + dx, pos_y + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        if abs(dx) + abs(dy) <= 1:  # 只标记邻近点，不包括对角
                            vis_map[nx, ny] = [255, 0, 0]
        
        # 保存图片
        plt.figure(figsize=(10, 10))
        plt.imshow(vis_map)
        plt.title("Occupancy Map - Red: Current Position, Black: Obstacles, Gray: Free Space, White: Unknown")
        plt.axis('off')
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()
        
        self.logger.info(f"地图已导出到: {filepath}")
        return filepath


class TargetSearchEnvironment:
    """
    目标搜索环境
    """
    def __init__(self, target_description="gate"):
        self.controller = ImprovedMovementController()
        self.target_description = target_description
        self.step_count = 0
        self.max_steps = 20  # 增加最大步数，给更多探索机会
        self.last_detection_result = None
        self.last_center_distance = float('inf')
        self.logger = logging.getLogger(__name__)
        
        # 记录探索历史，帮助判断是否在原地打转
        self.position_history = []
        self.max_history_length = 10
        
        # 初始化占用地图
        self.occupancy_map = OccupancyMap(map_size=100, resolution=1.0)
        self.prev_explored_cells = 0  # 用于计算信息增益
        
        # 预先初始化检测模型，确保在执行任何动作前模型已加载
        self._warm_up_detection_model()
    
    def _warm_up_detection_model(self):
        """
        预热检测模型，确保在训练开始前模型已加载到内存
        """
        self.logger.info("正在预热检测模型，确保模型已加载...")
        try:
            # 在环境初始化时立即加载模型
            dummy_image = self.capture_screen()
            if dummy_image is not None and dummy_image.size > 0:
                # 对当前屏幕截图进行检测，触发模型加载
                dummy_result = self.detect_target(dummy_image)
                self.logger.info("检测模型已预热完成")
            else:
                self.logger.warning("无法获取初始截图进行模型预热")
        except Exception as e:
            self.logger.warning(f"模型预热过程中出现错误: {e}")
    
    def capture_screen(self):
        """
        截取当前屏幕画面
        """
        # 使用现有的浏览器截图功能，修改为截取名称包含"sifu"的窗口
        try:
            from computer_server.prtsc import capture_window_by_title
            # 截取标题包含"sifu"的窗口
            result = capture_window_by_title("sifu", "sifu_window_capture.png")
            if result:
                # 读取保存的图片
                from PIL import Image
                screenshot = Image.open("sifu_window_capture.png")
                # 转换为numpy数组
                import numpy as np
                screenshot = np.array(screenshot)
                # 由于PIL读取的图像格式是RGB，而OpenCV使用BGR，所以需要转换
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
            else:
                self.logger.warning("未找到包含 'sifu' 的窗口，使用全屏截图")
                import pyautogui
                import numpy as np
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
        except ImportError:
            # 如果没有截图功能，模拟返回一张图片
            self.logger.warning("截图功能不可用，使用模拟图片")
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
            self.logger.debug(f"检测到 {len(detection_results)} 个目标")
            return detection_results
        except Exception as e:
            self.logger.error(f"检测过程中出错: {e}")
            return []
    
    def calculate_exploration_bonus(self):
        """
        计算探索奖励，避免在原地打转
        """
        if len(self.position_history) < 2:
            return 0.0
            
        # 计算与最近位置的差异（这里简单用步数差异表示）
        # 实际应用中可以加入坐标位置的记录
        exploration_bonus = 0.1  # 每步都给一点探索奖励
        
        # 如果位置历史中有很多重复位置，则降低奖励
        unique_positions = len(set(self.position_history))
        total_positions = len(self.position_history)
        
        if total_positions > 0:
            repetition_factor = 1.0 - (unique_positions / total_positions)
            exploration_bonus *= (1.0 - repetition_factor * 0.5)  # 重复越多，奖励越低
        
        return exploration_bonus
    
    def calculate_reward(self, detection_results, prev_distance, action_taken=None, image_shape=None):
        """
        计算奖励值
        """
        reward = 0.0
        
        # 1. 检测到目标的奖励（平滑奖励设计）
        if detection_results:
            # 找到最近的检测框
            min_distance = float('inf')
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2  # 修复：应该是bbox[3]而不是bbox[2]
                
                # 获取图像中心点，如果未提供image_shape，则使用默认值
                if image_shape is not None:
                    img_height, img_width = image_shape[0], image_shape[1]
                    img_center_x = img_width / 2
                    img_center_y = img_height / 2
                else:
                    # 默认值，仅用于向后兼容
                    img_center_x = 320  # 假设图像宽度为640
                    img_center_y = 240  # 假设图像高度为480
                
                # 计算到图像中心的距离
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
            
            # 基于目标中心与图像中心的距离给予奖励，距离越近奖励越高（平滑奖励）
            max_distance = np.sqrt((img_center_x)**2 + (img_center_y)**2)  # 图像对角线长度，即最大可能距离
            centering_reward = 1.0 * (1 - min_distance / max_distance)  # 距离中心越近，奖励越高，降低幅度
            
            # 基于距离改善的奖励（平滑过渡）
            distance_improvement = 0.0
            if prev_distance != float('inf'):  # 只有在之前有有效距离的情况下才比较
                distance_change = prev_distance - min_distance  # 正值表示距离变近
                # 标准化距离改善，使其在合理范围内
                normalized_improvement = distance_change / max_distance
                distance_improvement = 1.0 * normalized_improvement  # 奖励接近目标的行为
            else:
                # 第一次检测到目标时给予一定奖励
                distance_improvement = 0.5
                self.logger.debug(f"首次检测到目标，距离奖励: {distance_improvement:.2f}")
            
            # 组合奖励
            reward = centering_reward + distance_improvement
            
            # 额外奖励：目标在图像中心区域
            if min_distance < 30:  # 非常接近中心
                reward += 2.0
                self.logger.info(f"目标非常接近中心，额外奖励，总奖励: {reward:.2f}")
            elif min_distance < 60:  # 中等距离中心
                reward += 1.0
                self.logger.debug(f"目标接近中心，额外奖励，总奖励: {reward:.2f}")
            elif min_distance < 100:  # 较远但仍在中心区域
                reward += 0.5
                self.logger.debug(f"目标在中心区域，轻微奖励，总奖励: {reward:.2f}")
                
        else:
            # 2. 未检测到目标时的奖励（修改为负奖励鼓励真正找到目标）
            # 基于探索的奖励
            exploration_bonus = self.calculate_exploration_bonus()
            
            # 信息增益奖励 - 鼓励进入新区域
            newly_explored_ratio = self.occupancy_map.get_newly_explored_ratio(self.prev_explored_cells, radius=5)
            info_gain_bonus = newly_explored_ratio * 2.0  # 信息增益奖励
            
            # 鼓励持续探索的小奖励
            exploration_motivation = 0.1  # 鼓励继续探索
            
            # 基于行动的奖励 - 根据执行的动作类型给予小奖励
            action_reward = 0.05  # 每次行动给予小奖励，鼓励探索
            
            # 组合探索奖励 - 但整体为负值，鼓励真正找到目标
            exploration_reward = exploration_bonus + info_gain_bonus + exploration_motivation + action_reward
            
            # 修改：未找到目标时给予负奖励，以更明确地鼓励找到目标
            reward =  0.5*exploration_reward  # 未找到目标时的基础惩罚
            self.logger.debug(f"未检测到目标，探索奖励: {exploration_reward:.2f}, 基础惩罚后: {reward:.2f}, 信息增益: {info_gain_bonus:.2f}")
        # 额外的稳定性奖励/惩罚
        # 避免长时间在原地打转
        if len(self.position_history) > 1:
            recent_positions = self.position_history[-5:] if len(self.position_history) >= 5 else self.position_history
            if len(set(recent_positions)) <= 2:  # 最近位置变化很少
                reward -= 0.2  # 小惩罚，避免原地打转
        
        # 限制奖励范围，防止奖励值过大导致训练不稳定
        reward = max(-2.0, min(reward, 10.0))  # 限制奖励在[-2, 10]范围内
        
        return reward
    
    def step(self, action):
        """
        执行动作并返回新的状态、奖励和是否结束
        动作空间: 0-forward, 1-backward, 2-turn_left, 3-turn_right, 4-strafe_left, 5-strafe_right
        """
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        self.logger.debug(f"执行动作: {action_names[action]}")
        
        # 记录执行动作前的探索格子数
        self.prev_explored_cells = np.sum(self.occupancy_map.explored_map)
        
        # 执行动作 - 增加移动距离和旋转角度
        if action == 0:  # forward - 增加移动距离
            self.controller.move_forward(3)  # 从1改为3
        elif action == 1:  # backward
            self.controller.move_backward(2)  # 从1改为2
        elif action == 2:  # turn_left - 增加旋转角度
            self.controller.turn_left(60)  # 从30改为60
        elif action == 3:  # turn_right - 增加旋转角度
            self.controller.turn_right(60)  # 从30改为60
        elif action == 4:  # strafe_left
            self.controller.strafe_left(2)  # 从1改为2
        elif action == 5:  # strafe_right
            self.controller.strafe_right(2)  # 从1改为2
        
        time.sleep(0.3)  # 等待动作完成，稍作延长以确保动作完成
        
        # 获取新状态（截图）
        new_state = self.capture_screen()
        
        # 更新占用地图中的探索区域
        self.occupancy_map.mark_explored_around(radius=5)
        
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
        
        # 传递图像形状给奖励计算函数
        reward = self.calculate_reward(detection_results, self.last_center_distance, action, new_state.shape)
        
        # 打印动作和奖励
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        print(f"动作: {action_names[action]}, 奖励: {reward:.2f}")
        
        self.last_center_distance = current_distance
        self.last_detection_result = detection_results
        
        # 更新步数
        self.step_count += 1
        
        # 更新位置历史，记录当前状态的特征（如检测结果数量）
        state_feature = len(detection_results)  # 这里用检测到的对象数量作为状态特征
        self.position_history.append(state_feature)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # 检查是否结束
        done = self.step_count >= self.max_steps or (detection_results and current_distance < 50)
        
        if done:
            if detection_results and current_distance < 50:
                self.logger.info(f"在第 {self.step_count} 步找到目标")
            else:
                self.logger.info(f"达到最大步数 {self.max_steps}")
        
        return new_state, reward, done, detection_results
    
    def reset(self):
        """
        重置环境
        """
        self.logger.debug("重置环境")
        self.step_count = 0
        self.last_center_distance = float('inf')
        self.last_detection_result = None
        self.position_history = []  # 重置位置历史
        self.occupancy_map = OccupancyMap(map_size=100, resolution=1.0)  # 重置占用地图
        self.prev_explored_cells = 0
        initial_state = self.capture_screen()
        return initial_state


class ActorCritic(nn.Module):
    """
    PPO使用的Actor-Critic网络
    """
    def __init__(self, input_channels=3, action_space=6, image_height=480, image_width=640):
        super(ActorCritic, self).__init__()
        
        # 卷积层处理图像输入
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 自适应平均池化层，确保输出固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))  # 固定输出为10x10
        
        # 全连接层 - 输入大小是固定的，因为池化层输出固定尺寸
        conv_out_size = 64 * 10 * 10  # 64个通道 * 10 * 10 = 6400
        
        # Actor (策略网络) - 输出动作概率分布
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )
        
        # Critic (价值网络) - 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # 确保输出尺寸一致
        x = x.view(x.size(0), -1)  # 展平
        
        # 分别计算动作概率和状态价值
        action_logits = self.actor(x)  # 先计算未归一化的logits
        action_probs = F.softmax(action_logits, dim=-1)
        
        # 添加数值稳定性：确保概率分布不会变得过于尖锐或平坦
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0 - 1e-8)
        # 重新归一化以确保和为1
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        state_value = self.critic(x)
        
        return action_probs, state_value


class Memory:
    """
    存储智能体的经验数据
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPOAgent:
    """
    PPO智能体
    """
    def __init__(self, state_dim, action_dim, lr=0.0003, betas=(0.9, 0.999), gamma=0.99, K_epochs=80, eps_clip=0.2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        
        self.policy = ActorCritic(state_dim[0], action_dim, state_dim[1], state_dim[2])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim[0], action_dim, state_dim[1], state_dim[2])
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.logger = logging.getLogger(__name__)
    
    def update(self, memory):
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # 归一化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # 修复：添加数值稳定性检查，防止std为0或过小导致NaN
        rewards_std = rewards.std()
        if torch.isnan(rewards_std) or rewards_std < 1e-8:
            rewards_std = torch.tensor(1.0)  # 如果标准差太小或为NaN，使用默认值
        
        rewards = (rewards - rewards.mean()) / (rewards_std + 1e-8)
        
        # 转换为张量
        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        
        # K epochs更新策略
        for _ in range(self.K_epochs):
            # 计算优势
            logprobs, state_values = self.policy(old_states)
            
            # 修复：确保logprobs不会包含NaN或无穷大值
            if torch.isnan(logprobs).any() or torch.isinf(logprobs).any():
                print("警告: logprobs 包含 NaN 或无穷大值，跳过此批次")
                continue
                
            # 稳定化概率分布
            logprobs = torch.clamp(logprobs, min=1e-8, max=1.0 - 1e-8)
            
            dist = torch.distributions.Categorical(logprobs)
            entropy = dist.entropy().mean()
            
            # 计算动作的对数概率
            new_logprobs = dist.log_prob(old_actions)
            
            # 检查new_logprobs是否有异常值
            if torch.isnan(new_logprobs).any() or torch.isinf(new_logprobs).any():
                print("警告: new_logprobs 包含 NaN 或无穷大值，跳过此批次")
                continue
            
            # 计算优势
            advantages = rewards - state_values.detach().squeeze(-1)
            
            # 检查advantages是否有异常值
            if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                print("警告: advantages 包含 NaN 或无穷大值，跳过此批次")
                continue
            
            # 计算比率 (pi_theta / pi_theta__old)
            ratios = torch.exp(new_logprobs - old_logprobs.detach())
            
            # 检查ratios是否有异常值
            if torch.isnan(ratios).any() or torch.isinf(ratios).any():
                print("警告: ratios 包含 NaN 或无穷大值，跳过此批次")
                continue
            
            # 计算PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            critic_loss = self.MseLoss(state_values.squeeze(-1), rewards)
            
            # 检查损失是否有异常值
            if torch.isnan(actor_loss) or torch.isnan(critic_loss):
                print("警告: 损失包含 NaN，跳过此批次")
                continue
            
            # 总损失
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪以防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            
            self.optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def act(self, state, memory):
        """
        根据当前策略选择动作
        """
        state = self._preprocess_state(state)
        state = state.unsqueeze(0)  # 添加批次维度
        
        with torch.no_grad():
            action_probs, state_value = self.policy_old(state)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        
        # 存储状态、动作和对数概率
        memory.states.append(state.squeeze(0))  # 移除批次维度再存储
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.item()
    
    def evaluate(self, state):
        """
        评估状态-动作对的价值
        """
        action_probs, state_value = self.policy(state)
        dist = torch.distributions.Categorical(action_probs)
        
        return dist, state_value.squeeze(-1)
    
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
        
        # 确保图像尺寸与网络期望的一致
        # 如果图像尺寸不是640x480，进行缩放
        expected_height, expected_width = 480, 640
        if state_tensor.shape[1] != expected_height or state_tensor.shape[2] != expected_width:
            # 将单张图像转换为批处理格式以进行resize
            state_tensor = state_tensor.unsqueeze(0)  # 添加批次维度
            state_tensor = F.interpolate(state_tensor, size=(expected_height, expected_width), mode='bilinear', align_corners=False)
            state_tensor = state_tensor.squeeze(0)  # 移除批次维度
        
        return state_tensor


def train_gate_search_ppo_agent(episodes=200, model_path="gate_search_ppo_model.pth", target_description="gate"):
    """
    训练门搜索PPO智能体
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练门搜索PPO智能体，目标: {target_description}")
    
    # 确保模型保存目录存在
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 确保地图保存目录存在
    map_dir = "./map_export"
    if not os.path.exists(map_dir):
        os.makedirs(map_dir)
        logger.info(f"创建地图保存目录: {map_dir}")
    
    # 初始化环境和智能体
    env = TargetSearchEnvironment(target_description)
    
    # 定义状态和动作维度
    state_shape = (3, 480, 640)  # (channels, height, width)
    action_dim = 6  # 6种动作类型
    
    ppo_agent = PPOAgent(state_shape, action_dim)
    memory = Memory()
    
    scores = deque(maxlen=100)
    total_rewards = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        while not done:
            # 选择动作
            action = ppo_agent.act(state, memory)
            
            # 执行动作
            next_state, reward, done, detection_results = env.step(action)
            
            # 存储奖励和终止标志
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.2f}")
                print(f"Episode: {episode+1}/{episodes}, Score: {step_count}, Total Reward: {total_reward:.2f}")
                scores.append(step_count)
                total_rewards.append(total_reward)
                
                # 修复：删除注释符号后的错误代码
                map_path = export_current_map(env, f"./map_export/map_episode_{episode+1}.png")
                logger.info(f"Episode {episode+1} 地图已保存到: {map_path}")
            
                break
        
        # 更新PPO策略
        ppo_agent.update(memory)
        
        # 清空记忆
        memory.clear_memory()
        
        # 每50轮打印一次平均分数
        if episode % 50 == 0 and episode > 0:
            # 确保使用全局numpy实例
            avg_score = np.mean(scores)
            avg_total_reward = np.mean(total_rewards)
            logger.info(f"Episode {episode}, Average Score: {avg_score:.2f}, Average Total Reward: {avg_total_reward:.2f}")
    
    # 训练结束后保存最终地图
    final_map_path = export_current_map(env, f"./map_export/final_map_after_training.png")
    logger.info(f"训练完成后最终地图已保存到: {final_map_path}")
    
    logger.info("PPO训练完成！")
    
    # 保存模型
    torch.save(ppo_agent.policy.state_dict(), model_path)
    logger.info(f"PPO模型已保存为 {model_path}")
    
    return ppo_agent


def evaluate_trained_ppo_agent(model_path="gate_search_ppo_model.pth", episodes=10, target_description="gate"):
    """
    评估已训练好的PPO模型性能
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始评估PPO模型: {model_path}")
    
    # 定义状态和动作维度
    state_shape = (3, 480, 640)
    action_dim = 6
    
    # 创建PPO智能体
    ppo_agent = PPOAgent(state_shape, action_dim)
    
    # 加载已保存的模型
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy.eval()  # 设置为评估模式
        logger.info(f"PPO模型已从 {model_path} 加载")
    else:
        logger.warning(f"模型文件 {model_path} 不存在，使用随机模型")
    
    evaluation_results = []
    
    for episode in range(episodes):
        # 初始化环境
        env = TargetSearchEnvironment(target_description)
        state = env.reset()
        memory = Memory()  # 创建空的记忆对象，仅用于act函数
        
        total_reward = 0
        step_count = 0
        success = False
        
        done = False
        while not done and step_count < 50:
            action = ppo_agent.act(state, memory)
            
            # 清空临时记忆中的数据，因为我们只是在评估
            memory.clear_memory()
            
            # 打印动作
            action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
            logger.info(f"Episode {episode+1}, Step {step_count}: Taking action - {action_names[action]}")
            
            state, reward, done, detection_results = env.step(action)
            total_reward += reward
            step_count += 1
            
            if detection_results:
                logger.info(f"Episode {episode+1}: Detected {len(detection_results)} instances of {target_description}")
                
            if detection_results and env.last_center_distance < 50:
                success = True
                logger.info(f"Episode {episode+1}: 成功找到目标！")
        
        result = {
            'episode': episode + 1,
            'steps': step_count,
            'total_reward': total_reward,
            'success': success
        }
        evaluation_results.append(result)
        
        logger.info(f"Episode {episode+1} 完成 - Steps: {step_count}, Total Reward: {total_reward}, Success: {success}")
    
    # 计算总体统计信息
    successful_episodes = sum(1 for r in evaluation_results if r['success'])
    # 确保使用全局numpy实例
    avg_steps = np.mean([r['steps'] for r in evaluation_results])
    avg_reward = np.mean([r['total_reward'] for r in evaluation_results])
    
    stats = {
        'total_episodes': episodes,
        'successful_episodes': successful_episodes,
        'success_rate': successful_episodes / episodes if episodes > 0 else 0,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'details': evaluation_results
    }
    
    logger.info(f"PPO评估完成 - 总体成功率: {stats['success_rate']*100:.2f}% ({successful_episodes}/{episodes})")
    logger.info(f"平均步数: {avg_steps:.2f}, 平均奖励: {avg_reward:.2f}")
    
    return stats


def load_and_test_ppo_agent(model_path="gate_search_ppo_model.pth", target_description="gate"):
    """
    加载已训练的PPO模型并进行测试
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载PPO模型并测试: {model_path}")
    
    # 定义状态和动作维度
    state_shape = (3, 480, 640)
    action_dim = 6
    
    # 创建PPO智能体
    ppo_agent = PPOAgent(state_shape, action_dim)
    
    # 加载已保存的模型
    if os.path.exists(model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy.eval()  # 设置为评估模式
        logger.info(f"PPO模型已从 {model_path} 加载")
    else:
        logger.error(f"模型文件 {model_path} 不存在")
        return {"status": "error", "message": f"模型文件 {model_path} 不存在"}
    
    # 初始化环境
    env = TargetSearchEnvironment(target_description)
    state = env.reset()
    memory = Memory()  # 创建空的记忆对象，仅用于act函数
    
    total_reward = 0
    step_count = 0
    done = False
    
    logger.info("开始测试PPO智能体...")
    
    while not done and step_count < 50:
        action = ppo_agent.act(state, memory)
        
        # 清空临时记忆中的数据，因为我们只是在评估
        memory.clear_memory()
        
        # 打印动作
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        logger.info(f"Step {step_count}: Taking action - {action_names[action]}")
        
        state, reward, done, detection_results = env.step(action)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"Detected {len(detection_results)} instances of {target_description}")
        
        if done:
            if detection_results and env.last_center_distance < 50:
                logger.info("成功找到目标！")
                
                # 导出最终地图
                map_path = export_current_map(env)
                logger.info(f"最终地图已保存到: {map_path}")
                
                return {"status": "success", "result": f"成功找到{target_description}！", "steps": step_count, "total_reward": total_reward, "map_path": map_path}
            else:
                logger.info("未能找到目标")
                
                # 导出最终地图
                map_path = export_current_map(env)
                logger.info(f"最终地图已保存到: {map_path}")
                
                return {"status": "partial_success", "result": f"未能找到{target_description}", "steps": step_count, "total_reward": total_reward, "map_path": map_path}
    
    result = {"status": "timeout", "result": f"测试完成但未找到目标 - Steps: {step_count}, Total Reward: {total_reward}"}
    
    # 导出最终地图
    map_path = export_current_map(env)
    logger.info(f"最终地图已保存到: {map_path}")
    
    result["map_path"] = map_path
    logger.info(result["result"])
    return result


def find_gate_with_ppo(target_description="gate"):
    """
    使用PPO算法寻找门（训练+测试流程）
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练并使用PPO寻找目标: {target_description}")
    
    # 确保模型保存目录存在
    model_path = "./model/gate_search_ppo_model.pth"
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 训练PPO智能体
    trained_agent = train_gate_search_ppo_agent(target_description=target_description, model_path=model_path)
    
    # 初始化环境
    env = TargetSearchEnvironment(target_description)
    state = env.reset()
    memory = Memory()  # 创建空的记忆对象，仅用于act函数
    
    total_reward = 0
    step_count = 0
    done = False
    
    logger.info("开始测试PPO智能体...")
    
    while not done and step_count < 50:
        action = trained_agent.act(state, memory)
        
        # 清空临时记忆中的数据，因为我们只是在评估
        memory.clear_memory()
        
        # 打印动作
        action_names = ["forward", "backward", "turn_left", "turn_right", "strafe_left", "strafe_right"]
        logger.info(f"Step {step_count}: Taking action - {action_names[action]}")
        
        state, reward, done, detection_results = env.step(action)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"Detected {len(detection_results)} instances of {target_description}")
        
        if done:
            if detection_results and env.last_center_distance < 50:
                logger.info("成功找到目标！")
                
                # 导出最终地图
                map_path = export_current_map(env)
                logger.info(f"最终地图已保存到: {map_path}")
                
                return {"result": "成功找到目标！", "map_path": map_path}
            else:
                logger.info("未能找到目标")
                
                # 导出最终地图
                map_path = export_current_map(env)
                logger.info(f"最终地图已保存到: {map_path}")
                
                return {"result": "未能找到目标", "map_path": map_path}
    
    result = f"测试完成 - Steps: {step_count}, Total Reward: {total_reward}"
    
    # 导出最终地图
    map_path = export_current_map(env)
    logger.info(f"最终地图已保存到: {map_path}")
    
    return {"result": result, "map_path": map_path}


def export_current_map(env, filepath=None):
    """
    导出当前环境的地图
    :param env: TargetSearchEnvironment实例
    :param filepath: 保存路径，如果不提供则自动生成
    """
    if hasattr(env, 'occupancy_map'):
        return env.occupancy_map.export_map_image(filepath)
    else:
        print("环境没有占用地图功能")
        return None


def execute_ppo_tool(tool_name, *args):
    """
    根据工具名称执行对应的PPO操作
    
    Args:
        tool_name (str): 工具名称
        *args: 工具参数
        
    Returns:
        dict: API响应结果
    """
    logger = logging.getLogger(__name__)
    
    tool_functions = {
        "find_gate_with_ppo": find_gate_with_ppo,
        "train_gate_search_ppo_agent": train_gate_search_ppo_agent,
        "evaluate_trained_ppo_agent": evaluate_trained_ppo_agent,
        "load_and_test_ppo_agent": load_and_test_ppo_agent,
        "export_current_map": lambda env, *a: export_current_map(env, *a)
    }
    
    if tool_name in tool_functions:
        try:
            # 特殊处理带参数的函数
            if tool_name == "train_gate_search_ppo_agent":
                episodes = int(args[0]) if args else 200
                model_path = args[1] if len(args) > 1 else "gate_search_ppo_model.pth"
                target_desc = args[2] if len(args) > 2 else "gate"
                result = tool_functions[tool_name](episodes, model_path, target_desc)
            elif tool_name == "evaluate_trained_ppo_agent":
                model_path = args[0] if args else "gate_search_ppo_model.pth"
                episodes = int(args[1]) if len(args) > 1 else 10
                target_desc = args[2] if len(args) > 2 else "gate"
                result = tool_functions[tool_name](model_path, episodes, target_desc)
            elif tool_name == "load_and_test_ppo_agent":
                model_path = args[0] if args else "gate_search_ppo_model.pth"
                target_desc = args[1] if len(args) > 1 else "gate"
                result = tool_functions[tool_name](model_path, target_desc)
            elif tool_name == "find_gate_with_ppo":
                target_desc = args[0] if args else "gate"
                result = tool_functions[tool_name](target_desc)
            elif tool_name == "export_current_map":
                # 这个工具需要特殊的处理方式
                env = args[0] if args else None
                filepath = args[1] if len(args) > 1 else None
                result = export_current_map(env, filepath)
            else:
                result = tool_functions[tool_name]()
            logger.info(f"PPO工具执行成功: {tool_name}")
            return {"status": "success", "result": str(result)}
        except Exception as e:
            logger.error(f"执行PPO工具时出错: {str(e)}")
            import traceback
            traceback.print_exc()  # 添加详细的错误追踪
            return {"status": "error", "message": f"执行PPO工具时出错: {str(e)}"}
    else:
        logger.error(f"错误: 未知的PPO工具 '{tool_name}'")
        print(f"错误: 未知的PPO工具 '{tool_name}'")
        return {"status": "error", "message": f"未知的PPO工具 '{tool_name}'"}


def main():
    """
    主函数，用于直接运行此脚本
    """
    # 设置日志
    logger = setup_logging()
    
    if len(sys.argv) < 2:
        logger.info("用法: python gate_find_ppo.py <tool_name> [args...]")
        logger.info("可用PPO工具: find_gate_with_ppo, train_gate_search_ppo_agent, evaluate_trained_ppo_agent, load_and_test_ppo_agent")
        print("用法: python gate_find_ppo.py <tool_name> [args...]")
        print("可用PPO工具:")
        print("  find_gate_with_ppo [target_desc] - 训练并使用PPO寻找目标")
        print("  train_gate_search_ppo_agent [episodes] [model_path] [target_desc] - 训练PPO智能体")
        print("  evaluate_trained_ppo_agent [model_path] [episodes] [target_desc] - 评估已训练的PPO模型")
        print("  load_and_test_ppo_agent [model_path] [target_desc] - 加载并测试PPO模型")
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    print(response)


if __name__ == "__main__":
    main()