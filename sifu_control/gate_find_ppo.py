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
from pathlib import Path
from ultralytics import YOLO
import pyautogui  # 添加pyautogui库用于按键操作


# 添加当前目录到Python路径，确保能正确导入同目录下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入现有的移动控制器
from control_api_tool import ImprovedMovementController

# 添加项目根目录到路径，以便导入其他模块
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


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


class TargetSearchEnvironment:
    """
    目标搜索环境
    """
    def __init__(self, target_description="gate"):
        self.controller = ImprovedMovementController()
        self.target_description = target_description
        self.step_count = 0
        self.max_steps = 50  # 减少最大步数，加快训练
        self.last_detection_result = None
        self.last_center_distance = float('inf')
        self.last_area = 0  # 新增：跟踪上一帧目标区域
        self.logger = logging.getLogger(__name__)
        
        # 记录探索历史，帮助判断是否在原地打转
        self.position_history = []
        self.max_history_length = 10
        self.yolo_model = self._load_yolo_model()
        # 预先初始化检测模型，确保在执行任何动作前模型已加载
        self._warm_up_detection_model()
        
        # 新增：定义成功条件的阈值
        self.MIN_GATE_AREA = 15000  # 减小阈值，更容易成功
        self.CENTER_THRESHOLD = 200  # 放宽居中要求
        

    def reset_to_origin(self):
        """
        重置到原点操作：按下ESC键，按Q键，按下回车键，检测门，若无门则等待1秒按回车再次检测，最后再按一次回车
        """
        self.logger.info("执行重置到原点操作")
        print("执行重置到原点操作...")
        
        # 按键操作序列
        pyautogui.press('esc')
        time.sleep(0.3)  # 减少等待时间
        pyautogui.press('q')
        time.sleep(0.3)  # 减少等待时间
        pyautogui.press('enter')
        time.sleep(0.3)  # 减少等待时间
        pyautogui.press('enter')
        time.sleep(0.3)  # 减少等待时间
        # 检测门是否存在
        gate_detected = False

        while not gate_detected :  # 限制尝试次数
            # 重新截图并检测门
            new_state = self.capture_screen()
            detection_results = self.detect_target(new_state)
            
            # 检查是否检测到了门
            for detection in detection_results:
                if detection['label'].lower() == 'gate' or 'gate' in detection['label'].lower():
                    gate_detected = True
                    time.sleep(0.3)
                    pyautogui.press('enter')
                    time.sleep(0.5)
                    pyautogui.press('enter')
                    break
            
            if not gate_detected:
                print(f"未检测到门，等待1秒后按回车重新检测...")
                time.sleep(1)
                pyautogui.press('enter')
                time.sleep(0.3)  # 减少等待时间
               
                        
        # 最后再按一次回车
        pyautogui.press('enter')
        time.sleep(0.3)  # 减少等待时间
        pyautogui.press('enter')
        time.sleep(1.5)  # 减少等待时间
        print("重置到原点操作完成")
        self.logger.info("重置到原点操作完成")

    def _load_yolo_model(self):
        """
        加载YOLO模型
        """
        # 获取项目根目录
        current_dir = Path(__file__).parent
        model_path = current_dir.parent / "models" / "find_gate.pt"
        
        if not model_path.exists():
            self.logger.error(f"YOLO模型文件不存在: {model_path}")
            return None
        
        try:
            model = YOLO(str(model_path))
            self.logger.info(f"成功加载YOLO模型: {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"加载YOLO模型失败: {e}")
            return None

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
                screenshot = pyautogui.screenshot()
                screenshot = np.array(screenshot)
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
                return screenshot
        except ImportError:
            # 如果没有截图功能，模拟返回一张图片
            self.logger.warning("截图功能不可用，使用模拟图片")
            return np.zeros((480, 640, 3), dtype=np.uint8_)
    
    def detect_target(self, image):
        """
        使用YOLO检测目标
        """
        if self.yolo_model is None:
            self.logger.error("YOLO模型未加载，无法进行检测")
            return []
        
        try:
            # 进行预测 - 降低置信度阈值以增加检测敏感性
            results = self.yolo_model.predict(
                source=image,
                conf=0.7,  # 进一步降低置信度阈值，提高检测敏感性
                save=False,
                verbose=False
            )
            
            # 获取检测结果
            detections = []
            result = results[0]  # 获取第一个结果
            
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # 获取边界框坐标 [x1, y1, x2, y2]
                confs = result.boxes.conf.cpu().numpy()  # 获取置信度
                cls_ids = result.boxes.cls.cpu().numpy()  # 获取类别ID
                
                # 获取类别名称（如果模型有类别名称）
                names = result.names if hasattr(result, 'names') else {}
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    conf = confs[i]
                    cls_id = int(cls_ids[i])
                    class_name = names.get(cls_id, f"Class_{cls_id}")
                    
                    # 计算边界框中心坐标
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 计算边界框宽度和高度
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],  # 左上角和右下角坐标
                        'label': class_name,
                        'score': conf,
                        'width': width,
                        'height': height
                    })
            
            self.logger.debug(f"YOLO检测到 {len(detections)} 个目标")
            return detections
        except Exception as e:
            self.logger.error(f"YOLO检测过程中出错: {e}")
            return []
    

    def calculate_reward(self, detection_results, prev_distance, action_taken=None, prev_area=None):
        """
        改进的奖励函数：检测到门就给正反馈，基于目标绝对大小给予奖励
        """
        reward = 0.0
        
        # 基于探索的奖励 - 即使没有检测到门也给一定奖励
        exploration_bonus = 0.02  # 减少探索奖励
        
        if not detection_results or len(detection_results) == 0:
            # 没有检测到目标，给予轻微惩罚但加上探索奖励
            reward = -0.05 + exploration_bonus
            self.logger.debug(f"未检测到目标，奖励: {reward:.2f}")
            return reward, 0
        
        # 找到最近的检测框和最大面积
        min_distance = float('inf')
        max_area = 0
        
        for detection in detection_results:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            img_width = detection.get('img_width', 640)
            img_height = detection.get('img_height', 480)
            
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
            
            area = detection['width'] * detection['height']
            if area > max_area:
                max_area = area
        
        # 基础奖励：只要检测到门就给予正反馈
        base_detection_reward = 0.3  # 减少基础检测奖励
        reward += base_detection_reward
        self.logger.debug(f"检测到门，基础奖励: {base_detection_reward}")
        
        # 基于目标绝对大小的奖励：目标越大，奖励越高
        size_based_reward = 0.0

        size_based_reward = max_area / 10000  # 线性增长
        
        reward += size_based_reward
        self.logger.debug(f"基于目标大小的奖励: {size_based_reward}, 目标面积: {max_area}")
        
        # 基于接近目标的奖励
        if min_distance < prev_distance:
            approach_reward = 0.2
            reward += approach_reward
            self.logger.debug(f"接近目标奖励: {approach_reward}")
        
        # 基于远离中心的惩罚
        if min_distance > prev_distance:
            distance_penalty = -0.05
            reward += distance_penalty
            self.logger.debug(f"远离目标惩罚: {distance_penalty}")
        
        # 探索奖励
        reward += exploration_bonus
        
        return reward, max_area

    def step(self, action, move_forward_step=2, turn_angle=30):
        """
        执行动作并返回新的状态、奖励和是否结束
        动作空间: 0-forward, 1-backward, 2-turn_left, 3-turn_right, 4-strafe_left, 5-strafe_right
        """
        # 解析复合动作 (move_action, turn_action)
        if isinstance(action, tuple) and len(action) == 2:
            move_action, turn_action = action
        else:
            # 如果是单个数字，映射为复合动作
            move_action = action % 5  # 映射到5种移动动作
            turn_action = (action // 5) % 3  # 映射到3种转头动作
        
        move_action_names = ["no_move", "forward", "backward", "strafe_left", "strafe_right"]
        turn_action_names = ["no_turn", "turn_left", "turn_right"]
        
        self.logger.debug(f"执行动作: 移动-{move_action_names[move_action]}, 转头-{turn_action_names[turn_action]}, 步长: {move_forward_step}, 角度: {turn_angle}")
        
        # 根据动作类型动态计算等待时间
        move_wait_time = 0
        turn_wait_time = 0
        
        if move_action in [1, 2, 3, 4]:  # 移动动作
            # 等待时间与移动步长成正比
            move_wait_time = move_forward_step * 0.5  # 0.5秒/单位移动距离
        
        if turn_action in [1, 2]:  # 转向动作
            # 等待时间与转动角度成正比
            turn_wait_time = turn_angle * 0.02  # 0.02秒/度
        
        # 取两个等待时间的最大值
        wait_time = max(move_wait_time, turn_wait_time, 0.3)  # 至少0.3秒
        
        # 确保等待时间不会太长
        wait_time = min(wait_time, 2.0)  # 限制最大等待时间为2.0秒
        
        # 执行移动动作
        if move_action == 0:  # no_move
            pass  # 不执行任何移动动作
        elif move_action == 1:  # forward
            self.controller.move_forward(duration=move_forward_step*2)
        elif move_action == 2:  # backward
            self.controller.move_backward(duration=move_forward_step*2)
        elif move_action == 3:  # strafe_left
            self.controller.strafe_left(duration=move_forward_step*2)
        elif move_action == 4:  # strafe_right
            self.controller.strafe_right(duration=move_forward_step*2)
        
        # 执行转头动作
        if turn_action == 0:  # no_turn
            pass  # 不执行任何转头动作
        elif turn_action == 1:  # turn_left
            self.controller.turn_left(turn_angle*2, duration=turn_angle*0.02)
        elif turn_action == 2:  # turn_right
            self.controller.turn_right(turn_angle*2, duration=turn_angle*0.02)
        
        time.sleep(wait_time)  # 使用动态计算的等待时间
        
        # 获取新状态（截图）
        new_state = self.capture_screen()
        
        # 检测目标
        detection_results = self.detect_target(new_state)
        
        # 计算奖励 - 添加当前目标区域
        current_distance = self.last_center_distance
        current_area = self.last_area  # 获取上一帧的目标区域
        area = 0
        if detection_results:
            # 计算当前最近目标到中心的距离
            min_distance = float('inf')
            max_area = 0
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2  # 修复：应该是bbox[3]而不是bbox[2]
                img_center_x = new_state.shape[1] / 2
                img_center_y = new_state.shape[0] / 2
                # 添加检测信息到detection字典
                detection['img_width'] = new_state.shape[1]
                detection['img_height'] = new_state.shape[0]
                
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                
                # 计算目标区域
                area = detection['width'] * detection['height']
                if area > max_area:
                    max_area = area
            current_distance = min_distance
            current_area = max_area

        reward, new_area = self.calculate_reward(detection_results, self.last_center_distance, action, current_area)
        self.last_center_distance = current_distance
        self.last_area = new_area  # 保存当前目标区域
        self.last_detection_result = detection_results
        
        # 更新步数
        self.step_count += 1
        
        # 检查是否成功找到门（新条件）
        gate_found_and_close = (
            detection_results 
            and current_distance < self.CENTER_THRESHOLD 
            and current_area > self.MIN_GATE_AREA
        )
        
        # 检查是否结束
        done = self.step_count >= self.max_steps or gate_found_and_close
        
        # 如果成功找到门，给予额外奖励
        if gate_found_and_close:
            # 计算快速完成的额外奖励：基于剩余步数给予额外奖励
            remaining_steps = self.max_steps - self.step_count
            speed_bonus = remaining_steps * 0.3  # 减少速度奖励
            reward += 30 + speed_bonus  # 减少成功奖励
            self.logger.info(f"成功找到目标！基础奖励: 30.0, 速度奖励: {speed_bonus}, 当前面积: {current_area}")
        
        # 输出每步得分
        detected_targets = len(detection_results) if detection_results else 0
        print(f"Step {self.step_count}, Area: {current_area:.2f}, Reward: {reward:.2f}, "
            f"Targets Detected: {detected_targets}, Distance to Center: {current_distance:.2f}, "
            f"Done: {done}, Move Action: {move_action_names[move_action]}, Turn Action: {turn_action_names[turn_action]}, "
            f"Move Step: {move_forward_step}, Turn Angle: {turn_angle}, Wait Time: {wait_time:.2f}")
        
        # 更新位置历史，记录当前状态的特征（如检测结果数量）
        state_feature = len(detection_results)  # 这里用检测到的对象数量作为状态特征
        self.position_history.append(state_feature)
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        if done:
            if gate_found_and_close:
                self.logger.info(f"在第 {self.step_count} 步成功找到目标，面积: {current_area}")
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
        self.last_area = 0  # 重置目标区域
        self.last_detection_result = None
        self.position_history = []  # 重置位置历史
        initial_state = self.capture_screen()
        return initial_state


# 极简版CNN特征提取器
class SimpleFeatureExtractor(nn.Module):
    """
    极简版CNN特征提取器，大幅减少参数量
    """
    def __init__(self, input_channels=3):
        super(SimpleFeatureExtractor, self).__init__()
        
        # 极简CNN特征提取器
        self.conv_layers = nn.Sequential(
            # 第一层 - 减少通道数和步长
            nn.Conv2d(input_channels, 8, kernel_size=5, stride=4, padding=2, bias=False),  # 大步长减少尺寸
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            # 第二层
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 第三层
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化替代自适应池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_pool(x)
        return x.view(x.size(0), -1)  # 展平


class UltraSimplifiedActorCritic(nn.Module):
    """
    超简化版Actor-Critic网络，极致减少参数量
    """
    def __init__(self, input_channels=3, action_space=6, image_height=480, image_width=640):
        super(UltraSimplifiedActorCritic, self).__init__()
        
        # 使用极简特征提取器
        self.feature_extractor = SimpleFeatureExtractor(input_channels)
        
        conv_out_size = 32  # 32 * 1 * 1
        
        # 超简化的全连接层
        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out_size, 48),  # 减少隐藏层大小
            nn.ReLU(),
        )
        
        # 共享特征，分别输出不同的头
        self.move_action_head = nn.Linear(48, 5)  # 5种移动动作
        self.turn_action_head = nn.Linear(48, 3)  # 3种转头动作
        self.param_head = nn.Linear(48, 2)  # 步长和角度
        self.value_head = nn.Linear(48, 1)  # 价值估计

    def forward(self, x):
        features = self.feature_extractor(x)
        shared_features = self.shared_fc(features)  # [batch_size, 48]
        
        move_action_probs = F.softmax(self.move_action_head(shared_features), dim=-1)
        turn_action_probs = F.softmax(self.turn_action_head(shared_features), dim=-1)
        action_params = torch.sigmoid(self.param_head(shared_features))
        state_value = self.value_head(shared_features)
        
        return move_action_probs, turn_action_probs, action_params, state_value


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
        self.action_params = []


    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_params[:]


class PPOAgent:
    """
    PPO智能体 - 使用简化网络
    """
    def __init__(self, state_dim, action_dim, lr=0.001, betas=(0.9, 0.999), 
                 gamma=0.95, K_epochs=2, eps_clip=0.1):  # 进一步减少K_epochs和eps_clip
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs  # 进一步减少更新轮数
        self.eps_clip = eps_clip
        
        input_channels = 3
        height, width = 480, 640
        # 使用超简化网络
        self.policy = UltraSimplifiedActorCritic(input_channels, action_dim, height, width)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas, weight_decay=1e-6)  # 减少权重衰减
        self.policy_old = UltraSimplifiedActorCritic(input_channels, action_dim, height, width)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.logger = logging.getLogger(__name__)

    def update(self, memory):
        if len(memory.rewards) == 0:
            return
            
        # 计算折扣奖励
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        if len(memory.states) == 0:
            return
            
        # 减少批次大小以加快训练
        old_states = torch.stack(memory.states).detach()
        old_move_actions = torch.tensor([a[0] if isinstance(a, tuple) else a % 5 for a in memory.actions], dtype=torch.long)
        old_turn_actions = torch.tensor([a[1] if isinstance(a, tuple) else (a // 5) % 3 for a in memory.actions], dtype=torch.long)
        old_action_params = torch.tensor(memory.action_params if hasattr(memory, 'action_params') else [[0, 0]] * len(memory.actions), dtype=torch.float32)
        old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32)
        
        # 进一步减少更新轮数和批次大小
        for epoch in range(self.K_epochs):
            batch_size = 8  # 进一步减小批次大小以节省显存
            for i in range(0, len(old_states), batch_size):
                batch_states = old_states[i:i+batch_size]
                batch_move_actions = old_move_actions[i:i+batch_size]
                batch_turn_actions = old_turn_actions[i:i+batch_size]
                batch_action_params = old_action_params[i:i+batch_size]
                batch_logprobs = old_logprobs[i:i+batch_size]
                batch_rewards = rewards[i:i+batch_size]
                
                move_action_probs, turn_action_probs, action_params, state_values = self.policy(batch_states)
                
                move_dist = torch.distributions.Categorical(move_action_probs)
                turn_dist = torch.distributions.Categorical(turn_action_probs)
                
                new_move_logprobs = move_dist.log_prob(batch_move_actions)
                new_turn_logprobs = turn_dist.log_prob(batch_turn_actions)
                new_logprobs = new_move_logprobs + new_turn_logprobs
                
                advantages = batch_rewards - state_values.detach().squeeze(-1)
                ratios = torch.exp(new_logprobs - batch_logprobs.detach())
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                param_loss = F.mse_loss(action_params, batch_action_params) if batch_action_params.numel() > 0 else 0
                critic_loss = self.MseLoss(state_values.squeeze(-1), batch_rewards)
                
                # 简化损失计算
                loss = actor_loss + 0.5 * critic_loss + 0.1 * param_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                # 进一步减少梯度裁剪幅度
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
        
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def act(self, state, memory):
        """
        根据当前策略选择动作，添加调试信息
        """
        state = self._preprocess_state(state)
        
        # 添加批次维度进行推理
        state_batch = state.unsqueeze(0)  # 添加批次维度
        
        with torch.no_grad():
            move_action_probs, turn_action_probs, action_params, state_value = self.policy_old(state_batch)
            move_dist = torch.distributions.Categorical(move_action_probs)
            turn_dist = torch.distributions.Categorical(turn_action_probs)
            
            move_action = move_dist.sample()
            turn_action = turn_dist.sample()
            
            # 计算组合动作的对数概率
            move_logprob = move_dist.log_prob(move_action)
            turn_logprob = turn_dist.log_prob(turn_action)
            action_logprob = move_logprob + turn_logprob  # 总对数概率
        
        # 将连续参数转换为实际的步长和角度值
        # 将[0,1]范围映射到实际范围
        scaled_params = action_params.squeeze(0)  # 移除批次维度
        move_forward_step = 1 + 3 * scaled_params[0].item()  # 映射到[1, 4]范围
        turn_angle = 5 + 30 * scaled_params[1].item()  # 映射到[5, 35]范围
        
        # 存储状态、动作和对数概率
        # 不要存储带批次维度的状态，只存储原始状态
        memory.states.append(state)  # 不带批次维度
        memory.actions.append((move_action.item(), turn_action.item()))  # 存储复合动作元组
        memory.logprobs.append(action_logprob.item())  # 确保是标量值
        if not hasattr(memory, 'action_params'):
            memory.action_params = []
        memory.action_params.append(scaled_params.tolist())  # 存储动作参数
        
        return (move_action.item(), turn_action.item()), move_forward_step, turn_angle
    
    def _preprocess_state(self, state):
        """
        预处理状态（图像），保持原始尺寸但优化处理
        """
        # 将BGR转为RGB
        if len(state.shape) == 3:
            state_rgb = cv2.cvtColor(state, cv2.COLOR_BGR2RGB)
        else:
            state_rgb = state
        
        # 转换为tensor并归一化
        state_tensor = torch.FloatTensor(state_rgb).permute(2, 0, 1) / 255.0
        
        # 保持原始尺寸
        return state_tensor


def train_gate_search_ppo_agent_optimized(episodes=50, model_path="gate_search_ppo_model.pth", target_description="gate"):
    """
    优化的门搜索PPO智能体训练 - 快速版本
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练门搜索PPO智能体，目标: {target_description}")
    
    # 确保模型保存目录存在
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 初始化环境和智能体
    env = TargetSearchEnvironment(target_description)
    
    action_dim = 5
    
    # 使用优化的超参数 - 更快的训练
    ppo_agent = PPOAgent(
        (3, 480, 640), 
        action_dim,
        lr=0.001,        # 提高学习率以加快收敛
        gamma=0.9,       # 降低折扣因子以加快学习
        K_epochs=2,      # 减少更新轮数
        eps_clip=0.1     
    )
    memory = Memory()
    
    scores = deque(maxlen=50)  # 减少长度
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    batch_memory = Memory()
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        final_area = 0
        
        while not done:
            action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
            
            # 优化：减少环境交互延迟
            next_state, reward, done, detection_results = env.step(action, move_forward_step, turn_angle)
            
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if detection_results:
                final_area = max(d['width'] * d['height'] for d in detection_results)
            
            if done:
                if isinstance(action, tuple) and len(action) == 2:
                    move_action, turn_action = action
                else:
                    move_action = action % 5
                    turn_action = (action // 5) % 3
                    
                move_action_names = ["no_move", "forward", "backward", "strafe_left", "strafe_right"]
                turn_action_names = ["no_turn", "turn_left", "turn_right"]
                
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.2f}, Final Area: {final_area:.2f}")
                print(f"Ep: {episode+1}/{episodes}, S: {step_count}, R: {total_reward:.2f}, A: {final_area:.2f}")
                scores.append(step_count)
                total_rewards.append(total_reward)
                final_areas.append(final_area)
                
                env.reset_to_origin()
                break

        # 累积批量数据
        batch_memory.states.extend(memory.states)
        batch_memory.actions.extend(memory.actions)
        batch_memory.logprobs.extend(memory.logprobs)
        batch_memory.rewards.extend(memory.rewards)
        batch_memory.is_terminals.extend(memory.is_terminals)
        if hasattr(memory, 'action_params'):
            batch_memory.action_params.extend(memory.action_params)
        
        memory.clear_memory()
        
        # 每3个episode更新一次 - 更频繁更新
        if (episode + 1) % 3 == 0 and len(batch_memory.rewards) > 0:
            print(f"更新策略 - Episodes {(episode-2)} to {episode+1}...")
            update_start_time = time.time()
            ppo_agent.update(batch_memory)
            update_end_time = time.time()
            print(f"策略更新耗时: {update_end_time - update_start_time:.2f}s")
            
            batch_memory.clear_memory()
        
        # 保存检查点
        if (episode + 1) % 10 == 0:
            checkpoint_path = model_path.replace('.pth', f'_checkpoint_ep_{episode+1}.pth')
            torch.save(ppo_agent.policy.state_dict(), checkpoint_path)
            logger.info(f"检查点保存: {episode+1}")

        if episode % 5 == 0 and episode > 0:
            avg_score = np.mean(scores) if scores else 0
            avg_total_reward = np.mean(total_rewards) if total_rewards else 0
            avg_final_area = np.mean(final_areas) if final_areas else 0
            logger.info(f"Episode {episode}, Avg Score: {avg_score:.2f}, Avg Reward: {avg_total_reward:.2f}")
    
    # 最终更新
    if len(batch_memory.rewards) > 0:
        print("更新最终策略...")
        ppo_agent.update(batch_memory)
        batch_memory.clear_memory()
    
    logger.info("PPO训练完成！")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    logger.info(f"PPO模型已保存为 {model_path}")
    
    return ppo_agent


def evaluate_trained_ppo_agent(model_path="gate_search_ppo_model.pth", episodes=5, target_description="gate"):
    """
    评估已训练好的PPO模型性能 - 减少评估episode数量
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始评估PPO模型: {model_path}")
    
    # 定义状态和动作维度
    action_dim = 5
    
    # 创建PPO智能体
    ppo_agent = PPOAgent((3, 480, 640), action_dim)
    
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
        while not done and step_count < 30:  # 减少最大步数
            action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
            
            # 清空临时记忆中的数据，因为我们只是在评估
            memory.clear_memory()
            
            # 打印动作
            if isinstance(action, tuple) and len(action) == 2:
                move_action, turn_action = action
            else:
                move_action = action % 5
                turn_action = (action // 5) % 3
                
            move_action_names = ["no_move", "forward", "backward", "strafe_left", "strafe_right"]
            turn_action_names = ["no_turn", "turn_left", "turn_right"]
            
            logger.info(f"Episode {episode+1}, Step {step_count}: Taking action - Move: {move_action_names[move_action]}, Turn: {turn_action_names[turn_action]}, "
                       f"Step: {move_forward_step}, Turn Angle: {turn_angle}")
            
            state, reward, done, detection_results = env.step(action, move_forward_step, turn_angle)
            total_reward += reward
            step_count += 1
            
            if detection_results:
                logger.info(f"Episode {episode+1}: Detected {len(detection_results)} instances of {target_description}")
                
            # 检查是否成功找到门
            current_distance = float('inf')
            current_area = 0
            if detection_results:
                for detection in detection_results:
                    bbox = detection['bbox']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    img_center_x = state.shape[1] / 2
                    img_center_y = state.shape[0] / 2
                    distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                    area = detection['width'] * detection['height']
                    if distance < current_distance:
                        current_distance = distance
                        current_area = area
            
            if detection_results and current_distance < env.CENTER_THRESHOLD and current_area > env.MIN_GATE_AREA:
                success = True
                logger.info(f"Episode {episode+1}: 成功找到目标！")
        
        result = {
            'episode': episode + 1,
            'steps': step_count,
            'total_reward': total_reward,
            'success': success,
            'final_area': current_area
        }
        evaluation_results.append(result)
        
        # 在每个episode结束后执行重置到原点操作
        env.reset_to_origin()
        
        logger.info(f"Episode {episode+1} 完成 - Steps: {step_count}, Total Reward: {total_reward}, Success: {success}, Final Area: {current_area}")
    
    # 计算总体统计信息
    successful_episodes = sum(1 for r in evaluation_results if r['success'])
    avg_steps = np.mean([r['steps'] for r in evaluation_results])
    avg_reward = np.mean([r['total_reward'] for r in evaluation_results])
    avg_final_area = np.mean([r['final_area'] for r in evaluation_results])
    
    stats = {
        'total_episodes': episodes,
        'successful_episodes': successful_episodes,
        'success_rate': successful_episodes / episodes if episodes > 0 else 0,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
        'avg_final_area': avg_final_area,
        'details': evaluation_results
    }
    
    logger.info(f"PPO评估完成 - 总体成功率: {stats['success_rate']*100:.2f}% ({successful_episodes}/{episodes})")
    logger.info(f"平均步数: {avg_steps:.2f}, 平均奖励: {avg_reward:.2f}, 平均最终面积: {avg_final_area:.2f}")
    
    return stats


def load_and_test_ppo_agent(model_path="gate_search_ppo_model.pth", target_description="gate"):
    """
    加载已训练的PPO模型并进行测试
    """
    logger = logging.getLogger(__name__)
    logger.info(f"加载PPO模型并测试: {model_path}")
    
    # 定义动作维度
    action_dim = 5
    
    # 创建PPO智能体
    ppo_agent = PPOAgent((3, 480, 640), action_dim)
    
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
    
    while not done and step_count < 30:  # 减少最大步数
        action, move_forward_step, turn_angle = ppo_agent.act(state, memory)
        
        # 清空临时记忆中的数据，因为我们只是在评估
        memory.clear_memory()
        
        # 打印动作
        if isinstance(action, tuple) and len(action) == 2:
            move_action, turn_action = action
        else:
            move_action = action % 5
            turn_action = (action // 5) % 3
            
        move_action_names = ["no_move", "forward", "backward", "strafe_left", "strafe_right"]
        turn_action_names = ["no_turn", "turn_left", "turn_right"]
        
        logger.info(f"Step {step_count}: Taking action - Move: {move_action_names[move_action]}, Turn: {turn_action_names[turn_action]}, "
                   f"Step: {move_forward_step}, Turn Angle: {turn_angle}")
        
        state, reward, done, detection_results = env.step(action, move_forward_step, turn_angle)
        total_reward += reward
        step_count += 1
        
        if detection_results:
            logger.info(f"Detected {len(detection_results)} instances of {target_description}")
        
        # 检查是否成功找到门
        current_distance = float('inf')
        current_area = 0
        if detection_results:
            for detection in detection_results:
                bbox = detection['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                img_center_x = state.shape[1] / 2
                img_center_y = state.shape[0] / 2
                distance = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
                area = detection['width'] * detection['height']
                if distance < current_distance:
                    current_distance = distance
                    current_area = area
        
        if done and detection_results and current_distance < env.CENTER_THRESHOLD and current_area > env.MIN_GATE_AREA:
            logger.info("成功找到目标！")
            return {"status": "success", "result": f"成功找到{target_description}！", "steps": step_count, "total_reward": total_reward}
        elif done:
            logger.info("未能找到目标")
            return {"status": "partial_success", "result": f"未能找到{target_description}", "steps": step_count, "total_reward": total_reward}
    
    # 执行重置到原点操作
    env.reset_to_origin()
    
    result = {"status": "timeout", "result": f"测试完成但未找到目标 - Steps: {step_count}, Total Reward: {total_reward}"}
    logger.info(result["result"])
    return result


def find_gate_with_ppo(target_description="gate"):
    """
    使用PPO算法寻找门（训练+测试流程）- 快速版本
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始训练并使用PPO寻找目标: {target_description}")
    
    # 确保模型保存目录存在
    model_path = "./model/gate_search_ppo_model.pth"
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
        logger.info(f"创建模型保存目录: {model_dir}")
    
    # 训练PPO智能体 - 减少训练episode数量
    trained_agent = train_gate_search_ppo_agent_optimized(
        episodes=30,  # 减少训练episodes数量
        target_description=target_description, 
        model_path=model_path
    )
    
    # 测试训练好的智能体
    test_result = load_and_test_ppo_agent(model_path, target_description)
    
    return test_result


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
        "train_gate_search_ppo_agent": train_gate_search_ppo_agent_optimized,
        "evaluate_trained_ppo_agent": evaluate_trained_ppo_agent,
        "load_and_test_ppo_agent": load_and_test_ppo_agent,
    }
    
    if tool_name in tool_functions:
        try:
            # 特殊处理带参数的函数
            if tool_name == "train_gate_search_ppo_agent":
                episodes = int(args[0]) if args else 30  # 默认减少训练episode
                model_path = args[1] if len(args) > 1 else "gate_search_ppo_model.pth"
                target_desc = args[2] if len(args) > 2 else "gate"
                result = tool_functions[tool_name](episodes, model_path, target_desc)
            elif tool_name == "evaluate_trained_ppo_agent":
                model_path = args[0] if args else "gate_search_ppo_model.pth"
                episodes = int(args[1]) if len(args) > 1 else 5  # 减少评估episode
                target_desc = args[2] if len(args) > 2 else "gate"
                result = tool_functions[tool_name](model_path, episodes, target_desc)
            elif tool_name == "load_and_test_ppo_agent":
                model_path = args[0] if args else "gate_search_ppo_model.pth"
                target_desc = args[1] if len(args) > 1 else "gate"
                result = tool_functions[tool_name](model_path, target_desc)
            elif tool_name == "find_gate_with_ppo":
                target_desc = args[0] if args else "gate"
                result = tool_functions[tool_name](target_desc)
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
        print("导入成功！")
        print("\n可用的功能:")
        print("1. 训练门搜索智能体: train_gate_search_ppo_agent")
        print("2. 寻找门（包含训练）: find_gate_with_ppo")
        print("3. 评估已训练模型: evaluate_trained_ppo_agent")
        print("4. 加载并测试模型: load_and_test_ppo_agent")
        
        # 运行快速训练
        print("\n=== 开始快速训练门搜索智能体 ===")
        try:
            agent = train_gate_search_ppo_agent_optimized(
                episodes=30, 
                model_path="./model/fast_gate_search_ppo_model.pth",
                target_description="gate"
            )
            print("\n快速训练完成！")
        except Exception as e:
            logger.error(f"快速训练出错: {str(e)}")
            import traceback
            traceback.print_exc()
        
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    print(response)


if __name__ == "__main__":
    main()