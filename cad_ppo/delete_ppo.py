# ppo_dino_reward.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import random
import requests
from PIL import Image
import io
import base64
import tempfile
import os

class CADDINORewardEnvironment(gym.Env):
    """
    CAD删除操作环境 - 使用DINO特征相似度增强奖励函数
    """
    def __init__(self, autocad_server_url="http://localhost:5300", dino_server_url="http://localhost:5200"):
        super(CADDINORewardEnvironment, self).__init__()
        
        # 保存服务器URL
        self.autocad_server_url = autocad_server_url
        self.dino_server_url = dino_server_url
        
        # 加载参考图像路径
        self.reference_image_path = r"E:\code\cad_ppo\图片\right.png"
        
        # 动作空间：连续坐标 (min_x, min_y, max_x, max_y)
        # 输出归一化的坐标值 [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        # 观察空间：CAD界面截图 (84x84 RGB图像)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        
        
        
    def _send_to_autocad_command_line(self, message):
        """
        发送消息到AutoCAD命令行显示
        """
        try:
            response = requests.post(
                f"{self.autocad_server_url}/command/echo",
                json={"message": message},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send message to AutoCAD: {e}")
            return False
    
    def _get_model_bounds(self):
        """
        获取AutoCAD模型空间的边界框
        """
        try:
            response = requests.get(
                f"{self.autocad_server_url}/model/bounds",
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    self._send_to_autocad_command_line(f" Model Bounds: [{result['min_x']:.1f}, {result['min_y']:.1f}, {result['max_x']:.1f}, {result['max_y']:.1f}]")
                    return (
                        result['min_x'], 
                        result['min_y'], 
                        result['max_x'], 
                        result['max_y']
                    )
        except Exception as e:
            print(f"Error getting model bounds: {e}")
        
        # 返回默认边界框
        return (0, 0, 1000, 1000)
        
    # 修改ppo.py文件中的reset方法
    def reset(self, seed=None, options=None):
        """
        重置环境，获取当前CAD界面截图
        """
        super().reset(seed=seed)  # 调用父类的reset方法
        self.step_count = 0
        self.deleted_objects_count = 0
        self.previous_image_path = None
        
        # 执行5次撤销操作
        for i in range(5):
            try:
                response = requests.get(
                    f"{self.autocad_server_url}/edit/undo",
                    timeout=5
                )
                if response.status_code != 200:
                    print(f"Undo operation {i+1} failed with status code: {response.status_code}")
            except Exception as e:
                print(f"Error during undo operation {i+1}: {e}")
        
        obs = self._get_observation()
        
        # 发送重置信息到AutoCAD命令行
        self._send_to_autocad_command_line("Environment reset with 5 undo operations")
        
        return obs, {}  # 返回观测和额外信息字典
    
    def step(self, action):
        """
        执行动作：选择一个区域并删除其中的对象
        action: [min_x, min_y, max_x, max_y] 归一化坐标
        """
        self.step_count += 1
        
        # 确保坐标顺序正确
        min_x_norm = min(action[0], action[2])
        max_x_norm = max(action[0], action[2])
        min_y_norm = min(action[1], action[3])
        max_y_norm = max(action[1], action[3])
        
        # 获取模型空间边界框
        min_x_bound, min_y_bound, max_x_bound, max_y_bound = self._get_model_bounds()
        
        # 将归一化坐标转换为实际坐标
        min_x_actual = min_x_bound + min_x_norm * (max_x_bound - min_x_bound)
        max_x_actual = min_x_bound + max_x_norm * (max_x_bound - min_x_bound)
        min_y_actual = min_y_bound + min_y_norm * (max_y_bound - min_y_bound)
        max_y_actual = min_y_bound + max_y_norm * (max_y_bound - min_y_bound)
        
        # 发送动作信息到AutoCAD命令行
        self._send_to_autocad_command_line(f"Step {self.step_count}: Norm Coords [{min_x_norm:.3f}, {min_y_norm:.3f}, {max_x_norm:.3f}, {max_y_norm:.3f}]")
        self._send_to_autocad_command_line(f"Actual Coords: [{min_x_actual:.1f}, {min_y_actual:.1f}, {max_x_actual:.1f}, {max_y_actual:.1f}]")
        
        # 发送删除区域请求到AutoCAD服务器
        try:
            response = requests.get(
                f"{self.autocad_server_url}/delete-area",
                params={
                    'min_x': min_x_actual,
                    'min_y': min_y_actual,
                    'max_x': max_x_actual,
                    'max_y': max_y_actual
                },
                timeout=10
            )
            
            result = response.json()
            if result['status'] == 'success':
                deleted_count = result['deleted_count']
                self.deleted_objects_count += deleted_count
                
                # 发送删除成功信息到AutoCAD命令行
                self._send_to_autocad_command_line(f"Deleted {deleted_count} objects. Total: {self.deleted_objects_count}")
                
                # 获取操作后的图像
                after_delete_image_path = self._save_current_image_temp()
                
                # 计算奖励（完全基于DINO特征相似度）
                reward = self._calculate_reward(
                    deleted_count, 
                    min_x_norm, min_y_norm, max_x_norm, max_y_norm,  # 使用归一化坐标计算惩罚
                    None, after_delete_image_path
                )
                
                # 清理临时文件
                if after_delete_image_path and os.path.exists(after_delete_image_path):
                    os.remove(after_delete_image_path)
            else:
                reward = -5  # 操作失败惩罚
                # 发送操作失败信息到AutoCAD命令行
                self._send_to_autocad_command_line("Operation failed!")
                
        except Exception as e:
            print(f"Error calling AutoCAD server: {e}")
            reward = -10  # 严重错误惩罚
            # 发送错误信息到AutoCAD命令行
            self._send_to_autocad_command_line(f"Error: {str(e)}")
            
        # 检查是否完成
        terminated = False
        truncated = False
        if self.step_count >= 5:  # 最多10步
            truncated = True
            self._send_to_autocad_command_line("Episode finished!")
            
        # 获取新的观察状态
        obs = self._get_observation()
        
        # 返回五个值: obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, {}
    
 
  
    
    def _save_current_image_temp(self):
        """
        保存当前图像到临时文件，用于DINO特征提取
        """
        try:
            response = requests.get(
                f"{self.autocad_server_url}/screenshot/region/base64",
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['status'] == 'success':
                    # 解码base64图像数据
                    image_data = result['image']
                    image_bytes = base64.b64decode(image_data)
                    
                    # 保存到临时文件
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    temp_file.write(image_bytes)
                    temp_file.close()
                    return temp_file.name
                    
        except Exception as e:
            print(f"Error saving current image: {e}")
        
        return None
    
    def _get_observation(self):
        """
        从AutoCAD服务器获取当前界面截图
        """
        try:
            response = requests.get(
                f"{self.autocad_server_url}/screenshot/region",
                timeout=10
            )
            
            if response.status_code == 200:
                # 将响应内容转换为图像
                image = Image.open(io.BytesIO(response.content))
                # 调整图像大小为84x84
                image = image.resize((84, 84))
                # 转换为numpy数组
                obs = np.array(image)
                return obs
            else:
                # 如果获取失败，返回空白图像
                return np.ones((84, 84, 3), dtype=np.uint8) * 255
                
        except Exception as e:
            print(f"Error getting screenshot: {e}")
            # 返回空白图像
            return np.ones((84, 84, 3), dtype=np.uint8) * 255
    
    def render(self, mode='human'):
        """
        渲染环境（用于调试）
        """
        obs = self._get_observation()
        if mode == 'human':
            cv2.imshow('CAD Environment', obs)
            cv2.waitKey(1)
        return obs

def create_dino_reward_model(env):
    """
    创建使用DINO特征奖励的PPO模型
    """
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./cad_dino_reward_ppo_tensorboard/"
    )
    return model

def train_dino_reward_model():
    """
    训练使用DINO特征奖励的模型
    """
    # 创建环境
    env = CADDINORewardEnvironment()
    
    # 检查环境
    check_env(env)
    
    # 创建模型
    model = create_dino_reward_model(env)
    
    # 训练模型
    print("开始训练使用DINO特征奖励的模型...")
    env._send_to_autocad_command_line("开始训练使用DINO特征奖励的模型...")
    total_timesteps = 50000
    
    # 直接训练，不进行中间评估
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型
    model.save("cad_dino_reward_ppo")
    print("使用DINO特征奖励的模型训练完成并已保存")
    env._send_to_autocad_command_line("使用DINO特征奖励的模型训练完成并已保存")
    
    return model

def evaluate_model(model, env, num_episodes=5):
    """
    评估模型性能
    """
    total_rewards = 0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        total_rewards += episode_reward
    return total_rewards / num_episodes

def test_dino_reward_model():
    """
    测试训练好的使用DINO特征奖励的模型
    """
    # 加载模型
    env = CADDINORewardEnvironment()
    model = PPO.load("cad_dino_reward_ppo", env=env)
    
    # 测试
    obs, _ = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    
    print("开始测试使用DINO特征奖励的模型...")
    env._send_to_autocad_command_line("Starting model test...")
    
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # 打印预测的坐标
        print(f"预测坐标: min_x={action[0]:.3f}, min_y={action[1]:.3f}, max_x={action[2]:.3f}, max_y={action[3]:.3f}")
        print(f"步骤奖励: {reward:.2f}")
        
        # 发送到AutoCAD命令行
        env._send_to_autocad_command_line(f"Predicted coords: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]")
        env._send_to_autocad_command_line(f"Step reward: {reward:.2f}")
    
    print(f"测试完成，总奖励: {total_reward:.2f}")
    env._send_to_autocad_command_line(f"Test completed. Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    # 训练使用坐标输出和DINO奖励的模型
    trained_model = train_dino_reward_model()
    
    # 测试模型
    test_dino_reward_model()