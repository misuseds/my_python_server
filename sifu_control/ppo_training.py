import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import os
import glob
import re
from collections import deque
from torch.distributions import Categorical
import json
from pathlib import Path


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "ppo_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config['config']

CONFIG = load_config()


def setup_logging():
    """设置日志配置"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"ppo_networks_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    # 创建logger实例
    logger = logging.getLogger('ppo_training')
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器，避免重复
    logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter(' %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_and_print(logger, message):
    """
    同时记录日志和打印消息
    """
    
    logger.info(message)


def find_latest_checkpoint(model_path):
    """
    查找最新的检查点文件
    """
    model_dir = os.path.dirname(model_path)
    model_base_name = os.path.basename(model_path).replace('.pth', '')
    
    # 查找所有相关的检查点文件
    checkpoint_pattern = re.compile(rf'{re.escape(model_base_name)}_checkpoint_ep_(\d+)\.pth$')
    found_checkpoints = []
    
    if os.path.exists(model_dir):
        for file in os.listdir(model_dir):
            match = checkpoint_pattern.match(file)
            if match:
                full_path = os.path.join(model_dir, file)
                epoch_num = int(match.group(1))
                found_checkpoints.append((full_path, epoch_num))
    
    if found_checkpoints:
        # 返回具有最高epoch编号的检查点
        latest_checkpoint = max(found_checkpoints, key=lambda x: x[1])
        return latest_checkpoint[0]
    
    return None


def create_environment_and_agent():
    """
    创建环境和智能体
    """
    from ppo_agents import EnhancedTargetSearchEnvironment
    from ppo_agents import EnhancedGRUPPOAgent
    from ppo_agents import GRUMemory
    
    env = EnhancedTargetSearchEnvironment(CONFIG['TARGET_DESCRIPTION'])
    move_action_dim = 4
    turn_action_dim = 2
    
    ppo_agent = EnhancedGRUPPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim
    )
    
    return env, ppo_agent


def load_model(ppo_agent, model_path, logger):
    """
    加载模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"模型加载成功: {model_path}")
        return True
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return False


def initialize_model(model_path, logger, load_existing=True):
    """
    初始化模型，包括加载已存在的模型或创建新模型
    """
    env, ppo_agent = create_environment_and_agent()
    
    start_episode = 0
    
    if load_existing:
        # 首先查找检查点文件
        latest_checkpoint = find_latest_checkpoint(model_path)
        
        if latest_checkpoint:
            # 如果找到检查点，优先加载检查点
            start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
            logger.info(f"从最新检查点加载: {latest_checkpoint}")
        elif os.path.exists(model_path):
            # 没有检查点但主模型存在，加载主模型
            if not load_model(ppo_agent, model_path, logger):
                raise Exception(f"加载模型失败: {model_path}")
            start_episode = 0
        else:
            logger.info(f"模型文件和检查点都不存在: {model_path}，创建新模型开始训练")
            # 确保模型目录存在
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
    
    return env, ppo_agent, start_episode


def run_episode(env, ppo_agent, episode_num, total_episodes, training_mode=True, logger=None, print_debug=False):
    """
    运行单个episode，确保在结束时重置环境
    """
    state = env.reset()  # 重置环境
    total_reward = 0
    step_count = 0
    done = False
    final_area = 0
    success_flag = False
    
    # 为每个episode创建独立的记忆
    from ppo_agents import GRUMemory
    episode_memory = GRUMemory(sequence_length=CONFIG['SEQUENCE_LENGTH'])
    ppo_agent.state_history.clear()
    
    move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
    turn_action_names = ["turn_left", "turn_right"]
    
    while not done:
        if training_mode:
            # 训练模式：使用act方法
            if print_debug:
                move_action, turn_action, move_forward_step, turn_angle, debug_info = ppo_agent.act(
                    state, episode_memory, return_debug_info=True)
            else:
                move_action, turn_action, move_forward_step, turn_angle = ppo_agent.act(
                    state, episode_memory)
        else:
            # 评估模式：使用确定性动作
            move_action, turn_action, move_forward_step, turn_angle = get_deterministic_action(ppo_agent, state)
        
        next_state, reward, done, detection_results = env.step(
            move_action, turn_action, move_forward_step, turn_angle)
        
        # 更新episode_memory中的奖励和终端状态
        if len(episode_memory.rewards) > 0:
            episode_memory.rewards[-1] = reward
            episode_memory.is_terminals[-1] = done
        
        state = next_state
        total_reward += reward
        step_count += 1
        
        if detection_results:
            final_area = max(d['width'] * d['height'] for d in detection_results 
                           if 'width' in d and 'height' in d)
    
    # 检查是否成功找到目标
    climb_detected = any(
        detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
        for detection in detection_results
    )
    success_flag = climb_detected
    
    # 在训练模式下，需要更新智能体
    if training_mode and len(episode_memory.rewards) > 0:
        # 更新智能体
        ppo_agent.update(episode_memory)
    
    # 确保在episode结束时重置环境
    env.reset()
    
    return {
        'total_reward': total_reward,
        'step_count': step_count,
        'final_area': final_area,
        'success_flag': success_flag,
        'detection_results': detection_results
    }
def get_deterministic_action(ppo_agent, state):
    """
    在评估模式下获取确定性动作
    """
    state_tensor = ppo_agent._preprocess_state(state)
    
    # 将当前状态添加到历史记录
    ppo_agent.state_history.append(state_tensor)
    
    # 如果历史记录长度不足，用当前状态填充
    while len(ppo_agent.state_history) < ppo_agent.sequence_length:
        ppo_agent.state_history.appendleft(state_tensor)
    
    # 转换为张量并添加批次维度
    state_seq = torch.stack(list(ppo_agent.state_history)).unsqueeze(0)
    
    # 使用旧策略获取动作概率（但不采样，而是选择最高概率的动作）
    with torch.no_grad():
        move_probs, turn_probs, action_params, state_val, _ = ppo_agent.policy_old(state_seq)
        
        # 选择最高概率的动作
        move_action = torch.argmax(move_probs, dim=-1).item()
        turn_action = torch.argmax(turn_probs, dim=-1).item()
        
        # 获取动作参数
        move_forward_step_raw = action_params[0][0].item()
        turn_angle_raw = action_params[0][1].item()
        
        # 使用配置文件中的参数范围
        MOVE_STEP_MIN = CONFIG.get('MOVE_STEP_MIN', 0.0)
        MOVE_STEP_MAX = CONFIG.get('MOVE_STEP_MAX', 1.0)
        TURN_ANGLE_MIN = CONFIG.get('TURN_ANGLE_MIN', 5.0)
        TURN_ANGLE_MAX = CONFIG.get('TURN_ANGLE_MAX', 60.0)
        
        # 映射到实际范围
        move_forward_step = (torch.tanh(torch.tensor(move_forward_step_raw)) + 1) / 2 * (MOVE_STEP_MAX - MOVE_STEP_MIN) + MOVE_STEP_MIN
        turn_angle = (torch.tanh(torch.tensor(turn_angle_raw)) + 1) / 2 * (TURN_ANGLE_MAX - TURN_ANGLE_MIN) + TURN_ANGLE_MIN

    return move_action, turn_action, move_forward_step, turn_angle


def perform_training_loop(env, ppo_agent, start_episode, total_episodes, logger, model_path):
    """
    执行训练循环
    """
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    # 训练统计变量
    training_stats = {
        'episode_count': 0,
        'successful_episodes': 0,
        'average_reward_history': [],
        'action_distribution': {'forward': 0, 'backward': 0, 'strafe_left': 0, 'strafe_right': 0, 
                               'turn_left': 0, 'turn_right': 0},
        'action_count': 0,
        'training_update_count': 0
    }
    
    for episode in range(start_episode, total_episodes):
        # 每隔一定轮次打印调试信息
        print_debug = (episode % 10 == 0)  # 每10轮打印一次调试信息
        result = run_episode(env, ppo_agent, episode, total_episodes, training_mode=True, logger=logger, print_debug=print_debug)
        
        # 更新统计数据
        scores.append(result['step_count'])
        total_rewards.append(result['total_reward'])
        final_areas.append(result['final_area'])
        training_stats['episode_count'] += 1
        
        if result['success_flag']:
            training_stats['successful_episodes'] += 1
        
        # 使用收敛监控
        convergence_info = ppo_agent.check_convergence_status(
            result['total_reward'], result['step_count'], result['success_flag'])
        
        # 每10轮打印一次收敛报告
        if episode % 10 == 0:
            avg_reward = np.mean(total_rewards) if total_rewards else 0
            log_and_print(logger, f"收敛监控 - 平均奖励: {avg_reward:.3f}, "
                         f"成功率: {convergence_info['recent_success_rate']:.3f}, "
                         f"学习率: {convergence_info['current_learning_rate']:.6f}")
        
        # 检查早停条件
        if convergence_info['should_stop_early']:
            logger.info(f"早停触发: 连续{ppo_agent.patience}轮无改善")
            break
    
    return training_stats

def generate_training_report(training_stats, ppo_agent, logger):
    """
    生成训练报告
    """
    # 打印最终收敛报告
    final_report = ppo_agent.get_convergence_report()
    log_and_print(logger, "\n=== 训练收敛报告 ===")
    for key, value in final_report.items():
        log_and_print(logger, f"{key}: {value}")
    
    # 打印最终统计
    success_rate = training_stats['successful_episodes'] / training_stats['episode_count'] if training_stats['episode_count'] > 0 else 0
    log_and_print(logger, f"\n=== 训练统计 ===")
    log_and_print(logger, f"总轮数: {training_stats['episode_count']}")
    log_and_print(logger, f"成功轮数: {training_stats['successful_episodes']}")
    log_and_print(logger, f"成功率: {success_rate:.3f}")
    log_and_print(logger, f"训练更新次数: {training_stats['training_update_count']}")
    
    return final_report


def continue_training_gru_ppo_agent(model_path=None):
    """
    基于现有GRU模型继续训练 - 使用全局配置，带收敛监控
    """
    config = CONFIG
    if model_path is None:
        model_path = config['MODEL_PATH']
    
    logger = setup_logging()
    logger.info(f"基于现有GRU模型继续训练: {model_path}, 额外训练 {config['EPISODES']} 轮")
    
    # 初始化模型
    env, ppo_agent, start_episode = initialize_model(model_path, logger, load_existing=True)
    
    total_training_episodes = start_episode + config['EPISODES']
    logger.info(f"从第 {start_episode} 轮开始，继续训练 {config['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")
    
    # 执行训练循环
    training_stats = perform_training_loop(env, ppo_agent, start_episode, total_training_episodes, logger, model_path)
    
    # 生成报告
    final_report = generate_training_report(training_stats, ppo_agent, logger)
    
    logger.info("GRU继续训练完成！")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    logger.info(f"更新后的GRU PPO模型已保存为 {model_path}")
    
    return {
        "status": "success", 
        "message": f"继续训练完成，共训练了 {config['EPISODES']} 轮", 
        "final_episode": total_training_episodes,
        "convergence_report": final_report
    }


def train_new_gru_ppo_agent(model_path=None):
    """
    从头开始训练GRU PPO智能体
    """
    if model_path is None:
        model_path = CONFIG['MODEL_PATH']
    
    logger = setup_logging()
    logger.info(f"开始从头训练GRU PPO智能体: {model_path}")
    
    # 初始化模型（不加载现有模型）
    env, ppo_agent, start_episode = initialize_model(model_path, logger, load_existing=False)
    
    total_training_episodes = CONFIG['EPISODES']
    logger.info(f"从第 {start_episode} 轮开始，训练 {CONFIG['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")
    
    # 执行训练循环
    training_stats = perform_training_loop(env, ppo_agent, start_episode, total_training_episodes, logger, model_path)
    
    # 生成报告
    final_report = generate_training_report(training_stats, ppo_agent, logger)
    
    logger.info("GRU从头训练完成！")
    
    torch.save(ppo_agent.policy.state_dict(), model_path)
    logger.info(f"新训练的GRU PPO模型已保存为 {model_path}")
    
    return {
        "status": "success", 
        "message": f"从头训练完成，共训练了 {CONFIG['EPISODES']} 轮", 
        "final_episode": total_training_episodes,
        "convergence_report": final_report
    }


def evaluate_trained_gru_ppo_agent(model_path=None):
    """
    评估已训练的GRU PPO模型
    """
    if model_path is None:
        model_path = CONFIG['MODEL_PATH']
    
    logger = setup_logging()
    logger.info(f"评估已训练的GRU PPO模型: {model_path}")
    
    # 创建环境和智能体
    env, ppo_agent = create_environment_and_agent()
    
    # 加载训练好的模型
    if not load_model(ppo_agent, model_path, logger):
        return {"status": "error", "message": f"模型加载失败"}
    
    # 设置为评估模式
    ppo_agent.policy.eval()
    ppo_agent.policy_old.eval()
    
    evaluation_episodes = CONFIG.get('EVALUATION_EPISODES', 10)
    scores = []
    total_rewards = []
    success_count = 0
    
    for episode in range(evaluation_episodes):
        # 每隔几轮打印一次调试信息
        print_debug = (episode % 2 == 0)  # 每2轮打印一次调试信息
        result = run_episode(env, ppo_agent, episode, evaluation_episodes, training_mode=False, logger=logger, print_debug=print_debug)
        
        scores.append(result['step_count'])
        total_rewards.append(result['total_reward'])
        
        if result['success_flag']:
            success_count += 1
        
        log_and_print(logger, f"Eval Ep: {episode+1}/{evaluation_episodes}, S: {result['step_count']}, "
                     f"R: {result['total_reward']:.3f}, Success: {result['success_flag']}")
    
    # 计算评估指标
    avg_score = np.mean(scores) if scores else 0
    avg_reward = np.mean(total_rewards) if total_rewards else 0
    success_rate = success_count / evaluation_episodes if evaluation_episodes > 0 else 0
    
    evaluation_result = {
        'avg_score': avg_score,
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'total_evaluated_episodes': evaluation_episodes,
        'success_count': success_count
    }
    
    log_and_print(logger, f"\n=== 评估结果 ===")
    log_and_print(logger, f"平均步数: {avg_score:.3f}")
    log_and_print(logger, f"平均奖励: {avg_reward:.3f}")
    log_and_print(logger, f"成功率: {success_rate:.3f}")
    log_and_print(logger, f"成功次数: {success_count}/{evaluation_episodes}")
    
    logger.info("模型评估完成！")
    
    return {
        "status": "success",
        "message": "模型评估完成",
        "evaluation_result": evaluation_result
    }


def execute_ppo_tool(tool_name, *args):
    """
    根据工具名称执行对应的PPO操作
    """
    logger = setup_logging()
    
    tool_functions = {
        "continue_train_gru_ppo_agent": continue_training_gru_ppo_agent,
        "train_new_gru_ppo_agent": train_new_gru_ppo_agent,
        "evaluate_trained_gru_ppo_agent": evaluate_trained_gru_ppo_agent
    }
    
    if tool_name not in tool_functions:
        error_msg = f"错误: 未知的PPO工具 '{tool_name}'"
        logger.error(error_msg)
        log_and_print(logger, error_msg)
        return {"status": "error", "message": error_msg}

    try:
        result = tool_functions[tool_name](*args)
        logger.info(f"PPO工具执行成功: {tool_name}")
        return {"status": "success", "result": str(result)}
    except Exception as e:
        logger.error(f"执行PPO工具时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"执行PPO工具时出错: {str(e)}"}


def main():
    """
    主函数，用于直接运行此脚本
    """
    # 设置日志
    logger = setup_logging()
    import sys
    if len(sys.argv) < 2:
        continue_training_gru_ppo_agent()
        log_and_print(logger, "导入成功！")
        log_and_print(logger, "\n可用的功能:")
        log_and_print(logger, "1. 训练GRU门搜索智能体: train_new_gru_ppo_agent")
        log_and_print(logger, "2. 继续训练GRU门搜索智能体: continue_train_gru_ppo_agent")
        log_and_print(logger, "3. 评估已训练GRU模型: evaluate_trained_gru_ppo_agent")
        
        return

    tool_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 执行对应的工具
    response = execute_ppo_tool(tool_name, *args)
    log_and_print(logger, str(response))


if __name__ == "__main__":
    main()