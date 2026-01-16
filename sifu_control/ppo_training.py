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
    import time
    import logging
    import os
    from pathlib import Path
    
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
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    print(message)
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


def continue_training_gru_ppo_agent(model_path=None):
    """
    基于现有GRU模型继续训练 - 使用全局配置，带收敛监控
    """
    config = CONFIG
    convergence_info = {}
    # 如果没有传入model_path，使用默认路径
    if model_path is None:
        model_path = config['MODEL_PATH']
    
    logger = setup_logging()
    logger.info(f"基于现有GRU模型继续训练: {model_path}, 额外训练 {config['EPISODES']} 轮")
    
    from ppo_agents import EnhancedTargetSearchEnvironment
    env = EnhancedTargetSearchEnvironment(config['TARGET_DESCRIPTION'])
    
    move_action_dim = 4
    turn_action_dim = 2
    
    from ppo_agents import EnhancedGRUPPOAgent
    ppo_agent = EnhancedGRUPPOAgent(
        (3, 480, 640), 
        move_action_dim,
        turn_action_dim
    )
    
    # 首先查找检查点文件
    latest_checkpoint = find_latest_checkpoint(model_path)
    
    if latest_checkpoint:
        # 如果找到检查点，优先加载检查点
        start_episode = ppo_agent.load_checkpoint(latest_checkpoint)
        logger.info(f"从最新检查点加载: {latest_checkpoint}")
    elif os.path.exists(model_path):
        # 没有检查点但主模型存在，加载主模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
            ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"从主模型文件加载: {model_path}")
            start_episode = 0
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return {"status": "error", "message": f"加载模型失败: {e}"}
    else:
        # 模型文件和检查点都不存在，创建新模型
        logger.info(f"模型文件和检查点都不存在: {model_path}，创建新模型开始训练")
        start_episode = 0
        
        # 确保模型目录存在
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    total_training_episodes = start_episode + config['EPISODES']
    
    logger.info(f"从第 {start_episode} 轮开始，继续训练 {config['EPISODES']} 轮，总共到第 {total_training_episodes} 轮")
    
    # 训练统计变量
    training_stats = {
        'episode_count': 0,
        'successful_episodes': 0,
        'average_reward_history': [],
        'action_distribution': {'forward': 0, 'backward': 0, 'strafe_left': 0, 'strafe_right': 0, 
                               'turn_left': 0, 'turn_right': 0},
        'action_count': 0,
        'training_update_count': 0  # 训练更新次数
    }
    
    for episode in range(start_episode, total_training_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        final_area = 0
        success_flag = False  # 任务成功标志
        
        # 为每个episode创建独立的记忆
        episode_memory = GRUMemory(sequence_length=config['SEQUENCE_LENGTH'])
        ppo_agent.state_history.clear()
        
        while not done:
            move_action, turn_action, move_forward_step, turn_angle, debug_info = ppo_agent.act(state, episode_memory, return_debug_info=True)
            
            # 更新动作分布统计
            move_action_names = ["forward", "backward", "strafe_left", "strafe_right"]
            turn_action_names = ["turn_left", "turn_right"]
            training_stats['action_distribution'][move_action_names[move_action]] += 1
            training_stats['action_distribution'][turn_action_names[turn_action]] += 1
            training_stats['action_count'] += 1
            
            # 打印神经网络输出和动作参数在一起 - 记录到日志
            if step_count % 5 == 0:  # 每5步打印一次，避免日志过多
                log_and_print(logger, f"\n--- Episode {episode+1}, Step {step_count} ---")
                log_and_print(logger, f"GRU前5个值: {[f'{val:.2f}' for val in debug_info['gru_last_output'][0][:5]]}")
                
                # Move动作分支输出 - 4个离散动作的概率
                move_probs = debug_info['move_logits'][0]
                move_softmax = F.softmax(torch.tensor(move_probs), dim=0).numpy()
                log_and_print(logger, f"MoveProbabilities: {[f'{p:.3f}' for p in move_softmax]}")
                
                # Turn动作分支输出 - 2个离散动作的概率
                turn_probs = debug_info['turn_logits'][0]
                turn_softmax = F.softmax(torch.tensor(turn_probs), dim=0).numpy()
                log_and_print(logger, f"TurnProbabilities: {[f'{p:.3f}' for p in turn_softmax]}")
                
                # 价值函数输出
                log_and_print(logger, f"价值函数输出: {debug_info['value'][0][0]:.3f}")
                
                # 动作和参数信息
                log_and_print(logger, f"执行动作: Move Action: {move_action_names[move_action]}, Turn Action: {turn_action_names[turn_action]}, Move Step: {move_forward_step:.3f}, Turn Angle: {turn_angle:.3f}")
                log_and_print(logger, f"------------------\n")
            
            next_state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
            
            # 更新episode_memory中的奖励和终端状态
            if len(episode_memory.rewards) > 0:
                episode_memory.rewards[-1] = reward
                episode_memory.is_terminals[-1] = done
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if detection_results:
                final_area = max(d['width'] * d['height'] for d in detection_results if 'width' in d and 'height' in d)
            
            if done:
                # 检查是否成功找到目标
                climb_detected = any(
                    detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
                    for detection in detection_results
                )
                success_flag = climb_detected
                
                if success_flag:
                    training_stats['successful_episodes'] += 1
                
                logger.info(f"Episode: {episode}, Score: {step_count}, Total Reward: {total_reward:.3f}, Final Area: {final_area:.3f}, Success: {success_flag}")
                log_and_print(logger, f"Ep: {episode+1}/{total_training_episodes}, S: {step_count}, R: {total_reward:.3f}, A: {final_area:.3f}, "
                    f"Success: {success_flag}, MAct: {move_action_names[move_action]}, TAct: {turn_action_names[turn_action]}, "
                    f"MStep: {move_forward_step:.3f}, TAngle: {turn_angle:.3f}")
                
                scores.append(step_count)
                total_rewards.append(total_reward)
                final_areas.append(final_area)
                training_stats['episode_count'] += 1
                
                # 使用收敛监控
                convergence_info = ppo_agent.check_convergence_status(total_reward, step_count, success_flag)
                
                # 每10轮打印一次收敛报告
                if episode % 10 == 0:
                    avg_reward = np.mean(total_rewards) if total_rewards else 0
                    avg_steps = np.mean(scores) if scores else 0
                    success_rate = training_stats['successful_episodes'] / training_stats['episode_count'] if training_stats['episode_count'] > 0 else 0
                    
                    log_and_print(logger, f"收敛监控 - 平均奖励: {avg_reward:.3f}, "
                        f"平均步数: {avg_steps:.1f}, "
                        f"成功率: {success_rate:.3f}, "
                        f"学习率: {convergence_info['current_learning_rate']:.6f}, "
                        f"剩余耐心: {convergence_info['current_patience']}")

                    # 打印动作分布
                    if training_stats['action_count'] > 0:
                        log_and_print(logger, f"动作分布 - 前进: {training_stats['action_distribution']['forward']/training_stats['action_count']:.2f}, "
                                              f"后退: {training_stats['action_distribution']['backward']/training_stats['action_count']:.2f}, "
                                              f"左平移: {training_stats['action_distribution']['strafe_left']/training_stats['action_count']:.2f}, "
                                              f"右平移: {training_stats['action_distribution']['strafe_right']/training_stats['action_count']:.2f}, "
                                              f"左转: {training_stats['action_distribution']['turn_left']/training_stats['action_count']:.2f}, "
                                              f"右转: {training_stats['action_distribution']['turn_right']/training_stats['action_count']:.2f}")
                
                # 检查早停条件
                if convergence_info['should_stop_early']:
                    logger.info(f"早停触发: 连续{ppo_agent.patience}轮无改善")
                    log_and_print(logger, f"早停触发: 连续{ppo_agent.patience}轮无改善，训练结束")
                    break
                
                # 重置环境并获取新的初始状态
                env.reset_to_origin()
                state = env.reset()
                break

        # 每个episode结束后立即更新策略（关键改进）
        # 实现第一轮后，之后每5轮训练一次
        should_update = (episode == start_episode) or ((episode - start_episode) % 5 == 4)  # 第一轮或者每5轮训练一次
        
        if len(episode_memory.rewards) > 0 and should_update:
            log_and_print(logger, f"更新策略 - Episode {episode+1} (训练第 {training_stats['training_update_count']+1} 次)...")
            update_start_time = time.time()
            ppo_agent.update(episode_memory)
            update_end_time = time.time()
            log_and_print(logger, f"策略更新耗时: {update_end_time - update_start_time:.2f}s")
            training_stats['training_update_count'] += 1
            
            # 清空episode记忆
            episode_memory.clear_memory()
        elif len(episode_memory.rewards) > 0:
            # 如果不需要训练，但内存有数据，也要清空
            episode_memory.clear_memory()
        
        if (episode + 1) % config['CHECKPOINT_INTERVAL'] == 0:
            checkpoint_path = model_path.replace('.pth', f'_checkpoint_ep_{episode+1}.pth')
            ppo_agent.save_checkpoint(checkpoint_path, episode, ppo_agent.optimizer.state_dict())
            logger.info(f"检查点保存: {episode+1}")

        if episode % 5 == 0 and episode > 0:
            avg_score = np.mean(scores) if scores else 0
            avg_total_reward = np.mean(total_rewards) if total_rewards else 0
            avg_final_area = np.mean(final_areas) if final_areas else 0
            logger.info(f"Episode {episode}, Avg Score: {avg_score:.3f}, Avg Reward: {avg_total_reward:.3f}")
        
        # 检查早停
        if convergence_info.get('should_stop_early', False):
            break
    
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
    
    # 确保模型目录存在
    model_dir = os.path.dirname(model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    scores = deque(maxlen=50)
    total_rewards = deque(maxlen=50)
    final_areas = deque(maxlen=50)
    
    total_training_episodes = CONFIG['EPISODES']
    
    # 训练统计变量
    training_stats = {
        'episode_count': 0,
        'successful_episodes': 0,
        'training_update_count': 0
    }
    
    for episode in range(total_training_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        final_area = 0
        success_flag = False
        
        # 为每个episode创建独立的记忆
        episode_memory = GRUMemory(sequence_length=CONFIG['SEQUENCE_LENGTH'])
        ppo_agent.state_history.clear()
        
        while not done:
            move_action, turn_action, move_forward_step, turn_angle, debug_info = ppo_agent.act(state, episode_memory, return_debug_info=True)
            
            next_state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step, turn_angle)
            
            # 更新episode_memory中的奖励和终端状态
            if len(episode_memory.rewards) > 0:
                episode_memory.rewards[-1] = reward
                episode_memory.is_terminals[-1] = done
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if detection_results:
                final_area = max(d['width'] * d['height'] for d in detection_results if 'width' in d and 'height' in d)
            
            if done:
                # 检查是否成功找到目标
                climb_detected = any(
                    detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
                    for detection in detection_results
                )
                success_flag = climb_detected
                
                if success_flag:
                    training_stats['successful_episodes'] += 1
                
                log_and_print(logger, f"Ep: {episode+1}/{total_training_episodes}, S: {step_count}, R: {total_reward:.3f}, A: {final_area:.3f}, "
                    f"Success: {success_flag}")
                
                scores.append(step_count)
                total_rewards.append(total_reward)
                final_areas.append(final_area)
                training_stats['episode_count'] += 1
                
                # 使用收敛监控
                convergence_info = ppo_agent.check_convergence_status(total_reward, step_count, success_flag)
                
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
                
                # 重置环境并获取新的初始状态
                env.reset_to_origin()
                state = env.reset()
                break

        # 实现第一轮后，之后每5轮训练一次
        should_update = (episode == 0) or (episode % 5 == 4)  # 第一轮或者每5轮训练一次
        
        if len(episode_memory.rewards) > 0 and should_update:
            log_and_print(logger, f"更新策略 - Episode {episode+1} (训练第 {training_stats['training_update_count']+1} 次)...")
            ppo_agent.update(episode_memory)
            training_stats['training_update_count'] += 1
            episode_memory.clear_memory()
        elif len(episode_memory.rewards) > 0:
            # 如果不需要训练，但内存有数据，也要清空
            episode_memory.clear_memory()
        
        if (episode + 1) % CONFIG['CHECKPOINT_INTERVAL'] == 0:
            checkpoint_path = model_path.replace('.pth', f'_checkpoint_ep_{episode+1}.pth')
            ppo_agent.save_checkpoint(checkpoint_path, episode, ppo_agent.optimizer.state_dict())
            logger.info(f"检查点保存: {episode+1}")

        if episode % 5 == 0 and episode > 0:
            avg_score = np.mean(scores) if scores else 0
            avg_total_reward = np.mean(total_rewards) if total_rewards else 0
            logger.info(f"Episode {episode}, Avg Score: {avg_score:.3f}, Avg Reward: {avg_total_reward:.3f}")
        
        # 检查早停
        if convergence_info.get('should_stop_early', False):
            break
    
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
    
    # 加载训练好的模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        ppo_agent.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo_agent.policy_old.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"模型加载成功: {model_path}")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return {"status": "error", "message": f"模型加载失败: {e}"}
    
    # 设置为评估模式
    ppo_agent.policy.eval()
    ppo_agent.policy_old.eval()
    
    evaluation_episodes = CONFIG.get('EVALUATION_EPISODES', 10)
    scores = []
    total_rewards = []
    success_count = 0
    
    for episode in range(evaluation_episodes):
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        done = False
        ppo_agent.state_history.clear()
        
        # 评估时不使用探索噪声
        original_exploration_rate = CONFIG.get('INITIAL_EXPLORATION_RATE', 1.0)
        CONFIG['INITIAL_EXPLORATION_RATE'] = 0.0  # 不使用探索噪声
        
        while not done:
            # 使用确定性动作选择
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
        
            next_state, reward, done, detection_results = env.step(move_action, turn_action, move_forward_step.item(), turn_angle.item())
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                # 检查是否成功找到目标
                climb_detected = any(
                    detection['label'].lower() == 'climb' or 'climb' in detection['label'].lower()
                    for detection in detection_results
                )
                
                if climb_detected:
                    success_count += 1
                
                scores.append(step_count)
                total_rewards.append(total_reward)
                
                log_and_print(logger, f"Eval Ep: {episode+1}/{evaluation_episodes}, S: {step_count}, R: {total_reward:.3f}, Success: {climb_detected}")
                
                # 重置环境
                env.reset_to_origin()
                state = env.reset()
                break
        
        # 恢复原始探索率
        CONFIG['INITIAL_EXPLORATION_RATE'] = original_exploration_rate
    
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
    
    if tool_name == "continue_train_gru_ppo_agent":
        func = continue_training_gru_ppo_agent
    elif tool_name == "train_new_gru_ppo_agent":
        func = train_new_gru_ppo_agent
    elif tool_name == "evaluate_trained_gru_ppo_agent":
        func = evaluate_trained_gru_ppo_agent
    else:
        logger.error(f"错误: 未知的PPO工具 '{tool_name}'")
        log_and_print(logger, f"错误: 未知的PPO工具 '{tool_name}'")
        return {"status": "error", "message": f"未知的PPO工具 '{tool_name}'"}

    try:
        result = func(*args)
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