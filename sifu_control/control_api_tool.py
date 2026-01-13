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
from pathlib import Path
from ultralytics import YOLO
import pyautogui  # 添加pyautogui库用于按键操作

# 添加所需库
import subprocess
import ctypes
from ctypes import wintypes
from pynput import mouse, keyboard

# 改进的移动控制器类定义
class ImprovedMovementController:
    """
    改进的移动控制器类，使用更兼容游戏的输入方式
    """
    
    def __init__(self):
        pass  # 不再跟踪位置和方向

    def move_forward(self, duration=1.0):
        """
        向前移动指定时间 - 持续按住W键
        :param duration: 按键持续时间，默认为1.0秒
        """
        # 使用改进的按键方法，持续按住W键指定时间
        success = self._send_key_event_hold('w', duration)
        return f"向前移动，持续时间 {duration} 秒"
        
    def move_backward(self, duration=1.0):
        """
        向后移动指定时间 - 持续按住S键
        :param duration: 按键持续时间，默认为1.0秒
        """
        # 使用改进的按键方法，持续按住S键指定时间
        success = self._send_key_event_hold('s', duration)
        return f"向后移动，持续时间 {duration} 秒"
        
    def turn_left(self, angle=90, duration=0.1):
        """
        左转指定角度 - 鼠标向左移动
        :param angle: 转动角度，默认为90度
        :param duration: 转动持续时间，默认为0.1秒
        """
        # 通过相对鼠标移动来实现视角转动
        # 根据角度和持续时间计算移动量
        adjusted_movement = int(angle * 10)  # 调整系数使转动更精确
        self._send_mouse_move_relative(-adjusted_movement, 0, duration)
        return f"左转 {angle} 度"
        
    def turn_right(self, angle=90, duration=0.1):
        """
        右转指定角度 - 鼠标向右移动
        :param angle: 转动角度，默认为90度
        :param duration: 转动持续时间，默认为0.1秒
        """
        # 通过相对鼠标移动来实现视角转动
        # 根据角度和持续时间计算移动量
        adjusted_movement = int(angle * 10)  # 调整系数使转动更精确
        self._send_mouse_move_relative(adjusted_movement, 0, duration)
        return f"右转 {angle} 度"
        
    def strafe_left(self, duration=1.0):
        """
        左平移指定时间 - 持续按住A键
        :param duration: 按键持续时间，默认为1.0秒
        """
        # 使用改进的按键方法，持续按住A键指定时间
        success = self._send_key_event_hold('a', duration)
        return f"左平移，持续时间 {duration} 秒"
        
    def strafe_right(self, duration=1.0):
        """
        右平移指定时间 - 持续按住D键
        :param duration: 按键持续时间，默认为1.0秒
        """
        # 使用改进的按键方法，持续按住D键指定时间
        success = self._send_key_event_hold('d', duration)
        return f"右平移，持续时间 {duration} 秒"

    def _send_key_event_hold(self, key, duration=1.0):
        """
        使用更底层的API发送按键事件并保持按下状态一段时间，适用于游戏
        :param key: 要按下的键
        :param duration: 按键持续时间
        """
        try:
            # 尝试使用pynput库，它比pyautogui更兼容游戏
            from pynput.keyboard import Key, Controller
            
            keyboard = Controller()
            keyboard.press(key)
            time.sleep(duration)
            keyboard.release(key)
            return f"使用pynput发送按键: {key}，持续 {duration} 秒，成功"
        except ImportError:
            # 如果pynput不可用，使用ctypes直接调用Windows API
            try:
                import ctypes
                from ctypes import wintypes
                
                user32 = ctypes.WinDLL('user32', use_last_error=True)
                
                # 定义虚拟键码
                vk_code = ord(key.upper())
                
                # 定义输入结构
                class KEYBDINPUT(ctypes.Structure):
                    _fields_ = (("wVk", wintypes.WORD),
                                ("wScan", wintypes.WORD),
                                ("dwFlags", wintypes.DWORD),
                                ("time", wintypes.DWORD),
                                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG)))
                
                class INPUT(ctypes.Structure):
                    _anonymous_ = (("ki", KEYBDINPUT),)
                    _fields_ = (("type", wintypes.DWORD),
                                ("ki", KEYBDINPUT))
                
                # 定义标志
                KEYEVENTF_KEYUP = 0x0002
                KEYEVENTF_KEYDOWN = 0x0000
                
                # 按下键
                press_input = INPUT(type=1,  # INPUT_KEYBOARD
                                  ki=KEYBDINPUT(wVk=vk_code, wScan=0, dwFlags=KEYEVENTF_KEYDOWN, time=0, dwExtraInfo=None))
                
                # 释放键
                release_input = INPUT(type=1,  # INPUT_KEYBOARD
                                    ki=KEYBDINPUT(wVk=vk_code, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=None))
                
                # 按下键
                user32.SendInput(1, ctypes.byref(press_input), ctypes.sizeof(INPUT))
                
                # 等待指定时间
                time.sleep(duration)
                
                # 释放键
                user32.SendInput(1, ctypes.byref(release_input), ctypes.sizeof(INPUT))
                
                return f"使用Windows API发送按键: {key}，持续 {duration} 秒，成功"
            except Exception as e:
                print(f"使用Windows API发送按键事件时出错: {str(e)}")
                return f"使用Windows API发送按键事件时出错: {str(e)}"
        except Exception as e:
            return f"使用pynput发送按键事件时出错: {str(e)}"

    def _send_mouse_move_relative(self, x_offset, y_offset, duration=0.1):
        """
        使用更底层的API相对移动鼠标，适用于游戏视角控制
        """
        # 首先尝试使用pydirectinput，这是最适合游戏的
        try:
            import pydirectinput
            pydirectinput.moveRel(int(x_offset), int(y_offset), relative=True)
            return f"使用pydirectinput相对移动鼠标: ({x_offset}, {y_offset}) 成功"
        except ImportError:
            pass  # 继续尝试其他方法

        try:
            # 尝试使用pynput库
            from pynput.mouse import Controller
            import time

            mouse = Controller()
            # 实现平滑移动
            steps = max(abs(int(x_offset)), abs(int(y_offset)))
            if steps > 0:
                dx = x_offset / steps
                dy = y_offset / steps
                delay = duration / steps
                for _ in range(steps):
                    mouse.move(int(dx), int(dy))
                    time.sleep(delay)
            else:
                mouse.move(int(x_offset), int(y_offset))
            return f"使用pynput相对移动鼠标: ({x_offset}, {y_offset}) 成功"
        except ImportError:
            pass  # 继续尝试其他方法

        try:
            # 如果pynput不可用，使用ctypes直接调用Windows API
            import ctypes
            from ctypes import wintypes
            import time

            user32 = ctypes.WinDLL('user32', use_last_error=True)

            # 使用MOUSEEVENTF_MOVE标志进行相对移动
            MOUSEEVENTF_MOVE = 0x0001

            # 实现平滑移动
            steps = max(abs(int(x_offset)), abs(int(y_offset)))
            if steps > 0:
                dx = x_offset / steps
                dy = y_offset / steps
                delay = duration / steps
                for _ in range(steps):
                    user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
                    time.sleep(delay)
            else:
                success = user32.mouse_event(MOUSEEVENTF_MOVE, int(x_offset), int(y_offset), 0, 0)
                if not success:
                    return f"使用Windows API移动鼠标失败: {ctypes.FormatError(ctypes.get_last_error())}"
            return f"使用Windows API相对移动鼠标: ({x_offset}, {y_offset}) 成功"
        except Exception as e:
            return f"使用Windows API移动鼠标时出错: {str(e)}"