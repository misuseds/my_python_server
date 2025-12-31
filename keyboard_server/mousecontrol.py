#!/usr/bin/env python3
"""
鼠标操作工具 - 适配executor.py的格式
参数: 函数名 参数列表
"""
import sys
import pyautogui
from PIL import Image
import base64
import io

# 禁用pyautogui的安全限制（在生产环境中请谨慎使用）
pyautogui.FAILSAFE = False

def click_mouse(x=None, y=None, click_type='left', clicks=1, interval=0.0):
    """
    鼠标点击操作
    """
    try:
        if x is not None and y is not None:
            pyautogui.click(x=int(float(x)), y=int(float(y)), clicks=int(float(clicks)), interval=float(interval), button=click_type)
            return f"点击成功: ({x}, {y}), 类型: {click_type}, 次数: {clicks}"
        else:
            pyautogui.click(clicks=int(float(clicks)), interval=float(interval), button=click_type)
            return f"点击当前位置成功, 类型: {click_type}, 次数: {clicks}"
    except Exception as e:
        return f"点击失败: {str(e)}"

def get_position():
    """
    获取当前鼠标位置
    """
    try:
        x, y = pyautogui.position()
        return f"当前位置: ({x}, {y})"
    except Exception as e:
        return f"获取位置失败: {str(e)}"

def move_mouse(x, y, duration=0.0):
    """
    移动鼠标到指定位置
    """
    try:
        pyautogui.moveTo(x=int(float(x)), y=int(float(y)), duration=float(duration))
        return f"移动鼠标成功: ({x}, {y})"
    except Exception as e:
        return f"移动鼠标失败: {str(e)}"

def main():
    if len(sys.argv) < 2:
        print("用法: python mousecontrol.py <函数名> [参数...]")
        sys.exit(0)  # 改为0，避免在帮助信息时退出码为1
    
    function_name = sys.argv[1]
    args = sys.argv[2:]
    
    try:
        if function_name == 'click_mouse':
            # 处理可能的None值，同时确保类型转换正确
            x = float(args[0]) if args and args[0] not in ['None', 'null', ''] else None
            y = float(args[1]) if len(args) > 1 and args[1] not in ['None', 'null', ''] else None
            click_type = args[2] if len(args) > 2 and args[2] not in ['None', 'null', ''] else 'left'
            clicks = float(args[3]) if len(args) > 3 and args[3] not in ['None', 'null', ''] else 1
            interval = float(args[4]) if len(args) > 4 and args[4] not in ['None', 'null', ''] else 0.0
            
            result = click_mouse(x, y, click_type, clicks, interval)
            print(result)
            
        elif function_name == 'get_position':
            result = get_position()
            print(result)
            
        elif function_name == 'move_mouse':
            x = float(args[0]) if args and args[0] not in ['None', 'null', ''] else None
            y = float(args[1]) if len(args) > 1 and args[1] not in ['None', 'null', ''] else None
            duration = float(args[2]) if len(args) > 2 and args[2] not in ['None', 'null', ''] else 0.0
            
            result = move_mouse(x, y, duration)
            print(result)
            
        else:
            print(f"未知函数: {function_name}")
            sys.exit(1)
            
    except IndexError:
        # 参数数量不够时的处理
        print(f"参数数量不足: 需要为 {function_name} 提供足够的参数")
        sys.exit(1)
    except Exception as e:
        print(f"执行错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()