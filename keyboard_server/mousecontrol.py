#!/usr/bin/env python3
"""
鼠标和键盘操作工具 - 适配executor.py的格式
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

def scroll_mouse(units, x, y):
    """
    鼠标滚轮滚动操作，必须指定坐标位置
    :param units: 滚动单位，正数向上滚动，负数向下滚动
    :param x: 滚动操作的x坐标
    :param y: 滚动操作的y坐标
    """
    try:
        units = int(float(units))
        x = int(float(x))
        y = int(float(y))
        
        # 移动到指定坐标再滚动
        pyautogui.moveTo(x=x, y=y)
        pyautogui.scroll(units)
        return f"滚动成功: {units}单位，位置: ({x}, {y})"
    except Exception as e:
        return f"滚动失败: {str(e)}"

def press_key(*keys):
    """
    按下键盘按键
    :param keys: 要按下的键，可以是单个键或组合键
    """
    try:
        if len(keys) == 1:
            # 单个按键
            pyautogui.press(keys[0])
            return f"按下按键成功: {keys[0]}"
        else:
            # 组合键
            pyautogui.hotkey(*keys)
            return f"按下组合键成功: {'+'.join(keys)}"
    except Exception as e:
        return f"按键操作失败: {str(e)}"

def key_down(*keys):
    """
    按住键盘按键（不释放）
    :param keys: 要按住的键
    """
    try:
        pyautogui.keyDown(*keys)
        return f"按住按键成功: {'+'.join(keys)}"
    except Exception as e:
        return f"按住按键失败: {str(e)}"

def key_up(*keys):
    """
    释放键盘按键
    :param keys: 要释放的键
    """
    try:
        pyautogui.keyUp(*keys)
        return f"释放按键成功: {'+'.join(keys)}"
    except Exception as e:
        return f"释放按键失败: {str(e)}"

def type_text(text, interval=0.0):
    """
    输入文本
    :param text: 要输入的文本
    :param interval: 每个字符之间的输入间隔时间
    """
    try:
        pyautogui.typewrite(text, interval=float(interval))
        return f"输入文本成功: {text}"
    except Exception as e:
        return f"输入文本失败: {str(e)}"

def main():
    if len(sys.argv) < 2:
        print("用法: python mousecontrol.py <函数名> [参数...]")
        print("支持的函数:")
        print("  click_mouse x y click_type clicks interval")
        print("  get_position")
        print("  move_mouse x y duration")
        print("  scroll_mouse units x y")
        print("  press_key key1 [key2...] (如: press_key ctrl j)")
        print("  key_down key1 [key2...]")
        print("  key_up key1 [key2...]")
        print("  type_text text interval")
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
            
        elif function_name == 'scroll_mouse':
            if len(args) < 3:
                print("错误: scroll_mouse 需要提供 units, x, y 三个参数")
                sys.exit(1)
                
            units = float(args[0]) if args and args[0] not in ['None', 'null', ''] else 1
            x = float(args[1]) if len(args) > 1 and args[1] not in ['None', 'null', ''] else 0
            y = float(args[2]) if len(args) > 2 and args[2] not in ['None', 'null', ''] else 0
            
            result = scroll_mouse(units, x, y)
            print(result)
            
        elif function_name == 'press_key':
            # 处理按键参数
            keys = []
            for arg in args:
                if arg not in ['None', 'null', '']:
                    keys.append(arg)
            
            result = press_key(*keys)
            print(result)
            
        elif function_name == 'key_down':
            # 处理按住按键参数
            keys = []
            for arg in args:
                if arg not in ['None', 'null', '']:
                    keys.append(arg)
            
            result = key_down(*keys)
            print(result)
            
        elif function_name == 'key_up':
            # 处理释放按键参数
            keys = []
            for arg in args:
                if arg not in ['None', 'null', '']:
                    keys.append(arg)
            
            result = key_up(*keys)
            print(result)
            
        elif function_name == 'type_text':
            text = args[0] if args and args[0] not in ['None', 'null', ''] else ""
            interval = float(args[1]) if len(args) > 1 and args[1] not in ['None', 'null', ''] else 0.0
            
            result = type_text(text, interval)
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