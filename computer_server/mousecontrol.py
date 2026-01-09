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
import time

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

def parse_args(args):
    """
    解析参数，支持位置参数和 key=value 格式
    """
    positional_args = []
    named_args = {}
    
    for arg in args:
        if '=' in arg and not arg.startswith('='):
            key, value = arg.split('=', 1)
            named_args[key] = value
        else:
            positional_args.append(arg)
    
    return positional_args, named_args

def safe_float_convert(value, default=0.0):
    """安全转换为浮点数，默认返回默认值"""
    try:
        if value in ['None', 'null', '']:
            return default
        return float(value)
    except ValueError:
        return default

def safe_int_convert(value, default=0):
    """安全转换为整数，默认返回默认值"""
    try:
        if value in ['None', 'null', '']:
            return default
        return int(float(value))
    except ValueError:
        return default

def main():
    if len(sys.argv) < 2:
        print("用法: python mousecontrol.py <函数名> [参数...]")
        print("支持的函数:")
        print("  click_mouse x y click_type clicks interval (或使用命名参数: x=1492 y=466 click_type=left clicks=1 interval=0.1)")
        print("  get_position")
        print("  move_mouse x y duration (或使用命名参数: x=100 y=200 duration=0.5)")
        print("  scroll_mouse units x y (或使用命名参数: units=3 x=100 y=200)")
        print("  press_key key1 [key2...] (如: press_key ctrl j)")
        print("  key_down key1 [key2...]")
        print("  key_up key1 [key2...]")
        print("  type_text text interval (或使用命名参数: text='hello' interval=0.1)")
        sys.exit(0)
    
    function_name = sys.argv[1]
    args = sys.argv[2:]
    
    # 解析参数
    positional_args, named_args = parse_args(args)
    
    try:
        if function_name == 'click_mouse':
            # 设置默认值
            params = {
                'x': None,
                'y': None,
                'click_type': 'left',
                'clicks': 1,
                'interval': 0.0
            }
            
            # 处理位置参数
            param_names = ['x', 'y', 'click_type', 'clicks', 'interval']
            for i, param_name in enumerate(param_names):
                if i < len(positional_args) and positional_args[i] not in ['None', 'null', '']:
                    if param_name in ['x', 'y']:
                        params[param_name] = safe_float_convert(positional_args[i])
                    elif param_name in ['clicks', 'interval']:
                        if param_name == 'clicks':
                            params[param_name] = safe_int_convert(positional_args[i], 1)
                        else:
                            params[param_name] = safe_float_convert(positional_args[i], 0.0)
                    else:
                        params[param_name] = positional_args[i]
            
            # 处理命名参数（命名参数优先级更高）
            for param_name in params.keys():
                if param_name in named_args and named_args[param_name] not in ['None', 'null', '']:
                    if param_name in ['x', 'y']:
                        params[param_name] = safe_float_convert(named_args[param_name])
                    elif param_name == 'clicks':
                        params[param_name] = safe_int_convert(named_args[param_name], 1)
                    elif param_name == 'interval':
                        params[param_name] = safe_float_convert(named_args[param_name], 0.0)
                    else:
                        params[param_name] = named_args[param_name]
            
            result = click_mouse(params['x'], params['y'], params['click_type'], params['clicks'], params['interval'])
            time.sleep(2)
            print(result)
            
        elif function_name == 'get_position':
            result = get_position()
            print(result)
            
        elif function_name == 'move_mouse':
            # 设置默认值
            params = {
                'x': None,
                'y': None,
                'duration': 0.0
            }
            
            # 处理位置参数
            param_names = ['x', 'y', 'duration']
            for i, param_name in enumerate(param_names):
                if i < len(positional_args) and positional_args[i] not in ['None', 'null', '']:
                    if param_name in ['x', 'y']:
                        params[param_name] = safe_float_convert(positional_args[i])
                    else:
                        params[param_name] = safe_float_convert(positional_args[i], 0.0)
            
            # 处理命名参数
            for param_name in params.keys():
                if param_name in named_args and named_args[param_name] not in ['None', 'null', '']:
                    if param_name in ['x', 'y']:
                        params[param_name] = safe_float_convert(named_args[param_name])
                    else:
                        params[param_name] = safe_float_convert(named_args[param_name], 0.0)
            
            result = move_mouse(params['x'], params['y'], params['duration'])
            print(result)
            
        elif function_name == 'scroll_mouse':
            # 设置默认值
            params = {
                'units': 1,
                'x': 0,
                'y': 0
            }
            
            # 检查必需参数
            required_args_count = 0
            for arg in positional_args:
                if arg not in ['None', 'null', '']:
                    required_args_count += 1
            
            required_named_count = 0
            for key in ['units', 'x', 'y']:
                if key in named_args and named_args[key] not in ['None', 'null', '']:
                    required_named_count += 1
            
            if required_args_count < 3 and required_named_count < 3:
                # 检查是否所有必需参数都有定义
                has_all_named = all(key in named_args and named_args[key] not in ['None', 'null', ''] for key in ['units', 'x', 'y'])
                if not has_all_named and required_args_count < 3:
                    print("错误: scroll_mouse 需要提供 units, x, y 三个参数")
                    sys.exit(1)
            
            # 处理位置参数
            param_names = ['units', 'x', 'y']
            for i, param_name in enumerate(param_names):
                if i < len(positional_args) and positional_args[i] not in ['None', 'null', '']:
                    params[param_name] = safe_float_convert(positional_args[i])
            
            # 处理命名参数
            for param_name in params.keys():
                if param_name in named_args and named_args[param_name] not in ['None', 'null', '']:
                    params[param_name] = safe_float_convert(named_args[param_name])
            
            result = scroll_mouse(params['units'], params['x'], params['y'])
            print(result)
            
        elif function_name == 'press_key':
            # 处理按键参数 - 只支持位置参数
            keys = []
            for arg in positional_args:
                if arg not in ['None', 'null', '']:
                    keys.append(arg)
            
            result = press_key(*keys)
            print(result)
            
        elif function_name == 'key_down':
            # 处理按住按键参数 - 只支持位置参数
            keys = []
            for arg in positional_args:
                if arg not in ['None', 'null', '']:
                    keys.append(arg)
            
            result = key_down(*keys)
            print(result)
            
        elif function_name == 'key_up':
            # 处理释放按键参数 - 只支持位置参数
            keys = []
            for arg in positional_args:
                if arg not in ['None', 'null', '']:
                    keys.append(arg)
            
            result = key_up(*keys)
            print(result)
            
        elif function_name == 'type_text':
            # 设置默认值
            params = {
                'text': "",
                'interval': 0.0
            }
            
            # 处理位置参数
            param_names = ['text', 'interval']
            for i, param_name in enumerate(param_names):
                if i < len(positional_args) and positional_args[i] not in ['None', 'null', '']:
                    if param_name == 'interval':
                        params[param_name] = safe_float_convert(positional_args[i], 0.0)
                    else:
                        params[param_name] = positional_args[i]
            
            # 处理命名参数
            for param_name in params.keys():
                if param_name in named_args and named_args[param_name] not in ['None', 'null', '']:
                    if param_name == 'interval':
                        params[param_name] = safe_float_convert(named_args[param_name], 0.0)
                    else:
                        params[param_name] = named_args[param_name]
            
            result = type_text(params['text'], params['interval'])
            print(result)
            
        else:
            print(f"未知函数: {function_name}")
            sys.exit(1)
            
    except ValueError as e:
        print(f"参数值错误: {str(e)}")
        sys.exit(1)
    except IndexError:
        print(f"参数数量不足: 需要为 {function_name} 提供足够的参数")
        sys.exit(1)
    except Exception as e:
        print(f"执行错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()