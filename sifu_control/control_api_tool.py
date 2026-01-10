
import time
import math
import subprocess
import sys

class MovementController:
    """
    移动控制器类，用于处理前进、后退、左转和右转动作
    """
    
    def __init__(self):
        pass  # 不再跟踪位置和方向
        
    def move_forward(self, distance=1):
        """
        向前移动指定距离 - 按W键
        :param distance: 移动距离，默认为1
        """
        print(f"向前移动 {distance} 单位")
        # 模拟按W键
        for _ in range(int(distance)):
            self._press_key('w')
            time.sleep(0.1)
        #print("移动完成")
        return f"向前移动 {distance} 单位"
        
    def move_backward(self, distance=1):
        """
        向后移动指定距离 - 按S键
        :param distance: 移动距离，默认为1
        """
        print(f"向后移动 {distance} 单位")
        # 模拟按S键
        for _ in range(int(distance)):
            self._press_key('s')
            time.sleep(0.1)
        #print("移动完成")
        return f"向后移动 {distance} 单位"
        
    def turn_left(self, angle=90):
        """
        左转指定角度 - 鼠标向左移动
        :param angle: 转动角度，默认为90度
        """
        print(f"左转 {angle} 度")
        # 通过鼠标移动来实现视角转动
        self._move_mouse_relative(-angle, 0)
        #print("转动完成")
        return f"左转 {angle} 度"
        
    def turn_right(self, angle=90):
        """
        右转指定角度 - 鼠标向右移动
        :param angle: 转动角度，默认为90度
        """
        print(f"右转 {angle} 度")
        # 通过鼠标移动来实现视角转动
        self._move_mouse_relative(angle, 0)
        #print("转动完成")
        return f"右转 {angle} 度"
        
    def strafe_left(self, distance=1):
        """
        左平移 - 按A键
        :param distance: 移动距离，默认为1
        """
        print(f"左平移 {distance} 单位")
        # 模拟按A键
        for _ in range(int(distance)):
            self._press_key('a')
            time.sleep(0.1)
        #print("平移完成")
        return f"左平移 {distance} 单位"
        
    def strafe_right(self, distance=1):
        """
        右平移 - 按D键
        :param distance: 移动距离，默认为1
        """
        print(f"右平移 {distance} 单位")
        # 模拟按D键
        for _ in range(int(distance)):
            self._press_key('d')
            time.sleep(0.1)
        #print("平移完成")
        return f"右平移 {distance} 单位"
        
    def move_sequence(self, commands):
        """
        执行移动序列
        :param commands: 移动命令列表，支持 w(前), s(后), a(左平移), d(右平移) 和数字的组合
        例如: [("w", 2), ("d", 90), ("s", 1), ("a", 90)]
        """
        for command in commands:
            action, value = command[0], command[1]
            if action.lower() == "w":
                self.move_forward(value)
            elif action.lower() == "s":
                self.move_backward(value)
            elif action.lower() == "a":
                self.strafe_left(value)
            elif action.lower() == "d":
                self.strafe_right(value)
            elif action.lower() == "turn_left":
                self.turn_left(value)
            elif action.lower() == "turn_right":
                self.turn_right(value)
            time.sleep(0.5)  # 短暂延迟，模拟移动时间
        return f"移动序列执行完成"
            
    def get_position(self):
        """
        获取当前位置
        :return: 当前位置 [x, y]
        """
        return "位置追踪已禁用"
        
    def get_direction(self):
        """
        获取当前方向
        :return: 当前方向 (角度)
        """
        return "方向追踪已禁用"

    def _press_key(self, key):
        """
        内部方法：模拟按键
        :param key: 要按下的键
        """
        try:
            # 调用keyboard_server中的mousecontrol.py来执行按键操作
            result = subprocess.run([
                sys.executable, 
                "e:/code/my_python_server/keyboard_server/mousecontrol.py", 
                "press_key", 
                key
            ], capture_output=True, text=True, cwd="e:/code/my_python_server")
            
            if result.returncode != 0:
                print(f"按键 {key} 失败: {result.stderr}")
                return f"按键 {key} 失败: {result.stderr}"
            else:
                print(f"按键 {key} 成功: {result.stdout.strip()}")
                return f"按键 {key} 成功: {result.stdout.strip()}"
        except Exception as e:
            print(f"执行按键操作时出错: {str(e)}")
            return f"执行按键操作时出错: {str(e)}"

    def _move_mouse_relative(self, x_offset, y_offset):
        """
        内部方法：相对移动鼠标
        :param x_offset: X轴偏移量
        :param y_offset: Y轴偏移量
        """
        try:
            # 首先获取当前鼠标位置
            result = subprocess.run([
                sys.executable, 
                "e:/code/my_python_server/keyboard_server/mousecontrol.py", 
                "get_position"
            ], capture_output=True, text=True, cwd="e:/code/my_python_server")
            
            if result.returncode != 0:
                print(f"获取鼠标位置失败: {result.stderr}")
                return f"获取鼠标位置失败: {result.stderr}"
            else:
                # 解析当前位置
                output = result.stdout.strip()
                # 格式为 "当前位置: (x, y)"，提取坐标
                pos_str = output.split('(')[1].split(')')[0]
                x, y = map(int, pos_str.split(', '))
                
                # 计算新位置
                new_x = x + x_offset
                new_y = y + y_offset
                
                # 移动鼠标到新位置
                result = subprocess.run([
                    sys.executable, 
                    "e:/code/my_python_server/keyboard_server/mousecontrol.py", 
                    "move_mouse", 
                    str(new_x), 
                    str(new_y)
                ], capture_output=True, text=True, cwd="e:/code/my_python_server")
                
                if result.returncode != 0:
                    print(f"移动鼠标失败: {result.stderr}")
                    return f"移动鼠标失败: {result.stderr}"
                else:
                    print(f"鼠标移动成功: ({x}, {y}) -> ({new_x}, {new_y})")
                    return f"鼠标移动成功: ({x}, {y}) -> ({new_x}, {new_y})"
        except Exception as e:
            print(f"执行鼠标移动操作时出错: {str(e)}")
            return f"执行鼠标移动操作时出错: {str(e)}"


def send_key_event(key, duration=0.1):
    """
    使用更底层的API发送按键事件，适用于游戏
    """
    try:
        # 尝试使用pynput库，它比pyautogui更兼容游戏
        from pynput.keyboard import Key, Controller
        import time

        keyboard = Controller()
        keyboard.press(key)
        time.sleep(duration)
        keyboard.release(key)
        #print(f"使用pynput发送按键: {key} 成功")
        return f"使用pynput发送按键: {key} 成功"
    except ImportError:
        print("pynput库未安装，尝试安装: pip install pynput")
        try:
            # 如果pynput不可用，尝试使用ctypes直接调用Windows API
            import ctypes
            from ctypes import wintypes
            import time

            # 定义必要的Windows API常量和结构
            user32 = ctypes.WinDLL('user32', use_last_error=True)

            # 定义输入类型
            INPUT_KEYBOARD = 1
            KEYEVENTF_EXTENDEDKEY = 0x0001
            KEYEVENTF_KEYUP = 0x0002

            # 虚拟键码映射
            vk_codes = {
                'w': 0x57,
                's': 0x53,
                'a': 0x41,
                'd': 0x44,
                'left': 0x25,  # 左箭头
                'right': 0x27,  # 右箭头
            }

            # 获取虚拟键码
            vk_code = vk_codes.get(key.lower(), ord(key.upper()))

            # 定义INPUT结构
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

            # 创建键盘输入事件
            inputs = [INPUT(type=INPUT_KEYBOARD,
                           ki=KEYBDINPUT(wVk=vk_code, wScan=0, dwFlags=0, time=0, dwExtraInfo=None)),
                     INPUT(type=INPUT_KEYBOARD,
                           ki=KEYBDINPUT(wVk=vk_code, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=None))]

            # 发送输入事件
            ret = user32.SendInput(2, inputs, ctypes.sizeof(INPUT))
            if ret != 2:
                print(f"发送按键事件失败: {ctypes.FormatError(ctypes.get_last_error())}")
                return f"发送按键事件失败: {ctypes.FormatError(ctypes.get_last_error())}"

            print(f"使用Windows API发送按键: {key} 成功")
            return f"使用Windows API发送按键: {key} 成功"
        except Exception as e:
            print(f"使用Windows API发送按键事件时出错: {str(e)}")
            return f"使用Windows API发送按键事件时出错: {str(e)}"


def send_mouse_move_relative(x_offset, y_offset, use_raw_input=False):
    """
    使用更底层的API相对移动鼠标，适用于游戏视角控制
    """
    # 首先尝试使用pydirectinput，这是最适合游戏的
    try:
        import pydirectinput
        pydirectinput.moveRel(int(x_offset), int(y_offset), relative=True)
        #print(f"使用pydirectinput相对移动鼠标: ({x_offset}, {y_offset}) 成功")
        return f"使用pydirectinput相对移动鼠标: ({x_offset}, {y_offset}) 成功"
    except ImportError:
        print("pydirectinput库未安装，尝试其他方法。请运行: pip install pydirectinput")
        pass  # 继续尝试其他方法

    try:
        # 尝试使用pynput库
        from pynput.mouse import Controller
        import time

        mouse = Controller()
        mouse.move(int(x_offset), int(y_offset))
        #print(f"使用pynput相对移动鼠标: ({x_offset}, {y_offset}) 成功")
        return f"使用pynput相对移动鼠标: ({x_offset}, {y_offset}) 成功"
    except ImportError:
        print("pynput库未安装，尝试安装: pip install pynput")
        pass  # 继续尝试其他方法

    try:
        # 如果pynput不可用，使用ctypes直接调用Windows API
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL('user32', use_last_error=True)

        # 使用MOUSEEVENTF_MOVE标志进行相对移动
        MOUSEEVENTF_MOVE = 0x0001

        success = user32.mouse_event(MOUSEEVENTF_MOVE, int(x_offset), int(y_offset), 0, 0)
        if success:
            print(f"使用Windows API相对移动鼠标: ({x_offset}, {y_offset}) 成功")
            return f"使用Windows API相对移动鼠标: ({x_offset}, {y_offset}) 成功"
        else:
            print(f"使用Windows API移动鼠标失败: {ctypes.FormatError(ctypes.get_last_error())}")
            return f"使用Windows API移动鼠标失败: {ctypes.FormatError(ctypes.get_last_error())}"
    except Exception as e:
        print(f"使用Windows API移动鼠标时出错: {str(e)}")
        return f"使用Windows API移动鼠标时出错: {str(e)}"


class ImprovedMovementController:
    """
    改进的移动控制器类，使用更兼容游戏的输入方式
    """
    
    def __init__(self):
        pass  # 不再跟踪位置和方向
        
    def move_forward(self, distance=1):
        """
        向前移动指定距离 - 按W键
        :param distance: 移动距离，默认为1
        """
        print(f"向前移动 {distance} 单位")
        # 使用改进的按键方法
        for _ in range(int(distance)):
            success = send_key_event('w', 0.1)
            if not success.startswith("使用"):
                print("按键发送失败，使用原始方法")
                result = self._press_key('w')
            time.sleep(0.1)
        #print("移动完成")
        return f"向前移动 {distance} 单位"
        
    def move_backward(self, distance=1):
        """
        向后移动指定距离 - 按S键
        :param distance: 移动距离，默认为1
        """
        print(f"向后移动 {distance} 单位")
        # 使用改进的按键方法
        for _ in range(int(distance)):
            success = send_key_event('s', 0.1)
            if not success.startswith("使用"):
                print("按键发送失败，使用原始方法")
                result = self._press_key('s')
            time.sleep(0.1)
        #print("移动完成")
        return f"向后移动 {distance} 单位"
        
    def turn_left(self, angle=90):
        """
        左转指定角度 - 鼠标向左移动
        :param angle: 转动角度，默认为90度
        """
        print(f"左转 {angle} 度")
        # 通过相对鼠标移动来实现视角转动
        success = send_mouse_move_relative(-int(angle), 0)
        if not success.startswith("使用"):
            print("鼠标移动失败，使用原始方法")
            result = self._move_mouse_relative(-angle, 0)
        #print("转动完成")
        return f"左转 {angle} 度"
        
    def turn_right(self, angle=90):
        """
        右转指定角度 - 鼠标向右移动
        :param angle: 转动角度，默认为90度
        """
        print(f"右转 {angle} 度")
        # 通过相对鼠标移动来实现视角转动
        success = send_mouse_move_relative(int(angle), 0)
        if not success.startswith("使用"):
            print("鼠标移动失败，使用原始方法")
            result = self._move_mouse_relative(angle, 0)
        #print("转动完成")
        return f"右转 {angle} 度"
        
    def strafe_left(self, distance=1):
        """
        左平移 - 按A键
        :param distance: 移动距离，默认为1
        """
        print(f"左平移 {distance} 单位")
        # 使用改进的按键方法
        for _ in range(int(distance)):
            success = send_key_event('a', 0.1)
            if not success.startswith("使用"):
                print("按键发送失败，使用原始方法")
                result = self._press_key('a')
            time.sleep(0.1)
        #print("平移完成")
        return f"左平移 {distance} 单位"
        
    def strafe_right(self, distance=1):
        """
        右平移 - 按D键
        :param distance: 移动距离，默认为1
        """
        print(f"右平移 {distance} 单位")
        # 使用改进的按键方法
        for _ in range(int(distance)):
            success = send_key_event('d', 0.1)
            if not success.startswith("使用"):
                print("按键发送失败，使用原始方法")
                result = self._press_key('d')
            time.sleep(0.1)
        #print("平移完成")
        return f"右平移 {distance} 单位"
        
    def move_sequence(self, commands):
        """
        执行移动序列
        :param commands: 移动命令列表，支持 w(前), s(后), a(左平移), d(右平移) 和数字的组合
        例如: [("w", 2), ("d", 90), ("s", 1), ("a", 90)]
        """
        for command in commands:
            action, value = command[0], command[1]
            if action.lower() == "w":
                self.move_forward(value)
            elif action.lower() == "s":
                self.move_backward(value)
            elif action.lower() == "a":
                self.strafe_left(value)
            elif action.lower() == "d":
                self.strafe_right(value)
            elif action.lower() == "turn_left":
                self.turn_left(value)
            elif action.lower() == "turn_right":
                self.turn_right(value)
            time.sleep(0.5)  # 短暂延迟，模拟移动时间
        return f"移动序列执行完成"
            
    def get_position(self):
        """
        获取当前位置
        :return: 当前位置 [x, y]
        """
        return "位置追踪已禁用"
        
    def get_direction(self):
        """
        获取当前方向
        :return: 当前方向 (角度)
        """
        return "方向追踪已禁用"

    def _press_key(self, key):
        """
        内部方法：模拟按键（原始方法，作为备选）
        :param key: 要按下的键
        """
        try:
            # 调用keyboard_server中的mousecontrol.py来执行按键操作
            result = subprocess.run([
                sys.executable, 
                "e:/code/my_python_server/keyboard_server/mousecontrol.py", 
                "press_key", 
                key
            ], capture_output=True, text=True, cwd="e:/code/my_python_server")
            
            if result.returncode != 0:
                print(f"按键 {key} 失败: {result.stderr}")
                return f"按键 {key} 失败: {result.stderr}"
            else:
                print(f"按键 {key} 成功: {result.stdout.strip()}")
                return f"按键 {key} 成功: {result.stdout.strip()}"
        except Exception as e:
            print(f"执行按键操作时出错: {str(e)}")
            return f"执行按键操作时出错: {str(e)}"

    def _move_mouse_relative(self, x_offset, y_offset):
        """
        内部方法：相对移动鼠标（原始方法，作为备选）
        :param x_offset: X轴偏移量
        :param y_offset: Y轴偏移量
        """
        try:
            # 首先获取当前鼠标位置
            result = subprocess.run([
                sys.executable, 
                "e:/code/my_python_server/keyboard_server/mousecontrol.py", 
                "get_position"
            ], capture_output=True, text=True, cwd="e:/code/my_python_server")
            
            if result.returncode != 0:
                print(f"获取鼠标位置失败: {result.stderr}")
                return f"获取鼠标位置失败: {result.stderr}"
            else:
                # 解析当前位置
                output = result.stdout.strip()
                # 格式为 "当前位置: (x, y)"，提取坐标
                pos_str = output.split('(')[1].split(')')[0]
                x, y = map(int, pos_str.split(', '))
                
                # 计算新位置
                new_x = x + x_offset
                new_y = y + y_offset
                
                # 移动鼠标到新位置
                result = subprocess.run([
                    sys.executable, 
                    "e:/code/my_python_server/keyboard_server/mousecontrol.py", 
                    "move_mouse", 
                    str(new_x), 
                    str(new_y)
                ], capture_output=True, text=True, cwd="e:/code/my_python_server")
                
                if result.returncode != 0:
                    print(f"移动鼠标失败: {result.stderr}")
                    return f"移动鼠标失败: {result.stderr}"
                else:
                    print(f"鼠标移动成功: ({x}, {y}) -> ({new_x}, {new_y})")
                    return f"鼠标移动成功: ({x}, {y}) -> ({new_x}, {new_y})"
        except Exception as e:
            print(f"执行鼠标移动操作时出错: {str(e)}")
            return f"执行鼠标移动操作时出错: {str(e)}"


