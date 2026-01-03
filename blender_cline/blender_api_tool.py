import sys
import os
import requests
import json
import subprocess
import platform


def call_blender_api(endpoint, code):
    """
    调用Blender API执行代码
    """
    url = f"http://localhost:8080{endpoint}"
    
    payload = {
        "code": code
    }
    
    try:
        response = requests.post(url, json=payload)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return None


def activate_blender_window():
    """
    激活Blender窗口
    """
    try:
        system = platform.system()
        
        if system == "Windows":
            # Windows系统使用powershell激活Blender窗口
            script = '''
            $blender_process = Get-Process -Name "blender" -ErrorAction SilentlyContinue
            if ($blender_process) {
                Add-Type -TypeDefinition @"
                    using System;
                    using System.Runtime.InteropServices;
                    public class WindowActivator {
                        [DllImport("user32.dll")]
                        public static extern bool SetForegroundWindow(IntPtr hWnd);
                        [DllImport("user32.dll")]
                        public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
                    }
"@
                foreach ($process in $blender_process) {
                    [WindowActivator]::ShowWindow($process.MainWindowHandle, 5)
                    [WindowActivator]::SetForegroundWindow($process.MainWindowHandle)
                }
            }
            '''
            subprocess.run(["powershell", "-Command", script], check=True)
            
        elif system == "Darwin":  # macOS
            script = '''
            tell application "Blender"
                activate
            end tell
            '''
            subprocess.run(["osascript", "-e", script], check=True)
            
        else:
            print(f"当前系统 {system} 不支持窗口激活")
            return False
            
        print("Blender窗口已激活")
        return True
        
    except subprocess.CalledProcessError:
        print("激活Blender窗口失败，请确保Blender正在运行")
        return False
    except FileNotFoundError:
        print("未找到必要的工具")
        return False
    except Exception as e:
        print(f"激活Blender窗口时出错: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("错误: 缺少工具名称参数")
        return
    
    tool_name = sys.argv[1]
    
    if tool_name == "get_all_objects":
        code = '''
import bpy

# 获取当前场景中的所有物体
all_objects = bpy.context.scene.objects

# 打印每个物体的名称
result = []
for obj in all_objects:
    obj_info = f"obj: {obj.name}"
    result.append(obj_info)
    print(obj_info)

result
'''
        response = call_blender_api('/api/exec', code)
        
    elif tool_name == "delete_all_objects":
        code = '''
import bpy

# 选择所有对象
bpy.ops.object.select_all(action='SELECT')

# 删除选中的对象
bpy.ops.object.delete()
"所有物体已删除"
'''
        response = call_blender_api('/api/exec', code)
        
    elif tool_name == "create_cube":
        code = '''
import bpy

# 添加一个立方体
bpy.ops.mesh.primitive_cube_add(
    location=(0, 0, 0)  # 设置立方体的位置
)

# 获取新创建的立方体对象
cube = bpy.context.active_object
cube.name = "MyCube"  # 重命名立方体

f"立方体已创建，名称: {cube.name}"
'''
        response = call_blender_api('/api/exec', code)
    
    elif tool_name == "activate_blender":
        activate_blender_window()
        return
    
    else:
        print(f"错误: 未知的Blender工具 '{tool_name}'")
        return
    
    if response:
        if response['status'] == 'success':
            print(f"{tool_name} 执行成功!")
            print(f"返回结果: {response['result']}")
        else:
            print(f"{tool_name} 执行失败: {response['message']}")
    else:
        print("无法连接到Blender服务器")


if __name__ == "__main__":
    main()