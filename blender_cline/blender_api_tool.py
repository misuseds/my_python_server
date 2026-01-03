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
    激活Blender窗口（精确匹配窗口标题）
    """
    try:
        import pygetwindow as gw
    except ImportError:
        print("pygetwindow未安装，请运行: pip install pygetwindow")
        return False
    
    try:
        # 获取所有窗口
        all_windows = gw.getAllWindows()
        blender_window = None
        
        for window in all_windows:
            window_title = window.title.strip()
            
            # 多种可能的Blender标题格式
            if (window_title == 'Blender' or  # 基础标题
                window_title.startswith('Blender') and 
                not any(exclude in window_title.lower() for exclude in ['vscode', 'visual studio', 'code'])):
                
                # 额外检查确保不是VSCode或其他编辑器
                if ' - ' not in window_title or 'blender.exe' in window_title.lower():
                    blender_window = window
                    break
        
        if blender_window:
            print(f"激活窗口: {blender_window.title}")
            
            try:
                import win32gui
                import win32con
                
                hwnd = blender_window._hWnd
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                
                print(f"窗口已放到最前端: {blender_window.title}")
                return True
            except ImportError:
                print("pywin32未安装，请运行: pip install pywin32")
                if blender_window.isMinimized:
                    blender_window.restore()
                blender_window.activate()
                print("Blender窗口已激活（使用pygetwindow方法）")
                return True
        else:
            print("未找到Blender窗口")
            # 显示所有窗口标题用于调试
            all_titles = gw.getAllTitles()
            print("所有窗口标题（前10个）:")
            for i, title in enumerate(all_titles[:10]):
                if title.strip():
                    print(f"  {i+1}. {title}")
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