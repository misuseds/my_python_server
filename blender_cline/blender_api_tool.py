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


def import_pmx_file():
    """导入PMX文件"""
    code = '''
import bpy

def open_pmx_file_selector():  
    """打开PMX文件选择器"""  
    # 直接调用CATS的通用导入器，它会自动过滤PMX文件  
    bpy.ops.cats_importer.import_any_model('INVOKE_DEFAULT')  

# 在Blender中运行  
open_pmx_file_selector()
    '''
    return call_blender_api('/api/exec', code)


def delete_all_objects():
    """删除所有对象"""
    code = '''
import bpy
# 选择所有对象
bpy.ops.object.select_all(action='SELECT')
# 删除选中的对象
bpy.ops.object.delete()
"所有物体已删除"
'''
    return call_blender_api('/api/exec', code)


def set_blender_scale_settings():
    """
    设置Blender场景的单位比例和当前对象的缩放
    """
    code = '''
import bpy

# 设置场景单位比例
bpy.context.scene.unit_settings.scale_length = 0.01

# 设置当前选中对象的缩放（如果存在）
if bpy.context.object:
    bpy.context.object.scale[0] = 100
    bpy.context.object.scale[1] = 100
    bpy.context.object.scale[2] = 100

# 设置3D视口的远裁剪平面
if bpy.context.space_data:
    bpy.context.space_data.clip_end = 300000

print("Blender缩放设置已应用")
'''
    return call_blender_api('/api/exec', code)



def import_psk_file():
    """导入PSK文件（外部选择文件路径）"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        try:
            import Tkinter as tk
            from tkFileDialog import askopenfilename
        except ImportError:
            print("错误: 无法导入tkinter模块")
            return None

    # 创建一个隐藏的tkinter根窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 打开文件选择对话框
    filepath = filedialog.askopenfilename(
        title="选择PSK文件",
        filetypes=[("PSK files", "*.psk"), ("All files", "*.*")]
    )
    
    # 销毁tkinter根窗口
    root.destroy()
    print(f"选择的文件路径: {filepath}")
    if filepath and filepath.endswith('.psk'):
        # 构建执行PSK导入的代码，使用Blender操作符方式
        code = f'''
import bpy  
from io_scene_psk_psa.psk.reader import read_psk  
from io_scene_psk_psa.psk.importer import import_psk, PskImportOptions  
# 配置导入选项（需要添加 name 属性）  
options = PskImportOptions()  
options.name = 'ObjectName'  # 添加这个属性  
options.should_import_mesh = True  
options.should_import_skeleton = True  # 使用 skeleton 而不是 armature  
options.scale = 1.0  
  
# 读取PSK文件  
psk = read_psk('{filepath}')  
  
# 调用你的版本（3个参数）  
result = import_psk(psk, bpy.context, options)
'''
        return call_blender_api('/api/exec', code)
    else:
        print("未选择有效的PSK文件")
        return {"status": "error", "message": "未选择有效的PSK文件"}


def execute_tool(tool_name):
    """
    根据工具名称执行对应的Blender操作
    
    Args:
        tool_name (str): 工具名称
        
    Returns:
        dict: API响应结果
    """
    tool_functions = {
        "import_pmx_file": import_pmx_file,
        "import_psk_file": import_psk_file,
        
        "delete_all_objects": delete_all_objects,
        "activate_blender": activate_blender_window,
        "set_blender_scale": set_blender_scale_settings,
        
    }
    
    if tool_name in tool_functions:
        if tool_name == "activate_blender":
            # 特殊处理：激活Blender窗口不需要API调用
            result = tool_functions[tool_name]()
            return {"status": "success", "result": result} if result else {"status": "error", "message": "Failed to activate Blender"}
        else:
            # 其他工具调用API
            return tool_functions[tool_name]()
    else:
        print(f"错误: 未知的Blender工具 '{tool_name}'")
        return None

def print_response_result(tool_name, response):
    """
    打印API响应结果
    
    Args:
        tool_name (str): 工具名称
        response (dict): API响应
    """
    if response:
        if response['status'] == 'success':
            print(f"{tool_name} 执行成功!")
            print(f"返回结果: {response['result']}")
        else:
            print(f"{tool_name} 执行失败: {response['message']}")
    else:
        print("无法连接到Blender服务器")


def main():
    tool_name = sys.argv[1] if len(sys.argv) > 1 else "import_psk_file"
    
    # 执行对应的工具
    response = execute_tool(tool_name)
    
    # 如果不是激活窗口操作，则打印响应结果
    if tool_name != "activate_blender":
        print_response_result(tool_name, response)


if __name__ == "__main__":
    main()