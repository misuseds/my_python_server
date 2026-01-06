import requests
import json
import tkinter as tk
from tkinter import filedialog
import os

def select_fbx_and_send_to_ue():
    """
    使用tkinter选择FBX文件，然后发送导入命令到UE服务器
    """
    # 隐藏主窗口
    root = tk.Tk()
    root.withdraw()
    
    # 打开文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择FBX文件",
        filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")]
    )
    
    # 销毁根窗口
    root.destroy()
    
    if not file_path or not os.path.exists(file_path):
        print("未选择有效文件")
        return
    
    # 检查是否是 FBX 文件
    if not file_path.lower().endswith('.fbx'):
        print(f"文件 {file_path} 不是有效的 FBX 文件")
        return
    
    # 服务器地址和端口
    url = "http://localhost:8070/exec"
    
    # 构建导入命令 - 使用正确的AssetImportTask API
    command = f'''import unreal
import os

def import_fbx_safely():
    try:
        # 定义目标路径
        destination_path = "/Game/Characters/MainChar/W/Meshes/"
        
        # 检查源文件是否存在
        source_file_path = r"{file_path}"
        if not os.path.exists(source_file_path):
            unreal.log_error(f"源文件不存在: {{source_file_path}}")
            return False
        
        # 创建导入任务
        asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
        
        # 创建导入任务对象
        import_task = unreal.AssetImportTask()
        import_task.set_editor_property('filename', source_file_path)
        import_task.set_editor_property('destination_path', destination_path)
        import_task.set_editor_property('save', True)
        import_task.set_editor_property('automated', True)
        import_task.set_editor_property('replace_existing', True)
        
        # 执行导入任务
        asset_tools.import_asset_tasks([import_task])
        
        unreal.log(f"成功开始导入文件: {{os.path.basename(source_file_path)}}")
        unreal.log(f"导入到路径: {{destination_path}}")
        return True
            
    except Exception as e:
        unreal.log_error(f"导入过程中发生错误: {{str(e)}}")
        import traceback
        unreal.log_error(f"详细错误信息: {{traceback.format_exc()}}")
        return False

# 执行导入
import_fbx_safely()'''
    
    # 构建请求数据
    payload = {
        "command": command
    }
    
    try:
        # 发送POST请求到UE服务器
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        
        # 检查响应
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                print(f"✓ FBX导入命令已发送到UE: {os.path.basename(file_path)}")
                print(f"  导入到路径: /Game/Characters/MainChar/W/Meshes/")
                print(f"  请检查UE编辑器中的导入进度")
            else:
                print(f"✗ UE执行失败: {result}")
        else:
            print(f"✗ 请求失败，状态码: {response.status_code}")
            print(f"  错误信息: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ 请求异常: {e}")

def send_simple_command_to_ue(command_str):
    """
    发送简单命令到UE服务器
    """
    url = "http://localhost:8070/exec"
    
    payload = {
        "command": command_str
    }
    
    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                print(f"✓ 命令执行成功: {result.get('message', '')}")
            else:
                print(f"✗ 命令执行失败: {result}")
        else:
            print(f"✗ 请求失败: {response.status_code}, {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ 请求异常: {e}")

def hello_ue():
    """
    发送Hello UE命令
    """
    command = '''import unreal
unreal.log("Hello UE5 - From HTTP API")'''
    
    send_simple_command_to_ue(command)

if __name__ == "__main__":
    print("选择操作:")
    print("1. 导入FBX文件")
    print("2. 发送Hello UE命令")
    
    choice = input("请输入选择 (1 或 2): ")
    
    if choice == "1":
        select_fbx_and_send_to_ue()
    elif choice == "2":
        hello_ue()
    else:
        print("无效选择")