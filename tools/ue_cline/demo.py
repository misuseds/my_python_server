import socket
import json

def send_json_request(host="localhost", port=8070, code=None):
    """
    发送Python代码执行请求到服务器 - 使用纯JSON格式
    """
    try:
        # 创建socket连接
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        
        # 准备数据
        payload = {
            "code": code or "unreal.SystemLibrary.print_string(None, 'Hello from remote execution!', True, True, unreal.LinearColor(0,1,0,1), 5.0)"
        }
        
        # 发送JSON数据
        json_data = json.dumps(payload)
        sock.send(json_data.encode('utf-8'))
        
        # 接收响应
        response = sock.recv(4096).decode('utf-8')
        print(f"服务器响应: {response}")
        
        sock.close()
        return True
        
    except Exception as e:
        print(f"发送请求时出错: {str(e)}")
        return False

def send_fbx_import_request(host="localhost", port=8070):
    """
    发送FBX导入请求到服务器 - 使用固定路径
    """
    code = """
import unreal
import os

# 检查文件是否存在
fbx_path = "E:\\\\blender\\\\SK_W_MainChar_01.fbx"
destination_path = "/Game/Characters/MainChar/W/Meshes/"

if not os.path.exists(fbx_path):
    result = "文件路径无效: " + fbx_path
else:
    # 创建导入任务
    import_task = unreal.AssetImportTask()
    import_task.set_editor_property('filename', fbx_path)
    import_task.set_editor_property('destination_path', destination_path)
    import_task.set_editor_property('save', True)
    import_task.set_editor_property('automated', True)
    import_task.set_editor_property('replace_existing', True)
    
    # 执行导入任务
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    asset_tools.import_asset_tasks([import_task])
    
    # 获取文件名
    filename = os.path.basename(fbx_path)
    result = "成功导入文件: " + filename + " 到 " + destination_path

result
"""
    return send_json_request(host, port, code)

# 使用示例
if __name__ == "__main__":

    # 发送FBX导入请求 - 使用固定路径
    print("\n发送FBX导入请求...")
    success = send_fbx_import_request()

    if success:
        print("FBX导入请求发送成功")
    else:
        print("FBX导入请求发送失败")