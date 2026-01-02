import requests
import json

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

def get_all_objects():
    """
    获取当前场景中的所有物体并打印名称
    """
    # 您提供的代码
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
    
    # 使用 /api/exec 端点执行代码块
    response = call_blender_api('/api/exec', code)
    
    if response:
        if response['status'] == 'success':
            print("执行成功!")
            print("返回结果:", response['result'])
        else:
            print("执行失败:", response['message'])
    else:
        print("无法连接到Blender服务器")

def delete_all_objects():
    """
    删除当前场景中的所有物体
    """
    code = '''
import bpy


# 选择所有对象
bpy.ops.object.select_all(action='SELECT')

# 删除选中的对象
bpy.ops.object.delete()
"所有物体已删除"
'''
    
    # 使用 /api/exec 端点执行代码块
    response = call_blender_api('/api/exec', code)
    
    if response:
        if response['status'] == 'success':
            print("删除所有物体成功!")
            print("返回结果:", response['result'])
        else:
            print("删除失败:", response['message'])
    else:
        print("无法连接到Blender服务器")

def create_cube():
    """
    在Blender场景中创建一个立方体
    """
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
    
    # 使用 /api/exec 端点执行代码块
    response = call_blender_api('/api/exec', code)
    
    if response:
        if response['status'] == 'success':
            print("立方体创建成功!")
            print("返回结果:", response['result'])
        else:
            print("创建失败:", response['message'])
    else:
        print("无法连接到Blender服务器")


if __name__ == "__main__":
    # 确保Blender服务器正在运行
    print("正在连接到Blender服务器...")
    
    # 方法1: 使用exec端点执行完整代码块（推荐）
    print("\n--- 使用 exec 端点 ---")
    #get_all_objects()
    
    # 示例：删除所有物体
    print("\n--- 删除所有物体 ---")
    #delete_all_objects()

    print("\n--- 创建立方体 ---")
    create_cube()