# __init__.py
"""
Blender API Server Plugin
Blender API服务器插件，允许通过HTTP API控制Blender
"""

import bpy
import sys
import os
from bpy.props import IntProperty, StringProperty, BoolProperty
from bpy.types import AddonPreferences, Operator, Panel

# 将当前目录添加到Python路径
blender_api_dir = os.path.dirname(__file__)
if blender_api_dir not in sys.path:
    sys.path.insert(0, blender_api_dir)

# 导入API模块
from .api import init_blender_server, stop_blender_server_timer, is_server_running_timer

bl_info = {
    "name": "exec Server",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > API Server",
    "description": "提供HTTP API接口来控制Blender",
    "warning": "",
    "doc_url": "",
    "category": "Development",
}

class BLENDER_API_PT_server_panel(Panel):
    """API服务器控制面板"""
    bl_label = "API Server"
    bl_idname = "BLENDER_API_PT_server_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'API Server'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # 服务器状态
        col = layout.column()
        if is_server_running_timer():
            col.label(text="Server Status: Running", icon='CHECKMARK')
        else:
            col.label(text="Server Status: Stopped", icon='X')
        
        # 服务器控制按钮
        col.separator()
        row = col.row()
        if is_server_running_timer():
            row.operator("blender_api.stop_server", text="Stop Server", icon='X')
        else:
            row.operator("blender_api.start_server", text="Start Server", icon='PLAY')
        
        # 服务器信息
        col.separator()
        col.label(text="Server Info:")
        col.label(text="Host: localhost")
        col.label(text="Port: 8080")
        col.label(text="Endpoints:")
        col.label(text="  - POST /api/eval")
        col.label(text="  - POST /api/exec") 
        col.label(text="  - POST /api/clear_scene")

class BLENDER_API_OT_start_server(Operator):
    """启动API服务器"""
    bl_idname = "blender_api.start_server"
    bl_label = "Start API Server"
    bl_description = "启动HTTP API服务器"
    
    def execute(self, context):
        success = init_blender_server()
        if success:
            self.report({'INFO'}, "API Server started successfully")
        else:
            self.report({'ERROR'}, "Failed to start API Server")
        return {'FINISHED'}

class BLENDER_API_OT_stop_server(Operator):
    """停止API服务器"""
    bl_idname = "blender_api.stop_server"
    bl_label = "Stop API Server"
    bl_description = "停止HTTP API服务器"
    
    def execute(self, context):
        from .api import stop_blender_server_timer
        stop_blender_server_timer()
        self.report({'INFO'}, "API Server stopped")
        return {'FINISHED'}

classes = (
    BLENDER_API_PT_server_panel,
    BLENDER_API_OT_start_server,
    BLENDER_API_OT_stop_server,
)

def register():
    """注册插件"""
    from .api import register as api_register
    api_register()  # 调用API模块的注册函数
    
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """注销插件"""
    from .api import unregister as api_unregister
    api_unregister()  # 调用API模块的注销函数
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()