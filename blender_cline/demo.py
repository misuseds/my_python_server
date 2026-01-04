import bpy
import bmesh
from mathutils import Vector

def get_object_height_z(obj):
    """
    获取单个物体在Z轴方向的尺寸
    对于网格物体，使用边界框计算；对于其他类型物体，使用尺寸属性
    """
    if obj.type == 'MESH':
        # 对于网格物体，使用边界框计算Z方向尺寸
        bbox = obj.bound_box
        if bbox:
            # 将边界框顶点转换到世界坐标系
            world_bbox = [obj.matrix_world @ Vector(co) for co in bbox]
            # 计算Z轴方向的最大最小值差
            z_coords = [v.z for v in world_bbox]
            height_z = max(z_coords) - min(z_coords)
            return height_z
    elif obj.type in ['CURVE', 'SURFACE', 'FONT', 'META']:
        # 对于其他类型的几何体，使用尺寸属性
        return obj.dimensions.z
    else:
        # 对于其他类型（如空对象），返回尺寸的Z分量
        return obj.dimensions.z
    
    return 0.0

# 获取场景中所有物体的Z方向尺寸
heights_z = {}
for obj in bpy.context.scene.objects:
    if obj.type in ['MESH', 'CURVE', 'SURFACE', 'FONT', 'META', 'EMPTY']:
        height_z = get_object_height_z(obj)
        heights_z[obj.name] = round(height_z, 6)  # 保留6位小数

# 返回结果
heights_z