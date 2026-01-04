import win32com.client
import pythoncom
import math
import numpy as np
from win32com.client import VARIANT
import ezdxf
import os
from uuid import uuid4

def get_selected_faces_edges_and_obb():
    """
    获取SolidWorks中选中面的所有边，并计算这些边的OBB包围盒
    """
    try:
        # 初始化COM库
        pythoncom.CoInitialize()
        
        # 连接到SolidWorks应用程序
        sw_app = win32com.client.Dispatch("SldWorks.Application")
        
        # 获取活动文档
        active_doc = sw_app.ActiveDoc
        if not active_doc:
            print("没有打开的文档")
            return False
            
        # 检查是否为零件文档 (1表示零件文档)
        if active_doc.GetType != 1:  
            print("当前文档不是零件文档")
            return False
        
        # 获取选中的对象
        selected_objects = active_doc.SelectionManager.GetSelectedObjectsComponent4(1, -1)
        
        # 获取选中的面
        selected_faces = []
        sel_mgr = active_doc.SelectionManager
        face_count = sel_mgr.GetSelectedObjectCount2(-1)
        
        for i in range(1, face_count + 1):
            face = sel_mgr.GetSelectedObject6(i, -1)
            if face:
                selected_faces.append(face)
                
        if not selected_faces:
            print("未选中任何面")
            return False
            
        # 提取所有选中面上的边
        all_points = []
        spline_control_points = []  # 专门存储样条线控制点用于绘制
        print(f"selected_faces数：{len(selected_faces)}")
        for face in selected_faces:
            # 获取面上的所有边
            edges = face.GetEdges
            print(f"面上的边数：{len(edges)}")
            if edges:
                for edge in edges:
                    if edge:
                        # 获取边的起点和终点
                        print("-------------------------")
                        curve = edge.GetCurve
                        if curve.IsCircle:
                            print("是圆")
               
                            
                        if curve.IsLine:
                            print("是线")
                            
                            start_pt = edge.GetStartVertex.GetPoint
                            end_pt = edge.GetEndVertex.GetPoint
                            print(f"线起点: {start_pt}")
                            print(f"线终点: {end_pt}")
                            
                            # 转换为毫米单位并添加到点列表
                            all_points.append((
                                start_pt[0] * 1000, 
                                start_pt[1] * 1000, 
                                start_pt[2] * 1000
                            ))
                            all_points.append((
                                end_pt[0] * 1000, 
                                end_pt[1] * 1000, 
                                end_pt[2] * 1000
                            ))
                            
                        if curve.IsBcurve:
                            print("是样条线")
                            swSplineParaData = curve.GetBCurveParams5(False, False, True, True)
                            # 使用VARIANT处理out参数
                            konts_variant = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_VARIANT, None)
                            
                            success = swSplineParaData.GetKnotPoints(konts_variant)
                            if success:
                                konts = konts_variant.value
                            for i in range(len(konts)):
                                kont_pt = konts[i]
                                print(f"样条线控制点: {kont_pt}")
                                # 转换为毫米单位并添加到点列表
                                spline_control_points.append((
                                    kont_pt[0] * 1000, 
                                    kont_pt[1] * 1000, 
                                    kont_pt[2] * 1000
                                ))
                  
                        if curve.IsEllipse:
                            print("是椭圆")
         
                            
        # 去重点坐标
        unique_points = list(set(all_points))
        print(f"唯一三维点数: {len(unique_points)}")
        
        if len(unique_points) < 2:
            print("点数不足，无法计算包围盒")
            return False
            
        # 投影到XY平面进行2D OBB计算（假设主要关注XY平面）
        points_2d = [(p[0], p[1]) for p in unique_points]
        points_2d = list(set(points_2d))  # 再次去重
        print(f"二维投影点数: {len(points_2d)}")
        
        # 绘制控制点到DXF文件
        if spline_control_points:
            draw_control_points_to_dxf(active_doc, spline_control_points)
        
        # 计算OBB包围盒
        obb_result = calculate_obb(points_2d)
        
        if obb_result:
            print("OBB包围盒计算结果:")
            print(f"  角度: {obb_result['angle']:.2f}度")
            print(f"  宽度: {obb_result['width']:.3f} mm")
            print(f"  高度: {obb_result['height']:.3f} mm")
            print(f"  面积: {obb_result['area']:.3f} mm²")
            print(f"  中心点: ({obb_result['center'][0]:.3f}, {obb_result['center'][1]:.3f}) mm")
            
            # 输出角点坐标
            print("  角点坐标:")
            for i, corner in enumerate(obb_result['corners']):
                print(f"    角点{i+1}: ({corner[0]:.3f}, {corner[1]:.3f}) mm")
                
        return obb_result
        
    except Exception as e:
        print(f"发生错误: {e}")
        return False
 
    finally:
        # 清理COM资源
        pythoncom.CoUninitialize()

def draw_control_points_to_dxf(sw_document, control_points):
    """
    将控制点绘制到DXF文件并保存在SolidWorks文档同目录下
    """
    try:
        # 创建新的DXF文档
        doc = ezdxf.new(dxfversion='R2010')
        msp = doc.modelspace()
        
        # 绘制点
        for point in control_points:
            msp.add_point((point[0], point[1]))
        
        # 获取SolidWorks文档路径
        sw_file_path = sw_document.GetPathName
        if sw_file_path:
            # 获取目录路径
            directory = os.path.dirname(sw_file_path)
            # 生成文件名
            filename = f"控制点{uuid4().hex}.dxf"
            full_path = os.path.join(directory, filename)
            
            # 保存DXF文件
            doc.saveas(full_path)
            print(f"控制点已保存到: {full_path}")
        else:
            # 如果文档未保存，则保存到当前工作目录
            filename = f"控制点{uuid4().hex}.dxf"
            doc.saveas(filename)
            print(f"控制点已保存到: {os.path.abspath(filename)}")
            
    except Exception as e:
        print(f"绘制控制点时出错: {e}")

def rotate_point(point, angle, center=(0, 0)):
    """绕指定中心点旋转点"""
    x, y = point
    cx, cy = center
    
    x -= cx
    y -= cy
    
    rad = math.radians(angle)
    cos_rad, sin_rad = math.cos(rad), math.sin(rad)
    new_x = x * cos_rad - y * sin_rad
    new_y = x * sin_rad + y * cos_rad
    
    new_x += cx
    new_y += cy
    
    return (new_x, new_y)

def get_bounding_box_area(points, angle):
    """计算给定角度下的包围盒面积和边界信息"""
    rotated_points = [rotate_point(p, -angle) for p in points]
    
    x_coords = [p[0] for p in rotated_points]
    y_coords = [p[1] for p in rotated_points]
    
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    width = max_x - min_x
    height = max_y - min_y
    area = width * height
    
    return area, (min_x, max_x, min_y, max_y)

def ternary_search_min_area(points, left, right, eps=1e-6):
    """使用三分法搜索最小面积角度"""
    while right - left > eps:
        mid1 = left + (right - left) / 3
        mid2 = right - (right - left) / 3
        
        area1, _ = get_bounding_box_area(points, mid1)
        area2, _ = get_bounding_box_area(points, mid2)
        
        if area1 < area2:
            right = mid2
        else:
            left = mid1
    
    optimal_angle = (left + right) / 2
    min_area, bounds = get_bounding_box_area(points, optimal_angle)
    return optimal_angle, min_area, bounds

def calculate_obb(points):
    """计算最小面积包围矩形"""
    if len(points) < 2:
        return None
    
    # 首先找到一个较好的初始角度范围
    angles_to_check = []
    
    # 使用较小步长进行初步搜索
    for i in range(0, 180, 2):
        angles_to_check.append(i)
    
    # 找到距离最远的点对
    max_dist = 0
    farthest_pair = None
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = math.sqrt((points[i][0]-points[j][0])**2 + (points[i][1]-points[j][1])**2)
            if dist > max_dist:
                max_dist = dist
                farthest_pair = (points[i], points[j])
    
    if farthest_pair:
        p1, p2 = farthest_pair
        angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        angles_to_check.extend([angle, angle + 90])
    
    angles_to_check = list(set([a % 180 for a in angles_to_check]))
    
    # 找到初步的最小面积角度
    min_area = float('inf')
    best_angle = 0
    best_bounds = None
    
    for angle in angles_to_check:
        area, bounds = get_bounding_box_area(points, angle)
        if area < min_area:
            min_area = area
            best_angle = angle
            best_bounds = bounds
    
    # 在最优角度附近使用三分法进行精细搜索
    search_range = 5  # 搜索范围±5度
    left_angle = (best_angle - search_range) % 180
    right_angle = (best_angle + search_range) % 180
    
    # 处理跨越0度的情况
    if left_angle > right_angle:
        # 在[0, right_angle]和[left_angle, 180]两个区间分别搜索
        optimal_angle1, min_area1, bounds1 = ternary_search_min_area(points, 0, right_angle)
        optimal_angle2, min_area2, bounds2 = ternary_search_min_area(points, left_angle, 180)
        
        if min_area1 < min_area2:
            optimal_angle = optimal_angle1
            min_area = min_area1
            best_bounds = bounds1
        else:
            optimal_angle = optimal_angle2
            min_area = min_area2
            best_bounds = bounds2
    else:
        optimal_angle, min_area, best_bounds = ternary_search_min_area(points, left_angle, right_angle)
    
    # 构造最终的包围盒
    min_x, max_x, min_y, max_y = best_bounds
    width = max_x - min_x
    height = max_y - min_y
    
    # 计算旋转后的四个角点
    rotated_corners = [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y)
    ]
    
    # 将角点旋转回原始坐标系
    corners = [rotate_point(p, optimal_angle) for p in rotated_corners]
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_original = rotate_point((center_x, center_y), optimal_angle)
    
    return {
        'type': 'OBB',
        'corners': corners,
        'width': width,
        'height': height,
        'area': min_area,
        'angle': optimal_angle,
        'center': center_original
    }

# 主函数调用示例
if __name__ == "__main__":
    get_selected_faces_edges_and_obb()