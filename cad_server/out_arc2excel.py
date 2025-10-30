# export_arc_data.py
from pyautocad import Autocad, APoint
import math
import openpyxl
from openpyxl import Workbook
import os
from collections import Counter

def get_all_arcs_from_selection(acad, doc):
    """
    一次性获取所有选中的圆弧对象
    
    Args:
        acad: AutoCAD应用程序对象
        doc: AutoCAD文档对象
    
    Returns:
        list: 所有选中的圆弧对象列表
    """
    arcs = []
    
    try:
        # 提示用户选择对象
        selection_set = doc.SelectionSets.Add("SelectAllArcs")
        
        print("请在AutoCAD中选择所有需要处理的圆弧对象...")
        selection_set.SelectOnScreen()
        
        # 遍历选中的对象，筛选出圆弧
        for i in range(selection_set.Count):
            try:
                entity = selection_set.Item(i)
                if entity.ObjectName == "AcDbArc":
                    arcs.append(entity)
            except Exception as e:
                print(f"检查选中对象 {i} 时出错: {e}")
                continue
        
        # 清理选择集
        selection_set.Delete()
        
        print(f"总共选中了 {len(arcs)} 个圆弧对象")
        
    except Exception as e:
        print(f"获取选中对象时出错: {e}")
        
    return arcs

def separate_arcs_by_layer(arcs):
    """
    根据图层名称将圆弧分为外弧和内弧
    
    Args:
        arcs: 圆弧对象列表
    
    Returns:
        tuple: (外弧列表, 内弧列表)
    """
    outer_arcs = []
    inner_arcs = []
    
    for arc in arcs:
        if arc.Layer == "外弧":
            outer_arcs.append(arc)
        elif arc.Layer == "内弧":
            inner_arcs.append(arc)
    
    print(f"分离结果: 外弧 {len(outer_arcs)} 个, 内弧 {len(inner_arcs)} 个")
    return outer_arcs, inner_arcs

def sort_arcs_by_start_point(arcs, y_tolerance=1000):
    """
    按起点Y坐标分组（考虑容差），然后每组内按X坐标顺序对圆弧进行排序
    修改为按Y坐标降序排列
    
    Args:
        arcs: 圆弧对象列表
        y_tolerance: Y坐标容差值，默认为10
    
    Returns:
        list: 排序后的圆弧对象列表
    """
    def get_arc_start_point(arc):
        # 获取圆弧起点坐标
        start_point = arc.StartPoint
        return (start_point[0], start_point[1])  # (x, y)
    
    # 先按Y坐标分组（考虑容差）
    grouped_arcs = []
    used_arcs = set()
    
    # 按Y坐标降序排序所有圆弧
    arcs_sorted_by_y = sorted(arcs, key=lambda a: get_arc_start_point(a)[1], reverse=True)
    
    for i, arc in enumerate(arcs_sorted_by_y):
        if i in used_arcs:
            continue
            
        current_y = get_arc_start_point(arc)[1]
        current_group = [arc]
        used_arcs.add(i)
        
        # 查找Y坐标在容差范围内的其他圆弧
        for j, other_arc in enumerate(arcs_sorted_by_y):
            if j in used_arcs:
                continue
                
            other_y = get_arc_start_point(other_arc)[1]
            if abs(current_y - other_y) <= y_tolerance:
                current_group.append(other_arc)
                used_arcs.add(j)
        
        # 对当前组内的圆弧按X坐标升序排序
        current_group.sort(key=lambda a: get_arc_start_point(a)[0])
        grouped_arcs.extend(current_group)
    
    print(f"排序前圆弧数量: {len(arcs)}")
    print(f"排序后圆弧数量: {len(grouped_arcs)}")
    
    return grouped_arcs

def ensure_layer_exists(doc, layer_name):
    """
    确保指定图层存在，如果不存在则创建
    
    Args:
        doc: AutoCAD文档对象
        layer_name: 图层名称
    
    Returns:
        layer: 图层对象
    """
    try:
        # 尝试获取现有图层
        layer = doc.Layers.Item(layer_name)
        print(f"图层 '{layer_name}' 已存在")
        return layer
    except:
        # 如果图层不存在，则创建新图层
        try:
            layer = doc.Layers.Add(layer_name)
            print(f"已创建图层 '{layer_name}'")
            return layer
        except Exception as e:
            print(f"创建图层 '{layer_name}' 失败: {e}")
            return None

def add_number_to_arc(acad, doc, arc, number):  
    try:  
        # 获取圆弧中点坐标
        start_point = arc.StartPoint
        end_point = arc.EndPoint
        center = arc.Center   
        start_angle = arc.StartAngle
        end_angle = arc.EndAngle
        
        # 计算中点坐标
        mid_x = (start_point[0] + end_point[0]) / 2
        radius = arc.Radius  
        mid_y = center[1] + radius * math.sin((start_angle + end_angle) / 2)  
        
        # 调整文本位置，使其位于圆弧内侧
        # 计算圆心到起点的向量
        dx = start_point[0] - center[0]
        dy = start_point[1] - center[1]
        
        # 计算从圆心指向圆弧中点的向量，并将其反向以放置在圆弧内侧
        direction_x = mid_x - center[0]
        direction_y = mid_y - center[1]
        
        # 归一化方向向量
        length = math.sqrt(direction_x**2 + direction_y**2)
        if length > 0:
            direction_x /= length
            direction_y /= length
            
            # 将文本位置稍微偏移至圆弧内侧
            text_x = mid_x - direction_x * radius * 0.1
            text_y = mid_y - direction_y * radius * 0.1
        else:
            # 如果出现问题，回退到原始计算方法
            text_x = mid_x
            text_y = start_point[1] - 3 * (mid_y - start_point[1])
            
        # 使用 APoint 创建坐标点
        text_point = APoint(text_x, text_y)  
          
        # 添加文本，减小字体大小为半径的0.05倍
        text_obj = doc.ModelSpace.AddText(str(number), text_point, radius * 0.08)  
        text_obj.Layer = "编号"
          
        print(f"已为圆弧添加序号: {number}")  
    except Exception as e:  
        print(f"为圆弧添加序号 {number} 时出错: {e}")

def calculate_arc_length(arc):
    """
    计算圆弧的长度并向上取整
    
    Args:
        arc: AutoCAD圆弧对象
    
    Returns:
        int: 向上取整后的圆弧长度和半径
    """
    try:
        radius = arc.Radius
        start_angle = arc.StartAngle
        end_angle = arc.EndAngle
        
        # 计算圆弧角度差
        if end_angle < start_angle:
            angle_diff = (2 * math.pi - start_angle) + end_angle
        else:
            angle_diff = end_angle - start_angle
            
        # 计算弧长 = 半径 × 弧度
        arc_length = radius * angle_diff
        
        # 向上取整
        arc_length_ceil = math.ceil(arc_length)
        radius_ceil = math.ceil(radius)
        
        return arc_length_ceil, radius_ceil
    except Exception as e:
        print(f"计算圆弧长度时出错: {e}")
        return 0, 0

def export_arc_data_to_excel(outer_arcs, inner_arcs, filename="dxf_output/arc2excel/arc_data.xlsx"):
    """
    将外弧长度和内弧半径导出到Excel文件
    修改为外弧使用内弧图层的半径
    
    Args:
        outer_arcs: 外弧对象列表
        inner_arcs: 内弧对象列表
        filename: 导出的Excel文件名
    """
    try:
        # 确保文件路径存在
        file_dir = os.path.dirname(filename)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir)
            print(f"创建目录: {file_dir}")
        
        # 计算外弧的长度，但使用内弧的半径
        outer_arc_data = []
        for i, arc in enumerate(outer_arcs):
            length, _ = calculate_arc_length(arc)
            # 使用对应内弧的半径（如果内弧数量足够）
            if i < len(inner_arcs):
                radius = math.ceil(inner_arcs[i].Radius)
            else:
                # 如果内弧数量不足，回退到使用外弧自身的半径
                _, radius = calculate_arc_length(arc)
            outer_arc_data.append((length, radius))
        
        # 计算内弧的半径
        inner_arc_data = []
        for arc in inner_arcs:
            radius = math.ceil(arc.Radius)
            inner_arc_data.append(radius)
        
        # 创建工作簿和工作表
        wb = Workbook()
        
        # 外弧工作表
        ws_outer = wb.active
        ws_outer.title = "外弧长度"
        
        # 写入表头
        ws_outer['A1'] = "序号"
        ws_outer['B1'] = "弧长(取整)"
        ws_outer['C1'] = "半径"  # 这里的半径将使用内弧图层的半径
        
        # 写入外弧数据
        for idx, (length, radius) in enumerate(outer_arc_data, start=1):
            ws_outer[f'A{idx+1}'] = idx
            ws_outer[f'B{idx+1}'] = length
            ws_outer[f'C{idx+1}'] = radius
            print(f"外弧 {idx}: 弧长={length}, 半径={radius}")
        
        # 内弧工作表
        ws_inner = wb.create_sheet("内弧半径")
        
        # 写入表头
        ws_inner['A1'] = "序号"
        ws_inner['B1'] = "半径"
        
        # 写入内弧数据
        for idx, radius in enumerate(inner_arc_data, start=1):
            ws_inner[f'A{idx+1}'] = idx
            ws_inner[f'B{idx+1}'] = radius
            print(f"内弧 {idx}: 半径={radius}")
        
        # 保存文件
        wb.save(filename)
        print(f"成功导出弧数据到 {filename}")
        
    except Exception as e:
        print(f"导出到Excel时出错: {e}")

def main():
    """
    主函数 - 一次性选择所有弧，然后分离内外弧进行处理
    """
    # 连接到正在运行的 AutoCAD
    try:
        acad = Autocad(create_if_not_exists=True)
        doc = acad.doc
        print(f"成功连接到 AutoCAD 文档: {doc.Name}")
    except Exception as e:
        print("无法连接到 AutoCAD:", e)
        return
    
    try:
        # 确保"编号"图层存在（只需执行一次）
        ensure_layer_exists(doc, "编号")
        
        # 一次性选择所有圆弧对象
        all_arcs = get_all_arcs_from_selection(acad, doc)
        
        if not all_arcs:
            print("没有选中任何圆弧对象")
            return
        
        # 根据图层分离内外弧
        outer_arcs, inner_arcs = separate_arcs_by_layer(all_arcs)
        
        if not outer_arcs and not inner_arcs:
            print("选中的圆弧中不包含'外弧'或'内弧'图层的对象")
            return
            
        # 处理外弧（如果存在）
        sorted_outer_arcs = []
        if outer_arcs:
            print(f"找到 {len(outer_arcs)} 个外弧对象")
            
            # 对外弧按起点坐标进行排序
            sorted_outer_arcs = sort_arcs_by_start_point(outer_arcs)
            
            # 为每个排序后的外弧添加序号
            for index, arc in enumerate(sorted_outer_arcs, start=1):
                add_number_to_arc(acad, doc, arc, index)
        else:
            print("未找到外弧对象")
        
        # 处理内弧（如果存在）
        sorted_inner_arcs = []
        if inner_arcs:
            print(f"找到 {len(inner_arcs)} 个内弧对象")
            
            # 对内弧按起点坐标进行排序
            sorted_inner_arcs = sort_arcs_by_start_point(inner_arcs)
            
            # 为每个排序后的内弧添加序号
            for index, arc in enumerate(sorted_inner_arcs, start=1):
                add_number_to_arc(acad, doc, arc, index)
        else:
            print("未找到内弧对象")
        
        # 如果至少有一种类型的弧存在，就导出数据到Excel
        if outer_arcs or inner_arcs:
            export_arc_data_to_excel(
                sorted_outer_arcs if outer_arcs else [], 
                sorted_inner_arcs if inner_arcs else [], 
                "dxf_output/arc2excel/弧数据.xlsx"
            )
        
        # 重新生成视图以显示新添加的编号
        doc.Regen()
            
    except Exception as e:
        print(f"处理对象时出错: {e}")

if __name__ == "__main__":
    main()