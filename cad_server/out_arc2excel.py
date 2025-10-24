# export_arc_length.py
from pyautocad import Autocad, APoint
import math
import openpyxl
from openpyxl import Workbook
import os
from collections import Counter

def get_arcs_from_layer(acad, doc, layer_name="外弧"):
    """
    获取用户选择的圆弧对象
    
    Args:
        acad: AutoCAD应用程序对象
        doc: AutoCAD文档对象
    
    Returns:
        list: 用户选择的圆弧对象列表
    """
    arcs = []
    
    try:
        # 提示用户选择对象
        selection_set = doc.SelectionSets.Add("SelectArcs")
        
        print("请在AutoCAD中选择圆弧对象...")
        selection_set.SelectOnScreen()
        
        # 遍历选中的对象，筛选出圆弧
        for i in range(selection_set.Count):
            try:
                entity = selection_set.Item(i)
                if entity.ObjectName == "AcDbArc" and entity.Layer != layer_name:
                    arcs.append(entity)
            except Exception as e:
                print(f"检查选中对象 {i} 时出错: {e}")
                continue
        
        # 清理选择集
        selection_set.Delete()
        
        print(f"选中了 {len(arcs)} 个圆弧对象")
        
    except Exception as e:
        print(f"获取选中对象时出错: {e}")
        
    return arcs

def sort_arcs_by_start_point(arcs, y_tolerance=1000):
    """
    按起点Y坐标分组（考虑容差），然后每组内按X坐标顺序对圆弧进行排序
    
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
    
    for i, arc in enumerate(arcs):
        if i in used_arcs:
            continue
            
        current_y = get_arc_start_point(arc)[1]
        current_group = [arc]
        used_arcs.add(i)
        
        # 查找Y坐标在容差范围内的其他圆弧
        for j, other_arc in enumerate(arcs):
            if j in used_arcs:
                continue
                
            other_y = get_arc_start_point(other_arc)[1]
            if abs(current_y - other_y) <= y_tolerance:
                current_group.append(other_arc)
                used_arcs.add(j)
        
        # 对当前组内的圆弧按X坐标排序
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
        # 确保"编号"图层存在
        ensure_layer_exists(doc, "编号")
        
        # 获取圆弧中点坐标
        start_point = arc.StartPoint
        end_point = arc.EndPoint
        center = arc.Center   
        start_angle = arc.StartAngle
        end_angle = arc.EndAngle
        start_x = start_point[0]  
        start_y = start_point[1]  
        # 计算中点坐标
        mid_x = (start_point[0] + end_point[0]) / 2

        radius = arc.Radius  
        mid_y = center[1] + radius * math.sin((start_angle + end_angle) / 2)  
        # 使用 APoint 创建坐标点，放在中点附近
        text_point = APoint(mid_x, start_y - 3 * (mid_y - start_y))  
          
        # 减小字体大小为半径的0.15倍（原来是0.3倍）
        text_obj = doc.ModelSpace.AddText(str(number), text_point, radius * 0.05)  
        text_obj.Layer = "编号"  
        text_obj.TextMovement = 2  # 固定文字位置
        text_obj.TextPosition = text_point
          
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

def export_arc_lengths_to_excel(arcs, filename="dxf_output/arc2excel/arc_lengths.xlsx"):
    """
    将圆弧长度和半径导出到Excel文件
    
    Args:
        arcs: 圆弧对象列表
        filename: 导出的Excel文件名
    """
    try:
        # 确保文件路径存在
        file_dir = os.path.dirname(filename)
        if file_dir and not os.path.exists(file_dir):
            os.makedirs(file_dir)
            print(f"创建目录: {file_dir}")
        
        # 计算所有圆弧的长度和半径
        arc_data = []
        for arc in arcs:
            length, radius = calculate_arc_length(arc)
            arc_data.append((length, radius))
        
        # 创建工作簿和工作表
        wb = Workbook()
        ws = wb.active
        ws.title = "圆弧长度"
        
        # 写入表头
        ws['A1'] = "弧长(取整)"
        ws['B1'] = "半径"
        
        # 写入每个弧长和半径数据
        for idx, (length, radius) in enumerate(arc_data, start=2):
            ws[f'A{idx}'] = length
            ws[f'B{idx}'] = radius
            print(f"弧长: {length}, 半径: {radius}")
        
        # 保存文件
        wb.save(filename)
        print(f"成功导出弧长和半径信息到 {filename}")
        
    except Exception as e:
        print(f"导出到Excel时出错: {e}")

def main():
    """
    主函数 - 导出"外弧"图层中圆弧的弧长到Excel（只包含取整后的弧长和数量）
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
        # 确保"编号"图层存在
        ensure_layer_exists(doc, "编号")
        
        # 从"外弧"图层获取圆弧对象
        arcs = get_arcs_from_layer(acad, doc, "外弧")
        
        if not arcs:
            print("在'外弧'图层中没有找到圆弧对象")
            return
        
        print(f"找到 {len(arcs)} 个圆弧对象")
        
        # 对圆弧按起点坐标进行排序
        sorted_arcs = sort_arcs_by_start_point(arcs)
        
        # 为每个排序后的圆弧添加序号
        for index, arc in enumerate(sorted_arcs, start=1):
            add_number_to_arc(acad, doc, arc, index)
        
        # 导出圆弧长度到Excel
        export_arc_lengths_to_excel(sorted_arcs, "dxf_output/arc2excel/外弧长度数据.xlsx")
        
        # 重新生成视图以显示新添加的编号
        doc.Regen()
            
    except Exception as e:
        print(f"处理对象时出错: {e}")

if __name__ == "__main__":
    main()