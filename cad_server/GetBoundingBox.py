from pyautocad import Autocad, APoint
import pythoncom

# 连接到AutoCAD
acad = Autocad(create_if_not_exists=True)
doc = acad.doc

# 获取当前选择集
try:
    # 获取当前活动的选择集
    selection_set = doc.ActiveSelectionSet
except:
    print("请先在AutoCAD中选择对象。")
    exit()

# 如果没有选中对象，则提示用户选择
if selection_set.Count == 0:
    print("未检测到已选择的对象，正在等待用户选择对象...")
    # 创建一个新的选择集
    try:
        # 尝试使用现有选择集或创建新的
        selection_set = doc.SelectionSets.Add("TempSelection")
    except:
        # 如果同名选择集已存在，先删除再创建
        try:
            existing_ss = doc.SelectionSets.Item("TempSelection")
            existing_ss.Delete()
            selection_set = doc.SelectionSets.Add("TempSelection")
        except:
            # 如果还失败，则使用不同的名称
            import time
            selection_set = doc.SelectionSets.Add(f"TempSelection_{int(time.time())}")
            
    # 提示用户在屏幕上选择对象
    try:
        selection_set.SelectOnScreen()
    except:
        print("选择操作被取消或出现错误。")
        exit()
    
    if selection_set.Count == 0:
        print("用户未选择任何对象，程序退出。")
        exit()

# 初始化边界值
min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

valid_objects_count = 0

# 遍历选中对象，计算整体边界框
for entity in selection_set:
    try:
        # 正确调用GetBoundingBox方法，传入两个None参数来接收返回值
        min_point, max_point = entity.GetBoundingBox(None, None)
        
        # 提取坐标值
        obj_min_x, obj_min_y, obj_min_z = min_point
        obj_max_x, obj_max_y, obj_max_z = max_point
        
        # 更新整体边界
        min_x = min(min_x, obj_min_x)
        min_y = min(min_y, obj_min_y)
        min_z = min(min_z, obj_min_z)
        max_x = max(max_x, obj_max_x)
        max_y = max(max_y, obj_max_y)
        max_z = max(max_z, obj_max_z)
        
        valid_objects_count += 1
        
        # 如果是线段对象，添加线段标注
        try:
            # 检查对象类型是否为线段 (AcDbLine)
            if entity.ObjectName == 'AcDbLine':
                # 获取线段起点和终点
                start_point = entity.StartPoint
                end_point = entity.EndPoint
                
                start_x, start_y, start_z = start_point
                end_x, end_y, end_z = end_point
                
                # 计算线段长度
                line_length = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
                
                # 创建线段标注
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                
                # 根据线段方向确定标注位置
                if abs(end_x - start_x) > abs(end_y - start_y):  # 接近水平线
                    # 水平标注放在上方
                    dim_pt = APoint(mid_x, mid_y + 2, start_z)
                else:  # 接近垂直线
                    # 垂直标注放在右方
                    dim_pt = APoint(mid_x + 2, mid_y, start_z)
                
                # 添加线段长度标注
                start_apoint = APoint(start_x, start_y, start_z)
                end_apoint = APoint(end_x, end_y, end_z)
                dim = acad.model.AddDimAligned(start_apoint, end_apoint, dim_pt)
                print(f"已为线段添加长度标注: {line_length:.2f}")
        except Exception as line_e:
            print(f"为线段添加标注时出错: {line_e}")
            
    except Exception as e:
        print(f"无法获取对象 {entity} 的边界框: {e}")

if valid_objects_count == 0:
    print("没有有效的对象可用于计算边界框。")
    exit()

# 计算整体长度和宽度
length = max_x - min_x
width = max_y - min_y

print(f"选中物体整体长度: {length:.2f}, 宽度: {width:.2f}")

# 添加整体边界标注 (水平和垂直线性标注)
# 定义标注点
pt1 = APoint(min_x, min_y, min_z)  # 起点
pt2_length = APoint(max_x, min_y, min_z)  # 长度终点
pt2_width = APoint(min_x, max_y, min_z)   # 宽度终点

# 添加长度标注（水平）
try:
    dim_length = acad.model.AddDimAligned(pt1, pt2_length, APoint((min_x + max_x) / 2, min_y - 5, min_z))
    print("已添加整体长度标注。")
except Exception as e:
    print(f"添加整体长度标注失败: {e}")

# 添加宽度标注（垂直）
try:
    dim_width = acad.model.AddDimAligned(pt1, pt2_width, APoint(min_x - 5, (min_y + max_y) / 2, min_z))
    print("已添加整体宽度标注。")
except Exception as e:
    print(f"添加整体宽度标注失败: {e}")

# 清理临时选择集（如果创建了的话）
try:
    # 查找并删除我们创建的临时选择集
    for ss in doc.SelectionSets:
        if "TempSelection" in ss.Name:
            ss.Delete()
            break
except:
    pass