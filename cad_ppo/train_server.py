# train_server.py (简化版)
import ezdxf
import numpy as np
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Arc
import os 
import subprocess
import time
import json
from datetime import datetime
matplotlib.rcParams['figure.max_open_warning'] = 0
# 控制是否显示耗时信息的全局变量
SHOW_TIMING = True

def convert_dwg_to_dxf(dwg_file_path):
    """
    使用ODA File Converter将DWG文件转换为DXF文件
    
    Args:
        dwg_file_path (str): DWG文件路径
        
    Returns:
        dict: 转换结果，包含状态和DXF文件路径或错误信息
    """
    start_time = time.time()
    try:
        # 检查DWG文件是否存在
        if not os.path.exists(dwg_file_path):
            return {
                "status": "error",
                "message": f"DWG文件不存在: {dwg_file_path}"
            }

        # 获取文件目录和文件名
        dwg_dir = os.path.dirname(dwg_file_path)
        dwg_filename = os.path.basename(dwg_file_path)
        dxf_filename = dwg_filename.replace('.dwg', '.dxf')
        dxf_file_path = dwg_file_path.replace('.dwg', '.dxf')
        if dxf_file_path == dwg_file_path:
            dxf_file_path = f"{dwg_file_path}.dxf"

        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(dxf_file_path)
        os.makedirs(output_dir, exist_ok=True)

        # 参数定义（按官方规范）
        TEIGHA_PATH = "ODAFileConverter"
        INPUT_FOLDER = dwg_dir
        OUTPUT_FOLDER = output_dir
        OUTVER = "ACAD2018"           # 输入版本（如原文件是 ACAD2018）
        OUTFORMAT = "DXF"             # 输出格式：只能是 DXF / DWG / DXB
        RECURSIVE = "0"
        AUDIT = "1"
        INPUTFILTER = "*.DWG"         # 只处理DWG文件

        # 构建命令（注意顺序！）
        cmd = [
            TEIGHA_PATH,
            INPUT_FOLDER,
            OUTPUT_FOLDER,
            OUTVER,           # 输入版本
            OUTFORMAT,        # 输出格式 ← 必须是 DXF/DWG/DXB
            RECURSIVE,
            AUDIT,
            INPUTFILTER
        ]

        print(f"执行命令: {' '.join(cmd)}")

        # 执行转换命令
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

        # 打印调试信息
        print("ODA 输出:", result.stdout)
        print("ODA 错误:", result.stderr)

        if result.returncode == 0:
            # 检查DXF文件是否生成
            if os.path.exists(dxf_file_path):
                # 验证DXF文件是否有效
                if is_valid_dxf(dxf_file_path):
                    response = {
                        "status": "success",
                        "message": f"DWG文件已成功转换为DXF: {dxf_file_path}",
                        "dxf_path": dxf_file_path
                    }
                    if SHOW_TIMING:
                        response["processing_time"] = time.time() - start_time
                    return response
                else:
                    response = {
                        "status": "error",
                        "message": f"生成的DXF文件结构不完整或损坏: {dxf_file_path}"
                    }
                    if SHOW_TIMING:
                        response["processing_time"] = time.time() - start_time
                    return response
            else:
                response = {
                    "status": "error",
                    "message": f"转换完成但未生成DXF文件。请检查输出目录: {output_dir}"
                }
                if SHOW_TIMING:
                        response["processing_time"] = time.time() - start_time
                return response
        else:
            response = {
                "status": "error",
                "message": f"ODA转换失败: {result.stderr}"
            }
            if SHOW_TIMING:
                response["processing_time"] = time.time() - start_time
            return response

    except subprocess.TimeoutExpired:
        response = {
            "status": "error",
            "message": "转换超时"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response
    except FileNotFoundError:
        response = {
            "status": "error",
            "message": "未找到ODAFileConverter命令，请确保ODA File Converter已正确安装并加入PATH"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response
    except Exception as e:
        response = {
            "status": "error",
            "message": f"转换过程中发生错误: {str(e)}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response

def get_dxf_objects(dxf_file_path):
    """
    获取DXF文件中所有对象的签名信息
    
    Args:
        dxf_file_path: DXF文件路径
        
    Returns:
        dict: 对象签名列表
    """
    start_time = time.time()
    
    if not os.path.exists(dxf_file_path):
        response = {
            "status": "error", 
            "message": f"文件不存在: {dxf_file_path}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response

    try:
        # 加载DXF文件并收集对象签名
        doc_read_start = time.time()
        print(f"开始读取DXF文件: {dxf_file_path}")
        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()
        doc_read_time = time.time() - doc_read_start
        print(f"成功读取DXF文件，包含 {len(msp)} 个实体")

        # 收集所有对象签名
        signature_collection_start = time.time()
        signatures = []
        entity_types = {}
        for entity in msp:
            entity_type = entity.dxftype()
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            signature = generate_entity_signature(entity)
            signatures.append(signature)
        signature_collection_time = time.time() - signature_collection_start

        print(f"实体类型统计: {entity_types}")
        print(f"生成了 {len(signatures)} 个签名")

        response = {
            'status': 'success',
            'signatures': signatures,
            'object_count': len(signatures),
            'file_path': dxf_file_path,
            'entity_types': entity_types
        }
        
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
            response["timing_details"] = {
                "doc_read_time": doc_read_time,
                "signature_collection_time": signature_collection_time
            }
            
        return response

    except Exception as e:
        response = {
            'status': 'error',
            'message': f"处理文件时出错: {str(e)}"
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response

def is_valid_dxf(dxf_path):
    """检查DXF文件是否有效"""
    try:
        doc = ezdxf.readfile(dxf_path)
        return True
    except Exception as e:
        print(f"DXF文件校验失败: {e}")
        return False

def generate_entity_signature(entity):
    """
    为实体生成特征签名，用于跨文件匹配
    
    Args:
        entity: DXF实体对象
        
    Returns:
        dict: 实体的特征签名
    """
    signature = {
        "type": entity.dxftype(),
        "handle": entity.dxf.handle if hasattr(entity.dxf, 'handle') else None
    }
    
    if entity.dxftype() == 'LINE':
        start = entity.dxf.start
        end = entity.dxf.end
        # 为了便于匹配，对起点和终点进行排序
        points = sorted([(round(start.x, 6), round(start.y, 6)), 
                         (round(end.x, 6), round(end.y, 6))])
        signature.update({
            "start": points[0],
            "end": points[1],
            "length": round(((end.x - start.x)**2 + (end.y - start.y)**2)**0.5, 6)
        })
    elif entity.dxftype() == 'CIRCLE':
        center = entity.dxf.center
        signature.update({
            "center": (round(center.x, 6), round(center.y, 6)),
            "radius": round(entity.dxf.radius, 6)
        })
    elif entity.dxftype() == 'ARC':
        center = entity.dxf.center
        signature.update({
            "center": (round(center.x, 6), round(center.y, 6)),
            "radius": round(entity.dxf.radius, 6),
            "start_angle": round(entity.dxf.start_angle, 6),
            "end_angle": round(entity.dxf.end_angle, 6)
        })
    elif entity.dxftype() == 'SPLINE':
        signature.update({
            "degree": entity.dxf.degree if hasattr(entity.dxf, 'degree') else None,
            "fit_points_count": len(entity.fit_points) if hasattr(entity, 'fit_points') else 0,
            "control_points_count": len(entity.control_points) if hasattr(entity, 'control_points') else 0
        })
        
    return signature

def process_dxf_file_with_specific_highlight(dxf_file_path, target_entity):
    """
    处理DXF文件并生成图像，将高亮实体标红，其他实体保持原有颜色
    
    Args:
        dxf_file_path (str): DXF文件路径
        target_entity: 需要高亮的特定实体对象
        
    Returns:
        dict: 处理结果
    """
    start_time = time.time()
    # DXF颜色索引到matplotlib颜色的映射
    dxf_color_map = {
        0: 'black', 1: 'red', 2: 'yellow', 3: 'green', 4: 'cyan',
        5: 'blue', 6: 'magenta', 7: 'white', 8: '#a5a5a5', 9: '#c0c0c0',
        10: 'red', 11: '#ffaaaa', 12: '#bd0000', 13: '#bd7373', 14: '#800000',
        15: '#ff0000', 16: '#ffff00', 17: '#ffff73', 18: '#bda000', 19: '#bdae73',
        20: '#808000', 21: '#ffff00', 22: '#00ff00', 23: '#aaffaa', 24: '#00bd00',
        25: '#73bd73', 26: '#008000', 27: '#00ff00', 28: '#00ffff', 29: '#aaffff',
        30: '#00bfbf', 31: '#73bfbf', 32: '#008080', 33: '#00ffff', 34: '#0000ff',
        35: '#aaaaff', 36: '#0000bd', 37: '#7373bf', 38: '#000080', 39: '#0000ff',
        40: '#ff00ff', 41: '#ffaaff', 42: '#bd00bd', 43: '#bd73bd', 44: '#800080',
        45: '#ff00ff', 'default': 'black'
    }

    try:
        # 检查文件是否存在
        if not os.path.exists(dxf_file_path):
            response = {
                "status": "error",
                "message": f"文件不存在: {dxf_file_path}"
            }
            if SHOW_TIMING:
                response["processing_time"] = time.time() - start_time
            return response

        # 加载DXF文件
        print(f"开始加载DXF文件: {dxf_file_path}")
        doc = ezdxf.readfile(dxf_file_path)
        msp = doc.modelspace()
        print(f"成功加载DXF文件，包含 {len(msp)} 个实体")

        # 创建图形对象
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        
        # 确定高亮实体
        highlighted_entity = target_entity
        if highlighted_entity:
            print(f"指定高亮实体: {highlighted_entity.dxftype()}")

        all_x_coords = []
        all_y_coords = []
        highlighted_entity_signature = None

        # 处理实体
        entity_counts = {}  # 统计各类实体数量
        processed_entities = 0
        
        for entity in msp:
            # 统计实体类型
            entity_type = entity.dxftype()
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
            
            # 判断当前实体是否为高亮实体
            is_highlighted = (entity.dxf.handle == highlighted_entity.dxf.handle 
                  if highlighted_entity and hasattr(entity.dxf, 'handle') and hasattr(highlighted_entity.dxf, 'handle') 
                  else False)
            linewidth = 1.5 if is_highlighted else 0.5

            # 获取实体颜色
            try:
                if hasattr(entity.dxf, 'color') and entity.dxf.color in dxf_color_map:
                    actual_color = dxf_color_map[entity.dxf.color]
                else:
                    actual_color = dxf_color_map['default']
            except:
                actual_color = dxf_color_map['default']
            
            # 如果是高亮状态，使用红色
            color = 'red' if is_highlighted else actual_color
           
    
            if entity.dxftype() == 'LINE':
                start = entity.dxf.start
                end = entity.dxf.end
                # 直接绘制所有线条，避免使用LineCollection
                ax.plot([start.x, end.x], [start.y, end.y], color=color, linewidth=linewidth)
                
                if is_highlighted:
                    highlighted_entity_signature = generate_entity_signature(entity)
                    print(f"高亮LINE实体: 起点({start.x}, {start.y}), 终点({end.x}, {end.y})")
                
                all_x_coords.extend([start.x, end.x])
                all_y_coords.extend([start.y, end.y])
                processed_entities += 1
                    
            elif entity.dxftype() == 'SPLINE':
                try:
                    if hasattr(entity, 'fit_points') and entity.fit_points:
                        points = [(p.x, p.y) for p in entity.fit_points]
                    else:
                        points = [(p[0], p[1]) for p in entity.control_points]
                        
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    ax.plot(x_coords, y_coords, color=color, linewidth=linewidth)
                    
                    if is_highlighted:
                        highlighted_entity_signature = generate_entity_signature(entity)
                        print(f"高亮SPLINE实体: {len(points)} 个点")
                            
                    all_x_coords.extend([p[0] for p in points])
                    all_y_coords.extend([p[1] for p in points])
                    processed_entities += 1
                except Exception as e:
                    print(f'处理SPLINE实体时出错: {e}')
                    
            elif entity.dxftype() == 'ARC':
                center = entity.dxf.center
                radius = entity.dxf.radius
                start_angle = entity.dxf.start_angle
                end_angle = entity.dxf.end_angle
                
                if start_angle > end_angle:
                    end_angle += 360
                    
                arc_patch = Arc((center.x, center.y), 2*radius, 2*radius, angle=0,
                                theta1=start_angle, theta2=end_angle,
                                color=color, linewidth=linewidth)
                ax.add_patch(arc_patch)
                
                if is_highlighted:
                    highlighted_entity_signature = generate_entity_signature(entity)
                    print(f"高亮ARC实体: 中心({center.x}, {center.y}), 半径{radius}, 角度{start_angle}-{end_angle}")
                    
                all_x_coords.extend([center.x - radius, center.x + radius])
                all_y_coords.extend([center.y - radius, center.y + radius])
                processed_entities += 1
                    
            elif entity.dxftype() == 'CIRCLE':
                center = entity.dxf.center
                radius = entity.dxf.radius
                
                circle_patch = plt.Circle((center.x, center.y), radius, fill=False, 
                                        color=color, linewidth=linewidth)
                ax.add_patch(circle_patch)
                
                if is_highlighted:
                    highlighted_entity_signature = generate_entity_signature(entity)
                    print(f"高亮CIRCLE实体: 中心({center.x}, {center.y}), 半径{radius}")
                    
                all_x_coords.extend([center.x - radius, center.x + radius])
                all_y_coords.extend([center.y - radius, center.y + radius])
                processed_entities += 1
          

        # 设置坐标范围
        if all_x_coords and all_y_coords:
            margin = 5
            ax.set_xlim(min(all_x_coords) - margin, max(all_x_coords) + margin)
            ax.set_ylim(min(all_y_coords) - margin, max(all_y_coords) + margin)
        else:
            ax.set_xlim(-10, 10)
            ax.set_ylim(-10, 10)

        # 保存图像
        plt.gca().set_aspect('equal', adjustable='box')
        
        # 如果是指定高亮实体，生成特殊文件名
        if target_entity and highlighted_entity:
            base_name = os.path.splitext(os.path.basename(dxf_file_path))[0]
            target_signature = generate_entity_signature(target_entity)
            unique_id = f"{base_name}_{target_entity.dxftype()}_{hash(str(target_signature)) % 10000}"
            output_path = os.path.join("delete_annotion_output", "pictures", f"{unique_id}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = f"{dxf_file_path}.png"
        
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=100, format='png')
        plt.close(fig)
        plt.close('all')

        response = {
            "status": "success",
            "message": f"DXF文件已处理并保存为 {output_path}",
            "output_path": output_path,
            "entity_stats": {
                "total_processed": processed_entities,
                "type_breakdown": entity_counts
            }
        }
        
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        
        # 如果有高亮实体，返回其类型和特征信息及签名
        if highlighted_entity:
            entity_info = {
                "type": highlighted_entity.dxftype(),
                "signature": highlighted_entity_signature
            }
            
            if highlighted_entity.dxftype() == 'LINE':
                start = highlighted_entity.dxf.start
                end = highlighted_entity.dxf.end
                entity_info["start"] = {"x": start.x, "y": start.y}
                entity_info["end"] = {"x": end.x, "y": end.y}
                entity_info["length"] = ((end.x - start.x)**2 + (end.y - start.y)**2)**0.5
            elif highlighted_entity.dxftype() == 'CIRCLE':
                center = highlighted_entity.dxf.center
                entity_info["center"] = {"x": center.x, "y": center.y}
                entity_info["radius"] = highlighted_entity.dxf.radius
            elif highlighted_entity.dxftype() == 'ARC':
                center = highlighted_entity.dxf.center
                entity_info["center"] = {"x": center.x, "y": center.y}
                entity_info["radius"] = highlighted_entity.dxf.radius
                entity_info["start_angle"] = highlighted_entity.dxf.start_angle
                entity_info["end_angle"] = highlighted_entity.dxf.end_angle
            elif highlighted_entity.dxftype() == 'SPLINE':
                if hasattr(highlighted_entity, 'fit_points') and highlighted_entity.fit_points:
                    entity_info["fit_points_count"] = len(highlighted_entity.fit_points)
                if hasattr(highlighted_entity, 'control_points'):
                    entity_info["control_points_count"] = len(highlighted_entity.control_points)
                    
            response["highlighted_entity"] = entity_info

        return response

    except Exception as e:
        response = {
            "status": "error",
            "message": str(e)
        }
        if SHOW_TIMING:
            response["processing_time"] = time.time() - start_time
        return response

def generate_training_data(dxf_file_path):
    """
    为RL训练生成数据集，包括图像和标注信息
    
    Args:
        dxf_file_path (str): DXF文件路径
        
    Returns:
        dict: 生成结果信息
    """
    start_time = time.time()
   
    output_dir="delete_annotion_output"
    try:
        print(f"开始生成训练数据: {dxf_file_path}")
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        student_dir = os.path.join(output_dir, "student")
        teacher_dir = os.path.join(output_dir, "teacher")
        picture_dir = os.path.join(output_dir, "pictures")
        os.makedirs(student_dir, exist_ok=True)
        os.makedirs(teacher_dir, exist_ok=True)
 
         # 确保图片目录存在
        
        # 处理文件
        file_ext = os.path.splitext(dxf_file_path)[1].lower()
        
        # 如果是DWG文件，先转换为DXF
        if file_ext == '.dwg':
            conversion_result = convert_dwg_to_dxf(dxf_file_path)
            if conversion_result["status"] != "success":
                return conversion_result
            dxf_path = conversion_result["dxf_path"]
        elif file_ext == '.dxf':
            dxf_path = dxf_file_path
        else:
            return {
                "status": "error",
                "message": f"不支持的文件格式: {file_ext}"
            }
        
        # 获取文件基础名
        base_name = os.path.splitext(os.path.basename(dxf_path))[0]
        print(f"处理文件基础名: {base_name}")
        
        # 根据文件来源确定教师签名目录
        # 如果当前处理的是student目录中的文件，则从teacher目录获取签名
        # 如果当前处理的是teacher目录中的文件，则从student目录获取签名
        file_parent_dir = os.path.basename(os.path.dirname(os.path.abspath(dxf_file_path)))
        print(f"文件来源目录: {file_parent_dir}")
        
        if file_parent_dir == "student":
            # 查找teacher目录中同名的文件
            teacher_dxf_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(dxf_file_path))), 
                "teacher", 
                os.path.basename(dxf_file_path)
            )
            
            print(f"查找对应的教师文件: {teacher_dxf_path}")
            if os.path.exists(teacher_dxf_path):
                # 获取教师签名
                teacher_signatures_result = get_dxf_objects(teacher_dxf_path)
                if teacher_signatures_result["status"] != "success":
                    return teacher_signatures_result
                    
                signatures = teacher_signatures_result["signatures"]
                print(f"成功获取教师签名，共 {len(signatures)} 个")
            else:
                return {
                    "status": "error",
                    "message": f"找不到对应的教师文件: {teacher_dxf_path}"
                }
                
        elif file_parent_dir == "teacher":
            # 查找student目录中同名的文件
            student_dxf_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(dxf_file_path))), 
                "student", 
                os.path.basename(dxf_file_path)
            )
            
            print(f"查找对应的学生文件: {student_dxf_path}")
            if os.path.exists(student_dxf_path):
                # 获取学生签名
                student_signatures_result = get_dxf_objects(student_dxf_path)
                if student_signatures_result["status"] != "success":
                    return student_signatures_result
                    
                signatures = student_signatures_result["signatures"]
                print(f"成功获取学生签名，共 {len(signatures)} 个")
            else:
                return {
                    "status": "error",
                    "message": f"找不到对应的学生文件: {student_dxf_path}"
                }
        else:
            return {
                "status": "error",
                "message": f"文件必须位于student或teacher目录中: {dxf_file_path}"
            }
        
        # 生成训练数据
        print(f"开始读取当前文件: {dxf_path}")
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        print(f"当前文件包含 {len(msp)} 个实体")
        
        # 为每个可显示实体生成训练样本
        training_samples = []
        sample_count = 0
        # 在函数开始处添加
        retain_count = 0
        delete_count = 0
        entity_count = 0
        for entity in msp:
            entity_count += 1
            if entity.dxftype() in ['LINE', 'CIRCLE', 'ARC']:
                print(f"处理第 {entity_count} 个实体: {entity.dxftype()}")
                # 重新处理文件，这次高亮当前实体
                specific_result = process_dxf_file_with_specific_highlight(dxf_path, entity)
                
                if specific_result["status"] == "success":
                    sample_count += 1
                    entity_signature = generate_entity_signature(entity)
                    print(f"成功生成第 {sample_count} 个样本")
                    
                    # 判断该实体是否应该被删除（基于对比签名）
                    should_delete = not is_entity_in_signal_signatures(entity_signature, signatures)
                    print(f"实体是否应该删除: {should_delete}")
                    if should_delete:
                        delete_count += 1
                    else:
                        retain_count += 1
                    sample_data = {
                        "sample_id": f"{base_name}_{sample_count}",
                        "image_path": specific_result["output_path"],
                        "entity_info": {
                            "type": entity.dxftype(),
                            "signature": entity_signature
                        },
                        "label": {
                            "should_delete": should_delete,
                            "action": 1 if should_delete else 0  # 1表示删除，0表示保留
                        }
                    }
                    
                    training_samples.append(sample_data)
                else:
                    print(f"处理实体时出错: {specific_result['message']}")
            else:
                print(f"跳过第 {entity_count} 个实体: {entity.dxftype()} (不支持的类型)")
        
        print(f"总共生成了 {len(training_samples)} 个训练样本")
        print(f"文件 {dxf_file_path} 中:")
        print(f"  应保留的实体数量 (should_delete=false): {retain_count}")
        print(f"  应删除的实体数量 (should_delete=true): {delete_count}")
        # 保存训练数据索引
        index_data = {
            "file": dxf_path,
            "generated_time": datetime.now().isoformat(),
            "total_samples": len(training_samples),
            "samples": training_samples
        }
        
        # 根据源目录决定保存到哪个输出目录
        if file_parent_dir == "student":
            index_file = os.path.join(student_dir, f"{base_name}.json")
        else:  # teacher
            index_file = os.path.join(teacher_dir, f"{base_name}.json")
            
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        print(f"保存索引文件: {index_file}")
        
        # 保存签名文件
        label_data = {
            "file": dxf_path,
            "generated_time": datetime.now().isoformat(),
            "signatures": signatures
        }
        
        # 签名文件保存在同一目录下
        label_file = os.path.join(
            student_dir if file_parent_dir == "student" else teacher_dir, 
            f"{base_name}_reference.json"
        )
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, ensure_ascii=False, indent=2)
        print(f"保存标签文件: {label_file}")
        
        return {
            "status": "success",
            "message": f"训练数据生成完成，共{len(training_samples)}个样本",
            "index_file": index_file,
            "label_file": label_file,
            "sample_count": len(training_samples)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"生成训练数据时出错: {str(e)}"
        }


def is_entity_in_signal_signatures(entity_signature, signal_signatures, tolerance=1e-3):
    """
    检查实体是否在信号签名中
    
    Args:
        entity_signature (dict): 实体签名
        signal_signatures (list): 信号签名列表
        tolerance (float): 匹配容差
        
    Returns:
        bool: 是否在信号签名中
    """
    print(f"检查实体签名是否匹配: {entity_signature.get('type')}")
    match_count = 0
    for signal_sig in signal_signatures:
        if entity_signature.get("type") != signal_sig.get("type"):
            continue
            
        obj_type = entity_signature.get("type")
        match_count += 1
        
        if obj_type == "LINE":
            if (abs(entity_signature.get("start", [0,0])[0] - signal_sig.get("start", [0,0])[0]) < tolerance and
                abs(entity_signature.get("start", [0,0])[1] - signal_sig.get("start", [0,0])[1]) < tolerance and
                abs(entity_signature.get("end", [0,0])[0] - signal_sig.get("end", [0,0])[0]) < tolerance and
                abs(entity_signature.get("end", [0,0])[1] - signal_sig.get("end", [0,0])[1]) < tolerance):
                print(f"找到匹配的LINE实体")
                return True
        elif obj_type == "CIRCLE":
            if (abs(entity_signature.get("center", [0,0])[0] - signal_sig.get("center", [0,0])[0]) < tolerance and
                abs(entity_signature.get("center", [0,0])[1] - signal_sig.get("center", [0,0])[1]) < tolerance and
                abs(entity_signature.get("radius", 0) - signal_sig.get("radius", 0)) < tolerance):
                print(f"找到匹配的CIRCLE实体")
                return True
        elif obj_type == "ARC":
            if (abs(entity_signature.get("center", [0,0])[0] - signal_sig.get("center", [0,0])[0]) < tolerance and
                abs(entity_signature.get("center", [0,0])[1] - signal_sig.get("center", [0,0])[1]) < tolerance and
                abs(entity_signature.get("radius", 0) - signal_sig.get("radius", 0)) < tolerance):
                print(f"找到匹配的ARC实体")
                return True
                
    print(f"未找到匹配的实体，检查了 {match_count} 个相同类型的实体")
    return False

def generate_training_data_batch_student_teacher(student_directory='dataset/student', teacher_directory='dataset/teacher'):
    """
    批量生成训练数据，只使用student目录中的文件生成训练集
    
    Args:
        student_directory: 学生DXF文件目录路径
        teacher_directory: 教师DXF文件目录路径
        
    Returns:
        dict: 处理结果
    """
    print("开始批量处理训练数据")
    # 规范化路径
    student_directory = os.path.abspath(student_directory)
    teacher_directory = os.path.abspath(teacher_directory)
    output_dir = "delete_annotion_output"
    picture_dir = os.path.join(output_dir, "pictures")
    if os.path.exists(picture_dir):
        import shutil
        shutil.rmtree(picture_dir)
    os.makedirs(picture_dir, exist_ok=True)
    if not os.path.exists(student_directory):
        return {
            "status": "error",
            "message": f"学生目录不存在: {student_directory}"
        }
        
    if not os.path.exists(teacher_directory):
        return {
            "status": "error",
            "message": f"教师目录不存在: {teacher_directory}"
        }
    
    if not os.path.isdir(student_directory):
        return {
            "status": "error",
            "message": f"学生路径不是目录: {student_directory}"
        }
        
    if not os.path.isdir(teacher_directory):
        return {
            "status": "error",
            "message": f"教师路径不是目录: {teacher_directory}"
        }
    
    # 获取所有DXF文件
    student_dxf_files = []
    teacher_dxf_files = []
    
    for file in os.listdir(student_directory):
        if file.lower().endswith('.dxf'):
            student_dxf_files.append(file)
            
    for file in os.listdir(teacher_directory):
        if file.lower().endswith('.dxf'):
            teacher_dxf_files.append(file)
    
    print(f"学生目录中的DXF文件: {student_dxf_files}")
    print(f"教师目录中的DXF文件: {teacher_dxf_files}")
    
    # 检查是否有共同的文件名
    common_files = set(student_dxf_files) & set(teacher_dxf_files)
    
    # 找出student中有但teacher中没有的文件
    student_only_files = set(student_dxf_files) - set(teacher_dxf_files)
    
    print(f"共同文件: {common_files}")
    print(f"仅在学生目录中的文件: {student_only_files}")
    
    results = []
    
    # 只为student目录中的文件生成训练数据，并使用teacher作为参考标准
    for dxf_file in student_dxf_files:  # 修改这里，遍历所有学生文件
        print(f"处理学生文件: {dxf_file}")
        student_file_path = os.path.join(student_directory, dxf_file)
        student_file_path = os.path.normpath(student_file_path)
        
        if not os.path.exists(student_file_path):
            results.append({
                "file": dxf_file,
                "status": "error",
                "message": f"学生文件不存在: {student_file_path}"
            })
            continue
            
        # 调用现有的generate_training_data函数处理student文件
        print(f"处理学生文件: {student_file_path}")
        result = generate_training_data(student_file_path)  # 这个函数内部已经会参照teacher文件
        result["file"] = dxf_file
        result["source"] = "student"
        # 检查是否为仅在student中存在的文件
        result["delete"] = dxf_file in student_only_files
        results.append(result)
        
    plt.close('all')
    success_count = sum(1 for r in results if r.get("status") == "success")
    
    # 创建包含图片路径的统一label文件
    combined_labels = []
    output_dir = "delete_annotion_output"
    pictures_dir = os.path.join(output_dir, "pictures")
    os.makedirs(pictures_dir, exist_ok=True)
    
    # 收集所有成功的处理结果中的图片信息
    for result in results:
        if result.get("status") == "success":
            # 读取生成的索引文件以获取图片路径信息
            index_file = result.get("index_file")
            if index_file and os.path.exists(index_file):
                try:
                    with open(index_file, 'r', encoding='utf-8') as f:
                        index_data = json.load(f)
                    
                    # 提取图片路径信息
                    for sample in index_data.get("samples", []):
                        combined_labels.append({
                            "file": result.get("file"),
                            "source": result.get("source"),
                            "sample_id": sample.get("sample_id"),
                            "image_path": sample.get("image_path"),
                            "entity_info": sample.get("entity_info"),
                            "label": sample.get("label")
                        })
                except Exception as e:
                    print(f"读取索引文件时出错 {index_file}: {e}")
    
    # 保存包含所有图片路径的综合label文件
    combined_label_file = os.path.join(output_dir, "combined_labels.json")
    combined_label_data = {
        "generated_time": datetime.now().isoformat(),
        "total_samples": len(combined_labels),
        "samples": combined_labels
    }
    
    with open(combined_label_file, 'w', encoding='utf-8') as f:
        json.dump(combined_label_data, f, ensure_ascii=False, indent=2)
    print(f"保存综合标签文件: {combined_label_file}")
    
    return {
        "status": "success",
        "message": f"批量处理完成: {success_count}/{len(results)} 个文件处理成功",
        "total_files": len(results),
        "success_count": success_count,
        "failed_count": len(results) - success_count,
        "results": results,
        "common_files": list(common_files),
        "student_only_files": list(student_only_files),
        "combined_label_file": combined_label_file
    }
def compare_directly_and_generate_labels(teacher_dir='dataset/teacher', student_dir='dataset/student'):
    """
    直接比较teacher和student目录中的文件并生成标签
    
    Args:
        teacher_dir: 教师DXF文件目录路径
        student_dir: 学生DXF文件目录路径
        
    Returns:
        dict: 处理结果
    """
    try:
        print("开始直接比较并生成标签")
        # 规范化路径
        teacher_dir = os.path.abspath(teacher_dir)
        student_dir = os.path.abspath(student_dir)
        
        # 检查目录是否存在
        if not os.path.exists(teacher_dir):
            return {"status": "error", "message": f"教师目录不存在: {teacher_dir}"}
            
        if not os.path.exists(student_dir):
            return {"status": "error", "message": f"学生目录不存在: {student_dir}"}
        
        # 获取所有DXF文件
        teacher_files = [f for f in os.listdir(teacher_dir) if f.lower().endswith('.dxf')]
        student_files = [f for f in os.listdir(student_dir) if f.lower().endswith('.dxf')]
        
        print(f"教师目录中的文件: {teacher_files}")
        print(f"学生目录中的文件: {student_files}")
        
        # 找出共同的文件
        common_files = set(teacher_files) & set(student_files)
        print(f"共同文件: {common_files}")
        
        # 创建输出目录
        output_dir = "delete_annotion_output"
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        results = []
        
        # 处理共同文件
        for filename in common_files:
            print(f"处理文件: {filename}")
            teacher_file = os.path.join(teacher_dir, filename)
            student_file = os.path.join(student_dir, filename)
            
            # 获取两个文件的签名
            teacher_signatures_result = get_dxf_objects(teacher_file)
            student_signatures_result = get_dxf_objects(student_file)
            
            if teacher_signatures_result["status"] != "success":
                results.append({
                    "file": filename,
                    "status": "error",
                    "message": f"无法读取教师文件: {teacher_signatures_result['message']}"
                })
                continue
                
            if student_signatures_result["status"] != "success":
                results.append({
                    "file": filename,
                    "status": "error",
                    "message": f"无法读取学生文件: {student_signatures_result['message']}"
                })
                continue
            
            # 比较签名
            teacher_sigs = teacher_signatures_result["signatures"]
            student_sigs = student_signatures_result["signatures"]
            
            print(f"教师文件签名数: {len(teacher_sigs)}")
            print(f"学生文件签名数: {len(student_sigs)}")
            
            # 找出student中应该删除的实体(在student中但不在teacher中的)
            student_to_delete = []
            for student_sig in student_sigs:
                if not is_entity_in_signal_signatures(student_sig, teacher_sigs):
                    student_to_delete.append({
                        "signature": student_sig,
                        "should_delete": True
                    })
            
            # 找出teacher中缺失的实体(在teacher中但不在student中的)
            missing_in_student = []
            for teacher_sig in teacher_sigs:
                if not is_entity_in_signal_signatures(teacher_sig, student_sigs):
                    missing_in_student.append({
                        "signature": teacher_sig,
                        "missing": True
                    })
            
            # 保存比较结果
            comparison_data = {
                "file": filename,
                "teacher_file": teacher_file,
                "student_file": student_file,
                "generated_time": datetime.now().isoformat(),
                "teacher_entities_count": len(teacher_sigs),
                "student_entities_count": len(student_sigs),
                "entities_to_delete_from_student": len(student_to_delete),
                "entities_missing_in_student": len(missing_in_student),
                "delete_list": student_to_delete,
                "missing_list": missing_in_student
            }
            
            # 保存标签文件
            label_file = os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}_comparison.json")
            with open(label_file, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, ensure_ascii=False, indent=2)
            print(f"保存比较结果: {label_file}")
            
            results.append({
                "file": filename,
                "status": "success",
                "label_file": label_file,
                "entities_to_delete": len(student_to_delete),
                "entities_missing": len(missing_in_student)
            })
            
            print(f"处理文件 {filename}:")
            print(f"  教师实体数: {len(teacher_sigs)}")
            print(f"  学生实体数: {len(student_sigs)}")
            print(f"  学生文件中需删除实体: {len(student_to_delete)}")
            print(f"  学生文件中缺失实体: {len(missing_in_student)}")
        
        success_count = sum(1 for r in results if r["status"] == "success")
        
        return {
            "status": "success",
            "message": f"比较完成: {success_count}/{len(results)} 个文件处理成功",
            "total_files": len(results),
            "success_count": success_count,
            "results": results
        }
        
    except Exception as e:
        return {"status": "error", "message": f"处理过程中出错: {str(e)}"}

# 主函数示例
if __name__ == '__main__':
    # 批量处理student和teacher目录中的文件进行比较
    result = generate_training_data_batch_student_teacher()
    print(result)
    
    # 或者直接比较并生成标签
    # comparison_result = compare_directly_and_generate_labels()
    # print(comparison_result)