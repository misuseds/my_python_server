# delete_entities_with_cnn_gui.py
import torch
import ezdxf
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import io
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from delete_cnn import ImprovedDXFEntityCNN

def preprocess_image(image_path):
    """
    预处理图像以供模型预测
    """
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_entity_action(model, image_path, device):
    """
    使用模型预测实体是否应该被删除
    """
    model.eval()
    with torch.no_grad():
        input_tensor = preprocess_image(image_path).to(device)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1)
        
        # 返回动作(1=删除, 0=保留)和置信度
        action = prediction.item()
        confidence = probabilities[0][action].item()
        
        return action, confidence

def entity_to_image(entity, output_path, img_size=96):
    """
    将DXF实体转换为图像
    """
    # 创建一个新的临时DXF文档用于渲染实体
    doc = ezdxf.new()
    msp = doc.modelspace()
    
    # 复制实体到新文档
    try:
        # 根据实体类型复制
        if entity.dxftype() == 'LINE':
            msp.add_line(entity.dxf.start, entity.dxf.end)
        elif entity.dxftype() == 'CIRCLE':
            msp.add_circle(entity.dxf.center, entity.dxf.radius)
        elif entity.dxftype() == 'ARC':
            msp.add_arc(entity.dxf.center, entity.dxf.radius, 
                       entity.dxf.start_angle, entity.dxf.end_angle)
        elif entity.dxftype() == 'LWPOLYLINE':
            msp.add_lwpolyline(entity.get_points())
        elif entity.dxftype() == 'POLYLINE':
            points = [vertex.dxf.location for vertex in entity.vertices]
            msp.add_polyline3d(points)
        elif entity.dxftype() == 'TEXT':
            msp.add_text(entity.dxf.text, dxfattribs={'insert': entity.dxf.insert})
        elif entity.dxftype() == 'MTEXT':
            msp.add_mtext(entity.text, dxfattribs={'insert': entity.dxf.insert})
        else:
            # 对于不支持的实体类型，创建一个简单的表示
            # 获取实体边界框并绘制矩形
            try:
                extents = entity.bbox()
                if extents:
                    min_point, max_point = extents
                    msp.add_lwpolyline([
                        min_point[:2],
                        (max_point[0], min_point[1]),
                        max_point[:2],
                        (min_point[0], max_point[1]),
                        min_point[:2]
                    ])
            except:
                # 如果无法获取边界框，创建默认表示
                msp.add_line((0, 0), (10, 10))
                msp.add_line((10, 0), (0, 10))
    except Exception as e:
        # 出错时创建默认图形
        msp.add_line((0, 0), (10, 10))
        msp.add_line((10, 0), (0, 10))
    
    # 设置视图以适应实体
    try:
        extents = msp.bbox()
        if extents:
            min_point, max_point = extents
            center = ((min_point[0] + max_point[0]) / 2, 
                     (min_point[1] + max_point[1]) / 2)
            width = max_point[0] - min_point[0]
            height = max_point[1] - min_point[1]
            size = max(width, height) * 1.1  # 添加边距
            
            # 创建图像
            doc.set_modelspace_vport(size, center)
    except:
        pass
    
    # 导出为图像
    try:
        # 使用matplotlib渲染（需要安装matplotlib）
        import matplotlib.pyplot as plt
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
        
        fig = plt.figure(figsize=(2, 2), dpi=img_size//2)
        ax = fig.add_axes([0, 0, 1, 1])
        ctx = RenderContext(doc)
        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp)
        ax.set_xlim(-50, 50)
        ax.set_ylim(-50, 50)
        ax.axis('off')
        fig.savefig(output_path, dpi=img_size//2, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        return True
    except ImportError:
        # 如果没有matplotlib，使用简单方法
        try:
            doc.saveas('temp.dxf')
            # 这里可以调用其他工具将DXF转换为图像
            # 暂时返回False表示无法生成图像
            return False
        except:
            return False
    except Exception:
        return False

def process_dxf_file(input_dxf_path, model_path, confidence_threshold=0.5):
    """
    处理DXF文件，删除预测为"删除"的实体
    
    Args:
        input_dxf_path: 输入DXF文件路径
        model_path: 模型文件路径
        confidence_threshold: 删除置信度阈值
    """
    # 生成输出文件路径（在相同目录下）
    base_name = os.path.splitext(input_dxf_path)[0]
    output_dxf_path = f"{base_name}_processed.dxf"
    
    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImprovedDXFEntityCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 加载DXF文件
    doc = ezdxf.readfile(input_dxf_path)
    msp = doc.modelspace()
    
    # 创建临时目录存储实体图像
    temp_dir = 'temp_entity_images'
    os.makedirs(temp_dir, exist_ok=True)
    
    entities_to_delete = []
    
    # 遍历所有实体
    for i, entity in enumerate(msp):
        # 为每个实体生成图像
        image_path = os.path.join(temp_dir, f'entity_{i}.png')
        
        # 将实体转换为图像
        if entity_to_image(entity, image_path):
            # 使用模型预测
            try:
                action, confidence = predict_entity_action(model, image_path, device)
                
                # 如果预测为删除且置信度超过阈值，则标记为删除
                if action == 1 and confidence >= confidence_threshold:
                    entities_to_delete.append((entity, confidence))
                    print(f"标记删除实体 {entity.dxftype()} (置信度: {confidence:.2f})")
            except Exception as e:
                print(f"预测实体 {entity.dxftype()} 时出错: {e}")
        else:
            print(f"无法为实体 {entity.dxftype()} 生成图像")
    
    # 删除标记的实体
    deleted_count = 0
    for entity, confidence in entities_to_delete:
        try:
            msp.delete_entity(entity)
            deleted_count += 1
        except Exception as e:
            print(f"删除实体时出错: {e}")
    
    # 保存修改后的DXF文件
    doc.saveas(output_dxf_path)
    
    # 清理临时图像文件
    for filename in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, filename))
    if os.path.exists(temp_dir):  # 确保目录存在再尝试删除
        os.rmdir(temp_dir)
    
    print(f"处理完成: 删除了 {deleted_count} 个实体")
    print(f"输出文件保存到: {output_dxf_path}")
    
    return output_dxf_path, deleted_count

def select_files_and_process():
    """
    通过GUI选择文件并处理
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 选择模型文件
    model_path = filedialog.askopenfilename(
        title="选择训练好的模型文件",
        filetypes=[("PyTorch Model Files", "*.pth"), ("All Files", "*.*")]
    )
    
    if not model_path:
        messagebox.showinfo("取消", "未选择模型文件")
        return
    
    # 选择DXF文件
    dxf_path = filedialog.askopenfilename(
        title="选择要处理的DXF文件",
        filetypes=[("DXF Files", "*.dxf"), ("All Files", "*.*")]
    )
    
    if not dxf_path:
        messagebox.showinfo("取消", "未选择DXF文件")
        return
    
    try:
        # 设置默认置信度阈值
        confidence_threshold = 0.5
        
        # 处理文件
        output_path, deleted_count = process_dxf_file(
            dxf_path, model_path, confidence_threshold
        )
        
        # 显示成功消息
        messagebox.showinfo(
            "处理完成",
            f"DXF文件处理完成!\n\n"
            f"删除了 {deleted_count} 个实体\n"
            f"输出文件: {output_path}"
        )
        
    except Exception as e:
        messagebox.showerror("错误", f"处理过程中发生错误:\n{str(e)}")
        print(f"错误详情: {e}")

def main():
    """
    主函数 - 启动GUI文件选择器
    """
    try:
        select_files_and_process()
    except Exception as e:
        print(f"程序执行出错: {e}")
        messagebox.showerror("错误", f"程序执行出错:\n{str(e)}")

if __name__ == "__main__":
    main()