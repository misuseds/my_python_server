# delete_entities_with_cnn_pyautocad.py
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from delete_cnn import ImprovedDXFEntityCNN
from pyautocad import Autocad, APoint
import matplotlib.pyplot as plt
import tempfile
import fitz  # PyMuPDF for PDF processing
import math

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

def entity_to_image_pdf_obb(entity, output_path, img_size=96):
    """
    使用PDF导出方式将实体转换为图像，通过用户选择两点定义截图区域
    """
    try:
        # 初始化AutoCAD
        acad = Autocad(create_if_not_exists=True)
        doc = acad.doc
        
        # 提示用户选择打印窗口的第一个角点
        doc.Utility.Prompt("请选择截图区域的第一个角点:\n")
        point1_raw = doc.Utility.GetPoint()
        point1 = acad.aDouble(point1_raw[0], point1_raw[1])

        # 提示用户选择打印窗口的第二个角点
        doc.Utility.Prompt("请选择截图区域的第二个角点:\n")
        point2_raw = doc.Utility.GetPoint()
        point2 = acad.aDouble(point2_raw[0], point2_raw[1])
        
        # 保存临时DWG文件
        temp_dwg = tempfile.NamedTemporaryFile(suffix='.dwg', delete=False)
        temp_dwg.close()
        temp_dwg_path = temp_dwg.name
        doc.SaveAs(temp_dwg_path)
        
        # 导出为PDF
        temp_pdf_path = temp_dwg_path.replace('.dwg', '.pdf')
        doc.ActiveLayout.ConfigName = "DWG To PDF.pc3"
        doc.ActiveLayout.CanonicalMediaName = "ISO_A3_(297.00_x_420.00_MM)"
        doc.ActiveLayout.PlotRotation = 0
        doc.ActiveLayout.CenterPlot = True
        
        # 使用用户选择的点设置打印窗口
        doc.ActiveLayout.SetWindowToPlot(point1, point2)
        doc.ActiveLayout.PlotType = 4  # Window
        
        # 导出PDF
        doc.Plot.PlotToFile(temp_pdf_path)
        
        # 将PDF转换为图像
        pdf_document = fitz.open(temp_pdf_path)
        page = pdf_document[0]
        
        # 提取页面并转换为图像
        mat = fitz.Matrix(3, 3)  # 缩放因子
        pix = page.get_pixmap(matrix=mat)
        pix.save(output_path)
        
        # 清理临时文件
        pdf_document.close()
        os.remove(temp_dwg_path)
        os.remove(temp_pdf_path)
        
        return True
    except Exception as e:
        print(f"转换实体为图像时出错: {e}")
        return False

def process_dxf_file_pdf_obb_current(model_path, confidence_threshold=0.5):
    """
    使用PDF方法处理当前AutoCAD文档，删除预测为"删除"的实体，使用用户选择范围截图
    
    Args:
        model_path: 模型文件路径
        confidence_threshold: 删除置信度阈值
    """
    # 初始化AutoCAD
    acad = Autocad(create_if_not_exists=True)
    doc = acad.doc
    
    # 获取当前文档路径
    input_dxf_path = doc.FullName
    if not input_dxf_path:
        raise Exception("当前没有打开任何DXF文件")
    
    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImprovedDXFEntityCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 创建临时目录存储实体图像
    temp_dir = 'temp_entity_images'
    os.makedirs(temp_dir, exist_ok=True)
    
    entities_to_delete = []
    
    # 获取所有实体
    try:
        entities = list(doc.ModelSpace)
        
        # 遍历所有实体
        for i, entity in enumerate(entities):
            # 为每个实体生成图像
            image_path = os.path.join(temp_dir, f'entity_{i}.png')
            
            # 将实体转换为图像（通过PDF方式，使用用户选择范围）
            if entity_to_image_pdf_obb(entity, image_path):
                # 使用模型预测
                try:
                    action, confidence = predict_entity_action(model, image_path, device)
                    
                    # 如果预测为删除且置信度超过阈值，则标记为删除
                    if action == 1 and confidence >= confidence_threshold:
                        entities_to_delete.append((entity, confidence))
                        print(f"标记删除实体 {entity.ObjectName} (置信度: {confidence:.2f})")
                except Exception as e:
                    print(f"预测实体 {entity.ObjectName} 时出错: {e}")
            else:
                print(f"无法为实体 {entity.ObjectName} 生成图像")
                
    except Exception as e:
        print(f"遍历实体时出错: {e}")
    
    # 删除标记的实体
    deleted_count = 0
    for entity, confidence in entities_to_delete:
        try:
            entity.Delete()
            deleted_count += 1
        except Exception as e:
            print(f"删除实体时出错: {e}")
    
    # 清理临时图像文件
    for filename in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, filename))
    os.rmdir(temp_dir)
    
    print(f"处理完成: 删除了 {deleted_count} 个实体")
    
    return deleted_count

def select_model_and_process_current():
    """
    通过GUI选择模型文件并处理当前AutoCAD中的DXF文件 (PDF+用户选择区域版本)
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
    
    try:
        # 设置默认置信度阈值
        confidence_threshold = 0.5
        
        # 处理当前AutoCAD文档
        deleted_count = process_dxf_file_pdf_obb_current(
            model_path, confidence_threshold
        )
        
        # 显示成功消息
        messagebox.showinfo(
            "处理完成",
            f"DXF文件处理完成!\n\n"
            f"删除了 {deleted_count} 个实体\n"
            f"更改已直接应用到当前AutoCAD文档中"
        )
        
    except Exception as e:
        messagebox.showerror("错误", f"处理过程中发生错误:\n{str(e)}")
        print(f"错误详情: {e}")

def main_pdf_obb():
    """
    主函数 - 启动GUI文件选择器 (PDF+用户选择区域版本)
    """
    try:
        select_model_and_process_current()
    except Exception as e:
        print(f"程序执行出错: {e}")
        messagebox.showerror("错误", f"程序执行出错:\n{str(e)}")

if __name__ == "__main__":
    main_pdf_obb()