import os
import cv2
import json
from typing import List, Tuple, Dict
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import shutil
# 添加用于弹窗选择文件夹的库
import tkinter as tk
from tkinter import filedialog

class YOLODatasetCreator:
    def __init__(self, image_folder: str, output_dir: str):
        """
        初始化YOLO数据集创建器
        
        Args:
        
            image_folder: 包含图片的文件夹路径
            output_dir: 输出目录路径
        """
        self.image_folder = image_folder
        self.output_dir = output_dir
        self.current_image = None
        self.current_image_path = None
        self.annotations = []
        self.current_bboxes = []
        self.drawing = False
        self.start_point = (0, 0)
        self.end_point = (0, 0)
        self.current_class_id = 0
        
    def get_image_files(self) -> List[str]:
        """获取文件夹中所有图片文件"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for file in os.listdir(self.image_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(self.image_folder, file))
        
        return sorted(image_files)
    
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        """鼠标回调函数，用于绘制边界框"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # 确保坐标顺序正确
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1, x2)
            bottom = max(y1, y2)
            
            # 添加边界框到当前图片的标注
            height, width = self.current_image.shape[:2]
            bbox = {
                'class_id': self.current_class_id,
                'bbox': [left, top, right, bottom],  # 原始坐标
                'bbox_normalized': [  # YOLO格式 (x_center, y_center, width, height)
                    (left + right) / 2.0 / width,  # x_center
                    (top + bottom) / 2.0 / height,  # y_center
                    (right - left) / width,  # width
                    (bottom - top) / height   # height
                ]
            }
            self.current_bboxes.append(bbox)
            
    def draw_bboxes(self, img: np.ndarray) -> np.ndarray:
        """在图片上绘制当前的边界框"""
        img_copy = img.copy()
        
        # 绘制临时正在绘制的框
        if self.drawing:
            cv2.rectangle(img_copy, self.start_point, self.end_point, (0, 255, 0), 2)
        
        # 绘制已确认的框
        for bbox in self.current_bboxes:
            x1, y1, x2, y2 = bbox['bbox']
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            # 显示类别ID
            cv2.putText(img_copy, f"Class: {bbox['class_id']}", 
                       (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return img_copy
    
    def run(self):
        """运行标注工具"""
        image_files = self.get_image_files()
        if not image_files:
            print(f"在文件夹 {self.image_folder} 中未找到图片文件")
            return
        
        cv2.namedWindow('YOLO Dataset Creator')
        cv2.setMouseCallback('YOLO Dataset Creator', self.mouse_callback)
        
        current_idx = 0
        
        while current_idx < len(image_files):
            self.current_image_path = image_files[current_idx]
            self.current_image = cv2.imread(self.current_image_path)
            
            # 检查图像是否成功加载
            if self.current_image is None:
                print(f"无法加载图像: {self.current_image_path}")
                current_idx += 1
                continue
            
            self.current_bboxes = []
            
            print(f"正在标注: {os.path.basename(self.current_image_path)} ({current_idx + 1}/{len(image_files)})")
            print("操作说明:")
            print("- 鼠标拖拽绘制边界框")
            print("- 按 'n' 进入下一张图片")
            print("- 按 'p' 返回上一张图片")
            print("- 按 'c' 切换类别ID (当前: {})".format(self.current_class_id))
            print("- 按 'd' 删除最后一个标注框")
            print("- 按 's' 保存当前进度")
            print("- 按 'q' 退出程序")
            
            while True:
                display_img = self.draw_bboxes(self.current_image)
                
                # 显示当前类别ID
                cv2.putText(display_img, f"Current Class: {self.current_class_id}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_img, "Press 'h' for help", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                cv2.imshow('YOLO Dataset Creator', display_img)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('n'):  # 下一张
                    break
                elif key == ord('p'):  # 上一张
                    if current_idx > 0:
                        current_idx -= 2  # -1是为了抵消循环末尾的+1
                        if current_idx < -1:
                            current_idx = -1
                    break
                elif key == ord('c'):  # 切换类别
                    self.current_class_id = (self.current_class_id + 1) % 10  # 支持0-9类
                    print(f"类别ID已切换为: {self.current_class_id}")
                elif key == ord('d'):  # 删除最后一个框
                    if self.current_bboxes:
                        self.current_bboxes.pop()
                        print("删除了最后一个边界框")
                elif key == ord('s'):  # 保存当前进度
                    self.save_annotations()
                    print("当前进度已保存")
                elif key == ord('h'):  # 显示帮助
                    print("\n快捷键说明:")
                    print("n - 下一张图片")
                    print("p - 上一张图片")
                    print("c - 切换类别ID")
                    print("d - 删除最后一个标注框")
                    print("s - 保存当前进度")
                    print("q - 退出程序")
                    print("h - 显示帮助\n")
                elif key == ord('q'):  # 退出
                    cv2.destroyAllWindows()
                    self.save_annotations()
                    print("程序已退出，数据已保存")
                    return
            
            # 保存当前图片的标注
            if self.current_bboxes:
                annotation = {
                    'image_path': self.current_image_path,
                    'image_width': self.current_image.shape[1],  # 图像宽度
                    'image_height': self.current_image.shape[0],  # 图像高度
                    'annotations': self.current_bboxes
                }
                self.annotations.append(annotation)
            
            current_idx += 1
        
        cv2.destroyAllWindows()
        self.save_annotations()
        print(f"标注完成！数据已保存到 {self.output_dir}")
    
    def save_annotations(self):
        """保存标注数据到JSON文件"""
        data = {
            'dataset_info': {
                'total_images': len(self.annotations),
                'created_at': str(cv2.getTickCount() / cv2.getTickFrequency()),
                'image_folder': self.image_folder,  # 添加图片文件夹路径信息
                'image_width': self.current_image.shape[1] if self.current_image is not None else 0,  # 当前图像宽度
                'image_height': self.current_image.shape[0] if self.current_image is not None else 0  # 当前图像高度
            },
            'annotations': self.annotations
        }
        
        json_path = os.path.join(self.output_dir, "annotations.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def convert_to_yolo_format_and_split(json_file: str, output_dir: str, class_names: List[str] = None):
    """
    将JSON格式的标注转换为YOLO格式的txt文件，并自动划分训练、验证、测试集
    
    Args:
        json_file: 输入的JSON文件路径
        output_dir: 输出的YOLO格式文件目录
        class_names: 类别名称列表
    """
    # 设置默认类别名
    if class_names is None:
        class_names = [f"class_{i}" for i in range(10)]
    
    # 创建目录结构
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 创建训练、验证、测试子目录
    for subdir in ["train", "val", "test"]:
        os.makedirs(os.path.join(images_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, subdir), exist_ok=True)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 收集所有图片路径和对应的标注
    all_image_paths = []
    all_annotation_paths = []
    
    for annotation in data['annotations']:
        image_path = annotation['image_path']
        bboxes = annotation['annotations']
        
        # 复制图片到相应目录
        image_filename = os.path.basename(image_path)
        image_name, image_ext = os.path.splitext(image_filename)
        
        # 为每张图片生成YOLO格式的txt文件
        txt_path = os.path.join(labels_dir, "train", f"{image_name}.txt")
        
        with open(txt_path, 'w') as f:
            for bbox in bboxes:
                class_id = bbox['class_id']
                # YOLO格式: class_id x_center y_center width height (全部归一化)
                x_center, y_center, width, height = bbox['bbox_normalized']
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # 复制图片到images/train目录
        dst_image_path = os.path.join(images_dir, "train", image_filename)
        shutil.copy2(image_path, dst_image_path)
        
        all_image_paths.append(dst_image_path)
        all_annotation_paths.append(txt_path)
    
    # 检查数据集大小，如果样本数太少则不进行划分
    num_samples = len(all_image_paths)
    if num_samples < 3:
        print(f"数据集样本数为 {num_samples}，不足以进行标准数据集划分，所有数据将放在训练集")
        
        # 确定各类别数量
        all_class_ids = set()
        for annotation in data['annotations']:
            for bbox in annotation['annotations']:
                all_class_ids.add(bbox['class_id'])
        num_classes = max(all_class_ids) + 1 if all_class_ids else 10
        
        # 生成data.yaml配置文件
        data_yaml = {
            'path': output_dir,
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': num_classes,
            'names': class_names[:num_classes]
        }
        
        yaml_path = os.path.join(output_dir, "data.yaml")
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        
        print(f"YOLO格式文件已保存到 {output_dir}")
        print(f"由于样本数量不足，所有 {num_samples} 张图片都放在训练集中")
        print(f"配置文件已生成: {yaml_path}")
        return
    
    # 划分数据集（70%训练，20%验证，10%测试）
    train_img, temp_img, train_lbl, temp_lbl = train_test_split(
        all_image_paths, all_annotation_paths, test_size=0.3, random_state=42
    )
    val_img, test_img, val_lbl, test_lbl = train_test_split(
        temp_img, temp_lbl, test_size=1/3, random_state=42
    )
    
    # 移动文件到对应的子目录
    def move_files_to_subdir(file_paths, subdir):
        """将文件移动到指定子目录"""
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            dst_path = os.path.join(os.path.dirname(os.path.dirname(file_path)), subdir, filename)
            
            # 如果是图片文件，需要重新复制
            if os.path.splitext(filename)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                src_img_path = os.path.join(output_dir, "images", "train", filename)
                dst_img_path = os.path.join(output_dir, "images", subdir, filename)
                if os.path.exists(src_img_path):
                    shutil.move(src_img_path, dst_img_path)
            else:  # 标注文件
                dst_lbl_path = os.path.join(output_dir, "labels", subdir, filename)
                if os.path.exists(file_path):
                    shutil.move(file_path, dst_lbl_path)
    
    move_files_to_subdir(val_img, "val")
    move_files_to_subdir(test_img, "test")
    
    # 生成data.yaml配置文件
    num_classes = max([int(class_name.split('_')[-1]) if '_' in class_name else i for i, class_name in enumerate(class_names)]) + 1
    if num_classes < 10:
        num_classes = 10  # 假设最多支持10个类别
    
    data_yaml = {
        'path': output_dir,  # 数据集根目录
        'train': 'images/train',  # 训练图像目录
        'val': 'images/val',  # 验证图像目录
        'test': 'images/test',  # 测试图像目录（可选）
        'nc': num_classes,  # 类别数量
        'names': class_names[:num_classes]  # 类别名称列表
    }
    
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
    
    print(f"YOLO格式文件已保存到 {output_dir}")
    print(f"数据集划分完成：")
    print(f"- 训练集: {len(train_img)} 张图片")
    print(f"- 验证集: {len(val_img)} 张图片")
    print(f"- 测试集: {len(test_img)} 张图片")
    print(f"配置文件已生成: {yaml_path}")

    
def select_folder_dialog():
    """弹窗选择文件夹"""
    # 创建一个隐藏的tkinter窗口
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 显示文件夹选择对话框
    folder_path = filedialog.askdirectory(title="选择图片文件夹")
    
    # 销毁tkinter窗口
    root.destroy()
    
    return folder_path

# 使用示例
if __name__ == "__main__":
    # 弹窗选择图片文件夹
    image_folder = select_folder_dialog()
    
    # 如果用户取消选择，则退出
    if not image_folder:
        print("未选择文件夹，程序退出")
        exit()
    
    # 设置输出目录
    output_dir = os.path.join(image_folder, "yolo_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建数据集标注工具
    creator = YOLODatasetCreator(image_folder, output_dir)
    
    # 运行标注工具
    creator.run()
    
    # 生成YOLO格式的数据集和配置文件
    json_path = os.path.join(output_dir, "annotations.json")
    if os.path.exists(json_path):
        # 可以自定义类别名称
        class_names = ["person", "bicycle", "car", "motorcycle", "airplane", 
                      "bus", "train", "truck", "boat", "traffic light"]
        convert_to_yolo_format_and_split(json_path, output_dir, class_names)
    else:
        print("未找到标注文件，跳过YOLO格式转换")