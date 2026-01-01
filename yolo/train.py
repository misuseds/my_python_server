import os
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

def select_dataset_folder():
    """
    弹窗选择数据集文件夹
    
    Returns:
        str: 选择的文件夹路径，如果取消选择则返回None
    """
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    folder_path = filedialog.askdirectory(title="选择数据集文件夹")
    root.destroy()
    return folder_path

def train_yolo_with_single_image(data_yaml_path, model_type='yolov8n.pt'):
    """
    训练YOLO模型（虽然只有一张图片，但仍演示训练流程）
    
    Args:
        data_yaml_path: 数据配置文件路径
        model_type: YOLO模型类型
    """
    # 加载预训练模型
    model = YOLO(model_type)
    
    # 开始训练
    results = model.train(
        data=data_yaml_path,      # 数据配置文件
        epochs=100,               # 训练轮数
        imgsz=640,                # 图像尺寸
        batch=16,                 # 批次大小
        save_period=10,           # 每10轮保存一次
        save=True,                # 保存模型
        project='runs/train',     # 保存项目目录
        name='single_image_exp',  # 实验名称
        exist_ok=True,            # 允许覆盖
        pretrained=True           # 使用预训练权重
    )
    
    return model, results

def augment_single_image_dataset(image_path, label_path, output_dir, augmentation_factor=10):
    """
    通过数据增强扩展单张图片数据集
    
    Args:
        image_path: 原始图片路径
        label_path: 原始标签路径
        output_dir: 输出目录
        augmentation_factor: 扩增倍数
    """
    # 读取原始图片和标签
    original_img = cv2.imread(image_path)
    with open(label_path, 'r') as f:
        original_labels = f.readlines()
    
    # 创建输出目录
    aug_img_dir = os.path.join(output_dir, 'images', 'train')
    aug_lbl_dir = os.path.join(output_dir, 'labels', 'train')
    os.makedirs(aug_img_dir, exist_ok=True)
    os.makedirs(aug_lbl_dir, exist_ok=True)
    
    # 复制原始图片
    original_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(aug_img_dir, f"{original_name}_0.jpg"), original_img)
    with open(os.path.join(aug_lbl_dir, f"{original_name}_0.txt"), 'w') as f:
        f.writelines(original_labels)
    
    # 应用各种数据增强
    import albumentations as A
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.GaussNoise(p=0.3)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    for i in range(1, augmentation_factor):
        # 解析原始标签
        bboxes = []
        class_labels = []
        
        for line in original_labels:
            parts = line.strip().split()
            if len(parts) == 5:  # class_id x_center y_center width height
                class_id, x_center, y_center, width, height = map(float, parts)
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(int(class_id))
        
        # 应用增强
        transformed = transform(
            image=original_img,
            bboxes=bboxes,
            class_labels=class_labels
        )
        
        # 保存增强后的图片
        aug_img = transformed['image']
        aug_bboxes = transformed['bboxes']
        
        aug_img_path = os.path.join(aug_img_dir, f"{original_name}_aug_{i}.jpg")
        cv2.imwrite(aug_img_path, aug_img)
        
        # 保存增强后的标签
        aug_lbl_path = os.path.join(aug_lbl_dir, f"{original_name}_aug_{i}.txt")
        with open(aug_lbl_path, 'w') as f:
            for j, bbox in enumerate(aug_bboxes):
                class_id = class_labels[j]
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def prepare_data_yaml_for_training(output_dir, class_names):
    """
    准备训练所需的数据配置文件
    """
    data_yaml = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    return yaml_path

# 使用示例
if __name__ == "__main__":
    # 弹窗选择数据集文件夹
    dataset_dir = select_dataset_folder()
    
    if not dataset_dir:
        print("未选择数据集文件夹，程序退出。")
        exit()
    
    print(f"选择的数据集文件夹: {dataset_dir}")
    
    # 假设你已经有了单张图片的数据集
    class_names = ["person", "car", "dog"]  # 替换为你的类别名称
    
    # 扩充数据集
    images_dir = os.path.join(dataset_dir, "images", "train")
    labels_dir = os.path.join(dataset_dir, "labels", "train")
    
    # 检查目录是否存在
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"数据集目录结构不正确。请确保存在 {images_dir} 和 {labels_dir} 目录。")
        exit()
    
    # 找到第一个图片和标签文件
    img_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        print("在图片目录中未找到图片文件。")
        exit()
    
    first_img = os.path.join(images_dir, img_files[0])
    first_lbl = os.path.join(labels_dir, os.path.splitext(img_files[0])[0] + '.txt')
    
    if not os.path.exists(first_lbl):
        print(f"未找到对应的标签文件: {first_lbl}")
        exit()
    
    print(f"使用图片: {first_img}")
    print(f"使用标签: {first_lbl}")
    
    # 扩增数据集
    augment_single_image_dataset(
        first_img, first_lbl, 
        dataset_dir, 
        augmentation_factor=50  # 扩增50倍
    )
    
    # 准备数据配置文件
    data_yaml_path = prepare_data_yaml_for_training(dataset_dir, class_names)
    
    # 开始训练
    model, results = train_yolo_with_single_image(data_yaml_path)
    
    print("训练完成！")
    print("注意：由于数据量极小，模型性能可能很差，仅用于演示目的。")