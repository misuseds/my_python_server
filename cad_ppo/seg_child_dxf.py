import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# 使用原始字符串处理路径
img_path = r'E:\code\my_python_server\dxf_output\pictures\a.png'

# 检查文件是否存在
if not os.path.exists(img_path):
    print(f"错误：找不到文件 {img_path}")
else:
    # 读取图像
    img = cv2.imread(img_path)
    
    # 检查图像是否成功加载
    if img is None:
        print("错误：无法读取图像文件")
    else:
        # 改进的图像预处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用自适应阈值（推荐用于复杂图像）
        binary = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 2
        )
        
        # 或者使用Canny边缘检测
        # edges = cv2.Canny(gray, 50, 150)
        # binary = edges
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建结果图像
        result_img = img.copy()
        segmented_images = []
        
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            
            # 更严格的尺寸过滤
            min_size = 50  # 调整这个值来控制最小分割区域
            if w > min_size and h > min_size:
                # 在原图上绘制矩形框
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 提取子图
                subgraph = img[y:y+h, x:x+w]
                output_path = f'seg_child_output\\subgraph_{i}_{x}_{y}.png'
                cv2.imwrite(output_path, subgraph)
                segmented_images.append((subgraph, output_path))
        
        # 显示结果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(binary, cmap='gray')
        plt.title('Binary Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title('Detected Contours')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 打印分割结果
        print(f"成功分割出 {len(segmented_images)} 个子图")