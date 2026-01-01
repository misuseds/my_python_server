# e:\code\my_python_server\yolo\detect_like_favorite.py
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def detect_like_favorite(image_path=None, screenshot=None):
    """
    检测点赞和收藏按钮
    可以传入图像路径或直接传入截图对象
    """
    # 获取项目根目录
    current_dir = Path(__file__).parent.parent
    model_path = current_dir / "models" / "like_favorite.pt"
    
    if not model_path.exists():
        return f"错误: 模型文件不存在 - {model_path}"
    
    try:
        # 加载模型
        model = YOLO(str(model_path))
        
        # 如果没有提供图像路径，使用传入的截图对象
        if image_path:
            image = cv2.imread(image_path)
        elif screenshot is not None:
            # 将PIL图像转换为OpenCV格式
            image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        else:
            return "错误: 必须提供图像路径或截图对象"
        
        if image is None:
            return "错误: 无法读取图像"
        
        # 进行预测
        results = model.predict(source=image, conf=0.5, save=False)
        
        detections = []
        result = results[0]
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # 边界框坐标 [x1, y1, x2, y2]
            confs = result.boxes.conf.cpu().numpy()  # 置信度
            cls_ids = result.boxes.cls.cpu().numpy()  # 类别ID
            
            names = result.names if hasattr(result, 'names') else {}
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                conf = confs[i]
                cls_id = int(cls_ids[i])
                class_name = names.get(cls_id, f"Class_{cls_id}")
                
                # 计算中心坐标
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detections.append({
                    'class_name': class_name,
                    'center': [center_x, center_y],
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        
        if not detections:
            return "未检测到点赞或收藏按钮"
        
        result_str = f"检测到 {len(detections)} 个元素:\n"
        for det in detections:
            result_str += f"- {det['class_name']}: 位置({det['center'][0]:.1f}, {det['center'][1]:.1f}), 置信度{det['confidence']:.2f}\n"
        
        return result_str.strip()
        
    except Exception as e:
        return f"检测过程中出错: {str(e)}"

if __name__ == "__main__":
    # 测试函数
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = detect_like_favorite(image_path=image_path)
        print(result)
    else:
        print("请提供图像路径作为参数")