import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import requests
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from modelscope import AutoProcessor, AutoModelForZeroShotObjectDetection 

def visualize_results(image, results, text_labels):
    """
    可视化检测结果
    """
    draw = ImageDraw.Draw(image)
    
    # 从结果中获取检测框、标签和分数
    boxes = results[0]['boxes']
    labels = results[0]['text_labels'] if 'text_labels' in results[0] else results[0]['labels']
    scores = results[0]['scores']
    
    for i in range(len(boxes)):
        box = boxes[i].cpu().numpy()
        score = scores[i].item()
        label = labels[i]  # 现在是字符串，不需要 .item()
        
        # 只绘制置信度高的框
        if score > 0.4:
            x0, y0, x1, y1 = box
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            
            # 使用标签文本
            text_to_draw = f"{label} {score:.2f}"
            
            # 绘制标签
            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), text_to_draw, font)
            else:
                w, h = draw.textsize(text_to_draw, font)
                bbox = (x0, y0, x0 + w, y0 + h)
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), text_to_draw, fill="white", font=font)
    
    return image

def select_image_and_labels():
    """
    弹窗选择图片和获取标签输入
    """
    # 创建主窗口但不显示
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    try:
        # 选择图片文件
        image_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                ("All files", "*.*")
            ]
        )
        
        if not image_path:
            print("未选择图片，程序退出")
            return None, None
        
        # 获取标签输入
        labels = simpledialog.askstring(
            "输入标签",
            "请输入检测对象（用点分隔，如：a cat. a remote control.）：",
            initialvalue="a cat. a dog."
        )
        
        if not labels:
            print("未输入标签，程序退出")
            return None, None
        
        # 确保标签以点结尾
        if not labels.endswith('.'):
            labels += '.'
        
        return image_path, labels
        
    finally:
        root.destroy()

def main():
    # 弹窗选择图片和标签
    image_path, text = select_image_and_labels()
    
    if not image_path or not text:
        return
    
    # 加载模型
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # 加载图片
    image = Image.open(image_path)

    # 处理文本输入
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 使用正确的参数名
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs.input_ids,
        threshold=0.4,        # 使用 threshold 而不是 box_threshold
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )

    # 打印结果
    print("检测结果:")
    for i, result in enumerate(results):
        print(f"Result {i}:")
        print(f"  Boxes shape: {result['boxes'].shape}")
        print(f"  Labels: {result['labels']}")
        print(f"  Scores: {result['scores']}")
        print(f"  Text Labels: {result.get('text_labels', 'N/A')}")

    # 可视化结果
    result_image = visualize_results(image.copy(), results, text)
    
    # 保存结果
    output_path = "result.jpg"
    result_image.save(output_path)
    print(f"结果已保存到 {output_path}")
    
    # 显示结果图片
    result_image.show()

if __name__ == "__main__":
    main()