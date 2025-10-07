# 在 processordeemo1.py 基础上添加 Flask 服务
import io
import os
import pickle
import tarfile
import urllib
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
import torch
import torchvision.transforms.functional as TF

# Flask 应用实例
app = Flask(__name__)

# 加载DINOv3模型
MODEL_DINOV3_VITL = "dinov3_vitl16"
MODEL_NAME = "dinov3_vitl16_pretrain_lvd1689m"

# 加载预训练模型
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image

pretrained_model_name = os.path.abspath("E:\\code\\model\\dinov3-convnext-tiny-pretrain-lvd1689m")
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
model.cuda()

def load_image_from_path(path: str) -> Image:
    """支持本地和网络图片路径"""
    if path.startswith('http'):
        # 网络图片
        with urllib.request.urlopen(path) as f:
            return Image.open(f).convert("RGB")
    else:
        # 本地图片
        return Image.open(path).convert("RGB")

@app.route('/process_image', methods=['GET'])
def process_image():
    # 从请求参数中获取图片路径
    test_image_fpath = request.args.get('image_path')
    
    if not test_image_fpath:
        return jsonify({"error": "Missing image_path parameter"}), 400
    
    try:
        # 加载图片
        test_image = load_image_from_path(test_image_fpath)
        
        # 使用processor处理图像
        inputs = processor(images=test_image, return_tensors="pt")
        
        # 将输入数据移动到GPU
        inputs['pixel_values'] = inputs['pixel_values'].to('cuda')
        
        # 使用模型进行推理
        with torch.no_grad():
            outputs = model(inputs['pixel_values'])
            features = outputs.last_hidden_state
        
        # 获取处理后的图像用于可视化
        test_image_processed = inputs['pixel_values'][0]
        
        # 创建内存缓冲区保存图像
        img_buffer = io.BytesIO()
        
        # 创建可视化结果
        plt.figure(figsize=(9, 3), dpi=300)
        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow(test_image)  # 显示原图
        plt.title('original image')
        
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow(TF.to_pil_image(test_image_processed))  # 显示处理后的图像
        plt.title('processed image')
        
        plt.tight_layout()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        plt.close()  # 关闭图表以释放内存
        
        # 返回图像
        return send_file(img_buffer, mimetype='image/png')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/image_info', methods=['GET'])
def image_info():
    # 从请求参数中获取图片路径
    test_image_fpath = request.args.get('image_path')
    
    if not test_image_fpath:
        return jsonify({"error": "Missing image_path parameter"}), 400
    
    try:
        # 加载图片
        test_image = load_image_from_path(test_image_fpath)
        
        # 使用processor处理图像
        inputs = processor(images=test_image, return_tensors="pt")
        
        # 将输入数据移动到GPU
        inputs['pixel_values'] = inputs['pixel_values'].to('cuda')
        
        # 使用模型进行推理
        with torch.no_grad():
            outputs = model(inputs['pixel_values'])
            features = outputs.last_hidden_state
        
        # 返回模型输出信息
        return jsonify({
            "input_shape": list(inputs['pixel_values'].shape),
            "features_shape": list(features.shape)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5200, debug=True)