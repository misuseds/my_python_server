import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class ImprovedDXFEntityDataset(Dataset):
    """
    改进的DXF实体数据集类，支持数据平衡和增强
    """
    def __init__(self, json_file, root_dir, transform=None, balance_data=True):
        """
        Args:
            json_file (string): 包含样本信息的json文件路径
            root_dir (string): 图像文件的根目录
            transform (callable, optional): 可选的图像变换
            balance_data (bool): 是否平衡数据集
        """
        with open(json_file, 'r') as f:
            self.data_info = json.load(f)
        
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self.data_info.get('samples', [])
        
        # 分析数据分布
        self.delete_count = sum(1 for s in self.samples if s['label']['action'] == 1)
        self.retain_count = len(self.samples) - self.delete_count
        
        print(f"数据分布 - 删除: {self.delete_count}, 保留: {self.retain_count}")
        
        # 如果需要平衡数据
        if balance_data and self.delete_count != self.retain_count:
            self._balance_dataset()
    
    def _balance_dataset(self):
        """
        平衡数据集，使删除和保留的样本数量相近
        """
        delete_samples = [s for s in self.samples if s['label']['action'] == 1]
        retain_samples = [s for s in self.samples if s['label']['action'] == 0]
        
        # 过采样少数类，使用更积极的上采样策略
        if len(delete_samples) < len(retain_samples):
            # 上采样删除样本
            factor = len(retain_samples) // len(delete_samples) + 1
            delete_samples = (delete_samples * factor)[:len(retain_samples)]
        elif len(retain_samples) < len(delete_samples):
            # 上采样保留样本
            factor = len(delete_samples) // len(retain_samples) + 1
            retain_samples = (retain_samples * factor)[:len(delete_samples)]
        
        self.samples = delete_samples + retain_samples
        print(f"平衡后数据分布 - 删除: {len(delete_samples)}, 保留: {len(retain_samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.samples[idx]
        img_path = sample['image_path']
        
        # 检查图像是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图像文件不存在 {img_path}")
            # 返回默认图像或跳过
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 无法加载图像 {img_path}: {e}")
            # 创建一个默认图像
            image = Image.new('RGB', (96, 96), color='black')
        
        # 获取标签 (1表示删除，0表示保留)
        label = sample['label']['action']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

class ImprovedDXFEntityCNN(nn.Module):
    """
    改进的用于判断DXF实体是否应该删除的CNN网络
    """
    def __init__(self, num_classes=2):
        super(ImprovedDXFEntityCNN, self).__init__()
        
        # 改进的卷积层，增加批归一化
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # 第三个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        # 改进的分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 12 * 12, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 实现Focal Loss来处理类别不平衡
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train_improved_cnn_model(json_file, image_dir, model_save_path='improved_dxf_cnn_model.pth', 
                            num_epochs=50, learning_rate=0.001, batch_size=32):
    """
    改进的CNN模型训练函数
    
    Args:
        json_file: 包含训练数据信息的JSON文件路径
        image_dir: 图像文件所在的目录
        model_save_path: 模型保存路径
        num_epochs: 训练轮数
        learning_rate: 学习率
        batch_size: 批处理大小
    """
    
    # 改进的数据预处理和增强，加强少数类的增强
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),  # 增大旋转角度
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # 添加平移变换
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    full_dataset = ImprovedDXFEntityDataset(json_file=json_file, root_dir=image_dir, 
                                          transform=train_transform, balance_data=True)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # 为验证集应用不同的变换
    val_dataset.dataset.transform = val_transform
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 检查是否有GPU可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建模型、损失函数和优化器
    model = ImprovedDXFEntityCNN(num_classes=2).to(device)
    
    # 使用Focal Loss处理类别不平衡
    criterion = FocalLoss(alpha=0.25, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # 训练模型
    best_val_loss = float('inf')
    best_f1_score = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计信息
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = 0  # True positives
        val_fp = 0  # False positives
        val_fn = 0  # False negatives
        
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # 计算F1分数所需统计量
                val_tp += ((predicted == 1) & (labels == 1)).sum().item()
                val_fp += ((predicted == 1) & (labels == 0)).sum().item()
                val_fn += ((predicted == 0) & (labels == 1)).sum().item()
        
        # 计算F1分数
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, F1 Score: {f1_score:.4f}')
        
        # 保存最佳模型（基于F1分数）
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            torch.save(model.state_dict(), model_save_path)
            print(f'  Best model saved with validation F1 score: {best_f1_score:.4f}')
    
    print(f'Training completed. Best validation F1 score: {best_f1_score:.4f}')
    return model

def evaluate_detailed_model(model, json_file, image_dir):
    """
    详细评估模型性能，包括混淆矩阵和分类报告
    """
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImprovedDXFEntityDataset(json_file=json_file, root_dir=image_dir, 
                                     transform=transform, balance_data=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算详细指标
    accuracy = 100 * sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    print(f'Overall Accuracy: {accuracy:.2f}%')
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # 分类报告
    target_names = ['Retain', 'Delete']
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    
    return accuracy

# 在 delete_cnn.py 文件末尾添加以下代码
from flask import Flask, request, jsonify
import io
import base64
from PIL import Image as PILImage
import torchvision.transforms as transforms
import urllib.parse

app = Flask(__name__)

# 全局变量存储模型和配置
model = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_model(model_path):
    """
    初始化模型
    """
    global model
    model = ImprovedDXFEntityCNN(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

@app.route('/train', methods=['GET'])
def train_model_api():
    """
    训练模型的API接口 (GET)
    ---
    参数:
    - json_file: 训练数据JSON文件路径
    - image_dir: 图片目录路径
    - model_save_path: 模型保存路径
    - num_epochs: 训练轮数（可选，默认50）
    - learning_rate: 学习率（可选，默认0.001）
    - batch_size: 批处理大小（可选，默认32）
    """
    try:
        # 从查询参数获取值
        json_file = request.args.get('json_file', 'output/combined_labels.json')
        image_dir = request.args.get('image_dir', 'output/pictures')
        model_save_path = request.args.get('model_save_path', 'models/improved_dxf_entity_cnn_model.pth')
        num_epochs = int(request.args.get('num_epochs', 50))
        learning_rate = float(request.args.get('learning_rate', 0.001))
        batch_size = int(request.args.get('batch_size', 32))
        
        # 创建模型保存目录
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # 调用训练函数
        trained_model = train_improved_cnn_model(
            json_file=json_file,
            image_dir=image_dir,
            model_save_path=model_save_path,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        
        # 更新全局模型
        global model
        model = trained_model
        
        return jsonify({
            'status': 'success',
            'message': 'Model training completed successfully'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/evaluate', methods=['GET'])
def evaluate_model_api():
    """
    评估模型的API接口 (GET)
    ---
    参数:
    - json_file: 评估数据JSON文件路径
    - image_dir: 图片目录路径
    """
    try:
        # 从查询参数获取值
        json_file = request.args.get('json_file', 'output/combined_labels.json')
        image_dir = request.args.get('image_dir', 'output/pictures')
        
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized. Please train or load a model first.'
            }), 400
            
        # 执行评估
        accuracy = evaluate_detailed_model(model, json_file, image_dir)
        
        return jsonify({
            'status': 'success',
            'accuracy': accuracy
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict', methods=['GET'])
def predict_single_image():
    """
    对单张图片进行预测的API接口 (GET)
    ---
    参数:
    - image_path: 待预测图片路径
    - format: 返回格式 ("json" 或 "html")，默认为 "json"
    """
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized. Please train or load a model first.'
            }), 400
        
        # 从查询参数获取图片路径和格式
        image_path = request.args.get('image_path')
        format_type = request.args.get('format', 'json')  # 默认返回JSON格式
        
        if not image_path:
            return jsonify({
                'status': 'error',
                'message': 'image_path parameter is required'
            }), 400
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载并预处理图像
        image = PILImage.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # 执行预测
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1)
            
        # 解析结果
        action = "delete" if prediction.item() == 1 else "retain"
        confidence = probabilities[0][prediction.item()].item()
        
        result = {
            'status': 'success',
            'prediction': {
                'action': action,
                'confidence': confidence,
                'probabilities': {
                    'retain': probabilities[0][0].item(),
                    'delete': probabilities[0][1].item()
                }
            },
            'image_path': image_path
        }
        
        # 根据format参数决定返回JSON还是HTML
        if format_type.lower() == 'html':
            # 读取图片并转换为base64以便在HTML中显示
            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>DXF Entity Prediction Result</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 800px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .result-card {{
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 15px;
                        margin: 15px 0;
                        background-color: #f9f9f9;
                    }}
                    .prediction-image {{
                        max-width: 100%;
                        height: auto;
                        border: 1px solid #ccc;
                        border-radius: 4px;
                    }}
                    .confidence-high {{
                        color: #28a745;
                        font-weight: bold;
                    }}
                    .confidence-medium {{
                        color: #ffc107;
                        font-weight: bold;
                    }}
                    .confidence-low {{
                        color: #dc3545;
                        font-weight: bold;
                    }}
                    .action-delete {{
                        color: #dc3545;
                        font-weight: bold;
                    }}
                    .action-retain {{
                        color: #28a745;
                        font-weight: bold;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>DXF Entity Prediction Result</h1>
                    
                    <div class="result-card">
                        <h2>Prediction Result</h2>
                        <p><strong>Image Path:</strong> {image_path}</p>
                        <p><strong>Action:</strong> 
                            <span class="action-{'delete' if action == 'delete' else 'retain'}">{action.upper()}</span>
                        </p>
                        <p><strong>Confidence:</strong> 
                            <span class="{'confidence-high' if confidence > 0.9 else 'confidence-medium' if confidence > 0.7 else 'confidence-low'}">
                                {confidence:.2%}
                            </span>
                        </p>
                        <p><strong>Probabilities:</strong></p>
                        <ul>
                            <li>Retain: {(probabilities[0][0].item()):.2%}</li>
                            <li>Delete: {(probabilities[0][1].item()):.2%}</li>
                        </ul>
                    </div>
                    
                    <div class="result-card">
                        <h2>Entity Image</h2>
                        <img src="data:image/png;base64,{img_data}" alt="DXF Entity" class="prediction-image">
                    </div>
                    
                    <div class="result-card">
                        <h2>Interpretation</h2>
                        <p>The AI model predicts this DXF entity should be <strong>{action.upper()}</strong> with <strong>{confidence:.2%}</strong> confidence.</p>
                        {'''
                        <p style="color:#28a745;"><strong>Recommendation:</strong> The entity is likely safe to delete as indicated by high confidence.</p>
                        ''' if action == 'delete' and confidence > 0.9 else ''}
                        {'''
                        <p style="color:#ffc107;"><strong>Recommendation:</strong> Consider manual review before deleting as confidence is moderate.</p>
                        ''' if confidence <= 0.9 and confidence > 0.7 else ''}
                        {'''
                        <p style="color:#dc3545;"><strong>Recommendation:</strong> Low confidence - strongly recommend manual verification before taking action.</p>
                        ''' if confidence <= 0.7 else ''}
                    </div>
                </div>
            </body>
            </html>
            """
            return html_template
        
        # 默认返回JSON格式
        return jsonify(result)
    
    except Exception as e:
        if request.args.get('format', 'json').lower() == 'html':
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Error - DXF Entity Prediction</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 800px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    }}
                    .error {{
                        color: #dc3545;
                        padding: 15px;
                        border: 1px solid #dc3545;
                        border-radius: 5px;
                        background-color: #f8d7da;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Error - DXF Entity Prediction</h1>
                    <div class="error">
                        <h2>Error occurred:</h2>
                        <p>{str(e)}</p>
                    </div>
                </div>
            </body>
            </html>
            """, 500
        else:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500

@app.route('/predict_batch', methods=['GET'])
def predict_batch_images():
    """
    批量预测图片的API接口 (GET)
    ---
    参数:
    - json_file: 包含图片路径的JSON文件
    - image_dir: 图片目录路径（可选）
    """
    try:
        if model is None:
            return jsonify({
                'status': 'error',
                'message': 'Model not initialized. Please train or load a model first.'
            }), 400
        
        # 从查询参数获取值
        json_file = request.args.get('json_file')
        image_dir = request.args.get('image_dir', '')
        
        if not json_file:
            return jsonify({
                'status': 'error',
                'message': 'json_file parameter is required'
            }), 400
        
        # 加载JSON文件
        with open(json_file, 'r') as f:
            samples_data = json.load(f)
        
        samples = samples_data.get('samples', [])
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        results = []
        for sample in samples:
            try:
                image_path = sample.get('image_path')
                if image_dir and not os.path.isabs(image_path):
                    image_path = os.path.join(image_dir, image_path)
                
                # 加载并预处理图像
                image = PILImage.open(image_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                # 执行预测
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    prediction = torch.argmax(output, dim=1)
                
                # 解析结果
                action = "delete" if prediction.item() == 1 else "retain"
                confidence = probabilities[0][prediction.item()].item()
                
                results.append({
                    'image_path': image_path,
                    'prediction': {
                        'action': action,
                        'confidence': confidence
                    },
                    'ground_truth': sample.get('label', {})
                })
                
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/load_model', methods=['GET'])
def load_model_api():
    """
    加载已训练模型的API接口 (GET)
    ---
    参数:
    - model_path: 模型文件路径
    """
    try:
        # 从查询参数获取模型路径
        model_path = request.args.get('model_path')
        
        if not model_path or not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'message': 'Model file not found'
            }), 400
        
        init_model(model_path)
        
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# 将文件末尾的 __main__ 部分替换为以下代码
if __name__ == "__main__":
    # 默认模型路径
    default_model_path = "models/improved_dxf_entity_cnn_model.pth"
    
    # 检查命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='Run DXF Entity CNN API')
    parser.add_argument('--model-path', type=str, help='Path to the trained model')
    args = parser.parse_args()
    
    # 尝试加载模型（优先使用命令行参数，否则使用默认路径）
    model_path = args.model_path if args.model_path else default_model_path
    
    # 如果模型文件存在，则加载
    if os.path.exists(model_path):
        init_model(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file not found at {model_path}, starting server without preloaded model")
        print("You can train a model using the /train endpoint")
    
    app.run(host='0.0.0.0', port=5000, debug=True)