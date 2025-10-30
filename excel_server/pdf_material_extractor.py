# pdf_material_extractor.py
import os
import json
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
import time
import logging
from dotenv import load_dotenv
import http.client
from urllib.parse import urlparse
import base64
from flask import Flask, request, jsonify, send_file
import threading
import webbrowser
import io
from tkinter import Tk
from tkinter.filedialog import askdirectory

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFToImageConverter:
    @staticmethod
    def convert_pdf_to_pil_images(pdf_path):
        """
        将PDF文件转换为PIL图像对象列表（不保存到磁盘）
        
        Args:
            pdf_path (str): PDF文件路径
            
        Returns:
            list: PIL图像对象列表
        """
        logger.info(f"正在转换PDF为内存图像: {pdf_path}")
        
        try:
            # 打开PDF文件
            pdf_document = fitz.open(pdf_path)
            pil_images = []
            
            # 遍历每一页
            for page_num in range(len(pdf_document)):
                try:
                    # 获取页面
                    page = pdf_document[page_num]
                    
                    # 渲染页面为图像
                    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    
                    # 直接转换为PIL图像而不保存到磁盘
                    img_data = pix.tobytes("ppm")
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    pil_images.append(pil_image)
                    
                    logger.info(f"已转换PDF第 {page_num + 1} 页为内存图像")
                except Exception as page_error:
                    logger.error(f"处理PDF第 {page_num + 1} 页时出错: {str(page_error)}")
                    continue  # 继续处理其他页面
            
            pdf_document.close()
            return pil_images
        except Exception as e:
            logger.error(f"打开PDF文件 {pdf_path} 时出错: {str(e)}")
            return []

class VLMProcessor:
    def __init__(self):
        """初始化VLM处理器"""
        dotenv_path = r'E:\code\apikey\.env'
        load_dotenv(dotenv_path)
        self.api_url = os.getenv('VLM_OPENAI_API_URL')
        self.model_name = os.getenv('VLM_MODEL_NAME')
        self.api_key = os.getenv('VLM_OPENAI_API_KEY')
        
        if not all([self.api_url, self.model_name, self.api_key]):
            raise ValueError("缺少VLM API配置，请检查.env文件")
    
    def encode_pil_image(self, pil_image):
        """
        将PIL图像编码为base64
        
        Args:
            pil_image (PIL.Image): PIL图像对象
            
        Returns:
            str: base64编码的图像
        """
        # 将PIL图像保存到内存中的字节流
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        # 编码为base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def extract_material_info(self, pil_image):
        """
        使用VLM提取材料信息
        
        Args:
            pil_image (PIL.Image): PIL图像对象
            
        Returns:
            dict: 材料信息
        """
        logger.info("正在提取材料信息")
        
        # 编码图像
        base64_image = self.encode_pil_image(pil_image)
        
        # 构造消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请从这张图纸中提取零件信息，包括零件名称、材料类型、数量以及每个零件的独立文件名。如果有多个零件，请分别列出。以JSON数组格式返回，例如：[{'零件': '零件1', '材料': 'Q235B', '数量': 1, '文件名': '文件1.dwg'}, {'零件': '零件2', '材料': 'Q345B', '数量': 2, '文件名': '文件2.dwg'}]。如果无法识别，请返回[{'零件': '未知', '材料': '未知', '数量': '未知', '文件名': '未知'}]"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        return self._send_request(messages)
    
    def _send_request(self, messages):
        """
        发送请求到VLM API
        
        Args:
            messages (list): 消息列表
            
        Returns:
            dict: API响应结果
        """
        # 解析URL
        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, path = parsed.hostname, parsed.path
        
        # 创建连接
        conn = http.client.HTTPSConnection(host)
        
        # 构造请求体
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7
        }
        
        # 发送请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        conn.request(
            "POST",
            path,
            body=json.dumps(request_body),
            headers=headers
        )
        
        # 获取响应
        response = conn.getresponse()
        
        if response.status != 200:
            error_msg = response.read().decode('utf-8')
            conn.close()
            raise Exception(f"VLM服务器错误: {response.status} - {error_msg}")
        
        # 解析响应
        response_data = response.read().decode('utf-8')
        data = json.loads(response_data)
        conn.close()
        
        # 提取结果
        try:
            content = data['choices'][0]['message']['content']
            # 处理可能的Markdown包装
            if content.startswith("```json"):
                content = content[7:]  # 移除 ```json
                if content.endswith("```"):
                    content = content[:-3]  # 移除 ```
            
            result = json.loads(content)
            logger.info(f"VLM处理结果: {result}")
            return result
        except Exception as e:
            logger.error(f"解析VLM响应失败: {e}")
            logger.error(f"原始响应: {content}")
            raise

class ExcelUpdater:
    @staticmethod
    def update_excel(excel_path, file_name, material_info_list):
        """
        更新Excel文件中的信息
        
        Args:
            excel_path (str): Excel文件路径
            file_name (str): PDF文件名(不含扩展名)
            material_info_list (list): 材料信息列表
        """
        logger.info(f"正在更新Excel: {excel_path}")
        
        # 确保 material_info_list 是一个列表
        if not isinstance(material_info_list, list):
            material_info_list = [material_info_list]
        
        # 读取Excel文件
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            logger.info(f"成功读取Excel文件，现有 {len(df)} 行数据")
            
            # 确保必要的列存在
            required_columns = ['PDF文件名', '零件', '零件文件名', '材料', '数量']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ""  # 添加缺失的列
        else:
            # 如果文件不存在，创建新的DataFrame，添加'零件文件名'列
            df = pd.DataFrame(columns=['PDF文件名', '零件', '零件文件名', '材料', '数量'])
            logger.info("Excel文件不存在，创建新的DataFrame")
        
        # 安全地删除现有的与此文件相关的记录
        if not df.empty and 'PDF文件名' in df.columns:
            df = df[df['PDF文件名'] != file_name]
        else:
            # 如果DataFrame为空或没有'PDF文件名'列，则从空开始
            df = pd.DataFrame(columns=['PDF文件名', '零件', '零件文件名', '材料', '数量'])
        
        # 为每个零件添加新行
        new_rows = []
        for material_info in material_info_list:
            part_name = material_info.get('零件', '未知')
            part_filename = material_info.get('文件名', '未知')  # 提取零件文件名
            material = material_info.get('材料', '未知')
            quantity = material_info.get('数量', '未知')
            
            new_rows.append({
                'PDF文件名': file_name,
                '零件': part_name,
                '零件文件名': part_filename,  # 添加零件文件名列
                '材料': material,
                '数量': quantity
            })
        
        # 添加新行到DataFrame
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)
        
        logger.info(f"添加了 {len(new_rows)} 个零件信息")
        
        # 保存Excel文件
        try:
            df.to_excel(excel_path, index=False)
            logger.info(f"已成功更新Excel文件，保存路径: {excel_path}")
        except Exception as e:
            logger.error(f"保存Excel文件失败: {str(e)}")
            raise
        logger.info("已更新Excel文件")

# 全局变量存储当前状态
current_state = {
    'pdf_files': [],
    'current_index': 0,
    'current_pdf': None,
    'current_image': None,
    'current_pil_image': None,
    'processing': False,
    'converter': None,
    'processor': None,
    'updater': None,
    'pdf_directory': '',
    'excel_path': '',
    'image_paths': {},
    'pil_images_cache': {},
    'processing_complete': False
}

app = Flask(__name__)

@app.route('/')
def index():
    """Web界面主页"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF材料信息提取工具</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            button { padding: 12px 24px; margin: 10px; font-size: 16px; border: none; border-radius: 4px; cursor: pointer; transition: background-color 0.3s; }
            .btn-primary { background-color: #17a2b8; color: white; }
            .btn-primary:hover { background-color: #138496; }
            .btn-start { background-color: #4CAF50; color: white; }
            .btn-start:hover { background-color: #45a049; }
            .btn-reset { background-color: #607D8B; color: white; }
            .btn-reset:hover { background-color: #455A64; }
            button:disabled { background-color: #cccccc; cursor: not-allowed; }
            #image-container { margin-top: 20px; text-align: center; min-height: 300px; display: flex; align-items: center; justify-content: center; }
            img { max-width: 100%; max-height: 70vh; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
            .status { margin: 20px 0; padding: 15px; background-color: #e3f2fd; border-radius: 4px; }
            .controls { margin: 20px 0; text-align: center; }
            h1 { color: #333; text-align: center; }
            .progress-bar { height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0; }
            .progress-fill { height: 100%; background-color: #4CAF50; transition: width 0.3s; }
            .material-result { 
                margin: 15px 0; 
                padding: 10px; 
                background-color: #e8f5e9; 
                border: 1px solid #c8e6c9; 
                border-radius: 4px; 
                font-weight: bold;
                font-size: 16px;
            }
            .path-input {
                width: 70%;
                padding: 8px;
                margin-right: 10px;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .browse-btn { padding: 8px 16px; background-color: #6c757d; color: white; border: none; border-radius: 4px; cursor: pointer; }
            .browse-btn:hover { background-color: #5a6268; }
            .processing-status {
                margin: 15px 0;
                padding: 10px;
                background-color: #e3f2fd;
                border-radius: 4px;
                text-align: center;
                font-weight: bold;
            }
            .part-item {
                margin: 8px 0;
                padding: 8px;
                background-color: #f1f8e9;
                border-left: 3px solid #4CAF50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PDF材料信息提取工具</h1>
            
            <div class="status">
                <div style="margin-bottom: 15px;">
                    <label for="pdf-directory">PDF目录:</label>
                    <input type="text" id="pdf-directory" class="path-input" placeholder="请选择PDF目录">
                    <button class="browse-btn" onclick="browseDirectory()">浏览</button>
                </div>
                
                <button class="btn-primary" onclick="setPaths()" id="set-paths-btn">设置路径并开始处理</button>
                <button class="btn-reset" onclick="resetProcessing()" id="reset-btn">重置</button>
                
                <hr style="margin: 20px 0;">
                
                <p>当前文件: <strong><span id="current-file">无</span></strong></p>
                <p>进度: <span id="progress">0/0</span></p>
                <div class="progress-bar">
                    <div class="progress-fill" id="progress-bar-fill" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="processing-status" id="processing-status" style="display: none;">
                <span id="processing-text">正在处理中...</span>
            </div>
            
            <div class="material-result" id="material-result" style="display: none;">
                <div>PDF文件名: <strong><span id="pdf-file-name"></span></strong></div>
                <div id="parts-container"></div>
            </div>
            
            <div id="image-container">
                <p id="no-image-msg">尚未开始处理</p>
                <img id="preview-image" src="" alt="PDF预览" style="display:none;">
            </div>
        </div>
        
        <script>
            function browseDirectory() {
                fetch('/browse_directory', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.directory) {
                            document.getElementById('pdf-directory').value = data.directory;
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                    });
            }
            
            function setPaths() {
                const pdfDirectory = document.getElementById('pdf-directory').value;
                
                if (!pdfDirectory) {
                    alert('请先选择PDF目录');
                    return;
                }
                
                fetch('/set_paths', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        pdf_directory: pdfDirectory
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert('错误: ' + data.error);
                    } else {
                        // 开始处理
                        startProcessing();
                    }
                })
                .catch(error => {
                    alert('设置路径时出错: ' + error.message);
                });
            }
            
            function startProcessing() {
                // 显示处理状态
                document.getElementById('processing-status').style.display = 'block';
                document.getElementById('set-paths-btn').disabled = true;
                
                fetch('/start_processing', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            alert('错误: ' + data.error);
                            document.getElementById('processing-status').style.display = 'none';
                            document.getElementById('set-paths-btn').disabled = false;
                            return;
                        }
                        
                        updateStatus(data);
                        if (data.current_file) {
                            loadImage();
                        } else {
                            // 处理完成
                            document.getElementById('processing-status').style.display = 'none';
                            document.getElementById('set-paths-btn').disabled = false;
                            document.getElementById('processing-text').textContent = '处理完成!';
                        }
                    })
                    .catch(error => {
                        alert('处理过程中出现错误: ' + error.message);
                        document.getElementById('processing-status').style.display = 'none';
                        document.getElementById('set-paths-btn').disabled = false;
                    });
            }
            
            function resetProcessing() {
                fetch('/reset_processing', {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        updateStatus(data);
                        hideImage();
                        hideMaterialInfo();
                        document.getElementById('processing-status').style.display = 'none';
                        document.getElementById('set-paths-btn').disabled = false;
                        
                        // 清空路径输入框
                        document.getElementById('pdf-directory').value = '';
                    });
            }
            
            function loadImage() {
                const timestamp = new Date().getTime(); // 防止缓存
                const img = document.getElementById('preview-image');
                const msg = document.getElementById('no-image-msg');
                if (img && msg) {
                    img.src = '/image/' + timestamp;
                    img.style.display = 'inline';
                    msg.style.display = 'none';
                }
                
                // 继续处理下一个文件
                setTimeout(() => {
                    fetch('/process_next', {method: 'POST'})
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                alert('错误: ' + data.error);
                                document.getElementById('processing-status').style.display = 'none';
                                document.getElementById('set-paths-btn').disabled = false;
                                return;
                            }
                            
                            updateStatus(data);
                            if (data.current_file) {
                                loadImage();
                            } else {
                                // 处理完成
                                document.getElementById('processing-status').style.display = 'none';
                                document.getElementById('set-paths-btn').disabled = false;
                                document.getElementById('processing-text').textContent = '处理完成!';
                            }
                        })
                        .catch(error => {
                            alert('处理过程中出现错误: ' + error.message);
                            document.getElementById('processing-status').style.display = 'none';
                            document.getElementById('set-paths-btn').disabled = false;
                        });
                }, 1000); // 1秒后处理下一个文件
            }
            
            function hideImage() {
                const img = document.getElementById('preview-image');
                const msg = document.getElementById('no-image-msg');
                if (img && msg) {
                    img.style.display = 'none';
                    msg.style.display = 'block';
                    msg.textContent = '处理已完成';
                }
            }
            
            function updateStatus(data) {
                const currentFileEl = document.getElementById('current-file');
                const progressEl = document.getElementById('progress');
                const progressBarFill = document.getElementById('progress-bar-fill');
                
                if (currentFileEl) {
                    currentFileEl.textContent = data.current_file || '无';
                }
                if (progressEl) {
                    progressEl.textContent = data.progress || '0/0';
                }
                
                if (progressBarFill && data.total > 0) {
                    const percentage = (data.processed / data.total) * 100;
                    progressBarFill.style.width = percentage + '%';
                } else if (progressBarFill) {
                    progressBarFill.style.width = '0%';
                }
                
                // 显示材料信息
                if (data.material_info) {
                    showMaterialInfo(data.material_info, data.file_name);
                } else {
                    hideMaterialInfo();
                }
            }
            
            function showMaterialInfo(materialInfo, fileName) {
                const resultEl = document.getElementById('material-result');
                const fileNameEl = document.getElementById('pdf-file-name');
                const partsContainer = document.getElementById('parts-container');
                
                if (resultEl && fileNameEl && partsContainer && (materialInfo || fileName)) {
                    // 显示PDF文件名
                    if (fileName) {
                        fileNameEl.textContent = fileName;
                    }
                    
                    // 清空零件容器
                    partsContainer.innerHTML = '';
                    
                    if (Array.isArray(materialInfo) && materialInfo.length > 0) {
                        // 处理多个零件的情况
                        materialInfo.forEach((item, index) => {
                            const partDiv = document.createElement('div');
                            partDiv.className = 'part-item';
                            
                            let partText = `零件${index+1}: `;
                            if (item.零件) partText += `${item.零件} `;
                            if (item.文件名) partText += `文件名: ${item.文件名} `;
                            if (item.材料) partText += `材料: ${item.材料} `;
                            if (item.数量 !== undefined) partText += `数量: ${item.数量}`;
                            
                            partDiv.textContent = partText;
                            partsContainer.appendChild(partDiv);
                        });
                    } else if (materialInfo) {
                        // 处理单个零件的情况
                        const partDiv = document.createElement('div');
                        partDiv.className = 'part-item';
                        
                        let partText = '零件1: ';
                        if (materialInfo.零件) partText += `${materialInfo.零件} `;
                        if (materialInfo.文件名) partText += `文件名: ${materialInfo.文件名} `;
                        if (materialInfo.材料) partText += `材料: ${materialInfo.材料} `;
                        if (materialInfo.数量 !== undefined) partText += `数量: ${materialInfo.数量}`;
                        
                        partDiv.textContent = partText;
                        partsContainer.appendChild(partDiv);
                    }
                    
                    resultEl.style.display = 'block';
                } else if (resultEl) {
                    resultEl.style.display = 'none';
                }
            }
            
            function hideMaterialInfo() {
                const resultEl = document.getElementById('material-result');
                if (resultEl) {
                    resultEl.style.display = 'none';
                }
            }
        </script>
    </body>
    </html>
    '''

# 添加一个新的路由来提供图片，使用时间戳作为参数
@app.route('/image/<timestamp>')
def serve_image(timestamp):
    """提供当前图像文件"""
    logger.info(f"请求图片服务: current_pdf={current_state['current_pdf']}")
    
    if current_state['current_pdf'] and current_state['current_pil_image']:
        # 将PIL图像对象转换为响应
        img_io = io.BytesIO()
        current_state['current_pil_image'].save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')
    
    logger.warning("没有当前图片可提供")
    return '', 404

@app.route('/browse_directory', methods=['POST'])
def browse_directory():
    """打开目录选择对话框"""
    try:
        # 隐藏Flask服务器窗口
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        directory = askdirectory(title="选择PDF文件所在目录")
        root.destroy()
        
        if directory:
            return jsonify({'directory': directory})
        else:
            return jsonify({'directory': ''})
    except Exception as e:
        logger.error(f"选择目录时出错: {str(e)}")
        return jsonify({'error': '选择目录时出错'})

# 修改 start_processing 函数中的这部分代码
@app.route('/set_paths', methods=['POST'])
def set_paths():
    """设置PDF目录和Excel文件路径"""
    global current_state
    
    data = request.get_json()
    pdf_directory = data.get('pdf_directory')
    
    if not pdf_directory:
        return jsonify({'error': '请提供PDF目录'})
    
    if not os.path.exists(pdf_directory):
        return jsonify({'error': '指定的PDF目录不存在'})
        
    # 设置固定的Excel输出路径
    excel_output_dir = os.path.join('excel_output', 'pdf_vlm_excel')
    os.makedirs(excel_output_dir, exist_ok=True)  # 自动创建目录（如果不存在）
    
    # 使用固定名称的Excel文件
    excel_path = os.path.join(excel_output_dir, 'pdf_material_info.xlsx')
    
    current_state['pdf_directory'] = pdf_directory
    current_state['excel_path'] = excel_path
    
    logger.info(f"设置路径完成 - PDF目录: {pdf_directory}")
    logger.info(f"Excel文件将保存至: {excel_path}")
    
    return jsonify({
        'message': '路径设置成功',
        'pdf_directory': pdf_directory,
        'excel_path': excel_path
    })

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """开始处理PDF文件"""
    global current_state
    
    # 检查是否已设置路径
    if not current_state['pdf_directory'] or not current_state['excel_path']:
        return jsonify({'error': '请先设置PDF目录和Excel文件路径'})
    
    logger.info(f"开始处理，PDF目录: {current_state['pdf_directory']}")
    logger.info(f"Excel文件路径: {current_state['excel_path']}")
    
    # 清空之前的缓存
    current_state['pil_images_cache'] = {}
    current_state['image_paths'] = {}
    
    # 初始化组件
    current_state['converter'] = PDFToImageConverter()
    current_state['processor'] = VLMProcessor()
    current_state['updater'] = ExcelUpdater()
    
    # 获取所有PDF文件
    if not os.path.exists(current_state['pdf_directory']):
        logger.error(f"PDF目录不存在: {current_state['pdf_directory']}")
        return jsonify({'error': f'PDF目录不存在: {current_state["pdf_directory"]}'})
        
    current_state['pdf_files'] = [f for f in os.listdir(current_state['pdf_directory']) if f.lower().endswith('.pdf')]
    current_state['current_index'] = 0
    
    logger.info(f"找到 {len(current_state['pdf_files'])} 个PDF文件")
    for i, pdf_file in enumerate(current_state['pdf_files']):
        logger.info(f"  {i+1}. {pdf_file}")
    
    if not current_state['pdf_files']:
        return jsonify({'error': '目录中没有找到PDF文件'})
    
    # 处理第一个文件
    return process_next_file()

@app.route('/process_next', methods=['POST'])
def process_next():
    """处理下一个文件"""
    global current_state
    
    # 移动到下一个文件
    current_state['current_index'] += 1
    return process_next_file()

@app.route('/reset_processing', methods=['POST'])
def reset_processing():
    """重置处理状态"""
    global current_state
    
    logger.info("重置处理状态")
    
    current_state['pdf_files'] = []
    current_state['current_index'] = 0
    current_state['current_pdf'] = None
    current_state['current_image'] = None
    current_state['current_pil_image'] = None
    current_state['processing'] = False
    current_state['image_paths'] = {}
    current_state['pil_images_cache'] = {}
    current_state['processing_complete'] = False
    
    return jsonify({
        'current_file': None,
        'progress': '0/0',
        'processed': 0,
        'total': 0
    })

def process_next_file():
    """处理下一个PDF文件"""
    global current_state
    
    while current_state['current_index'] < len(current_state['pdf_files']):
        # 获取当前PDF文件
        current_state['current_pdf'] = current_state['pdf_files'][current_state['current_index']]
        pdf_path = os.path.join(current_state['pdf_directory'], current_state['current_pdf'])
        
        logger.info(f"开始处理PDF文件 ({current_state['current_index']+1}/{len(current_state['pdf_files'])}): {pdf_path}")
        
        try:
            # 检查是否已经转换过该PDF
            pdf_name = Path(current_state['current_pdf']).stem
            if pdf_name in current_state['pil_images_cache']:
                # 如果已经转换过，直接使用已有的图像对象
                pil_images = current_state['pil_images_cache'][pdf_name]
                logger.info(f"使用已缓存的图像对象，共 {len(pil_images)} 页")
            else:
                # 转换PDF为图像对象
                pil_images = current_state['converter'].convert_pdf_to_pil_images(pdf_path)
                
                # 缓存图像对象
                current_state['pil_images_cache'][pdf_name] = pil_images
                logger.info(f"新生成图像对象并缓存，共 {len(pil_images)} 页")
            
            material_info_list = None
            
            # 使用第一页作为预览
            if pil_images and len(pil_images) > 0:
                current_state['current_pil_image'] = pil_images[0]
                logger.info("设置当前PIL图像对象用于预览")
                
                # 提取材料信息
                try:
                    material_result = current_state['processor'].extract_material_info(current_state['current_pil_image'])
                    logger.info(f"材料信息提取结果: {material_result}")
                    
                    # 确保结果是列表格式
                    if isinstance(material_result, dict):
                        material_info_list = [material_result]
                    elif isinstance(material_result, list):
                        material_info_list = material_result
                    else:
                        material_info_list = [{"零件": "未知", "材料": "提取失败", "数量": "提取失败", "文件名": "提取失败"}]
                        
                except Exception as e:
                    logger.error(f"材料信息提取失败: {str(e)}")
                    material_info_list = [{"零件": "未知", "材料": "提取失败", "数量": "提取失败", "文件名": "提取失败"}]
                
                # 更新Excel
                try:
                    current_state['updater'].update_excel(
                        current_state['excel_path'], 
                        pdf_name, 
                        material_info_list
                    )
                    logger.info("Excel更新完成")
                except Exception as e:
                    logger.error(f"更新Excel时出错: {str(e)}")
            else:
                current_state['current_pil_image'] = None
                logger.warning(f"未生成图像对象 for {current_state['current_pdf']}")
            
            # 返回完整的零件列表用于前端显示
            return jsonify({
                'current_file': current_state['current_pdf'],
                'progress': f'{current_state["current_index"] + 1}/{len(current_state["pdf_files"])}',
                'processed': current_state["current_index"] + 1,
                'total': len(current_state["pdf_files"]),
                'material_info': material_info_list,  # 返回完整列表
                'file_name': pdf_name
            })
        except Exception as e:
            logger.error(f"处理文件 {current_state['current_pdf']} 时出错: {str(e)}")
            # 即使出错也继续处理下一个文件
            current_state['current_index'] += 1
    
    # 已经处理完所有文件
    logger.info("所有文件处理完成")
    current_state['current_pdf'] = None
    current_state['current_image'] = None
    current_state['current_pil_image'] = None
    current_state['processing_complete'] = True
    return jsonify({
        'current_file': None,
        'progress': f'{len(current_state["pdf_files"])}/{len(current_state["pdf_files"])}',
        'processed': len(current_state["pdf_files"]),
        'total': len(current_state["pdf_files"]),
        'message': '所有文件处理完成'
    })

def run_web_interface():
    """运行Web界面"""
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)

def open_browser():
    """打开浏览器"""
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    print("启动PDF材料信息提取工具...")
    
    # 启动Web服务线程
    web_thread = threading.Thread(target=run_web_interface)
    web_thread.daemon = True
    web_thread.start()
    
    # 等待一秒确保服务器启动后再打开浏览器
    time.sleep(1)
    open_browser()
    
    print("Web界面已在 http://127.0.0.1:5000 启动")
    print("请在浏览器中查看界面")
    print("按 Ctrl+C 退出程序")
    
    # 主线程继续运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序已退出")