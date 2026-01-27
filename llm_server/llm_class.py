import os
import json
from urllib.parse import urlparse
import base64
from io import BytesIO
import tempfile
import urllib3
from datetime import datetime
from PIL import Image
import requests

# 禁用 SSL 警告（仅开发环境）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

debug = False
VLM_DIR = f"/root/my_python_server/models/OpenBMB_MiniCPM-V-2_6-int4"

class VLMService:
    def __init__(self):
        # 使用 WSL 本地服务
        self.api_url = 'http://localhost:8001/v1'
        self.model_name = VLM_DIR
        self.api_key = 'sk-xxx'  # 本地服务不需要真实 API 密钥

        print(f"VLM服务初始化完成，模型: {self.model_name}")

    def create_with_image(self, messages, image_source=None, tools=None):
        """
        使用OpenAI兼容API发送带有图像的消息到VLM服务
        
        Args:
            messages: 消息列表，支持多轮对话
            image_source: 图像源（路径或URL），仅在第一轮对话中使用
            tools: 可用工具列表
            
        Returns:
            API响应结果
        """
        # 准备消息内容
        openai_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                # 纯文本消息
                openai_messages.append({
                    'role': role,
                    'content': content
                })
            elif isinstance(content, list):
                # 复合消息（可能包含图像）
                openai_messages.append({
                    'role': role,
                    'content': content
                })

        # 如果提供了图像源且是第一轮对话（没有图像消息），将其添加到第一个用户消息
        if image_source and not any(isinstance(msg['content'], list) and any('image_url' in item for item in msg['content']) for msg in openai_messages):
            if openai_messages and openai_messages[0]['role'] == 'user':
                # 读取图像并转换为base64
                with Image.open(image_source) as img:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                # 构建data URI
                image_data = f"data:image/png;base64,{img_str}"
                
                # 更新消息内容
                if isinstance(openai_messages[0]['content'], str):
                    # 将纯文本转换为复合内容
                    openai_messages[0]['content'] = [
                        {'type': 'text', 'text': openai_messages[0]['content']},
                        {'type': 'image_url', 'image_url': {'url': image_data}}
                    ]
                else:
                    # 添加图像到现有复合内容
                    openai_messages[0]['content'].append({
                        'type': 'image_url', 'image_url': {'url': image_data}
                    })

        # 构建请求体
        request_body = {
            "model": self.model_name,
            "messages": openai_messages,
            "tools": tools,
            "temperature": 0.9
        }

        # 构建请求URL
        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, port, path = parsed.hostname, parsed.port, parsed.path
        if not host:
            raise ValueError("API URL 无效，无法解析主机名")
        if not port:
            port = 80  # 默认端口

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 发送请求
        try:
            response = requests.post(
                f"http://{host}:{port}{path}",
                json=request_body,
                headers=headers,
                verify=False,
                timeout=60  # 设置60秒超时
            )
            
            # 检查响应是否成功
            if response.status_code == 200:
                result = response.json()
                print(f"[VLM调试] 成功收到响应")
                return result
            else:
                raise Exception(f"VLM服务器错误: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[VLM错误] 调用失败: {str(e)}")
            raise

    def create_with_multiple_images(self, messages, image_sources=None):
        """
        使用OpenAI兼容API发送多张图像到VLM服务
        
        Args:
            messages: 消息列表
            image_sources: 图像源列表（路径或URL）
            
        Returns:
            API响应结果
        """
        if not image_sources:
            return self.create_with_image(messages)
        
        # 准备消息内容
        openai_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if isinstance(content, str):
                # 纯文本消息
                openai_messages.append({
                    'role': role,
                    'content': content
                })
            elif isinstance(content, list):
                openai_messages.append({
                    'role': role,
                    'content': content
                })

        # 如果有第一个用户消息，添加所有图像
        if openai_messages and openai_messages[0]['role'] == 'user':
            # 确保content是列表格式
            if isinstance(openai_messages[0]['content'], str):
                # 将纯文本转换为复合内容
                base_content = [{'type': 'text', 'text': openai_messages[0]['content']}]
            else:
                # 保留现有复合内容
                base_content = openai_messages[0]['content']
            
            # 添加所有图像
            for image_source in image_sources:
                # 读取图像并转换为base64
                with Image.open(image_source) as img:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                # 构建data URI
                image_data = f"data:image/png;base64,{img_str}"
                
                # 添加图像到内容
                base_content.append({
                    'type': 'image_url', 'image_url': {'url': image_data}
                })
            
            # 更新消息内容
            openai_messages[0]['content'] = base_content

        # 调用create_with_image方法
        return self.create_with_image(openai_messages)

    def create_multimodal_conversation(self, conversation_history, current_query, image_source=None):
        """
        支持多轮对话的多模态对话
        
        Args:
            conversation_history: 历史对话列表
            current_query: 当前查询
            image_source: 图像源（仅在第一轮对话中使用）
            
        Returns:
            API响应结果
        """
        # 构建完整的消息列表
        messages = []
        
        # 添加历史对话
        for conv in conversation_history:
            messages.append({
                'role': conv['role'],
                'content': conv['content']
            })
        
        # 添加当前查询
        messages.append({
            'role': 'user',
            'content': current_query
        })
        
        # 调用create_with_image方法
        return self.create_with_image(messages, image_source=image_source)
