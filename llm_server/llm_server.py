import os
from dotenv import load_dotenv
import json
import http.client
from urllib.parse import urlparse
import base64
import requests
from io import BytesIO
from PIL import Image

class LLMService:
    def __init__(self):
        # 从环境变量中获取 DeepSeek 参数
        # 加载 .env 文件中的环境变量
        dotenv_path = r'E:\code\apikey\.env'
        load_dotenv(dotenv_path)
        self.api_url = os.getenv('LLM_OPENAI_API_URL')
        self.model_name = os.getenv('LLM_MODEL_NAME')
        self.api_key = os.getenv('LLM_OPENAI_API_KEY')
            # 检查必需的环境变量是否存在
        if not self.api_url:
            raise ValueError("环境变量 'deepseek_OPENAI_API_URL' 未设置或为空")
        if not self.model_name:
            raise ValueError("环境变量 'deepseek_MODEL_NAME' 未设置或为空")
        if not self.api_key:
            raise ValueError("环境变量 'deepseek_OPENAI_API_KEY' 未设置或为空")
        print(f"LLM服务初始化完成，模型: {self.model_name}")

    def create(self, messages, tools=None):
        print("开始调用LLM服务")
        
        # 解析 URL（去掉协议部分）
        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, path = parsed.hostname, parsed.path
        if not host:
            raise ValueError("API URL 无效，无法解析主机名")

        # 创建 HTTP 连接
        conn = http.client.HTTPSConnection(host)

        # 构造请求体
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.9  # 添加温度参数
        }

        # 发送 POST 请求
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
        print(f"LLM服务响应状态码: {response.status}")
        
        if response.status != 200:
            error_msg = response.read().decode('utf-8')
            raise Exception(f"LLM服务器错误: {response.status} - {error_msg}")

        # 读取响应内容
        response_data = response.read().decode('utf-8')
        data = json.loads(response_data)

        # 确保output目录存在
        os.makedirs('output', exist_ok=True)
        
        # 将响应保存到文件 (修复路径分隔符问题)
        output_file_path = os.path.join('output', 'formatted_data.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # 关闭连接
        conn.close()
         
        print("LLM服务调用完成")
        return data

class VLMService:
    def __init__(self):
        # 从环境变量中获取 VLM 参数
        dotenv_path = r'E:\code\my_python_server_private\.env'
        load_dotenv(dotenv_path)
        self.api_url = os.getenv('VLM_OPENAI_API_URL')
        self.model_name = os.getenv('VLM_MODEL_NAME')
        self.api_key = os.getenv('VLM_OPENAI_API_KEY')
        
        # 检查必需的环境变量是否存在
        if not self.api_url:
            raise ValueError("环境变量 'VLM_OPENAI_API_URL' 未设置或为空")
        if not self.model_name:
            raise ValueError("环境变量 'VLM_MODEL_NAME' 未设置或为空")
        if not self.api_key:
            raise ValueError("环境变量 'VLM_OPENAI_API_KEY' 未设置或为空")
        print(f"VLM服务初始化完成，模型: {self.model_name}")

    def encode_image(self, image_source):
        """
        编码图像为base64字符串
        支持URL和本地文件路径
        """
        try:
            if image_source.startswith(('http://', 'https://')):
                # 从URL获取图像
                response = requests.get(image_source)
                image_data = response.content
            else:
                # 从本地文件路径获取图像
                with open(image_source, "rb") as image_file:
                    image_data = image_file.read()
            
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise Exception(f"图像编码失败: {str(e)}")

    def create_with_image(self, messages, image_source=None, tools=None):
        print("开始调用VLM服务")
        
        # 如果提供了图像源，则处理图像
        if image_source:
            # 编码图像
            base64_image = self.encode_image(image_source)
            
            # 在第一条消息中添加图像
            if messages and len(messages) > 0:
                # 构建包含图像的消息内容
                image_content = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
                
                # 如果第一条消息的内容是字符串，转换为列表格式
                if isinstance(messages[0]["content"], str):
                    messages[0]["content"] = [
                        {
                            "type": "text",
                            "text": messages[0]["content"]
                        },
                        image_content
                    ]
                # 如果已经是列表格式，追加图像内容
                elif isinstance(messages[0]["content"], list):
                    messages[0]["content"].append(image_content)
        
        # 解析 URL（去掉协议部分）
        parsed = urlparse(f"{self.api_url}/chat/completions")
        host, path = parsed.hostname, parsed.path
        if not host:
            raise ValueError("API URL 无效，无法解析主机名")

        # 创建 HTTP 连接
        conn = http.client.HTTPSConnection(host)

        # 构造请求体
        request_body = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.7
        }

        # 发送 POST 请求
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
        print(f"VLM服务响应状态码: {response.status}")
        
        if response.status != 200:
            error_msg = response.read().decode('utf-8')
            raise Exception(f"VLM服务器错误: {response.status} - {error_msg}")

        # 读取响应内容
        response_data = response.read().decode('utf-8')
        data = json.loads(response_data)

        # 确保output目录存在
        os.makedirs('output', exist_ok=True)
        
        # 将响应保存到文件
        output_file_path = os.path.join('output', 'vlm_formatted_data.json')
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # 关闭连接
        conn.close()
        
        print("VLM服务调用完成")
        return data