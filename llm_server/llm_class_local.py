import os
import json
import time
from datetime import datetime
from PIL import Image
import numpy as np
import torch


# 导入必要的库
from huggingface_hub import snapshot_download
from modelscope import AutoModelForCausalLM, AutoTokenizer, Qwen3VLForConditionalGeneration, AutoProcessor

# 启用调试模式
debug = True

class LLMServiceLocal:
    def __init__(self):
        """
        初始化本地LLM服务
        使用modelscope加载模型
        """
        # 模型配置
        self.llm_model_name = "Qwen/Qwen2.5-0.5B"
        self.vlm_model_name = "OpenBMB/MiniCPM-V-2_6-int4"  # 使用 int4 量化版本，只需要约 7GB GPU 内存

        # 初始化模型
        self.llm_model = None
        self.llm_tokenizer = None
        self.vlm_model = None
        self.vlm_tokenizer = None

        # 只加载LLM模型，VLM模型在需要时才加载
        self._load_llm_model()

        print(f"本地LLM服务初始化完成，模型: {self.llm_model_name}")
        print(f"使用modelscope加载模型")
        print(f"VLM模型将在需要时动态加载")
    
    def _load_models(self):
        """
        加载本地模型
        如果模型不存在，使用ModelScope下载
        """
        try:
            # 下载并加载LLM模型
            print(f"正在加载LLM模型: {self.llm_model_name}")
            self._load_llm_model()
            
            # 下载并加载VLM模型
            print(f"正在加载VLM模型: {self.vlm_model_name}")
            self._load_vlm_model()
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
    
    def _load_llm_model(self):
        """
        使用modelscope加载LLM模型
        """
        try:
            print(f"使用modelscope加载LLM模型: {self.llm_model_name}")

            # 加载模型和分词器
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name, trust_remote_code=True)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )

            print(f"LLM模型加载成功(modelscope): {self.llm_model_name}")

        except Exception as e:
            print(f"加载LLM模型失败: {e}")
            raise
    
    def _load_vlm_model(self):
        """
        加载VLM模型
        """
        try:
            # 使用ModelScope加载模型
            try:
                print("正在使用内存优化参数加载VLM模型...")
                # 从modelscope导入相关的类
                from modelscope import AutoModel, AutoTokenizer
                
                self.vlm_model = AutoModel.from_pretrained(
                    self.vlm_model_name,
                    trust_remote_code=True
                )
                self.vlm_tokenizer = AutoTokenizer.from_pretrained(
                    self.vlm_model_name,
                    trust_remote_code=True
                )
                # 设置模型为评估模式
                self.vlm_model.eval()
                print(f"VLM模型加载成功: {self.vlm_model_name}")
            except Exception as e:
                print(f"VLM模型加载失败: {e}")
                raise
            
        except Exception as e:
            print(f"加载VLM模型失败: {e}")
            # VLM模型加载失败不影响LLM模型的使用
            print("VLM模型加载失败，继续使用LLM模型")
    
    def clear_memory(self):
        """
        释放模型内存
        """
        try:
            if hasattr(self, 'llm_model') and self.llm_model:
                del self.llm_model
                self.llm_model = None
                self.llm_tokenizer = None
                print("已释放LLM模型内存")
            
            if hasattr(self, 'vlm_model') and self.vlm_model:
                del self.vlm_model
                self.vlm_model = None
                self.vlm_tokenizer = None
                print("已释放VLM模型内存")
            
            # 清理GPU内存
            import torch
            torch.cuda.empty_cache()
            print("已清理GPU缓存")
            
        except Exception as e:
            print(f"释放内存失败: {e}")
    
    def create(self, messages, tools=None):
        """
        非流式调用：返回完整响应
        """
        print("开始调用本地LLM服务（非流式）")
        
        # 构建提示词
        prompt = self._build_prompt(messages)
        
        # ===== 立即保存发送给LLM的messages到文件 =====
        os.makedirs('output', exist_ok=True)
        request_file_path = os.path.join('output', 'llm_request.json')
        with open(request_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'request': {
                    'model': self.llm_model_name,
                    'messages': messages,
                    'temperature': 0.9
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"LLM请求已保存到: {request_file_path}")
        # ================================================
        
        print("正在调用本地LLM服务...")
        
        try:
            # 检查并加载LLM模型
            if not self.llm_model or not self.llm_tokenizer:
                print("LLM模型未加载，正在重新加载...")
                self._load_llm_model()
            
            # 生成响应
            start_time = time.time()
            response = self._generate_response(prompt)
            end_time = time.time()
            print(f"本地LLM服务调用完成，耗时: {(end_time - start_time):.2f}秒")
            
            # 构建响应格式
            response_data = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response
                        }
                    }
                ]
            }
            
            # ===== 保存LLM响应到文件 =====
            output_file_path = os.path.join('output', 'llm_response.json')
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'response': response_data
                }, f, indent=2, ensure_ascii=False)
            print(f"LLM响应已保存到: {output_file_path}")
            # ================================================
            
            return response_data
            
        except Exception as e:
            print(f"调用本地LLM服务失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    def create_stream(self, messages, tools=None, callback=None):
        """
        流式调用：生成器，每次 yield 一个解析后的 chunk 字典
        兼容旧版 callback（可选）
        """
        print("开始调用本地LLM服务（流式）")
        
        # 构建提示词
        prompt = self._build_prompt(messages)
        
        # ===== 立即保存发送给LLM的流式messages到文件 =====
        os.makedirs('output', exist_ok=True)
        request_file_path = os.path.join('output', 'llm_stream_request.json')
        with open(request_file_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'request': {
                    'model': self.llm_model_name,
                    'messages': messages,
                    'temperature': 0.9,
                    'stream': True
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"LLM流式请求已保存到: {request_file_path}")
        # ================================================
        
        print("正在调用本地LLM服务（流式）...")
        
        try:
            # 生成响应（模拟流式）
            start_time = time.time()
            response = self._generate_response(prompt)
            end_time = time.time()
            print(f"本地LLM服务流式调用完成，耗时: {(end_time - start_time):.2f}秒")
            
            # 模拟流式响应
            chunks = response.split(' ')
            for i, chunk in enumerate(chunks):
                chunk_text = chunk + (' ' if i < len(chunks) - 1 else '')
                
                # 构建chunk响应
                chunk_response = {
                    "choices": [
                        {
                            "delta": {
                                "content": chunk_text
                            }
                        }
                    ]
                }
                
                if callback and isinstance(chunk_text, str) and chunk_text:
                    callback(chunk_text)
                
                yield chunk_response
                time.sleep(0.05)  # 模拟流式输出延迟
            
        except Exception as e:
            print(f"调用本地LLM服务失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    def _build_prompt(self, messages):
        """
        构建提示词
        """
        prompt = ""
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt += f"系统: {content}\n\n"
            elif role == 'user':
                prompt += f"用户: {content}\n\n"
            elif role == 'assistant':
                prompt += f"助手: {content}\n\n"
        
        prompt += "助手: "
        return prompt
    
    def _generate_response(self, prompt):
        """
        使用modelscope生成响应
        """
        if not self.llm_model or not self.llm_tokenizer:
            raise Exception("LLM模型未加载")

        try:
            print("使用modelscope生成响应")

            # 编码输入
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.llm_model.device)

            # 生成响应
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.1,
                do_sample=True
            )

            # 解码输出
            response = self.llm_tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            print(f"生成响应失败: {e}")
            import traceback
            print(traceback.format_exc())
            raise
    
    def analyze_images(self, images, prompt):
        """
        分析图像
        """
        if not self.vlm_model or not self.vlm_processor:
            raise Exception("VLM模型未加载")
        
        try:
            print(f"正在分析 {len(images)} 张图像")
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 处理图像
            for i, image_path in enumerate(images):
                try:
                    # 读取图像
                    image = Image.open(image_path).convert('RGB')
                    
                    # 添加图像到消息
                    messages[0]["content"].append({"type": "image", "image": image})
                    
                except Exception as e:
                    print(f"处理图像 {image_path} 失败: {e}")
                    continue
            
            # 编码输入
            inputs = self.vlm_processor.apply_chat_template(
                vlm_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.vlm_model.device)
            
            # 生成响应
            start_time = time.time()
            generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=128, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.vlm_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            end_time = time.time()
            print(f"VLM分析完成，耗时: {(end_time - start_time):.2f}秒")
            
            return response
            
        except Exception as e:
            print(f"分析图像失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            raise
    
    def create_with_image(self, prompt, image_path):
        """
        单个图像分析
        """
        try:
            print(f"正在分析单个图像: {image_path}")
            
            # 构建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"请分析以下图像:\n\n{prompt}"}
                    ]
                }
            ]
            
            # 处理图像
            try:
                # 读取图像
                image = Image.open(image_path).convert('RGB')
                
                # 添加图像到消息
                messages[0]["content"].append({"type": "image", "image": image})
                
            except Exception as e:
                print(f"处理图像 {image_path} 失败: {e}")
                return {"error": str(e)}
            
            # 编码输入
            inputs = self.vlm_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.vlm_model.device)
            
            # 生成响应
            start_time = time.time()
            generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=128, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.vlm_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            end_time = time.time()
            print(f"VLM分析完成，耗时: {(end_time - start_time):.2f}秒")
            
            # 构建响应格式
            response_data = {
                "text": response
            }
            
            # 释放内存
            self.clear_memory()
            
            return response_data
            
        except Exception as e:
            print(f"分析图像失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def create_with_multiple_images(self, messages, image_sources=None):
        """
        多图像分析
        """
        try:
            if not image_sources:
                print("没有图像源")
                return {"error": "没有图像源"}
                
            print(f"正在分析多个图像: {len(image_sources)}张")
            
            # 加载VLM模型（仅在需要时）
            if not self.vlm_model or not self.vlm_tokenizer:
                print("正在加载VLM模型...")
                self._load_vlm_model()
            
            # 构建消息
            if not messages:
                # 如果没有消息，创建默认消息
                prompt = "请分析以下图像"
            else:
                # 提取提示文本
                if isinstance(messages, list) and len(messages) > 0:
                    if isinstance(messages[0], dict) and "content" in messages[0]:
                        if isinstance(messages[0]["content"], str):
                            prompt = messages[0]["content"]
                        else:
                            prompt = str(messages[0]["content"])
                    else:
                        prompt = str(messages)
                else:
                    prompt = str(messages)
            
            # 处理图像（调整大小以减少内存使用）
            max_size = 512  # 最大图像尺寸
            images = []
            for i, image_path in enumerate(image_sources):
                try:
                    # 读取图像
                    image = Image.open(image_path).convert('RGB')
                    
                    # 调整图像大小
                    image.thumbnail((max_size, max_size), Image.LANCZOS)
                    print(f"已调整图像 {image_path} 大小: {image.size}")
                    
                    # 添加图像到列表
                    images.append(image)
                    
                except Exception as e:
                    print(f"处理图像 {image_path} 失败: {e}")
                    continue
            
            # 构建消息格式（适配 MiniCPM-V 2.6 int4）
            msgs = [{'role': 'user', 'content': images + [prompt]}]
            
            # 生成响应
            start_time = time.time()
            res = self.vlm_model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.vlm_tokenizer
            )
            end_time = time.time()
            print(f"VLM分析完成，耗时: {(end_time - start_time):.2f}秒")
            
            # 构建响应格式（适配 _extract_content_from_vlm_result 方法）
            response_data = {
                "choices": [
                    {
                        "message": {
                            "content": res
                        }
                    }
                ]
            }
            
            # 释放内存
            self.clear_memory()
            
            return response_data
            
        except Exception as e:
            print(f"分析图像失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def _generate_vlm_response(self, prompt):
        """
        生成VLM响应
        """
        if not self.vlm_model or not self.vlm_processor:
            raise Exception("VLM模型未加载")
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # 编码输入
        inputs = self.vlm_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.vlm_model.device)
        
        # 生成响应
        generated_ids = self.vlm_model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # 解码输出
        response = self.vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response.strip()


# 测试代码
if __name__ == "__main__":
    print("测试本地LLM服务...")
    
    try:
        # 初始化本地LLM服务
        llm_service = LLMServiceLocal()
        
        # 测试非流式调用
        print("\n测试非流式调用:")
        messages = [
            {"role": "system", "content": "你是一个智能助手"},
            {"role": "user", "content": "你好，你是谁？"}
        ]
        
        response = llm_service.create(messages)
        print(f"响应: {response}")
        
        # 测试流式调用
        print("\n测试流式调用:")
        
        def callback(text):
            print(text, end='', flush=True)
        
        for chunk in llm_service.create_stream(messages, callback=callback):
            pass
        
        print("\n测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
