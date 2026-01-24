#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 内存优化测试脚本
测试使用 GLM-4.6V-Flash 模型和内存优化后的 VLM 多图分析功能
"""

import os
import sys
import time
from llm_server.llm_class_local import LLMServiceLocal

def main():
    """测试 VLM 内存优化"""
    print("[测试] 开始 VLM 内存优化测试...")
    
    # 选择测试图像（从 screenshots 文件夹中选择前 2 张）
    screenshots_dir = "screenshots"
    image_files = [f for f in os.listdir(screenshots_dir) if f.endswith('.png')][:2]
    image_paths = [os.path.join(screenshots_dir, f) for f in image_files]
    
    print(f"[测试] 选择的测试图像: {image_paths}")
    
    # 测试多次分析，验证内存是否正确释放
    for i in range(2):
        print(f"\n[测试] 第 {i+1} 次分析...")
        
        # 初始化 LLM 服务
        llm_service = LLMServiceLocal()
        
        # 测试使用 messages 和 image_sources 参数
        prompt = "请描述这两张图像的内容，并比较它们的不同之处"
        messages = [{"role": "user", "content": prompt}]
        
        try:
            start_time = time.time()
            vlm_result = llm_service.create_with_multiple_images(messages, image_sources=image_paths)
            end_time = time.time()
            
            print(f"[测试] 多图分析完成，耗时: {end_time - start_time:.2f}秒")
            print(f"[测试] 分析结果: {vlm_result}")
            
            # 等待 2 秒，让内存释放完成
            time.sleep(2)
            
        except Exception as e:
            print(f"[测试] VLM 多图分析测试失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 手动释放内存
            if 'llm_service' in locals():
                llm_service.clear_memory()
                del llm_service
            
            # 等待 2 秒，让内存释放完成
            time.sleep(2)
    
    print("\n[测试] VLM 内存优化测试完成!")

if __name__ == "__main__":
    main()
