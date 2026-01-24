#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 多图分析测试脚本
"""

import os
import sys
from llm_server.llm_class_local import LLMServiceLocal

def main():
    """测试 VLM 多图分析功能"""
    print("[测试] 开始 VLM 多图分析测试...")
    
    # 初始化 LLM 服务
    llm_service = LLMServiceLocal()
    
    # 选择测试图像（从 screenshots 文件夹中选择前 2 张）
    screenshots_dir = "screenshots"
    image_files = [f for f in os.listdir(screenshots_dir) if f.endswith('.png')][:2]
    image_paths = [os.path.join(screenshots_dir, f) for f in image_files]
    
    print(f"[测试] 选择的测试图像: {image_paths}")
    
    # 测试多图分析
    prompt = "请描述这两张图像的内容，并比较它们的不同之处"
    print(f"[测试] 测试提示: {prompt}")
    
    try:
        response = llm_service.create_with_multiple_images(prompt, image_paths)
        print("[测试] 多图分析响应:")
        print(response)
        print("[测试] VLM 多图分析测试成功!")
    except Exception as e:
        print(f"[测试] VLM 多图分析测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
