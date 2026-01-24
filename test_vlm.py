#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VLM 服务测试脚本
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_server.llm_class import VLMService

def test_vlm_basic():
    """测试基本的 VLM 功能"""
    print("\n=== 测试基本 VLM 功能 ===")
    
    # 初始化 VLM 服务
    vlm = VLMService()
    
    # 准备测试消息
    messages = [
        {
            "role": "user",
            "content": "这张图片里有什么？"
        }
    ]
    
    # 使用一张测试图片（如果存在）
    test_image = None
    # 检查是否有测试图片
    if os.path.exists("yolo/findgate_data/Snipaste_2026-01-13_23-40-40.png"):
        test_image = "yolo/findgate_data/Snipaste_2026-01-13_23-40-40.png"
        print(f"使用测试图片: {test_image}")
    else:
        print("警告: 未找到测试图片，将只测试文本功能")
    
    print("发送消息到 VLM 服务...")
    
    try:
        # 调用 VLM 服务
        response = vlm.create_with_image(messages, image_source=test_image)
        
        print("\n=== VLM 响应 ===")
        print(f"响应状态: 成功")
        print(f"响应类型: {type(response)}")
        
        # 提取并打印生成的内容
        if "choices" in response:
            for i, choice in enumerate(response["choices"]):
                if "message" in choice and "content" in choice["message"]:
                    print(f"\n生成内容 {i+1}:")
                    print(choice["message"]["content"])
        
        print("\n=== 测试完成 ===")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        return False

def test_vlm_multiple_images():
    """测试 VLM 多图功能"""
    print("\n=== 测试 VLM 多图功能 ===")
    
    # 初始化 VLM 服务
    vlm = VLMService()
    
    # 准备测试消息
    messages = [
        {
            "role": "user",
            "content": "比较这两张图片的不同之处"
        }
    ]
    
    # 收集测试图片
    test_images = []
    image_dir = "yolo/findgate_data"
    if os.path.exists(image_dir):
        # 获取前两张图片
        for img_file in os.listdir(image_dir):
            if img_file.endswith(".png") and len(test_images) < 2:
                test_images.append(os.path.join(image_dir, img_file))
    
    if len(test_images) >= 2:
        print(f"使用测试图片: {test_images}")
    else:
        print("警告: 未找到足够的测试图片，将只测试文本功能")
    
    print("发送多图请求到 VLM 服务...")
    
    try:
        # 调用 VLM 服务
        response = vlm.create_with_multiple_images(messages, image_sources=test_images)
        
        print("\n=== VLM 多图响应 ===")
        print(f"响应状态: 成功")
        print(f"响应类型: {type(response)}")
        
        # 提取并打印生成的内容
        if "choices" in response:
            for i, choice in enumerate(response["choices"]):
                if "message" in choice and "content" in choice["message"]:
                    print(f"\n生成内容 {i+1}:")
                    print(choice["message"]["content"])
        
        print("\n=== 多图测试完成 ===")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        return False

def test_vlm_conversation():
    """测试 VLM 多轮对话功能"""
    print("\n=== 测试 VLM 多轮对话功能 ===")
    
    # 初始化 VLM 服务
    vlm = VLMService()
    
    # 准备历史对话
    conversation_history = [
        {
            "role": "user",
            "content": "你好"
        },
        {
            "role": "assistant",
            "content": "你好！我是一个视觉语言模型，可以理解图片和文字。请问有什么可以帮助你的？"
        }
    ]
    
    # 当前查询
    current_query = "请描述一下这张图片"
    
    # 使用一张测试图片（如果存在）
    test_image = None
    if os.path.exists("yolo/findgate_data/Snipaste_2026-01-13_23-40-40.png"):
        test_image = "yolo/findgate_data/Snipaste_2026-01-13_23-40-40.png"
        print(f"使用测试图片: {test_image}")
    
    print("发送多轮对话请求到 VLM 服务...")
    
    try:
        # 调用 VLM 服务
        response = vlm.create_multimodal_conversation(conversation_history, current_query, image_source=test_image)
        
        print("\n=== VLM 多轮对话响应 ===")
        print(f"响应状态: 成功")
        print(f"响应类型: {type(response)}")
        
        # 提取并打印生成的内容
        if "choices" in response:
            for i, choice in enumerate(response["choices"]):
                if "message" in choice and "content" in choice["message"]:
                    print(f"\n生成内容 {i+1}:")
                    print(choice["message"]["content"])
        
        print("\n=== 多轮对话测试完成 ===")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试 VLM 服务...")
    
    # 运行各项测试
    test_results = []
    
    test_results.append("基本功能测试: " + ("通过" if test_vlm_basic() else "失败"))
    test_results.append("多图功能测试: " + ("通过" if test_vlm_multiple_images() else "失败"))
    test_results.append("多轮对话测试: " + ("通过" if test_vlm_conversation() else "失败"))
    
    # 打印测试结果摘要
    print("\n=== 测试结果摘要 ===")
    for result in test_results:
        print(result)
    
    print("\nVLM 测试完成！")
