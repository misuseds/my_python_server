#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM 服务测试脚本
"""

import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_server.llm_class import LLMService


def test_llm_basic():
    """测试基本的 LLM 功能"""
    print("\n=== 测试基本 LLM 功能 ===")
    
    # 初始化 LLM 服务
    llm = LLMService()
    
    # 准备测试消息
    messages = [
        {
            "role": "user",
            "content": "你好，你是谁？"
        }
    ]
    
    print("发送消息到 LLM 服务...")
    
    try:
        # 调用 LLM 服务（非流式）
        response = llm.create(messages)
        
        print("\n=== LLM 响应 ===")
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


def test_llm_stream():
    """测试流式 LLM 功能"""
    print("\n=== 测试流式 LLM 功能 ===")
    
    # 初始化 LLM 服务
    llm = LLMService()
    
    # 准备测试消息
    messages = [
        {
            "role": "user", 
            "content": "请简要介绍一下人工智能的发展历史"
        }
    ]
    
    print("发送流式请求到 LLM 服务...")
    
    try:
        # 回调函数，用于打印流式输出
        def callback(chunk):
            print(chunk, end="", flush=True)
        
        # 调用流式方法
        print("\n=== 流式响应 ===")
        chunks = []
        for chunk in llm.create_stream(messages, callback=callback):
            chunks.append(chunk)
        
        print("\n\n=== 流式测试完成 ===")
        print(f"共接收 {len(chunks)} 个响应块")
        return True
        
    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        return False


def test_llm_multi_turn():
    """测试多轮对话功能"""
    print("\n=== 测试多轮对话功能 ===")
    
    # 初始化 LLM 服务
    llm = LLMService()
    
    # 准备多轮对话消息
    messages = [
        {
            "role": "user",
            "content": "你好，我想了解一下 Python 编程语言"
        },
        {
            "role": "assistant",
            "content": "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。它广泛应用于数据分析、机器学习、Web 开发等领域。"
        },
        {
            "role": "user",
            "content": "那 Python 和 JavaScript 有什么区别呢？"
        }
    ]
    
    print("发送多轮对话请求到 LLM 服务...")
    
    try:
        # 调用 LLM 服务
        response = llm.create(messages)
        
        print("\n=== LLM 多轮对话响应 ===")
        
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
    print("开始测试 LLM 服务...")
    
    # 运行各项测试
    test_results = []
    
    test_results.append("基本功能测试: " + ("通过" if test_llm_basic() else "失败"))
    test_results.append("流式功能测试: " + ("通过" if test_llm_stream() else "失败"))
    test_results.append("多轮对话测试: " + ("通过" if test_llm_multi_turn() else "失败"))
    
    # 打印测试结果摘要
    print("\n=== 测试结果摘要 ===")
    for result in test_results:
        print(result)
    
    print("\nLLM 测试完成！")
