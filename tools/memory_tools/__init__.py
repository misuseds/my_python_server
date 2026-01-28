#!/usr/bin/env python3
"""
Memory工具包 - 提供记忆管理相关的工具
"""

from langchain.tools import tool
import sys
import os

# 添加memory目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入memory管理器
from memory_manager import MemoryManager

# 创建全局memory管理器实例
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
memory_manager = MemoryManager(base_dir)


@tool
def search_memory(query: str) -> str:
    """
    搜索记忆文档
    
    Args:
        query: 搜索查询字符串
        
    Returns:
        搜索结果
    """
    try:
        result = memory_manager.search(query)
        return f"搜索结果: {result}"
    except Exception as e:
        return f"搜索失败: {str(e)}"


@tool
def write_memory(content: str) -> str:
    """
    写入记忆文档
    
    Args:
        content: 要写入的内容
        
    Returns:
        写入结果
    """
    try:
        result = memory_manager.add(content)
        return f"写入成功: {result}"
    except Exception as e:
        return f"写入失败: {str(e)}"


@tool
def get_memory_stats() -> str:
    """
    获取记忆统计信息
    
    Returns:
        记忆统计信息
    """
    try:
        stats = memory_manager.get_stats()
        return f"记忆统计: {stats}"
    except Exception as e:
        return f"获取统计失败: {str(e)}"
