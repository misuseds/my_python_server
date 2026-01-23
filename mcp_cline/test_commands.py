#!/usr/bin/env python3
"""
测试新添加的命令
"""
import os
import sys
import time

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_memory import VectorMemory


def test_commands():
    """测试命令功能"""
    print("=" * 60)
    print("测试新添加的命令功能")
    print("=" * 60)
    
    # 创建向量记忆系统
    memory = VectorMemory()
    
    # 测试get_all_memories方法
    print("\n1. 测试get_all_memories方法...")
    all_memories = memory.get_all_memories()
    print(f"获取到 {len(all_memories)} 条记忆")
    
    # 显示前3条记忆
    print("\n前3条记忆:")
    for i, mem in enumerate(all_memories[:3]):
        print(f"{i+1}. ID: {mem.get('id', '无ID')}")
        print(f"   内容: {mem.get('document', '无内容')[:100]}...")
        print(f"   时间: {mem.get('metadata', {}).get('datetime', '无时间')}")
        print()
    
    # 测试get_stats方法
    print("\n2. 测试get_stats方法...")
    stats = memory.get_stats()
    print(f"总记忆数: {stats.get('total_memories', 0)}")
    print(f"使用模型: {stats.get('model', '未知')}")
    print(f"版本: {stats.get('version', '未知')}")
    print(f"持久化目录: {stats.get('persist_directory', '未知')}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_commands()
