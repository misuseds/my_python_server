#!/usr/bin/env python3
"""
测试/clearmemory命令的功能
"""
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_memory import VectorMemory


def test_clearmemory():
    """测试清空数据库功能"""
    print("=" * 60)
    print("测试/clearmemory命令功能")
    print("=" * 60)
    
    # 创建向量记忆系统
    memory = VectorMemory()
    
    # 检查初始记忆数量
    stats = memory.get_stats()
    print(f"初始记忆数量: {stats.get('total_memories', 0)}")
    
    # 保存一些测试记忆
    print("\n1. 保存测试记忆...")
    for i in range(3):
        memory.save_memory(
            vlm_analysis=f"测试记忆 {i}: 一只猫在沙发上睡觉",
            llm_commentary=f"测试吐槽 {i}: 这只猫看起来很嚣张"
        )
    
    # 检查保存后的记忆数量
    stats = memory.get_stats()
    print(f"保存后记忆数量: {stats.get('total_memories', 0)}")
    
    # 执行清空操作
    print("\n2. 执行清空操作...")
    memory.clear_all()
    print("[命令] 数据库已清空")
    
    # 检查清空后的记忆数量
    stats = memory.get_stats()
    print(f"清空后记忆数量: {stats.get('total_memories', 0)}")
    
    # 验证是否清空
    if stats.get('total_memories', 0) == 0:
        print("\n✅ 测试通过: 数据库已成功清空")
    else:
        print("\n❌ 测试失败: 数据库未清空")
    
    print("=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_clearmemory()
