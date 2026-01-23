"""
测试修复后的记忆系统
"""
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_cline.vector_memory import VectorMemory

def main():
    print("=" * 60)
    print("测试修复后的记忆系统")
    print("=" * 60)

    # 测试文件路径
    test_memory_dir = "test_vector_memory_db"
    
    # 清理旧测试目录
    if os.path.exists(test_memory_dir):
        import shutil
        shutil.rmtree(test_memory_dir)
        print(f"清理旧测试目录: {test_memory_dir}")

    # 创建记忆系统
    print("\n1. 创建记忆系统...")
    memory = VectorMemory(test_memory_dir)
    print("   ✓ 记忆系统创建成功")

    # 获取系统信息
    print("\n2. 记忆系统信息:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 测试保存记忆
    print("\n3. 测试保存记忆...")
    memory.save_memory(
        vlm_analysis="一只猫在沙发上睡觉",
        llm_commentary="这只猫看起来很嚣张",
        metadata={"category": "cat", "time": "morning"}
    )
    print("   ✓ 保存记忆1")

    memory.save_memory(
        vlm_analysis="猫从沙发上跳到地板",
        llm_commentary="哟，嚣张猫下地视察民情了？",
        metadata={"category": "cat", "time": "morning"}
    )
    print("   ✓ 保存记忆2")

    memory.save_memory(
        vlm_analysis="用户正在写代码",
        llm_commentary="代码写得不错，继续加油",
        metadata={"category": "work", "time": "afternoon"}
    )
    print("   ✓ 保存记忆3")

    # 测试检索记忆
    print("\n4. 测试检索记忆 (查询: '猫在地上')...")
    results = memory.retrieve_memory("猫在地上", top_k=2)
    print(f"   检索到 {len(results)} 条相关记忆:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['document'][:50]}... (相似度: {1 - result['distance']:.2f})")

    # 测试检索工作相关记忆
    print("\n5. 测试检索记忆 (查询: '写代码')...")
    results = memory.retrieve_memory("写代码", top_k=2)
    print(f"   检索到 {len(results)} 条相关记忆:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['document'][:50]}... (相似度: {1 - result['distance']:.2f})")

    # 测试格式化上下文
    print("\n6. 测试格式化上下文...")
    results = memory.retrieve_memory("猫", top_k=3)
    context = memory.format_memories_for_context(results, max_count=3)
    print("   格式化后的上下文:")
    print("   " + "\n   ".join(context.split("\n")))

    # 测试获取最近记忆
    print("\n7. 测试获取最近记忆...")
    recent_memories = memory.get_recent_memories(limit=3)
    print(f"   最近 {len(recent_memories)} 条记忆:")
    for i, mem in enumerate(recent_memories, 1):
        print(f"   {i}. {mem['document'][:50]}...")

    # 测试统计信息
    print("\n8. 记忆系统统计:")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # 测试保存和加载
    print("\n9. 测试保存和加载...")
    # 创建新实例加载同一个目录
    new_memory = VectorMemory(test_memory_dir)
    loaded_stats = new_memory.get_stats()
    print(f"   重新加载后记忆数量: {loaded_stats.get('total_memories', 0)}")
    if loaded_stats.get('total_memories', 0) > 0:
        print("   ✓ 记忆保存和加载成功")
    else:
        print("   ✗ 记忆保存和加载失败")

    # 清理测试目录
    if os.path.exists(test_memory_dir):
        import shutil
        shutil.rmtree(test_memory_dir)
        print(f"\n清理测试目录: {test_memory_dir}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出...")
