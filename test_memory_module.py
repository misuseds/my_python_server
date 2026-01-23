"""
测试记忆库模块
"""
import os
import sys

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_cline.memory_module import GameMemory

def main():
    print("=" * 60)
    print("测试记忆库模块")
    print("=" * 60)

    # 测试文件路径
    test_memory_file = "test_game_memory.json"
    
    # 清理旧测试文件
    if os.path.exists(test_memory_file):
        os.remove(test_memory_file)
        print(f"清理旧测试文件: {test_memory_file}")

    # 创建记忆系统
    print("\n1. 创建记忆系统...")
    memory = GameMemory(test_memory_file)
    print("   ✓ 记忆系统创建成功")

    # 测试添加记忆
    print("\n2. 测试添加记忆...")
    memory.add_memory(
        action="使用技能: 火焰箭",
        context="敌人数量: 3, 距离: 15米, 我方生命值: 80%",
        analysis="火焰箭可以同时伤害多个敌人，适合当前情况"
    )
    print("   ✓ 添加记忆1")

    memory.add_memory(
        action="移动到: 左侧掩体",
        context="敌人火力: 中等, 掩体状态: 完整",
        analysis="左侧掩体可以提供更好的防护，减少受到的伤害"
    )
    print("   ✓ 添加记忆2")

    memory.add_memory(
        action="使用道具: 治疗药水",
        context="我方生命值: 30%, 道具数量: 2",
        analysis="当前生命值较低，需要使用治疗药水恢复"
    )
    print("   ✓ 添加记忆3")

    # 测试获取最近记忆
    print("\n3. 测试获取最近记忆...")
    recent_memories = memory.get_recent_memories(2)
    print(f"   获取到 {len(recent_memories)} 条最近记忆:")
    for i, mem in enumerate(recent_memories, 1):
        print(f"   {i}. [{mem['timestamp']}] {mem['action']}")

    # 测试按关键词搜索
    print("\n4. 测试按关键词搜索记忆...")
    skill_memories = memory.get_memories_by_action("技能")
    print(f"   搜索 '技能' 相关记忆: {len(skill_memories)} 条")
    for mem in skill_memories:
        print(f"   - [{mem['timestamp']}] {mem['action']}")

    # 测试获取会话摘要
    print("\n5. 测试获取会话摘要...")
    summary = memory.get_session_summary()
    print("   会话摘要:")
    print("   " + "\n   ".join(summary.split("\n")))

    # 测试获取上下文
    print("\n6. 测试获取上下文...")
    context = memory.get_context_for_prompt(2)
    print("   上下文:")
    print("   " + "\n   ".join(context.split("\n")))

    # 测试分析记忆
    print("\n7. 测试分析记忆...")
    analysis = memory.analyze_memories()
    print("   记忆分析:")
    print("   " + "\n   ".join(analysis.split("\n")))

    # 测试保存和加载
    print("\n8. 测试保存和加载...")
    # 创建新实例加载同一个文件
    new_memory = GameMemory(test_memory_file)
    loaded_count = len(new_memory.get_all_memories())
    print(f"   重新加载后记忆数量: {loaded_count}")
    if loaded_count == 3:
        print("   ✓ 记忆保存和加载成功")
    else:
        print("   ✗ 记忆保存和加载失败")

    # 清理测试文件
    if os.path.exists(test_memory_file):
        os.remove(test_memory_file)
        print(f"\n清理测试文件: {test_memory_file}")

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
