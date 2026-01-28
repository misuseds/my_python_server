#!/usr/bin/env python3
"""
记忆工具测试脚本
"""

import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from memory_manager import MemoryManager


def test_memory_manager():
    """测试记忆管理器功能"""
    print("=" * 60)
    print("记忆管理器测试")
    print("=" * 60)
    
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manager = MemoryManager(base_dir)
    
    # 测试1：写入普通记忆
    print("\n[测试1] 写入普通记忆...")
    result = manager.write_memory(
        "用户偏好：使用Blender进行3D建模，喜欢使用快捷键提高效率",
        'general',
        append=True
    )
    print(f"结果: {result}")
    
    # 测试2：写入工具描述记忆
    print("\n[测试2] 写入工具描述记忆...")
    result = manager.write_memory(
        "Blender导入FBX的最佳实践：\n1. 先检查单位设置（米 vs 厘米）\n2. 确认轴心位置\n3. 检查材质路径是否正确",
        'tool',
        append=True
    )
    print(f"结果: {result}")
    
    # 测试3：读取普通记忆
    print("\n[测试3] 读取普通记忆...")
    content = manager.read_memory('general')
    print(f"内容长度: {len(content)} 字符")
    print(f"内容预览: {content[:100]}...")
    
    # 测试4：读取工具描述记忆
    print("\n[测试4] 读取工具描述记忆...")
    content = manager.read_memory('tool')
    print(f"内容长度: {len(content)} 字符")
    print(f"内容预览: {content[:100]}...")
    
    # 测试5：搜索普通记忆
    print("\n[测试5] 搜索普通记忆...")
    results = manager.search_memory('Blender', 'general', max_results=3)
    print(f"找到 {len(results)} 条结果:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [行{r['line_number']}] {r['content'][:60]}... (分数: {r['match_score']:.2f})")
    
    # 测试6：Grep搜索
    print("\n[测试6] Grep搜索...")
    results = manager.grep_memory('用户.*偏好', 'general', context_lines=1)
    print(f"找到 {len(results)} 条匹配:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [行{r['line_number']}] {r['matched_line'][:60]}...")
    
    # 测试7：获取统计信息
    print("\n[测试7] 获取统计信息...")
    stats = manager.get_memory_stats('general')
    print(f"普通记忆统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    stats = manager.get_memory_stats('tool')
    print(f"\n工具描述记忆统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 测试8：写入更多记忆
    print("\n[测试8] 写入更多记忆...")
    test_memories = [
        "重要设置：分辨率1920x1080，帧率60fps",
        "快捷键：Ctrl+S保存，Ctrl+Z撤销",
        "工作流：先建模，再贴图，最后导出"
    ]
    for memory in test_memories:
        result = manager.write_memory(memory, 'general', append=True)
        print(f"  - {result}")
    
    # 测试9：搜索新写入的记忆
    print("\n[测试9] 搜索新写入的记忆...")
    results = manager.search_memory('快捷键', 'general', max_results=5)
    print(f"找到 {len(results)} 条结果:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [行{r['line_number']}] {r['content'][:60]}...")
    
    # 测试10：清空记忆
    print("\n[测试10] 清空记忆（仅演示，不实际执行）...")
    print("  提示：取消下面的注释以实际清空记忆")
    # result = manager.clear_memory('general')
    # print(f"结果: {result}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print("\n记忆文件位置:")
    print(f"  普通记忆: {manager.general_memory_path}")
    print(f"  工具描述记忆: {manager.tool_description_path}")
    print("\n提示：可以使用任何文本编辑器查看记忆文件内容")


if __name__ == '__main__':
    try:
        test_memory_manager()
    except Exception as e:
        print(f"\n错误: 测试失败 - {e}")
        import traceback
        traceback.print_exc()
