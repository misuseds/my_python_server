"""
测试魔搭embedding模型集成
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_vector_memory import SimpleVectorMemory


def test_modelhub_embedding():
    """测试魔搭embedding模型"""
    print("=" * 60)
    print("测试魔搭embedding模型")
    print("=" * 60)

    # 创建记忆系统实例
    print("\n1. 创建向量记忆系统...")
    memory = SimpleVectorMemory()

    # 检查模型是否加载成功
    if memory.model is not None and memory.tokenizer is not None:
        print("[OK] 魔搭模型加载成功")
        print(f"  模型: damo/nlp_gte_sentence-embedding_chinese-base")
        print(f"  设备: {memory.device}")
    else:
        print("[WARN] 魔搭模型未加载，使用简单编码")

    # 测试编码
    print("\n2. 测试文本编码...")
    test_texts = [
        "一只猫在沙发上睡觉",
        "猫从沙发上跳到地板",
        "用户正在编写Python代码"
    ]

    embeddings = []
    for text in test_texts:
        embedding = memory._simple_encode(text)
        embeddings.append(embedding)
        print(f"  文本: {text}")
        print(f"  向量维度: {len(embedding)}")

    # 测试相似度计算
    print("\n3. 测试相似度计算...")
    similarity_0_1 = memory._cosine_similarity(embeddings[0], embeddings[1])
    similarity_0_2 = memory._cosine_similarity(embeddings[0], embeddings[2])

    print(f"  '猫在沙发上' vs '猫跳到地板': {similarity_0_1:.4f}")
    print(f"  '猫在沙发上' vs '编写代码': {similarity_0_2:.4f}")

    # 测试保存和检索
    print("\n4. 测试保存记忆...")
    memory.save_memory(
        vlm_analysis="一只猫在沙发上睡觉",
        llm_commentary="这只猫看起来很嚣张",
        metadata={"category": "cat", "action": "sleeping"}
    )

    memory.save_memory(
        vlm_analysis="猫从沙发上跳到地板",
        llm_commentary="哟，嚣张猫下地视察民情了？",
        metadata={"category": "cat", "action": "jumping"}
    )

    memory.save_memory(
        vlm_analysis="用户正在使用PyQt6开发界面",
        llm_commentary="代码写得很规范",
        metadata={"category": "coding", "action": "programming"}
    )

    print("\n5. 测试记忆检索...")
    query = "猫在地上"
    results = memory.retrieve_memory(query, top_k=2)

    print(f"  查询: {query}")
    print(f"  检索到 {len(results)} 条相关记忆:")
    for i, result in enumerate(results, 1):
        print(f"    {i}. {result['document']}")
        print(f"       相似度: {1 - result['distance']:.4f}")

    # 测试格式化上下文
    print("\n6. 测试格式化上下文...")
    context = memory.format_memories_for_context(results)
    print("  上下文字符串:")
    for line in context.split('\n'):
        print(f"    {line}")

    # 测试统计信息
    print("\n7. 获取系统统计...")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        test_modelhub_embedding()
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
