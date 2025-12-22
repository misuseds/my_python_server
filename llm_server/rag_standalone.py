# rag_simple.py
import os
import json
from datetime import datetime
from typing import List, Dict, Any

class SimpleRAG:
    """
    简化版RAG系统，不依赖模型和网络访问
    """
    
    def __init__(self, memory_file: str = "llm_server/memory.txt"):
        self.memory_file = memory_file
        # 确保目录存在
        os.makedirs(os.path.dirname(self.memory_file) if os.path.dirname(self.memory_file) else '.', exist_ok=True)
        
        # 创建内存文件如果不存在
        if not os.path.exists(self.memory_file):
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                pass
    
    def save_memory(self, role: str, content: str) -> bool:
        """
        保存记忆到文件
        
        Args:
            role: 角色 ("user", "assistant", "system")
            content: 内容
            
        Returns:
            是否保存成功
        """
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            memory_entry = {
                "timestamp": timestamp,
                "role": role,
                "content": content
            }
            
            # 读取现有记忆
            memories = self.load_memories()
            memories.append(memory_entry)
            
            # 保存所有记忆（只保留最近100条）
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memories[-100:], f, ensure_ascii=False, indent=2)
            
            print(f"记忆已保存: {role}: {content[:50]}...")
            return True
        except Exception as e:
            print(f"保存记忆失败: {e}")
            return False
    
    def load_memories(self, limit: int = None) -> List[Dict]:
        """
        从文件加载记忆
        
        Args:
            limit: 限制返回的记忆数量
            
        Returns:
            记忆列表
        """
        try:
            if not os.path.exists(self.memory_file):
                return []
            
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                memories = json.load(f)
            
            if limit:
                memories = memories[-limit:]  # 只返回最近的几条记忆
                
            return memories
        except Exception as e:
            print(f"加载记忆失败: {e}")
            return []
    
    def clear_memories(self) -> bool:
        """
        清空记忆
        
        Returns:
            是否清空成功
        """
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print("记忆已清空")
            return True
        except Exception as e:
            print(f"清空记忆失败: {e}")
            return False
    
    def get_recent_context(self, limit: int = 5) -> str:
        """
        获取最近的对话上下文
        
        Args:
            limit: 上下文条目数
            
        Returns:
            格式化的上下文字符串
        """
        memories = self.load_memories(limit)
        if not memories:
            return ""
        
        context_lines = []
        for memory in memories:
            context_lines.append(f"[{memory['timestamp']}] {memory['role']}: {memory['content']}")
        
        return "\n".join(context_lines)
    
    def search_memories(self, keyword: str) -> List[Dict]:
        """
        根据关键词搜索记忆
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            匹配的记忆列表
        """
        memories = self.load_memories()
        matched_memories = []
        
        for memory in memories:
            if keyword.lower() in memory['content'].lower():
                matched_memories.append(memory)
        
        return matched_memories

# 使用示例
if __name__ == "__main__":
    # 初始化简化版RAG系统
    rag = SimpleRAG()
    
    # 示例1: 保存记忆
    print("=== 示例1: 保存记忆 ===")
    rag.save_memory("user", "你好，我想了解人工智能")
    rag.save_memory("assistant", "你好！人工智能是计算机科学的一个分支...")
    rag.save_memory("user", "能详细说说机器学习吗？")
    
    # 示例2: 获取最近上下文
    print("\n=== 示例2: 获取最近上下文 ===")
    context = rag.get_recent_context(3)
    print("最近的对话上下文:")
    print(context)
    
    # 示例3: 加载所有记忆
    print("\n=== 示例3: 加载所有记忆 ===")
    all_memories = rag.load_memories()
    print(f"总共 {len(all_memories)} 条记忆")
    for i, memory in enumerate(all_memories, 1):
        print(f"{i}. [{memory['timestamp']}] {memory['role']}: {memory['content'][:50]}...")
    
    # 示例4: 搜索记忆
    print("\n=== 示例4: 搜索记忆 ===")
    results = rag.search_memories("人工智能")
    print(f"找到 {len(results)} 条包含'人工智能'的记忆:")
    for result in results:
        print(f"- {result['content']}")