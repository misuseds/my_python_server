#!/usr/bin/env python3
"""
记忆管理工具 - 提供AI读写记忆的能力

支持功能：
- 读取记忆文档
- 写入记忆文档
- 搜索记忆文档
- AI自主决定是否写入记忆
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import re


class MemoryManager:
    """记忆管理器"""
    
    def __init__(self, base_dir: str):
        """
        初始化记忆管理器
        
        Args:
            base_dir: 基础目录
        """
        self.base_dir = base_dir
        self.memory_dir = os.path.join(base_dir, 'memory')
        
        # 确保记忆目录存在
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # 记忆文档路径
        self.general_memory_path = os.path.join(self.memory_dir, 'general_memory.txt')
        self.tool_description_path = os.path.join(self.memory_dir, 'tool_description.txt')
        
        print(f"[记忆管理] 初始化完成，记忆目录: {self.memory_dir}")
    
    def read_memory(self, memory_type: str = 'general') -> str:
        """
        读取记忆文档
        
        Args:
            memory_type: 记忆类型 ('general' 或 'tool')
            
        Returns:
            记忆内容
        """
        if memory_type == 'general':
            path = self.general_memory_path
        elif memory_type == 'tool':
            path = self.tool_description_path
        else:
            return f"错误：未知的记忆类型 '{memory_type}'"
        
        if not os.path.exists(path):
            return ""
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"[记忆管理] 读取记忆: {memory_type} ({len(content)} 字符)")
            return content
        except Exception as e:
            return f"错误：读取记忆失败 - {str(e)}"
    
    def write_memory(self, content: str, memory_type: str = 'general', append: bool = True) -> str:
        """
        写入记忆文档
        
        Args:
            content: 要写入的内容
            memory_type: 记忆类型 ('general' 或 'tool')
            append: 是否追加到文件末尾
            
        Returns:
            操作结果
        """
        if memory_type == 'general':
            path = self.general_memory_path
        elif memory_type == 'tool':
            path = self.tool_description_path
        else:
            return f"错误：未知的记忆类型 '{memory_type}'"
        
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if append:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write(f"\n\n--- {timestamp} ---\n")
                    f.write(content)
            else:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            print(f"[记忆管理] 写入记忆: {memory_type} ({len(content)} 字符)")
            return f"成功：已写入 {memory_type} 记忆"
        except Exception as e:
            return f"错误：写入记忆失败 - {str(e)}"
    
    def search_memory(self, query: str, memory_type: str = 'general', max_results: int = 5) -> List[Dict]:
        """
        搜索记忆文档
        
        Args:
            query: 搜索关键词
            memory_type: 记忆类型 ('general' 或 'tool')
            max_results: 最大返回结果数
            
        Returns:
            搜索结果列表
        """
        if memory_type == 'general':
            path = self.general_memory_path
        elif memory_type == 'tool':
            path = self.tool_description_path
        else:
            return []
        
        if not os.path.exists(path):
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按行分割
            lines = content.split('\n')
            
            # 搜索匹配的行
            results = []
            for i, line in enumerate(lines):
                if query.lower() in line.lower():
                    results.append({
                        'line_number': i + 1,
                        'content': line.strip(),
                        'match_score': self._calculate_match_score(query, line)
                    })
            
            # 按匹配分数排序
            results.sort(key=lambda x: x['match_score'], reverse=True)
            
            # 返回前N个结果
            return results[:max_results]
        except Exception as e:
            print(f"[记忆管理] 搜索失败: {e}")
            return []
    
    def _calculate_match_score(self, query: str, content: str) -> float:
        """
        计算匹配分数
        
        Args:
            query: 查询词
            content: 内容
            
        Returns:
            匹配分数 (0-1)
        """
        query_lower = query.lower()
        content_lower = content.lower()
        
        # 完全匹配
        if query_lower in content_lower:
            return 1.0
        
        # 部分匹配
        words = query_lower.split()
        matched_words = sum(1 for word in words if word in content_lower)
        return matched_words / len(words) if words else 0.0
    
    def grep_memory(self, pattern: str, memory_type: str = 'general', context_lines: int = 2) -> List[Dict]:
        """
        在记忆文档中搜索模式（类似grep命令）
        
        Args:
            pattern: 正则表达式模式
            memory_type: 记忆类型 ('general' 或 'tool')
            context_lines: 上下文行数
            
        Returns:
            匹配结果列表
        """
        if memory_type == 'general':
            path = self.general_memory_path
        elif memory_type == 'tool':
            path = self.tool_description_path
        else:
            return []
        
        if not os.path.exists(path):
            return []
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 编译正则表达式
            try:
                regex = re.compile(pattern, re.IGNORECASE)
            except re.error:
                return []
            
            # 搜索匹配
            results = []
            for i, line in enumerate(lines):
                if regex.search(line):
                    # 获取上下文
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = ''.join(lines[start:end])
                    
                    results.append({
                        'line_number': i + 1,
                        'matched_line': line.strip(),
                        'context': context.strip()
                    })
            
            return results
        except Exception as e:
            print(f"[记忆管理] grep失败: {e}")
            return []
    
    def clear_memory(self, memory_type: str = 'general') -> str:
        """
        清空记忆文档
        
        Args:
            memory_type: 记忆类型 ('general' 或 'tool')
            
        Returns:
            操作结果
        """
        if memory_type == 'general':
            path = self.general_memory_path
        elif memory_type == 'tool':
            path = self.tool_description_path
        else:
            return f"错误：未知的记忆类型 '{memory_type}'"
        
        try:
            if os.path.exists(path):
                os.remove(path)
            print(f"[记忆管理] 清空记忆: {memory_type}")
            return f"成功：已清空 {memory_type} 记忆"
        except Exception as e:
            return f"错误：清空记忆失败 - {str(e)}"
    
    def get_memory_stats(self, memory_type: str = 'general') -> Dict:
        """
        获取记忆统计信息
        
        Args:
            memory_type: 记忆类型 ('general' 或 'tool')
            
        Returns:
            统计信息字典
        """
        if memory_type == 'general':
            path = self.general_memory_path
        elif memory_type == 'tool':
            path = self.tool_description_path
        else:
            return {}
        
        if not os.path.exists(path):
            return {
                'exists': False,
                'size': 0,
                'lines': 0,
                'last_modified': None
            }
        
        try:
            stat = os.stat(path)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            return {
                'exists': True,
                'size': stat.st_size,
                'lines': len(lines),
                'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        except Exception as e:
            print(f"[记忆管理] 获取统计失败: {e}")
            return {}


# MCP工具函数定义
def read_memory(memory_type: str = 'general') -> str:
    """
    读取记忆文档
    
    Args:
        memory_type: 记忆类型 ('general' 或 'tool')
        
    Returns:
        记忆内容
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manager = MemoryManager(base_dir)
    return manager.read_memory(memory_type)


def write_memory(content: str, memory_type: str = 'general', append: bool = True) -> str:
    """
    写入记忆文档
    
    Args:
        content: 要写入的内容
        memory_type: 记忆类型 ('general' 或 'tool')
        append: 是否追加到文件末尾
        
    Returns:
        操作结果
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manager = MemoryManager(base_dir)
    return manager.write_memory(content, memory_type, append)


def search_memory(query: str, memory_type: str = 'general', max_results: int = 5) -> str:
    """
    搜索记忆文档
    
    Args:
        query: 搜索关键词
        memory_type: 记忆类型 ('general' 或 'tool')
        max_results: 最大返回结果数
        
    Returns:
        搜索结果（JSON格式）
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manager = MemoryManager(base_dir)
    results = manager.search_memory(query, memory_type, max_results)
    return json.dumps(results, ensure_ascii=False, indent=2)


def grep_memory(pattern: str, memory_type: str = 'general', context_lines: int = 2) -> str:
    """
    在记忆文档中搜索模式（类似grep命令）
    
    Args:
        pattern: 正则表达式模式
        memory_type: 记忆类型 ('general' 或 'tool')
        context_lines: 上下文行数
        
    Returns:
        匹配结果（JSON格式）
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manager = MemoryManager(base_dir)
    results = manager.grep_memory(pattern, memory_type, context_lines)
    return json.dumps(results, ensure_ascii=False, indent=2)


def clear_memory(memory_type: str = 'general') -> str:
    """
    清空记忆文档
    
    Args:
        memory_type: 记忆类型 ('general' 或 'tool')
        
    Returns:
        操作结果
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manager = MemoryManager(base_dir)
    return manager.clear_memory(memory_type)


def get_memory_stats(memory_type: str = 'general') -> str:
    """
    获取记忆统计信息
    
    Args:
        memory_type: 记忆类型 ('general' 或 'tool')
        
    Returns:
        统计信息（JSON格式）
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manager = MemoryManager(base_dir)
    stats = manager.get_memory_stats(memory_type)
    return json.dumps(stats, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    import sys
    
    # 测试记忆管理功能
    print("=== 记忆管理工具测试 ===")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manager = MemoryManager(base_dir)
    
    # 写入测试
    print("\n1. 写入测试记忆...")
    result = manager.write_memory("这是一条测试记忆：用户喜欢使用Blender进行3D建模。", 'general')
    print(result)
    
    # 读取测试
    print("\n2. 读取测试记忆...")
    content = manager.read_memory('general')
    print(f"内容: {content}")
    
    # 搜索测试
    print("\n3. 搜索测试记忆...")
    results = manager.search_memory('Blender', 'general')
    print(f"找到 {len(results)} 条结果:")
    for r in results:
        print(f"  - {r['content'][:50]}...")
    
    # grep测试
    print("\n4. Grep测试记忆...")
    results = manager.grep_memory('用户.*喜欢', 'general')
    print(f"找到 {len(results)} 条匹配:")
    for r in results:
        print(f"  - 行{r['line_number']}: {r['matched_line'][:50]}...")
    
    # 统计测试
    print("\n5. 统计测试记忆...")
    stats = manager.get_memory_stats('general')
    print(f"统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")
