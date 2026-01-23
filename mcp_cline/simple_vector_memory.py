"""
简化的向量记忆系统 - 避免依赖冲突
使用内置库实现基本的向量存储和检索功能
"""
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional
import math
from dotenv import load_dotenv

# 加载环境变量（使用与llm_server相同的方式）
dotenv_path = r'E:\code\my_python_server\my_python_server_private\.env'
load_dotenv(dotenv_path)

# 尝试导入dashscope库来调用魔搭的embedding服务
MODEL_AVAILABLE = False
try:
    import dashscope
    from http import HTTPStatus
    # 从环境变量获取API密钥
    api_key = os.getenv('VLM_OPENAI_API_KEY')
    if api_key:
        dashscope.api_key = api_key
        MODEL_AVAILABLE = True
        print("[记忆系统] Dashscope库可用，API密钥已配置")
    else:
        print("[记忆系统] Dashscope库可用，但API密钥未配置")
except ImportError as e:
    print(f"[记忆系统] Dashscope库不可用: {e}")
except Exception as e:
    print(f"[记忆系统] 初始化Dashscope失败: {e}")


class SimpleVectorMemory:
    """简化的向量记忆系统"""

    def __init__(self, persist_directory: str = None):
        """
        初始化向量记忆系统

        Args:
            persist_directory: 向量数据库持久化存储路径
        """
        # 设置默认持久化路径
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(base_dir, "simple_vector_memory_db")

        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # 存储文件路径
        self.memory_file = os.path.join(persist_directory, "memories.json")
        
        # 记忆存储
        self.memories = []
        self._load_memories()

        # 初始化魔搭embedding模型
        self.embedding_dim = 1024  # text-embedding-v4模型的默认维度

        if MODEL_AVAILABLE:
            try:
                # 使用dashscope调用魔搭的embedding服务
                model_name = "text-embedding-v4"
                print(f"[记忆系统] 初始化Dashscope服务，使用模型: {model_name}...")
                print(f"[记忆系统] Dashscope服务初始化完成")
            except Exception as e:
                print(f"[记忆系统] 初始化Dashscope服务失败: {e}")
        else:
            print("[记忆系统] Dashscope库不可用，将使用简单编码")

        print("[记忆系统] 简化版向量记忆系统初始化完成")
        print(f"[记忆系统] 已加载 {len(self.memories)} 条记忆")

    def _load_memories(self):
        """加载记忆数据"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        self.memories = json.loads(content)
        except Exception as e:
            print(f"[记忆系统] 加载记忆失败: {e}")
            self.memories = []

    def _save_memories(self):
        """保存记忆数据"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[记忆系统] 保存记忆失败: {e}")

    def _simple_encode(self, text: str) -> List[float]:
        """
        文本编码方法
        使用dashscope调用魔搭的embedding服务

        Args:
            text: 要编码的文本

        Returns:
            向量列表
        """
        if MODEL_AVAILABLE:
            # 使用dashscope调用魔搭的embedding服务
            resp = dashscope.TextEmbedding.call(
                model="text-embedding-v4",
                input=text
            )
            if resp.status_code == HTTPStatus.OK:
                embedding = resp.output['embeddings'][0]['embedding']
                return embedding
            else:
                raise Exception(f"[记忆系统] Dashscope服务调用失败: {resp.message}")
        else:
            raise Exception("[记忆系统] Dashscope不可用，无法编码文本")

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度值
        """
        if not vec1 or not vec2:
            return 0.0
        
        # 确保向量长度相同
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        # 计算点积
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        
        # 计算模长
        norm1 = math.sqrt(sum(v ** 2 for v in vec1))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def save_memory(
        self,
        vlm_analysis: str,
        llm_commentary: str,
        metadata: Dict = None,
        timestamp: float = None
    ) -> str:
        """
        保存一条记忆

        Args:
            vlm_analysis: VLM分析结果
            llm_commentary: LLM吐槽
            metadata: 额外的元数据
            timestamp: 时间戳

        Returns:
            记忆ID
        """
        if timestamp is None:
            timestamp = time.time()

        # 生成唯一ID
        memory_id = f"mem_{timestamp}_{len(vlm_analysis) % 1000}"

        # 编码VLM分析
        embedding = self._simple_encode(vlm_analysis)

        # 准备元数据
        if metadata is None:
            metadata = {}

        metadata.update({
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "type": "monitoring"
        })

        # 存储记忆
        memory_info = {
            "id": memory_id,
            "vlm_analysis": vlm_analysis,
            "llm_commentary": llm_commentary,
            "metadata": metadata,
            "embedding": embedding
        }

        self.memories.append(memory_info)
        
        # 限制记忆数量
        if len(self.memories) > 1000:
            self.memories = self.memories[-1000:]

        # 保存到文件
        self._save_memories()

        print(f"[记忆系统] 保存记忆: {memory_id}")
        print(f"[记忆系统] VLM分析: {vlm_analysis[:50]}...")
        print(f"[记忆系统] LLM吐槽: {llm_commentary[:50]}...")

        return memory_id

    def retrieve_memory(
        self,
        query_text: str,
        top_k: int = 3,
        memory_type: str = None
    ) -> List[Dict]:
        """
        检索相关的记忆

        Args:
            query_text: 查询文本
            top_k: 返回前k条最相关的记忆
            memory_type: 记忆类型过滤

        Returns:
            相关记忆列表
        """
        if not self.memories:
            return []

        # 编码查询文本
        query_embedding = self._simple_encode(query_text)

        # 计算相似度
        results = []
        for memory in self.memories:
            # 类型过滤
            if memory_type and memory.get("metadata", {}).get("type") != memory_type:
                continue

            # 计算相似度
            similarity = self._cosine_similarity(query_embedding, memory.get("embedding", []))
            
            results.append({
                "id": memory.get("id"),
                "document": memory.get("vlm_analysis"),
                "metadata": memory.get("metadata"),
                "distance": 1.0 - similarity  # 转换为距离
            })

        # 按相似度排序
        results.sort(key=lambda x: x["distance"])

        # 返回前k条
        return results[:top_k]

    def get_recent_memories(self, limit: int = 5) -> List[Dict]:
        """
        获取最近的记忆

        Args:
            limit: 返回数量

        Returns:
            最近记忆列表
        """
        if not self.memories:
            return []

        # 按时间戳排序
        recent_memories = sorted(
            self.memories,
            key=lambda x: x.get("metadata", {}).get("timestamp", 0),
            reverse=True
        )

        # 转换为与完整版兼容的数据结构
        formatted_memories = []
        for mem in recent_memories[:limit]:
            formatted_mem = {
                'id': mem.get('id'),
                'document': mem.get('vlm_analysis'),
                'metadata': mem.get('metadata')
            }
            formatted_memories.append(formatted_mem)

        return formatted_memories

    def format_memories_for_context(self, memories: List[Dict], max_count: int = 3) -> str:
        """
        将记忆格式化为上下文字符串

        Args:
            memories: 记忆列表
            max_count: 最多包含几条记忆

        Returns:
            格式化后的上下文字符串
        """
        if not memories:
            return "暂无相关记忆"

        context_parts = []
        for i, memory in enumerate(memories[:max_count]):
            memory_time = memory.get("metadata", {}).get("datetime", "未知时间")
            memory_text = memory.get("document", "")
            memory_type = memory.get("metadata", {}).get("type", "unknown")

            context_parts.append(
                f"{i+1}. [{memory_time}] ({memory_type}) {memory_text}"
            )

        return "\n".join(context_parts)

    def clear_all(self):
        """清空所有记忆"""
        self.memories = []
        self._save_memories()
        print("[记忆系统] 已清空所有记忆")

    def get_stats(self) -> Dict:
        """获取记忆系统统计信息"""
        model_name = "text-embedding-v4 (dashscope)" if MODEL_AVAILABLE else "simple-encoder"
        return {
            "total_memories": len(self.memories),
            "persist_directory": self.persist_directory,
            "model": model_name,
            "embedding_dim": self.embedding_dim if MODEL_AVAILABLE else len(self._simple_encode("test"))
        }
