#!/usr/bin/env python3
"""
记忆工具MCP服务器 - 提供AI读写记忆的能力

支持功能：
- read_memory: 读取记忆文档
- write_memory: 写入记忆文档
- search_memory: 搜索记忆文档
- grep_memory: 在记忆文档中搜索模式
- clear_memory: 清空记忆文档
- get_memory_stats: 获取记忆统计信息
"""

import asyncio
import sys
import os

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from memory_manager import MemoryManager
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)


class MemoryToolServer:
    """记忆工具服务器"""
    
    def __init__(self):
        """初始化记忆工具服务器"""
        # 获取项目根目录
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.memory_manager = MemoryManager(self.base_dir)
        print(f"[记忆服务器] 初始化完成，基础目录: {self.base_dir}")
    
    async def read_memory_handler(self, args: dict) -> list[TextContent]:
        """
        读取记忆文档
        
        Args:
            memory_type: 记忆类型 ('general' 或 'tool')
            
        Returns:
            记忆内容
        """
        memory_type = args.get('memory_type', 'general')
        content = self.memory_manager.read_memory(memory_type)
        
        return [
            TextContent(
                type="text",
                text=f"记忆内容 ({memory_type}):\n\n{content}"
            )
        ]
    
    async def write_memory_handler(self, args: dict) -> list[TextContent]:
        """
        写入记忆文档
        
        Args:
            content: 要写入的内容
            memory_type: 记忆类型 ('general' 或 'tool')
            append: 是否追加到文件末尾 (默认: true)
            
        Returns:
            操作结果
        """
        content = args.get('content', '')
        memory_type = args.get('memory_type', 'general')
        append = args.get('append', True)
        
        if not content:
            return [
                TextContent(
                    type="text",
                    text="错误：内容不能为空"
                )
            ]
        
        result = self.memory_manager.write_memory(content, memory_type, append)
        
        return [
            TextContent(
                type="text",
                text=result
            )
        ]
    
    async def search_memory_handler(self, args: dict) -> list[TextContent]:
        """
        搜索记忆文档
        
        Args:
            query: 搜索关键词
            memory_type: 记忆类型 ('general' 或 'tool')
            max_results: 最大返回结果数 (默认: 5)
            
        Returns:
            搜索结果
        """
        query = args.get('query', '')
        memory_type = args.get('memory_type', 'general')
        max_results = args.get('max_results', 5)
        
        if not query:
            return [
                TextContent(
                    type="text",
                    text="错误：查询词不能为空"
                )
            ]
        
        results = self.memory_manager.search_memory(query, memory_type, max_results)
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. [行{result['line_number']}] {result['content'][:100]}...\n"
                f"   匹配分数: {result['match_score']:.2f}"
            )
        
        output = f"搜索结果 (查询: '{query}', 类型: {memory_type}):\n\n"
        output += "\n\n".join(formatted_results) if formatted_results else "未找到匹配结果"
        
        return [
            TextContent(
                type="text",
                text=output
            )
        ]
    
    async def grep_memory_handler(self, args: dict) -> list[TextContent]:
        """
        在记忆文档中搜索模式（类似grep命令）
        
        Args:
            pattern: 正则表达式模式
            memory_type: 记忆类型 ('general' 或 'tool')
            context_lines: 上下文行数 (默认: 2)
            
        Returns:
            匹配结果
        """
        pattern = args.get('pattern', '')
        memory_type = args.get('memory_type', 'general')
        context_lines = args.get('context_lines', 2)
        
        if not pattern:
            return [
                TextContent(
                    type="text",
                    text="错误：模式不能为空"
                )
            ]
        
        results = self.memory_manager.grep_memory(pattern, memory_type, context_lines)
        
        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. [行{result['line_number']}] {result['matched_line'][:80]}...\n"
                f"   上下文:\n{result['context']}"
            )
        
        output = f"Grep结果 (模式: '{pattern}', 类型: {memory_type}):\n\n"
        output += "\n\n".join(formatted_results) if formatted_results else "未找到匹配结果"
        
        return [
            TextContent(
                type="text",
                text=output
            )
        ]
    
    async def clear_memory_handler(self, args: dict) -> list[TextContent]:
        """
        清空记忆文档
        
        Args:
            memory_type: 记忆类型 ('general' 或 'tool')
            
        Returns:
            操作结果
        """
        memory_type = args.get('memory_type', 'general')
        result = self.memory_manager.clear_memory(memory_type)
        
        return [
            TextContent(
                type="text",
                text=result
            )
        ]
    
    async def get_memory_stats_handler(self, args: dict) -> list[TextContent]:
        """
        获取记忆统计信息
        
        Args:
            memory_type: 记忆类型 ('general' 或 'tool')
            
        Returns:
            统计信息
        """
        memory_type = args.get('memory_type', 'general')
        stats = self.memory_manager.get_memory_stats(memory_type)
        
        # 格式化统计信息
        output = f"记忆统计 ({memory_type}):\n\n"
        output += f"存在: {'是' if stats.get('exists') else '否'}\n"
        output += f"大小: {stats.get('size', 0)} 字节\n"
        output += f"行数: {stats.get('lines', 0)}\n"
        output += f"最后修改: {stats.get('last_modified', '无')}\n"
        
        return [
            TextContent(
                type="text",
                text=output
            )
        ]


async def main():
    """主函数"""
    # 创建记忆工具服务器
    server = MemoryToolServer()
    
    # 定义工具列表
    tools = [
        Tool(
            name="read_memory",
            description="读取记忆文档。可以读取普通记忆或工具描述记忆。",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "记忆类型: 'general' (普通记忆) 或 'tool' (工具描述)",
                        "enum": ["general", "tool"],
                        "default": "general"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="write_memory",
            description="写入记忆文档。AI可以自主决定是否需要写入记忆。支持追加或覆盖模式。",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要写入的内容"
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "记忆类型: 'general' (普通记忆) 或 'tool' (工具描述)",
                        "enum": ["general", "tool"],
                        "default": "general"
                    },
                    "append": {
                        "type": "boolean",
                        "description": "是否追加到文件末尾 (true) 或覆盖文件 (false)",
                        "default": True
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="search_memory",
            description="搜索记忆文档。支持关键词搜索，返回匹配分数最高的结果。",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "记忆类型: 'general' (普通记忆) 或 'tool' (工具描述)",
                        "enum": ["general", "tool"],
                        "default": "general"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "最大返回结果数",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="grep_memory",
            description="在记忆文档中搜索模式（类似grep命令）。支持正则表达式，返回匹配行和上下文。",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "正则表达式模式"
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "记忆类型: 'general' (普通记忆) 或 'tool' (工具描述)",
                        "enum": ["general", "tool"],
                        "default": "general"
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "上下文行数",
                        "default": 2,
                        "minimum": 0,
                        "maximum": 10
                    }
                },
                "required": ["pattern"]
            }
        ),
        Tool(
            name="clear_memory",
            description="清空记忆文档。谨慎使用，会删除所有内容。",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "记忆类型: 'general' (普通记忆) 或 'tool' (工具描述)",
                        "enum": ["general", "tool"],
                        "default": "general"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_memory_stats",
            description="获取记忆统计信息。包括文件大小、行数、最后修改时间等。",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "description": "记忆类型: 'general' (普通记忆) 或 'tool' (工具描述)",
                        "enum": ["general", "tool"],
                        "default": "general"
                    }
                },
                "required": []
            }
        )
    ]
    
    # 创建MCP服务器
    async with stdio_server() as (read_stream, write_stream):
        async with Server(
            "memory-tool",
            version="1.0.0",
            request_timeout=30.0,
            notification_timeout=5.0
        ) as mcp_server:
            # 注册工具
            for tool in tools:
                mcp_server.add_tool(
                    tool,
                    handler=getattr(server, f"{tool.name}_handler")
                )
            
            print("[记忆服务器] 已注册 6 个工具")
            print("[记忆服务器] 服务器已启动，等待连接...")
            
            # 运行服务器
            await mcp_server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="memory-tool",
                    server_version="1.0.0",
                    capabilities=mcp_server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
