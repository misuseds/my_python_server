#!/usr/bin/env python3
"""
Agent构建器 - 使用简单的工具调用模式

这个模块负责构建一个简单的工具调用处理器，避免使用可能不存在的Agent函数。
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.llms import VLLMOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Optional, Any, Dict


def build_agent(llm: VLLMOpenAI, tools: List) -> Any:
    """
    构建一个简单的工具调用处理器
    
    Args:
        llm: 语言模型实例
        tools: 工具列表
        
    Returns:
        一个简单的处理器函数
    """
    # 构建工具描述
    tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])
    
    # 构建Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"你是一个智能AI助手，能够帮助用户完成各种任务。你可以使用以下工具来获取信息和执行操作：\n{tools_description}\n请根据用户的需求，提供详细的回答。"),
        ("human", "{input}")
    ])
    
    # 创建一个简单的处理链
    chain = (
        RunnablePassthrough.assign(
            input=lambda x: x["input"]
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain


def create_chat_prompt(input_text: str, chat_history: List = None) -> ChatPromptTemplate:
    """
    创建聊天提示模板
    
    Args:
        input_text: 用户输入
        chat_history: 聊天历史
        
    Returns:
        聊天提示模板
    """
    messages = []
    
    # 添加系统消息
    messages.append(("system", "你是一个智能AI助手，能够帮助用户完成各种任务。"))
    
    # 添加聊天历史
    if chat_history:
        for message in chat_history:
            if isinstance(message, HumanMessage):
                messages.append(("human", message.content))
            else:
                messages.append(("ai", message.content))
    
    # 添加用户输入
    messages.append(("human", input_text))
    
    return ChatPromptTemplate.from_messages(messages)
