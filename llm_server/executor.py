#!/usr/bin/env python3
"""
通用Python脚本执行器
参数: 脚本名称 [脚本参数...]
"""
import sys
import os
import subprocess
import json
from pathlib import Path

def execute_python_script(script_path, *args):
    """
    执行指定路径的Python脚本
    """
    # 获取项目根目录（从当前脚本位置向上一级）
    current_dir = Path(__file__).parent.parent  # 回到项目根目录
    script_full_path = current_dir / script_path
    
    if not script_full_path.exists():
        return f"错误: 脚本 '{script_path}' 不存在"
    
    if script_full_path.suffix != '.py':
        return f"错误: 文件必须是Python脚本 (.py文件)"
    
    try:
        # 构建命令：工具名称作为脚本的第一个参数
        cmd = [sys.executable, str(script_full_path)] + list(args)
        print("cmd:", cmd)
        
        # 执行Python脚本，指定编码为UTF-8
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(current_dir),
            encoding='utf-8',      # 明确指定UTF-8编码
            errors='replace'       # 遇到编码错误时替换字符
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"脚本执行失败: {result.stderr.strip()}"
    
    except subprocess.TimeoutExpired:
        return f"脚本执行超时: {script_path}"
    except Exception as e:
        return f"执行脚本时出错: {str(e)}"

def list_available_tools():
    """
    从配置文件中列出所有可用工具
    """
    config_path = Path(__file__).parent / "tools_config.json"
    
    if not config_path.exists():
        return []
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('tools', [])
    except Exception as e:
        return []

def get_tool_by_name(tool_name):
    """
    根据工具名称获取工具信息
    """
    tools = list_available_tools()
    for tool in tools:
        if tool['name'] == tool_name:
            return tool
    return None

def main():
    if len(sys.argv) < 2:
        print("用法: python executor.py <工具名称> [参数...]")
        print("\n可用工具:")
        tools = list_available_tools()
        for tool in tools:
            params = ", ".join([p['name'] for p in tool['parameters']]) if tool['parameters'] else "无参数"
            print(f"  - {tool['name']}: {tool['description']} (参数: {params})")
        sys.exit(1)
    
    tool_name = sys.argv[1]
    tool_args = sys.argv[2:]
    
    # 根据工具名称查找对应的脚本路径
    tool_info = get_tool_by_name(tool_name)
    if not tool_info:
        print(f"错误: 未找到工具 '{tool_name}'")
        sys.exit(1)
    
    script_path = tool_info['path']
    
    # 修复：将工具名称作为脚本的第一个参数
    # 需要将工具名称添加到参数列表前面
    result = execute_python_script(script_path, tool_name, *tool_args)
    print(result)

if __name__ == "__main__":
    main()