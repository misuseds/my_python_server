# e:\code\my_python_server\sse\sse_server.py
from mcp import ClientSession
from mcp.client.sse import sse_client
import asyncio
from flask import Flask, request, jsonify

app = Flask(__name__)

async def read_excel_data(filepath, sheet_name=None):
    """通过 SSE 连接到 MCP 服务器并读取 Excel 数据"""
    url = "http://localhost:8017/sse"
    
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            print("已连接到 MCP 服务器")
            
            # 处理 sheet_name 参数
            # 如果没有提供 sheet_name，则使用默认值 "Sheet1"
            if sheet_name is None:
                sheet_name = "Sheet1"
            
            # 读取数据
            result = await session.call_tool(
                "read_data_from_excel",
                arguments={
                    "filepath": filepath,
                    "sheet_name": sheet_name,
                    "start_cell": "A1",
                    "preview_only": True
                }
            )
            print("已读取数据：", result)
            # 将结果转换为可序列化的格式
            return convert_result_to_serializable(result)

def convert_result_to_serializable(result):
    """将 CallToolResult 转换为可序列化的字典格式，并避免重复内容"""
    if hasattr(result, '__dict__'):
        # 如果结果对象有 __dict__ 属性，尝试转换其属性
        serializable = {}
        for key, value in result.__dict__.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                serializable[key] = value
            elif isinstance(value, (list, tuple)):
                serializable[key] = [convert_result_to_serializable(item) if hasattr(item, '__dict__') else item for item in value]
            elif isinstance(value, dict):
                serializable[key] = {k: convert_result_to_serializable(v) if hasattr(v, '__dict__') else v for k, v in value.items()}
            else:
                # 对于其他对象，转换为字符串表示
                serializable[key] = str(value)
        
        # 特殊处理 content 和 structuredContent 字段，避免重复数据
        if 'content' in serializable and 'structuredContent' in serializable:
            # 如果 structuredContent 中的 result 与 content 的 text 相同，则移除其中一个
            if (isinstance(serializable['content'], list) and 
                len(serializable['content']) > 0 and
                isinstance(serializable['content'][0], dict) and
                'text' in serializable['content'][0] and
                isinstance(serializable['structuredContent'], dict) and
                'result' in serializable['structuredContent']):
                
                # 比较内容是否重复
                if serializable['content'][0]['text'] == serializable['structuredContent']['result']:
                    # 移除重复的 structuredContent，只保留 content
                    del serializable['structuredContent']
        
        return serializable
    else:
        return str(result)

async def list_available_tools():
    """通过 SSE 连接到 MCP 服务器并列出所有可用工具"""
    url = "http://localhost:8017/sse"
    
    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化连接
            await session.initialize()
            print("已连接到 MCP 服务器")
            
            # 列出所有可用工具
            tools = await session.list_tools()
            return [{"name": tool.name, "description": tool.description} for tool in tools.tools]

@app.route('/read_excel', methods=['GET'])
def read_excel_endpoint():
    """Flask 路由，接受 filepath 和可选的 sheet_name 参数并返回读取结果"""
    filepath = request.args.get('filepath')
    sheet_name = request.args.get('sheet_name')
    
    if not filepath:
        return jsonify({"error": "缺少 filepath 参数"}), 400
    
    try:
        # 在异步环境中运行
        result = asyncio.run(read_excel_data(filepath, sheet_name))
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/list_tools', methods=['GET'])
def list_tools_endpoint():
    """Flask 路由，列出所有可用的 MCP 工具"""
    try:
        # 在异步环境中运行
        tools = asyncio.run(list_available_tools())
        return jsonify({"tools": tools})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)