# api.py
from flask import Flask, request, jsonify
import json
import logging
from llm_server import LLMService, VLMService

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# 创建服务实例
llm_service = LLMService()
vlm_service = VLMService()

logger = logging.getLogger(__name__)

@app.route('/chat', methods=['GET', 'POST'])
def chat_endpoint():
    """
    name:messages
     name:sender
    """
    try:
        logger.info("收到聊天请求")
        
        if request.method == 'GET':
            # 从查询参数中获取 messages 和 tools
            messages_json = request.args.get('messages', '[]')
            sender = request.args.get('sender', None)
            tools_json = request.args.get('tools', None)
            
            # 解析 JSON 字符串
            if sender:
                messages = [{
                    "role": sender,
                    "content": messages_json,
                }]
            else:
                messages = json.loads(messages_json) if messages_json else []
            tools = json.loads(tools_json) if tools_json else None
            
        elif request.method == 'POST':
            # 从请求体中获取数据
            data = request.get_json()
            if not data:
                logger.error("POST请求缺少JSON数据")
                return jsonify({"error": "请求体必须包含JSON数据"}), 400
                
            messages = data.get('messages', [])
            tools = data.get('tools', None)
        
        logger.debug(f"请求参数 - 消息数量: {len(messages)}, 工具数量: {len(tools) if tools else 0}")
        
        # 调用 LLMService 的 create 方法
        result = llm_service.create(messages, tools)
        logger.info("聊天请求处理完成")
        return jsonify(result)
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        return jsonify({"error": f"JSON解析失败: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"聊天请求处理失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/vlm/chat', methods=['GET'])
def vlm_chat_get_endpoint():
    """
    name:message
    name:image_source
    """
    try:
        logger.info("收到VLM聊天GET请求")
        
        # 从查询参数中获取数据
        messages_json = request.args.get('messages', '[]')
        image_source = request.args.get('image_source', None)
        tools_json = request.args.get('tools', None)
        
        # 解析 JSON 字符串
        try:
            messages = json.loads(messages_json) if messages_json else []
            tools = json.loads(tools_json) if tools_json else None
        except json.JSONDecodeError as e:
            logger.error(f"查询参数JSON解析失败: {str(e)}")
            return jsonify({"error": f"查询参数JSON解析失败: {str(e)}"}), 400
        
        logger.debug(f"VLM请求参数 - 消息数量: {len(messages)}, 包含图像: {image_source is not None}, 工具数量: {len(tools) if tools else 0}")
        
        # 调用 VLMService 的 create_with_image 方法
        result = vlm_service.create_with_image(messages, image_source, tools)
        logger.info("VLM聊天GET请求处理完成")
        return jsonify(result)
    except Exception as e:
        logger.error(f"VLM聊天GET请求处理失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    logger.info("健康检查请求")
    return jsonify({
        "status": "healthy",
        "message": "LLM/VLM服务运行正常",
        "models": {
            "llm": llm_service.model_name,
            "vlm": vlm_service.model_name
        }
    })

@app.route('/routes', methods=['GET'])
def show_routes():
    """
    显示所有可用路由
    """
    routes = []
    for rule in app.url_map.iter_rules():
        # 获取端点函数的docstring
        endpoint_func = app.view_functions.get(rule.endpoint)
        docstring = endpoint_func.__doc__.strip() if endpoint_func and endpoint_func.__doc__ else "无描述"
        
        routes.append({
            'endpoint': rule.endpoint,
            'methods': sorted(list(rule.methods - {'HEAD'})),  # 排除HEAD方法
            'rule': str(rule),
            'description': docstring
        })
    
    # 按路由规则排序
    routes.sort(key=lambda x: x['rule'])
    
    return jsonify({'routes': routes})

def create_app():
    """创建Flask应用实例"""
    return app

if __name__ == '__main__':
    # 如果直接运行此文件，需要获取端口参数
    import argparse
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001, help='Port to run the server on')
    parser.add_argument('--log-file', help='日志文件路径')
    args, unknown = parser.parse_known_args()

    # 配置日志
    if args.log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(args.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO)

    app.run(host='0.0.0.0', port=args.port, debug=True)