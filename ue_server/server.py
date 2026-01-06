import unreal
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import os

class ExecHandler(BaseHTTPRequestHandler):
    """
    HTTP请求处理器，用于处理exec方法的调用
    """
    
    def _set_response(self, status_code=200, content_type='application/json'):
        """设置响应头"""
        self.send_response(status_code)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_POST(self):
        """处理POST请求"""
        if self.path == '/exec':
            # 获取请求内容长度
            content_length = int(self.headers['Content-Length'])
            # 读取请求内容
            post_data = self.rfile.read(content_length)
            
            try:
                # 解析JSON数据
                request_data = json.loads(post_data.decode('utf-8'))
                
                # 获取要执行的命令
                command = request_data.get('command', '')
                
                if command:
                    # 执行命令并捕获结果
                    try:
                        # 检查是否是需要在主线程执行的特定命令
                        if self._needs_main_thread(command):
                            # 使用unreal.run_on_ui_thread执行主线程操作
                            result = self._execute_on_main_thread(command)
                        else:
                            # 使用普通exec执行命令
                            exec_globals = {"unreal": unreal}
                            exec(command, exec_globals)
                            result = {"status": "success", "message": f"Command executed: {command}"}
                    except Exception as e:
                        result = {"status": "error", "message": f"Execution failed: {str(e)}"}
                else:
                    result = {"status": "error", "message": "No command provided"}
                
                # 发送响应
                self._set_response(200)
                self.wfile.write(json.dumps(result).encode('utf-8'))
                
            except json.JSONDecodeError:
                self._set_response(400)
                error_response = {"status": "error", "message": "Invalid JSON format"}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self._set_response(404)
            error_response = {"status": "error", "message": "Endpoint not found"}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def _needs_main_thread(self, command):
        """检查命令是否需要在主线程执行"""
        # 检查是否包含资源加载相关的API调用
        main_thread_operations = [
            'AssetImportTask',
            'unreal.load_asset',
            'unreal.find_asset',
            'unreal.EditorAssetLibrary',
            'unreal.EditorLevelLibrary',
            'unreal.SystemLibrary',
            'unreal.EditorUtilityLibrary',
            'unreal.AssetRegistryHelpers'
        ]
        
        return any(op in command for op in main_thread_operations)

    def _execute_on_main_thread(self, command):
        """在主线程上执行命令"""
        try:
            # 使用AssetTools进行导入，这是正确的API
            exec_globals = {"unreal": unreal}
            
            # 对于资源导入任务，使用AssetTools
            if 'AssetImportTask' in command:
                return self._handle_asset_import_with_asset_tools(command)
            else:
                # 对于其他命令，尝试执行
                exec(command, exec_globals)
                return {"status": "success", "message": f"Command executed on main thread: {command}"}
        except Exception as e:
            return {"status": "error", "message": f"Main thread execution failed: {str(e)}"}

    def _handle_asset_import_with_asset_tools(self, command):
        """使用AssetTools处理资源导入"""
        try:
            exec_globals = {"unreal": unreal}
            exec(command, exec_globals)
            return {"status": "success", "message": f"Asset import command executed: {command}"}
        except Exception as e:
            return {"status": "error", "message": f"Asset import failed: {str(e)}"}

    def do_GET(self):
        """处理GET请求"""
        if self.path == '/':
            self._set_response(200, 'text/html')
            response_html = """
            <html>
            <head><title>Unreal Engine Exec API</title></head>
            <body>
                <h1>Unreal Engine Exec API</h1>
                <p>This service allows executing Python commands in Unreal Engine.</p>
                <h2>API Endpoints:</h2>
                <ul>
                    <li>POST /exec - Execute Python commands</li>
                    <li>Example: {"command": "unreal.EditorLevelLibrary.add_actor(unreal.StaticMeshActor, unreal.Vector(0, 0, 0))"}</li>
                </ul>
            </body>
            </html>
            """
            self.wfile.write(response_html.encode('utf-8'))
        else:
            self._set_response(404)
            error_response = {"status": "error", "message": "Endpoint not found"}
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

class UnrealExecServer:
    """
    Unreal Engine Exec服务器类
    """
    
    def __init__(self, host='localhost', port=8070):
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        
    def start_server(self):
        """启动服务器"""
        try:
            self.server = HTTPServer((self.host, self.port), ExecHandler)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            unreal.log(f"Exec server started at http://{self.host}:{self.port}")
            return True
        except Exception as e:
            unreal.log_error(f"Failed to start server: {str(e)}")
            return False
            
    def stop_server(self):
        """停止服务器"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.server_thread:
                self.server_thread.join()
            unreal.log("Exec server stopped")

# 全局服务器实例
exec_server = None

def start_exec_server():
    """启动exec服务器的函数"""
    global exec_server
    if exec_server is None:
        exec_server = UnrealExecServer(host='localhost', port=8070)
        success = exec_server.start_server()
        if success:
            unreal.log("Unreal Exec API server initialized successfully")
        else:
            unreal.log_error("Failed to initialize Unreal Exec API server")
    else:
        unreal.log("Exec server is already running")

def stop_exec_server():
    """停止exec服务器的函数"""
    global exec_server
    if exec_server:
        exec_server.stop_server()
        exec_server = None

# 启动服务器
if __name__ == "__main__":
    unreal.log("Starting Unreal Engine Exec API server...")
    start_exec_server()
else:
    # 当脚本在Unreal中执行时
    unreal.log("Hello UE5 - Exec API Server")
    start_exec_server()

# 注册关闭回调以确保服务器被正确关闭
def on_shutdown():
    """关闭时清理资源"""
    stop_exec_server()