import bpy
import sys
import json
import ast
from http.server import BaseHTTPRequestHandler
import threading
import socket
from io import StringIO
import select
import errno
from urllib.parse import urlparse, parse_qs

class BlenderAPIHandler:
    def __init__(self, request_data, client_socket, address):
        self.request_data = request_data
        self.client_socket = client_socket
        self.address = address
        
        # 解析请求
        self.parse_request()
    
    def parse_request(self):
        """解析HTTP请求"""
        request_str = self.request_data.decode('utf-8')
        lines = request_str.split('\r\n')
        
        # 解析请求行
        request_line = lines[0]
        parts = request_line.split(' ')
        self.method = parts[0]
        self.path = parts[1]
        
        # 解析头部
        self.headers = {}
        i = 1
        while i < len(lines) and lines[i] != '':
            header_line = lines[i]
            if ':' in header_line:
                key, value = header_line.split(':', 1)
                self.headers[key.strip().lower()] = value.strip()
            i += 1
        
        # 提取请求体
        body_start = request_str.find('\r\n\r\n') + 4
        self.body = request_str[body_start:] if body_start > 3 else ''
    
    def handle_request(self):
        """处理请求"""
        if self.method == 'POST':
            if self.path == '/api/eval':
                self.handle_eval()
            elif self.path == '/api/exec':
                self.handle_exec()
   
            else:
                self.send_response(404, {"status": "error", "message": "Endpoint not found"})
        else:
            self.send_response(405, {"status": "error", "message": "Method not allowed"})
    
    def handle_eval(self):
        """处理eval请求"""
        try:
            request_data = json.loads(self.body)
            code_string = request_data.get('code', '')
            
            if not code_string:
                response = {"status": "error", "message": "No code provided"}
            else:
                # 执行代码并捕获结果
                result = self.execute_blender_code(code_string)
                response = {"status": "success", "result": result}
                
        except json.JSONDecodeError:
            response = {"status": "error", "message": "Invalid JSON format"}
        except Exception as e:
            response = {"status": "error", "message": str(e)}
        
        self.send_response(200, response)
    
    def handle_exec(self):
        """处理exec请求"""
        try:
            request_data = json.loads(self.body)
            code_string = request_data.get('code', '')
            
            if not code_string:
                response = {"status": "error", "message": "No code provided"}
            else:
                # 使用exec执行代码块
                result = self.execute_blender_code_exec(code_string)
                response = {"status": "success", "result": result}
                
        except json.JSONDecodeError:
            response = {"status": "error", "message": "Invalid JSON format"}
        except Exception as e:
            response = {"status": "error", "message": str(e)}
        
        self.send_response(200, response)
    

    
    def execute_blender_code(self, code_string):
        """
        执行Blender代码字符串（用于单个表达式）
        """
        try:
            # 验证语法
            try:
                parsed = ast.parse(code_string, mode='eval')
            except SyntaxError as e:
                raise Exception(f"Syntax error in code: {str(e)}")
            
            # 完全开放的执行环境
            result = eval(code_string)
            return result
            
        except Exception as e:
            raise Exception(f"Error executing code: {str(e)}")
    
    def execute_blender_code_exec(self, code_string):
        """
        使用exec执行Blender代码块
        """
        try:
            # 验证语法
            try:
                parsed = ast.parse(code_string)
            except SyntaxError as e:
                raise Exception(f"Syntax error in code: {str(e)}")
            
            # 捕获print输出
            captured_output = StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured_output
            
            try:
                # 完全开放的执行环境
                exec(code_string)
            finally:
                # 恢复标准输出
                sys.stdout = old_stdout
            
            # 返回捕获的输出
            output = captured_output.getvalue()
            return output if output else "Code executed successfully"
            
        except Exception as e:
            raise Exception(f"Error executing code: {str(e)}")
    
    def send_response(self, status_code, data):
        """发送HTTP响应"""
        try:
            response_body = json.dumps(data, default=str)
            response_headers = [
                f"HTTP/1.1 {status_code} OK",
                "Content-Type: application/json",
                f"Content-Length: {len(response_body)}",
                "Connection: close",
                "\r\n"
            ]
            response_str = "\r\n".join(response_headers) + response_body
            
            self.client_socket.send(response_str.encode())
        except Exception as e:
            print(f"Error sending response: {e}")

class BlenderHTTPServer:
    """
    纯socket实现的HTTP服务器
    """
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
    def start(self):
        """启动服务器"""
        if self.running:
            print("Server already running")
            return False
            
        try:
            # 创建非阻塞socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.socket.setblocking(False)  # 设置为非阻塞模式
            
            self.running = True
            print(f"Blender API server running on {self.host}:{self.port}")
            
            # 注册定时器函数
            bpy.app.timers.register(self._handle_requests, persistent=True)
            
            return True
        except Exception as e:
            print(f"Failed to start server: {e}")
            return False
    
    def _handle_requests(self):
        """处理HTTP请求的定时器函数"""
        if not self.running:
            return None  # 停止定时器
            
        try:
            # 检查是否有新的连接
            ready, _, _ = select.select([self.socket], [], [], 0)  # 非阻塞检查
            
            if ready:
                client_socket, address = self.socket.accept()
                
                # 设置客户端socket为非阻塞模式
                client_socket.setblocking(False)
                
                # 读取请求数据
                request_data = self._read_request(client_socket)
                
                if request_data:
                    # 创建处理线程
                    thread = threading.Thread(
                        target=self._process_request_data,
                        args=(request_data, client_socket, address)
                    )
                    thread.daemon = True
                    thread.start()
                
        except socket.error as e:
            if e.errno != errno.EAGAIN and e.errno != errno.EWOULDBLOCK:
                print(f"Socket error: {e}")
        except Exception as e:
            print(f"Error handling requests: {e}")
        
        # 每0.1秒调用一次
        return 0.1
    
    def _read_request(self, client_socket):
        """读取HTTP请求数据"""
        try:
            # 首先尝试接收数据
            data = b""
            while True:
                try:
                    chunk = client_socket.recv(1024)
                    if not chunk:
                        break
                    data += chunk
                    
                    # 检查是否接收到了完整的HTTP头部
                    if b'\r\n\r\n' in data:
                        # 解析Content-Length头部
                        header_end = data.find(b'\r\n\r\n')
                        headers = data[:header_end].decode('utf-8')
                        
                        content_length = 0
                        for line in headers.split('\r\n'):
                            if line.lower().startswith('content-length:'):
                                content_length = int(line.split(':')[1].strip())
                                break
                        
                        # 检查是否接收了完整的请求体
                        body_start = header_end + 4
                        if len(data) >= body_start + content_length:
                            break
                except socket.error as e:
                    if e.errno == errno.EWOULDBLOCK or e.errno == errno.EAGAIN:
                        # 非阻塞模式下暂时没有数据
                        break
                    else:
                        raise e
            
            return data
        except Exception as e:
            print(f"Error reading request: {e}")
            return None
    
    def _process_request_data(self, request_data, client_socket, address):  
        """处理请求数据"""  
        try:  
            # 创建处理器  
            handler = BlenderAPIHandler(request_data, client_socket, address)  
            
            # 在主线程上调度执行  
            def execute_on_main_thread():  
                handler.handle_request()  
                return None  # 停止定时器  
            
            bpy.app.timers.register(execute_on_main_thread, first_interval=0.0)  
            
        except Exception as e:  
            print(f"Error processing request: {e}")
    
    def stop(self):
        """停止服务器"""
        self.running = False
        if self.socket:
            self.socket.close()
            self.socket = None
        print("Server stopped")

# 全局服务器实例
blender_timer_server = None

def start_blender_server_with_timer(port=8080):
    """启动使用Blender定时器的服务器"""
    global blender_timer_server
    if blender_timer_server is None:
        blender_timer_server = BlenderHTTPServer(port=port)
    return blender_timer_server.start()

def stop_blender_server_timer():
    """停止使用定时器的服务器"""
    global blender_timer_server
    if blender_timer_server:
        blender_timer_server.stop()
        blender_timer_server = None

def is_server_running_timer():
    """检查使用定时器的服务器是否正在运行"""
    global blender_timer_server
    return blender_timer_server is not None and blender_timer_server.running

def init_blender_server():
    """在Blender中初始化服务器（使用定时器方式）"""
    success = start_blender_server_with_timer(8080)
    if success:
        print("Blender server initialized with timer and running in background")
        return True
    else:
        print("Failed to initialize Blender server")
        return False

# Blender插件兼容代码
def register():
    """注册到Blender（用于插件）"""
    init_blender_server()

def unregister():
    """从Blender取消注册（用于插件）"""
    stop_blender_server_timer()

if __name__ == "__main__":
    # 直接运行时，模拟Blender环境
    print("Starting server in timer mode...")
    start_blender_server_with_timer(8080)
    
    # 保持运行
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        stop_blender_server_timer()