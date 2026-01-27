import sys
import asyncio
import os
import json
import threading
import queue
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# 添加当前目录到路径,支持直接运行
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit, QVBoxLayout,
    QWidget, QLabel, QDialog, QScrollArea, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread, QMetaObject, pyqtSlot, QEvent
from PyQt6.QtGui import QKeySequence, QShortcut, QColor
import importlib.util

# 导入拆分出来的组件
from components.content_extractor import ContentExtractor
from components.tools_dialog import ToolsDialog
from components.tool_loader import ToolLoader


# 以下是MCPAICaller类的定义


class MCPAICaller(QMainWindow):
    """MCP AI调用器主窗口类"""

    # 定义信号用于线程间通信
    add_caption_signal = pyqtSignal(str)  # 添加字幕信号

    # 工具调用事件流信号
    tool_call_start_signal = pyqtSignal(str, dict)  # 工具调用开始 (tool_name, arguments)
    tool_call_end_signal = pyqtSignal(str, dict, str)  # 工具调用结束 (tool_name, arguments, result)
    tool_result_signal = pyqtSignal(str, str)  # 工具结果 (tool_call_id, result)
    text_delta_signal = pyqtSignal(str)  # 文本增量 (text)
    text_end_signal = pyqtSignal(str)  # 文本结束 (full_text)
    message_end_signal = pyqtSignal(str)  # 消息结束 (role)
    lifecycle_signal = pyqtSignal(str, str)  # 生命周期事件 (event_type, data)

    def __init__(self):
        """初始化MCP AI调用器"""
        super().__init__()

        # 初始化变量
        self._initialize_variables()

        # 设置定时器
        self._setup_timers()

        # 设置窗口
        self._setup_window()

        # 初始化服务
        self._initialize_services()

        # 初始化客户端
        self._initialize_clients()

        # 设置UI
        self._setup_ui()

        # 设置快捷键
        self._setup_shortcuts()

        # 设置信号连接
        self._setup_connections()

        # 初始化自我监控
        self._initialize_self_monitoring()

        # 异步加载MCP工具
        self.async_refresh_tools_list()

        print("[初始化] 完成!")
        print("[启动] MCP AI Caller 已启动")

    def _initialize_variables(self):
        """初始化变量"""
        print("[初始化] 设置初始变量...")

        # 存储聊天历史
        self.chat_history = []

        # 存储最后一次用户输入的内容
        self.last_user_input = ""

        # 存储VLM分析结果
        self.vlm_results = []

        # 存储LLM回复
        self.llm_responses = []

        # 存储用户输入历史（保留用于记录）
        self.user_input_history = []

        # 最近一次记录的游戏状态
        self.last_game_state = ""

        # 存储分割后的内容
        self.split_contents = []

        # 存储工具列表 (all_tools_mapping: 字典格式 {"1": {...}, "2": {...}, ...})
        self.tools_list = {}
        # 存储按服务器分组的工具列表
        self.tools_by_server = {}

        # 存储对话历史，用于API调用
        self.conversation_history = []

        # 工具加载状态
        self.tools_loading = False

        # 工具调用状态
        self.tool_calling = False

        # 工具加载器
        self.tool_loader = None

        # 自我监控线程
        self.self_monitoring_thread = None

        # 工具是否已经加载过
        self.tools_loaded_once = False

        # 最后一次工具加载时间
        self.last_tools_load_time = 0

        # 工具加载间隔（秒）
        self.tools_load_interval = 60

        # 工具策略控制（类似Clawdbot）
        self.tools_policy = {
            'allow': [],  # 允许的工具列表（空表示全部允许）
            'deny': [],   # 禁止的工具列表（优先级高于allow）
            'by_provider': {},  # 按提供商的工具策略
            'sandbox': {}  # 沙箱环境下的额外限制
        }
        self.current_policy_mode = 'default'  # default, sandbox, restricted

        # 工具状态管理
        self.tool_state = 'initial'  # 'initial' (初始状态) 或 'active' (工具状态)
        self.tool_state_history = []  # 记录工具状态变化

    def _setup_timers(self):
        """设置定时器"""
        print("[初始化] 设置定时器...")

        # 设置定时器，每30秒自动记录游戏状态
        self.auto_record_timer = QTimer(self)
        self.auto_record_timer.timeout.connect(self.auto_record_game_state)
        self.auto_record_timer.start(30000)  # 30秒

        # 设置定时器，每1秒自动截图并调用VLM
        self.auto_screenshot_timer = QTimer(self)
        self.auto_screenshot_timer.timeout.connect(self.auto_screenshot_and_vlm)
        self.auto_screenshot_timer.start(1000)  # 1秒

        # 存储VLM回复的列表
        self.vlm_responses = []
        # 最大存储的VLM回复数量
        self.max_vlm_responses = 3

    def _setup_window(self):
        """设置窗口属性"""
        print("[初始化] 设置窗口...")

        # 设置无边框窗口和置顶
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)

        # 设置透明背景属性
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 获取屏幕尺寸
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()

        # 计算窗口位置，放在屏幕下方
        window_width = 1200
        window_height = 100
        x = (screen_geometry.width() - window_width) // 2
        y = screen_geometry.height() - window_height - 50  # 距离底部50px

        # 设置窗口位置和大小
        self.setGeometry(x, y, window_width, window_height)
        self.min_height = 180
        self.max_height = 500

        # 设置窗口不透明度为1.0，使其完全可见
        self.setWindowOpacity(1.0)

        # 添加调试信息
        print(f"[窗口调试] 窗口位置: ({x}, {y}), 大小: {window_width}x{window_height}")
        print(f"[窗口调试] 窗口是否可见: {self.isVisible()}")

    def _initialize_services(self):
        """初始化LLM和VLM服务"""
        print("[初始化] 初始化服务...")

        # 动态导入本地VLM服务
        llm_spec = importlib.util.spec_from_file_location(
            "llm_class",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "llm_server", "llm_class.py")
        )
        llm_module = importlib.util.module_from_spec(llm_spec)
        llm_spec.loader.exec_module(llm_module)

        # 初始化VLM服务（用于所有AI调用）
        self.VLMService = llm_module.VLMService
        self.vlm_service = self.VLMService()

    def _initialize_clients(self):
        """初始化MCP客户端"""
        print("[初始化] 初始化客户端...")

        # 动态导入MCP客户端
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mcp_client_path = os.path.join(current_dir, "mcp_client.py")
        spec = importlib.util.spec_from_file_location("mcp_client", mcp_client_path)
        mcp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mcp_module)
        self.mcp_client = mcp_module.MCPClient()

        # 初始化工具加载器
        self.tool_loader = ToolLoader(self.mcp_client)
        self.tool_loader.tools_loaded.connect(self.on_tools_loaded)
        self.tool_loader.loading_failed.connect(self.on_tools_loading_failed)

        # 启动时自动加载工具
        self.tools_loaded_once = False

    def _setup_ui(self):
        """设置UI"""
        print("[初始化] 设置UI...")

        # 创建主布局（仅包含字幕显示）
        main_layout = QVBoxLayout()

        # 创建字幕显示区域
        self.caption_display = QTextEdit(self)
        self.caption_display.setReadOnly(True)
        self.caption_display.setFrameShape(QFrame.Shape.NoFrame)
        self.caption_display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.caption_display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # 设置高透明黑色背景的样式（不设置字体样式，在HTML中设置）
        self.caption_display.setStyleSheet(
            """
            QTextEdit {
                background-color: transparent;
                padding: 10px;
            }
            """
        )
        # 不要对caption_display设置WA_TranslucentBackground，否则文本也会透明
        main_layout.addWidget(self.caption_display)

        # 创建主窗口部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        # 设置高透明的黑色背景（更透明）
        central_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0.2);")
        self.setCentralWidget(central_widget)

        # 创建独立的输入窗口
        self.input_window = QWidget()
        self.input_window.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.input_window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 设置输入窗口布局
        input_layout = QVBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(0)

        # 创建输入栏
        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("输入命令，例如：/h 打开工具窗口")
        self.input_line.setFixedHeight(30)
        self.input_line.setStyleSheet(
            "QLineEdit { background-color: white; color: black; font-size: 14px; padding: 5px; border: 1px solid #ccc; border-radius: 4px; }"
        )
        self.input_line.returnPressed.connect(self.on_input_submitted)
        input_layout.addWidget(self.input_line)

        self.input_window.setLayout(input_layout)

        # 设置输入窗口在屏幕左下角（x偏移200px，y离底部10px）
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        input_window_width = 300
        input_window_height = 40
        self.input_window.setGeometry(
            screen_geometry.left() + 10 + 200,  # 左边距 10px，再向右偏移 200px
            screen_geometry.bottom() - input_window_height -2,  # 距离底部 10px
            input_window_width,
            input_window_height
        )
        self.input_window.show()

        # 不显示欢迎信息，保持窗口空白

    def _setup_shortcuts(self):
        """设置快捷键（已禁用）"""
        print("[初始化] 设置快捷键...")
        # 禁用所有键盘快捷键
        pass

    def _setup_connections(self):
        """设置信号连接"""
        print("[初始化] 设置信号连接...")

        # 连接添加字幕信号
        self.add_caption_signal.connect(self.add_caption_line)

        # 连接工具调用事件流信号
        self.tool_call_start_signal.connect(self._on_tool_call_start)
        self.tool_call_end_signal.connect(self._on_tool_call_end)
        self.tool_result_signal.connect(self._on_tool_result)
        self.text_delta_signal.connect(self._on_text_delta)
        self.text_end_signal.connect(self._on_text_end)
        self.message_end_signal.connect(self._on_message_end)
        self.lifecycle_signal.connect(self._on_lifecycle_event)

    def _on_tool_call_start(self, tool_name, arguments):
        """工具调用开始事件处理"""
        print(f"[事件流] 工具调用开始: {tool_name}, 参数: {arguments}")

    def _on_tool_call_end(self, tool_name, arguments, result):
        """工具调用结束事件处理"""
        print(f"[事件流] 工具调用结束: {tool_name}, 结果: {result[:50] if result else 'None'}...")

    def _on_tool_result(self, tool_call_id, result):
        """工具结果事件处理"""
        print(f"[事件流] 工具结果: {tool_call_id}, 结果: {result[:50] if result else 'None'}...")

    def _on_text_delta(self, text):
        """文本增量事件处理"""
        print(f"[事件流] 文本增量: {text[:30]}...")

    def _on_text_end(self, full_text):
        """文本结束事件处理"""
        print(f"[事件流] 文本结束: {full_text[:30]}...")

    def _on_message_end(self, role):
        """消息结束事件处理"""
        print(f"[事件流] 消息结束: {role}")

    def _on_lifecycle_event(self, event_type, data):
        """生命周期事件处理"""
        print(f"[事件流] 生命周期事件: {event_type}, 数据: {data[:50] if data else 'None'}...")

    def _initialize_self_monitoring(self):
        """初始化自我监控线程"""
        print("[初始化] 初始化自我监控...")
        # 简化版本：不使用独立的自我监控线程
        # 直接使用auto_screenshot_and_vlm定时器进行截图和VLM分析
        self.self_monitoring_thread = None
        print("[自我监控] 使用简化版本（直接使用定时器）")

    def async_refresh_tools_list(self):
        """异步刷新工具列表"""
        if not self.tools_loading:
            self.tools_loading = True
            print("[工具加载] 开始加载MCP工具...")
            # 启动工具加载线程
            threading.Thread(target=self._load_tools_list).start()

    def _load_tools_list(self):
        """加载工具列表"""
        try:
            # 调用工具加载器加载工具
            self.tool_loader.load_tools()
        except Exception as e:
            print(f"[错误] 加载工具失败: {e}")
            self.tools_loading = False

    def on_tools_loaded(self, tools):
        """工具加载完成回调"""
        # tools 是一个元组 (all_tools_mapping, tools_by_server)
        all_tools_mapping, tools_by_server = tools

        # 存储完整的工具信息
        self.tools_list = all_tools_mapping
        self.tools_by_server = tools_by_server
        self.tools_loading = False
        self.tools_loaded_once = True
        self.last_tools_load_time = datetime.now().timestamp()

        # 显示工具加载完成信息
        tool_count = len(all_tools_mapping)
        print(f"[工具加载] MCP工具加载完成，共 {tool_count} 个工具")

        # 添加到字幕显示
        self.add_caption_line(f"[系统] MCP工具加载完成，共 {tool_count} 个工具")

        # 打印工具列表
        # all_tools_mapping 是字典格式: {"1": {...}, "2": {...}, ...}
        for tool_id, tool_info in all_tools_mapping.items():
            tool_name = tool_info.get('name', '无名称')
            tool_description = tool_info.get('description', '无描述')
            tool_server = tool_info.get('server', '未知服务器')
            print(f"- [{tool_id}] {tool_name} (服务器: {tool_server}): {tool_description}")

    def on_tools_loading_failed(self, error):
        """工具加载失败回调"""
        self.tools_loading = False
        print(f"[错误] 工具加载失败: {error}")

        # 添加到字幕显示
        self.add_caption_line(f"[错误] 工具加载失败: {error}")

    def open_tools_window(self):
        """打开工具窗口"""
        if not self.tools_loaded_once:
            # 如果工具还没有加载，先加载工具
            self.async_refresh_tools_list()
            QMessageBox.information(self, "提示", "工具正在加载中，请稍候...")
            return

        # 打开工具窗口
        try:
            dialog = ToolsDialog(self.tools_by_server, self.tools_list, self)
            dialog.exec()
        except Exception as e:
            print(f"[错误] 打开工具窗口失败: {e}")

    def add_caption_line(self, text):
        """添加字幕行"""
        try:
            print(f"[UI DEBUG] 尝试添加文本: {text}")
        except UnicodeEncodeError:
            print(f"[UI DEBUG] 尝试添加文本: {repr(text)}")
        # 检查是否在主线程
        if QThread.currentThread() == self.thread():
            try:
                print(f"[UI DEBUG] 当前线程: {QThread.currentThread()}, 主线程: {self.thread()}  ")
                print("[UI DEBUG] 在主线程，直接添加")
            except UnicodeEncodeError:
                pass
            # 确保窗口可见
            self.setWindowOpacity(1.0)
            if not self.isVisible():
                self.show()
            # 在主线程，直接添加
            # 只显示当前回合的内容，不保留历史
            # 白色文字，字号更小
            escaped_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br>')
            html_content = f"""
            <div style="
                color: #ffffff;
                font-size: 22px;
                font-weight: 900;
                font-family: 'Arial Black', 'Impact', 'Microsoft YaHei', sans-serif;
                text-shadow: 0 0 5px rgba(0,0,0,0.8);
            ">{escaped_text}</div>
            """
            self.caption_display.setHtml(html_content)
            # 滚动到底部
            self.caption_display.verticalScrollBar().setValue(self.caption_display.verticalScrollBar().maximum())

            # 添加调试信息
            try:
                print("[UI DEBUG] 文本已添加，只显示当前回合内容")
                print(f"[UI DEBUG] caption_display文本: {self.caption_display.toPlainText()[:50]}...")
                print(f"[UI DEBUG] caption_display是否可见: {self.caption_display.isVisible()}")
                print(f"[UI DEBUG] 主窗口是否可见: {self.isVisible()}")
            except UnicodeEncodeError:
                pass

            # 取消之前的定时器（如果存在）
            if hasattr(self, 'caption_timer') and self.caption_timer:
                self.caption_timer.stop()

            # 设置定时器，30秒后清空字幕
            self.caption_timer = QTimer(self)
            self.caption_timer.setSingleShot(True)
            self.caption_timer.timeout.connect(self.clear_caption)
            self.caption_timer.start(30000)  # 30秒后清空
        else:
            try:
                print(f"[UI DEBUG] 当前线程: {QThread.currentThread()}, 主线程: {self.thread()}  ")
                print("[UI DEBUG] 不在主线程，使用信号转发")
            except UnicodeEncodeError:
                pass
            # 不在主线程，使用信号转发
            self.add_caption_signal.emit(text)

    def clear_caption(self):
        """清空字幕显示区域"""
        print("[UI DEBUG] 清空字幕显示")
        if hasattr(self, 'caption_display') and self.caption_display:
            self.caption_display.setPlainText("")
        else:
            print("[UI DEBUG] caption_display不存在，跳过清空")

    def on_input_submitted(self):
        """处理用户输入"""
        input_text = self.input_line.text().strip()
        if not input_text:
            return

        # 清空输入栏
        self.input_line.clear()

        # 处理命令
        if input_text.startswith('/'):
            command = input_text[1:].lower()
            if command == 'h':
                # 打开工具窗口显示工具列表
                self.open_tools_window()
                self.add_caption_line(f"[命令] 打开工具窗口")
            elif command.startswith('r '):
                # /r 命令：直接发送给 LLM 执行（支持 function calling）
                query = command[2:].strip()
                if query:
                    # 无论什么命令，都调用LLM服务
                    # 但对于Blender相关命令，确保使用完全独立的进程
                    self._send_user_input_to_llm_with_tools(query)
                else:
                    self.add_caption_line(f"[命令] 用法: /r <问题描述>")
            else:
                self.add_caption_line(f"[命令] 未知命令: {input_text}")
        else:
            # 处理普通文本输入
            self.add_caption_line(f"[你] {input_text}")

            # 将用户输入添加到用户输入历史中，稍后与VLM信息一起发送给LLM
            self.user_input_history.append(input_text)
            print(f"[用户输入] 已添加到历史记录: {input_text[:30]}...")

            # 将用户输入添加到自我监控线程的历史记录中
            if hasattr(self, 'self_monitoring_thread') and self.self_monitoring_thread:
                self.self_monitoring_thread.add_user_input(input_text)
                print(f"[用户输入] 已添加到自我监控线程历史记录: {input_text[:30]}...")

    def _start_blender_directly(self):
        """直接启动Blender，不依赖LLM服务"""
        try:
            import subprocess
            import os
            import sys
            
            blender_path = r"D:\blender\blender.exe"
            
            if not os.path.exists(blender_path):
                error_msg = f"错误: 找不到Blender可执行文件: {blender_path}"
                print(error_msg)
                self.add_caption_line(f"[错误] {error_msg}")
                return
            
            print("正在启动Blender...")
            self.add_caption_line("[AI] 正在启动Blender...")
            
            if sys.platform == 'win32':
                # Windows平台使用CREATE_NEW_PROCESS_GROUP
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                DETACHED_PROCESS = 0x00000008
                
                # 在新的命令提示符窗口中启动Blender，完全独立
                subprocess.Popen(
                    ['start', 'cmd', '/k', blender_path], 
                    shell=True,
                    creationflags=CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS,
                    close_fds=True
                )
            else:
                # 其他平台
                subprocess.Popen(
                    [blender_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    close_fds=True
                )
            
            success_msg = f"Blender已在完全独立的进程中启动: {blender_path}"
            print(success_msg)
            self.add_caption_line(f"[AI] {success_msg}")
            
        except Exception as e:
            error_msg = f"启动Blender时出错: {e}"
            print(error_msg)
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            self.add_caption_line(f"[错误] {error_msg}")

    def _send_user_input_to_llm(self, input_text):
        """将用户输入发送给VLM处理"""
        import time
        start_time = time.time()
        print(f"[VLM] 处理用户输入: {input_text[:30]}...")
        
        # 将用户输入添加到对话历史
        self.conversation_history.append({
            'role': 'user',
            'content': input_text
        })
        
        # 构建messages格式，用于VLM服务
        messages = []
        
        # 添加系统消息
        messages.append({
            'role': 'system',
            'content': '你是一个智能助手，根据对话历史回答用户的问题'
        })
        
        # 添加最近的对话历史（最多10条）
        recent_history = self.conversation_history[-10:]
        for item in recent_history:
            messages.append(item)
        
        # 调用VLM服务
        try:
            vlm_start_time = time.time()
            response = self.vlm_service.create_with_image(messages, image_source=None)
            vlm_elapsed_time = time.time() - vlm_start_time
            print(f"[VLM] 调用完成，耗时: {vlm_elapsed_time:.2f}秒")
            
            if response:
                # 提取回复内容
                content = response['choices'][0]['message']['content']
                print(f"[VLM] 生成回复完成: {content[:30]}...")
                
                # 将VLM回复添加到对话历史
                self.conversation_history.append({
                    'role': 'assistant',
                    'content': content
                })
                
                # 显示在主窗口中
                self.add_caption_line(f"[AI] {content}")
        except Exception as e:
            print(f"[错误] 调用VLM服务失败: {e}")
            self.add_caption_line(f"[错误] 处理请求失败: {e}")
        finally:
            total_elapsed_time = time.time() - start_time
            print(f"[VLM处理] 完成，总耗时: {total_elapsed_time:.2f}秒")

    def _send_user_input_to_llm_with_tools(self, input_text):
        """将用户输入发送给VLM处理（支持 function calling）"""
        import time
        start_time = time.time()
        print(f"[VLM] 处理用户输入（带工具）: {input_text[:30]}...")

        # 检查工具列表是否已加载
        if not self.tools_loaded_once or len(self.tools_list) == 0:
            print("[警告] 工具列表未加载，正在重新加载...")
            self.add_caption_line("[系统] 工具列表未加载，正在重新加载...")
            # 同步加载工具
            self._load_tools_list()
            # 等待工具加载完成
            import time
            wait_time = 0
            max_wait_time = 10
            while not self.tools_loaded_once and wait_time < max_wait_time:
                time.sleep(1)
                wait_time += 1
            if not self.tools_loaded_once:
                print("[错误] 工具加载超时，无法使用function calling")
                self.add_caption_line("[错误] 工具加载超时，无法使用function calling")
                return

        # 构建工具列表格式（OpenAI function calling 格式）
        tools = self._convert_tools_to_openai_format()

        # 将用户输入添加到对话历史
        self.conversation_history.append({
            'role': 'user',
            'content': input_text
        })

        # 将用户输入添加到自我监控线程的历史记录中，以便吐槽vlm能够使用
        if hasattr(self, 'self_monitoring_thread') and self.self_monitoring_thread:
            self.self_monitoring_thread.add_user_input(input_text)
            print(f"[用户输入] 已添加到自我监控线程历史记录: {input_text[:30]}...")

        # 读取knowledge.txt文件
        knowledge_content = ""
        knowledge_path = os.path.join(os.path.dirname(__file__), 'knowledge.txt')
        try:
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                knowledge_content = f.read().strip()
        except Exception as e:
            print(f"[警告] 读取knowledge.txt失败: {e}")

        # 触发生命周期开始事件
        self.lifecycle_signal.emit('start', input_text)

        # 调用VLM服务（无限循环，支持动态工具状态）
        max_rounds = 10
        max_consecutive_tool_calls = 5  # 最大连续工具调用次数
        current_round = 0
        tool_call_count = 0
        consecutive_tool_calls = 0  # 连续工具调用计数
        last_tool_call = None  # 记录最后一次工具调用，用于检测重复
        
        # 生成工具摘要
        tools_summary = self._generate_tools_summary()
        print(f"[工具状态] 当前状态: {self.tool_state}, 工具摘要: {tools_summary}")
        
        while current_round < max_rounds:
            try:
                # 根据当前状态决定是否传递完整工具描述
                if self.tool_state == 'initial':
                    # 初始状态：不传递完整工具，只传递简短摘要
                    tools_to_pass = None  # 不传递tools参数
                    full_tools_available = False
                    system_prompt = self._build_system_prompt(knowledge_content, tools_summary, full_tools_available)
                    print(f"[工具状态] 初始状态 - 不传递完整工具描述")
                else:
                    # 工具状态：传递完整工具列表
                    tools_to_pass = tools
                    full_tools_available = True
                    system_prompt = self._build_system_prompt(knowledge_content, tools_summary, full_tools_available)
                    print(f"[工具状态] 工具状态 - 传递完整工具描述 ({len(tools)}个工具)")
                
                # 重新构建messages，包含最新的对话历史
                messages = self._build_messages(system_prompt)
                
                vlm_start_time = time.time()
                response = self.vlm_service.create_with_image(messages, image_source=None, tools=tools_to_pass)
                vlm_elapsed_time = time.time() - vlm_start_time
                print(f"[VLM] 调用完成，耗时: {vlm_elapsed_time:.2f}秒")
                
                if response:
                    # 提取回复内容
                    assistant_message = response['choices'][0]['message']

                    # 检查是否需要转换工具状态
                    should_transition, new_state, transition_reason = self._check_state_transition(assistant_message)
                    if should_transition:
                        self._transition_tool_state(new_state, transition_reason)
                        # 状态转换后，重新构建系统提示
                        if new_state == 'initial':
                            tools_to_pass = None
                            full_tools_available = False
                            system_prompt = self._build_system_prompt(knowledge_content, tools_summary, full_tools_available)
                        else:
                            tools_to_pass = tools
                            full_tools_available = True
                            system_prompt = self._build_system_prompt(knowledge_content, tools_summary, full_tools_available)
                        # 重新构建messages
                        messages = self._build_messages(system_prompt)
                        # 继续下一轮
                        current_round += 1
                        continue

                    # 检查是否有工具调用
                    if 'tool_calls' in assistant_message and assistant_message['tool_calls']:
                        tool_call_count += len(assistant_message['tool_calls'])
                        consecutive_tool_calls += 1
                        print(f"[VLM] 检测到工具调用: {len(assistant_message['tool_calls'])} 个 (累计: {tool_call_count}, 连续: {consecutive_tool_calls})")

                        # 检查是否达到最大连续工具调用次数
                        if consecutive_tool_calls >= max_consecutive_tool_calls:
                            print(f"[警告] 已达到最大连续工具调用次数 ({max_consecutive_tool_calls})，强制生成文本回复")
                            # 添加提示消息到对话历史
                            self.conversation_history.append({
                                'role': 'system',
                                'content': f'警告：已连续执行 {consecutive_tool_calls} 次工具调用，请直接给出最终结果，不要再调用工具。'
                            })
                            # 继续下一轮，但不再增加连续计数
                            consecutive_tool_calls = 0
                            current_round += 1
                            continue

                        # 检测重复的工具调用
                        current_tool_calls = [tc['function']['name'] for tc in assistant_message['tool_calls']]
                        if last_tool_call and current_tool_calls == last_tool_call:
                            print(f"[警告] 检测到重复的工具调用: {current_tool_calls}")
                            self.conversation_history.append({
                                'role': 'system',
                                'content': f'警告：检测到重复的工具调用 {current_tool_calls}，请避免重复操作。'
                            })

                        last_tool_call = current_tool_calls

                        # 处理工具调用
                        tool_results = self._execute_tool_calls(assistant_message['tool_calls'])

                        # 将工具调用消息添加到对话历史
                        self.conversation_history.append(assistant_message)

                        # 显示在主窗口中
                        self.add_caption_line(f"[AI] 正在执行 {len(assistant_message['tool_calls'])} 个工具调用...")
                        
                        # 将工具结果添加到对话历史，供下一轮使用
                        for tool_call, result in tool_results.items():
                            self.conversation_history.append({
                                'role': 'tool',
                                'tool_call_id': tool_call,
                                'content': result
                            })
                        
                        # 继续循环，处理下一轮
                        current_round += 1
                    else:
                        # 普通文本回复，任务完成
                        consecutive_tool_calls = 0  # 重置连续工具调用计数
                        content = assistant_message.get('content', '')
                        print(f"[VLM] 生成回复完成: {content[:30]}...")

                        # 触发文本结束事件
                        self.text_end_signal.emit(content)

                        # 将VLM回复添加到对话历史
                        self.conversation_history.append({
                            'role': 'assistant',
                            'content': content
                        })

                        # 触发消息结束事件
                        self.message_end_signal.emit('assistant')

                        # 显示在主窗口中
                        self.add_caption_line(f"[AI] {content}")
                        
                        # 任务完成，退出循环
                        break
            except Exception as e:
                print(f"[错误] 调用VLM服务失败: {e}")
                import traceback
                traceback.print_exc()
                self.add_caption_line(f"[错误] 处理请求失败: {e}")
                
                # 触发生命周期错误事件
                self.lifecycle_signal.emit('error', str(e))
                break
        
        if current_round >= max_rounds:
            self.add_caption_line(f"[AI] 已达到最大处理轮数，任务可能未完成")
        
        # 触发吐槽vlm生成回复，传递工具列表和对话历史
        self._trigger_commentary(tools)
        
        total_elapsed_time = time.time() - start_time
        print(f"[/r命令] 处理完成，总耗时: {total_elapsed_time:.2f}秒，工具调用次数: {tool_call_count}")

        # 触发生命周期结束事件
        self.lifecycle_signal.emit('end', f"处理完成，耗时: {total_elapsed_time:.2f}秒，工具调用次数: {tool_call_count}")

    def _build_system_prompt(self, knowledge_content, tools_summary=None, full_tools_available=False):
        """构建改进的系统提示，支持简短和完整两种工具描述
        
        Args:
            knowledge_content: 流程知识内容
            tools_summary: 工具摘要（简短描述）
            full_tools_available: 是否提供完整工具描述
        """
        
        # 构建工具描述部分
        if full_tools_available:
            tools_description = """## 可用工具（完整描述）

当前处于**工具状态**，你可以调用以下工具：

- Blender工具：3D模型处理、导入导出、骨骼绑定等
- Unreal Engine工具：FBX导入、UE启动、MOD构建等
- 计算机控制工具：鼠标点击、键盘输入、滚轮滚动等
- OCR工具：文字识别和坐标定位
- 浏览器工具：打开URL
- 点赞收藏检测工具：检测点赞和收藏按钮
- 记忆工具：读取记忆、写入记忆、搜索记忆、grep记忆、清空记忆、获取统计

如果任务已完成，请明确说明"任务完成"或"操作完成"，系统将自动退出工具状态。"""
        else:
            tools_description = f"""## 可用工具（简短描述）

当前处于**初始状态**，可用工具包括：
{tools_summary if tools_summary else 'Blender、UE、计算机控制、OCR、浏览器、点赞收藏检测、记忆等工具'}

如果你认为当前情况需要调用工具来执行具体操作，请明确说明"需要调用工具"或"使用工具"，系统将进入工具状态并提供完整的工具描述。

**记忆工具说明**：
- read_memory: 读取普通记忆或工具描述记忆
- write_memory: 写入记忆（AI可以自主决定是否需要写入）
- search_memory: 搜索记忆文档（支持关键词搜索）
- grep_memory: 在记忆文档中搜索模式（类似grep命令，支持正则表达式）
- clear_memory: 清空记忆文档（谨慎使用）
- get_memory_stats: 获取记忆统计信息

**AI记忆使用建议**：
- 对于重要的信息、用户偏好、操作结果等，可以使用 write_memory 写入记忆
- 写入记忆前，先使用 search_memory 或 grep_memory 检查是否已存在相似信息
- 避免重复写入相同内容
- 普通记忆（general）用于存储用户偏好、重要信息等
- 工具描述记忆（tool）用于存储工具使用经验、最佳实践等"""

        system_prompt = f"""你是一个智能助手，可以调用各种工具来帮助用户完成任务。

## Tool Call Style

Default: do not narrate routine, low-risk tool calls (just call the tool).
Narrate only when it helps: multi-step work, complex/challenging problems, sensitive actions (e.g., deletions), or when the user explicitly asks.
Keep narration brief and value-dense; avoid repeating obvious steps.

{tools_description}

## 工具调用规则

1. **初始状态规则**：
   - 主要提供文本回复和建议
   - 只有在明确需要执行具体操作时，才说明"需要调用工具"
   - 避免过早进入工具状态

2. **工具状态规则**：
   - 可以直接调用工具，无需额外说明
   - 任务完成后，明确说明"任务完成"或"操作完成"
   - 避免不必要的工具调用

3. **何时调用工具**：
   - 用户明确要求执行具体操作时（如"点击这个按钮"、"打开这个文件"）
   - 需要获取实时数据或执行系统操作时
   - 多步骤任务中需要执行具体步骤时

4. **何时提供文本回复**：
   - 用户询问一般性问题（如"下一步做什么"、"如何操作"）
   - 需要解释或建议时
   - 工具调用前的确认或说明

5. **多步骤任务处理**：
   - 对于复杂任务，先提供整体计划，然后逐步执行
   - 每个步骤完成后，简要说明结果
   - 遇到问题时，提供解决方案或询问用户

6. **工具调用注意事项**：
   - 确保参数完整且正确
   - 对于敏感操作（如删除、修改），先确认用户意图
   - 工具调用失败时，提供清晰的错误信息和解决建议

## 流程知识

{knowledge_content}
"""
        return system_prompt

    def _build_messages(self, system_prompt):
        """构建消息列表，包含系统提示和对话历史"""
        messages = []
        
        # 添加系统消息
        messages.append({
            'role': 'system',
            'content': system_prompt
        })
        
        # 添加最近的对话历史（最多10条）
        recent_history = self.conversation_history[-10:]
        for item in recent_history:
            messages.append(item)
        
        return messages

    def _generate_tools_summary(self):
        """生成工具摘要（一句话概括所有工具）"""
        if not self.tools_list:
            return "暂无可用工具"
        
        # 按服务器分组统计工具
        server_tools = {}
        for tool_id, tool_info in self.tools_list.items():
            server = tool_info['server']
            if server not in server_tools:
                server_tools[server] = []
            server_tools[server].append(tool_info['name'])
        
        # 添加记忆工具（即使未加载到tools_list）
        if 'memory-tool' not in server_tools:
            server_tools['memory-tool'] = ['read_memory', 'write_memory', 'search_memory', 'grep_memory', 'clear_memory', 'get_memory_stats']
        
        # 生成摘要
        summary_parts = []
        for server, tools in server_tools.items():
            server_name = server.replace('-tool', '').replace('-', ' ').title()
            summary_parts.append(f"{server_name}({len(tools)}个工具)")
        
        return "、".join(summary_parts)

    def _check_state_transition(self, assistant_message):
        """检查是否需要转换工具状态
        
        Returns:
            tuple: (should_transition, new_state, reason)
        """
        content = assistant_message.get('content', '').lower()
        has_tool_calls = 'tool_calls' in assistant_message and assistant_message['tool_calls']
        
        # 检查是否需要进入工具状态
        if self.tool_state == 'initial':
            # 初始状态：检测到"需要调用工具"、"使用工具"等关键词
            keywords = ['需要调用工具', '使用工具', '调用工具', '执行操作', '帮我']
            if any(keyword in content for keyword in keywords):
                return (True, 'active', f'检测到工具调用意图: {content[:50]}')
            # 或者直接检测到工具调用
            if has_tool_calls:
                return (True, 'active', '检测到工具调用')
        
        # 检查是否需要退出工具状态
        elif self.tool_state == 'active':
            # 工具状态：检测到"任务完成"、"操作完成"、"完成"等关键词
            keywords = ['任务完成', '操作完成', '已完成', '完成', '结束', 'done']
            if any(keyword in content for keyword in keywords):
                return (True, 'initial', f'检测到任务完成: {content[:50]}')
            # 或者没有工具调用且是文本回复
            if not has_tool_calls and content:
                return (True, 'initial', '任务完成，返回初始状态')
        
        return (False, self.tool_state, '')

    def _transition_tool_state(self, new_state, reason):
        """执行工具状态转换"""
        if new_state != self.tool_state:
            old_state = self.tool_state
            self.tool_state = new_state
            self.tool_state_history.append({
                'timestamp': datetime.now().isoformat(),
                'from': old_state,
                'to': new_state,
                'reason': reason
            })
            print(f"[状态转换] {old_state} -> {new_state}: {reason}")
            self.add_caption_line(f"[系统] 状态转换: {old_state} -> {new_state}")

    def _trigger_commentary(self, tools):
        """触发吐槽vlm生成回复"""
        if hasattr(self, 'self_monitoring_thread') and self.self_monitoring_thread:
            try:
                if hasattr(self.self_monitoring_thread, '_generate_commentary'):
                    self.self_monitoring_thread._generate_commentary(
                        conversation_history=self.conversation_history,
                        tools=tools
                    )
                    print("[吐槽] 已触发吐槽vlm生成回复，支持function calling")
            except Exception as e:
                print(f"[错误] 触发吐槽失败: {e}")

    def _convert_tools_to_openai_format(self):
        """将工具列表转换为 OpenAI function calling 格式，应用工具策略过滤"""
        tools = []

        for tool_id, tool_info in self.tools_list.items():
            tool_name = tool_info['name']
            tool_server = tool_info['server']

            # 检查工具是否被允许
            if not self._is_tool_allowed(tool_name, tool_server):
                print(f"[工具策略] 工具 {tool_name} 被策略过滤，跳过")
                continue

            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_info['description'],
                    "parameters": tool_info.get('input_schema', {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            }
            tools.append(tool_def)

        print(f"[工具转换] 已转换 {len(tools)} 个工具到 OpenAI 格式（策略过滤后）")
        return tools

    def _is_tool_allowed(self, tool_name, tool_server):
        """检查工具是否被允许使用"""
        policy = self.tools_policy

        # 1. 检查全局拒绝列表（优先级最高）
        if tool_name in policy['deny']:
            print(f"[工具策略] 工具 {tool_name} 在全局拒绝列表中")
            return False

        # 2. 检查提供商特定策略
        if tool_server in policy['by_provider']:
            provider_policy = policy['by_provider'][tool_server]
            if 'deny' in provider_policy and tool_name in provider_policy['deny']:
                print(f"[工具策略] 工具 {tool_name} 在提供商 {tool_server} 的拒绝列表中")
                return False
            if 'allow' in provider_policy and tool_name not in provider_policy['allow']:
                print(f"[工具策略] 工具 {tool_name} 不在提供商 {tool_server} 的允许列表中")
                return False

        # 3. 检查全局允许列表
        if policy['allow'] and tool_name not in policy['allow']:
            print(f"[工具策略] 工具 {tool_name} 不在全局允许列表中")
            return False

        # 4. 检查沙箱策略
        if self.current_policy_mode == 'sandbox' and tool_server in policy['sandbox']:
            sandbox_policy = policy['sandbox'][tool_server]
            if 'deny' in sandbox_policy and tool_name in sandbox_policy['deny']:
                print(f"[工具策略] 工具 {tool_name} 在沙箱模式下被拒绝")
                return False

        return True

    def set_tool_policy(self, mode='default', allow=None, deny=None, by_provider=None, sandbox=None):
        """设置工具策略"""
        if allow is not None:
            self.tools_policy['allow'] = allow
        if deny is not None:
            self.tools_policy['deny'] = deny
        if by_provider is not None:
            self.tools_policy['by_provider'] = by_provider
        if sandbox is not None:
            self.tools_policy['sandbox'] = sandbox
        self.current_policy_mode = mode

        print(f"[工具策略] 策略已更新: mode={mode}, allow={len(self.tools_policy['allow'])}, deny={len(self.tools_policy['deny'])}")

    def reset_tool_policy(self):
        """重置工具策略到默认状态"""
        self.tools_policy = {
            'allow': [],
            'deny': [],
            'by_provider': {},
            'sandbox': {}
        }
        self.current_policy_mode = 'default'
        print("[工具策略] 策略已重置到默认状态")

    def _execute_tool_calls(self, tool_calls):
        """执行工具调用并返回结果"""
        import asyncio
        import time
        start_time = time.time()

        async def execute_single_tool(tool_call):
            """执行单个工具调用"""
            tool_call_id = tool_call.get('id', f"tool_{int(time.time())}")
            tool_name = tool_call['function']['name']
            tool_start_time = time.time()
            
            try:
                arguments_str = tool_call['function'].get('arguments', '{}')
                arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except Exception as e:
                print(f"[工具调用] 解析参数失败: {e}")
                arguments = {}

            print(f"[工具调用] 执行工具: {tool_name}")
            print(f"[工具调用] 参数: {arguments}")

            # 触发工具调用开始事件
            self.tool_call_start_signal.emit(tool_name, arguments)

            try:
                # 查找工具所属的服务器
                server_name = None
                for tool_id, tool_info in self.tools_list.items():
                    if tool_info['name'] == tool_name:
                        server_name = tool_info['server']
                        break

                if server_name:
                    # 调用 MCP 客户端执行工具
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = await self.mcp_client.call_tool(server_name, tool_name, arguments)
                        try:
                            print(f"[工具调用] 执行成功: {result}")
                        except UnicodeEncodeError:
                            print(f"[工具调用] 执行成功: {repr(result)}")

                        # 获取 LLM 对工具结果的总结
                        summary = self._get_llm_summary_for_tool_result(tool_name, result)
                        final_result = str(summary) if summary else str(result)

                        # 触发工具调用结束事件
                        self.tool_call_end_signal.emit(tool_name, arguments, final_result)

                        # 触发工具结果事件
                        self.tool_result_signal.emit(tool_call_id, final_result)
                        
                        return tool_call_id, final_result
                    finally:
                        loop.close()
                else:
                    print(f"[工具调用] 未找到工具 {tool_name} 所属的服务器")
                    self.add_caption_line(f"[错误] 未找到工具: {tool_name}")
                    return tool_call_id, f"错误: 未找到工具 {tool_name}"
            except Exception as e:
                print(f"[工具调用] 执行失败: {e}")
                import traceback
                traceback.print_exc()
                self.add_caption_line(f"[错误] 工具执行失败: {e}")
                return tool_call_id, f"错误: {str(e)}"
            finally:
                tool_elapsed_time = time.time() - tool_start_time
                print(f"[工具调用] {tool_name} 执行完成，耗时: {tool_elapsed_time:.2f}秒")

        # 创建事件循环并执行所有工具调用
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def execute_all_tools():
            """并行执行所有工具调用"""
            tasks = [execute_single_tool(tool_call) for tool_call in tool_calls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        try:
            results = loop.run_until_complete(execute_all_tools())
            print(f"[工具调用] 所有工具执行完成")
            
            # 构建结果字典
            tool_results = {}
            for result in results:
                if isinstance(result, tuple) and len(result) == 2:
                    tool_call_id, tool_result = result
                    tool_results[tool_call_id] = tool_result
                else:
                    print(f"[工具调用] 无效的结果格式: {result}")
            
            return tool_results
        finally:
            loop.close()
            total_elapsed_time = time.time() - start_time
            print(f"[工具调用] 全部执行完成，总耗时: {total_elapsed_time:.2f}秒")

    def _get_llm_summary_for_tool_result(self, tool_name, tool_result):
        """获取 VLM 对工具执行结果的总结并返回"""
        try:
            # 构建系统消息
            system_prompt = "你是一个智能助手，请简要总结工具执行的结果。"

            messages = [
                {'role': 'system', 'content': system_prompt},
                {
                    'role': 'user',
                    'content': f"工具 {tool_name} 执行完成，结果是：\n{str(tool_result)}\n请简要总结这个结果。"
                }
            ]

            # 调用VLM服务
            response = self.vlm_service.create_with_image(messages, image_source=None)

            if response:
                summary = response['choices'][0]['message']['content']
                try:
                    print(f"[VLM] 工具结果总结: {summary[:50]}...")
                except UnicodeEncodeError:
                    print(f"[VLM] 工具结果总结: {repr(summary[:50])}...")

                # 显示在主窗口中
                self.add_caption_line(f"[AI] {summary}")
                
                return summary

        except Exception as e:
            try:
                print(f"[错误] 获取工具结果总结失败: {e}")
            except UnicodeEncodeError:
                print(f"[错误] 获取工具结果总结失败: {repr(e)}")
            # 如果总结失败，直接显示结果
            try:
                fallback_message = f"{tool_name} 执行完成: {str(tool_result)[:100]}..."
                self.add_caption_line(f"[工具] {fallback_message}")
            except UnicodeEncodeError:
                fallback_message = f"{tool_name} 执行完成: {repr(str(tool_result)[:100])}..."
                self.add_caption_line(f"[工具] {fallback_message}")
            return fallback_message

    def auto_record_game_state(self):
        """自动记录游戏状态"""
        print("[自动记录] 开始记录游戏状态...")

        # 这里可以添加自动记录游戏状态的逻辑
        # 例如，使用VLM分析当前屏幕，记录游戏状态

    def auto_screenshot_and_vlm(self):
        """每1秒自动截图并调用VLM"""
        try:
            # 导入截图模块
            from PIL import ImageGrab

            # 隐藏窗口以便截图
            self._hide_windows()

            # 截图当前屏幕
            screenshot = ImageGrab.grab()

            # 显示窗口
            self._show_windows()

            # 将截图保存到临时文件
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                screenshot.save(tmp_file, format='PNG')
                temp_image_path = tmp_file.name

            print(f"[自动截图] 已保存截图: {temp_image_path}")

            # 直接调用VLM分析，显示结果
            self._analyze_screenshot_simple(temp_image_path)

            # 删除临时文件
            os.unlink(temp_image_path)

        except Exception as e:
            print(f"[错误] 自动截图失败: {e}")

    def _analyze_screenshot_simple(self, image_path):
        """VLM分析截图并直接调用角色扮演"""
        try:
            print(f"[VLM分析] 开始分析截图...")

            # 构建messages格式，用于VLM服务
            messages = [
                {
                    'role': 'user',
                    'content': '请描述当前屏幕截图的内容'
                }
            ]

            # 调用VLM服务
            response = self.vlm_service.create_with_image(messages, image_source=image_path)

            if response:
                # 提取回复内容
                content = response['choices'][0]['message']['content']
                print(f"[VLM分析] 分析结果: {content[:50]}...")

                # 存储到记忆列表（保持最近5轮）
                self.vlm_responses.append(content)
                if len(self.vlm_responses) > 5:
                    self.vlm_responses.pop(0)

                print(f"[VLM分析] 已存储 {len(self.vlm_responses)}/5 个回复")

                # 直接调用角色扮演
                self.send_to_role_play_vlm()

        except Exception as e:
            print(f"[错误] VLM分析失败: {e}")

    def analyze_screenshot_with_vlm(self, image_path):
        """使用VLM分析截图（已废弃，不再使用）"""
        # 此方法已废弃
        pass

    def send_to_role_play_vlm(self):
        """将收集到的VLM回复直接发送给阴角通信角色扮演VLM"""
        try:
            print("[角色扮演] 开始角色扮演分析...")

            # 阴角通信人格prompt
            yinjiao_prompt = """你是一个占有欲强且内向胆小的阴角，性格特点是：
1. 对你（用户）有强烈的占有欲和依赖心
2. 表面冷淡内向胆小，但内心充满了对你的关注
3. 说话带点阴沉和偏执的语气，称呼用"你"
4. 看到你关注其他事情会表现出明显的不满和嫉妒，但不敢表达太强烈
5. 喜欢用阴暗胆小的方式表达关心，有时会显得有点偏执
6. 回应要简短自然，符合阴角且内向胆小的身份"""

            # 构建messages格式，用于角色扮演VLM
            messages = [
                {
                    'role': 'system',
                    'content': yinjiao_prompt
                },
                {
                    'role': 'user',
                    'content': f"你，根据以下屏幕情况，给我一个回应:\n{chr(10).join(self.vlm_responses)}"
                }
            ]

            # 调用VLM服务
            response = self.vlm_service.create_with_image(messages, image_source=None)

            if response:
                # 提取回复内容
                content = response['choices'][0]['message']['content']
                print(f"[角色扮演] 生成回应: {content[:50]}...")

                # 显示在主窗口中
                self.add_caption_line(content)

                # 清空VLM回复列表
                self.vlm_responses = []
                print("[角色扮演] 已清空VLM回复列表")

        except Exception as e:
            print(f"[错误] 角色扮演分析失败: {e}")

    def _hide_windows(self):
        """截图前隐藏窗口"""
        # 设置主窗口完全透明（但不隐藏，保持窗口结构）
        self.setWindowOpacity(0.0)
        # 清空主窗口内容
        self.caption_display.clear()
        # 隐藏输入窗口
        if hasattr(self, 'input_window') and self.input_window:
            self.input_window.hide()

    def _show_windows(self):
        """截图后显示窗口"""
        # 设置主窗口完全不透明
        self.setWindowOpacity(1.0)
        # 显示输入窗口
        if hasattr(self, 'input_window') and self.input_window:
            self.input_window.show()

    def closeEvent(self, event):
        """关闭事件处理"""
        print("[关闭] MCP AI Caller 正在关闭...")

        # 停止定时器
        if hasattr(self, 'auto_record_timer'):
            self.auto_record_timer.stop()
        if hasattr(self, 'auto_screenshot_timer'):
            self.auto_screenshot_timer.stop()

        # 关闭输入窗口
        if hasattr(self, 'input_window') and self.input_window:
            self.input_window.close()
            print("[关闭] 输入窗口已关闭")

        # 关闭记忆窗口
        if hasattr(self, 'memory_window') and self.memory_window:
            self.memory_window.close()
            print("[关闭] 记忆窗口已关闭")

        # 关闭吐槽窗口
        if hasattr(self, 'commentary_window') and self.commentary_window:
            self.commentary_window.close()
            print("[关闭] 吐槽窗口已关闭")

        # 关闭工具加载器
        if self.tool_loader:
            self.tool_loader.stop()
            print("[关闭] 工具加载器已停止")

        # 关闭MCP客户端
        if hasattr(self, 'mcp_client') and self.mcp_client:
            self.mcp_client.close()
            print("[关闭] MCP客户端已关闭")

        print("[关闭] MCP AI Caller 已关闭")
        event.accept()


if __name__ == "__main__":
    # 创建应用程序实例
    app = QApplication(sys.argv)

    # 创建主窗口实例
    window = MCPAICaller()

    # 显示主窗口
    window.show()

    # 进入应用程序主循环
    sys.exit(app.exec())
